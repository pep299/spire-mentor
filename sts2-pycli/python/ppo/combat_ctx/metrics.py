"""Teacher-regret metrics, CIs and sealing for the combat context probe (PREREG §5-§7).

On every TEST decision there are two independent searches of the same
position: A (primary; its recommendation was played and labels TRAIN) and B
(the shadow). With `Q_X(c)` the visit-weighted mean root Q of action class `c`
in search X:

    r_s(d)  = ½[(Qmax_A − Q_A(π_s(d))) + (Qmax_B − Q_B(π_s(d)))]
    R(arm)  = mean_{d∈E} (1/3) Σ_s r_s(d)
    R_ref   = ½(mean_E (Qmax_A − Q_A(rec_B)) + mean_E (Qmax_B − Q_B(rec_A)))
    Agr     = mean_d (1/3) Σ_s ½[1(π_s = rec_A) + 1(π_s = rec_B)],  Agr_ref = mean 1(rec_A = rec_B)

A student never saw A's or B's noise on a TEST decision, so `Q_A(π)` and
`Q_B(π)` are unbiased scores of its pick; the shared `Qmax` bias cancels in a
paired contrast. `R_ref` is the regret of one more independent teacher search:
the noise reference, not a floor.

CIs are a two-level bootstrap (10,000 resamples): runs with replacement, then
within the resample each arm's training seeds with replacement; the level is
97.06% (Pocock, two looks at an overall 0.05). A CI clustered by bank fight
(every play of one fight, whatever its rep) is reported too. `outcome`
applies the disjoint outcome table of §7.3.

The sealed read also carries every PREREG §5.7 / §6 diagnostic, so none needs
a script outside the seal: the per-rule E removal counts (and the rule-3 red
flag), `Δ_tok` when `S_ctx_tok` picks are given, R / Agr on all searched
decisions, and every contrast on the contested (frozen τ), high-visit (every
class ≥ 50 visits in A and B) and zero-UNK subsets. `eval --e-report` prints
the student-free E-rule counts alone, before the read.

Sealing: `freeze` writes `final/FROZEN.json` (each checkpoint's sha256, arm
and training seed, margins `M_ctx` / `M` from DEV, T*, tau) and refuses unless
every student arm has its training seeds (3, distinct); `eval --final --stage
1|2` refuses without it, or twice, and accepts pick files only when each was
predicted by the frozen checkpoint of the arm it is labelled with and they
cover every frozen checkpoint exactly once. Nothing in `train` reads TEST.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from typing import Any

import numpy as np

from .data import Split, class_stats

CI_LEVEL = 0.9706
N_BOOT = 10_000
MARGIN_FLOOR, MARGIN_CAP = 0.002, 0.003
REFERENCE_ARMS = ("ref_teacher", "rl2_zs", "heur", "uniform")
SEEDS_PER_ARM = 3
HIGH_VISITS = 50
RULE3_RED_FLAG = 0.10


# ── per-decision quantities ──────────────────────────────────────────────


def argmax_per_decision(sp: Split, scores: np.ndarray) -> np.ndarray:
    """Global action index of each decision's argmax (first on ties)."""
    a = sp.a
    out = np.empty(sp.n, np.int64)
    for d in range(sp.n):
        s, n = a["act_start"][d], a["n_act"][d]
        out[d] = s + int(np.argmax(scores[s : s + n]))
    return out


def pick_rows(sp: Split, pick: np.ndarray) -> dict[str, np.ndarray]:
    """Per decision: regret under A and B, agreement with each recommendation,
    class top-1 — for a global-action-index pick."""
    a = sp.a
    cls = a["cls"][pick].astype(np.int64)
    return {
        "regA": a["regA"][pick].astype(np.float64),
        "regB": a["regB"][pick].astype(np.float64),
        "agrA": (cls == a["rec_clsA"]).astype(np.float64),
        "agrB": (cls == a["rec_clsB"]).astype(np.float64),
    }


def ref_rows(sp: Split) -> dict[str, np.ndarray]:
    """The noise reference: A's pick scored by B and B's pick scored by A."""
    a = sp.a
    pa = a["act_start"] + np.maximum(a["recA"], 0)
    pb = a["act_start"] + np.maximum(a["recB"], 0)
    return {
        "regA": a["regA"][pb].astype(np.float64),
        "regB": a["regB"][pa].astype(np.float64),
        "agrA": (a["rec_clsA"] == a["rec_clsB"]).astype(np.float64),
        "agrB": (a["rec_clsA"] == a["rec_clsB"]).astype(np.float64),
    }


def uniform_rows(sp: Split) -> dict[str, np.ndarray]:
    """Uniform over action classes, in expectation."""
    a = sp.a
    out = {k: np.zeros(sp.n) for k in ("regA", "regB", "agrA", "agrB")}
    for d in range(sp.n):
        s, n = a["act_start"][d], a["n_act"][d]
        cls = a["cls"][s : s + n].astype(np.int64)
        first = np.unique(cls, return_index=True)[1]
        out["regA"][d] = np.mean(a["regA"][s + first])
        out["regB"][d] = np.mean(a["regB"][s + first])
        nc = len(first)
        out["agrA"][d] = (a["rec_clsA"][d] >= 0) / nc
        out["agrB"][d] = (a["rec_clsB"][d] >= 0) / nc
    return out


def reference_picks(sp: Split) -> dict[str, np.ndarray]:
    """Global action picks of the training-free arms: rl2 zero-shot (argmax of
    the built-in net's logits) and the heuristic (argmax of v1 feature 21)."""
    return {
        "rl2_zs": argmax_per_decision(sp, sp.a["rl2_logit"]),
        "heur": argmax_per_decision(sp, sp.a["rl2_x"][:, 21]),
    }


def arm_rows(sp: Split, picks: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Per decision, per training seed: `[S, n]` arrays."""
    rows = [pick_rows(sp, p) for p in picks]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def summarize(rows: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    """R, R_A, R_B, Agr over `mask` (rows may be `[n]` or `[S, n]`)."""

    def m(x: np.ndarray) -> float:
        x = x.mean(0) if x.ndim == 2 else x
        return float(np.nanmean(x[mask])) if mask.any() else float("nan")

    ra, rb = m(rows["regA"]), m(rows["regB"])
    return {
        "R": 0.5 * (ra + rb),
        "R_A": ra,
        "R_B": rb,
        "Agr": 0.5 * (m(rows["agrA"]) + m(rows["agrB"])),
    }


def dev_summary(sp: Split, pick: np.ndarray) -> dict[str, Any]:
    """The per-epoch DEV read a trainer selects on (E only for R and Agr)."""
    a = sp.a
    rows = pick_rows(sp, pick)
    e = a["in_E"]
    out = summarize(rows, e)
    has = a["rec_clsA"] >= 0
    out["top1"] = float(rows["agrA"][has].mean()) if has.any() else float("nan")
    out["n_E"] = int(e.sum())
    for room, rid in (("boss", 2), ("elite", 1)):
        out[f"R_{room}"] = summarize(rows, e & (a["room"] == rid))["R"]
    return out


# ── bootstrap ────────────────────────────────────────────────────────────


def _cluster_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(u))]
    return u, members


def bootstrap_contrast(
    reg_x: np.ndarray,
    reg_y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = N_BOOT,
    level: float = CI_LEVEL,
    seed: int = 0,
) -> dict[str, float]:
    """`Δ = mean(reg_x) − mean(reg_y)` over decisions, two-level bootstrap.

    Args:
        reg_x, reg_y: `[S, n]` per-seed per-decision regret (S may differ, 1
            for a seedless reference). Each resample draws clusters (`groups`,
            e.g. runs) with replacement, then each arm's seeds with
            replacement, and averages.
        groups: `[n]` cluster id per decision.
    """
    reg_x = np.atleast_2d(reg_x)
    reg_y = np.atleast_2d(reg_y)
    point = float(reg_x.mean(0).mean() - reg_y.mean(0).mean())
    _, members = _cluster_index(groups)
    k = len(members)
    rng = np.random.default_rng(seed)

    # Per cluster per seed sums, so a resample is cheap.
    def sums(r: np.ndarray) -> np.ndarray:
        return np.stack([r[:, m].sum(1) for m in members], 1)  # [S, k]

    sx, sy = sums(reg_x), sums(reg_y)
    cnt = np.asarray([len(m) for m in members], np.float64)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.integers(0, k, k)
        w = np.bincount(c, minlength=k).astype(np.float64)
        ix = rng.integers(0, sx.shape[0], sx.shape[0])
        iy = rng.integers(0, sy.shape[0], sy.shape[0])
        n = (w * cnt).sum()
        stats[b] = (sx[ix].mean(0) @ w - sy[iy].mean(0) @ w) / n
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "delta": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(stats.std()),
        "clusters": k,
    }


def outcome(lo: float, hi: float, point: float, margin: float, stage: int) -> str:
    """PREREG §7.3's disjoint outcome table (referee / agreement qualifiers apart)."""
    if lo > 0:
        return "PASS" if point >= margin else "SMALL"
    if hi < margin:
        return "NULL"
    return "INCONCLUSIVE" if stage == 1 else "UNDERPOWERED"


def margin(r_control_dev: float, r_ref_dev: float) -> float:
    """`clip(0.25·(R_DEV(control) − R_ref,DEV), 0.002, 0.003)` (§7.2)."""
    return float(np.clip(0.25 * (r_control_dev - r_ref_dev), MARGIN_FLOOR, MARGIN_CAP))


def contested_tau(sp: Split) -> float:
    """`max(0.02, 2·σ_AB)`, σ_AB the per-class SD of `Q_A(c) − Q_B(c)` on DEV shadows."""
    a = sp.a
    diffs = []
    for d in np.flatnonzero(a["has_B"]):
        s, n = a["act_start"][d], a["n_act"][d]
        cls = a["cls"][s : s + n].astype(np.int64)
        _, qa = class_stats(cls, a["visA"][s : s + n], a["qA"][s : s + n])
        _, qb = class_stats(cls, a["visB"][s : s + n], a["qB"][s : s + n])
        ok = np.isfinite(qa) & np.isfinite(qb)
        diffs.extend((qa - qb)[ok])
    sd = float(np.std(diffs)) if diffs else 0.0
    return max(0.02, 2 * sd)


# ── evaluation table ─────────────────────────────────────────────────────


def decision_flags(sp: Split) -> dict[str, np.ndarray]:
    """Per decision, from the search statistics alone: searched (rule 1),
    covered (rule 3), the smallest class visit count over A and B, and the
    range over classes of `½(Q_A(c) + Q_B(c))` (the contested statistic)."""
    a = sp.a
    n = sp.n
    out = {
        "searched": np.zeros(n, bool),
        "covered": np.zeros(n, bool),
        "min_visits": np.zeros(n),
        "q_range": np.full(n, np.nan),
    }
    for d in range(n):
        s, k = a["act_start"][d], a["n_act"][d]
        cls = a["cls"][s : s + k].astype(np.int64)
        va, qa = class_stats(cls, a["visA"][s : s + k], a["qA"][s : s + k])
        vb, qb = class_stats(cls, a["visB"][s : s + k], a["qB"][s : s + k])
        out["searched"][d] = va.sum() > 0
        out["covered"][d] = (
            bool(a["has_B"][d]) and len(va) >= 2 and (va > 0).all() and (vb > 0).all()
        )
        out["min_visits"][d] = min(va.min(), vb.min())
        m = 0.5 * (qa + qb)
        if np.isfinite(m).any():
            out["q_range"][d] = np.nanmax(m) - np.nanmin(m)
    e = out["searched"] & a["card_phase"] & out["covered"]
    if not np.array_equal(e, a["in_E"]):
        raise RuntimeError("the cache's E flag disagrees with PREREG §6's three rules")
    return out


def e_rule_report(sp: Split, flags: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """How many decisions each §6 rule removes, in order, and the rule-3 red
    flag (rule 3 removes > 10% of the searched card-play decisions)."""
    f = flags or decision_flags(sp)
    searched, cp, cov = f["searched"], sp.a["card_phase"], f["covered"]
    r1 = int((~searched).sum())
    r2 = int((searched & ~cp).sum())
    card_play = int((searched & cp).sum())
    r3 = int((searched & cp & ~cov).sum())
    share3 = r3 / max(1, card_play)
    return {
        "decisions": sp.n,
        "removed": {"rule1_not_searched": r1, "rule2_not_card_phase": r2, "rule3_not_covered": r3},
        "share_of_decisions": {
            "rule1": r1 / max(1, sp.n),
            "rule2": r2 / max(1, sp.n),
            "rule3": r3 / max(1, sp.n),
        },
        "searched_card_play": card_play,
        "rule3_share_of_card_play": share3,
        "rule3_red_flag": share3 > RULE3_RED_FLAG,
        "in_E": int(sp.a["in_E"].sum()),
    }


def arm_table(
    rows: dict[str, dict[str, np.ndarray]], sp: Split, mask: np.ndarray
) -> dict[str, Any]:
    out = {}
    for arm, r in rows.items():
        row = summarize(r, mask)
        for room, rid in (("boss", 2), ("elite", 1)):
            row[f"R_{room}"] = summarize(r, mask & (sp.a["room"] == rid))["R"]
        if arm == "rl2_zs":
            row["label"] = "in-sample for rl2"
        out[arm] = row
    return out


def contrast_table(
    rows: dict[str, dict[str, np.ndarray]],
    sp: Split,
    mask: np.ndarray,
    contrasts: list[tuple[str, str]],
    n_boot: int,
    fight_clustered: bool,
) -> dict[str, Any]:
    out = {}
    for ctrl, trt in contrasts:
        rx = 0.5 * (rows[ctrl]["regA"] + rows[ctrl]["regB"])[:, mask]
        ry = 0.5 * (rows[trt]["regA"] + rows[trt]["regB"])[:, mask]
        if not mask.any():
            out[f"{ctrl}-{trt}"] = {"delta": float("nan"), "n": 0}
            continue
        res = bootstrap_contrast(rx, ry, sp.run[mask], n_boot=n_boot)
        if fight_clustered:
            res["fight_clustered"] = bootstrap_contrast(
                rx, ry, sp.bank_fight[mask], n_boot=n_boot, seed=1
            )
        res["n"] = int(mask.sum())
        out[f"{ctrl}-{trt}"] = res
    return out


def evaluate(
    sp: Split,
    arm_picks: dict[str, list[np.ndarray]],
    contrasts: list[tuple[str, str]],
    margins: dict[str, float] | None = None,
    stage: int = 1,
    n_boot: int = N_BOOT,
    tau: float | None = None,
    unk: np.ndarray | None = None,
) -> dict[str, Any]:
    """The outcome table on `sp` (E only), every arm and reference, plus the
    PREREG §5.7 / §6 diagnostics.

    Args:
        arm_picks: arm -> one global-action pick array per training seed.
        contrasts: `(control, treatment)` pairs; Δ = R(control) − R(treatment).
        margins: contrast name -> margin (frozen); omitted -> no outcome.
        tau: the contested threshold (frozen); None -> no contested subset.
        unk: per decision, True if any student's inputs hold an UNK / rare
            id; None -> no zero-UNK subset.
    """
    a = sp.a
    e = a["in_E"]
    flags = decision_flags(sp)
    rows = {arm: arm_rows(sp, p) for arm, p in arm_picks.items()}
    rows["ref_teacher"] = {k: v[None] for k, v in ref_rows(sp).items()}
    for name, p in reference_picks(sp).items():
        rows[name] = {k: v[None] for k, v in pick_rows(sp, p).items()}
    rows["uniform"] = {k: v[None] for k, v in uniform_rows(sp).items()}
    table: dict[str, Any] = {
        "n_E": int(e.sum()),
        "n_runs": len(np.unique(sp.run[e])),
        "n_fights": len(np.unique(sp.bank_fight[e])),
        "n_fight_plays": len(np.unique(sp.fight[e])),
        "arms": arm_table(rows, sp, e),
    }
    out_c = contrast_table(rows, sp, e, contrasts, n_boot, fight_clustered=True)
    for name, res in out_c.items():
        if margins and name in margins:
            res["margin"] = margins[name]
            res["outcome"] = outcome(res["lo"], res["hi"], res["delta"], margins[name], stage)
    table["contrasts"] = out_c
    table["e_rules"] = e_rule_report(sp, flags)
    searched = flags["searched"] & a["has_B"]
    table["all_searched"] = {"n": int(searched.sum()), "arms": arm_table(rows, sp, searched)}
    subsets: dict[str, np.ndarray | None] = {
        "contested": None if tau is None else e & (flags["q_range"] >= tau),
        "high_visit": e & (flags["min_visits"] >= HIGH_VISITS),
        "zero_unk": None if unk is None else e & ~unk,
    }
    table["subsets"] = {}
    for name, m in subsets.items():
        if m is None:
            table["subsets"][name] = {
                "skipped": "tau not frozen" if name == "contested" else "no UNK flags"
            }
            continue
        table["subsets"][name] = {
            "n_E": int(m.sum()),
            "share_of_E": float(m.sum() / max(1, e.sum())),
            "arms": arm_table(rows, sp, m),
            "contrasts": contrast_table(rows, sp, m, contrasts, n_boot, fight_clustered=False),
        }
    if tau is not None:
        table["subsets"]["contested"]["tau"] = tau
    return table


# ── sealing ──────────────────────────────────────────────────────────────


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(final_dir: str, ckpts: list[str]) -> dict[str, Any]:
    """The FROZEN manifest, after checking every checkpoint's hash against it.

    Raises:
        SystemExit: No manifest, or a checkpoint not frozen / changed.
    """
    path = os.path.join(final_dir, "FROZEN.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: freeze every student first (PREREG §2 Sealing)")
    with open(path, encoding="utf-8") as f:
        frozen = json.load(f)
    want = frozen["checkpoints"]
    for c in ckpts:
        h = sha256(c)
        if os.path.abspath(c) not in want or want[os.path.abspath(c)] != h:
            raise SystemExit(f"{c} (sha256 {h[:12]}) is not a frozen checkpoint")
    return frozen


def ckpt_identity(path: str) -> dict[str, Any]:
    """`{"arm", "seed"}` a `train` checkpoint was trained as: a `best.pt`'s
    `arm` and `settings.seed`, an `rl2.json`'s `meta.combat_ctx`.

    Raises:
        SystemExit: The file records no arm or training seed (it was not
            written by `train`).
    """
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            meta = (json.load(f).get("meta") or {}).get("combat_ctx") or {}
        arm, seed, mode = meta.get("arm"), meta.get("seed"), meta.get("rl2_mode")
    else:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=True)
        settings = ck.get("settings") or {}
        arm, seed, mode = ck.get("arm"), settings.get("seed"), settings.get("rl2_mode")
    if arm is None or seed is None:
        raise SystemExit(f"{path}: no arm / training seed recorded (not a `train` checkpoint)")
    out: dict[str, Any] = {"arm": str(arm), "seed": int(seed)}
    if mode is not None:
        out["rl2_mode"] = str(mode)
    return out


def cmd_freeze(args: argparse.Namespace) -> int:
    os.makedirs(args.final_dir, exist_ok=True)
    path = os.path.join(args.final_dir, "FROZEN.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: the students are already frozen")
    with open(args.dev_reads, encoding="utf-8") as f:
        dev = json.load(f)
    identity = {}
    for c in args.ckpt:
        ident = ckpt_identity(c)
        if ident["arm"] == "S_rl2arch" and ident.get("rl2_mode") == "compat":
            raise SystemExit(f"{c}: the compat twin is the §3 cross-check, not a student")
        ident["sha256"] = sha256(c)
        identity[os.path.abspath(c)] = ident
    shas = [i["sha256"] for i in identity.values()]
    if len(set(shas)) != len(shas):
        raise SystemExit("two frozen checkpoints have one sha256")
    seeds: dict[str, list[int]] = collections.defaultdict(list)
    for i in identity.values():
        seeds[i["arm"]].append(i["seed"])
    for arm, ss in seeds.items():
        if len(set(ss)) != len(ss):
            raise SystemExit(f"{arm}: a training seed is frozen twice ({sorted(ss)})")
    for arm in STUDENTS:
        if len(seeds.get(arm, [])) != args.seeds_per_arm:
            raise SystemExit(
                f"{arm}: {len(seeds.get(arm, []))} training seeds, the freeze needs "
                f"{args.seeds_per_arm} (PREREG §4)"
            )
    frozen = {
        "checkpoints": {p: i["sha256"] for p, i in identity.items()},
        "identity": identity,
        "seeds_per_arm": args.seeds_per_arm,
        "M_ctx": margin(dev["R"]["S_ctxarch_rl2in"], dev["R"]["ref_teacher"]),
        "M": margin(dev["R"]["S_rl2arch"], dev["R"]["ref_teacher"]),
        "T_star": dev.get("T_star", {}),
        "tau": dev.get("tau"),
        "dev_reads": dev,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=1)
    print(json.dumps({k: v for k, v in frozen.items() if k != "dev_reads"}, indent=1))
    return 0


CONTRASTS = (
    ("S_ctxarch_rl2in", "S_ctx"),
    ("S_rl2arch", "S_ctx"),
    ("S_rl2arch", "S_ctxarch_rl2in"),
)
# Diagnostic contrasts (no margin, no outcome, no referee): Δ_tok.
DIAG_CONTRASTS = (("S_ctx_tok", "S_ctx"),)
STUDENTS = ("S_ctx", "S_ctxarch_rl2in", "S_rl2arch")


def load_picks(
    specs: list[str], sp: Split, frozen: dict[str, Any] | None
) -> tuple[dict[str, list[np.ndarray]], np.ndarray | None]:
    """`ARM=picks.npz` files (one per training seed), checked against the split
    and, on TEST, against the FROZEN manifest: each file was predicted by a
    frozen checkpoint of the arm it is labelled with (arm and seed as `freeze`
    recorded them), and the files cover every frozen checkpoint exactly once.

    Returns:
        `(arm -> picks per seed, per-decision OR of the files' UNK flags or
        None if a file has none)`.
    """
    picks: dict[str, list[np.ndarray]] = {}
    used: list[str] = []
    unk = np.zeros(sp.n, bool)
    have_unk = True
    for spec in specs:
        arm, path = spec.split("=", 1)
        z = np.load(path)
        if z["ids_sha"].item() != ids_sha(sp):
            raise SystemExit(f"{path}: picks were made on other decisions")
        if frozen is not None:
            if "identity" not in frozen:
                raise SystemExit("FROZEN.json records no arm per checkpoint: freeze again")
            sha = str(z["ckpt_sha256"].item()) if "ckpt_sha256" in z.files else None
            hit = [i for i in frozen["identity"].values() if i["sha256"] == sha]
            if not hit:
                raise SystemExit(f"{path}: not predicted by a frozen checkpoint")
            f_arm = str(z["arm"].item()) if "arm" in z.files else None
            f_seed = int(z["seed"].item()) if "seed" in z.files else None
            if hit[0]["arm"] != arm or f_arm != arm:
                raise SystemExit(
                    f"{path}: labelled {arm}, but predicted by the frozen {hit[0]['arm']} "
                    f"checkpoint (the file says {f_arm})"
                )
            if f_seed != hit[0]["seed"]:
                raise SystemExit(
                    f"{path}: seed {f_seed}, the frozen checkpoint's is {hit[0]['seed']}"
                )
            used.append(sha)
        picks.setdefault(arm, []).append(z["pick"])
        if "has_unk" in z.files:
            unk |= z["has_unk"].astype(bool)
        else:
            have_unk = False
    if frozen is not None:
        want = collections.Counter(i["sha256"] for i in frozen["identity"].values())
        got = collections.Counter(used)
        if got != want:
            missing = sorted(
                f"{i['arm']}@seed{i['seed']}"
                for i in frozen["identity"].values()
                if got[i["sha256"]] < 1
            )
            extra = sorted(h[:12] for h, n in got.items() if n > want.get(h, 0))
            raise SystemExit(
                f"the pick files must cover every frozen checkpoint once: missing {missing}, "
                f"repeated {extra}"
            )
    return picks, (unk if have_unk else None)


def disagreement_ids(sp: Split, picks: dict[str, list[np.ndarray]]) -> list[str]:
    """E decisions on which the frozen student picks are not all one class (§5.5)."""
    a = sp.a
    cls = np.stack([a["cls"][p] for arm in STUDENTS if arm in picks for p in picks[arm]])
    dis = (cls != cls[0]).any(0) & a["in_E"]
    return [sp.meta["ids"][i] for i in np.flatnonzero(dis)]


def referee_regrets(
    sp: Split, picks: dict[str, list[np.ndarray]], scores: dict[str, dict[str, Any]], referee: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(subset decision indices, arm -> [S, m] regret under the referee)`."""
    idx = {i: d for d, i in enumerate(sp.meta["ids"])}
    sub = np.asarray(sorted(idx[i] for i in scores if "error" not in scores[i]), np.int64)
    out = {}
    for arm, ps in picks.items():
        regs = []
        for p in ps:
            r = []
            for d in sub:
                q = np.asarray(scores[sp.meta["ids"][d]][referee], np.float64)
                c = int(sp.a["cls"][p[d]])
                r.append(np.nanmax(q) - q[c])
            regs.append(r)
        out[arm] = np.asarray(regs)
    return sub, out


def qualify(table: dict[str, Any], referee: dict[str, Any] | None) -> None:
    """The §7.3 qualifiers on a PASS (referee, agreement) and the CEILING label."""
    arms = table["arms"]
    for name, c in table["contrasts"].items():
        ctrl, trt = name.split("-", 1)
        out = c.get("outcome")
        if out is None:
            continue
        if out == "PASS":
            ref = (referee or {}).get(name)
            if ref is None or not (ref["F1"]["delta"] > 0 and ref["F2"]["delta"] > 0):
                c["outcome"] = "PASS (root-Q only)"
            elif arms[trt]["Agr"] < arms[ctrl]["Agr"] - 0.01:
                c["outcome"] = "PASS (regret only)"
        if out in ("NULL", "SMALL") and arms[ctrl]["R"] - arms["ref_teacher"]["R"] <= 0:
            c["label"] = "CEILING (teacher-limited)"


def cmd_eval(args: argparse.Namespace) -> int:
    """Score pick files on a split. TEST needs `--final --stage` and FROZEN.

    The sealed TEST read is two calls: `--disagreements IDS` writes the
    referee subset (no metric is computed), `referee` scores it, and
    `--referee scores.jsonl` then computes the table, the referee contrasts,
    the qualifiers and the §5.7 / §6 diagnostics, writes `stage<k>.json` and
    seals the stage. `--e-report` prints the per-rule E removal counts alone
    (no picks, no seal; on TEST once the students are frozen), so the rule-3
    red flag is checked before the read.
    """
    split = args.split
    if args.e_report:
        if split == "test":
            if not args.final_dir:
                raise SystemExit("--e-report on TEST needs --final-dir (freeze the students first)")
            check_frozen(args.final_dir, [])
        rep = e_rule_report(Split(args.cache, split))
        text = json.dumps(rep, indent=1)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        return 0
    if not args.picks:
        raise SystemExit("--picks ARM=picks.npz ... is required")
    frozen = None
    margins = None
    if split == "test":
        if not args.final:
            raise SystemExit("TEST is sealed: --final --stage 1|2 with a FROZEN manifest")
        frozen = check_frozen(args.final_dir, [])
        sealed = os.path.join(args.final_dir, f"SEALED-{args.stage}")
        if os.path.exists(sealed):
            raise SystemExit(f"{sealed} exists: stage {args.stage} has been read")
        if args.stage == 2:
            if not os.path.exists(os.path.join(args.final_dir, "SEALED-1")):
                raise SystemExit("stage 2 needs a stage-1 read")
            with open(os.path.join(args.final_dir, "stage1.json"), encoding="utf-8") as f:
                if (
                    json.load(f)["contrasts"]["S_ctxarch_rl2in-S_ctx"].get("outcome")
                    != "INCONCLUSIVE"
                ):
                    raise SystemExit("stage 2 only follows a stage-1 INCONCLUSIVE on Δ_ctx")
        margins = {
            "S_ctxarch_rl2in-S_ctx": frozen["M_ctx"],
            "S_rl2arch-S_ctx": frozen["M"],
            "S_rl2arch-S_ctxarch_rl2in": frozen["M"],
        }
    sp = Split(args.cache, split)
    picks, unk = load_picks(args.picks, sp, frozen)
    if args.disagreements:
        ids = disagreement_ids(sp, picks)
        with open(args.disagreements, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        print(f"{len(ids)} disagreement decisions written to {args.disagreements}")
        return 0
    scores = None
    if args.referee:
        with open(args.referee, encoding="utf-8") as f:
            scores = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        missing = set(disagreement_ids(sp, picks)) - set(scores)
        if missing:
            raise SystemExit(f"{len(missing)} disagreement decisions lack referee scores")
    elif split == "test":
        raise SystemExit("the sealed read needs --referee scores.jsonl (run --disagreements first)")
    contrasts = [(c, t) for c, t in CONTRASTS if c in picks and t in picks]
    diag = [(c, t) for c, t in DIAG_CONTRASTS if c in picks and t in picks]
    if frozen is not None:
        tau = frozen.get("tau")
    else:
        tau = contested_tau(sp) if sp.a["has_B"].any() else None
    table = evaluate(
        sp, picks, contrasts + diag, margins, args.stage, n_boot=args.n_boot, tau=tau, unk=unk
    )
    table["picks"] = {arm: len(ps) for arm, ps in picks.items()}
    ref_table = None
    if scores is not None:
        ref_table = {}
        n_e = int(sp.a["in_E"].sum())
        for c, t in contrasts:
            row = {}
            for f in ("F1", "F2"):
                sub, regs = referee_regrets(sp, picks, scores, f)
                if len(sub) == 0:
                    row[f] = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "subset": 0}
                    continue
                share = len(sub) / max(1, n_e)
                res = bootstrap_contrast(regs[c], regs[t], sp.run[sub], n_boot=args.n_boot)
                row[f] = {
                    k: (v * share if k in ("delta", "lo", "hi", "se") else v)
                    for k, v in res.items()
                }
                row[f]["subset"] = len(sub)
            ref_table[f"{c}-{t}"] = row
        table["referee"] = ref_table
    qualify(table, ref_table)
    if args.dev_reads_out:
        if split != "dev":
            raise SystemExit("--dev-reads-out reads DEV only")
        reads = {"R": {k: v["R"] for k, v in table["arms"].items()}, "tau": contested_tau(sp)}
        with open(args.dev_reads_out, "w", encoding="utf-8") as f:
            json.dump(reads, f, indent=1)
    text = json.dumps(table, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    if split == "test":
        with open(
            os.path.join(args.final_dir, f"stage{args.stage}.json"), "w", encoding="utf-8"
        ) as f:
            f.write(text)
        with open(os.path.join(args.final_dir, f"SEALED-{args.stage}"), "w", encoding="utf-8") as f:
            f.write(sha256(os.path.join(args.final_dir, f"stage{args.stage}.json")) + "\n")
    return 0


def ids_sha(sp: Split) -> str:
    return hashlib.sha256("\n".join(sp.meta["ids"]).encode()).hexdigest()


def add_parser(sub: Any) -> None:
    f = sub.add_parser("freeze", help="write final/FROZEN.json (checkpoints, margins, T*, tau)")
    f.add_argument("--final-dir", required=True)
    f.add_argument("--ckpt", nargs="+", required=True)
    f.add_argument("--dev-reads", required=True, help="JSON: R per arm on DEV, T_star, tau")
    f.add_argument(
        "--seeds-per-arm",
        type=int,
        default=SEEDS_PER_ARM,
        help="training seeds every student arm must have (PREREG: 3; the smoke uses 1)",
    )
    f.set_defaults(func=cmd_freeze)
    e = sub.add_parser("eval", help="regret / agreement / CIs of pick files on a split")
    e.add_argument("--cache", required=True)
    e.add_argument("--split", default="dev")
    e.add_argument("--picks", nargs="*", default=[], help="ARM=picks.npz (one per seed)")
    e.add_argument("--final", action="store_true")
    e.add_argument("--stage", type=int, default=1)
    e.add_argument("--final-dir")
    e.add_argument("--disagreements", help="write the referee subset's ids here and stop")
    e.add_argument("--referee", help="referee scores.jsonl covering the disagreement subset")
    e.add_argument("--dev-reads-out", help="DEV: write R per arm and tau for `freeze`")
    e.add_argument(
        "--e-report",
        action="store_true",
        help="print the per-rule E removal counts only (no picks, no seal)",
    )
    e.add_argument("--n-boot", type=int, default=N_BOOT)
    e.add_argument("--out")
    e.set_defaults(func=cmd_eval)
