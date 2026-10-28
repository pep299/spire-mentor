"""Data-to-Divergence Ratio (DDR) over recorded PPO campaigns — retroactive, zero compute.

Beukman et al., "Preventing Learning Stagnation in PPO by Scaling to 1 Million
Parallel Environments" (RLC 2026, `arXiv:2603.06009`) model PPO's OUTER loop as
stochastic optimization: a plateau appears when the sampled loss stops being a
good proxy for the true objective. Their diagnostic is

    DDR = (data points per update) / (KL divergence from the behaviour policy)

failing at both ends — low DDR plateaus early, high DDR learns slowly for a
fixed budget — with the prescription that DDR must RISE with the training
budget to keep improvement monotonic, and that you raise it by adding samples
while holding minibatch size and step size fixed.

Both ingredients are already in ``<save>.stats.jsonl``: ``n_transitions`` and
``kl`` (``trainer.py:362,368``). Every campaign we have ever run can therefore
be scored without collecting a single new game — which is the point, since
collection is ~87% of campaign wall clock and the machine is thermally bound.

**READ THIS BEFORE READING ANY DDR NUMBER.**

1. *Within* a campaign ``games_per_iter`` is a fixed hyperparameter, so
   ``n_transitions`` barely moves (CV 0.8–9%) while ``kl`` swings by 17–500%.
   DDR is then a monotone transform of ``1/kl`` — measured r > 0.92 on every
   campaign we have, > 0.99 on most. **The within-campaign DDR trace carries no
   information that an inverse-KL plot does not.** The ``degeneracy`` section
   prints that correlation so the reader cannot miss it. The paper's DDR varies
   because *they* vary the environment count; ours is a constant.

2. DDR is therefore only informative BETWEEN campaigns that chose different
   ``games_per_iter`` — see the ``regime`` section. That is a real comparison
   for us: 256 (legacy) vs 960 (the PR #482 iso-compute recipe).

3. Rising DDR is NOT automatically health. It rises when KL falls, and KL falls
   both when you add data (good, the paper's mechanism) and when the policy
   stops moving (stagnation, the thing being diagnosed). Late-campaign DDR
   rises here are confounded with the entropy-decay schedule. Only compare DDR
   at *matched training stage* across regimes.

4. The logged ``kl`` is a per-minibatch mean of the k1 estimator, not the
   end-of-update KL from the behaviour policy the paper defines — it
   underestimates by roughly 2x and can go negative. Absolute values are on an
   arbitrary scale. Never compare one to a number printed in the paper, and
   never set a threshold on it.

Usage (stdlib only — no venv, no wheel build):

    python3 python/ppo/ddr.py 'data/ppo/*.stats.jsonl' --html /tmp/ddr.html
    python3 python/ppo/ddr.py 'data/ppo/*.stats.jsonl' '~/ppo-*/*.stats.jsonl' \
        '~/ppo-*/*/*.stats.jsonl' --regime-window 0 60
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# Rolling window for smoothing. Per-iteration win rate over n_runs=256 carries
# SE ~ sqrt(.3*.7/256) ~ 2.9pp, so a raw per-iteration slope is noise; 15
# iterations pulls that to ~0.75pp, below the smallest effect worth naming.
SMOOTH_W = 15

# A plateau is a window whose smoothed win rate gains less than this, in points.
PLATEAU_GAIN_PP = 0.5

# Guard for the k1 KL estimator reaching zero or going negative.
KL_FLOOR = 1e-6

# Above this |r| between DDR and 1/KL, the DDR trace is declared degenerate —
# it is an inverse-KL plot and must not be read as an independent diagnostic.
DEGENERACY_R = 0.9


@dataclass
class Iter:
    it: int
    n_trans: int
    n_runs: int
    kl: float
    win_rate: float
    entropy_coef: float
    ent: float
    ev: float
    clipfrac: float
    collect_s: float
    train_s: float

    @property
    def ddr(self) -> float | None:
        if self.kl <= KL_FLOOR:
            return None
        return self.n_trans / self.kl


def load(path: Path) -> list[Iter]:
    rows: list[Iter] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # a live campaign's last line can be a torn partial write
        if "n_transitions" not in d or "kl" not in d:
            continue
        rows.append(
            Iter(
                it=int(d.get("iteration", len(rows))),
                n_trans=int(d["n_transitions"]),
                n_runs=int(d.get("n_runs") or 0),
                kl=float(d["kl"]),
                win_rate=float(d.get("win_rate", 0.0)),
                entropy_coef=float(d.get("entropy_coef", 0.0)),
                ent=float(d.get("ent", 0.0)),
                ev=float(d.get("explained_variance", 0.0)),
                clipfrac=float(d.get("clipfrac", 0.0)),
                collect_s=float(d.get("collect_s", 0.0)),
                train_s=float(d.get("train_s", 0.0)),
            )
        )
    rows.sort(key=lambda r: r.it)
    return rows


def smooth(xs: list[float | None], w: int = SMOOTH_W) -> list[float | None]:
    """Centred rolling median. Median, not mean: the k1 KL estimator throws
    occasional wild values and a mean lets one of them own the window."""
    out: list[float | None] = []
    for i in range(len(xs)):
        lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
        vals = [v for v in xs[lo:hi] if v is not None]
        out.append(statistics.median(vals) if vals else None)
    return out


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else None


def cv(xs: list[float]) -> float:
    m = statistics.fmean(xs)
    return statistics.pstdev(xs) / m if m else float("nan")


@dataclass
class Campaign:
    name: str
    rows: list[Iter]

    @property
    def games_per_iter(self) -> int:
        return int(statistics.median([r.n_runs for r in self.rows])) or 0

    @property
    def ddr_s(self) -> list[float | None]:
        return smooth([r.ddr for r in self.rows])

    @property
    def wr_s(self) -> list[float | None]:
        return smooth([r.win_rate for r in self.rows])

    def degeneracy(self) -> tuple[float, float, float] | None:
        """(CV of transitions, CV of KL, r between DDR and 1/KL).

        The load-bearing check. See caveat 1 in the module docstring.
        """
        good = [r for r in self.rows if r.kl > KL_FLOOR]
        if len(good) < 10:
            return None
        ddr = [r.n_trans / r.kl for r in good]
        inv = [1.0 / r.kl for r in good]
        r = pearson(inv, ddr)
        return (
            cv([float(x.n_trans) for x in good]),
            cv([x.kl for x in good]),
            r if r is not None else float("nan"),
        )

    def plateaus(self) -> list[tuple[int, int]]:
        """Contiguous spans where smoothed win rate gains < PLATEAU_GAIN_PP."""
        wr = self.wr_s
        flags: list[bool] = []
        for i in range(len(self.rows)):
            j = min(len(self.rows) - 1, i + SMOOTH_W)
            a, b = wr[i], wr[j]
            flags.append(a is not None and b is not None and (b - a) * 100.0 < PLATEAU_GAIN_PP)
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for i, f in enumerate(flags):
            if f and start is None:
                start = i
            elif not f and start is not None:
                if i - start >= SMOOTH_W:
                    spans.append((self.rows[start].it, self.rows[i - 1].it))
                start = None
        if start is not None and len(flags) - start >= SMOOTH_W:
            spans.append((self.rows[start].it, self.rows[-1].it))
        return spans

    def window(self, lo: int, hi: int) -> list[Iter]:
        return [r for r in self.rows if lo <= r.it < hi and r.kl > KL_FLOOR]


def section_degeneracy(camps: list[Campaign]) -> str:
    out = ["## Is DDR anything but 1/KL here?\n"]
    out.append(
        "`games_per_iter` is fixed within a campaign, so `n_transitions` is near-constant "
        "and DDR collapses onto `1/kl`. Where |r| > "
        f"{DEGENERACY_R}, the within-campaign DDR trace is **not an independent diagnostic**.\n"
    )
    out.append("| campaign | games/it | CV(trans) | CV(kl) | r(DDR, 1/KL) | verdict |")
    out.append("|---|---:|---:|---:|---:|---|")
    for c in camps:
        d = c.degeneracy()
        if d is None:
            out.append(f"| `{c.name}` | {c.games_per_iter} | — | — | — | too short |")
            continue
        cvt, cvk, r = d
        verdict = "**degenerate**" if abs(r) > DEGENERACY_R else "carries signal"
        out.append(
            f"| `{c.name}` | {c.games_per_iter} | {cvt:.4f} | {cvk:.3f} | {r:.4f} | {verdict} |"
        )
    return "\n".join(out)


def section_regime(camps: list[Campaign], lo: int, hi: int) -> str:
    """DDR across `games_per_iter` regimes at matched training stage.

    The only axis on which DDR can carry information for us (caveat 2).
    """
    out = [f"## DDR by games/iter regime — iterations [{lo}, {hi})\n"]
    out.append(
        "Matched training stage, since DDR drifts with convergence and the entropy "
        "schedule (caveat 3). This is the comparison the paper's mechanism actually "
        "predicts: more samples per update at unchanged step size ⇒ higher DDR.\n"
    )
    out.append(
        "| campaign | games/it | trans/it | KL | DDR | clipfrac | EV | entropy | WR | train% |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    groups: dict[int, list[float]] = {}
    for c in sorted(camps, key=lambda x: (x.games_per_iter, x.name)):
        w = c.window(lo, hi)
        if len(w) < 5:
            continue

        def med(f, rows: list[Iter] = w) -> float:
            return statistics.median([f(r) for r in rows])

        ddr = med(lambda r: r.n_trans / r.kl)
        if c.games_per_iter:  # older rows lack `n_runs`; ungrouped, still listed
            groups.setdefault(c.games_per_iter, []).append(ddr)
        # Share of iteration wall clock spent in the optimizer. `epochs` is the
        # knob the paper prescribes moving when samples per update grow, and
        # this prices it: cost of epochs 2->N is roughly (N/2 - 1) x train%.
        cs, ts = med(lambda r: r.collect_s), med(lambda r: r.train_s)
        train_pct = 100 * ts / (cs + ts) if (cs + ts) else float("nan")
        out.append(
            f"| `{c.name}` | {c.games_per_iter} | {med(lambda r: r.n_trans):.0f} "
            f"| {med(lambda r: r.kl):.5f} | {ddr:,.0f} | {med(lambda r: r.clipfrac):.3f} "
            f"| {med(lambda r: r.ev):+.3f} | {med(lambda r: r.ent):.3f} "
            f"| {med(lambda r: r.win_rate) * 100:.1f}% | {train_pct:.1f}% |"
        )
    if len(groups) > 1:
        out.append("\n**Regime medians**\n")
        out.append("| games/it | campaigns | median DDR | vs smallest |")
        out.append("|---:|---:|---:|---:|")
        base = statistics.median(groups[min(groups)])
        for g in sorted(groups):
            m = statistics.median(groups[g])
            out.append(f"| {g} | {len(groups[g])} | {m:,.0f} | {m / base:.2f}× |")
    return "\n".join(out)


def section_plateaus(camps: list[Campaign]) -> str:
    out = ["## Plateau spans (context only — see caveat 1)\n"]
    out.append(
        f"Spans where smoothed win rate gains < {PLATEAU_GAIN_PP}pp over {SMOOTH_W} "
        "iterations. Reported to locate stalls, **not** as DDR evidence: within a "
        "campaign the DDR contrast is an inverse-KL contrast.\n"
    )
    out.append("| campaign | iters | plateau spans | % of run |")
    out.append("|---|---:|---|---:|")
    for c in camps:
        spans = c.plateaus()
        n = sum(b - a + 1 for a, b in spans)
        txt = ", ".join(f"{a}–{b}" for a, b in spans) or "none"
        out.append(f"| `{c.name}` | {len(c.rows)} | {txt} | {100 * n / max(1, len(c.rows)):.0f}% |")
    return "\n".join(out)


def svg_chart(c: Campaign, w: int = 760, h: int = 200) -> str:
    """Twin-axis line chart: smoothed DDR (left) and win rate (right), plateau
    spans shaded. Inline SVG so the report needs no plotting library."""
    pad_l, pad_r, pad_t, pad_b = 62, 54, 14, 24
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    if not c.rows:
        return ""
    i0, i1 = c.rows[0].it, c.rows[-1].it
    span_i = max(1, i1 - i0)
    ddr_vals = [d for d in c.ddr_s if d is not None]
    wr_vals = [v for v in c.wr_s if v is not None]
    if not ddr_vals or not wr_vals:
        return ""
    ds = sorted(ddr_vals)
    # Clip at p95: one near-zero k1 KL spikes DDR and flattens everything else.
    dmax = ds[min(len(ds) - 1, int(0.95 * len(ds)))]
    dmin = ds[0]
    dspan = max(1e-9, dmax - dmin)
    wmin, wmax = min(wr_vals), max(wr_vals)
    wspan = max(1e-9, wmax - wmin)

    def px(it: int) -> float:
        return pad_l + iw * (it - i0) / span_i

    def py_d(v: float) -> float:
        return pad_t + ih * (1 - min(1.0, max(0.0, (v - dmin) / dspan)))

    def py_w(v: float) -> float:
        return pad_t + ih * (1 - (v - wmin) / wspan)

    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(c.name)}">'
    ]
    for a, b in c.plateaus():
        parts.append(
            f'<rect x="{px(a):.1f}" y="{pad_t}" width="{max(1.0, px(b) - px(a)):.1f}" '
            f'height="{ih}" fill="var(--plateau)"/>'
        )
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{iw}" height="{ih}" fill="none" stroke="var(--grid)"/>'
    )

    def path(vals: list[float | None], fn, colour: str) -> str:
        pts, pen = [], "M"
        for r, v in zip(c.rows, vals):
            if v is None:
                pen = "M"
                continue
            pts.append(f"{pen}{px(r.it):.1f},{fn(v):.1f}")
            pen = "L"
        return (
            f'<path d="{" ".join(pts)}" fill="none" stroke="{colour}" stroke-width="1.7"/>'
            if pts
            else ""
        )

    parts.append(path(c.ddr_s, py_d, "var(--ddr)"))
    parts.append(path(c.wr_s, py_w, "var(--wr)"))
    parts.append(
        f'<text x="4" y="{pad_t + 10}" class="ax ddr">{dmax / 1e6:.1f}M</text>'
        f'<text x="4" y="{pad_t + ih}" class="ax ddr">{dmin / 1e6:.1f}M</text>'
        f'<text x="{w - pad_r + 6}" y="{pad_t + 10}" class="ax wr">{wmax * 100:.0f}%</text>'
        f'<text x="{w - pad_r + 6}" y="{pad_t + ih}" class="ax wr">{wmin * 100:.0f}%</text>'
        f'<text x="{pad_l}" y="{h - 6}" class="ax">it {i0}</text>'
        f'<text x="{w - pad_r - 34}" y="{h - 6}" class="ax">it {i1}</text></svg>'
    )
    return "".join(parts)


def _inline(s: str) -> str:
    s = html.escape(s)
    for a, b, c2 in (("**", "<strong>", "</strong>"), ("`", "<code>", "</code>")):
        while s.count(a) >= 2:
            s = s.replace(a, b, 1).replace(a, c2, 1)
    return s


def md_to_html(md: str) -> str:
    """Minimal md → html for the tables this module emits (no dependency)."""
    out: list[str] = []
    in_tbl = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [x.strip() for x in s.strip("|").split("|")]
            if all(set(x) <= set("-: ") for x in cells):
                continue
            tag = "th" if not in_tbl else "td"
            if not in_tbl:
                out.append('<div class="wrap"><table>')
                in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(x)}</{tag}>" for x in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table></div>")
            in_tbl = False
        if s.startswith("## "):
            out.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s:
            out.append(f"<p>{_inline(s)}</p>")
    if in_tbl:
        out.append("</table></div>")
    return "\n".join(out)


CSS = """
:root { --bg:#fbfbfa; --fg:#1c1b19; --muted:#6b6862; --grid:#d9d6d0;
  --ddr:#2d6fb3; --wr:#b5541f; --plateau:rgba(120,120,120,.10); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#16151a; --fg:#e8e6e1; --muted:#9b968e; --grid:#3a3843;
  --ddr:#78b4ea; --wr:#e8975c; --plateau:rgba(200,200,200,.09); } }
:root[data-theme="dark"] { --bg:#16151a; --fg:#e8e6e1; --muted:#9b968e;
  --grid:#3a3843; --ddr:#78b4ea; --wr:#e8975c; --plateau:rgba(200,200,200,.09); }
body { background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem;
  max-width:58rem; font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
h1 { font-size:1.5rem; margin:0 0 .5rem; }
h2 { font-size:1.05rem; margin:2rem 0 .5rem; }
h3 { font-size:.9rem; margin:1.1rem 0 .2rem; color:var(--muted); font-weight:600;
  font-family:ui-monospace,monospace; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { border-bottom:1px solid var(--grid); padding:.32rem .5rem; text-align:left;
  white-space:nowrap; }
td:not(:first-child), th:not(:first-child) { text-align:right; }
.wrap { overflow-x:auto; }
.legend { color:var(--muted); font-size:13px; }
.sw { display:inline-block; width:.7rem; height:.7rem; border-radius:2px; }
.ax { font:10.5px ui-monospace,monospace; fill:var(--muted); }
.ax.ddr { fill:var(--ddr); } .ax.wr { fill:var(--wr); }
"""


def write_html(camps: list[Campaign], out: Path, body: str) -> None:
    charts = "".join(
        f"<section><h3>{html.escape(c.name)} · {c.games_per_iter} games/it</h3>{svg_chart(c)}</section>"
        for c in camps
        if svg_chart(c)
    )
    out.write_text(
        f"<title>PPO DDR diagnostic</title>\n<style>{CSS}</style>\n"
        "<h1>PPO Data-to-Divergence Ratio</h1>\n"
        '<p class="legend"><span class="sw" style="background:var(--ddr)"></span> smoothed DDR '
        '(left) &nbsp; <span class="sw" style="background:var(--wr)"></span> smoothed win rate '
        '(right) &nbsp; <span class="sw" style="background:var(--plateau)"></span> plateau span</p>\n'
        '<p class="legend">Diagnostic from arXiv:2603.06009, reconstructed from '
        "<code>stats.jsonl</code>. Trend only: the logged KL is a per-minibatch mean of the k1 "
        "estimator, so the absolute scale is not the paper's.</p>\n"
        f"{md_to_html(body)}\n<h2>Per-campaign traces</h2>\n{charts}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    ap.add_argument("paths", nargs="+", help="stats.jsonl files or globs")
    ap.add_argument("--html", type=Path, help="write a self-contained HTML report here")
    ap.add_argument("--min-iters", type=int, default=25, help="skip shorter campaigns")
    ap.add_argument(
        "--regime-window",
        type=int,
        nargs=2,
        default=(0, 60),
        metavar=("LO", "HI"),
        help="iteration window for the cross-regime comparison",
    )
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        files.extend(Path(x) for x in sorted(glob.glob(str(Path(p).expanduser()), recursive=True)))

    camps: list[Campaign] = []
    seen: set[str] = set()
    for f in files:
        rows = load(f)
        if len(rows) < args.min_iters:
            continue
        name = f.name.removesuffix(".stats.jsonl").removeprefix("ppo_fullrun_")
        if name in ("main", "leg", "polish"):
            parent = f.parent.name
            if parent in ("a", "b"):  # screen arms: <campaign>/<arm>/main
                parent = f"{f.parent.parent.name}/{parent}"
            name = f"{parent}/{name}"
        if name in seen:
            continue
        seen.add(name)
        camps.append(Campaign(name=name, rows=rows))

    if not camps:
        raise SystemExit("no campaigns with enough iterations")

    lo, hi = args.regime_window
    body = "\n\n".join(
        [
            section_degeneracy(camps),
            section_regime(camps, lo, hi),
            section_plateaus(camps),
        ]
    )
    print(body)
    if args.html:
        write_html(camps, args.html, body)
        print(f"\n# wrote {args.html}")


if __name__ == "__main__":
    main()
