import { useEffect, useState } from "react";
import { api, type BugReportDump } from "../api";
import {
  importReportPosition,
  loadReportIntoWorkbench,
  reportGamePosition,
  reportLoadKind,
  type LoadedReport,
} from "../bugreport";
import { isStaticMode } from "../transport";

/** Read-only inspection view for a captured bug report (no live engine).
 *
 *  In the browser build a report can also be TAKEN: a live report carries the
 *  reporter's persisted tree, a replay report the run log, and "Load into this
 *  workbench" puts either into this browser's engine (src/bugreport.ts) —
 *  the live run resumes on the reload, a log shows up in the Runs list.
 *
 *  A report from the real game (the mod's in-game Report a bug, or the
 *  mod-bundled workbench's) also carries the game's position: "Import the
 *  game position" rebuilds it as the live run, like the state import. Mod
 *  reports add the mod / game build, the coach mode and the log tails. */
export default function BugReportView({ id }: { id: string }) {
  const [dump, setDump] = useState<BugReportDump | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState<LoadedReport | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api
      .fetchBugReport(id)
      .then((d) => live && setDump(d))
      .catch(() => live && setNotFound(true));
    return () => {
      live = false;
    };
  }, [id]);

  if (notFound) return <div className="bugreport-view">Report not found: {id}</div>;
  if (!dump) return <div className="bugreport-view">Loading {id}…</div>;

  const meta = dump.meta as Record<string, unknown>;
  const loadKind = isStaticMode() ? reportLoadKind(dump) : null;
  const position = reportGamePosition(dump);
  const fromMod = meta.source === "mod";
  const coach = (meta.coach ?? null) as Record<string, unknown> | null;
  const logs = dump.logs ?? null;

  async function importPosition() {
    setImporting(true);
    setImportError(null);
    try {
      await importReportPosition(dump!);
      setImported(true);
    } catch (e) {
      setImportError(e instanceof Error ? e.message : "import failed");
    } finally {
      setImporting(false);
    }
  }

  async function load() {
    setLoading(true);
    setLoadError(null);
    try {
      setLoaded(await loadReportIntoWorkbench(dump!));
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "load failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bugreport-view">
      <h1>Bug report {String(meta.id ?? id)}</h1>
      <dl>
        <dt>captured</dt><dd>{String(meta.captured_at ?? "")}</dd>
        <dt>mode</dt><dd>{String(meta.mode ?? "")}{fromMod ? " (from the game: Spire Mentor)" : ""}</dd>
        <dt>character</dt><dd>{String(meta.character ?? "")}</dd>
        {fromMod ? (
          <>
            <dt>game / mod</dt><dd>{String(meta.game_version ?? "")} / Spire Mentor {String(meta.mod_version ?? "")}</dd>
            <dt>run</dt><dd>seed {String(meta.seed ?? "?")} · A{String(meta.ascension ?? "?")} · act {String(meta.act ?? "?")} floor {String(meta.floor ?? "?")}</dd>
            {coach && <><dt>coach</dt><dd>{String(coach.mode ?? "")} / {String(coach.style ?? "")} · advisor {coach.advisor_connected ? "connected" : "not connected"}</dd></>}
            {meta.os != null && <><dt>os</dt><dd>{String(meta.os)}</dd></>}
            {meta.captured_local != null && <><dt>player's clock</dt><dd>{String(meta.captured_local)}</dd></>}
          </>
        ) : (
          <><dt>engine</dt><dd>{String(meta.engine_build ?? "")} @ {String(meta.workbench_commit ?? "")}</dd></>
        )}
        {meta.host != null && <><dt>host</dt><dd>{String(meta.host)}</dd></>}
        {meta.user_agent != null && <><dt>browser</dt><dd>{String(meta.user_agent)}</dd></>}
        <dt>screen / overlay</dt><dd>{String(dump.ui.screen)} / {String(dump.ui.overlay)}</dd>
      </dl>
      {dump.note && <p className="bugreport-note">Note: {dump.note}</p>}
      {loadKind && !loaded && (
        <p className="bugreport-load">
          <button type="button" onClick={load} disabled={loading} data-testid="bugreport-load"
            title={loadKind === "live"
              ? "replace this browser's live run with the reporter's (it resumes where they stood)"
              : "add the reporter's run log to this browser's Runs list"}>
            {loading ? "Loading…" : loadKind === "live" ? "▶ Load into this workbench" : "▶ Add run to this workbench"}
          </button>
          {loadError && <span className="bugreport-error"> {loadError}</span>}
        </p>
      )}
      {position && !imported && (
        <p className="bugreport-load">
          <button type="button" onClick={importPosition} disabled={importing} data-testid="bugreport-import-position"
            title="rebuild the position the game was in as this workbench's live run (replaces the live run)">
            {importing ? "Importing…" : "▶ Import the game position"}
          </button>
          {importError && <span className="bugreport-error"> {importError}</span>}
        </p>
      )}
      {imported && (
        <p className="bugreport-load" data-testid="bugreport-imported">
          Imported — the game position is now the live run. <a href="/">Open the workbench</a>
        </p>
      )}
      {loaded && (
        <p className="bugreport-load" data-testid="bugreport-loaded">
          {loaded.kind === "live"
            ? "Loaded — this is now the live run. "
            : `Stored as ${loaded.path} — open it from the Runs list. `}
          <a href="/">Open the workbench</a>
        </p>
      )}
      <h2>repro</h2>
      <pre>{JSON.stringify(reproForDisplay(dump.repro), null, 2)}</pre>
      <h2>state</h2>
      <pre>{JSON.stringify(dump.state, null, 2)}</pre>
      {logs && (
        <>
          <h2>logs</h2>
          {Array.isArray(logs.mod) && (
            <details><summary>Spire Mentor action log ({logs.mod.length} lines)</summary><pre>{logs.mod.join("\n")}</pre></details>
          )}
          {typeof logs.game === "string" && (
            <details><summary>godot.log (tail, {logs.game.length} chars)</summary><pre>{logs.game}</pre></details>
          )}
          {typeof logs.advisor === "string" && (
            <details><summary>advisor stdout (tail, {logs.advisor.length} chars)</summary><pre>{logs.advisor}</pre></details>
          )}
        </>
      )}
      {dump.settings && (
        <details><summary>mod settings</summary><pre>{JSON.stringify(dump.settings, null, 2)}</pre></details>
      )}
    </div>
  );
}

/** The repro block without its bulk: a run log is thousands of lines and a
 *  persisted tree hundreds of nodes; show their size, not their bodies. */
function reproForDisplay(repro: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...repro };
  if (typeof out.run_log === "string") out.run_log = `<${out.run_log.length} chars, ${out.run_log.split("\n").filter(Boolean).length} records>`;
  const p = out.persist as { nodes?: unknown[] } | undefined;
  if (p && typeof p === "object") out.persist = `<persisted tree, ${Array.isArray(p.nodes) ? p.nodes.length : "?"} nodes>`;
  const save = out.run_save as { snapshot?: unknown; save?: unknown } | undefined;
  if (save && typeof save === "object" && save.save != null) {
    out.run_save = { snapshot: save.snapshot ?? null, save: `<run save, ${JSON.stringify(save.save).length} chars>` };
  }
  return out;
}
