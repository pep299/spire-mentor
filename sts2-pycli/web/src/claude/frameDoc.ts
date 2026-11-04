import type { DockItem } from "./useClaudeLink";

/**
 * The sandboxed iframe document for a Claude dock item (html / markdown).
 * Every document carries a tiny script that posts its height to the dock, so
 * the frame fits its content without the page reaching into the frame.
 */

const MARKED_CDN = "https://cdn.jsdelivr.net/npm/marked@15/marked.min.js";
const PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3/plotly.min.js";

/** JSON safe to inline in a `<script>` (no `</script>`, no `<!--`). */
function inlineJson(value: unknown): string {
  return JSON.stringify(value ?? null).replace(/</g, "\\u003c");
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

/** Posts the document's height to the dock so the frame fits its content —
 *  only when it changed, so a frame set to the reported height is a fixed
 *  point (the dock never pads it: a padded frame grew without end on content
 *  sized to the viewport, `height:100%` / `100vh`). */
function heightReporter(token: string): string {
  return `<script>(function(){var t=${JSON.stringify(token)},last=-1;function post(){var h=Math.ceil(document.documentElement.getBoundingClientRect().height);if(h===last)return;last=h;parent.postMessage({claudeDock:t,h:h},"*")}addEventListener("load",post);if(window.ResizeObserver)new ResizeObserver(post).observe(document.documentElement);setTimeout(post,50)})()</script>`;
}

const BASE_STYLE =
  "<style>body{font:14px/1.45 system-ui,sans-serif;margin:10px;color:#1d2330;background:#fff}" +
  "table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:3px 6px}" +
  "code,pre{font:12px ui-monospace,monospace;background:#f2f3f7}pre{padding:8px;overflow:auto}</style>";

/** The iframe document for a dock item. Plotly runs in here too, not in the
 *  page: a figure's `config` / layout come from Claude, and the page's origin
 *  holds the worker and the run. Both renderers load from a CDN — the link
 *  only exists while Claude Code (online) is running. */
export function frameDoc(item: DockItem, token: string): string {
  const reporter = heightReporter(token);
  if (item.kind === "plotly") {
    return (
      `<!doctype html><html><head><meta charset="utf-8">${BASE_STYLE}</head><body style="margin:0">` +
      `<div id="plot"></div><pre id="err" style="color:#b00;white-space:pre-wrap"></pre>` +
      `<script id="fig" type="application/json">${inlineJson(item.content)}</script>` +
      `<script src="${PLOTLY_CDN}"></script>` +
      `<script>(function(){var f=JSON.parse(document.getElementById("fig").textContent)||{};` +
      `if(!window.Plotly){document.getElementById("err").textContent="Plotly could not load (offline?)";return}` +
      `var l=Object.assign({autosize:true,height:360,margin:{t:40,r:12,b:40,l:48}},f.layout||{});` +
      `Plotly.newPlot("plot",f.data||[],l,{responsive:true,displaylogo:false}).catch(function(e){document.getElementById("err").textContent=String(e)})})()</script>` +
      `${reporter}</body></html>`
    );
  }
  if (item.kind === "markdown") {
    const md = String(item.content ?? "");
    // Rendered by `marked` from a CDN; offline, the raw markdown stays readable.
    return (
      `<!doctype html><html><head><meta charset="utf-8">${BASE_STYLE}</head><body>` +
      `<pre id="raw" style="white-space:pre-wrap;background:none">${escapeHtml(md)}</pre>` +
      // JSON, not a text/markdown block: inlineJson escapes every `<`, so no
      // `</script>` or `<!--` in the markdown can end or swallow the block.
      `<div id="out"></div><script id="md" type="application/json">${inlineJson(md)}</script>` +
      `<script src="${MARKED_CDN}"></script>` +
      `<script>if(window.marked){document.getElementById("out").innerHTML=marked.parse(JSON.parse(document.getElementById("md").textContent));document.getElementById("raw").remove()}</script>` +
      `${reporter}</body></html>`
    );
  }
  const html = String(item.content ?? "");
  const i = html.toLowerCase().lastIndexOf("</body>");
  if (i >= 0) return html.slice(0, i) + reporter + html.slice(i);
  return `<!doctype html><html><head><meta charset="utf-8">${BASE_STYLE}</head><body>${html}${reporter}</body></html>`;
}

export const MIN_FRAME_H = 60;
export const MAX_FRAME_H = 2000;

/** The frame height after a reported content height `h`. A document whose
 *  height follows the frame's own (`100vh` plus a margin) reports a little
 *  more than every height it is given; two growths in a row by the same step
 *  are that loop, and the frame stops there instead of creeping to the cap. */
export function nextFrameHeight(cur: number, h: number, lastGrowth: number | null):
  { height: number; growth: number | null; loop: boolean } {
  const want = Math.max(MIN_FRAME_H, Math.min(Math.ceil(h), MAX_FRAME_H));
  const growth = want - cur;
  if (growth > 0 && growth === lastGrowth) return { height: cur, growth: null, loop: true };
  return { height: want, growth: growth > 0 ? growth : null, loop: false };
}
