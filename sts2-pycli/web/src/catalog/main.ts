import type { Catalog, CatalogEdge, QuadrantSet, SortKey, SortDir } from "./data";
import { filterByQuadrant, sortRows } from "./data";
import { renderHeatmap } from "./heatmap";
import { versionedPublicPath } from "../gameVersion";
import type { CardTextEntry } from "./heatmap";

type Quadrant = CatalogEdge["quadrant"];

const ALL_QUADRANTS: Quadrant[] = ["confirmed", "myth", "undiscovered", "noise"];

const QUADRANT_COLOR: Record<Quadrant, string> = {
  confirmed: "#2ca02c",
  myth: "#ff7f0e",
  undiscovered: "#9467bd",
  noise: "#aaa",
};

const QUADRANT_LABEL: Record<Quadrant, string> = {
  confirmed: "Confirmed",
  myth: "Myth",
  undiscovered: "Undiscovered",
  noise: "Noise",
};

// Columns: key is either a SortKey or "pair"
const TABLE_COLS: { key: string; label: string; tip?: string }[] = [
  { key: "pair", label: "Pair", tip: "The two cards in this synergy pair. Hover a pair to see card descriptions." },
  {
    key: "pmi",
    label: "PMI",
    tip: "Pointwise Mutual Information: log P(A,B) / (P(A)·P(B)). How much more often the pair appears together than chance. Signal of believed synergy, not proof.",
  },
  {
    key: "econ_S",
    label: "Econ S",
    tip: "Economy synergy: value-net second difference V(D+A+B)−V(D+A)−V(D+B)+V(D). Cheap causal run-economy signal (null if not computed).",
  },
  {
    key: "combat_S",
    label: "Combat S",
    tip: "Combat synergy: forced-deck four-cell interaction in fight outcome. The authoritative causal 'does the combo actually fire' measure (null if not computed).",
  },
  {
    key: "confidence",
    label: "Confidence",
    tip: "Derived from the Combat S confidence interval width: higher = tighter CI, farther from zero. Null when Combat S is not computed.",
  },
  {
    key: "sample_n",
    label: "Sample N",
    tip: "Number of (baseline × encounter × seed) combos behind Combat S, or contexts behind Econ S. Larger = more reliable estimate.",
  },
  {
    key: "quadrant",
    label: "Quadrant",
    tip: "Observational × causal classification. See legend below for definitions.",
  },
];

// ── Characters ───────────────────────────────────────────────────────────────
// Display order + label for the character selector. Each maps to a
// `catalog_<id>.json` produced by the synergy pipeline.
const CHARACTERS: { id: string; label: string }[] = [
  { id: "ironclad", label: "Ironclad" },
  { id: "silent", label: "Silent" },
  { id: "defect", label: "Defect" },
  { id: "regent", label: "Regent" },
  { id: "necrobinder", label: "Necrobinder" },
];
const DEFAULT_CHARACTER = "ironclad";

// ── State ────────────────────────────────────────────────────────────────────
let catalog: Catalog | null = null;
let cardText: Record<string, CardTextEntry> = {};
let activeCharacter: string = DEFAULT_CHARACTER;
// Default: noise OFF (2,615 noise edges collapse the graph; user can turn on)
const enabledQuadrants: QuadrantSet = new Set(["confirmed", "myth", "undiscovered"] as Quadrant[]);
let activeSortKey: string = "pmi";   // may be "pair" or any SortKey
let sortDir: SortDir = "desc";
let cleanupGraph: (() => void) | null = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const statusEl       = document.getElementById("status")          as HTMLParagraphElement;
const metaEl         = document.getElementById("meta")            as HTMLDivElement;
const filtersEl      = document.getElementById("filters")         as HTMLElement;
const heatmapRoot    = document.getElementById("heatmap-root")    as HTMLDivElement;
const tableContainer = document.getElementById("table-container") as HTMLDivElement;
const charSelectEl   = document.getElementById("character-select") as HTMLSelectElement | null;

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(msg: string): void {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

function fmtNum(v: number | null, decimals = 2): string {
  if (v === null) return "—";
  return Number.isInteger(v) ? String(v) : v.toFixed(decimals);
}

// ── Render: meta ─────────────────────────────────────────────────────────────
function renderMeta(): void {
  if (!catalog) return;
  const date = new Date(catalog.generated_at).toLocaleDateString();
  metaEl.textContent =
    `${catalog.character} · ${catalog.run_count.toLocaleString()} runs · generated ${date}`;
}

// ── Render: quadrant filter checkboxes + legend ───────────────────────────────
function renderFilters(): void {
  const legend = document.createElement("div");
  legend.className = "quadrant-legend";

  for (const q of ALL_QUADRANTS) {
    const label = document.createElement("label");
    label.className = "quadrant-cb";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = q;
    cb.checked = enabledQuadrants.has(q);
    cb.addEventListener("change", () => {
      if (cb.checked) enabledQuadrants.add(q);
      else            enabledQuadrants.delete(q);
      renderAll();
    });

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = QUADRANT_COLOR[q];

    label.appendChild(cb);
    label.appendChild(swatch);
    label.appendChild(document.createTextNode(QUADRANT_LABEL[q]));
    legend.appendChild(label);
  }

  filtersEl.appendChild(legend);

  // ── Help / legend block ───────────────────────────────────────────────────
  const helpWrap = document.createElement("div");
  helpWrap.className = "help-wrap";

  const helpBtn = document.createElement("button");
  helpBtn.className = "help-btn";
  helpBtn.textContent = "?";
  helpBtn.title = "Show/hide quadrant and metric definitions";

  const helpBody = document.createElement("div");
  helpBody.className = "help-body";
  helpBody.style.display = "none";
  helpBody.innerHTML = `
    <strong>Quadrant definitions</strong>
    <ul>
      <li><span class="help-swatch" style="background:${QUADRANT_COLOR.confirmed}"></span>
        <strong>Confirmed</strong> — high observational <em>and</em> high causal: real synergy, the catalog core.</li>
      <li><span class="help-swatch" style="background:${QUADRANT_COLOR.myth}"></span>
        <strong>Myth</strong> — high observational, causal ≤ 0: community habit, not proven to actually synergize in combat.</li>
      <li><span class="help-swatch" style="background:${QUADRANT_COLOR.undiscovered}"></span>
        <strong>Undiscovered</strong> — low observational, high causal: combos players underuse but that test out.</li>
      <li><span class="help-swatch" style="background:${QUADRANT_COLOR.noise}"></span>
        <strong>Noise</strong> — low on both metrics.</li>
    </ul>
    <p class="help-note">With only the observational layer loaded (no <code>--combat</code> run), every pair is Myth or Noise until causal columns are computed.</p>
  `;

  helpBtn.addEventListener("click", () => {
    const isHidden = helpBody.style.display === "none";
    helpBody.style.display = isHidden ? "block" : "none";
    helpBtn.classList.toggle("help-btn-active", isHidden);
  });

  helpWrap.appendChild(helpBtn);
  helpWrap.appendChild(helpBody);
  filtersEl.appendChild(helpWrap);
}

// ── Render: clustered heatmap (primary visual) ────────────────────────────────
function renderGraphView(): void {
  if (!catalog) return;
  if (cleanupGraph) { cleanupGraph(); cleanupGraph = null; }
  cleanupGraph = renderHeatmap(heatmapRoot, catalog, cardText);
}

// ── Render: table ────────────────────────────────────────────────────────────
function sortEdges(rows: CatalogEdge[]): CatalogEdge[] {
  if (activeSortKey === "pair") {
    return [...rows].sort((a, b) => {
      const as = a.pair.join("|");
      const bs = b.pair.join("|");
      const cmp = as < bs ? -1 : as > bs ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
  }
  return sortRows(rows, activeSortKey as SortKey, sortDir);
}

function cardTip(token: string): string {
  const entry = cardText[token];
  if (!entry) return token;
  return `${entry.title}\n${entry.description}`;
}

function renderTable(): void {
  if (!catalog) return;

  const filtered = filterByQuadrant(catalog.edges, enabledQuadrants);
  const sorted   = sortEdges(filtered);

  tableContainer.textContent = ""; // clear safely

  if (sorted.length === 0) {
    const p = document.createElement("p");
    p.textContent = "No edges match the selected quadrants.";
    tableContainer.appendChild(p);
    return;
  }

  const table = document.createElement("table");
  table.className = "synergy-table";

  // thead
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const col of TABLE_COLS) {
    const th = document.createElement("th");
    th.style.cursor = "pointer";
    if (col.tip) th.title = col.tip;

    const txt = document.createTextNode(col.label);
    th.appendChild(txt);

    // Show a small indicator that this column has a tooltip
    if (col.tip) {
      const tipMark = document.createElement("span");
      tipMark.className = "col-tip-mark";
      tipMark.textContent = " ⓘ";
      tipMark.title = col.tip;
      th.appendChild(tipMark);
    }

    if (activeSortKey === col.key) {
      th.classList.add("sort-active");
      const arrow = document.createElement("span");
      arrow.className = "sort-arrow";
      arrow.textContent = sortDir === "asc" ? " ▲" : " ▼";
      th.appendChild(arrow);
    }

    const colKey = col.key; // capture for closure
    th.addEventListener("click", () => {
      if (activeSortKey === colKey) {
        sortDir = sortDir === "asc" ? "desc" : "asc";
      } else {
        activeSortKey = colKey;
        sortDir = colKey === "pair" ? "asc" : "desc";
      }
      renderAll();
    });

    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  // tbody
  const tbody = document.createElement("tbody");
  for (const edge of sorted) {
    const tr = document.createElement("tr");

    // Pair — with hover tooltip showing card descriptions
    const pairTd = document.createElement("td");
    const [t0, t1] = edge.pair;
    const e0 = cardText[t0];
    const e1 = cardText[t1];
    const pairLabel = `${e0?.title ?? t0} + ${e1?.title ?? t1}`;
    pairTd.textContent = pairLabel;
    pairTd.title = `${cardTip(t0)}\n\n${cardTip(t1)}`;
    tr.appendChild(pairTd);

    // Numeric columns
    const numFields: Array<keyof Pick<CatalogEdge, "pmi"|"econ_S"|"combat_S"|"confidence"|"sample_n">> =
      ["pmi", "econ_S", "combat_S", "confidence", "sample_n"];
    for (const field of numFields) {
      const td = document.createElement("td");
      td.textContent = fmtNum(edge[field] as number | null);
      tr.appendChild(td);
    }

    // Quadrant (with color swatch)
    const qtd = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = QUADRANT_COLOR[edge.quadrant];
    qtd.appendChild(swatch);
    qtd.appendChild(document.createTextNode(QUADRANT_LABEL[edge.quadrant]));
    tr.appendChild(qtd);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  tableContainer.appendChild(table);
}

// ── Combined render ──────────────────────────────────────────────────────────
function renderAll(): void {
  renderGraphView();
  renderTable();
}

// ── Character selector ─────────────────────────────────────────────────────────
function renderCharSelect(): void {
  if (!charSelectEl) return;
  charSelectEl.textContent = "";
  for (const c of CHARACTERS) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    if (c.id === activeCharacter) opt.selected = true;
    charSelectEl.appendChild(opt);
  }
  charSelectEl.addEventListener("change", () => {
    activeCharacter = charSelectEl.value;
    void loadCharacter(activeCharacter);
  });
}

// Resolve a per-character catalog URL, falling back to the legacy single-file
// catalog.json for Ironclad (kept for backward compatibility).
function catalogUrl(character: string): string {
  return `./catalog_${character}.json`;
}

async function loadCharacter(character: string): Promise<void> {
  setStatus(`Loading ${character}…`);
  try {
    let res = await fetch(catalogUrl(character));
    if (!res.ok && character === DEFAULT_CHARACTER) {
      // Backward-compat fallback to the original single-character file.
      res = await fetch("./catalog.json");
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    catalog = (await res.json()) as Catalog;

    setStatus("");
    renderMeta();
    renderAll();
  } catch (err) {
    setStatus(`Error loading ${character} catalog: ${String(err)}`);
  }
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
async function init(): Promise<void> {
  setStatus("Loading…");

  // Card text is shared across characters — fetch once. Per game build
  // (GH #877): the workbench opens this page with `?game_version=<ver>` for a
  // non-default build's run, whose own text lives at `./<ver>/card_text.json`;
  // a build without one (the public snapshot) reads the default build's.
  // The synergy catalogs themselves are the default build's only: they come
  // from a scraped v0.107 run corpus, so a card new in a later build simply
  // has no pairs here.
  const gameVersion = new URLSearchParams(window.location.search).get("game_version");
  const cardTextUrls = [versionedPublicPath("card_text.json", gameVersion), "/card_text.json"];
  for (const url of new Set(cardTextUrls)) {
    try {
      const cardTextRes = await fetch(`.${url}`);
      if (cardTextRes.ok) {
        cardText = (await cardTextRes.json()) as Record<string, CardTextEntry>;
        break;
      }
    } catch {
      // card text is optional; tooltips fall back to tokens
    }
  }

  // Static chrome that doesn't depend on the catalog.
  renderFilters();
  renderCharSelect();

  await loadCharacter(activeCharacter);
}

void init();
