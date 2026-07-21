const LABS = [
  [1, "Single-Hop Neighbor Copy", "Move one tile to a logical neighbor with remote DMA, entry barriers, and semaphore discipline; check the result against a reference permutation.", "foundations", "lab1_single_hop", ["DMA", "CORRECTNESS"]],
  [2, "Token-Passing Ring", "Build a dependent hop chain, make collective IDs explicit, and observe how per-hop latency accumulates around a ring.", "foundations", "lab2_token_ring", ["RING", "LATENCY"]],
  [3, "Pallas Memory Spaces", "Stage HBM data through VMEM, compute on tiles, and write back while respecting layout and memory-space constraints.", "foundations", "lab3_memory_spaces", ["HBM", "VMEM"]],
  [4, "Semaphore Bug Zoo", "Catalog missing waits, over-waits, over-signals, and buffer races so synchronization failures become recognizable invariants.", "foundations", "lab4_semaphore_bug_zoo", ["SYNC", "DEBUG"]],
  [5, "Ring All-Gather", "Track arrival order and chunk ownership while composing a custom ring all-gather with an explicit byte model.", "collectives", "lab5_ring_all_gather", ["GATHER", "RING"]],
  [6, "Ring Reduce-Scatter", "Assign chunk ownership, combine local reduction with remote movement, and compare the schedule with psum_scatter.", "collectives", "lab6_reduce_scatter", ["REDUCE", "SHARD"]],
  [7, "Composed All-Reduce", "Compose reduce-scatter and all-gather, restore canonical order, and validate the result against JAX psum.", "collectives", "lab7_all_reduce", ["ALL-REDUCE", "BYTES"]],
  [8, "Chunked Ring Pipeline", "Vary chunk size and buffer count, distinguish serialized chunking from real overlap, and expose the pipeline plan.", "collectives", "lab8_chunked_pipeline", ["CHUNKS", "OVERLAP"]],
  [9, "Bandwidth-Optimal All-Reduce", "Implement a shard-ring schedule, derive the optimal byte term, and locate the alpha-beta crossover with measurement.", "topology", "lab9_optimal_all_reduce", ["OPTIMAL", "ROOFLINE"]],
  [10, "2D Mesh Collectives", "Map a logical 2×2 mesh and compare x-then-y with y-then-x staging to make topology part of the algorithm.", "topology", "lab10_mesh_collectives", ["2D MESH", "TOPOLOGY"]],
  [11, "Multi-Host Smoke Test", "Inspect process and device topology, exercise process collectives, and produce a launch-ready hierarchy plan.", "topology", "lab11_multihost_smoke", ["MULTI-HOST", "LAUNCH"]]
];

const PHASE_LABELS = {
  foundations: "Foundations · 01—04",
  collectives: "Collectives · 05—08",
  topology: "Topology · 09—11"
};

const grid = document.querySelector("#lab-grid");
const results = document.querySelector("#results-count");
const search = document.querySelector("#lab-search");
const filters = [...document.querySelectorAll("[data-filter]")];
let phase = "all";

function labUrl(slug) {
  return `https://github.com/karlb-dev/labs/blob/main/collective_communication/labs/${slug}.md`;
}

function render() {
  const query = search.value.trim().toLowerCase();
  const visible = LABS.filter(([number, title, description, itemPhase, slug, badges]) => {
    const phaseMatch = phase === "all" || itemPhase === phase;
    const haystack = [number, title, description, slug, ...badges].join(" ").toLowerCase();
    return phaseMatch && haystack.includes(query);
  });

  results.textContent = `${String(visible.length).padStart(2, "0")} of ${LABS.length} labs`;
  grid.innerHTML = visible.length
    ? visible.map(([number, title, description, itemPhase, slug, badges]) => `
      <a class="lab-card ${itemPhase}" href="${labUrl(slug)}" target="_blank" rel="noreferrer">
        <div class="lab-top">
          <span class="lab-number">${String(number).padStart(2, "0")}</span>
          <small>${PHASE_LABELS[itemPhase]}</small>
          <i aria-hidden="true">↗</i>
        </div>
        <h3>${title}</h3>
        <p>${description}</p>
        <div class="evidence">${badges.map((badge) => `<span>${badge}</span>`).join("")}</div>
      </a>`).join("")
    : `<div class="empty"><span>∅</span><h3>No matching labs</h3><p>Try a broader search or another phase.</p></div>`;
}

filters.forEach((button) => {
  button.addEventListener("click", () => {
    phase = button.dataset.filter;
    filters.forEach((item) => item.classList.toggle("active", item === button));
    render();
  });
});

search.addEventListener("input", render);
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement !== search) {
    event.preventDefault();
    search.focus();
  }
  if (event.key === "Escape" && document.activeElement === search) {
    search.value = "";
    search.blur();
    render();
  }
});

render();
