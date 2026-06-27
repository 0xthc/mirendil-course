/* ============================================================
   Mirendil TSP Course — shared navigation + shape calculator
   ============================================================ */

const PAGES = [
  { file: "index.html",                    num: "",   title: "Course home" },
  { file: "01-big-picture.html",           num: "01", title: "The big picture" },
  { file: "02-transformers.html",          num: "02", title: "How a transformer computes" },
  { file: "03-attention-refresher.html",   num: "03", title: "Attention & the transformer block" },
  { file: "04-multi-gpu-primitives.html",  num: "04", title: "Multi-GPU primitives" },
  { file: "05-tensor-parallelism.html",    num: "05", title: "Tensor parallelism (TP)" },
  { file: "06-sequence-parallelism.html",  num: "06", title: "Sequence parallelism (SP)" },
  { file: "07-tensor-sequence-parallelism.html", num: "07", title: "Tensor-sequence parallelism (TSP)" },
  { file: "08-tradeoffs.html",             num: "08", title: "Tradeoffs & when to use what" },
  { file: "09-profiling-validation.html",  num: "09", title: "Profiling & validation" },
  { file: "10-glossary.html",              num: "10", title: "Glossary & cheat sheet" },
];

function currentFile() {
  const path = window.location.pathname.split("/").pop();
  return path === "" ? "index.html" : path;
}

function buildSidebar() {
  const here = currentFile();
  const links = PAGES.map((p) => {
    const active = p.file === here ? " class=\"active\"" : "";
    const num = p.num ? `<span class="num">${p.num}</span>` : `<span class="num">★</span>`;
    return `<li><a href="${p.file}"${active}>${num}<span>${p.title}</span></a></li>`;
  }).join("");

  return `
    <aside class="sidebar">
      <div class="brand">
        <a href="index.html">
          <span class="title">TSP for Engineers</span>
          <span class="subtitle">Mirendil AI Lab — concept course</span>
        </a>
      </div>
      <nav><ul class="nav-list">${links}</ul></nav>
    </aside>`;
}

function buildPageNav() {
  const here = currentFile();
  const idx = PAGES.findIndex((p) => p.file === here);
  if (idx < 0) return "";
  const prev = PAGES[idx - 1];
  const next = PAGES[idx + 1];
  const prevHtml = prev
    ? `<a href="${prev.file}"><span class="dir">← Previous</span><span class="ttl">${prev.title}</span></a>`
    : "<span></span>";
  const nextHtml = next
    ? `<a href="${next.file}" class="next"><span class="dir">Next →</span><span class="ttl">${next.title}</span></a>`
    : "<span></span>";
  return `<div class="pagenav">${prevHtml}${nextHtml}</div>`;
}

/* ---- Shape calculator ---- */
function fmt(arr) { return "[" + arr.join(", ") + "]"; }

function renderCalc(el) {
  const get = (id) => parseInt(el.querySelector("#" + id).value, 10);
  const B = get("c-B"), S = get("c-S"), H = get("c-H"), heads = get("c-heads"), D = get("c-D");
  const out = el.querySelector(".out");
  const errEl = el.querySelector(".err");
  errEl.textContent = "";

  const problems = [];
  if (![B, S, H, heads, D].every((n) => Number.isFinite(n) && n > 0)) {
    errEl.textContent = "Enter positive integers for every field.";
    out.innerHTML = "";
    return;
  }
  if (H % heads !== 0) problems.push("H must be divisible by num_heads.");
  if (heads % D !== 0) problems.push("num_heads must be divisible by D (so heads can be sharded).");
  if (S % D !== 0) problems.push("S must be divisible by D (so the sequence can be sharded).");
  if (problems.length) { errEl.innerHTML = problems.join("<br>"); out.innerHTML = ""; return; }

  const headDim = H / heads;
  const Hd = H / D;          // hidden per rank (head-sharded)
  const Sd = S / D;          // sequence per rank
  const localHeads = heads / D;

  const rows = [
    ["—", "Input X (full / replicated)", fmt([B, S, H])],
    ["TP", "X (replicated across ranks)", fmt([B, S, H])],
    ["TP", "Local Q/K/V (head-sharded)", fmt([B, localHeads, S, headDim])],
    ["TP", "Local attn out before Wo", fmt([B, S, Hd])],
    ["TP", "After Wo + all_reduce", fmt([B, S, H])],
    ["SP", "X_p (sequence-sharded)", fmt([B, Sd, H])],
    ["SP", "Local Q/K/V", fmt([B, heads, Sd, headDim])],
    ["SP", "K/V after all_gather", fmt([B, heads, S, headDim])],
    ["SP", "Output Y_p (this rank's tokens)", fmt([B, Sd, H])],
    ["TSP", "X_p (sequence-sharded)", fmt([B, Sd, H])],
    ["TSP", "Q/K/V for weight-shard r", fmt([B, localHeads, Sd, headDim])],
    ["TSP", "K/V after all_gather (over seq)", fmt([B, localHeads, S, headDim])],
    ["TSP", "Y_p (accumulated over r)", fmt([B, Sd, H])],
  ];

  out.innerHTML = `
    <p style="color:var(--text-dim);font-size:14px;margin:0 0 10px">
      head_dim = H / heads = <strong>${headDim}</strong> &nbsp;·&nbsp;
      hidden/rank = H / D = <strong>${Hd}</strong> &nbsp;·&nbsp;
      seq/rank = S / D = <strong>${Sd}</strong> &nbsp;·&nbsp;
      heads/rank = <strong>${localHeads}</strong>
    </p>
    <table>
      <thead><tr><th>Mode</th><th>Tensor</th><th>Shape</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td><strong>${r[0]}</strong></td><td>${r[1]}</td><td><span class="shape">${r[2]}</span></td></tr>`).join("")}</tbody>
    </table>`;
}

function wireCalc() {
  const el = document.querySelector(".calc");
  if (!el) return;
  el.querySelectorAll("input").forEach((i) => i.addEventListener("input", () => renderCalc(el)));
  renderCalc(el);
}

document.addEventListener("DOMContentLoaded", () => {
  const layout = document.querySelector(".layout");
  if (layout) layout.insertAdjacentHTML("afterbegin", buildSidebar());
  const slot = document.querySelector("[data-pagenav]");
  if (slot) slot.outerHTML = buildPageNav();
  wireCalc();
});
