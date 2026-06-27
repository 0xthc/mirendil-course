/* ============================================================
   Mirendil TSP Course — visualization toolkit (no dependencies)
   Auto-initializes every <div class="viz" data-viz="..."> on the page.
   ============================================================ */
(function () {
  "use strict";

  const COLORS = { tp: "#57b6f5", sp: "#f0c674", tsp: "#5fd38a", weight: "#c099f0", act: "#f0986b" };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const J = (el, attr, fallback) => { try { return JSON.parse(el.getAttribute(attr)); } catch (e) { return fallback; } };

  function frame(el, body) {
    const title = el.getAttribute("data-title");
    const sub = el.getAttribute("data-sub");
    const cap = el.getAttribute("data-cap");
    el.innerHTML =
      (title ? `<div class="viz-title">${esc(title)}</div>` : "") +
      (sub ? `<div class="viz-sub">${esc(sub)}</div>` : "") +
      body +
      (cap ? `<div class="viz-cap">${cap}</div>` : "");
  }

  /* ---------------- 1. Grouped bar chart (SVG) ---------------- */
  function barchart(el) {
    const groups = J(el, "data-groups", []);          // ["8K","16K",...]
    const series = J(el, "data-series", []);          // [{name,color,values:[...]}]
    const unit = el.getAttribute("data-unit") || "";
    const W = 640, H = 300, padL = 48, padR = 12, padT = 18, padB = 42;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    let ymax = parseFloat(el.getAttribute("data-ymax")) || 0;
    if (!ymax) series.forEach((s) => s.values.forEach((v) => { if (v > ymax) ymax = v; }));
    ymax = ymax * 1.12 || 1;

    const gW = plotW / groups.length;
    const nS = series.length;
    const bW = Math.min(46, (gW * 0.74) / nS);
    const y = (v) => padT + plotH - (v / ymax) * plotH;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    // y gridlines
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const val = (ymax / ticks) * i, yy = y(val);
      svg += `<line x1="${padL}" y1="${yy}" x2="${W - padR}" y2="${yy}" stroke="#2a3340" stroke-dasharray="${i ? "2 4" : "0"}"/>`;
      svg += `<text class="vlabel" x="${padL - 6}" y="${yy + 3}" text-anchor="end">${(+val.toFixed(val < 10 ? 1 : 0))}</text>`;
    }
    // bars
    groups.forEach((g, gi) => {
      const gx = padL + gi * gW;
      const groupInner = gW - bW * nS;
      series.forEach((s, si) => {
        const v = s.values[gi];
        const x = gx + groupInner / 2 + si * bW;
        const yy = y(v), h = padT + plotH - yy;
        svg += `<rect class="bar" x="${x + 2}" y="${yy}" width="${bW - 4}" height="${h}" rx="3" fill="${s.color}"><title>${esc(s.name)} · ${esc(g)}: ${v}${esc(unit)}</title></rect>`;
        svg += `<text class="vlabel" x="${x + bW / 2}" y="${yy - 4}" text-anchor="middle">${v}</text>`;
      });
      svg += `<text class="glabel" x="${gx + gW / 2}" y="${H - padB + 18}" text-anchor="middle">${esc(g)}</text>`;
    });
    svg += `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#2a3340"/></svg>`;

    const legend = `<div class="legend-row">${series.map((s) => `<span><span class="sw" style="background:${s.color}"></span>${esc(s.name)}</span>`).join("")}</div>`;
    frame(el, svg + legend);
  }

  /* ---------------- 2. Collective animation ---------------- */
  function collective(el) {
    const op = el.getAttribute("data-op") || "broadcast";
    const N = 4;
    const STAGES = {
      broadcast: [
        { chips: [["W", "d"], ["", "empty"], ["", "empty"], ["", "empty"]], label: "Start: only rank 0 holds the weight W. The others have empty receive buffers." },
        { chips: [["W", "d"], ["W", "d"], ["W", "d"], ["W", "d"]], label: "broadcast(src=0): every rank now has an identical copy of W.", flashAll: true },
      ],
      all_reduce: [
        { chips: [["[1,0,0]", "a"], ["[0,2,0]", "b"], ["[0,0,3]", "c"], ["[1,1,1]", "d"]], label: "Start: each rank computed a different PARTIAL of the same sum." },
        { chips: [["[2,3,4]", "sum"], ["[2,3,4]", "sum"], ["[2,3,4]", "sum"], ["[2,3,4]", "sum"]], label: "all_reduce(SUM): partials added element-wise → the total lands on every rank.", flashAll: true },
      ],
      all_gather: [
        { chips: [["A", "a"], ["B", "b"], ["C", "c"], ["D", "d"]], label: "Start: each rank holds a distinct CHUNK of a larger tensor." },
        { chips: [["A B C D", "full"], ["A B C D", "full"], ["A B C D", "full"], ["A B C D", "full"]], label: "all_gather: chunks concatenated (not summed) → everyone has the whole tensor.", flashAll: true },
      ],
    };
    const stages = STAGES[op];
    let cur = 0;

    el.innerHTML =
      (el.getAttribute("data-title") ? `<div class="viz-title">${esc(el.getAttribute("data-title"))}</div>` : "") +
      (el.getAttribute("data-sub") ? `<div class="viz-sub">${esc(el.getAttribute("data-sub"))}</div>` : "") +
      `<div class="gpu-row" data-gpus></div>` +
      `<div class="controls"><button class="btn primary" data-toggle></button><button class="btn" data-reset>Reset</button></div>` +
      `<div class="step-label" data-step></div>`;

    const gpusEl = el.querySelector("[data-gpus]");
    const stepEl = el.querySelector("[data-step]");
    const toggleEl = el.querySelector("[data-toggle]");

    function render() {
      const st = stages[cur];
      gpusEl.innerHTML = st.chips.map((c, i) =>
        `<div class="gpu ${st.flashAll ? "flash" : ""}"><span class="gpu-name">rank ${i}</span><span class="chip ${c[1]}">${c[0] === "" ? "∅" : esc(c[0])}</span></div>`
      ).join("");
      stepEl.textContent = `Step ${cur + 1}/${stages.length} — ${st.label}`;
      toggleEl.textContent = cur < stages.length - 1 ? `▶ Run ${op}` : "✓ Done — replay";
    }
    toggleEl.addEventListener("click", () => { cur = cur < stages.length - 1 ? cur + 1 : 0; render(); });
    el.querySelector("[data-reset]").addEventListener("click", () => { cur = 0; render(); });
    render();
  }

  /* ---------------- 3. Memory calculator ---------------- */
  function memcalc(el) {
    el.innerHTML =
      `<div class="viz-title">${esc(el.getAttribute("data-title") || "Weights vs. activations — feel the difference")}</div>` +
      `<div class="viz-sub">${esc(el.getAttribute("data-sub") || "Drag the sliders. Watch which bar moves when you change the input size vs. the model size.")}</div>` +
      slider("H", "H — hidden size", 512, 8192, 4096, 512) +
      slider("S", "S — sequence length", 1024, 131072, 8192, 1024) +
      slider("L", "L — number of layers", 1, 80, 32, 1) +
      `<div data-bars style="margin-top:16px"></div>` +
      `<div class="viz-cap" data-read></div>`;

    const bars = el.querySelector("[data-bars]");
    const read = el.querySelector("[data-read]");
    function slider(id, label, min, max, val, step) {
      return `<div class="slider-row"><label>${label}</label><input type="range" data-s="${id}" min="${min}" max="${max}" step="${step}" value="${val}"><span class="val" data-v="${id}">${val}</span></div>`;
    }
    function gb(bytes) { return bytes / 1e9; }
    function update() {
      const H = +el.querySelector('[data-s=H]').value;
      const S = +el.querySelector('[data-s=S]').value;
      const L = +el.querySelector('[data-s=L]').value;
      el.querySelector('[data-v=H]').textContent = H;
      el.querySelector('[data-v=S]').textContent = S >= 1024 ? (S / 1024) + "K" : S;
      el.querySelector('[data-v=L]').textContent = L;
      // bf16 = 2 bytes. weights ≈ 12·H²·L. activations (illustrative) ≈ ~10 S-scaled tensors of width H per layer.
      const wGB = gb(12 * H * H * L * 2);
      const aGB = gb(10 * H * S * L * 2);
      const mx = Math.max(wGB, aGB, 1);
      const row = (name, v, color) => {
        const pct = (v / mx) * 100;
        return `<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px"><span style="color:${color};font-weight:600">${name}</span><span style="font-family:var(--mono);color:var(--text)">${v.toFixed(2)} GB</span></div>`
          + `<div style="background:var(--bg);border-radius:6px;height:22px;overflow:hidden"><div style="height:100%;width:${pct}%;background:${color};border-radius:6px;transition:width .3s"></div></div></div>`;
      };
      bars.innerHTML = row("Model weights (model state)", wGB, COLORS.weight) + row("Activations (this forward pass)", aGB, COLORS.act);
      read.innerHTML = `Try it: drag <strong style="color:var(--accent)">S</strong> — only the activations bar moves (weights are flat, they don't know your input length). Now drag <strong style="color:var(--accent)">H</strong> or <strong style="color:var(--accent)">L</strong> — <em>both</em> move. <span style="color:var(--text-dim)">(bf16; activation constant is illustrative.)</span>`;
    }
    el.querySelectorAll("input[type=range]").forEach((i) => i.addEventListener("input", update));
    update();
  }

  /* ---------------- 4. Attention heatmap ---------------- */
  function attention(el) {
    const n = parseInt(el.getAttribute("data-n")) || 8;
    let causal = true, sel = n - 1;
    el.innerHTML =
      `<div class="viz-title">${esc(el.getAttribute("data-title") || "Causal attention — who looks at whom")}</div>` +
      `<div class="viz-sub">Rows = query tokens (the one doing the looking). Columns = key tokens (being looked at). Click a row to spotlight one token.</div>` +
      `<div class="controls"><div class="seg"><button data-c="1" class="on">Causal (masked)</button><button data-c="0">Full (no mask)</button></div></div>` +
      `<div data-grid></div>` +
      `<div class="attn-legend" data-leg></div>`;
    const gridEl = el.querySelector("[data-grid]");
    const legEl = el.querySelector("[data-leg]");

    function render() {
      let html = `<div class="attn-grid" style="grid-template-columns:22px repeat(${n}, 26px)">`;
      html += `<div></div>`;
      for (let j = 0; j < n; j++) html += `<div class="attn-axis-x" style="text-align:center">${j}</div>`;
      for (let i = 0; i < n; i++) {
        html += `<div class="attn-axis-y" style="display:flex;align-items:center;justify-content:flex-end;padding-right:4px">${i}</div>`;
        const allowed = causal ? i + 1 : n;
        for (let j = 0; j < n; j++) {
          const ok = causal ? j <= i : true;
          if (!ok) { html += `<div class="attn-cell masked" title="token ${i} cannot see future token ${j}">×</div>`; continue; }
          const w = 1 / allowed;                       // uniform illustrative weight
          const intensity = 0.25 + 0.75 * (w / (1 / 1)); // brighter when fewer keys
          const isSel = i === sel;
          const bg = `rgba(87,182,245,${(0.18 + 0.8 * w).toFixed(3)})`;
          html += `<div class="attn-cell ${isSel ? "q" : ""}" data-i="${i}" style="background:${bg};color:#cfe8ff" title="query ${i} → key ${j}, weight ≈ ${w.toFixed(2)}">${(w >= 0.5 ? "•" : "")}</div>`;
        }
      }
      html += `</div>`;
      gridEl.innerHTML = html;
      gridEl.querySelectorAll(".attn-cell[data-i]").forEach((c) => c.addEventListener("click", () => { sel = +c.getAttribute("data-i"); render(); }));
      const cnt = causal ? sel + 1 : n;
      legEl.innerHTML = `Spotlight: <strong style="color:var(--accent)">token ${sel}</strong> attends to <strong>${cnt}</strong> ${cnt === 1 ? "token" : "tokens"} ${causal ? `(itself + ${sel} earlier)` : "(all of them)"}. ` +
        (causal ? `Notice the dark triangle: future tokens are <strong>masked</strong>. Later tokens attend to more keys → more work — that's the load-imbalance problem in module 6.` : `Without the mask, every token sees all ${n} — but that would let the model "cheat" by peeking at the future.`);
    }
    el.querySelectorAll("[data-c]").forEach((b) => b.addEventListener("click", () => {
      causal = b.getAttribute("data-c") === "1";
      el.querySelectorAll("[data-c]").forEach((x) => x.classList.toggle("on", x === b));
      render();
    }));
    render();
  }

  /* ---------------- 5. Sharding grid ---------------- */
  function shard(el) {
    let mode = el.getAttribute("data-mode") || "tp";
    const D = 4;
    el.innerHTML =
      `<div class="viz-title">${esc(el.getAttribute("data-title") || "What lives on each GPU")}</div>` +
      `<div class="viz-sub">Same 4 GPUs, three strategies. Watch what gets split (small = good) and what stays full (big = memory cost).</div>` +
      `<div class="controls"><div class="seg">
        <button data-m="tp" class="on">TP (split weights)</button>
        <button data-m="sp">SP (split tokens)</button>
        <button data-m="tsp">TSP (split both)</button></div></div>` +
      `<div class="gpu-row" data-gpus></div>` +
      `<div class="viz-cap" data-cap></div>`;
    const gpusEl = el.querySelector("[data-gpus]");
    const capEl = el.querySelector("[data-cap]");

    const MODES = {
      tp: {
        gpu: (i) => `<span class="gpu-name">GPU ${i}</span><span class="chip a">tokens: ALL</span><span class="chip d">heads ${i} only</span>`,
        cap: `<strong style="color:var(--c-tp)">TP</strong>: weights are <strong>split by head</strong> (each GPU stores ¼) ✅, but every GPU keeps the <strong>full sequence</strong> of tokens ❌. Shrinks model state, not activations.`,
      },
      sp: {
        gpu: (i) => `<span class="gpu-name">GPU ${i}</span><span class="chip c">tokens ${i} only</span><span class="chip full">weights: ALL</span>`,
        cap: `<strong style="color:var(--c-sp)">SP</strong>: tokens are <strong>split</strong> (each GPU holds ¼ of the sequence) ✅, but every GPU keeps a <strong>full copy of all weights</strong> ❌. Shrinks activations, not model state.`,
      },
      tsp: {
        gpu: (i) => `<span class="gpu-name">GPU ${i}</span><span class="chip c">tokens ${i}</span><span class="chip d">heads ${i}</span>`,
        cap: `<strong style="color:var(--c-tsp)">TSP</strong>: <strong>both</strong> split on the same GPUs (the diagonal) ✅✅. Each GPU holds ¼ of the tokens <em>and</em> ¼ of the weights. The catch: weight shards must be passed around during the forward pass (module 7).`,
      },
    };
    function render() {
      gpusEl.innerHTML = Array.from({ length: D }, (_, i) => `<div class="gpu">${MODES[mode].gpu(i)}</div>`).join("");
      capEl.innerHTML = MODES[mode].cap;
    }
    el.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => {
      mode = b.getAttribute("data-m");
      el.querySelectorAll("[data-m]").forEach((x) => x.classList.toggle("on", x === b));
      render();
    }));
    render();
  }

  /* ---------------- 6. Shape-flow stepper ---------------- */
  function flow(el) {
    const steps = J(el, "data-steps", []); // [{op, shape, note}]
    let cur = 0;
    el.innerHTML =
      `<div class="viz-title">${esc(el.getAttribute("data-title") || "Shape flow")}</div>` +
      `<div class="viz-sub">${esc(el.getAttribute("data-sub") || "Step through the tensor as it moves through the computation.")}</div>` +
      `<div data-stage style="min-height:120px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;background:var(--bg);border-radius:10px;padding:18px"></div>` +
      `<div class="controls" style="justify-content:center"><button class="btn" data-prev>← Back</button><span data-prog style="font-family:var(--mono);color:var(--text-dim);align-self:center"></span><button class="btn primary" data-next>Next →</button></div>`;
    const stage = el.querySelector("[data-stage]");
    const prog = el.querySelector("[data-prog]");
    function render() {
      const s = steps[cur];
      stage.innerHTML =
        `<div style="font-family:var(--mono);color:var(--accent);font-size:14px">${esc(s.op || "")}</div>` +
        `<div style="font-family:var(--mono);font-size:22px;color:var(--text-bright);background:var(--accent-soft);padding:8px 16px;border-radius:8px">${esc(s.shape)}</div>` +
        (s.note ? `<div style="color:var(--text-dim);font-size:13.5px;text-align:center;max-width:480px">${s.note}</div>` : "");
      prog.textContent = `${cur + 1} / ${steps.length}`;
      el.querySelector("[data-prev]").disabled = cur === 0;
      el.querySelector("[data-next]").disabled = cur === steps.length - 1;
    }
    el.querySelector("[data-next]").addEventListener("click", () => { if (cur < steps.length - 1) { cur++; render(); } });
    el.querySelector("[data-prev]").addEventListener("click", () => { if (cur > 0) { cur--; render(); } });
    render();
  }

  const REGISTRY = { barchart, collective, memcalc, attention, shard, flow };

  function init() {
    document.querySelectorAll(".viz[data-viz]").forEach((el) => {
      if (el.getAttribute("data-rendered")) return;
      const fn = REGISTRY[el.getAttribute("data-viz")];
      if (fn) { try { fn(el); el.setAttribute("data-rendered", "1"); } catch (e) { el.innerHTML = `<div class="viz-sub">⚠ widget error: ${esc(e.message)}</div>`; } }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
