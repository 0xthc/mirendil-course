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

  /* ---------------- 7. Latency & cost calculator ---------------- */
  function costcalc(el) {
    // Illustrative per-1M-token prices (USD) and per-token decode latency (ms).
    // Editable defaults; clearly labeled as illustrative, not a price sheet.
    const MODELS = {
      "Small / fast": { inUsd: 0.80, outUsd: 4.0, msPerOut: 4 },
      "Mid": { inUsd: 3.0, outUsd: 15.0, msPerOut: 7 },
      "Large / best": { inUsd: 15.0, outUsd: 75.0, msPerOut: 12 },
    };
    el.innerHTML =
      `<div class="viz-title">${esc(el.getAttribute("data-title") || "Latency & cost calculator")}</div>` +
      `<div class="viz-sub">${esc(el.getAttribute("data-sub") || "Illustrative numbers — the point is the SHAPE of the tradeoffs, not exact prices. Output tokens dominate both cost and latency.")}</div>` +
      `<div class="controls">
        <div class="seg" data-models>${Object.keys(MODELS).map((k, i) => `<button data-model="${k}" class="${i === 1 ? "on" : ""}">${k}</button>`).join("")}</div>
      </div>` +
      sliderRow("in", "Input (prompt) tokens", 100, 200000, 4000, 100) +
      sliderRow("out", "Output tokens", 50, 8000, 600, 50) +
      sliderRow("rps", "Requests / day (thousands)", 1, 1000, 50, 1) +
      `<div class="controls"><label style="display:flex;gap:8px;align-items:center;font-size:14px;color:var(--text-dim);cursor:pointer"><input type="checkbox" data-cache style="accent-color:var(--accent)"> Prompt caching on (90% of input cached, 0.1× price)</label></div>` +
      `<div data-bars style="margin-top:14px"></div>` +
      `<div class="viz-cap" data-read></div>`;

    let model = "Mid";

    function sliderRow(id, label, min, max, val, step) {
      return `<div class="slider-row"><label>${label}</label><input type="range" data-s="${id}" min="${min}" max="${max}" step="${step}" value="${val}"><span class="val" data-v="${id}">${val}</span></div>`;
    }
    function bar(label, valueText, frac, color) {
      return `<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px"><span style="color:${color};font-weight:600">${label}</span><span style="font-family:var(--mono);color:var(--text)">${valueText}</span></div>`
        + `<div style="background:var(--bg);border-radius:6px;height:20px;overflow:hidden"><div style="height:100%;width:${Math.max(1, Math.min(100, frac * 100)).toFixed(1)}%;background:${color};border-radius:6px;transition:width .3s"></div></div></div>`;
    }
    function update() {
      const m = MODELS[model];
      const inTok = +el.querySelector('[data-s=in]').value;
      const outTok = +el.querySelector('[data-s=out]').value;
      const rpsK = +el.querySelector('[data-s=rps]').value;
      const cache = el.querySelector('[data-cache]').checked;
      el.querySelector('[data-v=in]').textContent = inTok.toLocaleString();
      el.querySelector('[data-v=out]').textContent = outTok.toLocaleString();
      el.querySelector('[data-v=rps]').textContent = rpsK + "K";

      const cachedFrac = cache ? 0.9 : 0;
      const inEffective = inTok * (1 - cachedFrac) + inTok * cachedFrac * 0.1;
      const inCost = (inEffective / 1e6) * m.inUsd;
      const outCost = (outTok / 1e6) * m.outUsd;
      const perReq = inCost + outCost;
      const perDay = perReq * rpsK * 1000;
      const totalCost = inCost + outCost;
      const latency = outTok * m.msPerOut; // decode dominates; prefill ~ smaller, omitted for clarity

      const bars = el.querySelector("[data-bars]");
      const maxc = Math.max(inCost, outCost, 1e-9);
      bars.innerHTML =
        bar("Input token cost / request", "$" + inCost.toFixed(5), inCost / maxc, "#c099f0") +
        bar("Output token cost / request", "$" + outCost.toFixed(5), outCost / maxc, "#f0986b") +
        bar("Est. generation latency", (latency / 1000).toFixed(2) + " s", latency / (8000 * 12), "#57b6f5");
      el.querySelector("[data-read]").innerHTML =
        `<strong style="color:var(--accent)">$${perReq.toFixed(5)}</strong> per request · ` +
        `<strong style="color:var(--accent)">$${perDay.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong> per day at ${rpsK}K req · ` +
        `~<strong>${(latency / 1000).toFixed(1)}s</strong> to generate. ` +
        `Notice: output tokens are priced ~5× input and drive latency — <em>shortening outputs</em> and <em>prompt caching</em> are your biggest levers.`;
    }
    el.querySelectorAll("[data-model]").forEach((b) => b.addEventListener("click", () => {
      model = b.getAttribute("data-model");
      el.querySelectorAll("[data-model]").forEach((x) => x.classList.toggle("on", x === b));
      update();
    }));
    el.querySelectorAll("input").forEach((i) => i.addEventListener("input", update));
    update();
  }

  /* ---------------- 8. Interactive architecture diagram ---------------- */
  const ARCH_PRESETS = {
    "chat-platform": {
      layers: {
        client:    { label: "Client",        color: "#57b6f5" },
        edge:      { label: "Edge",          color: "#f0c674" },
        gateway:   { label: "API gateway",   color: "#c099f0" },
        services:  { label: "Services",      color: "#5fd38a" },
        data:      { label: "Data stores",   color: "#f0986b" },
        inference: { label: "Inference",     color: "#f08b8b" },
      },
      nodes: [
        { id: "client",  title: "Browser / App", sub: "threads · SSE",        layer: "client",    col: 0, row: 0, desc: "The web or mobile client. Renders the threads list and streams the assistant reply token-by-token over SSE; handles optimistic UI and reconnection." },
        { id: "edge",    title: "Edge",          sub: "CDN · TLS · WAF",       layer: "edge",      col: 1, row: 0, desc: "CDN, TLS termination, WAF / DDoS protection and geo-routing. Forwards the request to the nearest API gateway." },
        { id: "gateway", title: "API Gateway",   sub: "auth · rate-limit",     layer: "gateway",   col: 2, row: 0, desc: "Verifies the session (Redis), enforces idempotency keys, applies token-bucket rate limits (by request AND by tokens), validates the body and attaches a trace id." },
        { id: "conv",    title: "Conversation",  sub: "threads & messages",    layer: "services",  col: 3, row: 0, desc: "Owns threads and messages. Enforces ACLs, loads history (hot path from Redis, source of truth in Postgres) and appends the new user message." },
        { id: "orch",    title: "Orchestration", sub: "prompt · routing",      layer: "services",  col: 4, row: 0, desc: "Assembles the final prompt (system + history + new turn), sets the prompt-cache breakpoint on the stable prefix, picks the model tier and enqueues the inference job." },
        { id: "igw",     title: "Inference GW",  sub: "admission · batching",  layer: "inference", col: 5, row: 0, desc: "Inference gateway — admission control, continuous (in-flight) batching and backpressure in front of the GPU fleet." },
        { id: "sched",   title: "Scheduler",     sub: "route · KV cache",      layer: "inference", col: 6, row: 0, desc: "Routes the request to a model server with capacity and checks its paged KV cache / prefix cache for a prompt-cache hit." },
        { id: "model",   title: "Model server",  sub: "embed→attn+MLP→sample", layer: "inference", col: 7, row: 0, desc: "A GPU model server. Runs the forward pass: embed → L transformer layers (attention + MLP) → sample. Maintains the KV cache and streams tokens out." },
        { id: "olap",    title: "Warehouse",     sub: "events · evals",        layer: "data",      col: 0, row: 2, desc: "Analytics warehouse (OLAP) for product events, evals and dashboards — fed asynchronously, off the request path." },
        { id: "redis",   title: "Redis",         sub: "cache · queue · pub/sub", layer: "data",    col: 2, row: 2, desc: "Sessions, rate-limit counters, hot-thread cache, idempotency keys, the inference job queue, and pub/sub channels that fan streamed tokens back to the client." },
        { id: "pg",      title: "PostgreSQL",    sub: "system of record",      layer: "data",      col: 3, row: 2, desc: "The durable, transactional source of truth for users, threads, messages and usage. Indexed by thread_id + created_at." },
        { id: "safety",  title: "Safety / Bill", sub: "moderation · metering", layer: "services",  col: 4, row: 2, desc: "Runs input and output moderation classifiers, meters token usage for billing and emits analytics events. Can cut a stream on a policy hit." },
        { id: "vec",     title: "Vector DB",     sub: "retrieval · memory",    layer: "data",      col: 5, row: 2, desc: "Vector database for retrieval, long-term memory and semantic search over past threads or documents (optional — powers RAG)." },
        { id: "s3",      title: "Object store",  sub: "blobs · files",         layer: "data",      col: 6, row: 2, desc: "Object storage for attachments, large message blobs and exported files. The database stores a pointer, not the bytes." },
      ],
      edges: [
        { from: "client", to: "edge" }, { from: "edge", to: "gateway" }, { from: "gateway", to: "conv" },
        { from: "conv", to: "orch" }, { from: "orch", to: "igw" }, { from: "igw", to: "sched" }, { from: "sched", to: "model" },
        { from: "gateway", to: "redis", kind: "data" }, { from: "conv", to: "pg", kind: "data" },
        { from: "conv", to: "redis", kind: "data" }, { from: "orch", to: "redis", kind: "data" },
        { from: "orch", to: "safety", kind: "data" }, { from: "orch", to: "vec", kind: "data" },
        { from: "conv", to: "s3", kind: "data" }, { from: "model", to: "redis", kind: "data" },
      ],
      path: [
        { node: "client",  label: "POST /v1/threads/{id}/messages with stream:true" },
        { node: "edge",    label: "TLS, WAF, geo-route to the nearest region" },
        { node: "gateway", label: "verify session, idempotency, token-bucket rate limit" },
        { node: "conv",    label: "load history (Redis → Postgres), append the message" },
        { node: "orch",    label: "assemble prompt + cache_control, enqueue the job" },
        { node: "igw",     label: "admission control + continuous batching" },
        { node: "sched",   label: "route to a GPU, check the KV / prefix cache" },
        { node: "model",   label: "prefill (attention + MLP × L) → first token → decode loop" },
        { node: "client",  label: "tokens stream back via Redis pub/sub → SSE" },
      ],
    },

    "aws": {
      layers: {
        client:    { label: "Client",                color: "#57b6f5" },
        edge:      { label: "Edge (Route 53 · CloudFront · WAF)", color: "#f0c674" },
        gateway:   { label: "Load balancing",        color: "#c099f0" },
        compute:   { label: "Compute (ECS / EKS)",   color: "#5fd38a" },
        messaging: { label: "Messaging (SQS)",       color: "#e08bd0" },
        data:      { label: "Data stores",           color: "#f0986b" },
        inference: { label: "Inference (GPU)",       color: "#f08b8b" },
        ops:       { label: "Observability",         color: "#8fa0b3" },
      },
      nodes: [
        { id: "client",  title: "Client",            sub: "browser / app",          layer: "client",    col: 0, row: 0, desc: "Web or mobile client. Opens a long-lived SSE stream for the reply." },
        { id: "cf",      title: "CloudFront",        sub: "CDN · WAF · Shield",      layer: "edge",      col: 1, row: 0, desc: "CloudFront edge with AWS WAF + Shield: TLS, static caching, and dropping attacks before they reach the VPC." },
        { id: "alb",     title: "ALB",               sub: "L7 load balancer",        layer: "gateway",   col: 2, row: 0, desc: "Application Load Balancer — terminates the SSE-friendly long-lived HTTP connection and routes to a healthy app task. Preferred over API Gateway for streaming, which has short timeouts." },
        { id: "app",     title: "ECS / EKS app",     sub: "auth · limit · enqueue",  layer: "compute",   col: 3, row: 0, desc: "The stateless app tier on ECS Fargate or EKS: authenticates (ElastiCache), rate-limits, loads the thread (Aurora), assembles the prompt, and enqueues the inference job (SQS)." },
        { id: "sqs",     title: "Amazon SQS",        sub: "job queue",               layer: "messaging", col: 4, row: 0, desc: "Durable job queue decoupling the fast API from slow GPU work. A visibility timeout hides an in-flight job; a redrive policy sends repeatedly-failing (poison) jobs to the DLQ." },
        { id: "infgw",   title: "Inference GW",      sub: "admission · batching",    layer: "inference", col: 5, row: 0, desc: "Consumes jobs from SQS, applies admission control and continuous batching, and routes to a model server (internal NLB / service mesh)." },
        { id: "model",   title: "GPU model server",  sub: "vLLM · KV cache in HBM",  layer: "inference", col: 6, row: 0, desc: "EC2 GPU (p5/p4d/g5) on EKS running vLLM/TGI. Runs prefill + decode; the KV cache lives in GPU HBM, paged, with CPU/NVMe offload under pressure." },
        { id: "route53", title: "Route 53",          sub: "DNS · geo",               layer: "edge",      col: 0, row: 2, desc: "DNS and latency/geo routing to the nearest CloudFront edge and a healthy region." },
        { id: "cw",      title: "CloudWatch / X-Ray", sub: "metrics · traces",       layer: "ops",       col: 1, row: 2, desc: "Metrics (TTFT, tokens/sec, queue depth, p99), distributed traces (X-Ray / OpenTelemetry) and logs. Alarms drive autoscaling." },
        { id: "redis",   title: "ElastiCache Redis", sub: "limit · cache · pub/sub", layer: "data",      col: 2, row: 2, desc: "Sessions, RPM/TPM token-bucket counters, hot-thread cache, idempotency keys, the prompt-cache routing map, and pub/sub channels that fan streamed tokens back to the client." },
        { id: "aurora",  title: "Aurora PostgreSQL", sub: "system of record",        layer: "data",      col: 3, row: 2, desc: "Durable, Multi-AZ source of truth for users, threads, messages and usage; read replicas for the history hot path." },
        { id: "dlq",     title: "SQS DLQ",           sub: "dead-letter queue",       layer: "messaging", col: 4, row: 2, desc: "Dead-letter queue. After maxReceiveCount failed deliveries a job lands here instead of looping forever and blocking the queue. Alarm on depth; fix and redrive." },
        { id: "oss",     title: "OpenSearch",        sub: "vector / pgvector",       layer: "data",      col: 5, row: 2, desc: "Vector store for retrieval, memory and semantic search (Amazon OpenSearch k-NN, or pgvector inside Aurora)." },
        { id: "s3",      title: "Amazon S3",         sub: "blobs · exports",         layer: "data",      col: 6, row: 2, desc: "Attachments, large message blobs and exports. The database stores an S3 pointer, not the bytes." },
      ],
      edges: [
        { from: "client", to: "cf" }, { from: "cf", to: "alb" }, { from: "alb", to: "app" },
        { from: "app", to: "sqs" }, { from: "sqs", to: "infgw" }, { from: "infgw", to: "model" },
        { from: "route53", to: "cf", kind: "data" }, { from: "app", to: "redis", kind: "data" },
        { from: "app", to: "aurora", kind: "data" }, { from: "app", to: "s3", kind: "data" },
        { from: "app", to: "oss", kind: "data" }, { from: "sqs", to: "dlq", kind: "data" },
        { from: "model", to: "redis", kind: "data" }, { from: "app", to: "cw", kind: "data" },
      ],
      path: [
        { node: "client",  label: "POST /v1/threads/{id}/messages with stream:true" },
        { node: "cf",      label: "CloudFront + WAF/Shield: TLS, cache, drop attacks" },
        { node: "alb",     label: "ALB routes to a healthy app task, holds the SSE connection" },
        { node: "app",     label: "auth + rate-limit (ElastiCache), load thread (Aurora), enqueue (SQS)" },
        { node: "sqs",     label: "SQS queues the inference job; DLQ catches poison messages" },
        { node: "infgw",   label: "inference gateway consumes it: admission + continuous batching" },
        { node: "model",   label: "GPU server (vLLM): prefill + decode, KV cache in HBM" },
        { node: "app",     label: "tokens stream back via ElastiCache pub/sub → ALB → SSE" },
      ],
    },
  };

  function architecture(el) {
    const spec = ARCH_PRESETS[el.getAttribute("data-arch") || "chat-platform"];
    if (!spec) { el.innerHTML = `<div class="viz-sub">unknown architecture preset</div>`; return; }
    const W = 150, H = 56, GX = 40, GY = 44, PAD = 18;
    const cols = Math.max(...spec.nodes.map((n) => n.col)) + 1;
    const rows = Math.max(...spec.nodes.map((n) => n.row)) + 1;
    const width = PAD * 2 + cols * W + (cols - 1) * GX;
    const height = PAD * 2 + rows * H + (rows - 1) * GY;
    const X = (c) => PAD + c * (W + GX), Y = (r) => PAD + r * (H + GY);
    const byId = {}; spec.nodes.forEach((n) => (byId[n.id] = n));
    const lc = (l) => (spec.layers[l] ? spec.layers[l].color : "#57b6f5");

    function anchor(f, t) {
      const fx = X(f.col), fy = Y(f.row), tx = X(t.col), ty = Y(t.row);
      const fcx = fx + W / 2, fcy = fy + H / 2, tcx = tx + W / 2, tcy = ty + H / 2;
      const dx = tcx - fcx, dy = tcy - fcy;
      if (Math.abs(dx) >= Math.abs(dy)) return [[dx > 0 ? fx + W : fx, fcy], [dx > 0 ? tx : tx + W, tcy]];
      return [[fcx, dy > 0 ? fy + H : fy], [tcx, dy > 0 ? ty : ty + H]];
    }
    const edgesSvg = spec.edges.map((e) => {
      const f = byId[e.from], t = byId[e.to]; if (!f || !t) return "";
      const [a, b] = anchor(f, t);
      const dash = e.kind === "data" ? 'stroke-dasharray="4 4"' : "";
      return `<line class="arch-edge ${e.kind || ""}" data-edge="${e.from}__${e.to}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" ${dash} marker-end="url(#arch-arrow)"/>`;
    }).join("");
    const nodesSvg = spec.nodes.map((n) => {
      const x = X(n.col), y = Y(n.row), col = lc(n.layer);
      return `<g class="arch-node" data-node="${n.id}" tabindex="0">
        <rect x="${x}" y="${y}" width="${W}" height="${H}" rx="9" fill="${col}22" stroke="${col}"/>
        <text class="arch-t" x="${x + W / 2}" y="${y + 23}" text-anchor="middle">${esc(n.title)}</text>
        <text class="arch-s" x="${x + W / 2}" y="${y + 40}" text-anchor="middle">${esc(n.sub || "")}</text>
      </g>`;
    }).join("");
    const legend = Object.keys(spec.layers).map((k) => `<span><span class="sw" style="background:${spec.layers[k].color}"></span>${esc(spec.layers[k].label)}</span>`).join("");

    el.innerHTML =
      (el.getAttribute("data-title") ? `<div class="viz-title">${esc(el.getAttribute("data-title"))}</div>` : "") +
      (el.getAttribute("data-sub") ? `<div class="viz-sub">${esc(el.getAttribute("data-sub"))}</div>` : "") +
      `<div class="controls"><button class="btn primary" data-play>▶ Animate a request</button><button class="btn" data-reset>Reset</button><span style="color:var(--text-dim);font-size:13px">Solid = request path · dashed = data access. Click any box.</span></div>` +
      `<div class="arch-scroll"><svg class="arch-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="System architecture">
         <defs><marker id="arch-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#5a6678"/></marker></defs>
         ${edgesSvg}${nodesSvg}</svg></div>` +
      `<div class="legend-row">${legend}</div>` +
      `<div class="arch-detail" data-detail></div>` +
      `<div class="step-label" data-step></div>`;

    const detail = el.querySelector("[data-detail]");
    const stepEl = el.querySelector("[data-step]");
    const playBtn = el.querySelector("[data-play]");
    const nodeEl = (id) => el.querySelector(`[data-node="${id}"]`);
    const defaultDetail = `<span style="color:var(--text-dim)">Hover or click a component to see what it does, or press <strong>Animate a request</strong> to watch one message flow through the whole system.</span>`;

    function showDetail(n) {
      detail.innerHTML = `<span class="badge" style="background:${lc(n.layer)}22;color:${lc(n.layer)}">${esc(spec.layers[n.layer].label)}</span> <strong style="color:var(--text-bright)">${esc(n.title)}</strong><br>${esc(n.desc || "")}`;
    }
    function select(id) {
      spec.nodes.forEach((n) => nodeEl(n.id).classList.toggle("sel", n.id === id));
      showDetail(byId[id]);
    }
    spec.nodes.forEach((n) => {
      const g = nodeEl(n.id);
      g.addEventListener("click", () => select(n.id));
      g.addEventListener("keypress", (e) => { if (e.key === "Enter") select(n.id); });
      g.addEventListener("mouseenter", () => showDetail(n));
    });
    detail.innerHTML = defaultDetail;

    let timer = null, step = -1;
    function clearAnim() {
      if (timer) { clearInterval(timer); timer = null; }
      el.querySelectorAll(".arch-node").forEach((g) => g.classList.remove("active", "dim", "sel"));
      el.querySelectorAll(".arch-edge").forEach((e) => e.classList.remove("active"));
    }
    function renderStep() {
      el.querySelectorAll(".arch-node").forEach((g) => g.classList.add("dim"));
      el.querySelectorAll(".arch-edge").forEach((e) => e.classList.remove("active"));
      for (let i = 0; i <= step; i++) {
        const id = spec.path[i].node, g = nodeEl(id);
        g.classList.remove("dim");
        g.classList.toggle("active", i === step);
        if (i > 0) {
          const prev = spec.path[i - 1].node;
          const e = el.querySelector(`[data-edge="${prev}__${id}"]`) || el.querySelector(`[data-edge="${id}__${prev}"]`);
          if (e) e.classList.add("active");
        }
      }
      const s = spec.path[step];
      stepEl.textContent = `Step ${step + 1}/${spec.path.length} — ${byId[s.node].title}: ${s.label}`;
      showDetail(byId[s.node]);
      nodeEl(s.node).classList.add("sel");
    }
    playBtn.addEventListener("click", () => {
      clearAnim(); step = -1; playBtn.textContent = "⏸ Playing…";
      timer = setInterval(() => {
        step++;
        if (step >= spec.path.length) {
          clearInterval(timer); timer = null; playBtn.textContent = "▶ Replay";
          el.querySelectorAll(".arch-node").forEach((g) => g.classList.remove("dim"));
          return;
        }
        renderStep();
      }, 850);
    });
    el.querySelector("[data-reset]").addEventListener("click", () => {
      clearAnim(); step = -1; stepEl.textContent = ""; detail.innerHTML = defaultDetail;
      playBtn.textContent = "▶ Animate a request";
    });
  }

  const REGISTRY = { barchart, collective, memcalc, attention, shard, flow, costcalc, architecture };

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
