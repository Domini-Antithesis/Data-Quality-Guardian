/* Data Quality Agent - frontend.
   Plain JavaScript, no framework. Talks to the FastAPI backend under /api. */

(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  const DESTRUCTIVE = new Set(["drop_rows", "drop_duplicates", "remove_outlier_rows", "drop_column"]);
  let STRATEGY_HELP = {};
  let current = null;        // the full run payload
  let selection = {};        // issue id -> {approved, strategy, fill_value}

  // ------------------------------------------------------------ helpers
  async function api(path, options = {}) {
    const res = await fetch(path, options);
    const type = res.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) {
      const d = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
      throw new Error(d);
    }
    return body;
  }
  const getJSON = (p) => api(p);
  const postJSON = (p, d) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(d || {}) });
  const putJSON = (p, d) => api(p, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(d || {}) });

  function toast(msg, ms = 3600) {
    const el = $("#toast");
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(() => (el.hidden = true), ms);
  }
  function setResult(id, ok, msg) {
    const el = $(id);
    el.textContent = msg;
    el.className = "result " + (ok ? "ok" : "err");
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function fmtDate(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString();
  }
  const pretty = (s) => String(s).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  // ------------------------------------------------------------ tabs
  function showTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + name));
    if (name === "history") loadHistory();
    if (name === "setup") loadSettings();
    window.scrollTo({ top: 0 });
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  document.addEventListener("click", (e) => {
    const g = e.target.closest("[data-goto]");
    if (g) { e.preventDefault(); showTab(g.dataset.goto); }
  });

  // ------------------------------------------------------------ health
  async function refreshHealth() {
    try {
      const h = await getJSON("/api/health");
      const label = h.provider === "mock" ? "Demo mode (built-in rules)" : `AI: ${h.provider} · ${h.model}`;
      const pills = [`<span class="pill ${h.llm_ready ? "ok" : "warn"}">${esc(h.llm_ready ? label : "AI provider: not configured")}</span>`];
      if (h.tracing) pills.push('<span class="pill ok">LangSmith tracing on</span>');
      $("#status-pills").innerHTML = pills.join("");
      $("#setup-notice").hidden = h.llm_ready;
      return h;
    } catch {
      $("#status-pills").innerHTML = '<span class="pill warn">Backend unreachable</span>';
      return null;
    }
  }

  // ------------------------------------------------------------ settings
  const KEY_ROWS = { groq: "#key-groq", gemini: "#key-gemini", openai: "#key-openai", ollama: "#key-ollama" };
  $("#llm_provider").addEventListener("change", (e) => {
    Object.entries(KEY_ROWS).forEach(([p, sel]) => ($(sel).hidden = p !== e.target.value));
    const s = window.__settings || {};
    $("#llm_model").placeholder = (s.default_models && s.default_models[e.target.value]) || "default";
  });
  function secretPlaceholder(input, isSet) {
    if (!input.dataset.ph) input.dataset.ph = input.placeholder;
    input.value = "";
    input.placeholder = isSet ? "•••••••• (saved - leave blank to keep)" : input.dataset.ph;
  }
  async function loadSettings() {
    const s = await getJSON("/api/settings");
    window.__settings = s;
    $("#llm_provider").value = s.llm_provider;
    $("#llm_model").value = s.llm_model || "";
    $("#llm_model").placeholder = s.default_models[s.llm_provider] || "default";
    Object.entries(KEY_ROWS).forEach(([p, sel]) => ($(sel).hidden = p !== s.llm_provider));
    secretPlaceholder($("#groq_api_key"), s.groq_api_key_set);
    secretPlaceholder($("#google_api_key"), s.google_api_key_set);
    secretPlaceholder($("#openai_api_key"), s.openai_api_key_set);
    $("#ollama_base_url").value = s.ollama_base_url;
    $("#missing_warn_pct").value = s.missing_warn_pct;
    $("#missing_critical_pct").value = s.missing_critical_pct;
    $("#outlier_iqr_multiplier").value = s.outlier_iqr_multiplier;
    $("#langchain_tracing_v2").checked = !!s.langchain_tracing_v2;
    $("#langchain_project").value = s.langchain_project;
    secretPlaceholder($("#langchain_api_key"), s.langchain_api_key_set);
  }
  async function saveSettings(payload, resultId) {
    try {
      await putJSON("/api/settings", payload);
      setResult(resultId, true, "Saved.");
      await refreshHealth();
      await loadSettings();
    } catch (err) { setResult(resultId, false, "Could not save: " + err.message); }
  }
  $("#btn-save-llm").addEventListener("click", () => saveSettings({
    llm_provider: $("#llm_provider").value,
    llm_model: $("#llm_model").value.trim(),
    groq_api_key: $("#groq_api_key").value.trim(),
    google_api_key: $("#google_api_key").value.trim(),
    openai_api_key: $("#openai_api_key").value.trim(),
    ollama_base_url: $("#ollama_base_url").value.trim() || "http://localhost:11434",
  }, "#res-llm"));
  $("#btn-test-llm").addEventListener("click", async () => {
    const b = $("#btn-test-llm"); b.disabled = true;
    setResult("#res-llm", true, "Testing… (save first if you changed something)");
    try { const r = await postJSON("/api/settings/test/llm"); setResult("#res-llm", r.ok, r.message); }
    catch (err) { setResult("#res-llm", false, err.message); }
    b.disabled = false;
  });
  $("#btn-save-rules").addEventListener("click", () => saveSettings({
    missing_warn_pct: Number($("#missing_warn_pct").value),
    missing_critical_pct: Number($("#missing_critical_pct").value),
    outlier_iqr_multiplier: Number($("#outlier_iqr_multiplier").value),
  }, "#res-rules"));
  $("#btn-save-tracing").addEventListener("click", () => saveSettings({
    langchain_tracing_v2: $("#langchain_tracing_v2").checked,
    langchain_project: $("#langchain_project").value.trim() || "data-quality-agent",
    langchain_api_key: $("#langchain_api_key").value.trim(),
  }, "#res-tracing"));

  // ------------------------------------------------------------ upload
  $("#file-input").addEventListener("change", () => { $("#btn-upload").disabled = !$("#file-input").files[0]; });
  $("#btn-upload").addEventListener("click", async () => {
    const file = $("#file-input").files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    await startRun(() => api("/api/analyze", { method: "POST", body: fd }), `Checking ${file.name}`);
  });

  async function loadSamples() {
    try {
      const samples = await getJSON("/api/samples");
      const box = $("#sample-buttons");
      if (!samples.length) { box.innerHTML = '<span class="muted">No samples found.</span>'; return; }
      box.innerHTML = samples.map((s) =>
        `<button class="btn" data-sample="${esc(s.name)}">Check ${esc(s.name.replace(/_/g, " ").replace(".csv", ""))}</button>` +
        `<a class="btn small" href="/api/samples/${encodeURIComponent(s.name)}" download>Download</a>`).join(" ");
      box.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () =>
        startRun(() => postJSON(`/api/samples/${encodeURIComponent(b.dataset.sample)}/analyze`), `Checking ${b.dataset.sample}`)));
    } catch { /* samples are optional */ }
  }

  function setSteps(activeIndex, doneUpTo) {
    $$("#steps li").forEach((li, i) => {
      li.className = i < doneUpTo ? "done" : i === activeIndex ? "active" : "";
    });
  }

  async function startRun(kickoff, title) {
    const card = $("#progress-card");
    card.hidden = false;
    $("#progress-error").hidden = true;
    $("#progress-title").textContent = title + "…";
    $("#progress-text").textContent = "Reading the file…";
    $("#progress-bar").style.width = "6%";
    setSteps(0, 0);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    let runId;
    try {
      runId = (await kickoff()).run_id;
    } catch (err) {
      $("#progress-error").textContent = err.message;
      $("#progress-error").hidden = false;
      $("#progress-title").textContent = "Could not start";
      return;
    }
    await pollRun(runId);
  }

  const STATUS_STEP = { queued: 1, analyzing: 2, awaiting_approval: 5, applying: 5, completed: 6 };
  async function pollRun(runId) {
    for (;;) {
      let payload;
      try { payload = await getJSON(`/api/runs/${runId}`); }
      catch (err) { $("#progress-error").textContent = err.message; $("#progress-error").hidden = false; return; }
      const run = payload.run;
      if (run.status === "failed") {
        setSteps(-1, 0);
        $("#progress-title").textContent = "The check failed";
        $("#progress-error").textContent = run.error || "Unknown error";
        $("#progress-error").hidden = false;
        return;
      }
      if (run.status === "awaiting_approval" || run.status === "completed") {
        setSteps(5, 5);
        $("#progress-bar").style.width = "100%";
        $("#progress-title").textContent = "Checks finished";
        $("#progress-text").textContent = `${run.issues_found} issues found. Quality score ${run.score_before.toFixed(1)}/100.`;
        renderRun(payload);
        showTab("review");
        return;
      }
      const step = STATUS_STEP[run.status] || 1;
      setSteps(step, step);
      $("#progress-bar").style.width = `${10 + step * 14}%`;
      $("#progress-text").textContent = run.status === "analyzing"
        ? "Running the nine checks and asking the AI for recommendations…"
        : "Waiting to start…";
      await new Promise((r) => setTimeout(r, 700));
    }
  }

  // ------------------------------------------------------------ review
  function kpi(label, value, sub, cls = "") {
    return `<div class="kpi ${cls}"><div class="label">${esc(label)}</div><div class="value">${value}</div><div class="sub">${esc(sub)}</div></div>`;
  }

  function renderRun(payload) {
    current = payload;
    const run = payload.run;
    const done = run.status === "completed";
    $("#review-empty").hidden = true;
    $("#review-body").hidden = false;
    $("#rv-title").textContent = `Run #${run.id} · ${run.filename}`;
    const advisor = run.provider === "mock" ? "Demo mode (built-in rules)" : `${run.provider} · ${run.model}`;
    $("#rv-meta").textContent =
      `${fmtDate(run.created_at)} · ${advisor} · ${run.llm_calls} AI calls · ${run.prompt_tokens + run.completion_tokens} tokens` +
      (run.advice_fallbacks ? ` · ${run.advice_fallbacks} recommendations from built-in rules` : "");

    $("#rv-top-actions").innerHTML =
      (done ? `<a class="btn primary" href="/api/runs/${run.id}/download" download>Download cleaned CSV</a>` : "") +
      `<a class="btn" href="/api/runs/${run.id}/audit.md" download>Audit report</a>` +
      `<button class="btn" data-goto="analyze">Check another file</button>`;

    // KPIs
    const scoreCell = done
      ? `<div class="score-arrow"><span class="from">${run.score_before.toFixed(1)}</span><span class="to ${run.score_after < run.score_before ? "worse" : ""}">${run.score_after.toFixed(1)}</span></div>`
      : `${run.score_before.toFixed(1)}`;
    const grade = (payload.score_after || payload.score_before || {}).score;
    $("#rv-kpis").innerHTML = [
      kpi("Quality score", scoreCell, done ? "out of 100, after fixes" : "out of 100, before fixes", grade >= 85 ? "positive" : grade >= 70 ? "neutral" : "negative"),
      kpi("Issues found", run.issues_found, done ? `${payload.issues_after.length} still outstanding` : "awaiting your approval"),
      kpi("Rows", done ? `${run.rows_before.toLocaleString()} → ${run.rows_after.toLocaleString()}` : run.rows_before.toLocaleString(), done && run.rows_after !== run.rows_before ? `${(run.rows_before - run.rows_after).toLocaleString()} removed` : ""),
      kpi("Columns", done ? `${run.columns_before} → ${run.columns_after}` : run.columns_before, ""),
    ].join("");

    // dimensions
    const before = payload.score_before, after = payload.score_after;
    if (before) {
      $("#rv-dimensions-card").hidden = false;
      $("#rv-dimensions").innerHTML = ["completeness", "uniqueness", "validity", "consistency"].map((d) => {
        const b = before[d], a = after ? after[d] : null;
        const cls = b >= 85 ? "" : b >= 60 ? "mid" : "bad";
        const arrow = a != null && Math.abs(a - b) >= 0.05 ? ` <span class="dim-after">→ ${a.toFixed(0)}</span>` : "";
        return `<div class="dim-row"><span>${pretty(d)}</span><div class="dim-track"><div class="dim-fill ${cls}" style="width:${Math.max(0, Math.min(100, b))}%"></div></div><span class="muted">${b.toFixed(0)}/100${arrow}</span></div>`;
      }).join("");
    } else $("#rv-dimensions-card").hidden = true;

    // issues + approval
    selection = {};
    payload.issues.forEach((i) => {
      const chosen = i.decision ? i.decision.strategy : (i.advice ? i.advice.strategy : i.default_strategy);
      selection[i.id] = {
        approved: i.decision ? i.decision.approved : chosen !== "none",
        strategy: chosen,
        fill_value: (i.decision && i.decision.fill_value) || (i.advice && i.advice.fill_value) || "",
      };
    });
    renderIssues(payload.issues, done);
    $("#rv-approve-card").hidden = payload.issues.length === 0;

    // applied + remaining
    if (payload.applied.length) {
      $("#rv-result-card").hidden = false;
      $("#rv-applied").innerHTML = `<div class="fix-log">` + payload.applied.map((f) =>
        `<div class="row-item"><span class="mark">${f.applied ? "✅" : "⏭️"}</span><div>` +
        `<strong>${esc(pretty(f.strategy))}</strong> <code>${esc(f.column || "whole table")}</code><br>` +
        `<span class="muted">${esc(f.detail)}</span></div></div>`).join("") + `</div>`;
    } else $("#rv-result-card").hidden = true;

    if (done) {
      $("#rv-remaining-card").hidden = false;
      $("#rv-remaining").innerHTML = payload.issues_after.length
        ? `<div class="fix-log">` + payload.issues_after.map((i) =>
            `<div class="row-item"><span class="mark"><span class="sev ${i.severity}">${i.severity}</span></span>` +
            `<div><strong>${esc(i.title)}</strong><br><span class="muted">${esc(i.detail)}</span></div></div>`).join("") + `</div>`
        : `<p class="empty">Nothing left. Every check passes on the cleaned data. 🎉</p>`;
    } else $("#rv-remaining-card").hidden = true;

    // audit
    $("#rv-audit").innerHTML =
      "<thead><tr><th>When</th><th>Who</th><th>Event</th><th>Detail</th></tr></thead><tbody>" +
      payload.audit.map((a) => `<tr><td>${esc(fmtDate(a.at))}</td><td><span class="actor ${esc(a.actor)}">${esc(a.actor)}</span></td><td>${esc(a.event)}</td><td title="${esc(a.detail)}">${esc(a.detail)}</td></tr>`).join("") +
      "</tbody>";

    $("#chip-cleaned").disabled = !done;
    $$("#rv-preview-toggle .chip").forEach((c) => c.classList.toggle("active", c.dataset.which === "original"));
    loadPreview("original");
  }

  function renderIssues(issues, done) {
    $("#rv-issues").innerHTML = issues.map((issue) => {
      const sel = selection[issue.id];
      const adv = issue.advice;
      const destructive = DESTRUCTIVE.has(sel.strategy);
      const options = issue.allowed_strategies.map((s) =>
        `<option value="${esc(s)}" ${s === sel.strategy ? "selected" : ""}>${esc(pretty(s))}</option>`).join("");
      return `<div class="issue-card ${sel.approved ? "" : "rejected"} ${destructive && sel.approved ? "destructive" : ""}" data-issue="${esc(issue.id)}">
        <div class="issue-head">
          <div class="grow">
            <p class="issue-title">${esc(issue.title)}</p>
            <div class="issue-meta">
              <span class="sev ${esc(issue.severity)}">${esc(issue.severity)}</span>
              <span class="muted">${esc(pretty(issue.issue_type))}</span>
              <span class="muted">·</span>
              <span class="muted">${issue.affected_rows.toLocaleString()} rows (${issue.affected_pct}%)</span>
            </div>
            <p class="issue-detail">${esc(issue.detail)}</p>
          </div>
        </div>
        ${issue.evidence.length ? `<div class="evidence">${issue.evidence.map((e) => `<span>${esc(e)}</span>`).join("")}</div>` : ""}
        ${adv ? `<div class="advice"><div class="who">${adv.source === "llm" ? "AI recommendation" : "Built-in rule"} · confidence ${(adv.confidence * 100).toFixed(0)}%</div><p>${esc(adv.reasoning)}</p></div>` : ""}
        <div class="fix-row">
          <label>Fix to apply
            <select data-role="strategy" ${done ? "disabled" : ""}>${options}</select>
          </label>
          <label class="approve"><input type="checkbox" data-role="approve" ${sel.approved ? "checked" : ""} ${done ? "disabled" : ""}> Approve</label>
          <div class="what-it-does" data-role="help">${esc(STRATEGY_HELP[sel.strategy] || "")}</div>
          <label data-role="fill-wrap" ${sel.strategy === "fill_constant" ? "" : "hidden"} style="grid-column:1/-1">Value to fill in
            <input data-role="fill" value="${esc(sel.fill_value)}" placeholder="for example Unknown or 0" ${done ? "disabled" : ""}>
          </label>
        </div>
      </div>`;
    }).join("");

    $$("#rv-issues .issue-card").forEach((card) => {
      const id = card.dataset.issue;
      card.querySelector("[data-role=strategy]").addEventListener("change", (e) => {
        selection[id].strategy = e.target.value;
        card.querySelector("[data-role=help]").textContent = STRATEGY_HELP[e.target.value] || "";
        card.querySelector("[data-role=fill-wrap]").hidden = e.target.value !== "fill_constant";
        card.classList.toggle("destructive", DESTRUCTIVE.has(e.target.value) && selection[id].approved);
        updateApplyBar();
      });
      card.querySelector("[data-role=approve]").addEventListener("change", (e) => {
        selection[id].approved = e.target.checked;
        card.classList.toggle("rejected", !e.target.checked);
        card.classList.toggle("destructive", DESTRUCTIVE.has(selection[id].strategy) && e.target.checked);
        updateApplyBar();
      });
      const fill = card.querySelector("[data-role=fill]");
      if (fill) fill.addEventListener("input", (e) => { selection[id].fill_value = e.target.value; });
    });
    $("#btn-apply").disabled = done;
    $("#btn-apply").textContent = done ? "Fixes already applied" : "Apply approved fixes";
    updateApplyBar();
  }

  function updateApplyBar() {
    const entries = Object.values(selection);
    const approved = entries.filter((s) => s.approved && s.strategy !== "none");
    const destructive = approved.filter((s) => DESTRUCTIVE.has(s.strategy));
    $("#rv-approved-count").textContent = `${approved.length} fix${approved.length === 1 ? "" : "es"} selected`;
    $("#rv-destructive-note").textContent = destructive.length
      ? `${destructive.length} of them remove rows or columns.` : "";
  }

  $("#btn-select-all").addEventListener("click", () => bulkSelect(() => true));
  $("#btn-select-none").addEventListener("click", () => bulkSelect(() => false));
  $("#btn-select-safe").addEventListener("click", () => bulkSelect((s) => !DESTRUCTIVE.has(s.strategy)));
  function bulkSelect(predicate) {
    Object.entries(selection).forEach(([id, s]) => { s.approved = predicate(s) && s.strategy !== "none"; });
    if (current) renderIssues(current.issues, current.run.status === "completed");
  }

  $("#btn-apply").addEventListener("click", async () => {
    if (!current) return;
    const btn = $("#btn-apply");
    const decisions = Object.entries(selection).map(([issue_id, s]) => ({
      issue_id, approved: s.approved, strategy: s.strategy, fill_value: s.fill_value || null,
    }));
    const destructive = decisions.filter((d) => d.approved && DESTRUCTIVE.has(d.strategy));
    if (destructive.length && !confirm(
      `${destructive.length} of the selected fixes remove rows or columns. Your original file is kept, ` +
      `and the result is saved as a separate cleaned copy. Continue?`)) return;

    btn.disabled = true;
    btn.textContent = "Applying…";
    $("#rv-apply-error").hidden = true;
    try {
      const payload = await postJSON(`/api/runs/${current.run.id}/apply`, { decisions });
      renderRun(payload);
      toast(`Done. Quality score ${payload.run.score_before.toFixed(1)} → ${payload.run.score_after.toFixed(1)}.`, 6000);
      $("#rv-result-card").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      $("#rv-apply-error").textContent = err.message;
      $("#rv-apply-error").hidden = false;
      btn.disabled = false;
      btn.textContent = "Apply approved fixes";
    }
  });

  // ------------------------------------------------------------ preview
  async function loadPreview(which) {
    if (!current) return;
    try {
      const p = await getJSON(`/api/runs/${current.run.id}/preview?which=${which}&rows=20`);
      $("#rv-preview").innerHTML =
        "<thead><tr>" + p.columns.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead><tbody>" +
        p.rows.map((r) => "<tr>" + r.map((v) => `<td title="${esc(v)}">${esc(v)}</td>`).join("") + "</tr>").join("") +
        "</tbody>";
      $("#rv-preview-note").textContent = `Showing ${p.rows.length} of ${p.total_rows.toLocaleString()} rows (${which}).`;
    } catch (err) {
      $("#rv-preview").innerHTML = `<tbody><tr><td class="empty">${esc(err.message)}</td></tr></tbody>`;
      $("#rv-preview-note").textContent = "";
    }
  }
  $$("#rv-preview-toggle .chip").forEach((c) => c.addEventListener("click", () => {
    if (c.disabled) return;
    $$("#rv-preview-toggle .chip").forEach((x) => x.classList.toggle("active", x === c));
    loadPreview(c.dataset.which);
  }));

  // ------------------------------------------------------------ history
  async function loadHistory() {
    const [overview, runs] = await Promise.all([getJSON("/api/overview"), getJSON("/api/runs?limit=100")]);
    $("#h-kpis").innerHTML = [
      kpi("Datasets checked", overview.runs, ""),
      kpi("Issues found", overview.issues_found, "all time"),
      kpi("Fixes applied", overview.fixes_applied, ""),
      kpi("Average improvement", (overview.average_score_lift >= 0 ? "+" : "") + overview.average_score_lift, "quality score points", "positive"),
    ].join("");
    const tbody = $("#h-runs tbody");
    tbody.innerHTML = runs.length ? runs.map((r) => {
      const score = r.status === "completed"
        ? `${r.score_before.toFixed(1)} → <strong>${r.score_after.toFixed(1)}</strong>`
        : r.score_before ? r.score_before.toFixed(1) : "-";
      return `<tr class="clickable" data-run="${r.id}"><td>${r.id}</td><td>${esc(r.filename)}</td>` +
        `<td>${esc(fmtDate(r.created_at))}</td><td>${r.rows_before.toLocaleString()}</td><td>${r.issues_found}</td>` +
        `<td>${score}</td><td>${esc(pretty(r.status))}${r.error ? ` <span class="hint">${esc(r.error.slice(0, 70))}</span>` : ""}</td>` +
        `<td><button class="btn small danger" data-delete="${r.id}">Delete</button></td></tr>`;
    }).join("") : '<tr><td colspan="8" class="empty">Nothing checked yet.</td></tr>';

    tbody.querySelectorAll("tr[data-run]").forEach((tr) => tr.addEventListener("click", async (e) => {
      if (e.target.closest("[data-delete]")) return;
      const run = runs.find((x) => x.id === Number(tr.dataset.run));
      if (run.status === "failed") { toast(run.error || "That run failed."); return; }
      renderRun(await getJSON(`/api/runs/${run.id}`));
      showTab("review");
    }));
    tbody.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete run #${b.dataset.delete} and its uploaded file?`)) return;
      await api(`/api/runs/${b.dataset.delete}`, { method: "DELETE" });
      loadHistory();
    }));
  }

  // ------------------------------------------------------------ boot
  (async function boot() {
    const h = await refreshHealth();
    try { STRATEGY_HELP = await getJSON("/api/strategies"); } catch { STRATEGY_HELP = {}; }
    await loadSamples();
    if (h && !h.llm_ready) showTab("setup"); else showTab("analyze");
    if (h) loadSettings();
    setInterval(refreshHealth, 30000);
  })();
})();
