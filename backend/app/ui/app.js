// UI <-> engine: REST for commands, one WebSocket for live events.
(() => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const $ = (id) => document.getElementById(id);
  let ui = localStorage.getItem("ui_language") || "ar";
  let state = { state: "idle", settings: {} };
  let languages = [];
  const T = (key, ...a) => { const v = (I18N[ui] || I18N.ar)[key]; return typeof v === "function" ? v(...a) : (v ?? key); };

  // ---------- http ----------
  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { "x-token": token, "content-type": "application/json", ...(opts.headers || {}) } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(body.message || body.detail || res.statusText), { code: body.code, status: res.status });
    return body;
  }

  // ---------- i18n ----------
  function applyI18n() {
    document.documentElement.lang = ui;
    document.documentElement.dir = ui === "ar" ? "rtl" : "ltr";
    document.title = T("app_name");
    document.querySelectorAll("[data-i18n]").forEach((el) => (el.innerHTML = T(el.dataset.i18n)));
    document.querySelectorAll("[data-i18n-title]").forEach((el) => (el.title = T(el.dataset.i18nTitle)));
    $("btn-lang-ui").textContent = ui === "ar" ? "EN" : "ع";
    render();
  }

  // ---------- toasts ----------
  let toastTimer;
  function toast(text, bad = false) {
    const t = $("toast");
    t.textContent = text;
    t.className = "toast" + (bad ? " bad" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), bad ? 7000 : 3000);
  }
  const errorText = (code, message) => (I18N[ui]["err_" + code] ? T("err_" + code) : `${T("err_engine")}: ${message || code}`);

  // ---------- selects ----------
  function fillLanguageSelect(sel, value, withAuto) {
    sel.innerHTML = "";
    if (withAuto) sel.add(new Option(T("auto"), "auto"));
    for (const l of languages) {
      if (sel.id === "my_language" && !l.can_listen) continue; // we must understand your speech
      const name = ui === "ar" ? (AR_NAMES[l.code] || l.name) : l.name;
      const native = l.native && l.native !== name && !l.code.startsWith("ar") ? ` (${l.native})` : "";
      const label = `${name}${native}${l.has_voice ? "" : " " + T("no_voice")}`;
      sel.add(new Option(label, l.code));
    }
    sel.value = value;
  }

  function fillDeviceSelect(sel, names, value) {
    sel.innerHTML = "";
    sel.add(new Option(T("default_device"), ""));
    for (const n of names) sel.add(new Option(n, n));
    sel.value = value || "";
  }

  async function loadMeetingApps(value) {
    const dev = await api("/api/devices");
    const sel = $("meeting_app");
    sel.innerHTML = "";
    sel.add(new Option(T("all_apps"), "system"));
    for (const app of dev.known_apps) {
      const running = dev.meeting_apps.includes(app);
      sel.add(new Option(app[0].toUpperCase() + app.slice(1) + (running ? ` — ${T("running_now")}` : ""), app));
    }
    sel.value = value || "system";
    return dev;
  }

  // ---------- rendering ----------
  function render() {
    const s = state.state;
    const running = ["running", "paused"].includes(s);
    const busy = ["starting", "loading_models", "stopping"].includes(s);
    $("status").className = "pill " + (busy ? "loading" : s);
    $("status-text").textContent = T("st_" + s);
    const start = $("btn-start");
    start.classList.toggle("on", running);
    start.classList.toggle("busy", busy);
    start.disabled = busy;
    $("start-text").textContent = busy ? T(s === "loading_models" ? "loading" : "starting") : T(running ? "stop" : "start");
    for (const id of ["btn-mute", "btn-pause", "btn-replay"]) $(id).disabled = !running;
    $("btn-pause").textContent = T(s === "paused" ? "resume" : "pause");
    for (const id of ["my_language", "other_language", "meeting_app", "btn-swap", "btn-demo"]) $(id).disabled = running || busy;
    const hint = $("hint");
    hint.hidden = !(running && state.mic_for_meeting);
    if (!hint.hidden) hint.innerHTML = T("hint_mic", state.mic_for_meeting);
  }

  function langInfo(code) { return languages.find((l) => l.code === code) || { code, flag: "", name: code, rtl: false }; }

  function addMessage(m) {
    $("empty").hidden = true;
    const box = $("messages");
    const div = document.createElement("div");
    div.className = "msg " + m.speaker;
    const src = langInfo(m.source_language), tgt = langInfo(m.target_language);
    const lat = m.latency_ms != null ? `${(m.latency_ms / 1000).toFixed(1)} ${T("ms")}` : T("text_only");
    const lname = (l) => (ui === "ar" ? AR_NAMES[l.code] || l.name : l.name);
    div.innerHTML = `<div class="who">${m.speaker === "me" ? T("you") : T("them")} · ${lname(src)} ← ${lname(tgt)}</div>
      <div class="orig" dir="auto"></div><div class="tr" dir="auto"></div><div class="meta">${lat}</div>`;
    div.querySelector(".orig").textContent = m.source_text;
    div.querySelector(".tr").textContent = m.translated_text;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  function setLive(direction, text) {
    const el = $(direction === "outgoing" ? "live-me" : "live-other");
    el.hidden = !text;
    el.textContent = text || "";
    el.dir = "auto";
  }

  // ---------- events ----------
  function connect() {
    const ws = new WebSocket(`ws://${location.host}/ws?t=${encodeURIComponent(token)}`);
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      switch (ev.type) {
        case "hello":
          state = ev;
          $("messages").querySelectorAll(".msg").forEach((n) => n.remove());
          (ev.conversation || []).forEach(addMessage);
          render();
          break;
        case "status":
          if (!ev.direction) { state.state = ev.state; render(); }
          break;
        case "ready":
          state.mic_for_meeting = ev.mic_for_meeting;
          render();
          break;
        case "partial":
          setLive(ev.direction, ev.text);
          break;
        case "final":
          setLive(ev.direction, "");
          if (ev.message) addMessage(ev.message);
          break;
        case "setup_progress":
          onSetupProgress(ev);
          break;
        case "interrupted":
          toast(T("interrupted"));
          break;
        case "error":
          toast(errorText(ev.code, ev.message), true);
          if (ev.code === "virtual_mic_missing") showSetup();
          break;
      }
    };
    ws.onclose = () => setTimeout(connect, 1000); // engine restarts / sleep: reconnect
  }

  // ---------- first run: model downloads ----------
  const size = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} ${T("gb")}` : `${mb} ${T("mb")}`);
  let setupPlan = null;
  async function checkFirstRun() {
    setupPlan = await api("/api/setup");
    $("firstrun").hidden = setupPlan.ready;
    if (setupPlan.ready) return;
    const missing = setupPlan.components.filter((c) => !c.installed);
    $("fr-list").innerHTML = missing.map((c) => `<div class="fr-item"><span>${c.title_ar}</span><span>${size(c.size_mb)}</span></div>`).join("")
      + `<div class="fr-item"><b>${T("fr_total")}</b><b>${size(missing.reduce((a, c) => a + c.size_mb, 0))}</b></div>`;
    if (setupPlan.downloading) startedDownload();
  }
  function startedDownload() { $("btn-download").disabled = true; $("btn-download").textContent = "…"; }
  $("btn-download").onclick = async () => { startedDownload(); await api("/api/setup/download", { method: "POST" }); };
  function onSetupProgress(ev) {
    if (ev.component === "all") {
      if (ev.finished) {
        $("fr-status").textContent = T("fr_done");
        setTimeout(() => { $("firstrun").hidden = true; showSetup(); }, 1200);
      } else {
        $("fr-status").textContent = `${T("fr_failed")}: ${ev.error}`;
        $("btn-download").disabled = false;
        $("btn-download").textContent = T("fr_retry");
      }
      return;
    }
    const missing = (setupPlan?.components || []).filter((c) => !c.installed);
    const totalMb = missing.reduce((a, c) => a + c.size_mb, 0) || 1;
    const before = missing.slice(0, missing.findIndex((c) => c.id === ev.component)).reduce((a, c) => a + c.size_mb, 0);
    const pct = Math.min(100, ((before + Math.min(ev.done / 1e6, ev.total / 1e6)) / totalMb) * 100);
    $("fr-bar").style.width = pct.toFixed(1) + "%";
    $("fr-status").textContent = `${ev.title} — ${Math.round(ev.done / 1e6)} / ${Math.round(ev.total / 1e6)} ${T("mb")} (${pct.toFixed(0)}%)`;
  }

  // ---------- setup check ----------
  async function showSetup(force = false) {
    const d = await api("/api/doctor");
    const box = $("setup");
    if (d.ready && !force) { box.hidden = true; return; }
    box.hidden = false;
    box.innerHTML = `<h3>${d.ready ? T("setup_ok") : T("setup_title")}</h3>` + d.checks.map((c) => `
      <div class="check"><span>${c.ok ? "✅" : c.required ? "❌" : "⚠️"}</span><b>${c.title_ar}</b>
      ${c.detail ? `<span class="detail">${escapeHtml(c.detail)}</span>` : ""}
      ${!c.ok && c.fix_ar ? `<span class="fix">${linkify(escapeHtml(c.fix_ar))}</span>` : ""}</div>`).join("");
  }
  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const linkify = (s) => s.replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

  // ---------- actions ----------
  async function saveSettings(changes) {
    try {
      state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(changes) });
    } catch (e) {
      toast(e.status === 409 ? T("stop_first") : e.message, true);
      throw e;
    }
  }

  async function action(name, query = "") {
    try {
      const r = await api(`/api/session/${name}${query}`, { method: "POST" });
      if (r.state) { state = { ...state, ...r }; render(); }
      return r;
    } catch (e) {
      toast(errorText(e.code, e.message), true);
      if (e.code === "virtual_mic_missing") showSetup();
    }
  }

  $("btn-start").onclick = async () => {
    if (["running", "paused"].includes(state.state)) return action("stop");
    state.state = "starting"; render();
    await saveSettings({ my_language: $("my_language").value, other_language: $("other_language").value,
                         meeting_app: $("meeting_app").value }).catch(() => {});
    await action("start");
    const r = await api("/api/state"); state = { ...state, ...r }; render();
  };
  let muted = false;
  $("btn-mute").onclick = async () => {
    muted = !muted;
    await action(muted ? "mute" : "unmute");
    $("btn-mute").classList.toggle("active", muted);
    $("btn-mute").textContent = T(muted ? "unmute" : "mute");
  };
  $("btn-demo").onclick = async () => {
    state.state = "starting"; render();
    toast(T("demo_running"));
    await action("demo");
    const r = await api("/api/state"); state = { ...state, ...r }; render();
  };
  $("btn-pause").onclick = () => action(state.state === "paused" ? "resume" : "pause");
  $("btn-replay").onclick = () => action("replay", "?speaker=other");
  $("btn-swap").onclick = () => {
    const a = $("my_language").value, b = $("other_language").value;
    if (b === "auto" || !langInfo(b).can_listen) return;
    $("my_language").value = b; $("other_language").value = a;
  };
  $("show_original").onchange = (e) => document.body.classList.toggle("hide-original", !e.target.checked);
  $("btn-lang-ui").onclick = () => {
    ui = ui === "ar" ? "en" : "ar";
    localStorage.setItem("ui_language", ui);
    applyI18n();
    fillLanguageSelect($("my_language"), state.settings.my_language, false);
    fillLanguageSelect($("other_language"), state.settings.other_language, true);
    loadMeetingApps(state.settings.meeting_app);
  };

  // settings dialog
  const dlg = $("settings");
  const toggles = ["headphones", "live_subtitles", "hear_my_translation", "save_history"];
  $("btn-settings").onclick = async () => {
    const s = state.settings, dev = await api("/api/devices");
    fillDeviceSelect($("input_device"), dev.inputs, s.input_device);
    fillDeviceSelect($("output_device"), dev.outputs, s.output_device);
    $("latency_mode").value = s.latency_mode;
    $("speech_speed").value = s.speech_speed;
    $("speed_out").textContent = `×${Number(s.speech_speed).toFixed(2)}`;
    toggles.forEach((k) => ($(k).checked = !!s[k]));
    dlg.showModal();
  };
  $("speech_speed").oninput = (e) => ($("speed_out").textContent = `×${Number(e.target.value).toFixed(2)}`);
  $("btn-save").onclick = async (e) => {
    e.preventDefault();
    const changes = { input_device: $("input_device").value || null, output_device: $("output_device").value || null,
      latency_mode: $("latency_mode").value, speech_speed: Number($("speech_speed").value) };
    toggles.forEach((k) => (changes[k] = $(k).checked));
    try { await saveSettings(changes); dlg.close(); toast(T("saved")); } catch {}
  };
  $("btn-history").onclick = async () => {
    const rows = await api("/api/history");
    dlg.close();
    if (!rows.length) return toast(T("history_empty"));
    $("messages").querySelectorAll(".msg").forEach((n) => n.remove());
    rows.forEach((r) => addMessage({ ...r, latency_ms: null }));
  };
  $("btn-clear-history").onclick = async () => { await api("/api/history", { method: "DELETE" }); toast(T("cleared")); };
  $("btn-doctor").onclick = () => { dlg.close(); showSetup(true); };

  // ---------- boot ----------
  (async () => {
    applyI18n();
    const [st, langs] = await Promise.all([api("/api/state"), api("/api/languages")]);
    state = st; languages = langs;
    fillLanguageSelect($("my_language"), st.settings.my_language, false);
    fillLanguageSelect($("other_language"), st.settings.other_language, true);
    await loadMeetingApps(st.settings.meeting_app);
    render();
    connect();
    await checkFirstRun();
    if (setupPlan.ready) showSetup();
  })().catch((e) => toast(e.message, true));
})();
