// UI <-> engine: REST for commands, one WebSocket for live events.
(() => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const $ = (id) => document.getElementById(id);
  let ui = localStorage.getItem("ui_language") || "ar";
  let state = { state: "idle", settings: {} };
  let languages = [];
  let convo = []; // messages on screen, for copy / export / average delay
  const escapeHtml = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
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
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => (el.placeholder = T(el.dataset.i18nPlaceholder)));
    $("btn-lang-ui").textContent = ui === "ar" ? "EN" : "ع";
    render();
  }

  // ---------- toasts ----------
  let toastTimer;
  function toast(text, kind = false) {
    const t = $("toast");
    const cls = kind === true ? "bad" : kind || "";
    t.textContent = text;
    t.className = "toast" + (cls ? " " + cls : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), cls ? 7000 : 3000);
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
    const muted = running && !!state.mic_muted;
    $("btn-mute").classList.toggle("active", muted);
    $("btn-mute").textContent = T(muted ? "unmute" : "mute");
    $("auto-note").hidden = $("other_language").value !== "auto";
    $("meters").hidden = !running;
    for (const id of ["btn-copy", "btn-export", "btn-clear-view"]) $(id).disabled = !convo.length;
    for (const id of ["my_language", "other_language", "meeting_app", "btn-swap", "btn-demo"]) $(id).disabled = running || busy;
    const hint = $("hint");
    hint.hidden = !(running && state.mic_for_meeting);
    if (!hint.hidden) hint.innerHTML = T("hint_mic", escapeHtml(state.mic_for_meeting));
  }

  function langInfo(code) { return languages.find((l) => l.code === code) || { code, flag: "", name: code, rtl: false }; }

  function updateAverage() {
    const lat = convo.map((m) => m.latency_ms).filter((v) => v != null);
    $("avg-latency").hidden = !lat.length;
    if (lat.length) $("avg-latency").textContent = T("avg_latency", (lat.reduce((a, b) => a + b, 0) / lat.length / 1000).toFixed(1));
  }

  function clearMessages() {
    convo = [];
    $("messages").querySelectorAll(".msg").forEach((n) => n.remove());
    $("empty").hidden = false;
    updateAverage();
    render();
  }

  function addMessage(m) {
    $("empty").hidden = true;
    convo.push(m);
    const box = $("messages");
    const div = document.createElement("div");
    div.className = "msg " + m.speaker;
    const src = langInfo(m.source_language), tgt = langInfo(m.target_language);
    const lat = m.latency_ms != null ? `${(m.latency_ms / 1000).toFixed(1)} ${T("ms")}`
      : m.same_language ? "" : m.spoken === false || m.text_only ? T("text_only") : "";
    const lname = (l) => (ui === "ar" ? AR_NAMES[l.code] || l.name : l.name);
    div.innerHTML = `<div class="who">${m.speaker === "me" ? T("you") : T("them")} · ${lname(src)} ← ${lname(tgt)}</div>
      <div class="orig" dir="auto"></div><div class="tr" dir="auto"></div>
      <div class="meta">${[lat, m.same_language ? T("same_language") : ""].filter(Boolean).join(" · ")}</div>`;
    div.querySelector(".orig").textContent = m.source_text;
    div.querySelector(".tr").textContent = m.translated_text;
    // Stay at the bottom only if the user hasn't scrolled up to re-read something.
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    box.appendChild(div);
    if (atBottom) box.scrollTop = box.scrollHeight;
    updateAverage();
    render();
  }

  // Live line per side: "● ● ●" as soon as someone starts talking, then the partial text.
  function setLive(direction, text, speaking = false) {
    const el = $(direction === "outgoing" ? "live-me" : "live-other");
    el.hidden = !text && !speaking;
    el.classList.toggle("speaking", speaking || !!text);
    el.querySelector(".speaker").textContent = (direction === "outgoing" ? T("you") : T("them")) + ":";
    const t = el.querySelector(".text");
    t.textContent = text || "";
    t.dir = "auto";
  }

  // Level meter: -60 dBFS (silence) .. 0 dBFS (full scale)
  function setLevel(ev) {
    const m = $(ev.direction === "outgoing" ? "meter-outgoing" : "meter-incoming");
    const pct = Math.max(0, Math.min(100, ((ev.db + 60) / 60) * 100));
    m.querySelector("i").style.width = pct.toFixed(0) + "%";
    m.classList.toggle("speaking", !!ev.speaking);
    m.classList.toggle("clip", !!ev.clipping);
  }

  function conversationText() {
    const lname = (code) => { const l = langInfo(code); return ui === "ar" ? AR_NAMES[l.code] || l.name : l.name; };
    return convo.map((m) => {
      const when = m.timestamp ? new Date(m.timestamp * 1000).toLocaleTimeString() : "";
      const who = m.speaker === "me" ? T("you") : T("them");
      return `[${when}] ${who} (${lname(m.source_language)}): ${m.source_text}\n    → (${lname(m.target_language)}): ${m.translated_text}`;
    }).join("\n\n");
  }

  // ---------- events ----------
  function connect() {
    const ws = new WebSocket(`ws://${location.host}/ws?t=${encodeURIComponent(token)}`);
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      switch (ev.type) {
        case "hello":
          state = ev;
          clearMessages();
          (ev.conversation || []).forEach(addMessage);
          render();
          break;
        case "status":
          if (!ev.direction) {
            state.state = ev.state;
            if (ev.state === "idle") { state.mic_muted = false; setLive("outgoing", ""); setLive("incoming", ""); }
            render();
          }
          break;
        case "mic":
          state.mic_muted = ev.muted;
          render();
          break;
        case "speech_started":
          setLive(ev.direction, "", true);
          break;
        case "warning":
          if (ev.code === "device_missing") toast(T("warn_device_missing", ev.device), "warn");
          else if (ev.code === "mic_quiet") toast(T("warn_mic_quiet"), "warn");
          break;
        case "level":
          setLevel(ev);
          break;
        case "recovered":
          toast(T("recovered"));
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
    $("fr-list").innerHTML = missing.map((c) => `<div class="fr-item"><span>${escapeHtml(c.title_ar)}</span><span>${size(c.size_mb)}</span></div>`).join("")
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
  $("btn-mute").onclick = async () => {
    const r = await action(state.mic_muted ? "unmute" : "mute");
    if (r) render();
  };
  $("btn-demo").onclick = async () => {
    state.state = "starting"; render();
    toast(T("demo_running"));
    await action("demo");
    const r = await api("/api/state"); state = { ...state, ...r }; render();
  };
  $("btn-pause").onclick = () => action(state.state === "paused" ? "resume" : "pause");
  $("btn-replay").onclick = async () => {
    const r = await action("replay", "?speaker=other");
    if (r && r.ok === false) toast(T("nothing_to_replay"));
  };
  $("other_language").onchange = render;
  $("btn-clear-view").onclick = clearMessages;
  $("btn-copy").onclick = async () => {
    try { await navigator.clipboard.writeText(conversationText()); toast(T("copied")); }
    catch (e) { toast(e.message, true); }
  };
  $("btn-export").onclick = () => {
    const blob = new Blob(["\ufeff" + conversationText()], { type: "text/plain;charset=utf-8" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob), download: `conversation-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.txt` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  // Keyboard shortcuts (handy while the meeting window has the focus... once you click back here)
  document.addEventListener("keydown", (e) => {
    if (!(e.ctrlKey || e.metaKey) || $("settings").open) return;
    const key = e.key.toLowerCase();
    const running = ["running", "paused"].includes(state.state);
    if (key === "enter" && !$("btn-start").disabled) { e.preventDefault(); $("btn-start").click(); }
    else if (key === "m" && running) { e.preventDefault(); $("btn-mute").click(); }
    else if (key === "r" && running) { e.preventDefault(); $("btn-replay").click(); }
  });
  $("btn-swap").onclick = () => {
    const a = $("my_language").value, b = $("other_language").value;
    if (b === "auto" || !langInfo(b).can_listen) return;
    $("my_language").value = b; $("other_language").value = a;
    render();
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
  const toggles = ["headphones", "live_subtitles", "hear_my_translation", "save_history", "auto_gain"];

  // ---------- voices ----------
  const voiceLabel = (v) => {
    const bits = [v.name.replace(/_/g, " ")];
    if (v.gender) bits.push(T(v.gender));
    if (v.quality) bits.push(T("q_" + v.quality));
    if (!v.installed) bits.push("⬇ " + T("voice_download", v.size_mb));
    return bits.join(" · ");
  };
  async function loadVoices(sel, label, language) {
    const l = langInfo(language);
    label.textContent = T("voice_for", ui === "ar" ? AR_NAMES[l.code] || l.name : l.name);
    sel.dataset.language = language;
    sel.innerHTML = "";
    sel.add(new Option(T("voice_auto"), ""));
    try {
      const r = await api(`/api/voices?language=${encodeURIComponent(language)}`);
      for (const v of r.voices) sel.add(new Option(voiceLabel(v), v.id));
      sel.value = r.selected && r.voices.some((v) => v.id === r.selected) ? r.selected : "";
    } catch (e) { /* offline and nothing installed: automatic only */ }
    sel.dataset.initial = sel.value;
  }
  async function preview(sel, btn) {
    btn.disabled = true;
    try {
      await api("/api/voices/preview", { method: "POST", body: JSON.stringify({ language: sel.dataset.language, voice: sel.value || null }) });
      if (sel.value) { // a downloaded voice is now installed: refresh the label
        const keep = sel.value; await loadVoices(sel, sel.previousElementSibling, sel.dataset.language); sel.value = keep;
      }
    } catch (e) { toast(T("err_voice_preview"), true); }
    btn.disabled = false;
  }
  $("preview-mine").onclick = () => preview($("voice-mine"), $("preview-mine"));
  $("preview-theirs").onclick = () => preview($("voice-theirs"), $("preview-theirs"));
  $("btn-settings").onclick = async () => {
    const s = state.settings, dev = await api("/api/devices");
    fillDeviceSelect($("input_device"), dev.inputs, s.input_device);
    fillDeviceSelect($("output_device"), dev.outputs, s.output_device);
    $("latency_mode").value = s.latency_mode;
    $("speech_speed").value = s.speech_speed;
    $("speed_out").textContent = `×${Number(s.speech_speed).toFixed(2)}`;
    toggles.forEach((k) => ($(k).checked = !!s[k]));
    $("smart_interruptions").checked = s.interruptions !== "off";
    $("translation_model").value = s.translation_model;
    $("noise_reduction").value = s.noise_reduction;
    $("glossary").value = s.glossary || "";
    dlg.showModal();
    loadVoices($("voice-mine"), $("voice-mine-label"), $("my_language").value);
    const other = $("other_language").value;
    $("voice-theirs-row").hidden = other === "auto";
    if (other !== "auto") loadVoices($("voice-theirs"), $("voice-theirs-label"), other);
  };
  $("speech_speed").oninput = (e) => ($("speed_out").textContent = `×${Number(e.target.value).toFixed(2)}`);
  $("btn-save").onclick = async (e) => {
    e.preventDefault();
    const changes = { input_device: $("input_device").value || null, output_device: $("output_device").value || null,
      latency_mode: $("latency_mode").value, speech_speed: Number($("speech_speed").value) };
    toggles.forEach((k) => (changes[k] = $(k).checked));
    changes.interruptions = $("smart_interruptions").checked ? "smart" : "off";
    changes.translation_model = $("translation_model").value;
    changes.noise_reduction = $("noise_reduction").value;
    changes.glossary = $("glossary").value.trim();
    const modelChanged = changes.translation_model !== state.settings.translation_model;
    try {
      await saveSettings(changes);
      for (const sel of [$("voice-mine"), $("voice-theirs")]) {
        if (sel.dataset.language && sel.value !== sel.dataset.initial && !sel.closest("[hidden]"))
          state.settings = await api("/api/voices", { method: "PUT", body: JSON.stringify({ language: sel.dataset.language, voice: sel.value || null }) });
      }
      dlg.close();
      toast(T(modelChanged ? "mt_download_needed" : "saved"));
      if (modelChanged) await checkFirstRun();
    } catch {}
  };
  $("btn-history").onclick = async () => {
    const rows = await api("/api/history");
    dlg.close();
    if (!rows.length) return toast(T("history_empty"));
    clearMessages();
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
