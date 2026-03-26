

        function jarvisDashboard() {
            return {
                currentView: "agent",
                commandPalette: { open: false, query: "", recent: [], activeIndex: 0, favorites: ["Open Voice Channel", "Execute Priority", "Open Mission Mode"], usage: {}, dragFavoriteIndex: null },
                missionMode: true,
                missionHud: { sidebandCollapsed: false, actionIndex: 0, lastCommandPulse: false },
                holoTilt: { x: 0, y: 0 },
                missionFeedPulse: false,
                missionEvents: [],
                agent: {
                    sessionId: null,
                    messages: [
                        { role: "assistant", text: "Jarvis online. Try /tool db_ping or /tool get_time." }
                    ],
                    tools: [],
                    input: "",
                    sending: false,
                    error: null,
                    streamEnabled: true,
                    llmMode: "quality",
                    llmModel: "",
                    llmModels: [],
                    llmTemperature: 0.2,
                    configLoading: false,
                    surface: "",
                },
                system: {
                    health: null,
                    error: null,
                },
                voice: {
                    supported: false,
                    listening: false,
                    speaking: false,
                    autoSpeak: true,
                    wakeEnabled: false,
                    handsFree: false,
                    wakeWord: "hey jarvis",
                    lastWakeAt: null,
                    transcript: "",
                    error: null,
                    recognition: null,
                    pipelineAvailable: false,
                    pipelineEnabled: false,
                    pipelineListening: false,
                    pipelineBusy: false,
                    pipelineStatus: "wake pipeline idle",
                    threshold: 0.45,
                    chunkMs: 960,
                    mediaStream: null,
                    audioContext: null,
                    wakeSource: null,
                    wakeProcessor: null,
                    wakeSink: null,
                    wakeBuffer: [],
                    wakeSampleCount: 0,
                    localAvailable: false,
                    localVoiceEnabled: false,
                    sttModel: "base",
                    ttsVoice: "mb-en1",
                    ttsRate: 138,
                    ttsPitch: 32,
                    localRecording: false,
                    localChunks: [],
                    localRecorder: null,
                    localStream: null,
                    localSink: null,
                    localAudio: null,
                    activePreset: "prime",
                    profiles: [
                        { id: "prime", label: "Prime", voice: "mb-en1", rate: 138, pitch: 32, style: "cinematic", hint: "cinematic male" },
                        { id: "ops", label: "Ops", voice: "mb-en1", rate: 148, pitch: 28, style: "operator", hint: "deep ops" },
                        { id: "arc", label: "Arc", voice: "en-us", rate: 154, pitch: 42, style: "crisp", hint: "future crisp" },
                    ],
                },
                profile: {
                    current: null,
                    dangerous: false,
                    selfTest: null,
                },
                memoryCore: {
                    overview: null,
                    archivedItems: [],
                    workspaces: [],
                    activeWorkspace: null,
                    workspacePolicy: null,
                    graph: null,
                    reminders: [],
                    nextActions: null,
                    loading: false,
                    archivedLoading: false,
                    consolidating: false,
                    windowHours: "720",
                    showArchived: false,
                    selectedIds: [],
                    draggedPinnedId: null,
                    draggedProjectId: null,
                    archivedQuery: "",
                    archivedTag: "",
                    archivedType: "",
                    savedFilters: [],
                    savedFilterName: "",
                    briefing: null,
                    workspaceName: "",
                    workspaceFocus: "",
                    workspaceDescription: "",
                    editModal: {
                        open: false,
                        id: null,
                        content: "",
                        subject: "",
                        memoryType: "general",
                        importance: 3,
                        pinned: false,
                        lane: "personal",
                        tags: [],
                        tagInput: "",
                    },
                    error: null,
                },
                goals: {
                    input: "",
                    autoApprove: false,
                    running: false,
                    lastRun: null,
                    history: [],
                },
                vision: {
                    capturing: false,
                    last: null,
                    provider: "auto",
                    model: "",
                    models: [],
                    configLoading: false,
                    ocrAvailable: false,
                    openaiConfigured: false,
                },
                schedules: {
                    items: [],
                    inputGoal: "",
                    inputMinutes: 60,
                    inputAutoApprove: false,
                    briefingJobs: [],
                    briefingMorningHour: 8,
                    briefingEveningHour: 18,
                    briefingTimezone: (Intl.DateTimeFormat().resolvedOptions().timeZone || "America/Chicago"),
                    briefingDelivery: {
                        enabled: false,
                        discord_enabled: false,
                        discord_webhook_url: "",
                        email_enabled: false,
                        email_to: "",
                        mobile_enabled: false,
                        mobile_push_url: "",
                        mobile_channel: "ntfy",
                    },
                    lastDeliveryTest: null,
                },
                autonomy: {
                    lastMission: null,
                    missionRuns: [],
                    running: false,
                    autoApprove: false,
                    retryLimit: 1,
                    watchers: [],
                    network: null,
                    watcherIntervalMinutes: 20,
                    watcherMinScore: 6,
                    watcherSaving: false,
                },
                integrations: {
                    config: {
                        github_enabled: false,
                        github_repo: "",
                        github_token_set: false,
                        calendar_enabled: false,
                        calendar_provider: "local",
                        calendar_id: "",
                        email_enabled: false,
                        email_to: "",
                    },
                    summary: null,
                    intelligence: null,
                    calendarEvents: [],
                },
                presence: {
                    snapshot: null,
                    workspace: null,
                    awareness: null,
                    next_actions: null,
                    reminders: [],
                    recent_vision: [],
                    autoSyncEnabled: true,
                    listenersBound: false,
                },
                control: {
                    config: {
                        browser_enabled: true,
                        desktop_enabled: true,
                        execute_on_host: false,
                        browser_name: "default",
                        search_engine: "google",
                        host_control_available: false,
                    },
                    browserUrl: "",
                    browserQuery: "",
                    desktopApp: "dashboard",
                    desktopUrl: "https://github.com/issues",
                    desktopText: "Jarvis desktop control ready",
                    desktopHotkey: "ctrl+l",
                    workflowUrl: "data:text/html,<html><body><h1 id='title'>Jarvis Browser Task</h1><input id='task' value='ready'><button id='go'>Go</button></body></html>",
                    workflowSteps: '[{"action":"wait_for","selector":"#title"},{"action":"extract_text","selector":"#title"}]',
                    templates: [],
                    authTemplates: [],
                    workflowLibrary: [],
                    sessions: [],
                    selectedTemplateName: "",
                    selectedSessionName: "",
                    saveTemplateName: "",
                    saveSession: false,
                    sessionNotes: "",
                    confirmActions: false,
                    githubReview: {
                        repo: "",
                        pullNumber: "",
                        body: "",
                        event: "COMMENT",
                    },
                    gmailCompose: {
                        to: "",
                        subject: "",
                        body: "",
                    },
                    workflowStatus: {},
                    githubPullSummary: null,
                    githubPullSummaryLoading: false,
                    lastResult: null,
                    sessionHealthRunning: false,
                },
                trust: {
                    report: null,
                    receipts: null,
                },
                quantum: {
                    status: null,
                    lastResult: null,
                    statesText: "alpha,beta,gamma",
                    systemA: "core",
                    systemB: "memory",
                    measurementBasis: "computational",
                    exportStart: "",
                    exportEnd: "",
                    stats24: null,
                    stats7d: null,
                    history: [],
                    decipher: null,
                    advancedResult: null,
                    noc: {
                        health_score: null,
                        active_alerts: null,
                        anomaly_count: null,
                        last_snapshot_delta_bias_pct: null,
                        events: null,
                    },
                    notifications: {
                        webhookUrl: "",
                        webhookUrlWarning: "",
                        webhookUrlCritical: "",
                        channel: "generic",
                    },
                    policyPack: "safe",
                    runbookCode: "entanglement_strength_low",
                    annotationNote: "",
                    streamConnected: false,
                    workspace: {
                        incident: null,
                        annotations: [],
                        incidents: [],
                    },
                    rbac: {
                        role: "operator",
                        actions: [],
                    },
                    risk: null,
                    riskTrend: [],
                    slo: null,
                    baselines: null,
                    correlations: null,
                    rootCause: null,
                    postmortem: null,
                    alerts: {
                        active: false,
                        items: [],
                        config: {
                            enabled: true,
                            window_hours: 24,
                            min_measurements: 5,
                            min_entangles: 3,
                            outcome_one_min_pct: 20,
                            outcome_one_max_pct: 80,
                            entanglement_strength_min: 0.9,
                        },
                    },
                },
                lastUpdatedText: "-",
                quantumStream: null,

                get viewTitle() {
                    if (this.currentView === "system") return "Core";
                    if (this.currentView === "quantum") return "Quantum Ops";
                    return "Command";
                },
                get viewDescription() {
                    if (this.currentView === "system") return "Health, links, and runtime status in one place.";
                    if (this.currentView === "quantum") return "Live quantum signals, simulations, and operator readings.";
                    return "Run missions, brief Jarvis, and move through tools without the clutter.";
                },
                get healthBadgeText() {
                    if (!this.system.health) return "unknown";
                    return this.system.health.status === "ok" ? "healthy" : "degraded";
                },
                systemHealthSummary() {
                    if (this.system.error) return this.system.error;
                    if (!this.system.health) return "Waiting for a fresh runtime health payload.";
                    const payload = this.system.health || {};
                    const status = payload.status || "unknown";
                    const signals = Object.keys(payload).filter((key) => key !== "status").length;
                    return `Runtime is ${status} across ${signals} monitored signals.`;
                },
                systemHealthLines() {
                    if (!this.system.health) return ["No runtime signals loaded yet."];
                    return Object.entries(this.system.health)
                        .filter(([key]) => key !== "status")
                        .slice(0, 5)
                        .map(([key, value]) => {
                            if (value && typeof value === "object") {
                                const brief = Object.entries(value).slice(0, 2).map(([subKey, subValue]) => `${subKey}: ${subValue}`).join(" | ");
                                return `${key}: ${brief || "available"}`;
                            }
                            return `${key}: ${value}`;
                        });
                },
                closeOverlays() {
                    this.commandPalette.open = false;
                    this.missionMode = false;
                },
                syncPaletteSelectionIntoView() {
                    this.$nextTick(() => {
                        const root = this.$refs && this.$refs.paletteList;
                        if (!root) return;
                        const active = root.querySelector('[data-palette-active="true"]');
                        if (!active || typeof active.scrollIntoView !== 'function') return;
                        active.scrollIntoView({ block: 'center', behavior: 'smooth' });
                    });
                },
                setMissionHudCollapsed(value) {
                    this.missionHud.sidebandCollapsed = !!value;
                    try { window.localStorage.setItem('jarvis-mission-hud-collapsed', this.missionHud.sidebandCollapsed ? '1' : '0'); } catch (_err) {}
                },
                toggleMissionHudCollapsed() {
                    this.setMissionHudCollapsed(!this.missionHud.sidebandCollapsed);
                },
                updateHoloTilt(event) {
                    const el = event && event.currentTarget;
                    if (!el || typeof el.getBoundingClientRect !== 'function') return;
                    const rect = el.getBoundingClientRect();
                    if (!rect.width || !rect.height) return;
                    const px = ((event.clientX - rect.left) / rect.width) - 0.5;
                    const py = ((event.clientY - rect.top) / rect.height) - 0.5;
                    this.holoTilt = {
                        x: Math.max(-7, Math.min(7, px * 12)),
                        y: Math.max(-7, Math.min(7, py * 12)),
                    };
                },
                resetHoloTilt() {
                    this.holoTilt = { x: 0, y: 0 };
                },
                missionActions() {
                    return [
                        { title: 'Voice', action: () => { this.toggleVoiceListening(); this.pushMissionEvent('Voice channel toggled'); } },
                        { title: 'Execute', action: () => { if (this.memoryCore.nextActions && this.memoryCore.nextActions.actions && this.memoryCore.nextActions.actions.length) this.executeNextAction(this.memoryCore.nextActions.actions[0]); this.pushMissionEvent('Priority executed'); } },
                        { title: 'Capture', action: () => { this.captureScreenVision(); this.pushMissionEvent('Vision frame requested'); } },
                    ];
                },
                moveMissionActionSelection(delta) {
                    const items = this.missionActions();
                    if (!items.length) return;
                    const current = Number.isFinite(this.missionHud.actionIndex) ? this.missionHud.actionIndex : 0;
                    let next = current + delta;
                    if (next < 0) next = items.length - 1;
                    if (next >= items.length) next = 0;
                    this.missionHud.actionIndex = next;
                },
                runMissionActionSelection() {
                    const items = this.missionActions();
                    if (!items.length) return;
                    const idx = Math.max(0, Math.min(this.missionHud.actionIndex || 0, items.length - 1));
                    items[idx].action();
                },
                toggleCommandPalette(forceState = null) {
                    this.commandPalette.open = forceState == null ? !this.commandPalette.open : !!forceState;
                    if (this.commandPalette.open) {
                        this.missionMode = false;
                        this.commandPalette.activeIndex = 0;
                        this.$nextTick(() => {
                            if (this.$refs && this.$refs.paletteInput) this.$refs.paletteInput.focus();
                            this.syncPaletteSelectionIntoView();
                        });
                    }
                },
                movePaletteSelection(delta) {
                    const items = this.paletteActions();
                    if (!items.length) {
                        this.commandPalette.activeIndex = 0;
                        return;
                    }
                    const current = Number.isFinite(this.commandPalette.activeIndex) ? this.commandPalette.activeIndex : 0;
                    let next = current + delta;
                    if (next < 0) next = items.length - 1;
                    if (next >= items.length) next = 0;
                    this.commandPalette.activeIndex = next;
                    this.syncPaletteSelectionIntoView();
                },
                runPaletteSelection() {
                    const items = this.paletteActions();
                    if (!items.length) return;
                    const idx = Math.max(0, Math.min(this.commandPalette.activeIndex || 0, items.length - 1));
                    this.runPaletteAction(items[idx]);
                },
                toggleMissionMode(forceState = null) {
                    this.missionMode = forceState == null ? !this.missionMode : !!forceState;
                    if (this.missionMode) this.commandPalette.open = false;
                    if (this.missionMode && this.currentView === 'agent') this.pushMissionEvent('Mission mode armed');
                },
                pushMissionEvent(text) {
                    const message = String(text || '').trim();
                    if (!message) return;
                    this.missionEvents.unshift({ text: message, at: new Date().toLocaleTimeString() });
                    this.missionEvents = this.missionEvents.slice(0, 6);
                    this.missionFeedPulse = true;
                    setTimeout(() => { this.missionFeedPulse = false; }, 900);
                },
                fuzzyScore(needle, haystack) {
                    const q = String(needle || '').toLowerCase();
                    const s = String(haystack || '').toLowerCase();
                    if (!q) return 1;
                    if (s.includes(q)) return 100 - (s.indexOf(q) * 0.5);
                    let qi = 0;
                    let score = 0;
                    for (let i = 0; i < s.length && qi < q.length; i += 1) {
                        if (s[i] === q[qi]) {
                            score += 3;
                            qi += 1;
                        }
                    }
                    return qi === q.length ? score : -1;
                },
                lastMissionCommand() {
                    if (this.commandPalette.recent && this.commandPalette.recent.length) return this.commandPalette.recent[0];
                    const lastUser = [...(this.agent.messages || [])].reverse().find((m) => m.role === 'user' && m.text);
                    return lastUser ? String(lastUser.text).slice(0, 64) : 'Standby';
                },
                favoritePaletteActions() {
                    const actions = this.paletteActions();
                    const order = this.commandPalette.favorites || [];
                    return order
                        .map((title) => actions.find((item) => item.title === title))
                        .filter(Boolean)
                        .slice(0, 6);
                },
                paletteUsageCount(title) {
                    return Number((this.commandPalette.usage || {})[title] || 0);
                },
                persistPaletteState() {
                    try {
                        window.localStorage.setItem('jarvis-command-palette-recent', JSON.stringify(this.commandPalette.recent || []));
                        window.localStorage.setItem('jarvis-command-palette-favorites', JSON.stringify(this.commandPalette.favorites || []));
                        window.localStorage.setItem('jarvis-command-palette-usage', JSON.stringify(this.commandPalette.usage || {}));
                    } catch (_err) {}
                },
                dropFavoriteAt(targetIndex) {
                    const from = this.commandPalette.dragFavoriteIndex;
                    const favorites = [...(this.commandPalette.favorites || [])];
                    if (from == null || from < 0 || from >= favorites.length || targetIndex < 0 || targetIndex >= favorites.length) return;
                    const [moved] = favorites.splice(from, 1);
                    favorites.splice(targetIndex, 0, moved);
                    this.commandPalette.favorites = favorites;
                    this.commandPalette.dragFavoriteIndex = null;
                    this.persistPaletteState();
                },
                resetPaletteState() {
                    this.commandPalette.recent = [];
                    this.commandPalette.usage = {};
                    this.commandPalette.favorites = ["Open Voice Channel", "Execute Priority", "Open Mission Mode"];
                    this.commandPalette.dragFavoriteIndex = null;
                    this.persistPaletteState();
                },
                mostUsedPaletteActions() {
                    const usage = this.commandPalette.usage || {};
                    return this.paletteActions()
                        .map((item) => ({ item, count: Number(usage[item.title] || 0) }))
                        .filter((entry) => entry.count > 0)
                        .sort((a, b) => b.count - a.count)
                        .slice(0, 4)
                        .map((entry) => entry.item);
                },
                isFavoriteAction(title) {
                    return (this.commandPalette.favorites || []).includes(title);
                },
                toggleFavoriteAction(item) {
                    if (!item || !item.title) return;
                    const current = new Set(this.commandPalette.favorites || []);
                    if (current.has(item.title)) current.delete(item.title);
                    else current.add(item.title);
                    this.commandPalette.favorites = Array.from(current);
                    this.persistPaletteState();
                },
                paletteActions() {
                    const actions = [
                        { title: 'Open Voice Channel', hint: 'voice', group: 'Actions', action: () => this.toggleVoiceListening() },
                        { title: 'Execute Priority', hint: 'mission', group: 'Actions', action: () => { if (this.memoryCore.nextActions && this.memoryCore.nextActions.actions && this.memoryCore.nextActions.actions.length) this.executeNextAction(this.memoryCore.nextActions.actions[0]); } },
                        { title: 'Capture Vision Frame', hint: 'vision', group: 'Actions', action: () => this.captureScreenVision() },
                        { title: 'Set Voice Prime', hint: 'voice preset male', group: 'Actions', action: () => this.applyVoicePreset('prime') },
                        { title: 'Set Voice Ops', hint: 'voice preset deep', group: 'Actions', action: () => this.applyVoicePreset('ops') },
                        { title: 'Set Voice Arc', hint: 'voice preset future', group: 'Actions', action: () => this.applyVoicePreset('arc') },
                        { title: 'Open Mission Mode', hint: 'command', group: 'Views', action: () => this.toggleMissionMode(true) },
                        { title: 'Switch To Core', hint: 'system', group: 'Views', action: () => { this.currentView = 'system'; } },
                        { title: 'Switch To Quantum Ops', hint: 'quantum', group: 'Views', action: () => { this.currentView = 'quantum'; } },
                        { title: 'Refresh Health', hint: 'status', group: 'System', action: () => this.refreshHealth() },
                        { title: 'Refresh Desktop Presence', hint: 'presence', group: 'System', action: () => this.loadDesktopPresence() },
                        { title: 'Sync Desktop Context', hint: 'awareness', group: 'System', action: () => this.captureDesktopPresenceAuto() },
                    ];
                    const recentTitles = new Set(this.commandPalette.recent || []);
                    const enriched = actions.map((item) => ({ ...item, recent: recentTitles.has(item.title) }));
                    const q = String(this.commandPalette.query || '').trim().toLowerCase();
                    let ranked = [];
                    const groupOrder = ['Actions', 'Views', 'System'];
                    if (!q) {
                        ranked = enriched.sort((a, b) => (groupOrder.indexOf(a.group) - groupOrder.indexOf(b.group)) || Number(b.recent) - Number(a.recent) || a.title.localeCompare(b.title));
                    } else {
                        ranked = enriched
                            .map((item) => ({
                                item,
                                score: Math.max(this.fuzzyScore(q, item.title), this.fuzzyScore(q, item.hint), this.fuzzyScore(q, item.group)) + (item.recent ? 40 : 0)
                            }))
                            .filter((entry) => entry.score >= 0)
                            .sort((a, b) => b.score - a.score || Number(b.item.recent) - Number(a.item.recent))
                            .map((entry) => entry.item);
                    }
                    ranked = ranked.map((item, idx) => ({ ...item, globalIndex: idx }));
                    if ((this.commandPalette.activeIndex || 0) >= ranked.length) this.commandPalette.activeIndex = Math.max(ranked.length - 1, 0);
                    return ranked;
                },
                paletteActionGroups() {
                    const iconMap = {
                        Actions: 'fas fa-bolt',
                        Views: 'fas fa-compass',
                        System: 'fas fa-shield-halved',
                    };
                    const groups = [];
                    for (const item of this.paletteActions()) {
                        let group = groups.find((entry) => entry.label === item.group);
                        if (!group) {
                            group = { label: item.group, icon: iconMap[item.group] || 'fas fa-circle', items: [] };
                            groups.push(group);
                        }
                        group.items.push(item);
                    }
                    return groups;
                },
                runPaletteAction(item) {
                    if (!item || typeof item.action !== 'function') return;
                    item.action();
                    this.commandPalette.recent = [item.title, ...(this.commandPalette.recent || []).filter((title) => title !== item.title)].slice(0, 8);
                    this.commandPalette.usage = { ...(this.commandPalette.usage || {}), [item.title]: Number((this.commandPalette.usage || {})[item.title] || 0) + 1 };
                    this.persistPaletteState();
                    this.pushMissionEvent(item.title);
                    this.missionHud.lastCommandPulse = true;
                    setTimeout(() => { this.missionHud.lastCommandPulse = false; }, 680);
                    this.commandPalette.open = false;
                    this.commandPalette.activeIndex = 0;
                },
                get voiceStatusText() {
                    if (!this.voice.supported) return "Voice not supported in this browser";
                    if (this.voice.error) return this.voice.error;
                    if (this.voice.pipelineListening) return this.voice.pipelineStatus;
                    if (this.voice.listening) return "Listening...";
                    if (this.voice.speaking) return "Speaking...";
                    return "Voice idle";
                },

                riskGaugeStyle() {
                    const score = (this.quantum.risk && this.quantum.risk.risk_score != null) ? Number(this.quantum.risk.risk_score) : 0;
                    const width = Math.max(0, Math.min(100, score));
                    let color = "#2ee6d6";
                    if (score >= 70) color = "#ef4444";
                    else if (score >= 35) color = "#f59e0b";
                    return "width:" + width + "%;background:" + color + ";box-shadow:0 0 12px " + color + ";";
                },

                riskTrendText() {
                    const t = Array.isArray(this.quantum.riskTrend) ? this.quantum.riskTrend : [];
                    if (t.length < 2) return "-";
                    const delta = Number(t[t.length - 1]) - Number(t[t.length - 2]);
                    if (delta > 0.2) return "up";
                    if (delta < -0.2) return "down";
                    return "flat";
                },

                severityBadgeStyle(severity) {
                    const s = (severity || "info").toLowerCase();
                    if (s === "critical") return "background: rgba(239,68,68,0.25); border:1px solid rgba(239,68,68,0.7); color:#fecaca;";
                    if (s === "warning") return "background: rgba(245,158,11,0.22); border:1px solid rgba(245,158,11,0.7); color:#fde68a;";
                    return "background: rgba(46,230,214,0.20); border:1px solid rgba(46,230,214,0.7); color:#b8fff9;";
                },

                async init() {
                    try {
                        this.missionHud.sidebandCollapsed = window.localStorage.getItem('jarvis-mission-hud-collapsed') === '1';
                        const paletteRecent = window.localStorage.getItem('jarvis-command-palette-recent');
                        this.commandPalette.recent = paletteRecent ? JSON.parse(paletteRecent) : [];
                        const paletteFavorites = window.localStorage.getItem('jarvis-command-palette-favorites');
                        this.commandPalette.favorites = paletteFavorites ? JSON.parse(paletteFavorites) : this.commandPalette.favorites;
                        const paletteUsage = window.localStorage.getItem('jarvis-command-palette-usage');
                        this.commandPalette.usage = paletteUsage ? JSON.parse(paletteUsage) : (this.commandPalette.usage || {});
                        const workflowStatus = window.localStorage.getItem('jarvis-workflow-status');
                        this.control.workflowStatus = workflowStatus ? JSON.parse(workflowStatus) : (this.control.workflowStatus || {});
                    } catch (_err) {}
                    await Promise.all([this.refreshHealth(), this.loadTools(), this.loadProfile(), this.loadMemoryOverview(), this.loadWorkspaces(), this.loadMemoryGraph(), this.loadReminders(), this.loadSavedArchivedFilters(), this.loadGoalHistory(), this.loadSchedules(), this.loadBriefingAutomations(), this.loadBriefingDeliveryConfig(), this.loadControlConfig(), this.loadBrowserTemplates(), this.loadBrowserAuthTemplates(), this.loadBrowserWorkflowLibrary(), this.loadTrustReport(), this.loadTrustReceipts(), this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNotificationConfig(), this.loadQuantumNoc(), this.loadQuantumRiskScore(), this.loadQuantumSlo(), this.loadQuantumRbac(), this.loadQuantumBaselines(), this.loadQuantumCorrelations(), this.loadQuantumRootCause(), this.loadLlmConfig(), this.loadVisionConfig(), this.loadWakeWordConfig(), this.loadLocalVoiceConfig(), this.loadWatchers(), this.loadWatcherNetwork(), this.loadMissionRuns(), this.loadIntegrationsSummary(), this.loadIntegrationsIntelligence(), this.loadCalendarEvents(), this.loadDesktopPresence(), this.loadDesktopAwareness()]);
                    await this.loadIncidentWorkspace();
                    this.initVoice();
                    this.bindDesktopAwarenessHooks();
                    this.startQuantumStream();
                    await this.maybeAutoBriefing();
                    if (this.currentView === 'agent') this.pushMissionEvent('Mission mode online');
                    setInterval(() => { this.refreshHealth(); }, 10000);
                    setInterval(() => {
                        if (this.currentView === "quantum") {
                            this.loadQuantumNoc();
                            this.loadQuantumAlerts();
                            this.loadQuantumAnalytics();
                        }
                    }, 8000);
                    setInterval(() => { this.pollReminderVoiceFeed(); }, 15000);
                    setInterval(() => { this.loadWatchers(); }, 30000);
                    setInterval(() => { this.loadMissionRuns(); }, 30000);
                    setInterval(() => { this.loadDesktopPresence(); }, 20000);
                    setInterval(() => { this.captureDesktopPresenceAuto(); }, 30000);
                },

                async refreshHealth() {
                    try {
                        const res = await fetch("/health", { headers: { "Accept": "application/json" } });
                        const data = await res.json();
                        this.system.health = data;
                        this.system.error = null;
                        this.lastUpdatedText = new Date().toLocaleTimeString();
                    } catch (err) {
                        this.system.error = (err && err.message) ? err.message : "Failed to fetch /health";
                    }
                },

                async loadTools() {
                    try {
                        const res = await fetch("/agent/tools", { headers: { "Accept": "application/json" } });
                        const data = await res.json();
                        this.agent.tools = Array.isArray(data) ? data : [];
                    } catch (err) {
                        this.agent.tools = [];
                    }
                },

                async loadLlmConfig() {
                    this.agent.configLoading = true;
                    try {
                        const res = await fetch("/agent/llm/config", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.agent.llmMode = data.mode || "quality";
                        this.agent.llmModels = Array.isArray(data.available_models) ? data.available_models : [];
                        if (this.agent.llmModels.length === 0 && data.model) this.agent.llmModels = [data.model];
                        this.agent.llmModel = data.model || (this.agent.llmModels[0] || "");
                        this.agent.llmTemperature = (data.temperature != null) ? data.temperature : 0.2;
                    } catch (_err) {
                        this.agent.llmModels = this.agent.llmModels || [];
                    } finally {
                        this.agent.configLoading = false;
                    }
                },

                async saveLlmConfig() {
                    try {
                        const res = await fetch("/agent/llm/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ mode: this.agent.llmMode || "quality", model: this.agent.llmModel || null }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.agent.llmMode = data.mode || this.agent.llmMode;
                        this.agent.llmTemperature = (data.temperature != null) ? data.temperature : this.agent.llmTemperature;
                        if (Array.isArray(data.available_models) && data.available_models.length) {
                            this.agent.llmModels = data.available_models;
                        }
                        if (data.model) this.agent.llmModel = data.model;
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save LLM config";
                    }
                },

                async loadVisionConfig() {
                    this.vision.configLoading = true;
                    try {
                        const res = await fetch("/agent/vision/config", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.vision.provider = data.provider || "auto";
                        this.vision.models = Array.isArray(data.available_models) ? data.available_models : [];
                        if (this.vision.models.length === 0 && data.model) this.vision.models = [data.model];
                        this.vision.model = data.model || (this.vision.models[0] || "");
                        this.vision.ocrAvailable = !!data.ocr_available;
                        this.vision.openaiConfigured = !!data.openai_configured;
                    } catch (_err) {
                        this.vision.models = this.vision.models || [];
                    } finally {
                        this.vision.configLoading = false;
                    }
                },

                async saveVisionConfig() {
                    try {
                        const res = await fetch("/agent/vision/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ provider: this.vision.provider || "auto", model: this.vision.model || null }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.vision.provider = data.provider || this.vision.provider;
                        this.vision.ocrAvailable = !!data.ocr_available;
                        this.vision.openaiConfigured = !!data.openai_configured;
                        if (Array.isArray(data.available_models) && data.available_models.length) {
                            this.vision.models = data.available_models;
                        }
                        if (data.model) this.vision.model = data.model;
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save vision config";
                    }
                },

                visionBoxStyle(box) {
                    if (!this.vision.last || !this.vision.last.details) return "";
                    const width = Number(this.vision.last.details.width || 1);
                    const height = Number(this.vision.last.details.height || 1);
                    const left = Math.max(0, Math.min(100, (Number(box.left || 0) / width) * 100));
                    const top = Math.max(0, Math.min(100, (Number(box.top || 0) / height) * 100));
                    const boxWidth = Math.max(0.5, Math.min(100, (Number(box.width || 0) / width) * 100));
                    const boxHeight = Math.max(0.5, Math.min(100, (Number(box.height || 0) / height) * 100));
                    return `left:${left}%;top:${top}%;width:${boxWidth}%;height:${boxHeight}%;`;
                },

                laneBadgeStyle(lane) {
                    const value = String(lane || "personal").toLowerCase();
                    if (value === "critical") return "background: rgba(239,68,68,0.18); border:1px solid rgba(239,68,68,0.45); color:#fecaca;";
                    if (value === "project") return "background: rgba(59,130,246,0.18); border:1px solid rgba(59,130,246,0.45); color:#bfdbfe;";
                    return "background: rgba(46,230,214,0.15); border:1px solid rgba(46,230,214,0.35); color:#b8fff9;";
                },

                async loadProfile() {
                    try {
                        const res = await fetch("/agent/profile", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error("Failed to load profile");
                        const data = await res.json();
                        this.profile.current = data.profile || null;
                        this.profile.dangerous = !!data.dangerous_full_access;
                    } catch (err) {
                        this.profile.current = null;
                    }
                },

                async loadMemoryOverview() {
                    this.memoryCore.loading = true;
                    try {
                        const params = new URLSearchParams({ limit_per_group: "6" });
                        if (this.memoryCore.windowHours && this.memoryCore.windowHours !== "all") {
                            params.set("since_hours", this.memoryCore.windowHours);
                        }
                        const res = await fetch("/agent/memory/overview?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.memoryCore.overview = await res.json();
                        this.memoryCore.selectedIds = [];
                        if (this.memoryCore.showArchived) await this.loadArchivedMemories();
                        this.memoryCore.error = null;
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to load memory overview";
                    } finally {
                        this.memoryCore.loading = false;
                    }
                },

                async loadWorkspaces() {
                    try {
                        const [listRes, activeRes] = await Promise.all([
                            fetch("/agent/memory/workspaces?limit=20", { headers: { "Accept": "application/json" } }),
                            fetch("/agent/memory/workspaces/active", { headers: { "Accept": "application/json" } }),
                        ]);
                        if (!listRes.ok) throw new Error(await listRes.text());
                        const listData = await listRes.json();
                        this.memoryCore.workspaces = Array.isArray(listData.items) ? listData.items : [];
                        if (activeRes.ok) {
                            const activeData = await activeRes.json();
                            this.memoryCore.activeWorkspace = activeData.workspace || null;
                        }
                        await Promise.all([this.loadWorkspacePolicy(), this.loadBrowserSessions(), this.loadNextBestActions(), this.loadDesktopPresence(), this.loadDesktopAwareness(), this.loadIntegrationsIntelligence()]);
                    } catch (_err) {
                        this.memoryCore.workspaces = [];
                        this.memoryCore.activeWorkspace = null;
                        this.memoryCore.workspacePolicy = null;
                        this.control.sessions = [];
                        this.memoryCore.nextActions = null;
                    }
                },

                async saveWorkspace() {
                    const name = String(this.memoryCore.workspaceName || "").trim();
                    if (!name) return;
                    try {
                        const memoryIds = (((this.memoryCore.overview || {}).projects || []).slice(0, 6)).map((item) => Number(item.id)).filter((id) => id > 0);
                        const res = await fetch("/agent/memory/workspaces", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                name: name,
                                focus: this.memoryCore.workspaceFocus || "",
                                description: this.memoryCore.workspaceDescription || "",
                                color: "#2ee6d6",
                                status: "active",
                                memory_ids: memoryIds,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.activeWorkspace = data.workspace || null;
                        this.memoryCore.workspaceName = "";
                        this.memoryCore.workspaceFocus = "";
                        this.memoryCore.workspaceDescription = "";
                        await this.loadWorkspaces();
                        await this.activateWorkspace(this.memoryCore.activeWorkspace);
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to save workspace";
                    }
                },

                async activateWorkspace(workspace) {
                    const workspaceId = workspace && workspace.id ? Number(workspace.id) : null;
                    try {
                        const res = await fetch("/agent/memory/workspaces/active", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ workspace_id: workspaceId || null }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.activeWorkspace = data.workspace || null;
                        await this.loadWorkspaces();
                        await this.loadMemoryGraph();
                        await this.loadReminders();
                        await this.loadWorkspacePolicy();
                        await this.loadBrowserSessions();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to activate workspace";
                    }
                },

                async loadMemoryGraph() {
                    try {
                        const params = new URLSearchParams({ limit: "60" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/memory/graph?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.memoryCore.graph = await res.json();
                    } catch (_err) {
                        this.memoryCore.graph = null;
                    }
                },

                graphEdgeSummary() {
                    const edges = ((this.memoryCore.graph || {}).edges || []);
                    const counts = {};
                    edges.forEach((edge) => {
                        const key = String(edge.type || "linked");
                        counts[key] = (counts[key] || 0) + 1;
                    });
                    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 4);
                },

                async loadReminders() {
                    try {
                        const params = new URLSearchParams({ limit: "12", status: "open" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/memory/reminders?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.reminders = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.memoryCore.reminders = [];
                    }
                },

                async generateReminders() {
                    try {
                        const params = new URLSearchParams({ limit: "4" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/memory/reminders/generate?" + params.toString(), {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        await this.loadReminders();
                        await this.loadNextBestActions();
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Proactive reminders generated for your current workspace and memory priorities.",
                            tool: data,
                        });
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to generate reminders";
                    }
                },

                async setReminderStatus(reminder, status) {
                    if (!reminder || !reminder.id) return;
                    try {
                        const res = await fetch("/agent/memory/reminders/" + reminder.id, {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ status: status }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadReminders();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to update reminder";
                    }
                },

                async loadArchivedMemories() {
                    this.memoryCore.archivedLoading = true;
                    try {
                        const params = new URLSearchParams({ limit: "40", archived: "true" });
                        if (this.memoryCore.windowHours && this.memoryCore.windowHours !== "all") {
                            params.set("since_hours", this.memoryCore.windowHours);
                        }
                        if (this.memoryCore.archivedQuery) params.set("query", this.memoryCore.archivedQuery);
                        if (this.memoryCore.archivedTag) params.set("tag", this.memoryCore.archivedTag);
                        if (this.memoryCore.archivedType) params.set("memory_type", this.memoryCore.archivedType);
                        const res = await fetch("/agent/memory/long-term?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.archivedItems = Array.isArray(data.items) ? data.items : [];
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to load archived memories";
                    } finally {
                        this.memoryCore.archivedLoading = false;
                    }
                },

                async loadSavedArchivedFilters() {
                    try {
                        const res = await fetch("/agent/memory/filters/saved", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.savedFilters = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.memoryCore.savedFilters = [];
                    }
                },

                async toggleArchivedView() {
                    this.memoryCore.showArchived = !this.memoryCore.showArchived;
                    if (this.memoryCore.showArchived) {
                        await this.loadArchivedMemories();
                        await this.loadSavedArchivedFilters();
                    }
                },

                memoryBriefingText() {
                    const overview = this.memoryCore.overview || {};
                    const pinned = Array.isArray(overview.pinned) ? overview.pinned : [];
                    const projects = Array.isArray(overview.projects) ? overview.projects : [];
                    const profile = Array.isArray(overview.profile) ? overview.profile : [];
                    const parts = ["Jarvis memory briefing."];
                    if (pinned.length) {
                        parts.push("Priority memories:");
                        pinned.slice(0, 3).forEach((m) => parts.push(String(m.content || "").trim()));
                    }
                    if (projects.length) {
                        parts.push("Active projects:");
                        projects.slice(0, 2).forEach((m) => parts.push(String(m.content || "").trim()));
                    }
                    if (!pinned.length && !projects.length && profile.length) {
                        parts.push("Profile memory:");
                        profile.slice(0, 2).forEach((m) => parts.push(String(m.content || "").trim()));
                    }
                    return parts.join(" ");
                },

                currentBriefingPeriod() {
                    const hour = new Date().getHours();
                    if (hour >= 5 && hour < 12) return "morning";
                    if (hour >= 17 && hour < 23) return "evening";
                    return "";
                },

                async loadMemoryBriefing(period) {
                    const p = period || this.currentBriefingPeriod() || "morning";
                    const res = await fetch(`/agent/memory/briefing?period=${encodeURIComponent(p)}`, { headers: { "Accept": "application/json" } });
                    if (!res.ok) throw new Error(await res.text());
                    const data = await res.json();
                    this.memoryCore.briefing = data;
                    return data;
                },

                async maybeAutoBriefing() {
                    const period = this.currentBriefingPeriod();
                    if (!period) return;
                    const today = new Date().toISOString().slice(0, 10);
                    const key = `jarvis-memory-briefing:${period}:${today}`;
                    try {
                        if (window.localStorage.getItem(key) === "played") return;
                        const briefing = await this.loadMemoryBriefing(period);
                        if (briefing && briefing.text) {
                            this.speakText(briefing.text, true);
                            this.agent.messages.push({ role: "assistant", text: briefing.text, tool: briefing });
                            this.$nextTick(() => this._scrollMessages());
                            window.localStorage.setItem(key, "played");
                        }
                    } catch (_err) {}
                },

                async playVoiceBriefing(periodOverride = null) {
                    try {
                        const briefing = await this.loadMemoryBriefing(periodOverride || this.currentBriefingPeriod() || "morning");
                        this.speakText(briefing.text || this.memoryBriefingText(), true);
                    } catch (_err) {
                        this.speakText(this.memoryBriefingText(), true);
                    }
                },

                isMemorySelected(id) {
                    return this.memoryCore.selectedIds.includes(Number(id));
                },

                toggleMemorySelection(id) {
                    const value = Number(id);
                    if (this.memoryCore.selectedIds.includes(value)) {
                        this.memoryCore.selectedIds = this.memoryCore.selectedIds.filter((item) => item !== value);
                    } else {
                        this.memoryCore.selectedIds = [...this.memoryCore.selectedIds, value];
                    }
                },

                clearMemorySelection() {
                    this.memoryCore.selectedIds = [];
                },

                dragPinnedStart(id) {
                    this.memoryCore.draggedPinnedId = Number(id);
                },

                async dropPinnedBefore(targetId) {
                    const draggedId = Number(this.memoryCore.draggedPinnedId || 0);
                    const target = Number(targetId || 0);
                    if (!draggedId || !target || draggedId === target) return;
                    const pinned = ((this.memoryCore.overview && this.memoryCore.overview.pinned) ? this.memoryCore.overview.pinned : []).slice();
                    const orderedIds = pinned.map((item) => Number(item.id)).filter((id) => id > 0);
                    const fromIndex = orderedIds.indexOf(draggedId);
                    const toIndex = orderedIds.indexOf(target);
                    if (fromIndex < 0 || toIndex < 0) return;
                    orderedIds.splice(fromIndex, 1);
                    orderedIds.splice(toIndex, 0, draggedId);
                    try {
                        const res = await fetch("/agent/memory/pinned/reorder", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ ids: orderedIds }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to reorder pinned memories";
                    } finally {
                        this.memoryCore.draggedPinnedId = null;
                    }
                },

                dragProjectStart(id) {
                    this.memoryCore.draggedProjectId = Number(id);
                },

                async dropProjectBefore(targetId) {
                    const draggedId = Number(this.memoryCore.draggedProjectId || 0);
                    const target = Number(targetId || 0);
                    if (!draggedId || !target || draggedId === target) return;
                    const projects = ((this.memoryCore.overview && this.memoryCore.overview.projects) ? this.memoryCore.overview.projects : []).slice();
                    const orderedIds = projects.map((item) => Number(item.id)).filter((id) => id > 0);
                    const fromIndex = orderedIds.indexOf(draggedId);
                    const toIndex = orderedIds.indexOf(target);
                    if (fromIndex < 0 || toIndex < 0) return;
                    orderedIds.splice(fromIndex, 1);
                    orderedIds.splice(toIndex, 0, draggedId);
                    try {
                        const res = await fetch("/agent/memory/projects/reorder", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ ids: orderedIds }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to reorder project memories";
                    } finally {
                        this.memoryCore.draggedProjectId = null;
                    }
                },

                async saveCurrentArchivedFilter() {
                    const name = String(this.memoryCore.savedFilterName || "").trim();
                    if (!name) return;
                    try {
                        const res = await fetch("/agent/memory/filters/saved", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                name,
                                query: this.memoryCore.archivedQuery || "",
                                tag: this.memoryCore.archivedTag || "",
                                memory_type: this.memoryCore.archivedType || "",
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.savedFilters = Array.isArray(data.items) ? data.items : [];
                        this.memoryCore.savedFilterName = "";
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to save archived filter";
                    }
                },

                applySavedArchivedFilter(filter) {
                    if (!filter) return;
                    this.memoryCore.archivedQuery = filter.query || "";
                    this.memoryCore.archivedTag = filter.tag || "";
                    this.memoryCore.archivedType = filter.memory_type || "";
                    this.loadArchivedMemories();
                },

                async deleteSavedArchivedFilter(name) {
                    if (!name) return;
                    try {
                        const res = await fetch("/agent/memory/filters/saved?name=" + encodeURIComponent(name), {
                            method: "DELETE",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.savedFilters = Array.isArray(data.items) ? data.items : [];
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to delete saved archived filter";
                    }
                },

                async applyBulkMemoryAction(action) {
                    const ids = (this.memoryCore.selectedIds || []).map((id) => Number(id)).filter((id) => id > 0);
                    if (ids.length === 0) return;
                    try {
                        const res = await fetch("/agent/memory/actions/bulk", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ ids, action }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to apply bulk action";
                    }
                },

                async togglePinMemory(item) {
                    try {
                        const pinned = !(item && item.pinned);
                        const res = await fetch(`/agent/memory/${item.id}/pin?pinned=${pinned ? "true" : "false"}`, {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to update pin";
                    }
                },

                addEditTag() {
                    const raw = String(this.memoryCore.editModal.tagInput || "").trim();
                    if (!raw) return;
                    const normalized = raw.replace(/\s+/g, " ");
                    if (!this.memoryCore.editModal.tags.includes(normalized)) {
                        this.memoryCore.editModal.tags = [...this.memoryCore.editModal.tags, normalized].slice(0, 12);
                    }
                    this.memoryCore.editModal.tagInput = "";
                },

                removeEditTag(tag) {
                    this.memoryCore.editModal.tags = (this.memoryCore.editModal.tags || []).filter((t) => t !== tag);
                },

                openEditMemory(item) {
                    this.memoryCore.editModal = {
                        open: true,
                        id: Number(item.id),
                        content: String(item.content || ""),
                        subject: String(item.subject || ""),
                        memoryType: String(item.memory_type || "general"),
                        importance: Number(item.importance || 3),
                        pinned: !!item.pinned,
                        lane: String(item.lane || (String(item.memory_type || "") === "project" ? "project" : "personal")),
                        tags: Array.isArray(item.tags) ? item.tags.slice(0, 12) : [],
                        tagInput: "",
                    };
                },

                closeEditMemory() {
                    this.memoryCore.editModal.open = false;
                },

                async saveMemoryEdit() {
                    const modal = this.memoryCore.editModal;
                    const trimmed = String(modal.content || "").trim();
                    if (!trimmed || !modal.id) return;
                    try {
                        const res = await fetch(`/agent/memory/${modal.id}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                content: trimmed,
                                subject: String(modal.subject || "").trim() || null,
                                memory_type: modal.memoryType || "general",
                                importance: Number(modal.importance || 3),
                                pinned: !!modal.pinned,
                                lane: modal.lane || "personal",
                                tags: Array.isArray(modal.tags) ? modal.tags : [],
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.closeEditMemory();
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to edit memory";
                    }
                },

                async archiveMemory(item) {
                    if (!item || !window.confirm("Archive this memory?")) return;
                    try {
                        const res = await fetch(`/agent/memory/${item.id}`, {
                            method: "DELETE",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to archive memory";
                    }
                },

                async restoreMemory(item) {
                    if (!item) return;
                    try {
                        const res = await fetch(`/agent/memory/${item.id}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ archived: false }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadMemoryOverview();
                        await this.loadArchivedMemories();
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to restore memory";
                    }
                },

                async consolidateMemory() {
                    this.memoryCore.consolidating = true;
                    try {
                        const res = await fetch("/agent/memory/consolidate?limit=300", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        await this.loadMemoryOverview();
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Memory consolidation complete. Merged " + String(data.merged || 0) + " duplicate records.",
                            tool: data,
                        });
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to consolidate memory";
                    } finally {
                        this.memoryCore.consolidating = false;
                    }
                },

                async setProfile(profileName) {
                    try {
                        const res = await fetch("/agent/profile", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ profile: profileName }),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        const data = await res.json();
                        this.profile.current = data.profile || null;
                        this.profile.dangerous = !!data.dangerous_full_access;
                        this.agent.messages.push({ role: "assistant", text: "Profile set to " + data.profile + "." });
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to set profile";
                    }
                },

                async runSelfTest() {
                    try {
                        const res = await fetch("/agent/self_test", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        this.profile.selfTest = await res.json();
                    } catch (err) {
                        this.profile.selfTest = { ok: false, error: (err && err.message) ? err.message : "Self-test failed" };
                    }
                },

                async runGoal() {
                    const goal = (this.goals.input || "").trim();
                    if (!goal) return;
                    this.goals.running = true;
                    try {
                        const res = await fetch("/agent/goals/run", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                goal: goal,
                                session_id: this.agent.sessionId,
                                auto_approve: !!this.goals.autoApprove,
                            }),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        const data = await res.json();
                        this.goals.lastRun = data;
                        this.agent.sessionId = data.session_id || this.agent.sessionId;
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Goal run " + data.status + " (run #" + data.run_id + ").",
                            tool: { plan: data.plan, steps: data.steps, result: data.result },
                        });
                        this.$nextTick(() => this._scrollMessages());
                        await this.loadGoalHistory();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Goal run failed";
                    } finally {
                        this.goals.running = false;
                    }
                },

                async loadGoalHistory() {
                    try {
                        const params = new URLSearchParams();
                        params.set("limit", "12");
                        if (this.agent.sessionId) params.set("session_id", this.agent.sessionId);
                        const res = await fetch("/agent/goals/history?" + params.toString(), {
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error("Failed to load goal history");
                        const data = await res.json();
                        this.goals.history = Array.isArray(data.runs) ? data.runs : [];
                    } catch (err) {
                        this.goals.history = [];
                    }
                },

                async loadSchedules() {
                    try {
                        const res = await fetch("/agent/goals/schedule?limit=100", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error("Failed to load schedules");
                        const data = await res.json();
                        this.schedules.items = Array.isArray(data.schedules) ? data.schedules : [];
                    } catch (_err) {
                        this.schedules.items = [];
                    }
                },

                async loadBriefingAutomations() {
                    try {
                        const res = await fetch("/agent/memory/briefings/automations", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.schedules.briefingJobs = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.schedules.briefingJobs = [];
                    }
                },

                async loadBriefingDeliveryConfig() {
                    try {
                        const res = await fetch("/agent/memory/briefings/delivery", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.schedules.briefingDelivery = await res.json();
                    } catch (_err) {}
                },

                async loadControlConfig() {
                    try {
                        const res = await fetch("/agent/control/config", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.config = await res.json();
                    } catch (_err) {}
                },

                async loadBrowserSessions() {
                    try {
                        const params = new URLSearchParams({ limit: "24" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/control/browser/sessions?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.sessions = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.control.sessions = [];
                    }
                },

                async checkBrowserSessionHealth(sessionName = null) {
                    this.control.sessionHealthRunning = true;
                    try {
                        const res = await fetch("/agent/control/browser/sessions/health", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                session_name: sessionName || this.control.selectedSessionName || null,
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                limit: sessionName ? 1 : 12,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        const items = Array.isArray(data.items) ? data.items : [];
                        if (sessionName) {
                            const fresh = items[0] || null;
                            if (fresh) {
                                const existing = Array.isArray(this.control.sessions) ? this.control.sessions.slice() : [];
                                const idx = existing.findIndex((item) => item.name === fresh.name);
                                if (idx >= 0) existing.splice(idx, 1, fresh);
                                else existing.unshift(fresh);
                                this.control.sessions = existing;
                            }
                        } else {
                            this.control.sessions = items;
                        }
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to check session health";
                    } finally {
                        this.control.sessionHealthRunning = false;
                    }
                },

                async loadWorkspacePolicy() {
                    try {
                        const url = (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id)
                            ? ("/agent/memory/workspaces/" + this.memoryCore.activeWorkspace.id + "/policy")
                            : "/agent/memory/workspaces/policy/current";
                        const res = await fetch(url, { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.workspacePolicy = data.policy || null;
                    } catch (_err) {
                        this.memoryCore.workspacePolicy = null;
                    }
                },

                async saveWorkspacePolicy() {
                    if (!(this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) || !this.memoryCore.workspacePolicy) {
                        this.memoryCore.error = "Activate a workspace before editing its permissions.";
                        return;
                    }
                    try {
                        const res = await fetch("/agent/memory/workspaces/" + this.memoryCore.activeWorkspace.id + "/policy", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify(this.memoryCore.workspacePolicy),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.memoryCore.workspacePolicy = data.policy || null;
                        this.memoryCore.error = null;
                    } catch (err) {
                        this.memoryCore.error = (err && err.message) ? err.message : "Failed to save workspace policy";
                    }
                },

                async loadNextBestActions() {
                    try {
                        const params = new URLSearchParams({ limit: "6" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/next-action?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.memoryCore.nextActions = await res.json();
                    } catch (_err) {
                        this.memoryCore.nextActions = null;
                    }
                },

                async loadBrowserTemplates() {
                    try {
                        const res = await fetch("/agent/control/browser/templates", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.templates = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.control.templates = [];
                    }
                },

                async loadBrowserAuthTemplates() {
                    try {
                        const res = await fetch("/agent/control/browser/templates/auth", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.authTemplates = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.control.authTemplates = [];
                    }
                },

                async loadBrowserWorkflowLibrary() {
                    try {
                        const res = await fetch("/agent/control/browser/templates/library", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.workflowLibrary = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.control.workflowLibrary = [];
                    }
                },

                applyBrowserTemplateByName() {
                    const name = String(this.control.selectedTemplateName || "").trim();
                    if (!name) return;
                    const item = this.selectedWorkflowTemplate();
                    if (!item) return;
                    this.control.workflowUrl = item.start_url || "";
                    this.control.workflowSteps = JSON.stringify(item.steps || [], null, 2);
                    if (item.auth_template) {
                        this.control.saveSession = true;
                        this.control.selectedSessionName = item.recommended_session_name || this.control.selectedSessionName || "";
                        this.control.sessionNotes = item.description || this.control.sessionNotes || "";
                    }
                    this.prefillWorkflowFormsFromTemplate(item);
                },

                selectedWorkflowTemplate() {
                    const name = String(this.control.selectedTemplateName || '').trim();
                    if (!name) return null;
                    const pools = [this.control.workflowLibrary || [], this.control.authTemplates || [], this.control.templates || []];
                    for (const pool of pools) {
                        const match = (pool || []).find((item) => String(item.name || '') === name);
                        if (match) return match;
                    }
                    return null;
                },

                selectedWorkflowProvider() {
                    const item = this.selectedWorkflowTemplate();
                    return item ? String(item.provider || '').toLowerCase() : '';
                },

                prefillWorkflowFormsFromTemplate(item = null) {
                    const tpl = item || this.selectedWorkflowTemplate();
                    const provider = tpl ? String(tpl.provider || '').toLowerCase() : '';
                    if (provider === 'github') {
                        this.control.githubPullSummary = null;
                        this.control.githubReview.repo = this.integrations.config.github_repo || this.control.githubReview.repo || '';
                        if (String(tpl.name || '').toLowerCase().includes('pr')) {
                            this.control.githubReview.body = this.control.githubReview.body || 'Jarvis review pass complete. See notes below.';
                            this.control.githubReview.event = this.control.githubReview.event || 'COMMENT';
                        }
                    }
                    if (provider === 'gmail') {
                        this.control.gmailCompose.to = this.integrations.config.email_to || this.control.gmailCompose.to || '';
                        if (!String(this.control.gmailCompose.subject || '').trim()) {
                            this.control.gmailCompose.subject = String(tpl && tpl.name ? tpl.name : 'Jarvis Gmail action');
                        }
                        if (!String(this.control.gmailCompose.body || '').trim()) {
                            this.control.gmailCompose.body = 'Jarvis draft ready for review.';
                        }
                    }
                },

                rememberWorkflowOutcome(name, result) {
                    const key = String(name || '').trim();
                    if (!key) return;
                    const previous = (this.control.workflowStatus || {})[key] || {};
                    const ok = !!(result && result.ok);
                    const next = Object.assign({}, this.control.workflowStatus || {});
                    next[key] = {
                        ok,
                        preview: !!(result && result.preview),
                        at: new Date().toISOString(),
                        successCount: Number(previous.successCount || 0) + (ok ? 1 : 0),
                        failureCount: Number(previous.failureCount || 0) + (ok ? 0 : 1),
                    };
                    this.control.workflowStatus = next;
                    try {
                        window.localStorage.setItem('jarvis-workflow-status', JSON.stringify(this.control.workflowStatus));
                    } catch (_err) {}
                },

                workflowSessionBadge(tpl) {
                    const template = tpl || {};
                    const provider = String(template.provider || '').toLowerCase();
                    const recommended = String(template.recommended_session_name || '').toLowerCase();
                    const sessions = Array.isArray(this.control.sessions) ? this.control.sessions : [];
                    const match = sessions.find((session) => {
                        const name = String(session.name || '').toLowerCase();
                        const sessionProvider = String(session.provider || '').toLowerCase();
                        const templateName = String(session.template_name || '').toLowerCase();
                        return (recommended && name === recommended) || (provider && sessionProvider === provider) || (templateName && templateName === String(template.name || '').toLowerCase());
                    });
                    if (!match) return { label: 'no session', tone: 'unknown' };
                    const status = String(match.health_status || 'unknown').toLowerCase();
                    if (status === 'healthy') return { label: 'session healthy', tone: 'ok' };
                    if (status === 'warning') return { label: 'session warn', tone: 'preview' };
                    if (status === 'error') return { label: 'session issue', tone: 'error' };
                    return { label: 'session ' + status, tone: 'unknown' };
                },

                workflowSuccessBadge(tpl) {
                    const key = String((tpl && tpl.name) || '').trim();
                    const status = (this.control.workflowStatus || {})[key] || null;
                    if (!status) return { label: 'standby', tone: 'unknown' };
                    if (status.ok && status.preview) return { label: 'preview ok', tone: 'preview' };
                    if (status.ok) return { label: 'success', tone: 'ok' };
                    return { label: 'needs check', tone: 'error' };
                },

                workflowCountBadge(tpl) {
                    const key = String((tpl && tpl.name) || '').trim();
                    const status = (this.control.workflowStatus || {})[key] || null;
                    const successCount = Number(status && status.successCount || 0);
                    const failureCount = Number(status && status.failureCount || 0);
                    if (!successCount && !failureCount) return { label: '0/0', tone: 'unknown' };
                    return { label: `${successCount}/${failureCount}`, tone: failureCount ? 'error' : 'ok' };
                },

                async loadGitHubPullSummary() {
                    const pullNumber = Number(this.control.githubReview.pullNumber || 0);
                    if (!pullNumber) {
                        this.agent.error = 'GitHub PR number is required';
                        return;
                    }
                    this.control.githubPullSummaryLoading = true;
                    try {
                        const res = await fetch('/agent/integrations/github/pulls/summary', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                            body: JSON.stringify({
                                repo: this.control.githubReview.repo || null,
                                pull_number: pullNumber,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.githubPullSummary = await res.json();
                    } catch (err) {
                        this.control.githubPullSummary = null;
                        this.agent.error = (err && err.message) ? err.message : 'Failed to load GitHub PR brief';
                    } finally {
                        this.control.githubPullSummaryLoading = false;
                    }
                },

                async loadWakeWordConfig() {
                    try {
                        const res = await fetch("/agent/voice/wake", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.voice.pipelineAvailable = !!data.available;
                        this.voice.pipelineEnabled = !!data.enabled;
                        this.voice.wakeEnabled = !!data.enabled;
                        this.voice.wakeWord = data.wake_word || this.voice.wakeWord;
                        this.voice.threshold = data.threshold != null ? data.threshold : this.voice.threshold;
                        this.voice.chunkMs = data.chunk_ms || this.voice.chunkMs;
                        this.voice.pipelineStatus = this.voice.pipelineAvailable ? "wake pipeline ready" : "wake pipeline unavailable";
                    } catch (_err) {
                        this.voice.pipelineAvailable = false;
                        this.voice.pipelineStatus = "wake pipeline unavailable";
                    }
                },

                async loadLocalVoiceConfig() {
                    try {
                        const [cfgRes, presetRes] = await Promise.all([
                            fetch("/agent/voice/local", { headers: { "Accept": "application/json" } }),
                            fetch("/agent/voice/local/presets", { headers: { "Accept": "application/json" } }),
                        ]);
                        if (!cfgRes.ok) throw new Error(await cfgRes.text());
                        const data = await cfgRes.json();
                        this.voice.localAvailable = !!(data.stt_available && data.tts_available);
                        this.voice.localVoiceEnabled = !!data.enabled;
                        this.voice.sttModel = data.stt_model || "base";
                        this.voice.ttsVoice = data.tts_voice || "mb-en1";
                        this.voice.ttsRate = data.tts_rate || 145;
                        this.voice.ttsPitch = data.tts_pitch || 38;
                        if (presetRes.ok) {
                            const presetData = await presetRes.json();
                            this.voice.profiles = Array.isArray(presetData.items)
                                ? presetData.items.map((item) => ({ id: item.id, label: item.name, voice: item.voice, rate: item.rate, pitch: item.pitch, style: item.style, hint: item.description || '' }))
                                : this.voice.profiles;
                        }
                        this.voice.activePreset = this.detectVoicePreset();
                    } catch (_err) {
                        this.voice.localAvailable = false;
                    }
                },

                async saveLocalVoiceConfig() {
                    try {
                        const active = (this.voice.profiles || []).find((item) => item.id === this.voice.activePreset);
                        const res = await fetch("/agent/voice/local", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                enabled: !!this.voice.localVoiceEnabled,
                                stt_model: this.voice.sttModel || "base",
                                tts_voice: this.voice.ttsVoice || "mb-en1",
                                tts_rate: this.voice.ttsRate,
                                tts_pitch: this.voice.ttsPitch,
                                tts_style: active && active.style ? active.style : "assistant",
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.voice.localAvailable = !!(data.stt_available && data.tts_available);
                        this.voice.activePreset = this.detectVoicePreset();
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Failed to save local voice config";
                    }
                },

                async saveWakeWordConfig() {
                    try {
                        const res = await fetch("/agent/voice/wake", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                enabled: !!this.voice.pipelineEnabled,
                                wake_word: this.voice.wakeWord || "hey jarvis",
                                threshold: this.voice.threshold,
                                chunk_ms: this.voice.chunkMs,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.voice.pipelineAvailable = !!data.available;
                        this.voice.pipelineEnabled = !!data.enabled;
                        this.voice.wakeEnabled = !!data.enabled;
                        this.voice.pipelineStatus = this.voice.pipelineAvailable ? "wake pipeline ready" : "wake pipeline unavailable";
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Failed to save wake pipeline";
                    }
                },

                async loadWatchers() {
                    try {
                        const res = await fetch("/agent/autonomy/watchers?limit=50", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.autonomy.watchers = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.autonomy.watchers = [];
                    }
                },

                async loadWatcherNetwork() {
                    try {
                        const res = await fetch("/agent/autonomy/watchers/network", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.autonomy.network = await res.json();
                    } catch (_err) {
                        this.autonomy.network = null;
                    }
                },

                async saveWorkspaceWatcher() {
                    if (!(this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id)) {
                        this.agent.error = "Activate a workspace before enabling a watcher.";
                        return;
                    }
                    this.autonomy.watcherSaving = true;
                    try {
                        const res = await fetch("/agent/autonomy/watchers", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                workspace_id: this.memoryCore.activeWorkspace.id,
                                interval_minutes: this.autonomy.watcherIntervalMinutes,
                                session_id: this.agent.sessionId,
                                auto_approve: !!this.autonomy.autoApprove,
                                min_score: this.autonomy.watcherMinScore,
                                enabled: true,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Project watcher armed for " + (this.memoryCore.activeWorkspace.name || "workspace") + ".",
                            tool: data,
                        });
                        await this.loadWatchers();
                        await this.loadWatcherNetwork();
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save project watcher";
                    } finally {
                        this.autonomy.watcherSaving = false;
                    }
                },

                async saveBrowserTemplate() {
                    const name = String(this.control.saveTemplateName || "").trim();
                    if (!name) return;
                    let steps = [];
                    try {
                        steps = JSON.parse(String(this.control.workflowSteps || "[]"));
                        if (!Array.isArray(steps) || !steps.length) throw new Error("No steps");
                    } catch (_err) {
                        this.agent.error = "Workflow steps must be valid JSON array before saving a template";
                        return;
                    }
                    try {
                        const res = await fetch("/agent/control/browser/templates", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                name: name,
                                description: "Saved from Command Deck",
                                start_url: this.control.workflowUrl || null,
                                steps: steps,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.templates = Array.isArray(data.items) ? data.items : [];
                        this.control.selectedTemplateName = name;
                        this.control.saveTemplateName = "";
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save browser template";
                    }
                },

                async loadTrustReport() {
                    try {
                        const res = await fetch("/agent/trust/report?limit=120", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.trust.report = await res.json();
                    } catch (_err) {
                        this.trust.report = null;
                    }
                },

                async loadTrustReceipts() {
                    try {
                        const params = new URLSearchParams();
                        params.set("limit", "24");
                        if (this.agent.sessionId) params.set("session_id", this.agent.sessionId);
                        const res = await fetch("/agent/trust/receipts?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.trust.receipts = await res.json();
                    } catch (_err) {
                        this.trust.receipts = null;
                    }
                },

                trustScorePct() {
                    const report = this.trust.report || {};
                    const total = Number(report.total_runs || 0);
                    const verified = Number(report.verified_runs || 0);
                    if (!total) return 0;
                    return Math.max(0, Math.min(100, (verified / total) * 100));
                },

                trustScoreStyle() {
                    const pct = this.trustScorePct();
                    let color = "#2ee6d6";
                    if (pct < 50) color = "#ef4444";
                    else if (pct < 80) color = "#f59e0b";
                    return "width:" + pct + "%;background:" + color + ";box-shadow:0 0 12px " + color + ";";
                },

                async saveControlConfig() {
                    try {
                        const res = await fetch("/agent/control/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify(this.control.config),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.config = await res.json();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save control config";
                    }
                },

                async runBrowserWorkflow() {
                    let steps = [];
                    try {
                        steps = JSON.parse(String(this.control.workflowSteps || "[]"));
                        if (!Array.isArray(steps) || !steps.length) throw new Error("No steps");
                    } catch (_err) {
                        this.agent.error = "Workflow steps must be valid JSON array";
                        return;
                    }
                    try {
                        const res = await fetch("/agent/control/browser/workflow", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                start_url: this.control.workflowUrl || null,
                                steps: steps,
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                headless: true,
                                session_name: this.control.selectedSessionName || null,
                                save_session: !!this.control.saveSession,
                                session_notes: this.control.sessionNotes || null,
                                template_name: this.control.selectedTemplateName || null,
                                confirm: !!this.control.confirmActions,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        this.rememberWorkflowOutcome(this.control.selectedTemplateName || 'Browser Workflow', this.control.lastResult);
                        await this.loadBrowserSessions();
                        await this.loadTrustReport();
                        await this.loadTrustReceipts();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to run browser workflow";
                    }
                },

                async runGitHubPullReview() {
                    const pullNumber = Number(this.control.githubReview.pullNumber || 0);
                    if (!pullNumber) {
                        this.agent.error = 'GitHub PR number is required';
                        return;
                    }
                    try {
                        const res = await fetch('/agent/integrations/github/pulls/review', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                            body: JSON.stringify({
                                repo: this.control.githubReview.repo || null,
                                pull_number: pullNumber,
                                body: this.control.githubReview.body || '',
                                event: this.control.githubReview.event || 'COMMENT',
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        this.rememberWorkflowOutcome('GitHub PR Review', this.control.lastResult);
                        await this.loadTrustReceipts();
                        await this.loadIntegrationsSummary();
                        await this.loadGitHubPullSummary();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : 'Failed to send GitHub PR review';
                    }
                },

                async sendGmailDraft() {
                    const subject = String(this.control.gmailCompose.subject || '').trim();
                    const body = String(this.control.gmailCompose.body || '').trim();
                    if (!subject || !body) {
                        this.agent.error = 'Gmail subject and body are required';
                        return;
                    }
                    try {
                        const res = await fetch('/agent/integrations/email/send', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                            body: JSON.stringify({
                                to: this.control.gmailCompose.to || null,
                                subject,
                                body,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        this.rememberWorkflowOutcome('Gmail Draft Send', this.control.lastResult);
                        await this.loadIntegrationsSummary();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : 'Failed to send Gmail draft';
                    }
                },

                async executeNextAction(action) {
                    if (!action) return;
                    try {
                        const res = await fetch("/agent/next-action/execute", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                action: action,
                                session_id: this.agent.sessionId,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.agent.sessionId = data.session_id || this.agent.sessionId;
                        const result = data.result || {};
                        const text = (result.reply || action.title || "Action executed.");
                        this.agent.messages.push({
                            role: "assistant",
                            text: text,
                            tool: data,
                        });
                        await this.loadNextBestActions();
                        await this.loadWorkspaces();
                        await this.loadReminders();
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to execute next action";
                    }
                },

                async runAutonomyMission() {
                    this.autonomy.running = true;
                    try {
                        const res = await fetch("/agent/autonomy/missions/start", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                session_id: this.agent.sessionId,
                                limit: 3,
                                auto_approve: !!this.autonomy.autoApprove,
                                retry_limit: Number(this.autonomy.retryLimit || 1),
                                goal: this.memoryCore.activeWorkspace ? ("Advance workspace " + this.memoryCore.activeWorkspace.name) : "Advance Jarvis priorities",
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.autonomy.lastMission = data;
                        this.agent.sessionId = data.session_id || this.agent.sessionId;
                        this.agent.messages.push({
                            role: "assistant",
                            text: data.summary || "Autonomy mission complete.",
                            tool: data,
                        });
                        await this.loadNextBestActions();
                        await this.loadReminders();
                        await this.loadWorkspaces();
                        await this.loadMissionRuns();
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Mission mode failed";
                    } finally {
                        this.autonomy.running = false;
                    }
                },

                async resumeLatestMission() {
                    const mission = (this.autonomy.missionRuns || [])[0];
                    if (!mission || !mission.id) return;
                    try {
                        const res = await fetch(`/agent/autonomy/missions/${mission.id}/resume?approve=${this.autonomy.autoApprove ? 'true' : 'false'}`, {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.autonomy.lastMission = data;
                        this.agent.sessionId = data.session_id || this.agent.sessionId;
                        this.pushMissionEvent('Mission resumed from checkpoint');
                        await this.loadMissionRuns();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to resume mission";
                    }
                },

                async loadMissionRuns() {
                    try {
                        const params = new URLSearchParams({ limit: "8" });
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/autonomy/missions?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.autonomy.missionRuns = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.autonomy.missionRuns = [];
                    }
                },

                async loadIntegrationsSummary() {
                    try {
                        const [cfgRes, summaryRes] = await Promise.all([
                            fetch("/agent/integrations/config", { headers: { "Accept": "application/json" } }),
                            fetch("/agent/integrations/summary", { headers: { "Accept": "application/json" } }),
                        ]);
                        if (!cfgRes.ok) throw new Error(await cfgRes.text());
                        if (!summaryRes.ok) throw new Error(await summaryRes.text());
                        this.integrations.config = await cfgRes.json();
                        this.integrations.summary = await summaryRes.json();
                        if (!String(this.control.githubReview.repo || '').trim()) {
                            this.control.githubReview.repo = this.integrations.config.github_repo || '';
                        }
                        if (!String(this.control.gmailCompose.to || '').trim()) {
                            this.control.gmailCompose.to = this.integrations.config.email_to || '';
                        }
                    } catch (_err) {
                        this.integrations.summary = null;
                    }
                },

                async loadIntegrationsIntelligence() {
                    try {
                        const res = await fetch("/agent/integrations/intelligence", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.integrations.intelligence = await res.json();
                    } catch (_err) {
                        this.integrations.intelligence = null;
                    }
                },

                async loadCalendarEvents() {
                    try {
                        const res = await fetch("/agent/integrations/calendar/events?limit=6", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.integrations.calendarEvents = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.integrations.calendarEvents = [];
                    }
                },

                async loadDesktopPresence() {
                    try {
                        const params = new URLSearchParams();
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/desktop/presence" + (params.toString() ? ("?" + params.toString()) : ""), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.presence.snapshot = data.snapshot || null;
                        this.presence.workspace = data.workspace || null;
                        this.presence.next_actions = data.next_actions || null;
                        this.presence.reminders = Array.isArray(data.reminders) ? data.reminders : [];
                        this.presence.recent_vision = Array.isArray(data.recent_vision) ? data.recent_vision : [];
                    } catch (_err) {
                        this.presence.snapshot = null;
                    }
                },

                async loadDesktopAwareness() {
                    try {
                        const params = new URLSearchParams();
                        if (this.memoryCore.activeWorkspace && this.memoryCore.activeWorkspace.id) {
                            params.set("workspace_id", String(this.memoryCore.activeWorkspace.id));
                        }
                        const res = await fetch("/agent/desktop/awareness" + (params.toString() ? ("?" + params.toString()) : ""), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.presence.awareness = await res.json();
                    } catch (_err) {
                        this.presence.awareness = null;
                    }
                },

                bindDesktopAwarenessHooks() {
                    if (this.presence.listenersBound) return;
                    this.presence.listenersBound = true;
                    const refreshAwareness = () => {
                        this.captureDesktopPresenceAuto();
                        this.loadDesktopAwareness();
                    };
                    window.addEventListener('focus', refreshAwareness);
                    window.addEventListener('resize', refreshAwareness);
                    document.addEventListener('visibilitychange', refreshAwareness);
                    document.addEventListener('selectionchange', () => {
                        if (document.visibilityState === 'visible') this.captureDesktopPresenceAuto();
                    });
                },

                buildDesktopAutoDetails() {
                    const activeElement = document.activeElement;
                    const selection = window.getSelection ? String(window.getSelection() || '').trim() : '';
                    return {
                        current_view: this.currentView,
                        mission_mode: !!this.missionMode,
                        page_url: window.location.href,
                        page_path: window.location.pathname,
                        visibility_state: document.visibilityState,
                        document_title: document.title || 'Jarvis AI Control Center',
                        has_active_session: !!this.agent.sessionId,
                        active_workspace: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.name : null,
                        top_next_action: (this.memoryCore.nextActions && this.memoryCore.nextActions.actions && this.memoryCore.nextActions.actions.length)
                            ? this.memoryCore.nextActions.actions[0].title
                            : null,
                        viewport: window.innerWidth + 'x' + window.innerHeight,
                        screen_resolution: (window.screen && window.screen.width && window.screen.height) ? (window.screen.width + 'x' + window.screen.height) : null,
                        color_scheme: (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light',
                        active_element: activeElement ? (activeElement.tagName || '').toLowerCase() : null,
                        selected_text: selection ? selection.slice(0, 180) : null,
                        recent_command: this.lastMissionCommand(),
                        integrations_connected: this.integrations.intelligence ? Number(this.integrations.intelligence.connected_count || 0) : 0,
                        watcher_count: Array.isArray(this.autonomy.watchers) ? this.autonomy.watchers.filter((job) => job.enabled).length : 0,
                        voice_state: this.voice.speaking ? 'speaking' : (this.voice.localRecording ? 'recording' : (this.voice.pipelineListening ? 'wake-listening' : (this.voice.listening ? 'listening' : 'idle'))),
                    };
                },

                async captureDesktopPresenceAuto() {
                    if (!this.presence.autoSyncEnabled || this.currentView !== "agent") return;
                    try {
                        const details = this.buildDesktopAutoDetails();
                        const payload = {
                            workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                            app_name: "Jarvis Dashboard",
                            window_title: document.title || "Jarvis AI Control Center",
                            summary: (this.missionMode ? 'Mission mode' : 'Command mode') + ' on ' + (this.currentView || 'agent') + ' with ' + (details.voice_state || 'idle') + ' voice',
                            details,
                        };
                        const res = await fetch("/agent/desktop/presence", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify(payload),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.presence.snapshot = data.snapshot || this.presence.snapshot;
                    } catch (_err) {}
                },

                async pollReminderVoiceFeed() {
                    if (!this.voice.autoSpeak) return;
                    try {
                        const res = await fetch("/agent/memory/reminders/voice-feed?limit=6", { headers: { "Accept": "application/json" } });
                        if (!res.ok) return;
                        const data = await res.json();
                        const items = Array.isArray(data.items) ? data.items : [];
                        for (const item of items) {
                            if (!item || !item.text) continue;
                            this.speakText(String(item.text), true);
                            this.agent.messages.push({ role: "assistant", text: String(item.text), tool: item });
                        }
                        if (items.length) this.$nextTick(() => this._scrollMessages());
                    } catch (_err) {}
                },

                async browserOpen() {
                    const url = String(this.control.browserUrl || "").trim();
                    if (!url) return;
                    try {
                        const res = await fetch("/agent/control/browser/open", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                url: url,
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                confirm: !!this.control.confirmActions,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        await this.loadTrustReport();
                        await this.loadTrustReceipts();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to open browser target";
                    }
                },

                async browserSearch() {
                    const query = String(this.control.browserQuery || "").trim();
                    if (!query) return;
                    try {
                        const res = await fetch("/agent/control/browser/search", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                query: query,
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                confirm: !!this.control.confirmActions,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        await this.loadTrustReport();
                        await this.loadTrustReceipts();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to run browser search";
                    }
                },

                async desktopLaunch() {
                    const app = String(this.control.desktopApp || "").trim();
                    if (!app) return;
                    try {
                        const res = await fetch("/agent/control/desktop/launch", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                app: app,
                                workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                                confirm: !!this.control.confirmActions,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        await this.loadTrustReport();
                        await this.loadTrustReceipts();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to launch desktop target";
                    }
                },

                async desktopAction(action) {
                    try {
                        const payload = {
                            action,
                            workspace_id: this.memoryCore.activeWorkspace ? this.memoryCore.activeWorkspace.id : null,
                            confirm: !!this.control.confirmActions,
                        };
                        if (action === 'open_url') payload.target = String(this.control.desktopUrl || '').trim();
                        if (action === 'type_text') payload.text = String(this.control.desktopText || '').trim();
                        if (action === 'hotkey') payload.keys = String(this.control.desktopHotkey || '').split('+').map((item) => item.trim()).filter(Boolean);
                        const res = await fetch("/agent/control/desktop/action", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify(payload),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.control.lastResult = await res.json();
                        await this.loadTrustReport();
                        await this.loadTrustReceipts();
                        await this.loadNextBestActions();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to run desktop control";
                    }
                },

                async runWorkflowQuickAction(templateName) {
                    if (!templateName) return;
                    this.control.selectedTemplateName = templateName;
                    this.applyBrowserTemplateByName();
                    await this.runBrowserWorkflow();
                },

                missionCheckpointCard() {
                    const mission = this.autonomy.lastMission || ((this.autonomy.missionRuns || []).length ? (this.autonomy.missionRuns[0].result || this.autonomy.missionRuns[0]) : null) || {};
                    const checkpoint = mission.checkpoint || {};
                    const failed = Array.isArray(mission.failed) ? mission.failed : [];
                    const blocked = Array.isArray(mission.blocked) ? mission.blocked : [];
                    const remaining = Array.isArray(checkpoint.remaining_actions) ? checkpoint.remaining_actions.length : 0;
                    const retryLimit = Number(mission.retry_limit || checkpoint.retry_limit || this.autonomy.retryLimit || 1);
                    const retryUsed = Number(checkpoint.retry_attempts_used || 0);
                    return {
                        status: mission.ok ? 'Mission stable' : (blocked.length ? 'Approval paused' : (failed.length ? 'Retry pending' : 'Standby')),
                        summary: mission.summary || 'No active mission checkpoint.',
                        retry: `${retryUsed}/${retryLimit} retries used`,
                        failed: failed.length ? failed[0].title || failed[0].kind || 'Failure recorded' : (blocked.length ? 'Awaiting approval' : 'No failures logged'),
                        remaining: remaining ? `${remaining} step${remaining === 1 ? '' : 's'} queued` : 'Queue clear',
                        checkpointLine: remaining ? `${remaining} queued with retry window ${retryUsed}/${retryLimit}.` : `Retry window ${retryUsed}/${retryLimit}.`,
                        canResume: !!remaining || !!blocked.length || !!failed.length,
                    };
                },

                async ensureDailyBriefings() {
                    try {
                        const res = await fetch("/agent/memory/briefings/automations", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                session_id: this.agent.sessionId,
                                timezone_name: this.schedules.briefingTimezone || "America/Chicago",
                                morning_hour: Number(this.schedules.briefingMorningHour || 8),
                                evening_hour: Number(this.schedules.briefingEveningHour || 18),
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.schedules.briefingJobs = Array.isArray(data.items) ? data.items : [];
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Daily Jarvis briefings are scheduled and persistent now.",
                            tool: data,
                        });
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to set daily briefings";
                    }
                },

                async saveBriefingDeliveryConfig() {
                    try {
                        const res = await fetch("/agent/memory/briefings/delivery", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify(this.schedules.briefingDelivery),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.schedules.briefingDelivery = await res.json();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to save briefing delivery";
                    }
                },

                async testBriefingDelivery() {
                    try {
                        const res = await fetch("/agent/memory/briefings/delivery/test?period=morning", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.schedules.lastDeliveryTest = await res.json();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to test briefing delivery";
                    }
                },

                async toggleBriefingAutomation(job) {
                    if (!job) return;
                    try {
                        const res = await fetch("/agent/autonomy/jobs/" + job.id, {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ enabled: !job.enabled }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadBriefingAutomations();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to toggle daily briefing";
                    }
                },

                async runBriefingAutomation(job) {
                    if (!job) return;
                    try {
                        const res = await fetch("/agent/autonomy/jobs/" + job.id + "/run", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        const text = (((data || {}).result || {}).text) || "Briefing executed.";
                        this.agent.messages.push({ role: "assistant", text: text, tool: data });
                        this.$nextTick(() => this._scrollMessages());
                        if (this.voice.autoSpeak) this.speakText(text, true);
                        await this.loadBriefingAutomations();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to run daily briefing";
                    }
                },

                async createSchedule() {
                    const goal = (this.schedules.inputGoal || "").trim();
                    if (!goal) return;
                    try {
                        const res = await fetch("/agent/goals/schedule", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                goal: goal,
                                interval_minutes: Number(this.schedules.inputMinutes || 60),
                                auto_approve: !!this.schedules.inputAutoApprove,
                                session_id: this.agent.sessionId,
                                enabled: true,
                            }),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        this.schedules.inputGoal = "";
                        await this.loadSchedules();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to create schedule";
                    }
                },

                async toggleSchedule(s) {
                    try {
                        const res = await fetch("/agent/goals/schedule/" + s.id, {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ enabled: !s.enabled }),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        await this.loadSchedules();
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Failed to update schedule";
                    }
                },

                quantumDecipherSchedules() {
                    const items = Array.isArray(this.schedules.items) ? this.schedules.items : [];
                    return items.filter(s => {
                        const goal = ((s && s.goal) ? String(s.goal) : "").toLowerCase();
                        return goal.includes("quantum decipher") || goal.includes("quantum decypher");
                    });
                },

                quantumDecipherScheduleSummary() {
                    const items = this.quantumDecipherSchedules();
                    const active = items.filter(s => !!s.enabled).length;
                    return { active: active, total: items.length };
                },

                async setDecipherSchedulesEnabled(enabled) {
                    try {
                        const targets = this.quantumDecipherSchedules();
                        for (const s of targets) {
                            if (!!s.enabled === !!enabled) continue;
                            await fetch("/agent/goals/schedule/" + s.id, {
                                method: "POST",
                                headers: { "Content-Type": "application/json", "Accept": "application/json" },
                                body: JSON.stringify({ enabled: !!enabled }),
                            });
                        }
                        await this.loadSchedules();
                        this.quantum.lastResult = { decipher_schedules_enabled: !!enabled, count: this.quantumDecipherSchedules().length };
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "failed to toggle decipher schedules" };
                    }
                },

                async loadQuantumStatus() {
                    try {
                        const res = await fetch("/agent/quantum/status", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error("Failed to load quantum status");
                        this.quantum.status = await res.json();
                    } catch (err) {
                        this.quantum.status = { available: false, error: (err && err.message) ? err.message : "status error" };
                    }
                },

                quantumOutcomePct() {
                    if (!this.quantum.stats24 || !this.quantum.stats24.measurements) return 0;
                    const m = this.quantum.stats24.measurement_outcomes || {};
                    const ones = Number(m["1"] || 0);
                    const total = Number(this.quantum.stats24.measurements || 0);
                    if (!total) return 0;
                    return (ones / total) * 100;
                },
                quantumStatusHighlights() {
                    const status = this.quantum.status || {};
                    const stats = this.quantum.stats24 || {};
                    const lines = [];
                    if (status.available != null) lines.push(`availability: ${status.available ? "online" : "offline"}`);
                    if (stats.total_events != null) lines.push(`24h events: ${stats.total_events}`);
                    if (stats.measurements != null) lines.push(`measurements: ${stats.measurements} | entangles: ${stats.entangles || 0}`);
                    if (stats.avg_entanglement_strength != null) lines.push(`avg entanglement strength: ${stats.avg_entanglement_strength}`);
                    if (this.quantum.alerts && this.quantum.alerts.active != null) lines.push(`alerts: ${this.quantum.alerts.active ? "active" : "clear"}`);
                    return lines.length ? lines : ["No quantum status loaded yet."];
                },

                quantumHealthTone() {
                    const score = (this.quantum.noc && this.quantum.noc.health_score != null) ? Number(this.quantum.noc.health_score) : null;
                    if (score == null) return "Awaiting live score";
                    if (score >= 90) return "Stable and healthy";
                    if (score >= 70) return "Watch but steady";
                    if (score >= 45) return "Degraded";
                    return "Needs attention";
                },

                quantumAlertThresholdText() {
                    const cfg = (this.quantum.alerts && this.quantum.alerts.config) ? this.quantum.alerts.config : {};
                    const stats = this.quantum.stats24 || {};
                    const measurements = Number(stats.measurements || 0);
                    const entangles = Number(stats.entangles || 0);
                    const minMeasurements = Number(cfg.min_measurements || 0);
                    const minEntangles = Number(cfg.min_entangles || 0);
                    const measurementGap = Math.max(0, minMeasurements - measurements);
                    const entangleGap = Math.max(0, minEntangles - entangles);
                    if (measurementGap > 0 || entangleGap > 0) {
                        return "Alerts are currently quiet partly because the sample size is still building. Jarvis needs "
                            + measurementGap + " more measurements and "
                            + entangleGap + " more entanglement events to fully evaluate every threshold.";
                    }
                    if (this.quantum.alerts && this.quantum.alerts.active) {
                        return "Threshold coverage is active and at least one live alert is firing right now.";
                    }
                    return "Threshold coverage is active and current telemetry is staying inside the configured safe range.";
                },

                quantumBriefingLines() {
                    const stats = this.quantum.stats24 || {};
                    const noc = this.quantum.noc || {};
                    const decipher = this.quantum.decipher || {};
                    const events = Number(stats.total_events || 0);
                    const measurements = Number(stats.measurements || 0);
                    const entangles = Number(stats.entangles || 0);
                    const superpositions = Number(stats.superpositions || 0);
                    const outcomePct = this.quantumOutcomePct();
                    const entanglement = Number(stats.avg_entanglement_strength || 0);
                    const anomalyCount = Number(noc.anomaly_count || 0);
                    const confidence = Number(decipher.confidence_pct || 0);
                    const recommendation = Array.isArray(decipher.recommendations) && decipher.recommendations.length
                        ? String(decipher.recommendations[0])
                        : "Jarvis is still collecting enough signal to write a stronger recommendation.";

                    const line1 = events
                        ? "Jarvis has logged " + events + " quantum events in the last 24 hours: "
                            + measurements + " measurements, "
                            + entangles + " entanglements, and "
                            + superpositions + " superpositions."
                        : "Jarvis has not received any recent quantum events yet.";

                    const line2 = measurements
                        ? "Measurement outcomes are currently " + outcomePct.toFixed(1) + "% outcome=1, which reads as "
                            + (outcomePct >= 45 && outcomePct <= 55 ? "near-balanced." : "slightly biased.")
                        : "There are no recent measurements yet, so balance cannot be estimated.";

                    const line3 = entangles
                        ? "Average entanglement strength is " + entanglement.toFixed(3) + ", which looks "
                            + (entanglement >= 0.9 ? "stable." : "weaker than ideal.")
                        : "No recent entanglement events are available to judge channel stability.";

                    const line4 = "NOC health is "
                        + ((noc.health_score != null) ? Number(noc.health_score).toFixed(1) : "unknown")
                        + " with "
                        + anomalyCount
                        + " anomalies detected. Decipher confidence is "
                        + confidence.toFixed(1)
                        + "%. "
                        + recommendation;

                    return [line1, line2, line3, line4];
                },

                async loadQuantumAnalytics() {
                    try {
                        const [s24, s7d, hist] = await Promise.all([
                            fetch("/agent/quantum/stats?hours=24", { headers: { "Accept": "application/json" } }),
                            fetch("/agent/quantum/stats?hours=168", { headers: { "Accept": "application/json" } }),
                            fetch("/agent/quantum/history?limit=50", { headers: { "Accept": "application/json" } }),
                        ]);
                        this.quantum.stats24 = s24.ok ? await s24.json() : null;
                        this.quantum.stats7d = s7d.ok ? await s7d.json() : null;
                        if (hist.ok) {
                            const data = await hist.json();
                            this.quantum.history = Array.isArray(data.events) ? data.events : [];
                        } else {
                            this.quantum.history = [];
                        }
                    } catch (_err) {
                        this.quantum.stats24 = null;
                        this.quantum.stats7d = null;
                        this.quantum.history = [];
                    }
                },

                async loadQuantumAlerts() {
                    try {
                        const res = await fetch("/agent/quantum/alerts", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error("Failed to load alerts");
                        const data = await res.json();
                        this.quantum.alerts.active = !!data.active;
                        this.quantum.alerts.items = Array.isArray(data.alerts) ? data.alerts : [];
                        if (data.config && typeof data.config === "object") {
                            this.quantum.alerts.config = { ...this.quantum.alerts.config, ...data.config };
                        }
                    } catch (_err) {
                        this.quantum.alerts.active = false;
                        this.quantum.alerts.items = [];
                    }
                },

                startQuantumStream() {
                    try {
                        if (this.quantumStream) {
                            this.quantumStream.close();
                            this.quantumStream = null;
                        }
                        const es = new EventSource("/agent/quantum/stream");
                        es.addEventListener("quantum", (evt) => {
                            try {
                                const data = JSON.parse(evt.data || "{}");
                                if (data.noc) this.quantum.noc = data.noc;
                                if (data.alerts) {
                                    this.quantum.alerts.active = !!data.alerts.active;
                                    this.quantum.alerts.items = Array.isArray(data.alerts.alerts) ? data.alerts.alerts : [];
                                    if (data.alerts.config && typeof data.alerts.config === "object") {
                                        this.quantum.alerts.config = { ...this.quantum.alerts.config, ...data.alerts.config };
                                    }
                                }
                                if (Array.isArray(data.incidents)) this.quantum.workspace.incidents = data.incidents;
                                if (data.risk) {
                                    this.quantum.risk = data.risk;
                                    if (data.risk.risk_score != null) {
                                        this.quantum.riskTrend.push(Number(data.risk.risk_score));
                                        if (this.quantum.riskTrend.length > 24) this.quantum.riskTrend.shift();
                                    }
                                }
                                if (data.slo) this.quantum.slo = data.slo;
                                this.quantum.streamConnected = true;
                            } catch (_e) {
                                this.quantum.streamConnected = false;
                            }
                        });
                        es.onerror = () => {
                            this.quantum.streamConnected = false;
                        };
                        this.quantumStream = es;
                    } catch (_err) {
                        this.quantum.streamConnected = false;
                    }
                },

                async loadIncidentWorkspace(incidentId) {
                    try {
                        const params = new URLSearchParams();
                        if (incidentId) params.set("incident_id", incidentId);
                        const url = "/agent/quantum/incidents/workspace" + (params.toString() ? ("?" + params.toString()) : "");
                        const res = await fetch(url, { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.quantum.workspace.incident = data.incident || null;
                        this.quantum.workspace.annotations = Array.isArray(data.annotations) ? data.annotations : [];
                        this.quantum.workspace.incidents = Array.isArray(data.incidents) ? data.incidents : [];
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "workspace load failed" };
                    }
                },

                async setIncidentStatus(status) {
                    const inc = this.quantum.workspace.incident;
                    if (!inc || !inc.id) return;
                    try {
                        const url = "/agent/quantum/incidents/" + encodeURIComponent(inc.id) + "/status?status=" + encodeURIComponent(status);
                        const res = await fetch(url, { method: "POST", headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadIncidentWorkspace(inc.id);
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "incident status update failed" };
                    }
                },

                async toggleIncidentChecklist(item, done) {
                    const inc = this.quantum.workspace.incident;
                    if (!inc || !inc.id || !item || !item.id) return;
                    try {
                        const url = "/agent/quantum/incidents/" + encodeURIComponent(inc.id) + "/checklist?item_id=" + encodeURIComponent(item.id) + "&done=" + (done ? "true" : "false");
                        const res = await fetch(url, { method: "POST", headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadIncidentWorkspace(inc.id);
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "checklist toggle failed" };
                    }
                },

                async drillSimulateFromAlert(a) {
                    await this.simulateQuantumScenario();
                },

                async drillRunbookFromAlert(a) {
                    this.quantum.runbookCode = (a && a.code) ? String(a.code) : (this.quantum.runbookCode || "general");
                    await this.loadQuantumRunbook();
                },

                async drillRemediateFromAlert(a) {
                    await this.runQuantumRemediation(true);
                },

                async drillAnnotateFromAlert(a) {
                    const code = (a && a.code) ? String(a.code) : "alert";
                    this.quantum.annotationNote = "Drill note for " + code + ": " + ((a && a.message) ? String(a.message) : "no message");
                    await this.addQuantumAnnotation();
                    await this.loadIncidentWorkspace();
                },

                async saveQuantumAlertConfig() {
                    try {
                        const cfg = this.quantum.alerts.config || {};
                        const res = await fetch("/agent/quantum/alerts/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                enabled: !!cfg.enabled,
                                window_hours: Number(cfg.window_hours || 24),
                                min_measurements: Number(cfg.min_measurements || 5),
                                min_entangles: Number(cfg.min_entangles || 3),
                                outcome_one_min_pct: Number(cfg.outcome_one_min_pct || 20),
                                outcome_one_max_pct: Number(cfg.outcome_one_max_pct || 80),
                                entanglement_strength_min: Number(cfg.entanglement_strength_min || 0.9),
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        await this.loadQuantumAlerts();
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "failed to save alert config" };
                    }
                },

                downloadQuantumExport(format) {
                    const params = new URLSearchParams();
                    params.set("format", format);
                    params.set("hours", "24");
                    if (this.quantum.exportStart) params.set("start_at", new Date(this.quantum.exportStart).toISOString());
                    if (this.quantum.exportEnd) params.set("end_at", new Date(this.quantum.exportEnd).toISOString());
                    const url = "/agent/quantum/export?" + params.toString();
                    window.open(url, "_blank");
                },

                downloadQuantumAlertsExport(format) {
                    const url = "/agent/quantum/alerts/export?format=" + encodeURIComponent(format);
                    window.open(url, "_blank");
                },

                downloadQuantumExportAll(format) {
                    const params = new URLSearchParams();
                    params.set("format", format);
                    params.set("hours", "24");
                    if (this.quantum.exportStart) params.set("start_at", new Date(this.quantum.exportStart).toISOString());
                    if (this.quantum.exportEnd) params.set("end_at", new Date(this.quantum.exportEnd).toISOString());
                    const url = "/agent/quantum/export/all?" + params.toString();
                    window.open(url, "_blank");
                },

                async runQuantumDecipher() {
                    try {
                        const params = new URLSearchParams();
                        params.set("hours", "24");
                        if (this.quantum.exportStart) params.set("start_at", new Date(this.quantum.exportStart).toISOString());
                        if (this.quantum.exportEnd) params.set("end_at", new Date(this.quantum.exportEnd).toISOString());
                        const res = await fetch("/agent/quantum/decipher?" + params.toString(), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.decipher = await res.json();
                    } catch (err) {
                        this.quantum.decipher = { error: (err && err.message) ? err.message : "decipher failed" };
                    }
                },

                async loadQuantumNoc() {
                    try {
                        const res = await fetch("/agent/quantum/noc?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.noc = await res.json();
                    } catch (err) {
                        this.quantum.noc = {
                            health_score: null,
                            active_alerts: null,
                            anomaly_count: null,
                            last_snapshot_delta_bias_pct: null,
                            events: null,
                            error: (err && err.message) ? err.message : "noc failed",
                        };
                    }
                },

                async loadQuantumTimeline() {
                    try {
                        const res = await fetch("/agent/quantum/timeline?hours=24&bucket_minutes=60", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "timeline failed" };
                    }
                },

                async loadQuantumAnomalies() {
                    try {
                        const res = await fetch("/agent/quantum/anomalies?hours=24&z_threshold=2.0", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "anomalies failed" };
                    }
                },

                async loadQuantumBasisAnalysis() {
                    try {
                        const res = await fetch("/agent/quantum/basis-analysis?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "basis analysis failed" };
                    }
                },

                async runQuantumExperiment(preset) {
                    try {
                        const res = await fetch("/agent/quantum/experiment/run", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ preset: preset || "quick" }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                        await Promise.all([this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "experiment failed" };
                    }
                },

                async saveQuantumSnapshot() {
                    try {
                        const res = await fetch("/agent/quantum/decipher/snapshot?hours=24", { method: "POST", headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "snapshot failed" };
                    }
                },

                async loadQuantumSnapshots() {
                    try {
                        const res = await fetch("/agent/quantum/decipher/snapshots?limit=20", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "snapshot list failed" };
                    }
                },

                async runQuantumRemediation(forceFlag) {
                    try {
                        const res = await fetch("/agent/quantum/remediation/run?hours=24&force=" + (forceFlag ? "true" : "false"), {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                        await Promise.all([this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "remediation failed" };
                    }
                },

                async tuneQuantumRemediation(applyFlag) {
                    try {
                        const url = "/agent/quantum/remediation/tune?hours=168&apply=" + (applyFlag ? "true" : "false");
                        const res = await fetch(url, { method: "POST", headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "remediation tune failed" };
                    }
                },

                async loadQuantumHealthScore() {
                    try {
                        const res = await fetch("/agent/quantum/health-score?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "health score failed" };
                    }
                },

                async loadQuantumReplay() {
                    try {
                        const res = await fetch("/agent/quantum/replay?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "replay failed" };
                    }
                },

                downloadQuantumSummaryPdf() {
                    window.open("/agent/quantum/summary.pdf?hours=24", "_blank");
                },

                async loadQuantumNotificationConfig() {
                    try {
                        const res = await fetch("/agent/quantum/notifications/config", { headers: { "Accept": "application/json" } });
                        if (!res.ok) return;
                        const data = await res.json();
                        this.quantum.notifications.webhookUrl = data.webhook_url || "";
                        this.quantum.notifications.webhookUrlWarning = data.webhook_url_warning || "";
                        this.quantum.notifications.webhookUrlCritical = data.webhook_url_critical || "";
                        this.quantum.notifications.channel = data.channel || "generic";
                    } catch (_err) {
                        this.quantum.notifications.webhookUrl = "";
                        this.quantum.notifications.webhookUrlWarning = "";
                        this.quantum.notifications.webhookUrlCritical = "";
                        this.quantum.notifications.channel = "generic";
                    }
                },

                async saveQuantumNotificationConfig() {
                    try {
                        const res = await fetch("/agent/quantum/notifications/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                enabled: !!((this.quantum.notifications.webhookUrl || "").trim() || (this.quantum.notifications.webhookUrlWarning || "").trim() || (this.quantum.notifications.webhookUrlCritical || "").trim()),
                                channel: this.quantum.notifications.channel || "generic",
                                webhook_url: (this.quantum.notifications.webhookUrl || "").trim(),
                                webhook_url_warning: (this.quantum.notifications.webhookUrlWarning || "").trim(),
                                webhook_url_critical: (this.quantum.notifications.webhookUrlCritical || "").trim(),
                                min_severity: "warning",
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "save webhook config failed" };
                    }
                },

                async testQuantumNotification() {
                    try {
                        const res = await fetch("/agent/quantum/notifications/test", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "test webhook failed" };
                    }
                },

                async sendQuantumNotifications() {
                    try {
                        const res = await fetch("/agent/quantum/notifications/dispatch?hours=24", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "dispatch webhook failed" };
                    }
                },

                async loadQuantumOpsAudit() {
                    try {
                        const res = await fetch("/agent/quantum/ops-audit?limit=100", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "ops audit load failed" };
                    }
                },

                async loadQuantumMemoryGraph() {
                    try {
                        const res = await fetch("/agent/quantum/memory-graph?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "memory graph failed" };
                    }
                },

                async loadQuantumAgentReplay() {
                    try {
                        const res = await fetch("/agent/quantum/replay/agent?limit=30", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "agent replay failed" };
                    }
                },

                async simulateQuantumScenario() {
                    try {
                        const res = await fetch("/agent/quantum/simulate?hours=24&outcome_one_min_pct=25&outcome_one_max_pct=75&entanglement_strength_min=0.88", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "scenario simulation failed" };
                    }
                },

                async applyQuantumPolicyPack() {
                    try {
                        const pack = this.quantum.policyPack || "safe";
                        const res = await fetch("/agent/quantum/policy-packs/" + encodeURIComponent(pack) + "/apply", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                        await Promise.all([this.loadQuantumAlerts(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "policy apply failed" };
                    }
                },

                async loadQuantumRunbook() {
                    try {
                        const code = encodeURIComponent((this.quantum.runbookCode || "").trim() || "general");
                        const res = await fetch("/agent/quantum/runbook/" + code, { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "runbook load failed" };
                    }
                },

                async loadQuantumIncidents() {
                    try {
                        const res = await fetch("/agent/quantum/incidents?limit=100", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "incidents load failed" };
                    }
                },

                async addQuantumAnnotation() {
                    const note = (this.quantum.annotationNote || "").trim();
                    if (!note) return;
                    try {
                        const res = await fetch("/agent/quantum/annotations", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                item_type: "incident",
                                item_id: "latest",
                                note: note,
                                author: "operator",
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.annotationNote = "";
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "annotation create failed" };
                    }
                },

                async loadQuantumAnnotations() {
                    try {
                        const res = await fetch("/agent/quantum/annotations?limit=100", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "annotations load failed" };
                    }
                },

                async loadQuantumCorrelations() {
                    try {
                        const res = await fetch("/agent/quantum/correlations?hours=24&window_minutes=30", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.correlations = await res.json();
                        this.quantum.advancedResult = this.quantum.correlations;
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "correlations failed" };
                    }
                },

                async recomputeQuantumBaselines() {
                    try {
                        const res = await fetch("/agent/quantum/baselines/recompute?hours=168", {
                            method: "POST",
                            headers: { "Accept": "application/json" },
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.baselines = await res.json();
                        this.quantum.advancedResult = this.quantum.baselines;
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "baseline recompute failed" };
                    }
                },

                async loadQuantumBaselines() {
                    try {
                        const res = await fetch("/agent/quantum/baselines?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.baselines = await res.json();
                        this.quantum.advancedResult = this.quantum.baselines;
                    } catch (err) {
                        this.quantum.baselines = { error: (err && err.message) ? err.message : "baseline load failed" };
                    }
                },

                async loadQuantumRootCause() {
                    try {
                        const id = (this.quantum.workspace.incident && this.quantum.workspace.incident.id) ? ("incident_id=" + encodeURIComponent(this.quantum.workspace.incident.id)) : "";
                        const res = await fetch("/agent/quantum/root-cause?hours=24" + (id ? "&" + id : ""), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.rootCause = await res.json();
                        this.quantum.advancedResult = this.quantum.rootCause;
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "root cause failed" };
                    }
                },

                async loadQuantumRiskScore() {
                    try {
                        const res = await fetch("/agent/quantum/risk-score?horizon_hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.risk = await res.json();
                        if (this.quantum.risk && this.quantum.risk.risk_score != null) {
                            this.quantum.riskTrend.push(Number(this.quantum.risk.risk_score));
                            if (this.quantum.riskTrend.length > 24) this.quantum.riskTrend.shift();
                        }
                        this.quantum.advancedResult = this.quantum.risk;
                    } catch (err) {
                        this.quantum.risk = { error: (err && err.message) ? err.message : "risk score failed" };
                    }
                },

                async runQuantumPlaybookV2(dryRun) {
                    try {
                        const res = await fetch("/agent/quantum/playbooks-v2/run", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                incident_id: this.quantum.workspace.incident ? this.quantum.workspace.incident.id : null,
                                approve: true,
                                dry_run: !!dryRun,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "playbook v2 failed" };
                    }
                },

                async loadQuantumPostmortem() {
                    try {
                        const id = (this.quantum.workspace.incident && this.quantum.workspace.incident.id) ? ("incident_id=" + encodeURIComponent(this.quantum.workspace.incident.id)) : "";
                        const res = await fetch("/agent/quantum/postmortem?hours=24" + (id ? "&" + id : ""), { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.postmortem = await res.json();
                        this.quantum.advancedResult = this.quantum.postmortem;
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "postmortem failed" };
                    }
                },

                downloadQuantumPostmortemPdf() {
                    const params = new URLSearchParams();
                    params.set("hours", "24");
                    if (this.quantum.workspace.incident && this.quantum.workspace.incident.id) {
                        params.set("incident_id", this.quantum.workspace.incident.id);
                    }
                    window.open("/agent/quantum/postmortem.pdf?" + params.toString(), "_blank");
                },

                async loadQuantumSlo() {
                    try {
                        const res = await fetch("/agent/quantum/slo?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.slo = await res.json();
                        this.quantum.advancedResult = this.quantum.slo;
                    } catch (err) {
                        this.quantum.slo = { error: (err && err.message) ? err.message : "slo failed" };
                    }
                },

                async loadQuantumDecypheringLab() {
                    try {
                        const res = await fetch("/agent/quantum/decyphering/lab?hours=24", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "decyphering lab failed" };
                    }
                },

                async loadQuantumRbac() {
                    try {
                        const res = await fetch("/agent/quantum/rbac", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.rbac = await res.json();
                    } catch (err) {
                        this.quantum.rbac = { role: "operator", actions: [], error: (err && err.message) ? err.message : "rbac load failed" };
                    }
                },

                async setQuantumRbacRole(role) {
                    try {
                        const res = await fetch("/agent/quantum/rbac", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ role: role }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.rbac = await res.json();
                        this.quantum.advancedResult = this.quantum.rbac;
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "rbac set failed (needs full profile/admin role)" };
                    }
                },

                async runQuantumSandbox() {
                    try {
                        const res = await fetch("/agent/quantum/sandbox/run", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                name: "dashboard_sandbox",
                                hours: 24,
                                inject_alert_code: "measurement_outcome_bias",
                                inject_severity: "warning",
                                drift_pct: 18.5,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.advancedResult = await res.json();
                    } catch (err) {
                        this.quantum.advancedResult = { error: (err && err.message) ? err.message : "sandbox simulation failed" };
                    }
                },

                async runQuantumSuperposition() {
                    const states = (this.quantum.statesText || "").split(",").map(s => s.trim()).filter(Boolean);
                    if (states.length < 2) return;
                    try {
                        const res = await fetch("/agent/quantum/superposition", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ states: states }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.lastResult = await res.json();
                        await Promise.all([this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "superposition failed" };
                    }
                },

                async runQuantumEntangle() {
                    try {
                        const res = await fetch("/agent/quantum/entangle", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ system_a: this.quantum.systemA, system_b: this.quantum.systemB }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.lastResult = await res.json();
                        await Promise.all([this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "entanglement failed" };
                    }
                },

                async runQuantumMeasure() {
                    try {
                        const res = await fetch("/agent/quantum/measure", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ measurement_basis: this.quantum.measurementBasis || "computational" }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.quantum.lastResult = await res.json();
                        await Promise.all([this.loadQuantumStatus(), this.loadQuantumAnalytics(), this.loadQuantumAlerts(), this.runQuantumDecipher(), this.loadQuantumNoc()]);
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "measurement failed" };
                    }
                },

                async createQuantumTemplate(kind) {
                    let goal = "";
                    let interval = 15;

                    if (kind === "status") {
                        goal = "quantum status";
                        interval = 15;
                    } else if (kind === "measure") {
                        goal = "quantum measure basis computational";
                        interval = 15;
                    } else if (kind === "superposition") {
                        goal = "quantum superposition: alpha,beta,gamma";
                        interval = 30;
                    } else if (kind === "entangle") {
                        goal = "quantum entangle: core,memory";
                        interval = 30;
                    } else if (kind === "decipher15") {
                        goal = "quantum decipher 24h";
                        interval = 15;
                    } else if (kind === "decipher30") {
                        goal = "quantum decipher 24h";
                        interval = 30;
                    } else {
                        return;
                    }

                    try {
                        const res = await fetch("/agent/goals/schedule", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                goal: goal,
                                interval_minutes: interval,
                                auto_approve: true,
                                session_id: this.agent.sessionId,
                                enabled: true,
                            }),
                        });
                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }
                        const data = await res.json();
                        this.quantum.lastResult = { template: kind, schedule_id: data.id, goal: goal, interval_minutes: interval };
                        this.agent.messages.push({ role: "assistant", text: "Quantum template scheduled (#" + data.id + ").", tool: this.quantum.lastResult });
                        this.$nextTick(() => this._scrollMessages());
                        await this.loadSchedules();
                    } catch (err) {
                        this.quantum.lastResult = { error: (err && err.message) ? err.message : "template scheduling failed" };
                    }
                },

                _scrollMessages() {
                    const el = document.getElementById("agentMessages");
                    if (!el) return;
                    el.scrollTop = el.scrollHeight;
                },

                async startWakePipeline() {
                    if (!this.voice.pipelineAvailable || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        this.voice.error = "Wake pipeline is unavailable on this device.";
                        return;
                    }
                    if (this.voice.pipelineListening) return;
                    this.stopVoicePlaybackOnly();
                    await this.saveWakeWordConfig();
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            audio: {
                                channelCount: 1,
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true,
                            },
                            video: false,
                        });
                        const AudioCtx = window.AudioContext || window.webkitAudioContext;
                        const audioContext = new AudioCtx();
                        if (audioContext.state === "suspended" && audioContext.resume) await audioContext.resume();
                        const source = audioContext.createMediaStreamSource(stream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);
                        const silentGain = audioContext.createGain();
                        silentGain.gain.value = 0;
                        this.voice.mediaStream = stream;
                        this.voice.audioContext = audioContext;
                        this.voice.wakeSource = source;
                        this.voice.wakeProcessor = processor;
                        this.voice.wakeSink = silentGain;
                        this.voice.wakeBuffer = [];
                        this.voice.wakeSampleCount = 0;
                        processor.onaudioprocess = async (event) => {
                            if (!this.voice.pipelineListening || this.voice.pipelineBusy) return;
                            const samples = event.inputBuffer.getChannelData(0);
                            this.voice.wakeBuffer.push(new Float32Array(samples));
                            this.voice.wakeSampleCount += samples.length;
                            const targetSamples = Math.max(2048, Math.round((audioContext.sampleRate * this.voice.chunkMs) / 1000));
                            if (this.voice.wakeSampleCount < targetSamples) return;
                            const merged = new Float32Array(this.voice.wakeSampleCount);
                            let offset = 0;
                            for (const chunk of this.voice.wakeBuffer) {
                                merged.set(chunk, offset);
                                offset += chunk.length;
                            }
                            this.voice.wakeBuffer = [];
                            this.voice.wakeSampleCount = 0;
                            this.voice.pipelineBusy = true;
                            try {
                                const pcm = new Int16Array(merged.length);
                                for (let i = 0; i < merged.length; i += 1) {
                                    const value = Math.max(-1, Math.min(1, merged[i] || 0));
                                    pcm[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
                                }
                                const bytes = new Uint8Array(pcm.buffer);
                                let binary = "";
                                for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
                                const res = await fetch("/agent/voice/wake/detect", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json", "Accept": "application/json" },
                                    body: JSON.stringify({
                                        pcm16_b64: btoa(binary),
                                        sample_rate: Math.round(audioContext.sampleRate),
                                        channels: 1,
                                        wake_word: this.voice.wakeWord || "hey jarvis",
                                        threshold: this.voice.threshold,
                                    }),
                                });
                                if (!res.ok) throw new Error(await res.text());
                                const data = await res.json();
                                this.voice.pipelineStatus = data.detected ? "wake detected" : ("listening for " + (this.voice.wakeWord || "hey jarvis"));
                                if (data.detected) {
                                    this.voice.lastWakeAt = new Date().toLocaleTimeString();
                                    if (this.voice.localVoiceEnabled && this.voice.localAvailable) {
                                        this.startLocalVoiceCapture(true);
                                    } else if (!this.voice.listening && this.voice.recognition) {
                                        try { this.voice.recognition.start(); } catch (_err) {}
                                    }
                                }
                            } catch (err) {
                                this.voice.error = (err && err.message) ? err.message : "Wake detection failed";
                            } finally {
                                this.voice.pipelineBusy = false;
                            }
                        };
                        source.connect(processor);
                        processor.connect(silentGain);
                        silentGain.connect(audioContext.destination);
                        this.voice.pipelineListening = true;
                        this.voice.pipelineStatus = "listening for " + (this.voice.wakeWord || "hey jarvis");
                        this.voice.error = null;
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Could not start wake pipeline";
                        await this.stopWakePipeline();
                    }
                },

                async stopWakePipeline() {
                    try {
                        if (this.voice.wakeProcessor) {
                            this.voice.wakeProcessor.disconnect();
                            this.voice.wakeProcessor.onaudioprocess = null;
                        }
                    } catch (_err) {}
                    try {
                        if (this.voice.wakeSource) this.voice.wakeSource.disconnect();
                    } catch (_err) {}
                    try {
                        if (this.voice.wakeSink) this.voice.wakeSink.disconnect();
                    } catch (_err) {}
                    try {
                        if (this.voice.mediaStream) {
                            this.voice.mediaStream.getTracks().forEach((track) => track.stop());
                        }
                    } catch (_err) {}
                    try {
                        if (this.voice.audioContext) await this.voice.audioContext.close();
                    } catch (_err) {}
                    this.voice.mediaStream = null;
                    this.voice.audioContext = null;
                    this.voice.wakeSource = null;
                    this.voice.wakeProcessor = null;
                    this.voice.wakeSink = null;
                    this.voice.pipelineListening = false;
                    this.voice.pipelineBusy = false;
                    this.voice.pipelineStatus = this.voice.pipelineAvailable ? "wake pipeline idle" : "wake pipeline unavailable";
                },

                encodeWavFromFloat32(float32Data, sampleRate = 16000) {
                    const length = float32Data.length;
                    const buffer = new ArrayBuffer(44 + length * 2);
                    const view = new DataView(buffer);
                    const writeString = (offset, value) => {
                        for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
                    };
                    writeString(0, "RIFF");
                    view.setUint32(4, 36 + length * 2, true);
                    writeString(8, "WAVE");
                    writeString(12, "fmt ");
                    view.setUint32(16, 16, true);
                    view.setUint16(20, 1, true);
                    view.setUint16(22, 1, true);
                    view.setUint32(24, sampleRate, true);
                    view.setUint32(28, sampleRate * 2, true);
                    view.setUint16(32, 2, true);
                    view.setUint16(34, 16, true);
                    writeString(36, "data");
                    view.setUint32(40, length * 2, true);
                    let offset = 44;
                    for (let i = 0; i < length; i += 1) {
                        const sample = Math.max(-1, Math.min(1, float32Data[i] || 0));
                        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                        offset += 2;
                    }
                    return new Blob([buffer], { type: "audio/wav" });
                },

                async startLocalVoiceCapture(autoStop = false) {
                    if (!this.voice.localVoiceEnabled || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        this.voice.error = "Local voice capture is unavailable.";
                        return;
                    }
                    if (this.voice.localRecording) return;
                    this.stopVoicePlaybackOnly();
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                        const AudioCtx = window.AudioContext || window.webkitAudioContext;
                        const audioContext = new AudioCtx({ sampleRate: 16000 });
                        if (audioContext.state === "suspended" && audioContext.resume) await audioContext.resume();
                        const source = audioContext.createMediaStreamSource(stream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);
                        const silentGain = audioContext.createGain();
                        silentGain.gain.value = 0;
                        const chunks = [];
                        processor.onaudioprocess = (event) => {
                            if (!this.voice.localRecording) return;
                            chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
                        };
                        source.connect(processor);
                        processor.connect(silentGain);
                        silentGain.connect(audioContext.destination);
                        this.voice.localStream = stream;
                        this.voice.localSink = silentGain;
                        this.voice.localRecorder = { processor, source, audioContext, autoStopTimer: null };
                        this.voice.localChunks = chunks;
                        this.voice.localRecording = true;
                        this.voice.error = null;
                        if (autoStop) {
                            this.voice.localRecorder.autoStopTimer = window.setTimeout(() => this.stopLocalVoiceCapture(true), 5000);
                        }
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Failed to start local recording";
                    }
                },

                async stopLocalVoiceCapture(submit = true) {
                    if (!this.voice.localRecording) return;
                    this.voice.localRecording = false;
                    const recorder = this.voice.localRecorder;
                    try {
                        if (recorder && recorder.autoStopTimer) window.clearTimeout(recorder.autoStopTimer);
                        if (recorder && recorder.processor) recorder.processor.disconnect();
                        if (recorder && recorder.source) recorder.source.disconnect();
                        if (this.voice.localSink) this.voice.localSink.disconnect();
                        if (this.voice.localStream) this.voice.localStream.getTracks().forEach((track) => track.stop());
                        if (recorder && recorder.audioContext) await recorder.audioContext.close();
                    } catch (_err) {}
                    const chunks = Array.isArray(this.voice.localChunks) ? this.voice.localChunks : [];
                    this.voice.localRecorder = null;
                    this.voice.localStream = null;
                    this.voice.localSink = null;
                    this.voice.localChunks = [];
                    if (!submit || !chunks.length) return;
                    let total = 0;
                    for (const chunk of chunks) total += chunk.length;
                    const merged = new Float32Array(total);
                    let offset = 0;
                    for (const chunk of chunks) {
                        merged.set(chunk, offset);
                        offset += chunk.length;
                    }
                    const wav = this.encodeWavFromFloat32(merged, 16000);
                    const formData = new FormData();
                    formData.append("file", wav, "jarvis-command.wav");
                    try {
                        const res = await fetch("/agent/voice/transcribe", { method: "POST", body: formData });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        const text = String(data.text || "").trim();
                        if (!text) return;
                        this.agent.input = text;
                        this.sendAgent();
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Local transcription failed";
                    }
                },

                initVoice() {
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    this.voice.supported = !!SpeechRecognition && !!window.speechSynthesis;
                    if (!this.voice.supported) return;
                    const recognition = new SpeechRecognition();
                    recognition.lang = "en-US";
                    recognition.continuous = true;
                    recognition.interimResults = true;
                    recognition.onstart = () => {
                        this.stopVoicePlaybackOnly();
                        this.voice.listening = true;
                        this.voice.error = null;
                    };
                    recognition.onend = () => {
                        this.voice.listening = false;
                        if (this.voice.handsFree && !this.voice.pipelineListening) {
                            window.setTimeout(() => {
                                try {
                                    if (this.voice.handsFree && this.voice.recognition && !this.voice.listening) {
                                        this.voice.recognition.start();
                                    }
                                } catch (_err) {}
                            }, 500);
                        }
                    };
                    recognition.onerror = (event) => {
                        this.voice.error = (event && event.error) ? ("Voice error: " + event.error) : "Voice recognition failed";
                    };
                    recognition.onresult = (event) => {
                        const resultIndex = event && typeof event.resultIndex === "number" ? event.resultIndex : 0;
                        const result = event && event.results && event.results[resultIndex] ? event.results[resultIndex] : null;
                        const text = result && result[0] ? String(result[0].transcript || "").trim() : "";
                        if (!text) return;
                        this.voice.transcript = text;
                        const isFinal = !!(result && result.isFinal);
                        if (!isFinal) return;
                        const lower = text.toLowerCase();
                        const wakePhrase = String(this.voice.wakeWord || "hey jarvis").trim().toLowerCase();
                        if (this.voice.wakeEnabled && !this.voice.pipelineListening) {
                            if (lower === wakePhrase || lower.startsWith(wakePhrase + " ")) {
                                this.voice.lastWakeAt = new Date().toLocaleTimeString();
                                const commandText = lower === wakePhrase ? "" : text.slice(wakePhrase.length).trim();
                                if (!commandText) {
                                    this.speakText("Listening.", true);
                                    return;
                                }
                                this.voice.transcript = commandText;
                                this.agent.input = commandText;
                                this.sendAgent();
                                return;
                            }
                            return;
                        }
                        if (lower === "brief me" || lower === "memory briefing" || lower === "jarvis briefing") {
                            this.playVoiceBriefing();
                            return;
                        }
                        this.agent.input = text;
                        this.sendAgent();
                    };
                    this.voice.recognition = recognition;
                },

                toggleVoiceListening() {
                    if (this.voice.pipelineEnabled && this.voice.pipelineAvailable) {
                        if (this.voice.pipelineListening) {
                            this.stopWakePipeline();
                            return;
                        }
                        this.stopVoicePlaybackOnly();
                        this.startWakePipeline();
                        return;
                    }
                    if (!this.voice.supported || !this.voice.recognition) return;
                    if (this.voice.listening) {
                        this.voice.recognition.stop();
                        return;
                    }
                    this.stopVoicePlaybackOnly();
                    this.voice.recognition.start();
                },

                async toggleLocalVoiceCapture() {
                    if (this.voice.localRecording) {
                        await this.stopLocalVoiceCapture(true);
                        return;
                    }
                    await this.startLocalVoiceCapture(false);
                },

                stopVoicePlaybackOnly() {
                    try {
                        window.speechSynthesis.cancel();
                        if (window.speechSynthesis.resume) window.speechSynthesis.resume();
                    } catch (_e) {}
                    try {
                        if (this.voice.localAudio) {
                            this.voice.localAudio.pause();
                            this.voice.localAudio.currentTime = 0;
                        }
                    } catch (_e) {}
                    this.voice.speaking = false;
                },

                detectVoicePreset() {
                    const profiles = Array.isArray(this.voice.profiles) ? this.voice.profiles : [];
                    const match = profiles.find((preset) => preset.voice === this.voice.ttsVoice && Number(preset.rate) === Number(this.voice.ttsRate) && Number(preset.pitch) === Number(this.voice.ttsPitch));
                    return match ? match.id : 'custom';
                },

                async applyVoicePreset(presetId) {
                    const preset = (this.voice.profiles || []).find((item) => item.id === presetId);
                    if (!preset) return;
                    this.voice.ttsVoice = preset.voice;
                    this.voice.ttsRate = preset.rate;
                    this.voice.ttsPitch = preset.pitch;
                    this.voice.activePreset = preset.id;
                    await this.saveLocalVoiceConfig();
                    this.pushMissionEvent('Voice profile ' + preset.label + ' online');
                },

                async stopVoice() {
                    try {
                        if (this.voice.recognition && this.voice.listening) this.voice.recognition.stop();
                    } catch (_e) {}
                    await this.stopLocalVoiceCapture(false);
                    await this.stopWakePipeline();
                    this.stopVoicePlaybackOnly();
                },

                async speakText(text, force = false) {
                    if ((!this.voice.autoSpeak && !force)) return;
                    const content = String(text || "").trim();
                    if (!content) return;
                    if (this.voice.localVoiceEnabled && this.voice.localAvailable) {
                        try {
                            const res = await fetch("/agent/voice/speak", {
                                method: "POST",
                                headers: { "Content-Type": "application/json", "Accept": "audio/wav" },
                                body: JSON.stringify({ text: content }),
                            });
                            if (!res.ok) throw new Error(await res.text());
                            const blob = await res.blob();
                            const url = URL.createObjectURL(blob);
                            if (this.voice.localAudio) {
                                try { this.voice.localAudio.pause(); } catch (_err) {}
                            }
                            const audio = new Audio(url);
                            audio.preload = "auto";
                            audio.volume = 1;
                            this.voice.localAudio = audio;
                            this.voice.speaking = true;
                            audio.onended = () => {
                                this.voice.speaking = false;
                                URL.revokeObjectURL(url);
                            };
                            audio.onerror = () => {
                                this.voice.speaking = false;
                                URL.revokeObjectURL(url);
                            };
                            await audio.play();
                            return;
                        } catch (err) {
                            this.voice.error = (err && err.message) ? err.message : "Local speech playback failed";
                        }
                    }
                    if (!this.voice.supported) return;
                    try {
                        window.speechSynthesis.cancel();
                        if (window.speechSynthesis.resume) window.speechSynthesis.resume();
                        const utterance = new SpeechSynthesisUtterance(content.slice(0, 900));
                        utterance.rate = 1.0;
                        utterance.pitch = 0.95;
                        utterance.onstart = () => { this.voice.speaking = true; };
                        utterance.onend = () => { this.voice.speaking = false; };
                        utterance.onerror = () => { this.voice.speaking = false; };
                        window.speechSynthesis.speak(utterance);
                    } catch (_e) {
                        this.voice.speaking = false;
                    }
                },

                async captureScreenVision() {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
                        this.agent.error = "Screen capture is not supported in this browser.";
                        return;
                    }
                    this.vision.capturing = true;
                    this.agent.error = null;
                    let stream = null;
                    try {
                        stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
                        const video = document.createElement("video");
                        video.srcObject = stream;
                        await video.play();
                        await new Promise((resolve) => setTimeout(resolve, 250));
                        const canvas = document.createElement("canvas");
                        canvas.width = video.videoWidth || 1280;
                        canvas.height = video.videoHeight || 720;
                        const ctx = canvas.getContext("2d");
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
                        if (!blob) throw new Error("Failed to capture image");
                        const imageUrl = canvas.toDataURL("image/png");
                        const formData = new FormData();
                        formData.append("file", blob, "screen-capture.png");
                        if (this.agent.sessionId) formData.append("session_id", this.agent.sessionId);
                        formData.append("source", "dashboard_screen_capture");
                        formData.append("prompt", "Describe this screen, visible text, current UI state, and anything Jarvis should notice.");
                        const res = await fetch("/agent/vision/analyze", { method: "POST", body: formData });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.vision.last = { ...data, imageUrl };
                        this.agent.messages.push({
                            role: "assistant",
                            text: "Screen analyzed: " + (data.summary || "Vision analysis complete."),
                            tool: data.details || null,
                        });
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Screen capture failed";
                    } finally {
                        if (stream) stream.getTracks().forEach((t) => t.stop());
                        this.vision.capturing = false;
                    }
                },

                async sendAgent() {
                    const message = (this.agent.input || "").trim();
                    if (!message) return;

                    this.stopVoicePlaybackOnly();
                    this.agent.error = null;
                    this.agent.sending = true;
                    this.agent.messages.push({ role: "user", text: message });
                    this.pushMissionEvent('Command sent');
                    this.agent.input = "";
                    this.$nextTick(() => this._scrollMessages());

                    try {
                        if (this.agent.streamEnabled) {
                            const response = await fetch("/agent/chat/stream", {
                                method: "POST",
                                headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
                                body: JSON.stringify({ message: message, session_id: this.agent.sessionId }),
                            });
                            if (!response.ok || !response.body) {
                                const text = await response.text();
                                throw new Error(text || ("HTTP " + response.status));
                            }

                            const decoder = new TextDecoder();
                            const reader = response.body.getReader();
                            let buffer = "";
                            let streamText = "";
                            let streamIndex = this.agent.messages.push({ role: "assistant", text: "" }) - 1;
                            let donePayload = null;

                            while (true) {
                                const { value, done } = await reader.read();
                                if (done) break;
                                buffer += decoder.decode(value, { stream: true });
                                const chunks = buffer.split("\n\n");
                                buffer = chunks.pop() || "";

                                for (const chunk of chunks) {
                                    const lines = chunk.split("\n").filter(Boolean);
                                    let eventType = "message";
                                    let dataLine = "";
                                    for (const line of lines) {
                                        if (line.startsWith("event:")) eventType = line.slice(6).trim();
                                        if (line.startsWith("data:")) dataLine += line.slice(5).trim();
                                    }
                                    if (!dataLine) continue;
                                    const payload = JSON.parse(dataLine);
                                    if (eventType === "delta") {
                                        streamText += payload.text || "";
                                        this.agent.messages[streamIndex].text = streamText;
                                        this.$nextTick(() => this._scrollMessages());
                                    } else if (eventType === "done") {
                                        donePayload = payload;
                                    } else if (eventType === "error") {
                                        throw new Error(payload.error || "Stream failed");
                                    }
                                }
                            }

                            if (donePayload) {
                                this.agent.sessionId = donePayload.session_id || this.agent.sessionId;
                                const debugPayload = {};
                                if (donePayload.plan) debugPayload.plan = donePayload.plan;
                                this.agent.messages[streamIndex].tool = Object.keys(debugPayload).length ? debugPayload : null;
                                this.speakText(this.agent.messages[streamIndex].text || "");
                            }
                            await this.loadGoalHistory();
                            this.$nextTick(() => this._scrollMessages());
                            return;
                        }

                        const res = await fetch("/agent/chat", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({ message: message, session_id: this.agent.sessionId }),
                        });

                        if (!res.ok) {
                            const text = await res.text();
                            throw new Error(text || ("HTTP " + res.status));
                        }

                        const data = await res.json();
                        const debugPayload = {};
                        if (data.tool_result) debugPayload.tool_result = data.tool_result;
                        if (data.plan) debugPayload.plan = data.plan;
                        this.agent.sessionId = data.session_id || this.agent.sessionId;
                        this.agent.messages.push({
                            role: "assistant",
                            text: data.reply || "",
                            tool: Object.keys(debugPayload).length ? debugPayload : null,
                        });
                        this.speakText(data.reply || "");
                        await this.loadGoalHistory();
                        this.$nextTick(() => this._scrollMessages());
                    } catch (err) {
                        this.agent.error = (err && err.message) ? err.message : "Request failed";
                    } finally {
                        this.agent.sending = false;
                    }
                },
            };
        }
    