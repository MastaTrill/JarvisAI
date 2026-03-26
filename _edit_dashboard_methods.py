from pathlib import Path
path = Path(r"C:\Users\willi\OneDrive\Documents\New project\JarvisAI-main\static\dashboard\index.html")
text = path.read_text(encoding="cp1252")

pairs = []

pairs.append((
'''this.loadControlConfig(), this.loadBrowserTemplates(), this.loadBrowserAuthTemplates(), this.loadTrustReport()''',
'''this.loadControlConfig(), this.loadBrowserTemplates(), this.loadBrowserAuthTemplates(), this.loadBrowserWorkflowLibrary(), this.loadTrustReport()'''
))

pairs.append((
'''                async loadBrowserAuthTemplates() {
                    try {
                        const res = await fetch("/agent/control/browser/templates/auth", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.control.authTemplates = Array.isArray(data.items) ? data.items : [];
                    } catch (_err) {
                        this.control.authTemplates = [];
                    }
                },

                applyBrowserTemplateByName() {''',
'''                async loadBrowserAuthTemplates() {
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

                applyBrowserTemplateByName() {'''
))

pairs.append((
'''                async loadLocalVoiceConfig() {
                    try {
                        const res = await fetch("/agent/voice/local", { headers: { "Accept": "application/json" } });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.voice.localAvailable = !!(data.stt_available && data.tts_available);
                        this.voice.localVoiceEnabled = !!data.enabled;
                        this.voice.sttModel = data.stt_model || "base";
                        this.voice.ttsVoice = data.tts_voice || "mb-en1";
                        this.voice.ttsRate = data.tts_rate || 145;
                        this.voice.ttsPitch = data.tts_pitch || 38;
                        this.voice.activePreset = this.detectVoicePreset();
                    } catch (_err) {
                        this.voice.localAvailable = false;
                    }
                },''',
'''                async loadLocalVoiceConfig() {
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
                },'''
))

pairs.append((
'''                async saveLocalVoiceConfig() {
                    try {
                        const res = await fetch("/agent/voice/local", {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "application/json" },
                            body: JSON.stringify({
                                enabled: !!this.voice.localVoiceEnabled,
                                stt_model: this.voice.sttModel || "base",
                                tts_voice: this.voice.ttsVoice || "mb-en1",
                                tts_rate: this.voice.ttsRate,
                                tts_pitch: this.voice.ttsPitch,
                            }),
                        });
                        if (!res.ok) throw new Error(await res.text());
                        const data = await res.json();
                        this.voice.localAvailable = !!(data.stt_available && data.tts_available);
                        this.voice.activePreset = this.detectVoicePreset();
                    } catch (err) {
                        this.voice.error = (err && err.message) ? err.message : "Failed to save local voice config";
                    }
                },''',
'''                async saveLocalVoiceConfig() {
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
                },'''
))

pairs.append((
'''                async runAutonomyMission() {
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
                },''',
'''                async runAutonomyMission() {
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
                },'''
))

pairs.append((
'''                async desktopLaunch() {
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
                },''',
'''                async desktopLaunch() {
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
                },'''
))

for old,new in pairs:
    if old not in text:
        raise SystemExit('missing method block')
    text = text.replace(old,new,1)

path.write_text(text, encoding='cp1252')
print('methods updated')
