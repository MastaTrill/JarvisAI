from pathlib import Path
path = Path(r"C:\Users\willi\OneDrive\Documents\New project\JarvisAI-main\static\dashboard\index.html")
text = path.read_text(encoding="cp1252")

pairs = [
(
'''                                <div class="mt-3 control-mini-grid">''',
'''                                <div class="mt-3 section-shell section-shell-compact">
                                    <div class="flex items-center justify-between gap-3">
                                        <div>
                                            <div class="section-kicker">Voice</div>
                                            <div class="font-semibold text-sm mt-1">Local voice presets</div>
                                        </div>
                                        <button class="cyber-btn-soft mono !px-2 !py-1" @click="loadLocalVoiceConfig()">Refresh</button>
                                    </div>
                                    <div class="mt-3 flex flex-wrap gap-2">
                                        <template x-for="preset in (voice.profiles || [])" :key="'control-voice-'+preset.id">
                                            <button class="chip-soft mono" :class="{ 'active': voice.activePreset === preset.id }" @click="applyVoicePreset(preset.id)">
                                                <span x-text="preset.label"></span>
                                            </button>
                                        </template>
                                    </div>
                                </div>
                                <div class="mt-3 control-mini-grid">'''
),
(
'''                                <div class="mt-2 compact-note mono">
                                    host control: <span x-text="control.config.host_control_available ? 'available' : 'preview only'"></span>
                                </div>''',
'''                                <div class="mt-3 control-mini-grid">
                                    <label class="control-mini-card text-xs">
                                        <div class="control-mini-head"><span>Open URL</span><span class="compact-note mono">desktop</span></div>
                                        <input class="field-shell mt-2 mono" placeholder="https://github.com/issues" x-model="control.desktopUrl" />
                                        <button class="cyber-btn mt-2 mono" @click="desktopAction('open_url')">Open</button>
                                    </label>
                                    <label class="control-mini-card text-xs">
                                        <div class="control-mini-head"><span>Type text</span><span class="compact-note mono">desktop</span></div>
                                        <input class="field-shell mt-2 mono" placeholder="type into focused app" x-model="control.desktopText" />
                                        <button class="cyber-btn mt-2 mono" @click="desktopAction('type_text')">Type</button>
                                    </label>
                                    <label class="control-mini-card text-xs">
                                        <div class="control-mini-head"><span>Hotkey</span><span class="compact-note mono">desktop</span></div>
                                        <input class="field-shell mt-2 mono" placeholder="ctrl+l" x-model="control.desktopHotkey" />
                                        <button class="cyber-btn mt-2 mono" @click="desktopAction('hotkey')">Send</button>
                                    </label>
                                </div>
                                <div class="mt-2 compact-note mono">
                                    host control: <span x-text="control.config.host_control_available ? 'available' : 'preview only'"></span>
                                </div>'''
),
(
'''                                    <div class="mt-3 flex flex-wrap gap-2">
                                        <template x-for="tpl in (control.authTemplates || []).slice(0, 6)" :key="'auth-template-chip-'+tpl.name">
                                            <button class="chip-soft mono" @click="control.selectedTemplateName = tpl.name; applyBrowserTemplateByName(); workflowModal = true; workflowStage = 1" x-text="tpl.name"></button>
                                        </template>
                                        </div>''',
'''                                    <div class="mt-3 flex flex-wrap gap-2">
                                        <template x-for="tpl in (control.authTemplates || []).slice(0, 6)" :key="'auth-template-chip-'+tpl.name">
                                            <button class="chip-soft mono" @click="control.selectedTemplateName = tpl.name; applyBrowserTemplateByName(); workflowModal = true; workflowStage = 1" x-text="tpl.name"></button>
                                        </template>
                                    </div>
                                    <div class="mt-3 section-shell section-shell-compact">
                                        <div class="section-kicker">Workflow Library</div>
                                        <div class="font-semibold text-sm mt-1">GitHub, Gmail, Notion, Figma</div>
                                        <div class="compact-note mt-1">One-tap flows for PR triage, draft review, workspace search, and recent design review.</div>
                                        <div class="mt-3 flex flex-wrap gap-2">
                                            <template x-for="tpl in (control.workflowLibrary || []).slice(0, 8)" :key="'workflow-library-'+tpl.name">
                                                <button class="chip-soft mono" @click="runWorkflowQuickAction(tpl.name)">
                                                    <span x-text="tpl.name"></span>
                                                </button>
                                            </template>
                                        </div>
                                    </div>'''
),
(
'''                        <div class="mission-last-command mono" :class="{ 'pulse': missionHud.lastCommandPulse }">
                            <span class="status-dot"></span>
                            <span>Last command</span>
                            <strong x-text="lastMissionCommand()"></strong>
                        </div>
                        <div class="center-chat-float mission-channel-compact">''',
'''                        <div class="mission-last-command mono" :class="{ 'pulse': missionHud.lastCommandPulse }">
                            <span class="status-dot"></span>
                            <span>Last command</span>
                            <strong x-text="lastMissionCommand()"></strong>
                        </div>
                        <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2">
                            <div class="panel-soft px-3 py-3 rounded-2xl text-left">
                                <div class="section-kicker">Mission</div>
                                <div class="font-semibold text-sm mt-1" x-text="missionCheckpointCard().status"></div>
                                <div class="compact-note mt-1" x-text="missionCheckpointCard().summary"></div>
                            </div>
                            <div class="panel-soft px-3 py-3 rounded-2xl text-left">
                                <div class="section-kicker">Retry</div>
                                <div class="font-semibold text-sm mt-1" x-text="missionCheckpointCard().retry"></div>
                                <div class="compact-note mt-1" x-text="missionCheckpointCard().failed"></div>
                            </div>
                            <div class="panel-soft px-3 py-3 rounded-2xl text-left">
                                <div class="section-kicker">Resume</div>
                                <div class="font-semibold text-sm mt-1" x-text="missionCheckpointCard().remaining"></div>
                                <button class="cyber-btn-soft mono !px-2 !py-1 mt-2" @click="resumeLatestMission()" :disabled="!missionCheckpointCard().canResume">Resume</button>
                            </div>
                        </div>
                        <div class="center-chat-float mission-channel-compact">'''
),
(
'''                        <div class="sideband-messages mission-transcript mission-transcript-stream space-y-2" :class="{ 'pulse': missionFeedPulse }" id="missionMessages">''',
'''                        <div class="p-4 border-b" style="border-color: var(--border);">
                            <div class="section-kicker">Checkpoint</div>
                            <div class="compact-note mt-1" x-text="missionCheckpointCard().checkpointLine"></div>
                        </div>
                        <div class="sideband-messages mission-transcript mission-transcript-stream space-y-2" :class="{ 'pulse': missionFeedPulse }" id="missionMessages">'''
),
]

for old,new in pairs:
    if old not in text:
        raise SystemExit('missing ui block')
    text = text.replace(old,new,1)

path.write_text(text, encoding='cp1252')
print('ui updated')
