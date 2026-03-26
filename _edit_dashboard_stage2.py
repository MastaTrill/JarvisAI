from pathlib import Path

path = Path(r"C:\Users\willi\OneDrive\Documents\New project\JarvisAI-main\static\dashboard\index.html")
text = path.read_text(encoding="cp1252")

pairs = []

pairs.append((
'''                    lastMission: null,
                    missionRuns: [],
                    running: false,
                    autoApprove: false,
                    watchers: [],
                    network: null,
                    watcherIntervalMinutes: 20,
                    watcherMinScore: 6,
                    watcherSaving: false,
                },''',
'''                    lastMission: null,
                    missionRuns: [],
                    running: false,
                    autoApprove: false,
                    retryLimit: 1,
                    watchers: [],
                    network: null,
                    watcherIntervalMinutes: 20,
                    watcherMinScore: 6,
                    watcherSaving: false,
                },'''
))

pairs.append((
'''                    browserUrl: "",
                    browserQuery: "",
                    desktopApp: "dashboard",
                    workflowUrl: "data:text/html,<html><body><h1 id='title'>Jarvis Browser Task</h1><input id='task' value='ready'><button id='go'>Go</button></body></html>",
                    workflowSteps: '[{"action":"wait_for","selector":"#title"},{"action":"extract_text","selector":"#title"}]',
                    templates: [],
                    authTemplates: [],
                    sessions: [],
                    selectedTemplateName: "",
                    selectedSessionName: "",
                    saveTemplateName: "",
                    saveSession: false,
                    sessionNotes: "",
                    confirmActions: false,
                    lastResult: null,
                    sessionHealthRunning: false,
                },''',
'''                    browserUrl: "",
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
                    lastResult: null,
                    sessionHealthRunning: false,
                },'''
))

pairs.append((
'''                    activePreset: "prime",
                    profiles: [
                        { id: "prime", label: "Prime", voice: "mb-en1", rate: 138, pitch: 32, hint: "male synth" },
                        { id: "ops", label: "Ops", voice: "mb-en1", rate: 126, pitch: 24, hint: "deep calm" },
                        { id: "arc", label: "Arc", voice: "en-gb", rate: 152, pitch: 40, hint: "future crisp" },
                    ],
                },''',
'''                    activePreset: "prime",
                    profiles: [
                        { id: "prime", label: "Prime", voice: "mb-en1", rate: 138, pitch: 32, style: "cinematic", hint: "cinematic male" },
                        { id: "ops", label: "Ops", voice: "mb-en1", rate: 148, pitch: 28, style: "operator", hint: "deep ops" },
                        { id: "arc", label: "Arc", voice: "en-us", rate: 154, pitch: 42, style: "crisp", hint: "future crisp" },
                    ],
                },'''
))

for old, new in pairs:
    if old not in text:
        raise SystemExit('missing base block')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='cp1252')
print('base updated')
