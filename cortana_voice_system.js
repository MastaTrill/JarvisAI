// --- 3D Cortana-Style Hologram Animation ---
function initHologram3D() {
    const holo = document.getElementById('hologram3d');
    if (!holo) return;
    const avatar = holo.querySelector('.holo-avatar');
    let rotX = 0, rotY = 0;
    let mouseX = 0.5, mouseY = 0.5;

    // Animate rotation
    function animate() {
        // Smoothly interpolate to mouse position
        rotY += ((mouseX - 0.5) * 60 - rotY) * 0.08;
        rotX += ((mouseY - 0.5) * 30 - rotX) * 0.08;
        avatar.style.transform = `rotateY(${rotY}deg) rotateX(${-rotX}deg)`;
        requestAnimationFrame(animate);
    }
    animate();

    // Mouse move effect
    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX / window.innerWidth;
        mouseY = e.clientY / window.innerHeight;
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initHologram3D, 1200);
});

// --- 3D Cortana-Style Hologram Styles ---
const hologramStyles = `
<style>
#hologram3d {
    position: fixed;
    left: 50%;
    top: 60px;
    transform: translateX(-50%);
    width: 240px;
    height: 380px;
    z-index: 10;
    pointer-events: none;
    perspective: 1200px;
    filter: drop-shadow(0 0 50px #4a90e2cc);
}
.holo-base {
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 140px;
    height: 8px;
    background: radial-gradient(ellipse at center, #4a90e2 40%, #6bb6ff 60%, #0ff0 100%);
    opacity: 0.8;
    border-radius: 50%;
    filter: blur(3px);
    z-index: 1;
    animation: cortanaBasePulse 3s infinite ease-in-out;
    box-shadow: 0 0 30px #4a90e2aa, 0 0 60px #4a90e244;
}
.holo-avatar {
    position: absolute;
    left: 50%;
    bottom: 40px;
    width: 140px;
    height: 240px;
    transform-style: preserve-3d;
    transform: rotateY(0deg) rotateX(0deg);
    transition: filter 0.4s ease;
    filter: drop-shadow(0 0 40px #4a90e2cc) brightness(1.3);
    z-index: 2;
}
.holo-face {
    width: 120px;
    height: 160px;
    margin: 0 auto;
    background: linear-gradient(135deg, 
        #4a90e2 0%, 
        #6bb6ff 25%, 
        #87ceeb 50%, 
        #4a90e2 75%, 
        #2e5ce6 100%);
    border-radius: 60px 60px 40px 40px;
    box-shadow: 
        0 0 50px #4a90e2cc, 
        0 0 100px #4a90e244,
        inset 0 0 30px rgba(255,255,255,0.1);
    position: relative;
    top: 20px;
    animation: cortanaPulse 4s ease-in-out infinite;
    overflow: hidden;
}
.holo-face::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 30%;
    width: 80px;
    height: 80px;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: cortanaCore 3s ease-in-out infinite;
}
.holo-face::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 40%;
    width: 40px;
    height: 40px;
    background: radial-gradient(circle, #ffffff 0%, #4a90e2 50%, transparent 100%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: cortanaEye 2s ease-in-out infinite;
    box-shadow: 0 0 20px #ffffff88;
}
.holo-scanlines {
    position: absolute;
    left: 10px; top: 10px;
    width: 120px; height: 160px;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom,
        rgba(74,144,226,0.15) 0px,
        rgba(74,144,226,0.15) 1px,
        transparent 1px,
        transparent 4px
    );
    border-radius: 60px 60px 40px 40px;
    animation: cortanaScan 2.5s linear infinite;
    opacity: 0.7;
}
@keyframes cortanaPulse {
    0%, 100% { 
        filter: brightness(1.3) hue-rotate(0deg);
        transform: scale(1);
    }
    25% { 
        filter: brightness(1.6) hue-rotate(5deg);
        transform: scale(1.02);
    }
    50% { 
        filter: brightness(1.8) hue-rotate(10deg);
        transform: scale(1.05);
    }
    75% { 
        filter: brightness(1.6) hue-rotate(5deg);
        transform: scale(1.02);
    }
}
@keyframes cortanaCore {
    0%, 100% { opacity: 0.3; transform: translateX(-50%) scale(1); }
    50% { opacity: 0.6; transform: translateX(-50%) scale(1.1); }
}
@keyframes cortanaEye {
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.2); }
}
@keyframes cortanaBasePulse {
    0%, 100% { 
        opacity: 0.8; 
        transform: translateX(-50%) scale(1);
        filter: blur(3px) hue-rotate(0deg);
    }
    50% { 
        opacity: 1; 
        transform: translateX(-50%) scale(1.1);
        filter: blur(2px) hue-rotate(10deg);
    }
}
@keyframes cortanaScan {
    0% { background-position-y: 0; opacity: 0.7; }
    50% { opacity: 0.4; }
    100% { background-position-y: 160px; opacity: 0.7; }
}
/* Speaking animation when voice is active */
.holo-avatar.speaking {
    animation: cortanaSpeaking 0.5s ease-in-out infinite alternate;
}
@keyframes cortanaSpeaking {
    0% { 
        filter: drop-shadow(0 0 40px #4a90e2cc) brightness(1.3);
        transform: scale(1);
    }
    100% { 
        filter: drop-shadow(0 0 60px #6bb6ffdd) brightness(1.8);
        transform: scale(1.08);
    }
}
@keyframes voiceIndicatorPulse {
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.05); }
}
</style>
`;
document.head.insertAdjacentHTML('beforeend', hologramStyles);

// Advanced Cortana-Jarvis Voice System
class AetheronVoiceSystem {
    constructor() {
        this.synthesis = window.speechSynthesis;
        this.voices = [];
        this.currentVoice = null;
        this.isEnabled = true;
        this.ambientMode = true;
        this.voiceConfig = {
            rate: 0.85,
            pitch: 0.9,
            volume: 0.8
        };
        this.jarvisResponses = {
            greetings: [
                "Good day, Sir. Cortana systems are online and fully operational.",
                "Welcome back, Commander. All neural networks are functioning within normal parameters.",
                "Aetheron Platform initialized successfully. Standing by for your instructions.",
                "Good to see you again. All systems are green and ready for deployment."
            ],
            training: [
                "Initiating neural network training protocol.",
                "Commencing machine learning optimization sequence.",
                "Training algorithms are now active. Monitoring progress.",
                "Neural pathways are being refined. Performance metrics updating."
            ],
            progress: [
                "Training progress is proceeding as expected, Sir.",
                "Neural network optimization continues within acceptable parameters.",
                "Model convergence detected. Accuracy metrics improving steadily.",
                "Learning algorithms are performing admirably, Commander."
            ],
            success: [
                "Training protocol completed successfully. Model performance is exemplary.",
                "Neural network optimization has concluded with outstanding results.",
                "Mission accomplished, Sir. The model has achieved optimal performance.",
                "Training sequence finalized. Results exceed expectations."
            ],
            errors: [
                "I'm afraid we've encountered a minor setback in the training protocol.",
                "Sir, there appears to be an anomaly in the data processing pipeline.",
                "System alert: Training sequence has encountered an unexpected error.",
                "Apologies, Commander. A technical difficulty has been detected."
            ],
            predictions: [
                "Analyzing data patterns and generating predictions.",
                "Running predictive algorithms. Confidence levels are high.",
                "Processing your request. Computational analysis in progress.",
                "Deploying prediction models. Results incoming momentarily."
            ],
            ambient: [
                "All systems operating within normal parameters.",
                "Monitoring network performance. Everything looks optimal.",
                "Standing by for further instructions, Sir.",
                "Neural networks are humming along nicely."
            ]
        };
        this.init();
    }

    async init() {
        await this.loadVoices();
        this.setupVoiceEvents();
        this.createVoiceUI();
        
        // Sophisticated welcome message
        setTimeout(() => {
            const greeting = this.getRandomResponse('greetings');
            this.speak(greeting, 'greeting');
        }, 2000);

        // Ambient voice notifications
        if (this.ambientMode) {
            this.startAmbientNotifications();
        }
    }

    async loadVoices() {
        return new Promise((resolve) => {
            const updateVoices = () => {
                this.voices = this.synthesis.getVoices();
                
                // Prefer British or sophisticated English voices for Jarvis effect
                const preferredVoices = [
                    'Google UK English Male',
                    'Microsoft George - English (United Kingdom)', 
                    'Alex',
                    'Daniel (Enhanced)',
                    'Microsoft David - English (United States)',
                    'Google US English Male'
                ];

                for (let preferred of preferredVoices) {
                    const voice = this.voices.find(v => 
                        v.name.toLowerCase().includes(preferred.toLowerCase()) ||
                        (v.lang.includes('en-GB') && preferred.includes('UK')) ||
                        (v.lang.includes('en-US') && preferred.includes('US'))
                    );
                    if (voice) {
                        this.currentVoice = voice;
                        break;
                    }
                }

                // Fallback to best available English voice
                if (!this.currentVoice && this.voices.length > 0) {
                    this.currentVoice = this.voices.find(v => 
                        v.lang.includes('en-GB') || 
                        v.lang.includes('en-US') || 
                        v.lang.includes('en')
                    ) || this.voices[0];
                }

                resolve();
            };

            if (this.voices.length === 0) {
                this.synthesis.addEventListener('voiceschanged', updateVoices);
            } else {
                updateVoices();
            }
        });
    }

    getRandomResponse(category) {
        const responses = this.jarvisResponses[category];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    speak(text, type = 'normal', options = {}) {
        if (!this.isEnabled || !this.synthesis) return;

        // Cancel any ongoing speech
        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        
        // Configure voice based on type - refined for Jarvis sophistication
        const configs = {
            greeting: { rate: 0.8, pitch: 0.85, volume: 0.9 },
            training: { rate: 0.85, pitch: 0.9, volume: 0.8 },
            alert: { rate: 0.9, pitch: 1.0, volume: 0.95 },
            success: { rate: 0.8, pitch: 0.95, volume: 0.85 },
            error: { rate: 0.75, pitch: 0.8, volume: 0.9 },
            ambient: { rate: 0.7, pitch: 0.9, volume: 0.6 }
        };

        const config = { ...this.voiceConfig, ...configs[type], ...options };
        
        utterance.voice = this.currentVoice;
        utterance.rate = config.rate;
        utterance.pitch = config.pitch;
        utterance.volume = config.volume;

        // Add visual effects during speech
        utterance.onstart = () => {
            this.addSpeechVisualEffects(type);
        };

        utterance.onend = () => {
            this.removeSpeechVisualEffects();
        };

        this.synthesis.speak(utterance);
    }

    addSpeechVisualEffects(type) {
        const avatar = document.querySelector('.holo-avatar');
        const face = document.querySelector('.holo-face');
        if (avatar && face) {
            avatar.classList.add('speaking');
            face.style.animation = 'cortanaSpeaking 0.4s ease-in-out infinite alternate';
            face.style.boxShadow = '0 0 80px #4a90e2, 0 0 160px #4a90e266';
        }

        // Add speaking indicator
        const indicator = document.createElement('div');
        indicator.id = 'voice-indicator';
        indicator.innerHTML = '🎙️ Cortana Online...';
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #4a90e2, #6bb6ff);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            z-index: 10001;
            animation: voiceIndicatorPulse 1s ease-in-out infinite;
            box-shadow: 0 0 20px rgba(74, 144, 226, 0.5);
        `;
        document.body.appendChild(indicator);
    }

    removeSpeechVisualEffects() {
        const avatar = document.querySelector('.holo-avatar');
        const face = document.querySelector('.holo-face');
        if (avatar && face) {
            avatar.classList.remove('speaking');
            face.style.animation = 'cortanaPulse 4s ease-in-out infinite';
            face.style.boxShadow = '0 0 50px #4a90e2cc, 0 0 100px #4a90e244';
        }

        // Remove speaking indicator
        const indicator = document.getElementById('voice-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    setupVoiceEvents() {
        // Training events with sophisticated Jarvis responses
        document.addEventListener('training-started', () => {
            const message = this.getRandomResponse('training');
            this.speak(message, 'training');
        });

        document.addEventListener('training-progress', (e) => {
            const { epoch, accuracy } = e.detail;
            if (epoch % 25 === 0) {
                const progressMsg = this.getRandomResponse('progress');
                this.speak(`${progressMsg} Epoch ${epoch} completed. Accuracy holding at ${(accuracy * 100).toFixed(1)} percent.`, 'training');
            }
        });

        document.addEventListener('training-complete', (e) => {
            const { accuracy, training_time } = e.detail;
            const successMsg = this.getRandomResponse('success');
            this.speak(`${successMsg} Final accuracy: ${(accuracy * 100).toFixed(1)} percent. Training duration: ${training_time} seconds.`, 'success');
        });

        // Prediction events
        document.addEventListener('prediction-started', () => {
            const message = this.getRandomResponse('predictions');
            this.speak(message, 'training');
        });

        document.addEventListener('prediction-complete', () => {
            this.speak('Prediction analysis concluded, Sir. Results have been compiled and are ready for your review.', 'success');
        });

        // Error events with sophisticated responses
        document.addEventListener('system-error', (e) => {
            const errorMsg = this.getRandomResponse('errors');
            this.speak(`${errorMsg} ${e.detail.message}`, 'error');
        });

        // UI interaction events
        document.addEventListener('tab-switched', (e) => {
            const tabResponses = {
                train: 'Neural training bay is now online, Sir.',
                analyze: 'Analytics suite initialized. All diagnostic tools are ready.',
                compare: 'Model comparison protocols activated. Performance metrics available.',
                deploy: 'Deployment systems standing by for your instructions.',
                data: 'Data management interface is ready for operation, Commander.'
            };
            if (this.ambientMode) {
                const response = tabResponses[e.detail.tab] || 'Interface module loaded successfully.';
                this.speak(response, 'ambient');
            }
        });
    }

    startAmbientNotifications() {
        // Sophisticated ambient messages in Jarvis style
        setInterval(() => {
            if (this.ambientMode && Math.random() > 0.8) {
                const message = this.getRandomResponse('ambient');
                this.speak(message, 'ambient');
            }
        }, 45000); // Every 45 seconds

        // Periodic status updates
        setInterval(() => {
            if (this.ambientMode && Math.random() > 0.9) {
                const statusMessages = [
                    'System diagnostics complete. All modules functioning at peak efficiency.',
                    'Neural network infrastructure operating within optimal parameters, Sir.',
                    'Computational resources available. Standing by for your next directive.',
                    'Data integrity verified. All systems green across the board.'
                ];
                const message = statusMessages[Math.floor(Math.random() * statusMessages.length)];
                this.speak(message, 'ambient');
            }
        }, 120000); // Every 2 minutes
    }

    createVoiceUI() {
        const existingPanel = document.getElementById('voice-control-panel');
        if (existingPanel) return;

        const panel = document.createElement('div');
        panel.id = 'voice-control-panel';
        panel.innerHTML = `
            <div class="voice-controls">
                <button id="voice-toggle" class="voice-btn ${this.isEnabled ? 'active' : ''}">
                    🔊 Voice
                </button>
                <button id="ambient-toggle" class="voice-btn ${this.ambientMode ? 'active' : ''}">
                    🌙 Ambient
                </button>
                <select id="voice-select" class="voice-select">
                    <option value="">Select Voice</option>
                </select>
            </div>
        `;

        const styles = `
        <style>
        #voice-control-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .voice-btn {
            background: linear-gradient(135deg, #4a90e2, #6bb6ff);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
        }
        .voice-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
        }
        .voice-btn.active {
            background: linear-gradient(135deg, #6bb6ff, #4a90e2);
            box-shadow: 0 0 20px rgba(74, 144, 226, 0.6);
        }
        .voice-select {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
            padding: 6px 10px;
            font-size: 12px;
        }
        .voice-select option {
            background: #1a1a3e;
            color: white;
        }
        </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
        document.body.appendChild(panel);

        // Populate voice select
        const voiceSelect = document.getElementById('voice-select');
        this.voices.forEach((voice, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${voice.name} (${voice.lang})`;
            if (voice === this.currentVoice) {
                option.selected = true;
            }
            voiceSelect.appendChild(option);
        });

        // Event listeners
        document.getElementById('voice-toggle').addEventListener('click', () => {
            this.toggleVoice();
        });

        document.getElementById('ambient-toggle').addEventListener('click', () => {
            this.toggleAmbient();
        });

        voiceSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.currentVoice = this.voices[e.target.value];
                this.speak('Voice updated successfully, Sir.', 'success');
            }
        });
    }

    toggleVoice() {
        this.isEnabled = !this.isEnabled;
        const btn = document.getElementById('voice-toggle');
        btn.classList.toggle('active', this.isEnabled);
        
        if (this.isEnabled) {
            this.speak('Voice system activated, Sir.', 'success');
        }
    }

    toggleAmbient() {
        this.ambientMode = !this.ambientMode;
        const btn = document.getElementById('ambient-toggle');
        btn.classList.toggle('active', this.ambientMode);
        
        if (this.ambientMode) {
            this.speak('Ambient monitoring enabled, Commander.', 'success');
        } else {
            this.speak('Ambient mode disabled, Sir.', 'success');
        }
    }

    // Public methods for manual triggering
    announceTraining(modelName) {
        this.speak(`Commencing neural network training for model: ${modelName}. Initializing optimization algorithms.`, 'training');
    }

    announceProgress(epoch, loss, accuracy) {
        if (epoch % 25 === 0) {
            this.speak(`Training progress update: Epoch ${epoch}. Loss: ${loss.toFixed(4)}. Accuracy: ${(accuracy * 100).toFixed(1)} percent.`, 'training');
        }
    }

    announceCompletion(accuracy, trainingTime) {
        this.speak(`Training protocol completed successfully, Sir. Final accuracy: ${(accuracy * 100).toFixed(1)} percent. Duration: ${trainingTime} seconds.`, 'success');
    }

    announceError(message) {
        this.speak(`I'm afraid there's been a complication, Sir. ${message}`, 'error');
    }
}

// Global voice system instance
let aetheronVoice;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        aetheronVoice = new AetheronVoiceSystem();
        window.aetheronVoice = aetheronVoice; // Make globally accessible
    }, 1000);
});
