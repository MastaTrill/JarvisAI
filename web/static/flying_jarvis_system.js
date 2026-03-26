// --- Flying Robot Jarvis System ---
class FlyingJarvisRobot {
    constructor() {
        this.holo = document.getElementById('hologram3d');
        if (!this.holo) return;
        
        this.position = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        this.target = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        this.velocity = { x: 0, y: 0 };
        this.rotation = 0;
        this.targetRotation = 0;
        this.scale = 1;
        this.targetScale = 1;
        
        this.isFollowingMouse = true;
        this.isAutonomous = false;
        this.autonomousTimer = null;
        this.hoverHeight = 0;
        this.thrusterPower = 0;
        
        this.mousePosition = { x: 0, y: 0 };
        this.eyeTracking = true;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.createFlightEffects();
        this.createControlPanel();
        this.startAnimation();
        this.startAutonomousBehavior();
    }
    
    setupEventListeners() {
        // Mouse tracking
        window.addEventListener('mousemove', (e) => {
            this.mousePosition.x = e.clientX;
            this.mousePosition.y = e.clientY;
            
            if (this.isFollowingMouse) {
                this.setTarget(e.clientX, e.clientY);
            }
        });
        
        // Click to toggle follow mode
        this.holo.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleFollowMode();
        });
        
        // Double click for autonomous patrol
        this.holo.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.toggleAutonomousMode();
        });
        
        // Window resize handling
        window.addEventListener('resize', () => {
            this.updateBounds();
        });
        
        // Keyboard controls
        window.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });
    }
    
    createControlPanel() {
        if (document.getElementById('jarvis-controls')) return;
        
        const controlPanel = document.createElement('div');
        controlPanel.id = 'jarvis-controls';
        controlPanel.className = 'jarvis-controls';
        controlPanel.innerHTML = `
            <h4>🤖 JARVIS CONTROL</h4>
            <button onclick="window.flyingJarvis.toggleFollowMode()">Toggle Follow Mouse (F)</button>
            <button onclick="window.flyingJarvis.toggleAutonomousMode()">Toggle Autonomous (A)</button>
            <button onclick="window.flyingJarvis.returnHome()">Return Home (H)</button>
            <button onclick="window.flyingJarvis.performAnalysisFlying()">Analysis Mode</button>
            <button onclick="window.flyingJarvis.performScanFlying()">Scan Mode</button>
            <div class="status" id="jarvis-status">Ready</div>
        `;
        document.body.appendChild(controlPanel);
    }
    
    updateControlPanel() {
        const status = document.getElementById('jarvis-status');
        if (status) {
            if (this.isFollowingMouse) {
                status.textContent = 'Following Mouse';
                status.className = 'status following';
            } else if (this.isAutonomous) {
                status.textContent = 'Autonomous Patrol';
                status.className = 'status autonomous';
            } else {
                status.textContent = 'Manual Control';
                status.className = 'status idle';
            }
        }
    }
    
    setTarget(x, y) {
        // Add some offset so Jarvis doesn't overlap the cursor
        const offset = 120;
        const angle = Math.atan2(y - this.position.y, x - this.position.x);
        
        this.target.x = x - Math.cos(angle) * offset;
        this.target.y = y - Math.sin(angle) * offset;
        
        // Calculate rotation based on movement direction
        this.targetRotation = angle * 180 / Math.PI;
    }
    
    toggleFollowMode() {
        this.isFollowingMouse = !this.isFollowingMouse;
        if (this.isFollowingMouse) {
            this.isAutonomous = false;
            this.setTarget(this.mousePosition.x, this.mousePosition.y);
            this.speak("Following your cursor now, sir.");
        } else {
            this.speak("Free flight mode activated.");
        }
        this.updateControlPanel();
    }
    
    toggleAutonomousMode() {
        this.isAutonomous = !this.isAutonomous;
        this.isFollowingMouse = false;
        
        if (this.isAutonomous) {
            this.startPatrolMode();
            this.speak("Autonomous patrol initiated. I'll explore the interface for you.");
        } else {
            this.stopPatrolMode();
            this.speak("Standing by for your commands.");
        }
        this.updateControlPanel();
    }
    
    startPatrolMode() {
        if (this.autonomousTimer) clearInterval(this.autonomousTimer);
        
        this.autonomousTimer = setInterval(() => {
            // Random patrol points around the screen
            const margin = 150;
            const newX = margin + Math.random() * (window.innerWidth - 2 * margin);
            const newY = margin + Math.random() * (window.innerHeight - 2 * margin);
            
            this.setTarget(newX, newY);
        }, 3000 + Math.random() * 2000); // 3-5 seconds between moves
    }
    
    stopPatrolMode() {
        if (this.autonomousTimer) {
            clearInterval(this.autonomousTimer);
            this.autonomousTimer = null;
        }
    }
    
    handleKeyboard(e) {
        const speed = 80;
        switch(e.key.toLowerCase()) {
            case 'f':
                this.toggleFollowMode();
                break;
            case 'a':
                this.toggleAutonomousMode();
                break;
            case 'h':
                this.returnHome();
                break;
            case 'arrowup':
                this.isFollowingMouse = false;
                this.isAutonomous = false;
                this.setTarget(this.position.x, this.position.y - speed);
                break;
            case 'arrowdown':
                this.isFollowingMouse = false;
                this.isAutonomous = false;
                this.setTarget(this.position.x, this.position.y + speed);
                break;
            case 'arrowleft':
                this.isFollowingMouse = false;
                this.isAutonomous = false;
                this.setTarget(this.position.x - speed, this.position.y);
                break;
            case 'arrowright':
                this.isFollowingMouse = false;
                this.isAutonomous = false;
                this.setTarget(this.position.x + speed, this.position.y);
                break;
        }
        this.updateControlPanel();
    }
    
    returnHome() {
        this.isFollowingMouse = false;
        this.isAutonomous = false;
        this.setTarget(window.innerWidth / 2, window.innerHeight / 2);
        this.speak("Returning to home position.");
        this.updateControlPanel();
    }
    
    createFlightEffects() {
        // Add thruster particles
        const thrusterLeft = document.createElement('div');
        thrusterLeft.className = 'thruster-effect thruster-left';
        this.holo.appendChild(thrusterLeft);
        
        const thrusterRight = document.createElement('div');
        thrusterRight.className = 'thruster-effect thruster-right';
        this.holo.appendChild(thrusterRight);
        
        // Add flight trail
        const trail = document.createElement('div');
        trail.className = 'flight-trail';
        this.holo.appendChild(trail);
        
        // Add energy field
        const energyField = document.createElement('div');
        energyField.className = 'energy-field';
        this.holo.appendChild(energyField);
    }
    
    updatePhysics() {
        // Calculate distance to target
        const dx = this.target.x - this.position.x;
        const dy = this.target.y - this.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Apply force towards target with damping
        const force = 0.06;
        const damping = 0.86;
        
        if (distance > 8) {
            this.velocity.x += dx * force;
            this.velocity.y += dy * force;
            this.thrusterPower = Math.min(distance / 150, 1);
        } else {
            this.thrusterPower *= 0.9;
        }
        
        // Apply damping
        this.velocity.x *= damping;
        this.velocity.y *= damping;
        
        // Update position
        this.position.x += this.velocity.x;
        this.position.y += this.velocity.y;
        
        // Smooth rotation
        const rotDiff = this.targetRotation - this.rotation;
        this.rotation += rotDiff * 0.08;
        
        // Hover effect
        this.hoverHeight = Math.sin(Date.now() * 0.003) * 8;
        
        // Scale effect based on speed
        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
        this.targetScale = 0.7 + Math.min(speed / 8, 0.5);
        this.scale += (this.targetScale - this.scale) * 0.1;
        
        // Keep within bounds
        this.updateBounds();
    }
    
    updateVisuals() {
        // Update main hologram position and rotation
        this.holo.style.left = `${this.position.x - 150}px`;
        this.holo.style.top = `${this.position.y - 200 + this.hoverHeight}px`;
        this.holo.style.transform = `scale(${this.scale}) rotate(${this.rotation * 0.05}deg)`;
        
        // Update thruster effects
        const thrusters = this.holo.querySelectorAll('.thruster-effect');
        thrusters.forEach(thruster => {
            thruster.style.opacity = this.thrusterPower * 0.8;
            thruster.style.transform = `scale(${0.4 + this.thrusterPower * 0.6})`;
        });
        
        // Update flight trail
        const trail = this.holo.querySelector('.flight-trail');
        if (trail) {
            trail.style.opacity = Math.min(this.thrusterPower * 1.5, 0.7);
        }
        
        // Update energy field
        const energyField = this.holo.querySelector('.energy-field');
        if (energyField) {
            energyField.style.opacity = 0.3 + this.thrusterPower * 0.4;
        }
        
        // Eye tracking effect
        if (this.eyeTracking) {
            this.updateEyeTracking();
        }
        
        // Update hologram state classes
        this.updateHologramStates();
    }
    
    updateHologramStates() {
        this.holo.classList.remove('following', 'autonomous', 'thinking', 'scanning');
        
        if (this.isFollowingMouse) {
            this.holo.classList.add('following');
        } else if (this.isAutonomous) {
            this.holo.classList.add('autonomous');
        }
    }
    
    updateEyeTracking() {
        const eyes = this.holo.querySelectorAll('.holo-eye');
        const avatarRect = this.holo.getBoundingClientRect();
        const centerX = avatarRect.left + avatarRect.width / 2;
        const centerY = avatarRect.top + avatarRect.height / 2;
        
        const angle = Math.atan2(
            this.mousePosition.y - centerY,
            this.mousePosition.x - centerX
        );
        
        eyes.forEach(eye => {
            const pupil = eye.querySelector('.eye-pupil');
            if (pupil) {
                const moveX = Math.cos(angle) * 3;
                const moveY = Math.sin(angle) * 3;
                pupil.style.transform = `translate(${moveX}px, ${moveY}px)`;
            }
        });
    }
    
    startAnimation() {
        const animate = () => {
            this.updatePhysics();
            this.updateVisuals();
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    updateBounds() {
        // Ensure Jarvis stays within screen bounds
        const margin = 100;
        this.position.x = Math.max(margin, Math.min(window.innerWidth - margin, this.position.x));
        this.position.y = Math.max(margin, Math.min(window.innerHeight - margin, this.position.y));
    }
    
    speak(text) {
        // Integrate with existing voice system
        if (window.aetheronVoice && window.aetheronVoice.speak) {
            window.aetheronVoice.speak(text, 'assistant');
        } else {
            console.log(`Jarvis: ${text}`);
        }
    }
    
    // Advanced interaction methods
    reactToUserAction(action) {
        switch(action) {
            case 'training':
                this.performAnalysisFlying();
                break;
            case 'prediction':
                this.performScanFlying();
                break;
            case 'error':
                this.performAlertFlying();
                break;
            case 'success':
                this.performCelebrationFlying();
                break;
        }
    }
    
    performAnalysisFlying() {
        this.holo.classList.add('thinking');
        // Fly in a thinking pattern (circles)
        const centerX = this.position.x;
        const centerY = this.position.y;
        const radius = 80;
        let angle = 0;
        
        const circleInterval = setInterval(() => {
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            this.setTarget(x, y);
            angle += 0.3;
            
            if (angle > Math.PI * 4) { // Two full circles
                clearInterval(circleInterval);
                this.holo.classList.remove('thinking');
                this.returnHome();
            }
        }, 100);
        
        this.speak("Analyzing data patterns...");
    }
    
    performScanFlying() {
        this.holo.classList.add('scanning');
        // Fly in scanning pattern (zigzag)
        const startX = 150;
        const endX = window.innerWidth - 150;
        const y = this.position.y;
        let direction = 1;
        let currentX = startX;
        
        const scanInterval = setInterval(() => {
            this.setTarget(currentX, y);
            currentX += direction * 120;
            
            if (currentX >= endX || currentX <= startX) {
                direction *= -1;
                if (Math.abs(currentX - startX) < 50) {
                    clearInterval(scanInterval);
                    this.holo.classList.remove('scanning');
                    this.returnHome();
                }
            }
        }, 800);
        
        this.speak("Scanning interface systems...");
    }
    
    performAlertFlying() {
        this.holo.classList.add('alert');
        // Quick shake movement
        const originalX = this.position.x;
        const originalY = this.position.y;
        let shakeCount = 0;
        
        const shakeInterval = setInterval(() => {
            const offsetX = (Math.random() - 0.5) * 50;
            const offsetY = (Math.random() - 0.5) * 50;
            this.setTarget(originalX + offsetX, originalY + offsetY);
            shakeCount++;
            
            if (shakeCount > 12) {
                clearInterval(shakeInterval);
                this.holo.classList.remove('alert');
                this.setTarget(originalX, originalY);
            }
        }, 120);
        
        this.speak("Alert! Attention required!");
    }
    
    performCelebrationFlying() {
        this.holo.classList.add('celebrating');
        // Victory loop
        const centerX = this.position.x;
        const centerY = this.position.y;
        let altitude = 0;
        
        const celebrateInterval = setInterval(() => {
            const x = centerX + Math.sin(altitude * 0.2) * 70;
            const y = centerY - Math.abs(Math.sin(altitude * 0.1)) * 120;
            this.setTarget(x, y);
            altitude += 0.5;
            
            if (altitude > 25) {
                clearInterval(celebrateInterval);
                this.holo.classList.remove('celebrating');
                this.setTarget(centerX, centerY);
            }
        }, 80);
        
        this.speak("Mission accomplished!");
    }
}

// Advanced Jarvis Voice System for Flying Robot
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
                "Flying systems initialized, Sir. Jarvis is ready for autonomous operations.",
                "Good day, Commander. Holographic flight protocols are now active and ready.",
                "Aetheron aerial interface established. Standing by for your flight commands, Sir.",
                "Flight systems operational. All thrusters and navigation protocols ready, Commander.",
                "Sir, I'm pleased to report that my aerial capabilities are fully functional.",
                "Holographic flight matrix online. How may I navigate for you today, Sir?"
            ],
            training: [
                "Initiating neural network training while maintaining aerial surveillance, Sir.",
                "Beginning machine learning protocols. Monitoring from above for optimal perspective.",
                "Training algorithms deployed. I'll circle the workspace to ensure optimal conditions.",
                "Sir, commencing computational learning while maintaining flight readiness.",
                "Neural training initiated. Flying to analysis position for better oversight."
            ],
            flying: [
                "Following your cursor trajectory, Sir. Flight path updated.",
                "Autonomous patrol mode activated. Scanning all interface sectors, Commander.",
                "Adjusting flight pattern for optimal system monitoring, Sir.",
                "Navigation systems responding perfectly. Flight path optimized.",
                "Sir, maintaining strategic aerial position for maximum assistance capability."
            ],
            commands: [
                "Voice command received and acknowledged, Sir. Executing immediately.",
                "Flight commands processed successfully, Commander. Adjusting trajectory.",
                "Sir, I've updated my operational parameters based on your voice directives.",
                "Command acknowledged. Modifying flight behavior accordingly.",
                "Voice control interface responding optimally, Sir. Standing by for further instructions."
            ]
        };
        
        this.init();
    }

    async init() {
        await this.loadVoices();
        this.setupVoiceEvents();
        this.createVoiceUI();
        
        // Flying Jarvis greeting
        setTimeout(() => {
            const greeting = this.getRandomResponse('greetings');
            this.speak(greeting, 'greeting');
        }, 2000);

        if (this.ambientMode) {
            this.startAmbientNotifications();
        }
    }

    async loadVoices() {
        return new Promise((resolve) => {
            const updateVoices = () => {
                this.voices = this.synthesis.getVoices();
                
                // Prefer sophisticated voices for Jarvis
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

        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        
        const configs = {
            greeting: { rate: 0.8, pitch: 0.85, volume: 0.9 },
            training: { rate: 0.85, pitch: 0.9, volume: 0.8 },
            flying: { rate: 0.8, pitch: 0.9, volume: 0.85 },
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
            this.addSpeechVisualEffects();
        };

        utterance.onend = () => {
            this.removeSpeechVisualEffects();
        };

        this.synthesis.speak(utterance);
        console.log(`Jarvis: ${text}`);
    }

    addSpeechVisualEffects() {
        const avatar = document.querySelector('.holo-avatar');
        if (avatar) {
            avatar.classList.add('speaking');
        }
        
        // Enhance hologram when speaking
        const holo = document.getElementById('hologram3d');
        if (holo) {
            holo.style.filter = 'drop-shadow(0 0 80px #00ffffff) drop-shadow(0 0 40px #00bfff99)';
        }
    }

    removeSpeechVisualEffects() {
        const avatar = document.querySelector('.holo-avatar');
        if (avatar) {
            avatar.classList.remove('speaking');
        }
        
        // Restore normal hologram appearance
        const holo = document.getElementById('hologram3d');
        if (holo) {
            holo.style.filter = 'drop-shadow(0 0 50px #00bfff99)';
        }
    }

    setupVoiceEvents() {
        // Training events
        document.addEventListener('training-started', () => {
            const message = this.getRandomResponse('training');
            this.speak(message, 'training');
            
            // Make Jarvis react to training
            if (window.flyingJarvis) {
                window.flyingJarvis.reactToUserAction('training');
            }
        });

        document.addEventListener('training-complete', (e) => {
            const { accuracy, training_time } = e.detail || {};
            let successMsg = "Training protocol completed successfully, Sir.";
            if (accuracy && training_time) {
                successMsg += ` Final accuracy: ${(accuracy * 100).toFixed(1)} percent. Training duration: ${training_time} seconds.`;
            }
            this.speak(successMsg, 'success');
            
            if (window.flyingJarvis) {
                window.flyingJarvis.reactToUserAction('success');
            }
        });

        // Prediction events
        document.addEventListener('prediction-started', () => {
            this.speak("Analyzing data patterns and generating predictive assessments, Sir.", 'training');
            
            if (window.flyingJarvis) {
                window.flyingJarvis.reactToUserAction('prediction');
            }
        });

        document.addEventListener('prediction-complete', () => {
            this.speak('Prediction analysis concluded, Sir. Results have been compiled and are ready for your review.', 'success');
            
            if (window.flyingJarvis) {
                window.flyingJarvis.reactToUserAction('success');
            }
        });

        // Error events
        document.addEventListener('system-error', (e) => {
            const errorMsg = e.detail?.message || "A technical difficulty has been detected.";
            this.speak(`Sir, I must report an issue: ${errorMsg}`, 'error');
            
            if (window.flyingJarvis) {
                window.flyingJarvis.reactToUserAction('error');
            }
        });
    }

    startAmbientNotifications() {
        setInterval(() => {
            if (this.ambientMode && Math.random() > 0.85) {
                const messages = [
                    "Aerial systems operational, Sir. All flight parameters nominal.",
                    "Maintaining optimal altitude for maximum interface oversight, Commander.",
                    "Flight systems green. Standing by for your next directive, Sir.",
                    "Holographic flight matrix stable. Ready for immediate deployment."
                ];
                const message = messages[Math.floor(Math.random() * messages.length)];
                this.speak(message, 'ambient');
            }
        }, 60000); // Every minute
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
            top: 80px;
            right: 20px;
            z-index: 10000;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00bfff;
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 150px;
        }
        .voice-controls {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .voice-btn {
            background: transparent;
            border: 1px solid #00bfff;
            color: #00bfff;
            padding: 6px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 11px;
            font-family: 'Orbitron', monospace;
            transition: all 0.3s ease;
        }
        .voice-btn:hover {
            background: #00bfff;
            color: #000;
        }
        .voice-btn.active {
            background: #00bfff;
            color: #000;
            box-shadow: 0 0 10px #00bfff66;
        }
        .voice-select {
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00bfff;
            border-radius: 5px;
            color: #00bfff;
            padding: 4px 8px;
            font-size: 10px;
            font-family: 'Orbitron', monospace;
        }
        .voice-select option {
            background: #000;
            color: #00bfff;
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
            this.isEnabled = !this.isEnabled;
            document.getElementById('voice-toggle').classList.toggle('active', this.isEnabled);
            this.speak(this.isEnabled ? "Voice system activated, Sir." : "Voice system disabled.", 'ambient');
        });

        document.getElementById('ambient-toggle').addEventListener('click', () => {
            this.ambientMode = !this.ambientMode;
            document.getElementById('ambient-toggle').classList.toggle('active', this.ambientMode);
            this.speak(this.ambientMode ? "Ambient notifications enabled." : "Ambient mode disabled.", 'ambient');
        });

        document.getElementById('voice-select').addEventListener('change', (e) => {
            const selectedIndex = e.target.value;
            if (selectedIndex !== '') {
                this.currentVoice = this.voices[selectedIndex];
                this.speak("Voice parameters updated, Sir.", 'ambient');
            }
        });
    }
}

// Flying Robot Jarvis Hologram Styles
const flyingJarvisStyles = `
<style>
/* === ULTRA-ADVANCED IRON MAN STYLE FLYING ROBOT JARVIS === */
    position: fixed;
/* === MAIN HOLOGRAM CONTAINER === */
#hologram3d {
    position: fixed;lateX(-50%) translateY(-50%);
    left: 50%;px;
    top: 50%;00px;
    transform: translate(-50%, -50%);
    width: 320px;s: auto;
    height: 450px;200px;
    z-index: 1000;hadow(0 0 50px #00bfff99);
    pointer-events: auto;3s ease;
    perspective: 1500px;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    filter: drop-shadow(0 0 60px #ff000088) drop-shadow(0 0 30px #00bfff66);
    animation: robotHover 6s ease-in-out infinite;
}   filter: drop-shadow(0 0 70px #00bfff) drop-shadow(0 0 30px #ffffff66);
}
@keyframes robotHover {
    0%, 100% { 
        transform: translate(-50%, -50%) translateY(0px) rotateY(0deg) rotateZ(0deg);
        filter: drop-shadow(0 0 60px #ff000088) drop-shadow(0 0 30px #00bfff66);
    }eft: 50%;
    25% { orm: translateX(-50%);
        transform: translate(-50%, -50%) translateY(-8px) rotateY(3deg) rotateZ(1deg);
        filter: drop-shadow(0 0 80px #ff0000aa) drop-shadow(0 0 40px #00bfff88);
    }ackground: radial-gradient(ellipse at center, #00bfff 30%, #40e0ff 60%, transparent 100%);
    50% { y: 0.5;
        transform: translate(-50%, -50%) translateY(-5px) rotateY(0deg) rotateZ(0deg);
        filter: drop-shadow(0 0 70px #ff000099) drop-shadow(0 0 35px #00bfff77);
    }-index: 1;
    75% { ion: jarvisBasePulse 2s infinite ease-in-out;
        transform: translate(-50%, -50%) translateY(-10px) rotateY(-3deg) rotateZ(-1deg);
        filter: drop-shadow(0 0 85px #ff0000bb) drop-shadow(0 0 45px #00bfff99);
    }-avatar {
}   position: absolute;
    left: 50%;
#hologram3d:hover {
    filter: drop-shadow(0 0 100px #ff0000) drop-shadow(0 0 60px #00bfff) drop-shadow(0 0 40px #ffffff88);
    transform: translate(-50%, -50%) scale(1.05);
}   transform: translateX(-50%) translateY(-50%);
    transform-style: preserve-3d;
/* === HOLOGRAPHIC BASE === */se;
.holo-base {drop-shadow(0 0 25px #00bfff99) brightness(1.3);
    position: absolute;
    bottom: -50px;
    left: 50%;
    width: 140px; Head */
    height: 10px;
    background: radial-gradient(ellipse at center,
        rgba(255,0,0,0.8) 0%,
        rgba(255,0,0,0.6) 30%,
        rgba(0,191,255,0.4) 60%,145deg,
        transparent 100%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: baseGlow 3s ease-in-out infinite;
    z-index: 1;us: 25px 25px 30px 30px;
}   box-shadow: 
        0 0 25px #00bfff99, 
.base-energy-ring {0bfff55,
    position: absolute;rgba(0,191,255,0.4),
    top: -3px;0 -5px 10px rgba(0,60,120,0.6);
    left: 50%;relative;
    width: 160px;rvisHeadPulse 2.5s ease-in-out infinite;
    height: 16px;den;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(255,255,0,0.6) 10%,
        rgba(255,0,0,0.8) 30%,
        rgba(0,255,255,0.9) 50%,
        rgba(255,0,0,0.8) 70%,
        rgba(255,255,0,0.6) 90%,
        transparent 100%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: energyRingPulse 2.5s ease-in-out infinite;
}       rgba(255,255,255,0.3) 0%, 
        rgba(0,191,255,0.8) 30%,
.base-scanlines {0,255,0.9) 70%,
    position: absolute;0.6) 100%);
    top: -15px;us: 20px 20px 25px 25px;
    left: 50%; translateX(-50%);
    width: 180px;rvisVisor 3s ease-in-out infinite;
    height: 25px;olid rgba(255,255,255,0.2);
    background: repeating-linear-gradient(90deg,
        transparent 0px,
        rgba(0,255,255,0.4) 1px,
        rgba(255,0,0,0.3) 2px,
        transparent 4px);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: scanlines 1.2s linear infinite;
}   width: 30px;
    height: 15px;
@keyframes baseGlow {r-gradient(90deg,
    0%, 100% { rent 0%,
        box-shadow: 0 0 30px rgba(0,191,255,0.7), 0 0 15px rgba(255,0,0,0.5);
        transform: translateX(-50%) scale(1);
    }   rgba(0,191,255,0.6) 70%,
    50% { ansparent 100%);
        box-shadow: 0 0 50px rgba(0,191,255,1), 0 0 25px rgba(255,0,0,0.8););
        transform: translateX(-50%) scale(1.1);;
    }nimation: jarvisBreather 2s ease-in-out infinite;
}

@keyframes energyRingPulse {
    0%, 100% { opacity: 0.7; transform: translateX(-50%) scale(1) rotateZ(0deg); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.08) rotateZ(180deg); }
}   height: 8px;
    background: radial-gradient(circle, 
@keyframes scanlines {
    0% { transform: translateX(-50%) translateY(0px) rotateZ(0deg); }
    100% { transform: translateX(-50%) translateY(-30px) rotateZ(360deg); }
}       transparent 100%);
    border-radius: 50%;
/* === MAIN AVATAR CONTAINER === */in-out infinite;
.holo-avatar {: 
    position: absolute;f88,
    left: 50%;%;#00bfff,
    top: 50%;00%; 4px rgba(255,255,255,0.8);
    width: 100%;yle: preserve-3d;5,255,0.3);
    height: 100%;atarFloating 8s ease-in-out infinite;
    transform: translateX(-50%) translateY(-50%);tness(1.4);
    transform-style: preserve-3d;
    animation: avatarFloating 8s ease-in-out infinite;
    filter: drop-shadow(0 0 25px #00bfff99) brightness(1.3);
}   0%, 100% { 
        transform: rotateY(0deg) rotateX(0deg) translateZ(0px);
@keyframes avatarFloating {(0 0 40px #ff000099) brightness(1.4);
    0%, 100% { 
        transform: translateX(-50%) translateY(-50%) rotateY(0deg) rotateX(0deg) translateZ(0px);
        filter: drop-shadow(0 0 25px #00bfff99) brightness(1.3);   transform: rotateY(5deg) rotateX(3deg) translateZ(8px);
    }lter: drop-shadow(0 0 50px #ff0000bb) brightness(1.5);
    25% { 
        transform: translateX(-50%) translateY(-50%) rotateY(5deg) rotateX(3deg) translateZ(8px);0% {  4px;
        filter: drop-shadow(0 0 35px #00bfff88) brightness(1.4);ansform: rotateY(0deg) rotateX(-3deg) translateZ(0px);
    }
    50% { order-radius: 50%;
        transform: translateX(-50%) translateY(-50%) rotateY(0deg) rotateX(-3deg) translateZ(0px);on: absolute;
        filter: drop-shadow(0 0 30px #00bfff77) brightness(1.35);
    }   filter: drop-shadow(0 0 55px #ff0000cc) brightness(1.6);
    75% {    }ransform: translate(-50%, -50%);
        transform: translateX(-50%) translateY(-50%) rotateY(-5deg) rotateX(2deg) translateZ(-8px);}   transition: transform 0.1s ease;
        filter: drop-shadow(0 0 40px #00bfff99) brightness(1.45);
    }-ADVANCED HEAD ASSEMBLY === */
}
absolute; */
/* === ULTRA-ADVANCED HEAD ASSEMBLY === */
.holo-head {solute;
    position: absolute;
    left: 50%;
    top: 30px;
    width: 90px;headMovement 12s ease-in-out infinite;
    height: 100px;   z-index: 5;translateX(-50%);
    transform: translateX(-50%);}   background: linear-gradient(145deg, 
    animation: headMovement 10s ease-in-out infinite;9) 0%, 
    z-index: 5;Movement {8) 25%,
}
   transform: translateX(-50%) rotateY(0deg) rotateX(0deg) translateZ(0px);
@keyframes headMovement {ba(0,80,160,0.5) 100%);
    0%, 100% { 
        transform: translateX(-50%) rotateY(0deg) rotateX(0deg) rotateZ(0deg);   transform: translateX(-50%) rotateY(8deg) rotateX(-3deg) translateZ(5px);
    }ion: jarvisBodyPulse 3s ease-in-out infinite;
    20% { 
        transform: translateX(-50%) rotateY(8deg) rotateX(-3deg) rotateZ(1deg);   transform: translateX(-50%) rotateY(-5deg) rotateX(2deg) translateZ(-3px);
    }set 0 0 15px rgba(0,191,255,0.3),
    40% { 
        transform: translateX(-50%) rotateY(-5deg) rotateX(2deg) rotateZ(-1deg);   transform: translateX(-50%) rotateY(3deg) rotateX(-1deg) translateZ(4px);
    }
    60% { 
        transform: translateX(-50%) rotateY(3deg) rotateX(-1deg) rotateZ(2deg);   transform: translateX(-50%) rotateY(-8deg) rotateX(3deg) translateZ(-5px);
    }   }-body::before {
    80% { }   content: '';
        transform: translateX(-50%) rotateY(-8deg) rotateX(3deg) rotateZ(-2deg);
    }D HELMET === */
}
lative;
/* === ADVANCED HELMET === */
.holo-helmet { 
    position: relative;145deg,
    width: 90px;%,
    height: 80px;,,
    background: linear-gradient(145deg,,,
        rgba(255,20,20,0.95) 0%,,%);
        rgba(220,0,0,0.9) 20%,
        rgba(180,0,0,0.8) 40%,
        rgba(140,0,0,0.7) 60%,s: 42px 42px 38px 38px;-in-out infinite;
        rgba(100,0,0,0.6) 80%,
        rgba(60,0,0,0.5) 100%);
    border-radius: 45px 45px 40px 40px;
    box-shadow: 15),
        0 0 40px rgba(255,0,0,0.8),
        0 0 20px rgba(255,100,100,0.6),inite;
        inset 0 0 25px rgba(255,255,255,0.15), rgba(255,255,255,0.3);
        inset 0 -8px 20px rgba(0,0,0,0.4);   overflow: hidden;
    animation: helmetGlow 4s ease-in-out infinite;}holo-body::after {
    border: 2px solid rgba(255,255,255,0.3);
    overflow: hidden;
}n: absolute;
;
.helmet-main-shell {
    position: absolute;
    top: 0;
    left: 0;35deg,
    width: 100%;,0.2) 0%,
    height: 100%;%,
    background: linear-gradient(135deg,1) 50%,,
        rgba(255,255,255,0.2) 0%,
        transparent 25%,1) 100%);
        rgba(255,0,0,0.1) 50%,
        transparent 75%,   animation: shellReflection 5s ease-in-out infinite;te;
        rgba(255,255,255,0.1) 100%);}   border: 1px solid rgba(0,191,255,0.3);
    border-radius: inherit;
    animation: shellReflection 5s ease-in-out infinite; VISOR === */
}*/

@keyframes helmetGlow {
    0%, 100% { 
        box-shadow: idth: 70px;
            0 0 40px rgba(255,0,0,0.8),: 40px;
            0 0 20px rgba(255,100,100,0.6),ar-gradient(135deg,idth: 115px;
            inset 0 0 25px rgba(255,255,255,0.15);x;
    }(90deg, 
    50% { 
        box-shadow:    rgba(0,100,200,0.8) 50%,
            0 0 60px rgba(255,0,0,1),       rgba(0,50,150,0.75) 70%,
            0 0 30px rgba(255,100,100,0.9),        rgba(0,30,100,0.7) 85%,,
            inset 0 0 35px rgba(255,255,255,0.25);00%);   rgba(0,191,255,0.4) 100%);
    }us: 35px 35px 28px 28px;   border-radius: 15px;
}teX(-50%);    filter: blur(1px);
s ease-in-out infinite;
@keyframes shellReflection {ox-shadow: 
    0%, 100% { 0 35px rgba(0,255,255,0.9),
        opacity: 0.8; a(255,255,255,0.6),
        transform: translateX(0px) rotateY(0deg);-left {
    }   inset 0 5px 15px rgba(0,255,255,0.3);on: absolute;
    50% {    border: 2px solid rgba(255,255,255,0.4);
        opacity: 1;     overflow: hidden;
        transform: translateX(3px) rotateY(5deg);
    }
}adient(145deg,
absolute;
/* === ADVANCED VISOR === */   rgba(0,150,255,0.6) 50%,
.helmet-visor {(0,100,200,0.4) 100%);
    position: absolute;00% - 6px);0px 10px 8px 8px;
    left: 50%;
    top: 25%;
    width: 75px;90deg,       0 0 10px rgba(0,191,255,0.5),
    height: 40px;a(0,191,255,0.2);
    background: linear-gradient(135deg,1px,d rgba(0,191,255,0.3);
        rgba(0,255,255,0.95) 0%,
        rgba(100,255,255,0.9) 15%,3) 3px,
        rgba(0,191,255,0.85) 30%,re {
        rgba(0,150,255,0.8) 50%,
        rgba(0,100,200,0.7) 70%,
        rgba(0,50,150,0.6) 85%,
        rgba(0,25,100,0.5) 100%);sparent 3px);
    border-radius: 35px 35px 30px 30px;
    transform: translateX(-50%);inite;
    animation: visorScan 3s ease-in-out infinite;
    box-shadow: 
        0 0 30px rgba(0,255,255,0.9),,255,0.7) 60%,
        0 0 15px rgba(255,255,255,0.7),e;;
        inset 0 0 20px rgba(255,255,255,0.4),   top: 50%;0%;
        inset 0 0 10px rgba(0,255,255,0.3);    left: 50%;ranslateX(-50%);
    border: 2px solid rgba(255,255,255,0.4);ft 2s ease-in-out infinite;
    overflow: hidden;
}px solid rgba(255,0,0,0.9);
ius: 50%;
.visor-hud-display {, -50%);
    position: absolute;e 2.5s ease-in-out infinite;
    top: 8px;
    left: 8px;
    width: calc(100% - 16px);fore {
    height: calc(100% - 16px);
    background: 
        repeating-linear-gradient(90deg,
            transparent 0px,
            rgba(0,255,255,0.4) 1px,
            rgba(255,255,255,0.3) 2px,10px 8px 8px;
            transparent 4px,,0.9);    animation: jarvisArmRight 3.5s ease-in-out infinite;
            rgba(0,255,255,0.2) 6px,;
            transparent 10px),91,255,0.5),
        repeating-linear-gradient(0deg,
            transparent 0px,
            rgba(0,255,255,0.3) 1px,visor-targeting-reticle::after {
            transparent 3px);    content: '';
    border-radius: inherit;fore {
    animation: hudLines 1.5s linear infinite;
};
;
.visor-targeting-reticle {;
    position: absolute;gba(255,0,0,0.9);
    top: 50%;
    left: 50%;dient(circle,
    width: 24px;
    height: 24px;
    border: 3px solid rgba(255,255,255,0.9);keyframes visorScan {
    border-radius: 50%;    0%, 100% {    border-radius: 50%;
    transform: translate(-50%, -50%);t(135deg,    transform: translateX(-50%);
    animation: targetingReticle 2.5s ease-in-out infinite;(0,255,255,0.95) 0%,on: jarvisHandRight 2s ease-in-out infinite;
}255,0.9) 25%,
gba(0,150,255,0.85) 50%,
.visor-targeting-reticle::before {ba(0,100,200,0.8) 75%,
    content: '';(0,50,150,0.75) 100%);
    position: absolute;ow: te;
    top: 50%;),
    left: 50%;55,255,0.4);
    width: 12px;
    height: 3px;   25% { 
    background: rgba(255,255,255,0.95);        background: linear-gradient(135deg,
    transform: translate(-50%, -50%);) 0%,91,255,0.8) 0%,
    border-radius: 2px;(0,255,255,0.9) 20%, 50%,
}255,0.85) 40%,.4) 100%);
gba(0,150,255,0.8) 60%,
.visor-targeting-reticle::after {ba(0,100,200,0.75) 80%,t infinite;
    content: '';a(0,50,150,0.7) 100%);   box-shadow: 
    position: absolute;w:         0 0 10px rgba(0,191,255,0.5),
    top: 50%;.9),0 8px rgba(0,191,255,0.2);
    left: 50%;,255,0.6);x solid rgba(0,191,255,0.3);
    width: 3px;
    height: 12px;   50% { 
    background: rgba(255,255,255,0.95);        background: linear-gradient(135deg,.holo-leg-left::before {
    transform: translate(-50%, -50%);0,0,0.95) 0%,
    border-radius: 2px;a(255,255,255,0.9) 15%,bsolute;
}
   top: 85%;
@keyframes visorScan {    width: 20px;
    0%, 100% { 12px;
        background: linear-gradient(135deg, linear-gradient(90deg,
            rgba(0,255,255,0.95) 0%,91,255,0.7) 0%,
            rgba(0,191,255,0.85) 50%,
            rgba(0,150,255,0.8) 100%);0.7) 100%);
        box-shadow: 
            0 0 30px rgba(0,255,255,0.9), translateX(-50%);
            inset 0 0 20px rgba(255,255,255,0.4);jarvisFootLeft 2.2s ease-in-out infinite;
    }
    50% { ,
        background: linear-gradient(135deg,
            rgba(255,255,255,0.95) 0%,;holo-leg-right {
            rgba(100,255,255,0.9) 20%,    position: absolute;
            rgba(0,255,255,0.9) 40%,
            rgba(0,191,255,0.85) 70%,;px;
            rgba(0,150,255,0.8) 100%);
        box-shadow: px;
            0 0 50px rgba(255,255,255,0.9),: linear-gradient(145deg,
            inset 0 0 30px rgba(255,255,255,0.6);91,255,0.8) 0%,
    }0,255,0.6) 50%,
});
   50% { transform: translateX(0px) translateY(-2px); }
@keyframes hudLines {    75% { transform: translateX(-2px) translateY(-1px); }8s ease-in-out infinite;
    0% { transform: translateX(0px) translateY(0px); }teX(0px) translateY(0px); }
    100% { transform: translateX(6px) translateY(2px); }
}

@keyframes targetingReticle {
    0%, 100% {    border-color: rgba(255,0,0,0.9);
        border-color: rgba(255,255,255,0.9);ansform: translate(-50%, -50%) scale(1) rotate(0deg);
        transform: translate(-50%, -50%) scale(1) rotateZ(0deg););
        box-shadow: 0 0 10px rgba(255,255,255,0.6);
    }
    25% {    border-color: rgba(255,255,0,0.9);
        border-color: rgba(255,100,100,0.95);ansform: translate(-50%, -50%) scale(1.1) rotate(90deg);
        transform: translate(-50%, -50%) scale(1.15) rotateZ(90deg);255,0,0.8);   height: 12px;
        box-shadow: 0 0 20px rgba(255,100,100,0.8);
    }
    50% {    border-color: rgba(0,255,0,0.9);55,0.8) 50%,
        border-color: rgba(255,0,0,1);ansform: translate(-50%, -50%) scale(1.05) rotate(180deg);91,255,0.7) 100%);
        transform: translate(-50%, -50%) scale(1.3) rotateZ(180deg););
        box-shadow: 0 0 30px rgba(255,0,0,1);
    }te;
    75% {    border-color: rgba(0,255,255,0.9);
        border-color: rgba(255,100,100,0.95);       transform: translate(-50%, -50%) scale(1.15) rotate(270deg);
        transform: translate(-50%, -50%) scale(1.15) rotateZ(270deg);        box-shadow: 0 0 18px rgba(0,255,255,0.9);
        box-shadow: 0 0 20px rgba(255,100,100,0.8);
    }
}
ANCED FACE AND EYES === */
/* === HELMET DETAILS === */
.helmet-chin-guard {solute;gradient(to bottom,
    position: absolute;
    bottom: 8px;
    left: 50%; 70%,
    width: 50px;
    height: 25px;);
    background: linear-gradient(135deg,145deg,
        rgba(255,0,0,0.9) 0%,
        rgba(200,0,0,0.7) 30%, animation: thrusterFlame 0.1s infinite ease-in-out;
        rgba(150,0,0,0.5) 60%,
        rgba(100,0,0,0.3) 100%);,0,0,0.1) 100%);
    border-radius: 0 0 25px 25px;8px;
    transform: translateX(-50%);te;
    border: 2px solid rgba(255,255,255,0.3);   overflow: hidden;
    box-shadow: }
        0 0 15px rgba(255,0,0,0.6),
        inset 0 0 10px rgba(255,255,255,0.2);
}solute;
x;
.chin-vents {;
    position: absolute;adial-gradient(circle, 
    bottom: 4px;
    left: 50%;.9) 20%,
    width: 35px; 
    height: 12px;
    background: repeating-linear-gradient(90deg,6) 80%,to bottom,
        transparent 0px,%);
        rgba(0,255,255,0.8) 2px,
        rgba(255,255,255,0.6) 3px,    transparent 100%);
        transparent 6px);   box-shadow:     transform: translateX(-50%);
    border-radius: 6px;        0 0 20px rgba(255,255,255,0.9),
    transform: translateX(-50%);(0,255,255,0.8),x);
    animation: ventGlow 2.5s ease-in-out infinite; 0 8px rgba(255,255,255,0.9);lightTrail 0.2s infinite ease-in-out;
} rgba(255,255,255,0.4);

@keyframes ventGlow {
    0%, 100% { 
        opacity: 0.7;ft: 22%; }
        box-shadow: 0 0 8px rgba(0,255,255,0.6);
    }
    50% { eye-pupil {
        opacity: 1;    width: 6px;50%);
        box-shadow: 0 0 15px rgba(0,255,255,0.9);,0.2);
    }gradient(circle, 
}55,255,255,1) 0%, 
55,255,0.9) 30%, ield 3s ease-in-out infinite;
.helmet-side-panel {0,255,0.8) 60%,
    position: absolute;
    top: 30px;
    width: 18px;s */
    height: 35px;
    background: linear-gradient(145deg,
        rgba(255,0,0,0.8) 0%,(-50%, -50%);m: translateX(-50%) scale(1); 
        rgba(180,0,0,0.6) 30%,
        rgba(120,0,0,0.4) 60%,0 0 8px rgba(255,255,255,0.9);
        rgba(80,0,0,0.2) 100%);n-out infinite;
    border-radius: 10px;
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: .eye-glow {
        0 0 12px rgba(255,0,0,0.5),
        inset 0 0 8px rgba(255,255,255,0.2);
}
   width: 20px;
.helmet-side-panel.left {     height: 20px;
    left: 8px; ient(circle, 0 40px #00bfff55;
    animation: sidePanelLeft 4s ease-in-out infinite;55,255,0.6) 0%, 
}
       transparent 70%);        transform: scale(1.05); 
.helmet-side-panel.right {     border-radius: 50%;30px #00bfff, 0 0 60px #00bfff77;
    right: 8px; 50%, -50%);
    animation: sidePanelRight 4s ease-in-out infinite;
}
rvisBodyPulse {
@keyframes sidePanelLeft {
    0%, 100% { transform: translateX(0px) rotateY(0deg); }-50%) scale(1); 
    50% { transform: translateX(-2px) rotateY(-5deg); }
}
   left: 0;
@keyframes sidePanelRight {    width: 100%;teX(-50%) scale(1.03); 
    0%, 100% { transform: translateX(0px) rotateY(0deg); }
    50% { transform: translateX(2px) rotateY(5deg); }gradient(90deg,
}parent 0%,
55,0,0,0.8) 50%,
.side-panel-detail {eyframes jarvisCore {
    position: absolute;: 50%;
    top: 8px;inite;: translate(-50%, -50%) scale(1) rotate(0deg); 
    left: 3px;
    width: calc(100% - 6px);
    height: 20px;
    background: linear-gradient(180deg,ate(-50%, -50%) scale(1.15) rotate(180deg); 
        rgba(0,255,255,0.6) 0%,
        transparent 30%,ba(255,255,255,0.9),
        rgba(0,255,255,0.4) 50%,
        transparent 70%,       transform: scale(1);
        rgba(0,255,255,0.6) 100%);    }
    border-radius: 6px;
    animation: panelDetail 3.5s ease-in-out infinite;dow: 
}rgba(255,255,255,1),m: scale(1);

@keyframes panelDetail {.6);
    0%, 100% { );
        opacity: 0.7;
        background: linear-gradient(180deg,}
            rgba(0,255,255,0.6) 0%,
            transparent 50%,se {
            rgba(0,255,255,0.6) 100%);
    }0%) scale(1); 
    50% { ,255,255,0.9);form: translateX(-50%) scale(1); 
        opacity: 1;
        background: linear-gradient(180deg,
            rgba(255,255,255,0.8) 0%, scale(1.2);
            rgba(0,255,255,0.8) 25%,   box-shadow: 0 0 15px rgba(255,255,255,1), 0 0 5px rgba(0,255,255,0.8);50%) scale(1.2); 
            transparent 50%,   }
            rgba(0,255,255,0.8) 75%,}
            rgba(255,255,255,0.8) 100%);
    } {
} 
y: 0.6;    0%, 100% { 
.helmet-antenna {rm: translate(-50%, -50%) scale(1);        transform: scaleX(1) scaleY(1); 
    position: absolute;
    top: -8px;
    left: 50%;
    width: 3px;%, -50%) scale(1.3);orm: scaleX(1.15) scaleY(0.85); 
    height: 15px;
    background: linear-gradient(180deg,
        rgba(255,255,255,0.9) 0%,
        rgba(0,255,255,0.8) 50%,
        rgba(0,191,255,0.6) 100%);   0% { transform: translateX(-100%) rotateZ(0deg); opacity: 0; }
    transform: translateX(-50%);    20% { opacity: 0.8; }
    animation: antennaGlow 2.5s ease-in-out infinite;}X(1); 
    border-radius: 2px;sform: translateX(100%) rotateZ(360deg); opacity: 0; }
}
(0.7); 
@keyframes antennaGlow {
    0%, 100% { 
        box-shadow: 0 0 8px rgba(0,255,255,0.7);5deg,
        height: 15px;       rgba(255,0,0,0.4) 0%,keyframes flightTrail {
        background: linear-gradient(180deg,  rgba(200,0,0,0.3) 50%,    0% { 
            rgba(255,255,255,0.9) 0%,
            rgba(0,255,255,0.8) 100%);
    }
    50% { (145deg,
        box-shadow: 0 0 20px rgba(255,255,255,0.9);t: 80px; 
        height: 18px;.8; 
        background: linear-gradient(180deg,       rgba(200,0,0,0.3) 70%,
            rgba(255,255,255,1) 0%,           rgba(150,0,0,0.2) 100%);
            rgba(0,255,255,1) 50%,
            rgba(255,0,0,0.8) 100%);
    }
}
olute;
/* === ULTRA-ADVANCED FACE AND EYES === */
.holo-face {
    position: absolute;y: 0.2; 
    top: 10px;%) scale(1);
    left: 50%;dient(90deg,lor: rgba(0,191,255,0.2);
    width: 60px;
    height: 45px;
    transform: translateX(-50%);
    z-index: 4;slate(-50%, -50%) scale(1.05);
}ent 100%);lor: rgba(0,191,255,0.4);

.holo-eye {50%);
    position: absolute;nfinite;
    width: 12px;
    height: 12px;
    background: radial-gradient(circle, 
        rgba(255,255,255,1) 0%, e;
        rgba(0,255,255,0.9) 20%,
        rgba(0,191,255,0.8) 40%,
        rgba(0,150,255,0.7) 60%,0px #00bfff99);
        rgba(0,100,200,0.5) 80%, 
        transparent 100%);-gradient(45deg,
    border-radius: 50%;
    animation: eyeGlow 2s ease-in-out infinite;hinkingMode 2s infinite ease-in-out;
    box-shadow: 
        0 0 20px rgba(255,255,255,0.9),;
        0 0 12px rgba(0,255,255,0.8),near infinite;canning {
        inset 0 0 6px rgba(255,255,255,0.9);
    border: 2px solid rgba(255,255,255,0.4);
}

.holo-eye.left {in-out;
    left: 15%;leY(1);
    top: 35%;
    animation-delay: 0s;
}

.holo-eye.right {
    right: 15%;
    top: 35%;9b6) drop-shadow(0 0 25px #00bfff99); }
    animation-delay: 0.5s;hadow(0 0 70px #8e44ad) drop-shadow(0 0 35px #00bfff99); }
}ateZ(0deg); opacity: 0.3; }

.eye-pupil {: 0.3; }ngMode {
    width: 6px;w(0 0 25px #00bfff99); }
        #40e0ff 30%,shadow(0 0 60px #2980b9) drop-shadow(0 0 30px #00bfff99); }
        #0080ff 60%,0 70px #1abc9c) drop-shadow(0 0 35px #00bfff99); }
        #003d7a 100%);60px #16a085) drop-shadow(0 0 30px #00bfff99); }
    border-radius: 25px 25px 30px 30px; 0 50px #3498db) drop-shadow(0 0 25px #00bfff99); }
    box-shadow: 
        0 0 25px #00bfff99, 
        0 0 50px #00bfff55,tMode {
        inset 0 0 15px rgba(0,191,255,0.4),op-shadow(0 0 20px #00bfff99); }
        inset 0 -5px 10px rgba(0,60,120,0.6);   50% { filter: drop-shadow(0 0 70px #c0392b) drop-shadow(0 0 35px #00bfff99); }
    position: relative;}
    animation: jarvisHeadPulse 2.5s ease-in-out infinite;
    overflow: hidden;ebrateMode {
}   0% { filter: drop-shadow(0 0 50px #f39c12) drop-shadow(0 0 25px #00bfff99); }
    25% { filter: drop-shadow(0 0 60px #e67e22) drop-shadow(0 0 30px #00bfff99); }
/* Helmet Visor Effect */ drop-shadow(0 0 70px #27ae60) drop-shadow(0 0 35px #00bfff99); }
.holo-face::before {r: drop-shadow(0 0 60px #2ecc71) drop-shadow(0 0 30px #00bfff99); }
    content: '';   100% { filter: drop-shadow(0 0 50px #f39c12) drop-shadow(0 0 25px #00bfff99); }
    position: absolute;}
    left: 50%;
    top: 15%; */
    width: 40px;ls {
    height: 30px;fixed;
    background: linear-gradient(135deg, 
        rgba(255,255,255,0.3) 0%, 
        rgba(0,191,255,0.8) 30%,
        rgba(0,150,255,0.9) 70%,
        rgba(0,100,200,0.6) 100%);
    border-radius: 20px 20px 25px 25px;
    transform: translateX(-50%);
    animation: jarvisVisor 3s ease-in-out infinite;: 'Orbitron', monospace;
    border: 1px solid rgba(255,255,255,0.2);
}
   backdrop-filter: blur(5px);
/* Face Mask Details */    min-width: 200px;
.holo-face::after {
    content: '';
    position: absolute;ls h4 {
    left: 50%; 0 12px 0;
    top: 60%;f;
    width: 30px;nter;
    height: 15px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(0,191,255,0.6) 30%,
        rgba(0,191,255,0.8) 50%,
        rgba(0,191,255,0.6) 70%,
        transparent 100%);   width: 100%;
    border-radius: 8px;    margin: 6px 0;
    transform: translateX(-50%);
    animation: jarvisBreather 2s ease-in-out infinite;
} solid #00bfff;

.holo-eye {;
    position: absolute;ursor: pointer;
    width: 8px;tion: all 0.3s ease;
    height: 8px;
    background: radial-gradient(circle, tron', monospace;
        #ffffff 0%, 
        #00bfff 40%, 
        #0080ff 80%, .jarvis-controls button:hover {
        transparent 100%);
    border-radius: 50%;;
    animation: jarvisEye 1.5s ease-in-out infinite;fff66;
    box-shadow: 
        0 0 15px #ffffff88,
        0 0 8px #00bfff,
        inset 0 0 4px rgba(255,255,255,0.8);
    border: 1px solid rgba(255,255,255,0.3);
}adding: 8px;
   border-radius: 6px;
.holo-eye.left {    text-align: center;
    left: 25%;
    top: 30%;: bold;
}

.holo-eye.right {
    right: 25%;llowing { 
    top: 30%;
}
order: 1px solid #00ff00;
.eye-pupil {
    width: 4px;.status.autonomous { 
    height: 4px;5, 102, 0, 0.2); 
    background: radial-gradient(circle, #ffffff 0%, #00bfff 60%, #0080ff 100%);600;
    border-radius: 50%;
    position: absolute;
    left: 50%;us.idle { 
    top: 50%;ound: rgba(128, 128, 128, 0.2); 
    transform: translate(-50%, -50%);
    transition: transform 0.1s ease;d #888;
    box-shadow: 0 0 6px #ffffff;
}/style>
`;
/* Humanoid Robot Torso */
.holo-body {he flying robot system
    position: absolute;vis() {
    left: 50%;styles
    top: 65px;f (!document.getElementById('flying-jarvis-styles')) {
    width: 65px;nst styleElement = document.createElement('div');
    height: 90px;.id = 'flying-jarvis-styles';
    transform: translateX(-50%); = flyingJarvisStyles;
    background: linear-gradient(145deg,    document.head.appendChild(styleElement);
        rgba(0,191,255,0.9) 0%,    }
        rgba(0,150,255,0.8) 25%,    
        rgba(0,120,255,0.7) 50%,entById('hologram3d')) {
        rgba(0,100,200,0.6) 75%,flyingJarvis = new FlyingJarvisRobot();
        rgba(0,80,160,0.5) 100%);
    border-radius: 30px 30px 25px 25px;
    filter: blur(0.3px);
    animation: jarvisBodyPulse 3s ease-in-out infinite;ddEventListener('DOMContentLoaded', () => {
    box-shadow: 
        initFlyingJarvis();
        
        // Initialize voice system
        window.aetheronVoice = new AetheronVoiceSystem();
        
        // Integration with flying robot
        if (window.flyingJarvis && window.aetheronVoice) {
            // Connect voice feedback to flying actions
            const originalSpeak = window.flyingJarvis.speak;
            window.flyingJarvis.speak = function(text) {
                window.aetheronVoice.speak(text, 'flying');
            };
        }
    }, 1200);
});
