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
        this.eyeTracking = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.createFlightEffects();
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
    
    setTarget(x, y) {
        // Add some offset so Jarvis doesn't overlap the cursor
        const offset = 100;
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
        const speed = 50;
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
                this.setTarget(this.position.x, this.position.y - speed);
                break;
            case 'arrowdown':
                this.setTarget(this.position.x, this.position.y + speed);
                break;
            case 'arrowleft':
                this.setTarget(this.position.x - speed, this.position.y);
                break;
            case 'arrowright':
                this.setTarget(this.position.x + speed, this.position.y);
                break;
        }
    }
    
    returnHome() {
        this.isFollowingMouse = false;
        this.isAutonomous = false;
        this.setTarget(window.innerWidth / 2, window.innerHeight / 2);
        this.speak("Returning to home position.");
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
    }
    
    updatePhysics() {
        // Calculate distance to target
        const dx = this.target.x - this.position.x;
        const dy = this.target.y - this.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Apply force towards target with damping
        const force = 0.05;
        const damping = 0.88;
        
        if (distance > 5) {
            this.velocity.x += dx * force;
            this.velocity.y += dy * force;
            this.thrusterPower = Math.min(distance / 200, 1);
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
        this.rotation += rotDiff * 0.1;
        
        // Hover effect
        this.hoverHeight = Math.sin(Date.now() * 0.003) * 10;
        
        // Scale effect based on speed
        const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
        this.targetScale = 0.8 + Math.min(speed / 10, 0.4);
        this.scale += (this.targetScale - this.scale) * 0.1;
    }
    
    updateVisuals() {
        // Update main hologram position and rotation
        this.holo.style.left = `${this.position.x - 150}px`;
        this.holo.style.top = `${this.position.y - 250 + this.hoverHeight}px`;
        this.holo.style.transform = `scale(${this.scale}) rotate(${this.rotation * 0.1}deg)`;
        
        // Update thruster effects
        const thrusters = this.holo.querySelectorAll('.thruster-effect');
        thrusters.forEach(thruster => {
            thruster.style.opacity = this.thrusterPower;
            thruster.style.transform = `scale(${0.5 + this.thrusterPower * 0.5})`;
        });
        
        // Update flight trail
        const trail = this.holo.querySelector('.flight-trail');
        if (trail) {
            trail.style.opacity = Math.min(this.thrusterPower * 2, 0.8);
        }
        
        // Eye tracking effect
        if (this.eyeTracking) {
            this.updateEyeTracking();
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
                this.returnHome();
            }
        }, 100);
        
        this.speak("Analyzing data patterns...");
    }
    
    performScanFlying() {
        // Fly in scanning pattern (zigzag)
        const startX = 100;
        const endX = window.innerWidth - 100;
        const y = this.position.y;
        let direction = 1;
        let currentX = startX;
        
        const scanInterval = setInterval(() => {
            this.setTarget(currentX, y);
            currentX += direction * 100;
            
            if (currentX >= endX || currentX <= startX) {
                direction *= -1;
                if (Math.abs(currentX - startX) < 50) {
                    clearInterval(scanInterval);
                    this.returnHome();
                }
            }
        }, 800);
        
        this.speak("Scanning interface systems...");
    }
    
    performAlertFlying() {
        // Quick shake movement
        const originalX = this.position.x;
        const originalY = this.position.y;
        let shakeCount = 0;
        
        const shakeInterval = setInterval(() => {
            const offsetX = (Math.random() - 0.5) * 40;
            const offsetY = (Math.random() - 0.5) * 40;
            this.setTarget(originalX + offsetX, originalY + offsetY);
            shakeCount++;
            
            if (shakeCount > 10) {
                clearInterval(shakeInterval);
                this.setTarget(originalX, originalY);
            }
        }, 150);
    }
    
    performCelebrationFlying() {
        // Victory loop
        const centerX = this.position.x;
        const centerY = this.position.y;
        let altitude = 0;
        
        const celebrateInterval = setInterval(() => {
            const x = centerX + Math.sin(altitude * 0.2) * 60;
            const y = centerY - Math.abs(Math.sin(altitude * 0.1)) * 100;
            this.setTarget(x, y);
            altitude += 0.5;
            
            if (altitude > 20) {
                clearInterval(celebrateInterval);
                this.setTarget(centerX, centerY);
            }
        }, 100);
        
        this.speak("Mission accomplished!");
    }
}

// Initialize the flying robot system
function initHologram3D() {
    if (document.getElementById('hologram3d')) {
        window.flyingJarvis = new FlyingJarvisRobot();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initHologram3D, 1200);
});

// --- Flying Robot Jarvis Hologram Styles ---
const hologramStyles = `
<style>
#hologram3d {
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translateX(-50%) translateY(-50%);
    width: 300px;
    height: 400px;
    z-index: 10;
    pointer-events: auto;
    perspective: 1200px;
    filter: drop-shadow(0 0 60px #00bfff99);
    transition: all 0.3s ease;
    cursor: pointer;
}

#hologram3d:hover {
    filter: drop-shadow(0 0 80px #00bfff) drop-shadow(0 0 40px #ffffff66);
}

.holo-base {
    position: absolute;
    bottom: 0px;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 6px;
    background: radial-gradient(ellipse at center, #00bfff 20%, #40e0ff 50%, transparent 100%);
    opacity: 0.6;
    border-radius: 50%;
    filter: blur(2px);
    z-index: 1;
    animation: jarvisBasePulse 2s infinite ease-in-out;
}

.holo-avatar {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 120px;
    height: 200px;
    transform: translateX(-50%) translateY(-50%);
    transform-style: preserve-3d;
    transition: filter 0.3s ease;
    filter: drop-shadow(0 0 30px #00bfff99) brightness(1.2);
    z-index: 2;
}

/* Compact Robot Head */
.holo-face {
    width: 60px;
    height: 60px;
    margin: 0 auto;
    background: radial-gradient(ellipse at center, 
        #00bfff 0%, 
        #40e0ff 50%, 
        #0080ff 80%, 
        transparent 100%);
    border-radius: 50%;
    box-shadow: 
        0 0 25px #00bfff99, 
        0 0 50px #00bfff55,
        inset 0 0 10px rgba(0,191,255,0.6);
    position: relative;
    animation: jarvisHeadPulse 2.5s ease-in-out infinite;
    overflow: hidden;
}

.holo-face::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 30px;
    height: 30px;
    background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(0,191,255,0.5) 70%, transparent 90%);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: jarvisCore 2s ease-in-out infinite;
}

.holo-eye {
    position: absolute;
    width: 8px;
    height: 8px;
    background: radial-gradient(circle, #ffffff 0%, #00bfff 70%, transparent 100%);
    border-radius: 50%;
    animation: jarvisEye 1.5s ease-in-out infinite;
    box-shadow: 0 0 15px #ffffff88;
}

.holo-eye.left {
    left: 35%;
    top: 40%;
}

.holo-eye.right {
    right: 35%;
    top: 40%;
}

.eye-pupil {
    width: 4px;
    height: 4px;
    background: #ffffff;
    border-radius: 50%;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    transition: transform 0.1s ease;
}

/* Compact Robot Body */
.holo-body {
    position: absolute;
    left: 50%;
    top: 70px;
    width: 80px;
    height: 100px;
    transform: translateX(-50%);
    background: linear-gradient(to bottom, 
        rgba(0,191,255,0.8) 0%, 
        rgba(0,191,255,0.6) 40%,
        rgba(0,128,255,0.5) 70%, 
        rgba(0,191,255,0.4) 100%);
    border-radius: 25px 25px 20px 20px;
    filter: blur(0.5px);
    animation: jarvisBodyPulse 3s ease-in-out infinite;
    box-shadow: 
        0 0 20px rgba(0,191,255,0.7),
        inset 0 0 15px rgba(0,191,255,0.3);
}

.holo-body::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 30%;
    width: 40px;
    height: 40px;
    background: radial-gradient(circle, rgba(0,191,255,0.9) 0%, transparent 70%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: jarvisChest 2.5s ease-in-out infinite;
}

/* Robot Arms/Wings */
.holo-body::after {
    content: '';
    position: absolute;
    left: -20px;
    top: 20px;
    width: 120px;
    height: 40px;
    background: linear-gradient(90deg, 
        rgba(0,191,255,0.3) 0%, 
        rgba(0,191,255,0.6) 20%,
        rgba(0,191,255,0.4) 50%,
        rgba(0,191,255,0.6) 80%,
        rgba(0,191,255,0.3) 100%);
    border-radius: 20px;
    filter: blur(1px);
    animation: jarvisWings 2s ease-in-out infinite;
}

/* Thruster Effects */
.thruster-effect {
    position: absolute;
    bottom: -10px;
    width: 20px;
    height: 40px;
    background: linear-gradient(to bottom,
        transparent 0%,
        rgba(0,191,255,0.8) 30%,
        rgba(255,255,255,0.9) 70%,
        rgba(0,255,255,1) 100%);
    border-radius: 10px 10px 50% 50%;
    filter: blur(2px);
    opacity: 0;
    animation: thrusterFlame 0.1s infinite ease-in-out;
}

.thruster-left {
    left: 25%;
}

.thruster-right {
    right: 25%;
}

.flight-trail {
    position: absolute;
    left: 50%;
    top: 100%;
    width: 4px;
    height: 80px;
    background: linear-gradient(to bottom,
        rgba(0,191,255,0.8) 0%,
        rgba(0,191,255,0.4) 50%,
        transparent 100%);
    transform: translateX(-50%);
    opacity: 0;
    filter: blur(1px);
    animation: flightTrail 0.2s infinite ease-in-out;
}

/* Enhanced Animations */
@keyframes jarvisBasePulse {
    0%, 100% { 
        transform: translateX(-50%) scale(1); 
        opacity: 0.6; 
    }
    50% { 
        transform: translateX(-50%) scale(1.1); 
        opacity: 0.8; 
    }
}

@keyframes jarvisHeadPulse {
    0%, 100% { 
        transform: scale(1); 
        box-shadow: 0 0 25px #00bfff99, 0 0 50px #00bfff55;
    }
    50% { 
        transform: scale(1.05); 
        box-shadow: 0 0 35px #00bfff, 0 0 70px #00bfff77;
    }
}

@keyframes jarvisBodyPulse {
    0%, 100% { 
        transform: translateX(-50%) scale(1); 
        opacity: 0.8;
    }
    50% { 
        transform: translateX(-50%) scale(1.02); 
        opacity: 0.9;
    }
}

@keyframes jarvisCore {
    0%, 100% { 
        transform: translate(-50%, -50%) scale(1); 
        opacity: 0.6; 
    }
    50% { 
        transform: translate(-50%, -50%) scale(1.1); 
        opacity: 0.9; 
    }
}

@keyframes jarvisEye {
    0%, 100% { 
        opacity: 0.8; 
        transform: scale(1);
    }
    50% { 
        opacity: 1; 
        transform: scale(1.2);
    }
}

@keyframes jarvisChest {
    0%, 100% { 
        transform: translateX(-50%) scale(1); 
        opacity: 0.6; 
    }
    50% { 
        transform: translateX(-50%) scale(1.15); 
        opacity: 0.9; 
    }
}

@keyframes jarvisWings {
    0%, 100% { 
        transform: scaleX(1) scaleY(1); 
        opacity: 0.4; 
    }
    50% { 
        transform: scaleX(1.1) scaleY(0.9); 
        opacity: 0.7; 
    }
}

@keyframes thrusterFlame {
    0%, 100% { 
        transform: scaleY(1) scaleX(1); 
    }
    50% { 
        transform: scaleY(1.2) scaleX(0.8); 
    }
}

@keyframes flightTrail {
    0% { 
        height: 60px; 
        opacity: 0.6; 
    }
    50% { 
        height: 100px; 
        opacity: 0.8; 
    }
    100% { 
        height: 80px; 
        opacity: 0.4; 
    }
}

/* Interactive States */
#hologram3d.following {
    filter: drop-shadow(0 0 80px #00ff00) drop-shadow(0 0 40px #00bfff99);
}

#hologram3d.autonomous {
    filter: drop-shadow(0 0 80px #ff6600) drop-shadow(0 0 40px #00bfff99);
}

#hologram3d.thinking {
    animation: thinkingMode 2s infinite ease-in-out;
}

#hologram3d.scanning {
    animation: scanningMode 1s infinite linear;
}

#hologram3d.alert {
    animation: alertMode 0.3s infinite ease-in-out;
}

#hologram3d.celebrating {
    animation: celebrateMode 0.5s infinite ease-in-out;
}

@keyframes thinkingMode {
    0%, 100% { transform: translateX(-50%) translateY(-50%) rotateZ(0deg); }
    25% { transform: translateX(-50%) translateY(-50%) rotateZ(2deg); }
    75% { transform: translateX(-50%) translateY(-50%) rotateZ(-2deg); }
}

@keyframes scanningMode {
    0% { transform: translateX(-50%) translateY(-50%) rotateY(0deg); }
    100% { transform: translateX(-50%) translateY(-50%) rotateY(360deg); }
}

@keyframes alertMode {
    0%, 100% { transform: translateX(-50%) translateY(-50%) scale(1); }
    50% { transform: translateX(-50%) translateY(-50%) scale(1.1); }
}

@keyframes celebrateMode {
    0%, 100% { transform: translateX(-50%) translateY(-50%) rotateZ(0deg) scale(1); }
    25% { transform: translateX(-50%) translateY(-50%) rotateZ(5deg) scale(1.05); }
    75% { transform: translateX(-50%) translateY(-50%) rotateZ(-5deg) scale(1.05); }
}

/* Control Panel Styles */
.jarvis-controls {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid #00bfff;
    border-radius: 10px;
    padding: 15px;
    color: #00bfff;
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    z-index: 1000;
}

.jarvis-controls h4 {
    margin: 0 0 10px 0;
    color: #ffffff;
    text-align: center;
}

.jarvis-controls button {
    display: block;
    width: 100%;
    margin: 5px 0;
    padding: 8px;
    background: transparent;
    border: 1px solid #00bfff;
    color: #00bfff;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.jarvis-controls button:hover {
    background: #00bfff;
    color: #000;
}

.jarvis-controls .status {
    margin-top: 10px;
    padding: 5px;
    border-radius: 3px;
    text-align: center;
    font-size: 10px;
}

.status.following { background: rgba(0, 255, 0, 0.2); }
.status.autonomous { background: rgba(255, 102, 0, 0.2); }
.status.idle { background: rgba(128, 128, 128, 0.2); }
</style>
`;
.holo-avatar {
    position: absolute;
    left: 50%;
    bottom: 30px;
    width: 120px;
    height: 400px;
    transform-style: preserve-3d;
    transform: rotateY(0deg) rotateX(0deg);
    transition: filter 0.3s ease;
    filter: drop-shadow(0 0 40px #00bfff99) brightness(1.3);
    z-index: 2;
}

/* Jarvis Head */
.holo-face {
    width: 80px;
    height: 100px;
    margin: 0 auto;
    background: radial-gradient(ellipse at center, 
        #00bfff 0%, 
        #40e0ff 40%, 
        #0080ff 70%, 
        transparent 100%);
    border-radius: 40px 40px 35px 35px;
    box-shadow: 
        0 0 30px #00bfff99, 
        0 0 60px #00bfff55,
        inset 0 0 15px rgba(0,191,255,0.4);
    position: relative;
    animation: jarvisHeadPulse 3s ease-in-out infinite;
    overflow: hidden;
}
.holo-face::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 40%;
    width: 50px;
    height: 50px;
    background: radial-gradient(circle, rgba(255,255,255,0.6) 0%, rgba(0,191,255,0.3) 60%, transparent 90%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: jarvisCore 2.5s ease-in-out infinite;
}
.holo-face::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 45%;
    width: 15px;
    height: 15px;
    background: radial-gradient(circle, #ffffff 0%, #00bfff 70%, transparent 100%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: jarvisEye 2s ease-in-out infinite;
    box-shadow: 0 0 20px #ffffff88;
}

/* Jarvis Torso */
.holo-body {
    position: absolute;
    left: 50%;
    top: 110px;
    width: 100px;
    height: 180px;
    transform: translateX(-50%);
    background: linear-gradient(to bottom, 
        rgba(0,191,255,0.7) 0%, 
        rgba(0,191,255,0.5) 30%,
        rgba(0,128,255,0.4) 60%, 
        rgba(0,191,255,0.3) 100%);
    border-radius: 50px 50px 40px 40px;
    filter: blur(1px);
    animation: jarvisBodyPulse 3.5s ease-in-out infinite;
    box-shadow: 
        0 0 25px rgba(0,191,255,0.6),
        inset 0 0 20px rgba(0,191,255,0.2);
}
.holo-body::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 20%;
    width: 60px;
    height: 60px;
    background: radial-gradient(circle, rgba(0,191,255,0.8) 0%, transparent 70%);
    border-radius: 50%;
    transform: translateX(-50%);
    animation: jarvisChest 3s ease-in-out infinite;
}

/* Enhanced Jarvis Arms with better positioning */
.holo-body::after {
    content: '';
    position: absolute;
    left: -25px;
    top: 35px;
    width: 150px;
    height: 80px;
    background: 
        radial-gradient(ellipse at 25% 40%, rgba(0,191,255,0.6) 0%, rgba(0,191,255,0.3) 40%, transparent 70%),
        radial-gradient(ellipse at 75% 40%, rgba(0,191,255,0.6) 0%, rgba(0,191,255,0.3) 40%, transparent 70%),
        linear-gradient(90deg, transparent 0%, rgba(0,191,255,0.2) 50%, transparent 100%);
    border-radius: 40px;
    animation: jarvisArms 4s ease-in-out infinite;
    filter: blur(0.5px);
}

/* Enhanced Jarvis Legs with better symmetry */
.holo-neural {
    position: absolute;
    left: 50%;
    top: 280px;
    width: 90px;
    height: 130px;
    transform: translateX(-50%);
    background: 
        linear-gradient(to bottom left, rgba(0,191,255,0.7) 0%, rgba(0,191,255,0.4) 60%, transparent 100%),
        linear-gradient(to bottom right, rgba(0,191,255,0.7) 0%, rgba(0,191,255,0.4) 60%, transparent 100%),
        radial-gradient(ellipse at 30% 20%, rgba(0,191,255,0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 70% 20%, rgba(0,191,255,0.3) 0%, transparent 50%);
    border-radius: 45px 45px 25px 25px;
    animation: jarvisLegs 3.5s ease-in-out infinite;
    filter: blur(0.8px);
    box-shadow: 
        0 0 20px rgba(0,191,255,0.4),
        inset 0 0 15px rgba(0,191,255,0.2);
}

/* Neural Network Overlay */
.holo-scanlines {
    position: absolute;
    left: 0; top: 0;
    width: 120px; height: 400px;
    pointer-events: none;
    background: 
        repeating-linear-gradient(
            to bottom,
            rgba(0,191,255,0.15) 0px,
            rgba(0,191,255,0.15) 1px,
            transparent 1px,
            transparent 4px
        ),
        radial-gradient(ellipse at 50% 30%, rgba(0,191,255,0.1) 0%, transparent 60%),
        radial-gradient(ellipse at 50% 70%, rgba(0,191,255,0.1) 0%, transparent 60%);
    border-radius: 60px;
    animation: jarvisScan 3s linear infinite;
    opacity: 0.7;
}

/* Enhanced Floating Particles with more variety */
.holo-avatar::before {
    content: '';
    position: absolute;
    left: -30px;
    top: -30px;
    width: 180px;
    height: 460px;
    background: 
        radial-gradient(circle at 15% 15%, rgba(0,191,255,0.4) 1.5px, transparent 3px),
        radial-gradient(circle at 85% 25%, rgba(64,224,255,0.3) 1px, transparent 2px),
        radial-gradient(circle at 25% 45%, rgba(0,191,255,0.5) 1px, transparent 2px),
        radial-gradient(circle at 75% 55%, rgba(64,224,255,0.2) 2px, transparent 4px),
        radial-gradient(circle at 50% 75%, rgba(0,191,255,0.4) 1px, transparent 2px),
        radial-gradient(circle at 90% 85%, rgba(64,224,255,0.3) 1.5px, transparent 3px),
        radial-gradient(circle at 10% 95%, rgba(0,191,255,0.2) 1px, transparent 2px);
    background-size: 50px 50px, 60px 60px, 40px 40px, 55px 55px, 45px 45px, 42px 42px, 48px 48px;
    animation: jarvisParticles 8s ease-in-out infinite;
    pointer-events: none;
}

/* Enhanced Energy Field with multiple layers */
.holo-avatar::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 150px;
    height: 430px;
    transform: translate(-50%, -50%);
    border: 2px solid rgba(0,191,255,0.25);
    border-radius: 75px;
    animation: jarvisField 5s ease-in-out infinite;
    box-shadow: 
        0 0 25px rgba(0,191,255,0.35),
        inset 0 0 25px rgba(0,191,255,0.15),
        0 0 50px rgba(0,191,255,0.2);
}
.holo-avatar .holo-field-inner {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 170px;
    height: 450px;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(64,224,255,0.15);
    border-radius: 85px;
    animation: jarvisFieldInner 4s ease-in-out infinite reverse;
    pointer-events: none;
}

@keyframes jarvisHeadPulse {
    0%, 100% { 
        filter: brightness(1.3) hue-rotate(0deg);
        transform: scale(1);
    }
    50% { 
        filter: brightness(1.6) hue-rotate(10deg);
        transform: scale(1.02);
    }
}
@keyframes jarvisCore {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1) rotate(0deg); }
    50% { opacity: 0.9; transform: translateX(-50%) scale(1.1) rotate(180deg); }
}
@keyframes jarvisEye {
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.2); }
}
@keyframes jarvisBodyPulse {
    0%, 100% { opacity: 0.7; transform: translateX(-50%) scaleY(1); }
    50% { opacity: 0.9; transform: translateX(-50%) scaleY(1.02); }
}
@keyframes jarvisChest {
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.1); }
}
@keyframes jarvisArms {
    0%, 100% { 
        opacity: 0.6; 
        transform: scaleX(1) scaleY(1) rotate(0deg);
        filter: hue-rotate(0deg);
    }
    33% { 
        opacity: 0.8; 
        transform: scaleX(1.03) scaleY(1.02) rotate(1deg);
        filter: hue-rotate(10deg);
    }
    66% { 
        opacity: 0.7; 
        transform: scaleX(0.98) scaleY(1.01) rotate(-1deg);
        filter: hue-rotate(-10deg);
    }
}
@keyframes jarvisLegs {
    0%, 100% { 
        opacity: 0.7; 
        transform: translateX(-50%) scaleY(1) scaleX(1);
        filter: hue-rotate(0deg);
    }
    50% { 
        opacity: 0.9; 
        transform: translateX(-50%) scaleY(1.01) scaleX(1.02);
        filter: hue-rotate(8deg);
    }
}
@keyframes jarvisBasePulse {
    0%, 100% { 
        opacity: 0.8; 
        transform: translateX(-50%) scale(1);
        filter: blur(2px) hue-rotate(0deg);
    }
    50% { 
        opacity: 1; 
        transform: translateX(-50%) scale(1.08);
        filter: blur(1px) hue-rotate(15deg);
    }
}
@keyframes jarvisBaseScan {
    0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scaleX(1); }
    50% { opacity: 0.8; transform: translate(-50%, -50%) scaleX(1.2); }
}
@keyframes jarvisScan {
    0% { background-position-y: 0; opacity: 0.7; }
    50% { opacity: 0.4; }
    100% { background-position-y: 400px; opacity: 0.7; }
}
@keyframes jarvisParticles {
    0%, 100% { 
        opacity: 0.7; 
        transform: scale(1) rotate(0deg);
        filter: hue-rotate(0deg) brightness(1);
    }
    25% { 
        opacity: 0.9; 
        transform: scale(1.02) rotate(90deg);
        filter: hue-rotate(15deg) brightness(1.1);
    }
    50% { 
        opacity: 0.8; 
        transform: scale(0.98) rotate(180deg);
        filter: hue-rotate(30deg) brightness(1.2);
    }
    75% { 
        opacity: 0.85; 
        transform: scale(1.01) rotate(270deg);
        filter: hue-rotate(15deg) brightness(1.1);
    }
}
@keyframes jarvisField {
    0%, 100% { 
        opacity: 0.25; 
        transform: translate(-50%, -50%) scale(1);
        border-color: rgba(0,191,255,0.25);
        box-shadow: 
            0 0 25px rgba(0,191,255,0.35),
            inset 0 0 25px rgba(0,191,255,0.15),
            0 0 50px rgba(0,191,255,0.2);
    }
    50% { 
        opacity: 0.45; 
        transform: translate(-50%, -50%) scale(1.01);
        border-color: rgba(64,224,255,0.4);
        box-shadow: 
            0 0 35px rgba(0,191,255,0.5),
            inset 0 0 35px rgba(0,191,255,0.25),
            0 0 70px rgba(0,191,255,0.3);
    }
}
@keyframes jarvisFieldInner {
    0%, 100% { 
        opacity: 0.15; 
        transform: translate(-50%, -50%) scale(1);
        border-color: rgba(64,224,255,0.15);
    }
    50% { 
        opacity: 0.3; 
        transform: translate(-50%, -50%) scale(1.015);
        border-color: rgba(64,224,255,0.25);
    }
}

/* Enhanced gesture animations for different interactions */
.holo-avatar.gesture-thinking {
    animation: jarvisThinking 2s ease-in-out infinite;
}
.holo-avatar.gesture-analyzing {
    animation: jarvisAnalyzing 1.5s ease-in-out infinite;
}
.holo-avatar.gesture-alert {
    animation: jarvisAlert 0.8s ease-in-out infinite;
}
.holo-avatar.gesture-success {
    animation: jarvisSuccess 2s ease-in-out;
}

@keyframes jarvisThinking {
    0%, 100% { 
        filter: drop-shadow(0 0 40px #00bfff99) brightness(1.3);
        transform: scale(1) rotateY(0deg);
    }
    25% { 
        filter: drop-shadow(0 0 50px #40e0ff99) brightness(1.5);
        transform: scale(1.02) rotateY(5deg);
    }
    75% { 
        filter: drop-shadow(0 0 50px #0080ff99) brightness(1.5);
        transform: scale(1.02) rotateY(-5deg);
    }
}

@keyframes jarvisAnalyzing {
    0%, 100% { 
        filter: drop-shadow(0 0 45px #00bfff99) brightness(1.4) hue-rotate(0deg);
        transform: scale(1);
    }
    50% { 
        filter: drop-shadow(0 0 65px #40e0ff99) brightness(1.8) hue-rotate(20deg);
        transform: scale(1.03);
    }
}

@keyframes jarvisAlert {
    0%, 100% { 
        filter: drop-shadow(0 0 40px #ff6b6b99) brightness(1.3);
        transform: scale(1);
    }
    50% { 
        filter: drop-shadow(0 0 60px #ff4757aa) brightness(1.7);
        transform: scale(1.05);
    }
}

@keyframes jarvisSuccess {
    0% { 
        filter: drop-shadow(0 0 40px #00bfff99) brightness(1.3);
        transform: scale(1);
    }
    25% { 
        filter: drop-shadow(0 0 70px #2ed573aa) brightness(2);
        transform: scale(1.08);
    }
    50% { 
        filter: drop-shadow(0 0 80px #2ed573aa) brightness(2.2);
        transform: scale(1.1);
    }
    100% { 
        filter: drop-shadow(0 0 40px #00bfff99) brightness(1.3);
        transform: scale(1);
    }
}

/* Enhanced Speaking animation when voice is active */
.holo-avatar.speaking {
    animation: jarvisSpeaking 0.35s ease-in-out infinite alternate;
}
.holo-avatar.speaking .holo-face {
    animation: jarvisHeadPulse 3s ease-in-out infinite, jarvisSpeakingHead 0.35s ease-in-out infinite alternate;
}
.holo-avatar.speaking .holo-body {
    animation: jarvisBodyPulse 3.5s ease-in-out infinite, jarvisSpeakingBody 0.35s ease-in-out infinite alternate;
}
@keyframes jarvisSpeaking {
    0% { 
        filter: drop-shadow(0 0 45px #00bfff99) brightness(1.3);
        transform: scale(1);
    }
    100% { 
        filter: drop-shadow(0 0 80px #00ffffff) brightness(1.9);
        transform: scale(1.025);
    }
}
@keyframes jarvisSpeakingHead {
    0% { 
        box-shadow: 
            0 0 30px #00bfff99, 
            0 0 60px #00bfff55,
            inset 0 0 15px rgba(0,191,255,0.4);
    }
    100% { 
        box-shadow: 
            0 0 50px #00ffffff, 
            0 0 100px #00bfff88,
            inset 0 0 25px rgba(0,191,255,0.6);
    }
}
@keyframes jarvisSpeakingBody {
    0% { 
        box-shadow: 
            0 0 25px rgba(0,191,255,0.6),
            inset 0 0 20px rgba(0,191,255,0.2);
    }
    100% { 
        box-shadow: 
            0 0 40px rgba(0,191,255,0.9),
            inset 0 0 30px rgba(0,191,255,0.4);
    }
}
@keyframes voiceIndicatorPulse {
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.05); }
}
</style>
`;
document.head.insertAdjacentHTML('beforeend', hologramStyles);

// Advanced Jarvis Voice System
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
                "Good day, Sir. Jarvis systems are online and fully operational.",
                "Welcome back, Commander. All neural networks are functioning within normal parameters.",
                "Aetheron Platform initialized successfully. Standing by for your instructions, Sir.",
                "Good to see you again. All systems are green and ready for deployment.",
                "Sir, the Aetheron artificial intelligence matrix is at your service.",
                "Holographic interface established. How may I assist you today, Commander?",
                "Neural processing cores are operating at peak efficiency. At your command, Sir."
            ],
            training: [
                "Initiating neural network training protocol, Sir.",
                "Commencing machine learning optimization sequence with advanced algorithms.",
                "Training algorithms are now active. Monitoring progress with real-time analytics.",
                "Neural pathways are being refined. Performance metrics updating continuously.",
                "Sir, I'm deploying sophisticated learning protocols for optimal model convergence.",
                "Beginning computational training sequence. All parameters are within acceptable ranges.",
                "Engaging deep learning architecture. Training commenced, Commander."
            ],
            progress: [
                "Training progress is proceeding as expected, Sir. All metrics are nominal.",
                "Neural network optimization continues within acceptable parameters, Commander.",
                "Model convergence detected. Accuracy metrics improving steadily as anticipated.",
                "Learning algorithms are performing admirably. Results are quite promising, Sir.",
                "Sir, the training regimen is yielding excellent performance indicators.",
                "Computational learning is progressing smoothly. Error rates declining steadily.",
                "Neural adaptation continues. The model's intellectual capacity is expanding, Commander."
            ],
            success: [
                "Training protocol completed successfully. Model performance is exemplary, Sir.",
                "Neural network optimization has concluded with outstanding results, Commander.",
                "Mission accomplished, Sir. The model has achieved optimal performance metrics.",
                "Training sequence finalized. Results exceed all expectations, as predicted.",
                "Sir, I'm pleased to report that the learning protocol has achieved remarkable success.",
                "Computational training complete. The neural architecture now operates at peak efficiency.",
                "Training objectives met with distinction. The model demonstrates exceptional accuracy, Commander."
            ],
            errors: [
                "I'm afraid we've encountered a minor setback in the training protocol, Sir.",
                "Sir, there appears to be an anomaly in the data processing pipeline requiring attention.",
                "System alert: Training sequence has encountered an unexpected computational error.",
                "Apologies, Commander. A technical difficulty has been detected in the learning matrix.",
                "Sir, I must report a temporary impediment in our neural processing operations.",
                "It seems we've hit a computational snag. Allow me to recalibrate the parameters.",
                "A minor algorithmic deviation has occurred. Initiating corrective measures, Sir."
            ],
            predictions: [
                "Analyzing data patterns and generating predictive assessments, Sir.",
                "Running advanced predictive algorithms. Confidence levels are exceptionally high.",
                "Processing your request with sophisticated computational analysis, Commander.",
                "Deploying prediction models. Results incoming momentarily with detailed insights.",
                "Sir, I'm conducting comprehensive pattern analysis for optimal prediction accuracy.",
                "Engaging predictive intelligence protocols. Data correlation is proceeding smoothly.",
                "Running deep analytical models. The predictions are forming quite nicely, Commander."
            ],
            ambient: [
                "All systems operating within normal parameters, Sir.",
                "Monitoring network performance continuously. Everything looks optimal, Commander.",
                "Standing by for further instructions. Ready to assist at a moment's notice, Sir.",
                "Neural networks are humming along nicely. Performance metrics remain excellent.",
                "Sir, all Aetheron systems are functioning at peak operational capacity.",
                "Holographic matrix stable. All computational cores operating smoothly, Commander.",
                "Maintaining optimal system status. Ready for your next directive, Sir."
            ],
            tabs: [
                "Transitioning to the requested interface section, Sir.",
                "Navigating to your selected workspace area, Commander.",
                "Switching operational focus as requested. New interface loaded.",
                "Sir, I've updated the display to reflect your preferred analytical view.",
                "Interface reconfiguration complete. Your new workspace is ready, Commander.",
                "Adaptive interface responding to your selection. How may I assist further, Sir?"
            ]
        };
        
        // Advanced Voice Commands System
        this.recognition = null;
        this.isListening = false;
        this.commandPatterns = {
            training: [
                /start.*train/i, /begin.*train/i, /initiate.*train/i, /commence.*train/i,
                /run.*model/i, /execute.*train/i, /launch.*train/i
            ],
            prediction: [
                /predict/i, /analyze.*data/i, /run.*prediction/i, /generate.*prediction/i,
                /forecast/i, /infer/i, /classify/i
            ],
            navigation: [
                /go.*to.*train/i, /switch.*train/i, /open.*train/i,
                /go.*to.*analyze/i, /switch.*analyze/i, /open.*analyze/i,
                /go.*to.*compare/i, /switch.*compare/i, /open.*compare/i,
                /go.*to.*deploy/i, /switch.*deploy/i, /open.*deploy/i,
                /go.*to.*data/i, /switch.*data/i, /open.*data/i
            ],
            status: [
                /status/i, /report/i, /how.*doing/i, /performance/i,
                /system.*check/i, /diagnostics/i, /health/i
            ],
            greeting: [
                /hello.*jarvis/i, /hi.*jarvis/i, /good.*morning.*jarvis/i,
                /good.*day.*jarvis/i, /wake.*up/i, /are.*you.*there/i
            ],
            help: [
                /help/i, /what.*can.*you.*do/i, /commands/i, /assist/i,
                /capabilities/i, /features/i
            ]
        };
        this.contextualResponses = {
            training: {
                idle: "Shall I commence the neural network training protocol, Sir?",
                active: "Training is currently in progress. Monitoring performance metrics.",
                complete: "Training completed successfully. Would you like to analyze the results?"
            },
            analysis: {
                idle: "Ready to analyze your data, Commander. What would you like to examine?",
                active: "Currently processing analytical computations. Results forthcoming.",
                complete: "Analysis complete. All findings are available for your review."
            }
        };
        this.userActivity = {
            lastAction: 'idle',
            currentTab: 'train',
            trainingActive: false,
            predictionActive: false,
            lastInteraction: Date.now()
        };
        this.smartSuggestions = [];
        
        this.init();
    }

    async init() {
        await this.loadVoices();
        this.setupVoiceEvents();
        this.createVoiceUI();
        this.initVoiceRecognition();
        
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

        // UI interaction events with enhanced responses
        document.addEventListener('tab-switched', (e) => {
            const tabResponses = {
                train: 'Neural training bay is now online, Sir. All learning protocols are at your disposal.',
                analyze: 'Analytics suite initialized, Commander. All diagnostic tools are ready for deployment.',
                compare: 'Model comparison protocols activated. Performance metrics and analysis available.',
                deploy: 'Deployment systems standing by for your instructions, Sir. Ready to launch.',
                data: 'Data management interface is ready for operation, Commander. All datasets accessible.'
            };
            if (this.ambientMode) {
                const response = tabResponses[e.detail.tab] || this.getRandomResponse('tabs');
                this.speak(response, 'ambient');
            }
        });

        // Voice command event handlers for advanced functionality
        document.addEventListener('voice-command-training', () => {
            // Trigger training start if on training tab
            const trainBtn = document.querySelector('#trainBtn, .train-button, [data-action="train"]');
            if (trainBtn) {
                trainBtn.click();
                this.userActivity.trainingActive = true;
                this.userActivity.lastAction = 'training';
            }
        });

        document.addEventListener('voice-command-prediction', () => {
            // Trigger prediction if on appropriate tab
            const predictBtn = document.querySelector('#predictBtn, .predict-button, [data-action="predict"]');
            if (predictBtn) {
                predictBtn.click();
                this.userActivity.predictionActive = true;
                this.userActivity.lastAction = 'prediction';
            }
        });

        document.addEventListener('voice-command-navigate', (e) => {
            // Navigate to requested tab
            const targetTab = e.detail.tab;
            const tabButton = document.querySelector(`[data-tab="${targetTab}"], .tab-${targetTab}`);
            if (tabButton) {
                tabButton.click();
            }
        });

        // Enhanced training events with activity tracking
        document.addEventListener('training-started', () => {
            this.userActivity.trainingActive = true;
            this.userActivity.lastAction = 'training';
            this.userActivity.lastInteraction = Date.now();
            
            const message = this.getRandomResponse('training');
            this.speak(message, 'training');
        });

        document.addEventListener('training-complete', (e) => {
            this.userActivity.trainingActive = false;
            this.userActivity.lastAction = 'training_complete';
            this.userActivity.lastInteraction = Date.now();
            
            const { accuracy, training_time } = e.detail;
            const successMsg = this.getRandomResponse('success');
            this.speak(`${successMsg} Final accuracy: ${(accuracy * 100).toFixed(1)} percent. Training duration: ${training_time} seconds.`, 'success');
        });

        document.addEventListener('prediction-started', () => {
            this.userActivity.predictionActive = true;
            this.userActivity.lastAction = 'prediction';
            this.userActivity.lastInteraction = Date.now();
            
            const message = this.getRandomResponse('predictions');
            this.speak(message, 'training');
        });

        document.addEventListener('prediction-complete', () => {
            this.userActivity.predictionActive = false;
            this.userActivity.lastAction = 'prediction_complete';
            this.userActivity.lastInteraction = Date.now();
            
            this.speak('Prediction analysis concluded, Sir. Results have been compiled and are ready for your review.', 'success');
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
                <button id="voice-command-toggle" class="voice-btn ${this.isListening ? 'active listening' : ''}">
                    🎤 Commands
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
        .voice-btn.listening {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            animation: voiceListening 1s ease-in-out infinite alternate;
        }
        @keyframes voiceListening {
            0% { box-shadow: 0 0 15px rgba(255, 107, 107, 0.6); }
            100% { box-shadow: 0 0 25px rgba(255, 107, 107, 0.9); }
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

        document.getElementById('voice-command-toggle').addEventListener('click', () => {
            this.toggleVoiceCommands();
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

    // Initialize voice command recognition
    initVoiceRecognition() {
        if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
            console.warn('Speech recognition not supported in this browser.');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            console.log('Voice command recognition activated.');
            this.speak('Voice command mode activated, Sir. I am listening.', 'system');
        };

        this.recognition.onend = () => {
            this.isListening = false;
            console.log('Voice command recognition deactivated.');
        };

        this.recognition.onresult = (event) => {
            const lastResult = event.results[event.results.length - 1];
            if (lastResult.isFinal) {
                const command = lastResult[0].transcript.toLowerCase().trim();
                this.processVoiceCommand(command);
            }
        };
    }

    // Process voice commands with sophisticated responses
    processVoiceCommand(command) {
        console.log('Processing voice command:', command);
        
        // Check for greeting patterns
        if (this.matchesPattern(command, this.commandPatterns.greeting)) {
            const response = this.getRandomResponse('greetings');
            this.speak(response, 'greeting');
            return;
        }

        // Check for help patterns
        if (this.matchesPattern(command, this.commandPatterns.help)) {
            this.speak('Sir, I can assist with training models, running predictions, navigating the platform, and providing status reports. Simply speak naturally and I will understand.', 'help');
            return;
        }

        // Check for status patterns
        if (this.matchesPattern(command, this.commandPatterns.status)) {
            this.provideStatusReport();
            return;
        }

        // Check for training patterns
        if (this.matchesPattern(command, this.commandPatterns.training)) {
            this.speak('Initiating training sequence as requested, Sir.', 'training');
            // Trigger training start event
            document.dispatchEvent(new CustomEvent('voice-command-training'));
            return;
        }

        // Check for prediction patterns
        if (this.matchesPattern(command, this.commandPatterns.prediction)) {
            this.speak('Commencing predictive analysis, Commander.', 'prediction');
            // Trigger prediction event
            document.dispatchEvent(new CustomEvent('voice-command-prediction'));
            return;
        }

        // Check for navigation patterns
        if (this.matchesPattern(command, this.commandPatterns.navigation)) {
            this.handleNavigationCommand(command);
            return;
        }

        // Default response for unrecognized commands
        this.speak('I apologize, Sir, but I did not understand that command. Please try rephrasing or say "help" for assistance.', 'error');
    }

    // Helper method to match command patterns
    matchesPattern(command, patterns) {
        return patterns.some(pattern => pattern.test(command));
    }

    // Handle navigation voice commands
    handleNavigationCommand(command) {
        let targetTab = null;
        if (/train/i.test(command)) targetTab = 'train';
        else if (/analyze/i.test(command)) targetTab = 'analyze';
        else if (/compare/i.test(command)) targetTab = 'compare';
        else if (/deploy/i.test(command)) targetTab = 'deploy';
        else if (/data/i.test(command)) targetTab = 'data';

        if (targetTab) {
            this.speak(`Navigating to ${targetTab} section, Sir.`, 'navigation');
            // Trigger tab switch
            document.dispatchEvent(new CustomEvent('voice-command-navigate', { detail: { tab: targetTab } }));
        }
    }

    // Provide comprehensive status report
    provideStatusReport() {
        const status = this.analyzeSystemStatus();
        const report = `System status report, Sir. ${status.overall} ${status.details}`;
        this.speak(report, 'status');
    }

    // Analyze current system status
    analyzeSystemStatus() {
        const now = Date.now();
        const timeSinceLastAction = now - this.userActivity.lastInteraction;
        
        if (this.userActivity.trainingActive) {
            return {
                overall: "Training protocol is currently active.",
                details: "Neural networks are learning and optimization is in progress."
            };
        } else if (this.userActivity.predictionActive) {
            return {
                overall: "Prediction analysis is underway.",
                details: "Computational models are processing data patterns."
            };
        } else if (timeSinceLastAction > 300000) { // 5 minutes
            return {
                overall: "All systems are in standby mode.",
                details: "Ready for your next directive, Commander."
            };
        } else {
            return {
                overall: "All systems operating within normal parameters.",
                details: "Platform is active and responsive to your commands."
            };
        }
    }

    // Toggle voice command listening with enhanced UI feedback
    toggleVoiceCommands() {
        if (this.isListening) {
            this.recognition.stop();
            this.speak('Voice command mode deactivated, Sir.', 'system');
            document.getElementById('voice-command-toggle').classList.remove('listening');
        } else {
            this.recognition.start();
            document.getElementById('voice-command-toggle').classList.add('listening');
        }
    }

    // Advanced hologram gesture system
    triggerHologramGesture(gestureType) {
        const avatar = document.querySelector('.holo-avatar');
        if (!avatar) return;

        // Remove any existing gesture classes
        avatar.classList.remove('gesture-thinking', 'gesture-analyzing', 'gesture-alert', 'gesture-success');
        
        // Add the new gesture class
        avatar.classList.add(`gesture-${gestureType}`);
        
        // Remove the gesture after animation completes
        setTimeout(() => {
            avatar.classList.remove(`gesture-${gestureType}`);
        }, 3000);
    }

    // Enhanced speak method with gesture integration
    speak(text, category = 'general') {
        if (!this.isEnabled || !this.synthesis) return;
        
        // Cancel any ongoing speech
        this.synthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = this.currentVoice;
        utterance.rate = this.voiceConfig.rate;
        utterance.pitch = this.voiceConfig.pitch;
        utterance.volume = this.voiceConfig.volume;
        
        // Trigger appropriate hologram gesture based on category
        const gestureMap = {
            'training': 'analyzing',
            'error': 'alert',
            'success': 'success',
            'prediction': 'thinking',
            'system': 'analyzing'
        };
        
        if (gestureMap[category]) {
            this.triggerHologramGesture(gestureMap[category]);
        }
        
        // Enhanced visual feedback during speech
        const avatar = document.querySelector('.holo-avatar');
        if (avatar) {
            avatar.classList.add('speaking');
            
            utterance.onend = () => {
                avatar.classList.remove('speaking');
            };
        }
        
        this.synthesis.speak(utterance);
        console.log(`Jarvis: ${text}`);
    }
}
