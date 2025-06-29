// --- 3D Hologram Animation ---
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
        filter: brightness(1.3) hue-rotate(0deg);
        transform: scale(1);
    }
    100% { 
        filter: brightness(1.8) hue-rotate(15deg);
        transform: scale(1.08);
    }
}
</style>
`;
document.head.insertAdjacentHTML('beforeend', hologramStyles);
// Aetheron Platform Advanced JavaScript
// Real-time ML Training and Management Interface

class AetheronPlatform {
    constructor() {
        this.ws = null;
        this.clientId = this.generateClientId();
        this.trainingCharts = {};
        this.systemCharts = {};
        this.currentTraining = null;
        this.models = [];
        this.init();
    }

    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }

    async init() {
        await this.setupWebSocket();
        await this.setupEventListeners();
        await this.loadModels();
        await this.initializeCharts();
        await this.startSystemMonitoring();
    }

    // WebSocket Management
    async setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.showNotification('Connected to Aetheron Platform', 'success');
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.showNotification('Disconnected from server', 'warning');
                // Attempt reconnection after 5 seconds
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('Connection error', 'error');
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'training_update':
                this.updateTrainingMetrics(data.data);
                // Voice announcement for training progress
                if (window.aetheronVoice && data.data.epoch % 25 === 0) {
                    window.aetheronVoice.announceProgress(data.data.epoch, data.data.loss, data.data.accuracy);
                }
                break;
            case 'training_complete':
                this.handleTrainingComplete(data.data);
                // Voice announcement for completion
                if (window.aetheronVoice) {
                    window.aetheronVoice.announceCompletion(data.data.accuracy, data.data.training_time);
                }
                break;
            case 'system_metrics':
                this.updateSystemMetrics(data.data);
                break;
            case 'error':
                this.showNotification(data.message, 'error');
                // Voice announcement for errors
                if (window.aetheronVoice) {
                    window.aetheronVoice.announceError(data.message);
                }
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    // Event Listeners
    async setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Training form
        const trainForm = document.getElementById('trainForm');
        if (trainForm) {
            trainForm.addEventListener('submit', (e) => this.handleTrainSubmit(e));
        }

        // Prediction form
        const predictForm = document.getElementById('predictForm');
        if (predictForm) {
            predictForm.addEventListener('submit', (e) => this.handlePredictSubmit(e));
        }

        // File upload
        const fileInput = document.getElementById('dataFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Model type change
        const modelTypeSelect = document.getElementById('modelType');
        if (modelTypeSelect) {
            modelTypeSelect.addEventListener('change', (e) => this.handleModelTypeChange(e));
        }
    }

    // Tab Management
    switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Remove active class from all buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        // Show selected tab content
        const tabContent = document.getElementById(tabName);
        if (tabContent) {
            tabContent.classList.add('active');
        }

        // Add active class to selected button
        const tabBtn = document.querySelector(`[data-tab="${tabName}"]`);
        if (tabBtn) {
            tabBtn.classList.add('active');
        }

        // Voice announcement for tab switch
        if (window.aetheronVoice) {
            const tabNames = {
                train: 'Training module activated',
                analyze: 'Analytics dashboard online',
                compare: 'Model comparison interface loaded',
                deploy: 'Deployment center ready',
                data: 'Data management system active'
            };
            window.aetheronVoice.speak(tabNames[tabName] || 'Interface module loaded', 'ambient');
        }

        // Initialize tab-specific content
        this.initializeTabContent(tabName);
    }

    async initializeTabContent(tabName) {
        switch (tabName) {
            case 'train':
                await this.refreshModelsList();
                break;
            case 'analyze':
                await this.updateAnalyticsCharts();
                break;
            case 'compare':
                await this.updateModelComparison();
                break;
            case 'deploy':
                await this.updateDeploymentOptions();
                break;
            case 'data':
                await this.updateDataExploration();
                break;
        }
    }

    // Training Management
    async handleTrainSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const config = this.buildTrainingConfig(formData);
        
        try {
            this.setTrainingState('starting');
            
            // Voice announcement for training start
            if (window.aetheronVoice) {
                window.aetheronVoice.announceTrainingStart(config.model_name || 'neural network');
            }
            
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                throw new Error(`Training failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.currentTraining = result.model_name;
            this.setTrainingState('training');
            
            this.showNotification('Training started successfully!', 'success');
            
        } catch (error) {
            console.error('Training error:', error);
            this.showNotification(`Training failed: ${error.message}`, 'error');
            this.setTrainingState('idle');
            
            // Voice announcement for error
            if (window.aetheronVoice) {
                window.aetheronVoice.announceError(error.message);
            }
        }
    }

    buildTrainingConfig(formData) {
        const config = {
            model_name: formData.get('modelName'),
            model_type: formData.get('modelType'),
            config: {
                hidden_sizes: formData.get('hiddenSizes').split(',').map(s => parseInt(s.trim())),
                epochs: parseInt(formData.get('epochs')),
                learning_rate: parseFloat(formData.get('learningRate')),
                batch_size: parseInt(formData.get('batchSize')),
                optimizer: formData.get('optimizer'),
                dropout_rate: parseFloat(formData.get('dropoutRate')),
                l1_regularization: parseFloat(formData.get('l1Reg')),
                l2_regularization: parseFloat(formData.get('l2Reg')),
                target_column: 'target'
            }
        };

        // Add advanced config for advanced models
        if (config.model_type === 'advanced') {
            config.config.activation = formData.get('activation');
            config.config.architecture = formData.get('architecture');
        }

        return config;
    }

    setTrainingState(state) {
        const trainBtn = document.getElementById('trainBtn');
        const trainBtnText = document.getElementById('trainBtnText');
        const trainBtnLoading = document.getElementById('trainBtnLoading');
        const progressContainer = document.getElementById('trainingProgress');

        switch (state) {
            case 'starting':
                trainBtn.disabled = true;
                trainBtnText.classList.add('hidden');
                trainBtnLoading.classList.remove('hidden');
                break;
            case 'training':
                progressContainer.classList.remove('hidden');
                break;
            case 'complete':
            case 'idle':
                trainBtn.disabled = false;
                trainBtnText.classList.remove('hidden');
                trainBtnLoading.classList.add('hidden');
                if (state === 'complete') {
                    progressContainer.classList.add('hidden');
                }
                break;
        }
    }

    updateTrainingMetrics(data) {
        // Update metric displays
        document.getElementById('currentEpoch').textContent = data.epoch || 0;
        document.getElementById('trainLoss').textContent = (data.train_loss || 0).toFixed(4);
        document.getElementById('valLoss').textContent = (data.val_loss || 0).toFixed(4);
        document.getElementById('trainAccuracy').textContent = `${((data.train_accuracy || 0) * 100).toFixed(1)}%`;

        // Update progress bar
        if (data.progress !== undefined) {
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressFill.style.width = `${data.progress}%`;
            progressText.textContent = `Training: ${data.progress.toFixed(1)}%`;
        }

        // Update training chart
        this.updateTrainingChart(data);
    }

    updateTrainingChart(data) {
        const ctx = document.getElementById('trainingChart');
        if (!ctx || !data.epoch) return;

        if (!this.trainingCharts.main) {
            this.trainingCharts.main = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }, {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        const chart = this.trainingCharts.main;
        chart.data.labels.push(data.epoch);
        chart.data.datasets[0].data.push(data.train_loss);
        chart.data.datasets[1].data.push(data.val_loss || data.train_loss);
        
        // Keep only last 50 points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        chart.update('none');
    }

    // Prediction Management
    async handlePredictSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const modelName = formData.get('predictModel');
        const inputData = formData.get('inputData');

        try {
            const data = JSON.parse(inputData);
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    data: data
                })
            });

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayPredictionResults(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification(`Prediction failed: ${error.message}`, 'error');
        }
    }

    displayPredictionResults(result) {
        const resultDiv = document.getElementById('predictResult');
        if (resultDiv) {
            resultDiv.innerHTML = `
                <h4>Prediction Results</h4>
                <p><strong>Model:</strong> ${result.model_name}</p>
                <p><strong>Predictions:</strong> ${JSON.stringify(result.predictions)}</p>
                <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
            `;
            resultDiv.classList.remove('hidden');
        }
    }

    // File Upload Management
    async handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showNotification('Uploading file...', 'info');
            
            const response = await fetch('/data/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayDataInfo(result);
            this.showNotification('File uploaded successfully!', 'success');
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        }
    }

    displayDataInfo(result) {
        const dataInfo = result.data_info;
        const infoDiv = document.getElementById('dataInfo');
        
        if (infoDiv) {
            infoDiv.innerHTML = `
                <h4>Data Information</h4>
                <p><strong>Filename:</strong> ${dataInfo.filename}</p>
                <p><strong>Shape:</strong> ${dataInfo.shape.join(' × ')}</p>
                <p><strong>Columns:</strong> ${dataInfo.columns.join(', ')}</p>
                <p><strong>Missing Values:</strong> ${Object.values(dataInfo.missing_values).reduce((a, b) => a + b, 0)} total</p>
            `;
            infoDiv.classList.remove('hidden');
        }
    }

    // Model Management
    async loadModels() {
        try {
            const response = await fetch('/api/models/list');
            if (response.ok) {
                const result = await response.json();
                this.models = result.models;
                this.updateModelSelects();
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    updateModelSelects() {
        const selects = ['predictModel', 'deployModel'];
        
        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                // Clear existing options except first
                while (select.children.length > 1) {
                    select.removeChild(select.lastChild);
                }
                
                // Add model options
                this.models.forEach(model => {
                    if (model.status === 'trained') {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = `${model.name} (${model.type})`;
                        select.appendChild(option);
                    }
                });
            }
        });
    }

    // System Monitoring
    updateSystemMetrics(metrics) {
        // Update system metric displays if they exist
        const cpuElement = document.getElementById('cpuUsage');
        const memoryElement = document.getElementById('memoryUsage');
        
        if (cpuElement) cpuElement.textContent = `${metrics.cpu_usage.toFixed(1)}%`;
        if (memoryElement) memoryElement.textContent = `${metrics.memory_usage.toFixed(1)}%`;
    }

    async startSystemMonitoring() {
        // Start periodic system monitoring
        setInterval(async () => {
            try {
                const response = await fetch('/api/system/metrics');
                if (response.ok) {
                    const metrics = await response.json();
                    this.updateSystemMetrics(metrics);
                }
            } catch (error) {
                console.error('Failed to fetch system metrics:', error);
            }
        }, 5000);
    }

    // Chart Management
    async initializeCharts() {
        // Initialize performance chart
        const perfCtx = document.getElementById('performanceChart');
        if (perfCtx) {
            this.initializePerformanceChart();
        }

        // Initialize other charts as needed
        this.initializeDistributionChart();
        this.initializeResourceChart();
    }

    initializePerformanceChart() {
        // Plotly.js performance chart
        const data = [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy',
            line: { color: 'rgb(75, 192, 192)' }
        }];

        const layout = {
            title: 'Model Performance Over Time',
            xaxis: { title: 'Training Steps' },
            yaxis: { title: 'Accuracy' },
            showlegend: true
        };

        Plotly.newPlot('performanceChart', data, layout);
    }

    initializeDistributionChart() {
        // Placeholder for distribution chart
        const data = [{
            values: [1, 2, 3, 4, 5],
            labels: ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
            type: 'pie'
        }];

        const layout = {
            title: 'Prediction Distribution'
        };

        Plotly.newPlot('distributionChart', data, layout);
    }

    initializeResourceChart() {
        // Placeholder for resource monitoring chart
        const data = [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'CPU Usage',
            line: { color: 'rgb(255, 99, 132)' }
        }, {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'Memory Usage',
            line: { color: 'rgb(54, 162, 235)' }
        }];

        const layout = {
            title: 'System Resources',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Usage (%)' }
        };

        Plotly.newPlot('resourceChart', data, layout);
    }

    // Utility Methods
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" style="margin-left: 10px; background: none; border: none; color: inherit; cursor: pointer;">&times;</button>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    async refreshModelsList() {
        await this.loadModels();
    }

    async updateAnalyticsCharts() {
        // Update analytics charts with latest data
        // Implementation depends on specific analytics requirements
    }

    async updateModelComparison() {
        // Update model comparison view
        // Implementation for comparing multiple models
    }

    async updateDeploymentOptions() {
        // Update deployment configuration options
        // Implementation for deployment management
    }

    async updateDataExploration() {
        // Update data exploration charts and statistics
        // Implementation for data analysis visualization
    }

    handleModelTypeChange(e) {
        const modelType = e.target.value;
        const advancedConfig = document.getElementById('advancedConfig');
        
        if (advancedConfig) {
            if (modelType === 'advanced') {
                advancedConfig.style.display = 'block';
            } else {
                advancedConfig.style.display = 'none';
            }
        }
    }
}

// Enhanced Space Animation System
class SpaceAnimationSystem {
    constructor() {
        this.particles = [];
        this.maxParticles = 50;
        this.init();
    }

    init() {
        this.createParticleSystem();
        this.createFloatingShapes();
        this.createConstellation();
        this.startAnimations();
        this.addInteractiveEffects();
    }

    createParticleSystem() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        // Create particles
        for (let i = 0; i < this.maxParticles; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            
            // Random properties
            const size = Math.random() * 6 + 2;
            const startX = Math.random() * window.innerWidth;
            const animationDuration = Math.random() * 20 + 15;
            const delay = Math.random() * 15;
            const opacity = Math.random() * 0.8 + 0.2;
            
            particle.style.cssText = `
                left: ${startX}px;
                width: ${size}px;
                height: ${size}px;
                animation-duration: ${animationDuration}s;
                animation-delay: ${delay}s;
                opacity: ${opacity};
            `;
            
            // Add random colors for some particles
            if (Math.random() > 0.7) {
                const colors = [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(240, 147, 251, 0.8)',
                    'rgba(0, 255, 255, 0.6)',
                    'rgba(255, 255, 0, 0.6)'
                ];
                particle.style.background = colors[Math.floor(Math.random() * colors.length)];
                particle.style.boxShadow = `0 0 20px ${particle.style.background}`;
            }
            
            particlesContainer.appendChild(particle);
            this.particles.push(particle);
        }
    }

    createFloatingShapes() {
        const shapesContainer = document.getElementById('floatingShapes');
        if (!shapesContainer) return;

        const shapes = ['circle', 'triangle', 'square', 'diamond'];
        const numShapes = 8;

        for (let i = 0; i < numShapes; i++) {
            const shape = document.createElement('div');
            const shapeType = shapes[Math.floor(Math.random() * shapes.length)];
            
            shape.className = `floating-shape floating-${shapeType}`;
            
            const size = Math.random() * 100 + 50;
            const startX = Math.random() * window.innerWidth;
            const startY = Math.random() * window.innerHeight;
            const animationDuration = Math.random() * 30 + 20;
            const delay = Math.random() * 10;
            
            shape.style.cssText = `
                position: absolute;
                left: ${startX}px;
                top: ${startY}px;
                width: ${size}px;
                height: ${size}px;
                animation: floatShape ${animationDuration}s ease-in-out infinite ${delay}s;
                opacity: 0.1;
            `;
            
            shapesContainer.appendChild(shape);
        }

        // Add floating shape styles
        this.addFloatingShapeStyles();
    }

    addFloatingShapeStyles() {
        const styles = `
            <style>
            .floating-shape {
                pointer-events: none;
                filter: blur(1px);
            }
            
            .floating-circle {
                background: radial-gradient(circle, rgba(102, 126, 234, 0.3), transparent);
                border-radius: 50%;
            }
            
            .floating-triangle {
                background: linear-gradient(45deg, rgba(118, 75, 162, 0.3), transparent);
                clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
            }
            
            .floating-square {
                background: linear-gradient(45deg, rgba(240, 147, 251, 0.3), transparent);
                transform: rotate(45deg);
            }
            
            .floating-diamond {
                background: linear-gradient(45deg, rgba(0, 255, 255, 0.3), transparent);
                clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
            }
            
            @keyframes floatShape {
                0%, 100% {
                    transform: translateY(0px) rotate(0deg) scale(1);
                }
                25% {
                    transform: translateY(-30px) rotate(90deg) scale(1.1);
                }
                50% {
                    transform: translateY(-60px) rotate(180deg) scale(0.9);
                }
                75% {
                    transform: translateY(-30px) rotate(270deg) scale(1.1);
                }
            }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', styles);
    }

    startAnimations() {
        // Add parallax effect on mouse move
        document.addEventListener('mousemove', (e) => {
            const mouseX = e.clientX / window.innerWidth;
            const mouseY = e.clientY / window.innerHeight;
            
            // Move nebula
            const nebula = document.querySelector('.nebula');
            if (nebula) {
                nebula.style.transform = `translate(${mouseX * 20 - 10}px, ${mouseY * 20 - 10}px)`;
            }
            
            // Move grid
            const grid = document.querySelector('.space-grid');
            if (grid) {
                grid.style.transform = `translate(${mouseX * 10 - 5}px, ${mouseY * 10 - 5}px)`;
            }
            
            // Move particles slightly
            this.particles.forEach((particle, index) => {
                const factor = (index % 3 + 1) * 0.5;
                particle.style.transform = `translate(${mouseX * factor - factor/2}px, ${mouseY * factor - factor/2}px)`;
            });
        });

        // Add scroll-based animations
        window.addEventListener('scroll', () => {
            const scrollY = window.scrollY;
            const nebula = document.querySelector('.nebula');
            if (nebula) {
                nebula.style.transform = `translateY(${scrollY * 0.5}px)`;
            }
        });

        // Add window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }

    handleResize() {
        // Reposition particles on resize
        this.particles.forEach(particle => {
            const newX = Math.random() * window.innerWidth;
            particle.style.left = `${newX}px`;
        });
    }

    createConstellation() {
        const constellationContainer = document.getElementById('constellation');
        if (!constellationContainer) return;

        // Create constellation points and connecting lines
        const points = [];
        const numPoints = 12;
        
        // Generate random constellation points
        for (let i = 0; i < numPoints; i++) {
            const point = {
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                brightness: Math.random() * 0.8 + 0.2
            };
            points.push(point);
            
            // Create star point
            const star = document.createElement('div');
            star.style.cssText = `
                position: absolute;
                left: ${point.x}px;
                top: ${point.y}px;
                width: 3px;
                height: 3px;
                background: white;
                border-radius: 50%;
                box-shadow: 0 0 ${point.brightness * 20}px rgba(255, 255, 255, ${point.brightness});
                animation: starTwinkle ${2 + Math.random() * 3}s ease-in-out infinite;
            `;
            constellationContainer.appendChild(star);
        }
        
        // Connect nearby points with lines
        for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
                const point1 = points[i];
                const point2 = points[j];
                const distance = Math.sqrt(
                    Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
                );
                
                // Only connect close points
                if (distance < 200) {
                    this.createConstellationLine(point1, point2, constellationContainer);
                }
            }
        }

        // Add constellation line styles
        const styles = `
            <style>
            @keyframes starTwinkle {
                0%, 100% { opacity: 0.3; transform: scale(1); }
                25% { opacity: 0.8; transform: scale(1.2); }
                50% { opacity: 1; transform: scale(1); }
                75% { opacity: 0.6; transform: scale(0.8); }
            }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', styles);
    }

    createConstellationLine(point1, point2, container) {
        const line = document.createElement('div');
        const length = Math.sqrt(
            Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
        );
        const angle = Math.atan2(point2.y - point1.y, point2.x - point1.x) * 180 / Math.PI;
        
        line.className = 'constellation-line';
        line.style.cssText = `
            position: absolute;
            left: ${point1.x}px;
            top: ${point1.y}px;
            width: ${length}px;
            height: 1px;
            transform: rotate(${angle}deg);
            transform-origin: 0 0;
            animation-delay: ${Math.random() * 2}s;
        `;
        
        container.appendChild(line);
    }

    addInteractiveEffects() {
        // Add click ripple effects
        document.addEventListener('click', (e) => {
            this.createRippleEffect(e.clientX, e.clientY);
        });

        // Add touch support for mobile
        document.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                this.createRippleEffect(touch.clientX, touch.clientY);
            }
        });
    }

    createRippleEffect(x, y) {
        const ripple = document.createElement('div');
        ripple.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(102, 126, 234, 0.6);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            z-index: 9999;
        `;
        
        document.body.appendChild(ripple);
        
        ripple.animate([
            { 
                transform: 'translate(-50%, -50%) scale(0)',
                opacity: 1
            },
            { 
                transform: 'translate(-50%, -50%) scale(3)',
                opacity: 0
            }
        ], {
            duration: 600,
            easing: 'ease-out'
        }).onfinish = () => {
            ripple.remove();
        };
    }

    // Add cosmic dust effect
    createCosmicDust() {
        const dustContainer = document.createElement('div');
        dustContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        `;
        document.body.appendChild(dustContainer);

        for (let i = 0; i < 30; i++) {
            const dust = document.createElement('div');
            const size = Math.random() * 3 + 1;
            const x = Math.random() * window.innerWidth;
            const y = Math.random() * window.innerHeight;
            const duration = Math.random() * 20 + 10;
            
            dust.style.cssText = `
                position: absolute;
                left: ${x}px;
                top: ${y}px;
                width: ${size}px;
                height: ${size}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                animation: dustFloat ${duration}s ease-in-out infinite;
            `;
            
            dustContainer.appendChild(dust);
        }

        // Add dust animation styles
        const dustStyles = `
            <style>
            @keyframes dustFloat {
                0%, 100% {
                    transform: translateY(0px) translateX(0px) rotate(0deg);
                    opacity: 0.3;
                }
                25% {
                    transform: translateY(-20px) translateX(10px) rotate(90deg);
                    opacity: 0.7;
                }
                50% {
                    transform: translateY(-40px) translateX(-5px) rotate(180deg);
                    opacity: 1;
                }
                75% {
                    transform: translateY(-20px) translateX(-10px) rotate(270deg);
                    opacity: 0.7;
                }
            }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', dustStyles);
    }
}

// Enhanced UI Animation System
class UIAnimationSystem {
    constructor() {
        this.init();
    }

    init() {
        this.addScrollAnimations();
        this.addHoverEffects();
        this.addTypingEffect();
        this.addCounterAnimations();
    }

    addScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'slideInUp 0.8s ease-out forwards';
                }
            });
        }, observerOptions);

        document.querySelectorAll('.card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            observer.observe(card);
        });

        // Add slideInUp keyframes
        const slideInUpStyles = `
            <style>
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', slideInUpStyles);
    }

    addHoverEffects() {
        // Add enhanced hover effects to all interactive elements
        document.querySelectorAll('.nav-tab, .btn, .chart-btn').forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                e.target.style.transform = 'translateY(-3px) scale(1.05)';
                e.target.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            });
            
            element.addEventListener('mouseleave', (e) => {
                e.target.style.transform = 'translateY(0) scale(1)';
            });
        });
    }

    addTypingEffect() {
        const title = document.querySelector('h1');
        if (title) {
            const text = title.textContent;
            title.textContent = '';
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    title.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 100);
                }
            };
            
            setTimeout(typeWriter, 1000);
        }
    }

    addCounterAnimations() {
        const animateCounter = (element, target, duration = 2000) => {
            const start = 0;
            const increment = target / (duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(current);
            }, 16);
        };

        // Animate metrics when they become visible
        const metricsObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = parseInt(entry.target.textContent) || 100;
                    animateCounter(entry.target, target);
                    metricsObserver.unobserve(entry.target);
                }
            });
        });

        document.querySelectorAll('.training-metric-value').forEach(metric => {
            metricsObserver.observe(metric);
        });
    }
}

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
            'All systems operational. Neural networks maintaining optimal performance.',
            'Quantum processing cores stable. Ready for computational tasks.',
            'Data streams flowing normally. Memory banks at optimal capacity.',Sensor arrays online. Environmental monitoring active.',
            'Sensor arrays online. Environmental monitoring active.', 'Machine learning protocols engaged. Continuous improvement in progress.'
            'Machine learning protocols engaged. Continuous improvement in progress.'   ];
        ];

        setInterval(() => {sages in Jarvis style
            if (this.ambientMode && Math.random() > 0.7) {
                const message = ambientMessages[Math.floor(Math.random() * ambientMessages.length)];
                this.speak(message, 'ambient');
            }
        }, 45000); // Every 45 seconds, 30% chance
    }

    createVoiceUI() {        // Periodic status updates
        const voicePanel = document.createElement('div');
        voicePanel.id = 'voice-control-panel';
        voicePanel.innerHTML = `
            <div class="voice-controls">e. All modules functioning at peak efficiency.',
                <button id="voice-toggle" class="voice-btn ${this.isEnabled ? 'active' : ''}">       'Neural network infrastructure operating within optimal parameters, Sir.',
                    🔊 Voiceable. Standing by for your next directive.',
                </button>               'Data integrity verified. All systems green across the board.'
                <button id="ambient-toggle" class="voice-btn ${this.ambientMode ? 'active' : ''}">                ];
                    🌌 Ambient message = statusMessages[Math.floor(Math.random() * statusMessages.length)];
                </button>
                <select id="voice-selector" class="voice-select">
                    ${this.voices.map(v => inutes
                        `<option value="${v.name}" ${v === this.currentVoice ? 'selected' : ''}>${v.name}</option>`ive' : ''}">
                    ).join('')}
                </select>
            </div>d="ambient-toggle" class="voice-btn ${this.ambientMode ? 'active' : ''}">
        `;

        voicePanel.style.cssText = `d="voice-selector" class="voice-select">
            position: fixed;
            top: 20px;.name}" ${v === this.currentVoice ? 'selected' : ''}>${v.name}</option>`
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 10px;
            z-index: 10000;        voicePanel.style.cssText = `
            border: 1px solid rgba(0, 255, 247, 0.3);
        `;
x;
        document.body.appendChild(voicePanel);rgba(0, 0, 0, 0.8);

        // Event listeners
        document.getElementById('voice-toggle').addEventListener('click', () => {
            this.toggleVoice();;
        });id rgba(0, 255, 247, 0.3);

        document.getElementById('ambient-toggle').addEventListener('click', () => {
            this.toggleAmbient();        document.body.appendChild(voicePanel);
        });
        // Event listeners
        document.getElementById('voice-selector').addEventListener('change', (e) => {tById('voice-toggle').addEventListener('click', () => {
            this.currentVoice = this.voices.find(v => v.name === e.target.value);
            this.speak('Voice configuration updated', 'ambient');
        });
    }        document.getElementById('ambient-toggle').addEventListener('click', () => {

    toggleVoice() {
        this.isEnabled = !this.isEnabled;
        const btn = document.getElementById('voice-toggle');        document.getElementById('voice-selector').addEventListener('change', (e) => {
        btn.classList.toggle('active', this.isEnabled);
        
        if (this.isEnabled) {
            this.speak('Voice system enabled', 'ambient');
        } else {
            this.synthesis.cancel();    toggleVoice() {
        }led = !this.isEnabled;
    }Id('voice-toggle');

    toggleAmbient() {
        this.ambientMode = !this.ambientMode;if (this.isEnabled) {
        const btn = document.getElementById('ambient-toggle'); system enabled', 'ambient');
        btn.classList.toggle('active', this.ambientMode);
        .synthesis.cancel();
        this.speak(`Ambient notifications ${this.ambientMode ? 'enabled' : 'disabled'}`, 'ambient');
    }

    // Public methods for external use    toggleAmbient() {
    announceTrainingStart(modelName) {ode = !this.ambientMode;
        this.speak(`Commencing neural network training for model: ${modelName}. Initializing optimization algorithms.`, 'training');ambient-toggle');
    }

    announceProgress(epoch, loss, accuracy) {this.speak(`Ambient notifications ${this.ambientMode ? 'enabled' : 'disabled'}`, 'ambient');
        if (epoch % 25 === 0) {
            this.speak(`Training epoch ${epoch}. Loss: ${loss.toFixed(4)}. Accuracy: ${(accuracy * 100).toFixed(1)} percent.`, 'training');
        }    // Public methods for external use
    }
network training for model: ${modelName}. Initializing optimization algorithms.`, 'training');
    announceCompletion(finalAccuracy, trainingTime) {
        this.speak(`Training sequence complete. Final accuracy: ${(finalAccuracy * 100).toFixed(1)} percent. Training duration: ${trainingTime} seconds. Model ready for deployment.`, 'success');
    }    announceProgress(epoch, loss, accuracy) {

    announceError(errorMessage) {g epoch ${epoch}. Loss: ${loss.toFixed(4)}. Accuracy: ${(accuracy * 100).toFixed(1)} percent.`, 'training');
        this.speak(`System alert detected: ${errorMessage}. Please review system status.`, 'error');
    }

    speakCustom(message, type = 'normal') {    announceCompletion(finalAccuracy, trainingTime) {
        this.speak(message, type); accuracy: ${(finalAccuracy * 100).toFixed(1)} percent. Training duration: ${trainingTime} seconds. Model ready for deployment.`, 'success');
    }
}
    announceError(errorMessage) {
// Voice system stylesdetected: ${errorMessage}. Please review system status.`, 'error');
const voiceStyles = `
<style>
.voice-controls {    speakCustom(message, type = 'normal') {
    display: flex;
    gap: 8px;
    align-items: center;
}
// Voice system styles
.voice-btn {
    background: linear-gradient(135deg, rgba(0, 255, 247, 0.2), rgba(0, 136, 255, 0.2));
    border: 1px solid rgba(0, 255, 247, 0.4);controls {
    color: #00fff7;;
    padding: 6px 12px;
    border-radius: 8px;ms: center;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.3s ease;.voice-btn {
}nd: linear-gradient(135deg, rgba(0, 255, 247, 0.2), rgba(0, 136, 255, 0.2));

.voice-btn:hover {
    background: linear-gradient(135deg, rgba(0, 255, 247, 0.3), rgba(0, 136, 255, 0.3));px;
    box-shadow: 0 0 15px rgba(0, 255, 247, 0.3);;
}

.voice-btn.active {0.3s ease;
    background: linear-gradient(135deg, #00fff7, #0088ff);
    color: white;
    box-shadow: 0 0 20px rgba(0, 255, 247, 0.5);.voice-btn:hover {
}near-gradient(135deg, rgba(0, 255, 247, 0.3), rgba(0, 136, 255, 0.3));

.voice-select {
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(0, 255, 247, 0.4);.voice-btn.active {
    color: #00fff7;ear-gradient(135deg, #00fff7, #0088ff);
    padding: 4px 8px;
    border-radius: 6px; 0 20px rgba(0, 255, 247, 0.5);
    font-size: 11px;
    max-width: 120px;
}.voice-select {
 rgba(0, 0, 0, 0.7);
@keyframes voiceIndicatorPulse {247, 0.4);
    0%, 100% { opacity: 0.8; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.05); }x;
}x;

@keyframes cortanaSpeaking {;
    0% { 
        filter: brightness(1.3) hue-rotate(0deg);
        transform: scale(1);@keyframes voiceIndicatorPulse {
    }form: translateX(-50%) scale(1); }
    100% { 
        filter: brightness(1.8) hue-rotate(15deg);
        transform: scale(1.08);
    }@keyframes holoSpeaking {
}ss(1.2) hue-rotate(0deg); }
</style>; }
`;
/style>
document.head.insertAdjacentHTML('beforeend', voiceStyles);

// Initialize the platform when DOM is loadeddocument.head.insertAdjacentHTML('beforeend', voiceStyles);
let platform;
let aetheronVoice;// Initialize the platform when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    platform = new AetheronPlatform();oice;
    istener('DOMContentLoaded', () => {
    // Initialize voice system
    setTimeout(() => {
        aetheronVoice = new AetheronVoiceSystem();// Initialize voice system
        window.aetheronVoice = aetheronVoice; // Make globally accessible
    }, 1000);= new AetheronVoiceSystem();
    ake globally accessible
    // Initialize space animations after a short delay
    setTimeout(() => {
        if (platform) {// Initialize space animations after a short delay
            platform.spaceAnimations = new SpaceAnimationSystem();
            platform.uiAnimations = new UIAnimationSystem();{
            aceAnimations = new SpaceAnimationSystem();
            // Add cosmic dust effect
            setTimeout(() => {
                platform.spaceAnimations.createCosmicDust();// Add cosmic dust effect
            }, 2000);
            Animations.createCosmicDust();
            // Start meteor shower occasionally
            setInterval(() => {
                if (Math.random() > 0.7) {// Start meteor shower occasionally
                    platform.spaceAnimations.createMeteorShower();
                }() > 0.7) {
            }, 30000); // Every 30 seconds, 30% chancens.createMeteorShower();
            
            console.log('🚀 Enhanced Aetheron Platform animations initialized');000); // Every 30 seconds, 30% chance
        }
    }, 1000);console.log('🚀 Enhanced Aetheron Platform animations initialized');
});
00);
// Add notification styles
const notificationStyles = `
<style>// Add notification styles
.notification { `
    position: fixed;
    top: 20px;cation {
    right: 20px;ixed;
    padding: 15px 20px;
    border-radius: 8px;x;
    color: white;x 20px;
    font-weight: 500;
    z-index: 10000;
    animation: slideIn 0.3s ease-out;500;
    max-width: 400px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);eIn 0.3s ease-out;
}
 12px rgba(0, 0, 0, 0.2);
.notification-success { background-color: #10b981; }
.notification-error { background-color: #ef4444; }
.notification-warning { background-color: #f59e0b; }.notification-success { background-color: #10b981; }
.notification-info { background-color: #3b82f6; }
 }
@keyframes slideIn {
    from {
        transform: translateX(100%);@keyframes slideIn {
        opacity: 0;
    }ansform: translateX(100%);
    to {
        transform: translateX(0);
        opacity: 1;o {
    }transform: translateX(0);
}

.loading {
    display: inline-block;
    width: 20px;.loading {
    height: 20px;y: inline-block;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;;
    border-radius: 50%;olid #f3f3f3;
    animation: spin 1s linear infinite;8db;
}
linear infinite;
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }@keyframes spin {
}m: rotate(0deg); }
); }
.hidden {
    display: none !important;
}.hidden {
</style>ay: none !important;
`;
/style>
document.head.insertAdjacentHTML('beforeend', notificationStyles);
