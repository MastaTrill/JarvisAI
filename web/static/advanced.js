// Advanced Jarvis AI Web Interface JavaScript
// Continuation of the advanced.html JavaScript

// Event listeners setup
function setupEventListeners() {
    // Train form handler
    document.getElementById('trainForm').addEventListener('submit', handleTrainSubmit);
    
    // Predict form handler
    document.getElementById('predictForm').addEventListener('submit', handlePredictSubmit);
    
    // Upload form handler
    document.getElementById('uploadForm').addEventListener('submit', handleUploadSubmit);
    
    // Modal close handler
    document.getElementsByClassName('close')[0].onclick = function() {
        document.getElementById('modelModal').style.display = 'none';
    };
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('modelModal');
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
}

// Handle train form submission
async function handleTrainSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const modelName = formData.get('modelName');
    const modelType = formData.get('modelType');
    
    // Basic configuration
    const hiddenSizes = formData.get('hiddenSizes').split(',').map(s => parseInt(s.trim()));
    const epochs = parseInt(formData.get('epochs'));
    const learningRate = parseFloat(formData.get('learningRate'));
    
    // Build configuration object
    const config = {
        hidden_sizes: hiddenSizes,
        epochs: epochs,
        learning_rate: learningRate,
        batch_size: parseInt(formData.get('batchSize')) || 32,
        target_column: 'target'
    };
    
    // Add advanced configuration if model type is advanced
    if (modelType === 'advanced') {
        config.activation = formData.get('activation');
        config.optimizer = formData.get('optimizer');
        config.dropout_rate = parseFloat(formData.get('dropoutRate'));
        config.l1_reg = parseFloat(formData.get('l1Reg'));
        config.l2_reg = parseFloat(formData.get('l2Reg'));
    }
    
    showLoading('trainBtn', 'trainBtnLoading', 'trainBtnText');
    hideStatus('trainStatus');
    
    try {
        const response = await fetch(`/models/${modelName}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: modelName,
                model_type: modelType,
                config: config
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('trainStatus', result.message, 'success');
            // Start training progress monitoring
            startTrainingMonitor(modelName);
        } else {
            showStatus('trainStatus', result.detail || 'Training failed', 'error');
        }
    } catch (error) {
        showStatus('trainStatus', `Error: ${error.message}`, 'error');
    } finally {
        hideLoading('trainBtn', 'trainBtnLoading', 'trainBtnText');
    }
}

// Training progress monitoring
let trainingInterval = null;

function startTrainingMonitor(modelName) {
    document.getElementById('trainingProgress').classList.remove('hidden');
    
    trainingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/models/${modelName}/status`);
            const status = await response.json();
            
            updateTrainingProgress(status);
            
            if (status.status === 'completed') {
                clearInterval(trainingInterval);
                showStatus('trainStatus', 'Training completed successfully! 🎉', 'success');
                document.getElementById('trainingProgress').classList.add('hidden');
                loadModels(); // Refresh models list
                loadTrainingChart(modelName); // Load training visualization
            } else if (status.status === 'failed') {
                clearInterval(trainingInterval);
                showStatus('trainStatus', `Training failed: ${status.error}`, 'error');
                document.getElementById('trainingProgress').classList.add('hidden');
            }
        } catch (error) {
            console.error('Error monitoring training:', error);
        }
    }, 2000);
}

function updateTrainingProgress(status) {
    const progress = status.progress || 0;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `Training: ${progress}%`;
    
    if (status.status === 'training') {
        showStatus('trainStatus', `Training in progress... ${progress}%`, 'info');
    }
}

// Load training visualization
async function loadTrainingChart(modelName) {
    try {
        const response = await fetch(`/models/${modelName}/metrics`);
        const data = await response.json();
        
        if (data.metrics && data.metrics.history) {
            renderTrainingChart(data.metrics.history);
        }
    } catch (error) {
        console.error('Error loading training chart:', error);
    }
}

function renderTrainingChart(history) {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    const epochs = Object.keys(history.train_loss || {}).map(Number);
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: Object.values(history.train_loss || {}),
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Validation Loss',
                    data: Object.values(history.val_loss || {}),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Training Progress'
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
}

// Handle predict form submission
async function handlePredictSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const modelName = formData.get('predModelName');
    const inputMethod = document.getElementById('inputMethod').value;
    
    if (!modelName) {
        showStatus('predictStatus', 'Please select a model', 'error');
        return;
    }
    
    let inputData;
    
    if (inputMethod === 'manual' || inputMethod === 'sample') {
        const inputDataStr = formData.get('inputData');
        try {
            inputData = JSON.parse(inputDataStr);
        } catch (error) {
            showStatus('predictStatus', 'Invalid JSON format for input data', 'error');
            return;
        }
    } else if (inputMethod === 'file') {
        const file = document.getElementById('predictionFile').files[0];
        if (!file) {
            showStatus('predictStatus', 'Please select a file', 'error');
            return;
        }
        
        // Parse CSV file
        try {
            const text = await file.text();
            inputData = parseCSVForPrediction(text);
        } catch (error) {
            showStatus('predictStatus', 'Error reading file', 'error');
            return;
        }
    }
    
    showLoading('predictBtn', 'predictBtnLoading', 'predictBtnText');
    hideStatus('predictStatus');
    
    try {
        const response = await fetch(`/models/${modelName}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: modelName,
                data: inputData
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('predictStatus', 'Predictions generated successfully! 🎯', 'success');
            displayPredictions(result.predictions, inputData);
            renderPredictionChart(result.predictions, inputData);
        } else {
            showStatus('predictStatus', result.detail || 'Prediction failed', 'error');
        }
    } catch (error) {
        showStatus('predictStatus', `Error: ${error.message}`, 'error');
    } finally {
        hideLoading('predictBtn', 'predictBtnLoading', 'predictBtnText');
    }
}

// Parse CSV for predictions
function parseCSVForPrediction(csvText) {
    const lines = csvText.trim().split('\n');
    const data = [];
    
    // Skip header row, parse data rows
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(val => parseFloat(val.trim()));
        if (values.every(val => !isNaN(val))) {
            data.push(values);
        }
    }
    
    return data;
}

// Display predictions
function displayPredictions(predictions, inputData) {
    const resultsEl = document.getElementById('predictResults');
    
    const resultsHTML = `
        <h4>🎯 Prediction Results</h4>
        <div style="max-height: 300px; overflow-y: auto;">
            ${predictions.map((pred, i) => `
                <div class="prediction-item">
                    <span><strong>Sample ${i + 1}:</strong></span>
                    <span style="font-weight: 600; color: var(--primary-color);">${pred.toFixed(4)}</span>
                </div>
            `).join('')}
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #f0f9ff; border-radius: 8px;">
            <strong>Statistics:</strong>
            <br>
            Mean: ${(predictions.reduce((a, b) => a + b, 0) / predictions.length).toFixed(4)}
            <br>
            Min: ${Math.min(...predictions).toFixed(4)} | 
            Max: ${Math.max(...predictions).toFixed(4)}
        </div>
    `;
    
    resultsEl.innerHTML = resultsHTML;
    resultsEl.classList.remove('hidden');
}

// Render prediction chart
function renderPredictionChart(predictions, inputData) {
    const data = predictions.map((pred, i) => ({
        x: i + 1,
        y: pred,
        type: 'scatter',
        mode: 'markers+lines',
        marker: { 
            color: '#667eea',
            size: 8
        },
        line: {
            color: '#667eea'
        },
        name: 'Predictions'
    }));
    
    const layout = {
        title: 'Prediction Results',
        xaxis: { title: 'Sample Index' },
        yaxis: { title: 'Predicted Value' },
        margin: { t: 50, r: 20, b: 50, l: 60 }
    };
    
    Plotly.newPlot('predictionChart', [data], layout, {responsive: true});
}

// Handle file upload
async function handleUploadSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    
    showLoading('uploadBtn', 'uploadBtnLoading', 'uploadBtnText');
    hideStatus('uploadStatus');
    
    try {
        const response = await fetch('/data/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('uploadStatus', `File uploaded successfully! Shape: ${result.data_info.shape} 📊`, 'success');
            displayDataPreview(result.data_info);
            displayQualityReport(result.quality_report);
        } else {
            showStatus('uploadStatus', result.detail || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `Error: ${error.message}`, 'error');
    } finally {
        hideLoading('uploadBtn', 'uploadBtnLoading', 'uploadBtnText');
    }
}

// Display data preview
function displayDataPreview(dataInfo) {
    const previewEl = document.getElementById('dataPreview');
    
    const previewHTML = `
        <h4>📋 Data Summary</h4>
        <p><strong>Shape:</strong> ${dataInfo.shape[0]} rows × ${dataInfo.shape[1]} columns</p>
        <p><strong>Columns:</strong> ${dataInfo.columns.join(', ')}</p>
        <h5>Data Types:</h5>
        <pre>${JSON.stringify(dataInfo.data_types, null, 2)}</pre>
    `;
    
    previewEl.innerHTML = previewHTML;
    previewEl.classList.remove('hidden');
}

// Display quality report
function displayQualityReport(report) {
    const reportEl = document.getElementById('qualityReport');
    
    const missingValues = report.missing_values;
    const outliers = report.outliers;
    
    const reportHTML = `
        <h4>🔍 Data Quality Assessment</h4>
        
        <div style="margin: 15px 0;">
            <h5>📊 Missing Values</h5>
            ${missingValues.total_missing === 0 ? 
                '<p style="color: var(--success-color);">✅ No missing values found</p>' :
                `<p style="color: var(--warning-color);">⚠️ ${missingValues.total_missing} missing values found</p>
                 <ul>${Object.entries(missingValues.missing_by_column)
                    .filter(([col, count]) => count > 0)
                    .map(([col, count]) => `<li>${col}: ${count} (${missingValues.missing_percentages[col].toFixed(1)}%)</li>`)
                    .join('')}</ul>`
            }
        </div>
        
        <div style="margin: 15px 0;">
            <h5>📈 Outliers Detected</h5>
            ${Object.keys(outliers).length === 0 ?
                '<p style="color: var(--success-color);">✅ No outliers detected</p>' :
                `<ul>${Object.entries(outliers)
                    .map(([col, info]) => `<li>${col}: ${info.count} outliers (${info.percentage.toFixed(1)}%)</li>`)
                    .join('')}</ul>`
            }
        </div>
        
        <div style="margin: 15px 0;">
            <h5>💾 Dataset Info</h5>
            <p>Memory Usage: ${(report.dataset_info.memory_usage / 1024).toFixed(1)} KB</p>
            <p>Generated: ${new Date(report.timestamp).toLocaleString()}</p>
        </div>
    `;
    
    reportEl.innerHTML = reportHTML;
}

// Load models
async function loadModels() {
    try {
        const response = await fetch('/models');
        models = await response.json();
        
        updateModelSelect(models);
        return models;
    } catch (error) {
        console.error('Error loading models:', error);
        return [];
    }
}

// Update model select dropdown
function updateModelSelect(models) {
    const selectEl = document.getElementById('predModelName');
    const trainedModels = models.filter(m => m.status === 'trained');
    
    selectEl.innerHTML = '<option value="">Choose a trained model...</option>' +
        trainedModels.map(model => `<option value="${model.name}">${model.name}</option>`).join('');
}

// Show model details modal
async function showModelDetails(modelName) {
    try {
        const response = await fetch(`/models/${modelName}/metrics`);
        const data = await response.json();
        
        document.getElementById('modalModelName').textContent = data.model_name;
        
        const modalContent = `
            <div style="margin: 20px 0;">
                <h4>📊 Model Information</h4>
                <p><strong>Type:</strong> ${data.type}</p>
                <p><strong>Status:</strong> ${data.status}</p>
                <p><strong>Created:</strong> ${new Date(data.created_at).toLocaleString()}</p>
            </div>
            
            ${data.metrics ? `
                <div style="margin: 20px 0;">
                    <h4>📈 Performance Metrics</h4>
                    <div class="metrics-grid">
                        <div class="metric">
                            <span class="metric-value">${data.metrics.train_r2?.toFixed(4) || 'N/A'}</span>
                            <span class="metric-label">Train R²</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">${data.metrics.test_r2?.toFixed(4) || 'N/A'}</span>
                            <span class="metric-label">Test R²</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">${data.metrics.train_mse?.toFixed(4) || 'N/A'}</span>
                            <span class="metric-label">Train MSE</span>
                        </div>
                        <div class="metric">
                            <span class="metric-value">${data.metrics.test_mse?.toFixed(4) || 'N/A'}</span>
                            <span class="metric-label">Test MSE</span>
                        </div>
                    </div>
                </div>
            ` : ''}
            
            <div style="margin: 20px 0;">
                <h4>⚙️ Configuration</h4>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto;">${JSON.stringify(data.config, null, 2)}</pre>
            </div>
        `;
        
        document.getElementById('modalContent').innerHTML = modalContent;
        document.getElementById('modelModal').style.display = 'block';
    } catch (error) {
        console.error('Error loading model details:', error);
    }
}

// Load analytics
async function loadAnalytics() {
    try {
        await loadModels();
        renderComparisonChart();
        renderHistoryChart();
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Render comparison chart
function renderComparisonChart() {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const trainedModels = models.filter(m => m.status === 'trained' && m.metrics);
    
    if (trainedModels.length === 0) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.font = '16px Inter';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';
        ctx.fillText('No trained models to compare', ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: trainedModels.map(m => m.name),
            datasets: [
                {
                    label: 'R² Score',
                    data: trainedModels.map(m => m.metrics.test_r2 || 0),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                },
                {
                    label: 'MSE (×10⁻²)',
                    data: trainedModels.map(m => (m.metrics.test_mse || 0) * 100),
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Score / Loss'
                    }
                }
            }
        }
    });
}

// Render history chart
function renderHistoryChart() {
    const historyData = models
        .filter(m => m.status === 'trained')
        .map(m => ({
            x: new Date(m.created_at),
            y: m.metrics?.test_r2 || 0,
            name: m.name
        }))
        .sort((a, b) => a.x - b.x);
    
    if (historyData.length === 0) {
        document.getElementById('historyChart').innerHTML = 
            '<p style="text-align: center; color: #6b7280; padding: 40px;">No training history available</p>';
        return;
    }
    
    const trace = {
        x: historyData.map(d => d.x),
        y: historyData.map(d => d.y),
        type: 'scatter',
        mode: 'markers+lines',
        marker: { 
            color: '#667eea',
            size: 10
        },
        line: {
            color: '#667eea',
            width: 2
        },
        text: historyData.map(d => d.name),
        hovertemplate: '<b>%{text}</b><br>R² Score: %{y:.4f}<br>Date: %{x}<extra></extra>'
    };
    
    const layout = {
        title: 'Model Performance Over Time',
        xaxis: { title: 'Training Date' },
        yaxis: { title: 'R² Score' },
        margin: { t: 50, r: 20, b: 50, l: 60 }
    };
    
    Plotly.newPlot('historyChart', [trace], layout, {responsive: true});
}

// Auto-refresh models every 30 seconds
setInterval(async () => {
    if (currentTab === 'dashboard') {
        await loadModels();
        updateDashboardStats();
        displayDashboardModels();
    }
}, 30000);
