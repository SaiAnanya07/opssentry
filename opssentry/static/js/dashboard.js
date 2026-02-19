// OpsSentry Dashboard JavaScript

const API_BASE = '';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadMetrics();
    loadPipelines();
    loadAlerts();
    
    // Set up prediction form
    const form = document.getElementById('predict-form');
    form.addEventListener('submit', handlePrediction);
    
    // Refresh data every 30 seconds
    setInterval(() => {
        loadStats();
        loadAlerts();
    }, 30000);
});

// Handle prediction form submission
async function handlePrediction(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    if (data.build_duration) data.build_duration = parseFloat(data.build_duration);
    if (data.num_jobs) data.num_jobs = parseInt(data.num_jobs);
    
    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayPredictionResult(result);
        
        // Refresh alerts if one was sent
        if (result.alert_sent) {
            setTimeout(loadAlerts, 1000);
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Failed to make prediction. Please ensure the model is trained.');
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const section = document.getElementById('result-section');
    const container = document.getElementById('prediction-result');
    
    const isFailure = result.prediction === 1;
    const className = isFailure ? 'failure' : 'success';
    
    container.className = `prediction-result ${className}`;
    container.innerHTML = `
        <div class="result-header">
            ${isFailure ? '❌' : '✅'} Prediction: ${result.prediction_label}
        </div>
        <div class="result-details">
            <div class="result-item">
                <div class="result-item-label">Failure Probability</div>
                <div class="result-item-value">${(result.failure_probability * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Success Probability</div>
                <div class="result-item-value">${(result.success_probability * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Risk Level</div>
                <div class="result-item-value">${result.risk_level}</div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Alert Sent</div>
                <div class="result-item-value">${result.alert_sent ? 'Yes' : 'No'}</div>
            </div>
        </div>
        ${result.recommendation ? `
            <div class="recommendation">
                <div class="recommendation-title">💡 Recommendations</div>
                <div>${result.recommendation}</div>
            </div>
        ` : ''}
    `;
    
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Load overall statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        document.getElementById('total-runs').textContent = data.total_runs || 0;
        document.getElementById('failure-rate').textContent = 
            data.overall_failure_rate ? `${(data.overall_failure_rate * 100).toFixed(1)}%` : '0%';
        document.getElementById('total-alerts').textContent = data.total_alerts || 0;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load model metrics
async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics`);
        const data = await response.json();
        
        document.getElementById('metric-accuracy').textContent = 
            data.accuracy ? data.accuracy.toFixed(3) : '-';
        document.getElementById('metric-precision').textContent = 
            data.precision ? data.precision.toFixed(3) : '-';
        document.getElementById('metric-recall').textContent = 
            data.recall ? data.recall.toFixed(3) : '-';
        document.getElementById('metric-f1').textContent = 
            data.f1 ? data.f1.toFixed(3) : '-';
        document.getElementById('metric-roc-auc').textContent = 
            data.roc_auc ? data.roc_auc.toFixed(3) : '-';
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Load monitored pipelines
async function loadPipelines() {
    try {
        const response = await fetch(`${API_BASE}/api/pipelines`);
        const data = await response.json();
        
        const container = document.getElementById('pipelines-list');
        
        if (!data.pipelines || data.pipelines.length === 0) {
            container.innerHTML = '<p style="color: var(--text-muted);">No pipelines found. Run data collection first.</p>';
            return;
        }
        
        container.innerHTML = data.pipelines.map(pipeline => `
            <div class="pipeline-item">
                <div class="pipeline-name">${pipeline.name}</div>
                <div class="pipeline-stats">
                    <div class="pipeline-stat">
                        <div class="pipeline-stat-value">${pipeline.total_runs}</div>
                        <div class="pipeline-stat-label">Total Runs</div>
                    </div>
                    <div class="pipeline-stat">
                        <div class="pipeline-stat-value">${(pipeline.failure_rate * 100).toFixed(1)}%</div>
                        <div class="pipeline-stat-label">Failure Rate</div>
                    </div>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading pipelines:', error);
    }
}

// Load recent alerts
async function loadAlerts() {
    try {
        const response = await fetch(`${API_BASE}/api/alerts?limit=10`);
        const data = await response.json();
        
        const container = document.getElementById('alerts-list');
        
        if (!data.alerts || data.alerts.length === 0) {
            container.innerHTML = '<p style="color: var(--text-muted);">No alerts yet. Make a prediction to generate alerts.</p>';
            return;
        }
        
        container.innerHTML = data.alerts.reverse().map(alert => {
            const riskClass = alert.risk_level.toLowerCase();
            const timestamp = new Date(alert.timestamp).toLocaleString();
            
            return `
                <div class="alert-item ${riskClass}">
                    <div class="alert-header">
                        <div class="alert-title">
                            ${getRiskIcon(alert.risk_level)} ${alert.run_name || 'Unknown Pipeline'}
                        </div>
                        <div class="alert-time">${timestamp}</div>
                    </div>
                    <div class="alert-details">
                        <strong>Prediction:</strong> ${alert.prediction_label}<br>
                        <strong>Failure Probability:</strong> ${(alert.failure_probability * 100).toFixed(1)}%<br>
                        ${alert.recommendation ? `<br><strong>Recommendation:</strong> ${alert.recommendation}` : ''}
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

// Get icon for risk level
function getRiskIcon(riskLevel) {
    const icons = {
        'CRITICAL': '🔴',
        'WARNING': '⚠️',
        'INFO': 'ℹ️',
        'LOW': '✅'
    };
    return icons[riskLevel] || '❓';
}

// Format percentage
function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// Format number
function formatNumber(value) {
    return value.toLocaleString();
}
