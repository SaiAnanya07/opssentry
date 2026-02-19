"""
OpsSentry Flask Application
REST API and Web Dashboard for CI/CD Pipeline Failure Prediction
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import config
from utils.logger import setup_logger
from models.predictor import Predictor
from alert.alert_manager import AlertManager
from alert.recommendation_engine import RecommendationEngine

# Initialize Flask app
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Initialize logger
logger = setup_logger(__name__, "app.log")

# Initialize components
predictor = None
alert_manager = AlertManager()
recommendation_engine = RecommendationEngine()


def init_predictor():
    """Initialize predictor (lazy loading)."""
    global predictor
    if predictor is None:
        try:
            predictor = Predictor()
            logger.info("✓ Predictor initialized")
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize predictor: {e}")
            predictor = None
    return predictor


@app.route('/')
def index():
    """Serve main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    pred = init_predictor()
    
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': pred is not None,
        'version': '1.0.0'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make prediction for a pipeline run.
    
    Request body:
    {
        "name": "CI Pipeline",
        "event": "push",
        "build_duration": 450.0,
        "num_jobs": 3,
        ...
    }
    
    Response:
    {
        "prediction": 0,
        "prediction_label": "SUCCESS",
        "failure_probability": 0.25,
        "risk_level": "LOW",
        "recommendation": "..."
    }
    """
    try:
        pred = init_predictor()
        if pred is None:
            return jsonify({
                'error': 'Predictor not initialized. Please train a model first.'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = pred.predict_pipeline_run(data)
        
        # Process alert if needed
        alert_sent = alert_manager.process_prediction(result)
        result['alert_sent'] = alert_sent
        
        logger.info(f"Prediction made for {data.get('name', 'unknown')}: {result['prediction_label']}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple pipeline runs.
    
    Request body:
    {
        "runs": [
            {"name": "CI Pipeline", "build_duration": 450.0, ...},
            {"name": "Deploy Pipeline", "build_duration": 300.0, ...}
        ]
    }
    
    Response:
    {
        "predictions": [...],
        "total": 2,
        "high_risk_count": 1
    }
    """
    try:
        pred = init_predictor()
        if pred is None:
            return jsonify({
                'error': 'Predictor not initialized. Please train a model first.'
            }), 500
        
        data = request.get_json()
        runs = data.get('runs', [])
        
        if not runs:
            return jsonify({'error': 'No runs provided'}), 400
        
        # Make predictions
        df = pd.DataFrame(runs)
        result_df = pred.batch_predict(df)
        
        predictions = result_df.to_dict('records')
        high_risk_count = sum(1 for p in predictions if p.get('risk_level') in ['CRITICAL', 'WARNING'])
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'high_risk_count': high_risk_count
        })
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get model performance metrics.
    
    Response:
    {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "roc_auc": 0.90
    }
    """
    try:
        # Try to load metrics from saved file or return placeholder
        metrics_file = config.MODELS_DIR / 'metrics.json'
        
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            # Placeholder metrics
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'note': 'Train a model to see actual metrics'
            }
        
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipelines', methods=['GET'])
def get_pipelines():
    """
    Get list of monitored pipelines.
    
    Response:
    {
        "pipelines": [
            {"name": "CI Pipeline", "total_runs": 100, "failure_rate": 0.15},
            ...
        ]
    }
    """
    try:
        # Load GitHub runs metadata
        if not config.RUNS_METADATA_CSV.exists():
            return jsonify({'pipelines': []})
        
        df = pd.read_csv(config.RUNS_METADATA_CSV)
        
        # Group by pipeline name
        if 'name' in df.columns:
            pipeline_stats = df.groupby('name').agg({
                'id': 'count',
                'conclusion': lambda x: (x == 'failure').sum() / len(x) if len(x) > 0 else 0
            }).reset_index()
            
            pipeline_stats.columns = ['name', 'total_runs', 'failure_rate']
            pipelines = pipeline_stats.to_dict('records')
        else:
            pipelines = []
        
        return jsonify({'pipelines': pipelines})
    
    except Exception as e:
        logger.error(f"Error getting pipelines: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    Get recent alerts.
    
    Query params:
    - limit: Number of alerts to return (default: 10)
    
    Response:
    {
        "alerts": [...],
        "total": 10
    }
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        alerts = alert_manager.get_alert_history(limit=limit)
        
        return jsonify({
            'alerts': alerts,
            'total': len(alerts)
        })
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendation', methods=['POST'])
def get_recommendation():
    """
    Get recommendation for a pipeline run.
    
    Request body:
    {
        "run_data": {...},
        "prediction": {...}
    }
    
    Response:
    {
        "recommendation": "..."
    }
    """
    try:
        data = request.get_json()
        run_data = data.get('run_data', {})
        prediction = data.get('prediction', {})
        
        recommendation = recommendation_engine.generate_comprehensive_recommendations(
            run_data, prediction
        )
        
        return jsonify({'recommendation': recommendation})
    
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get overall statistics.
    
    Response:
    {
        "total_runs": 1000,
        "total_failures": 150,
        "overall_failure_rate": 0.15,
        "total_alerts": 45
    }
    """
    try:
        stats = {
            'total_runs': 0,
            'total_failures': 0,
            'overall_failure_rate': 0.0,
            'total_alerts': len(alert_manager.alert_history)
        }
        
        # Load runs metadata
        if config.RUNS_METADATA_CSV.exists():
            df = pd.read_csv(config.RUNS_METADATA_CSV)
            stats['total_runs'] = len(df)
            
            if 'conclusion' in df.columns:
                stats['total_failures'] = (df['conclusion'] == 'failure').sum()
                stats['overall_failure_rate'] = stats['total_failures'] / stats['total_runs'] if stats['total_runs'] > 0 else 0
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting OpsSentry Server")
    logger.info("=" * 60)
    logger.info(f"Host: {config.FLASK_CONFIG['host']}")
    logger.info(f"Port: {config.FLASK_CONFIG['port']}")
    logger.info("=" * 60)
    
    app.run(
        host=config.FLASK_CONFIG['host'],
        port=config.FLASK_CONFIG['port'],
        debug=config.FLASK_CONFIG['debug']
    )
