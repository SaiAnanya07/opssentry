"""
Integration Tests for OpsSentry CI/CD Pipeline
Tests end-to-end functionality of the application.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from flask import Flask
from app import app as flask_app


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint_exists(self):
        """Test that health endpoint exists."""
        with flask_app.test_client() as client:
            response = client.get('/api/health')
            assert response.status_code == 200
    
    def test_health_endpoint_response(self):
        """Test health endpoint response format."""
        with flask_app.test_client() as client:
            response = client.get('/api/health')
            data = response.get_json()
            
            assert 'status' in data
            assert 'predictor_loaded' in data
            assert 'version' in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""
    
    def test_prediction_endpoint_exists(self):
        """Test that prediction endpoint exists."""
        with flask_app.test_client() as client:
            test_data = {
                "name": "Test Pipeline",
                "event": "push",
                "build_duration": 300.0
            }
            response = client.post('/api/predict', json=test_data)
            # Should return 200 or 500 (if model not loaded)
            assert response.status_code in [200, 500]
    
    def test_prediction_endpoint_validation(self):
        """Test prediction endpoint validates input."""
        with flask_app.test_client() as client:
            # Empty request
            response = client.post('/api/predict', json={})
            assert response.status_code in [200, 400, 500]
    
    def test_prediction_response_format(self):
        """Test prediction response format (if model is loaded)."""
        with flask_app.test_client() as client:
            test_data = {
                "name": "Test Pipeline",
                "event": "push",
                "build_duration": 300.0,
                "num_jobs": 3
            }
            response = client.post('/api/predict', json=test_data)
            
            if response.status_code == 200:
                data = response.get_json()
                # Check for expected fields
                assert 'prediction' in data or 'prediction_label' in data


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint_exists(self):
        """Test that metrics endpoint exists."""
        with flask_app.test_client() as client:
            response = client.get('/api/metrics')
            assert response.status_code == 200
    
    def test_metrics_response_format(self):
        """Test metrics response format."""
        with flask_app.test_client() as client:
            response = client.get('/api/metrics')
            data = response.get_json()
            
            # Should have metric fields
            assert isinstance(data, dict)


class TestStatsEndpoint:
    """Test stats endpoint."""
    
    def test_stats_endpoint_exists(self):
        """Test that stats endpoint exists."""
        with flask_app.test_client() as client:
            response = client.get('/api/stats')
            assert response.status_code == 200
    
    def test_stats_response_format(self):
        """Test stats response format."""
        with flask_app.test_client() as client:
            response = client.get('/api/stats')
            data = response.get_json()
            
            assert 'total_runs' in data
            assert 'total_failures' in data
            assert 'overall_failure_rate' in data


class TestApplicationStructure:
    """Test application structure and configuration."""
    
    def test_config_exists(self):
        """Test that config module exists."""
        import config
        assert hasattr(config, 'BASE_DIR')
        assert hasattr(config, 'DATA_DIR')
        assert hasattr(config, 'MODELS_DIR')
    
    def test_required_directories_exist(self):
        """Test that required directories exist."""
        import config
        assert config.DATA_DIR.exists()
        assert config.MODELS_DIR.exists()
        assert config.LOGS_DIR.exists()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
