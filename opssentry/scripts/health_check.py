"""
Health Check Script for Jenkins CI/CD
Verifies that the OpsSentry application is running and responding correctly.
"""
import sys
import requests
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 10
MAX_RETRIES = 3


def check_health_endpoint() -> bool:
    """Check if the health endpoint is responding."""
    print("=" * 60)
    print("HEALTH CHECK: API Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/health"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Attempt {attempt}/{MAX_RETRIES}: Checking {url}")
            response = requests.get(url, timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health endpoint responding")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Predictor loaded: {data.get('predictor_loaded', False)}")
                print(f"  Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"✗ Health endpoint returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Connection failed - application may not be running")
        except requests.exceptions.Timeout:
            print(f"✗ Request timed out")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        if attempt < MAX_RETRIES:
            print(f"  Retrying in 5 seconds...")
            time.sleep(5)
    
    return False


def check_prediction_endpoint() -> bool:
    """Check if the prediction endpoint is working."""
    print("\n" + "=" * 60)
    print("HEALTH CHECK: Prediction Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/predict"
    
    # Sample prediction data
    test_data = {
        "name": "Test Pipeline",
        "event": "push",
        "build_duration": 300.0,
        "num_jobs": 3,
        "hour_of_day": 14,
        "is_weekend": 0
    }
    
    try:
        print(f"Testing prediction endpoint: {url}")
        response = requests.post(url, json=test_data, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction endpoint responding")
            print(f"  Prediction: {data.get('prediction_label', 'unknown')}")
            print(f"  Failure probability: {data.get('failure_probability', 0):.2%}")
            print(f"  Risk level: {data.get('risk_level', 'unknown')}")
            return True
        else:
            print(f"✗ Prediction endpoint returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing prediction endpoint: {e}")
        return False


def check_metrics_endpoint() -> bool:
    """Check if the metrics endpoint is accessible."""
    print("\n" + "=" * 60)
    print("HEALTH CHECK: Metrics Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/api/metrics"
    
    try:
        print(f"Checking metrics endpoint: {url}")
        response = requests.get(url, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Metrics endpoint responding")
            print(f"  Accuracy: {data.get('accuracy', 0):.4f}")
            print(f"  F1 Score: {data.get('f1', 0):.4f}")
            return True
        else:
            print(f"⚠ Metrics endpoint returned status {response.status_code}")
            # Metrics endpoint is optional, so we don't fail on this
            return True
            
    except Exception as e:
        print(f"⚠ Warning: Could not check metrics endpoint: {e}")
        # Metrics endpoint is optional
        return True


def main():
    """Main health check function."""
    print("=" * 60)
    print("OPSSENTRY HEALTH CHECK")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Max retries: {MAX_RETRIES}")
    print()
    
    results = {
        'health': check_health_endpoint(),
        'prediction': False,
        'metrics': False
    }
    
    # Only check other endpoints if health check passed
    if results['health']:
        results['prediction'] = check_prediction_endpoint()
        results['metrics'] = check_metrics_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("HEALTH CHECK SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check.upper()}: {status}")
    
    print("=" * 60)
    
    # Exit with appropriate code
    if results['health'] and results['prediction']:
        print("✓ APPLICATION IS HEALTHY")
        sys.exit(0)
    else:
        print("✗ APPLICATION HEALTH CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
