"""
============================================================================
API TESTING SCRIPT FOR TMKOC PREDICTION API
============================================================================
Run this after starting the API to test all endpoints
Usage: python test_api.py
============================================================================
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def test_root():
    """Test root endpoint"""
    print_section("TEST 1: Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test health check endpoint"""
    print_section("TEST 2: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_features():
    """Test features endpoint"""
    print_section("TEST 3: Get Features")
    try:
        response = requests.get(f"{BASE_URL}/features")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Feature Count: {data.get('feature_count')}")
        print(f"Features: {data.get('features')[:5]}... (showing first 5)")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_section("TEST 4: Model Information")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_default():
    """Test prediction with default values"""
    print_section("TEST 5: Prediction - Default Values")
    
    payload = {
        "lead_cast_matches": 3,
        "supporting_cast_matches": 5,
        "desc_length": 300,
        "desc_word_count": 50,
        "question_count": 2,
        "exclamation_count": 1,
        "has_conflict": True,
        "has_main_char": True,
        "has_emotion": False,
        "has_society": True,
        "has_mystery": False,
        "has_family": True,
        "runtime_minutes": 22.0,
        "release_year": 2024,
        "release_month": 10,
        "release_day": 15,
        "is_weekend": False
    }
    
    try:
        print("Request Payload:")
        print(json.dumps(payload, indent=2))
        print()
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"\nPrediction Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_high_success():
    """Test prediction optimized for high success"""
    print_section("TEST 6: Prediction - High Success Scenario")
    
    payload = {
        "lead_cast_matches": 5,
        "supporting_cast_matches": 8,
        "desc_length": 450,
        "desc_word_count": 75,
        "question_count": 3,
        "exclamation_count": 2,
        "has_conflict": True,
        "has_main_char": True,
        "has_emotion": True,
        "has_society": True,
        "has_mystery": True,
        "has_family": True,
        "runtime_minutes": 24.0,
        "release_year": 2024,
        "release_month": 10,
        "release_day": 20,
        "is_weekend": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"\n🎯 Prediction: {result['prediction']} ({'Success' if result['prediction'] == 1 else 'Not Success'})")
        print(f"📊 Success Probability: {result['success_probability']}%")
        print(f"🎚️  Confidence Level: {result['confidence_level']}")
        print(f"💡 Recommendation: {result['recommendation']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_low_success():
    """Test prediction optimized for low success"""
    print_section("TEST 7: Prediction - Low Success Scenario")
    
    payload = {
        "lead_cast_matches": 1,
        "supporting_cast_matches": 2,
        "desc_length": 100,
        "desc_word_count": 15,
        "question_count": 0,
        "exclamation_count": 0,
        "has_conflict": False,
        "has_main_char": False,
        "has_emotion": False,
        "has_society": False,
        "has_mystery": False,
        "has_family": False,
        "runtime_minutes": 18.0,
        "release_year": 2024,
        "release_month": 3,
        "release_day": 5,
        "is_weekend": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"\n🎯 Prediction: {result['prediction']} ({'Success' if result['prediction'] == 1 else 'Not Success'})")
        print(f"📊 Success Probability: {result['success_probability']}%")
        print(f"🎚️  Confidence Level: {result['confidence_level']}")
        print(f"💡 Recommendation: {result['recommendation']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                  TMKOC API TESTING SUITE                          ║
    ║                  Testing all endpoints...                          ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Get Features", test_features),
        ("Model Info", test_model_info),
        ("Prediction (Default)", test_predict_default),
        ("Prediction (High Success)", test_predict_high_success),
        ("Prediction (Low Success)", test_predict_low_success)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test '{name}' failed with error: {e}")
            results.append((name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed_count}/{total_count} tests passed")
    print(f"{'='*70}\n")
    
    if passed_count == total_count:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()