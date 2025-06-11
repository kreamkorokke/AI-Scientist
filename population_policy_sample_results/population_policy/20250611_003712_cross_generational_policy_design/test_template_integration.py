#!/usr/bin/env python3
"""
Test script to verify template integration with launch_scientist.py
"""

import os
import sys
import json

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

def test_population_template_integration():
    """Test that population_policy template integrates correctly with AI-Scientist framework."""
    
    print("Testing population_policy template integration...")
    
    # Test path
    template_dir = ""
    
    if not os.path.exists(template_dir):
        print(f"❌ Template directory not found: {template_dir}")
        return False
    
    # Test 1: Check if template has baseline results
    baseline_path = os.path.join(template_dir, "run_0", "final_info.json")
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline results not found: {baseline_path}")
        return False
    
    # Test 2: Try to import template experiment module
    try:
        sys.path.insert(0, template_dir)
        import experiment
        print("✅ Successfully imported experiment module")
    except ImportError as e:
        print(f"❌ Failed to import experiment module: {e}")
        return False
    
    # Test 3: Check if extract_key_metrics_from_results function exists
    if not hasattr(experiment, 'extract_key_metrics_from_results'):
        print("❌ Template missing extract_key_metrics_from_results function")
        return False
    print("✅ Template has extract_key_metrics_from_results function")
    
    # Test 4: Try to extract metrics from baseline
    try:
        run_0_dir = os.path.join(template_dir, "run_0")
        metrics = experiment.extract_key_metrics_from_results(run_0_dir)
        if not metrics:
            print("❌ No metrics extracted from baseline")
            return False
        print(f"✅ Extracted {len(metrics)} baseline metrics")
        print("   Sample metrics:")
        for i, (key, value) in enumerate(list(metrics.items())[:5]):
            print(f"     {key}: {value}")
        if len(metrics) > 5:
            print(f"     ... and {len(metrics) - 5} more")
    except Exception as e:
        print(f"❌ Error extracting baseline metrics: {e}")
        return False
    
    # Test 5: Verify metric format (should be all numeric)
    non_numeric = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
    if non_numeric:
        print(f"❌ Found non-numeric metrics: {non_numeric}")
        return False
    print("✅ All metrics are numeric")
    
    # Test 6: Try launch_scientist.py integration functions (skip if openai not available)
    try:
        from launch_scientist import extract_baseline_metrics
        baseline_results = extract_baseline_metrics(template_dir)
        if not baseline_results:
            print("❌ launch_scientist.py baseline extraction failed")
            return False
        print(f"✅ launch_scientist.py extracted {len(baseline_results)} baseline metrics")
    except ImportError as e:
        if "openai" in str(e):
            print("⚠️  Skipping launch_scientist.py test (openai not available)")
        else:
            print(f"❌ Error with launch_scientist.py integration: {e}")
            return False
    except Exception as e:
        print(f"❌ Error with launch_scientist.py integration: {e}")
        return False
    
    print("\n🎉 All tests passed! Population template is ready for AI-Scientist integration.")
    return True

def test_fallback_compatibility():
    """Test that the system falls back gracefully for templates without key metrics extraction."""
    
    print("\nTesting fallback compatibility...")
    
    # Create a mock final_info.json with ML-style structure
    mock_ml_results = {
        "train_loss": {"means": [2.1, 1.8, 1.5], "stds": [0.1, 0.08, 0.06]},
        "val_loss": {"means": [2.3, 2.0, 1.7], "stds": [0.12, 0.09, 0.07]}
    }
    
    # Create temporary mock structure
    mock_dir = "/tmp/mock_template"
    os.makedirs(f"{mock_dir}/run_0", exist_ok=True)
    
    with open(f"{mock_dir}/run_0/final_info.json", "w") as f:
        json.dump(mock_ml_results, f)
    
    try:
        from launch_scientist import extract_baseline_metrics
        baseline_results = extract_baseline_metrics(mock_dir)
        
        expected_keys = ["train_loss", "val_loss"]
        if all(key in baseline_results for key in expected_keys):
            print("✅ Fallback ML extraction works correctly")
            print(f"   Extracted: {baseline_results}")
        else:
            print(f"❌ Fallback extraction failed: {baseline_results}")
            return False
            
    except ImportError as e:
        if "openai" in str(e):
            print("⚠️  Skipping fallback test (openai not available)")
            return True
        else:
            print(f"❌ Error with fallback extraction: {e}")
            return False
    except Exception as e:
        print(f"❌ Error with fallback extraction: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(mock_dir, ignore_errors=True)
    
    return True

if __name__ == "__main__":
    print("AI-Scientist Template Integration Test")
    print("=" * 50)
    
    success = True
    success &= test_population_template_integration()
    success &= test_fallback_compatibility()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Template integration is working correctly.")
        print("\nNext steps:")
        print("1. Run: python launch_scientist.py --experiment population_policy")
        print("2. The framework will now use demographic-specific metrics")
        print("3. Baseline comparisons will work for policy effectiveness")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")
        sys.exit(1)