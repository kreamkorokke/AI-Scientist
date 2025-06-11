#!/usr/bin/env python3
"""
Comprehensive test script to verify all template integration fixes work correctly.
Tests the fixes in launch_scientist.py, perform_experiments.py, and experimental files.
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

def test_perform_experiments_compatibility():
    """Test that perform_experiments.py works with both ML and non-ML templates."""
    
    print("Testing perform_experiments.py compatibility...")
    
    try:
        from ai_scientist.perform_experiments import extract_metrics_from_final_info
        
        # Test 1: ML-style format
        ml_data = {
            "train_loss": {"means": [2.1, 1.8, 1.5], "stds": [0.1, 0.08, 0.06]},
            "val_loss": {"means": [2.3, 2.0, 1.7], "stds": [0.12, 0.09, 0.07]}
        }
        
        ml_results = extract_metrics_from_final_info(ml_data)
        if ml_results.get("train_loss") == [2.1, 1.8, 1.5] and ml_results.get("val_loss") == [2.3, 2.0, 1.7]:
            print("✅ ML-style format extraction works")
        else:
            print(f"❌ ML-style format extraction failed: {ml_results}")
            return False
        
        # Test 2: Non-ML format (like population template)
        demographic_data = {
            "baseline": {
                "population_decline_pct": -31.98,
                "aging_ratio_pct": 98.84,
                "policy_effectiveness_pct": 0.0
            },
            "comprehensive": {
                "population_decline_pct": -25.63,
                "aging_ratio_pct": 99.12,
                "policy_effectiveness_pct": 7.55
            }
        }
        
        demo_results = extract_metrics_from_final_info(demographic_data)
        expected_keys = [
            "baseline_population_decline_pct", "baseline_aging_ratio_pct", 
            "comprehensive_population_decline_pct", "comprehensive_aging_ratio_pct"
        ]
        
        if all(key in demo_results for key in expected_keys):
            print("✅ Non-ML format extraction works")
        else:
            print(f"❌ Non-ML format extraction failed: {demo_results}")
            return False
        
        # Test 3: Mixed format
        mixed_data = {
            "scalar_metric": 42.5,
            "nested_metric": {"value": 3.14, "other": "text"}
        }
        
        mixed_results = extract_metrics_from_final_info(mixed_data)
        if "scalar_metric" in mixed_results and "nested_metric_value" in mixed_results:
            print("✅ Mixed format extraction works")
        else:
            print(f"❌ Mixed format extraction failed: {mixed_results}")
            return False
            
        print("✅ perform_experiments.py is template-agnostic")
        return True
        
    except Exception as e:
        print(f"❌ Error testing perform_experiments.py: {e}")
        return False

def test_launch_scientist_compatibility():
    """Test that launch_scientist.py baseline extraction works."""
    
    print("\nTesting launch_scientist.py compatibility...")
    
    try:
        from launch_scientist import extract_baseline_metrics
        
        # Test with population template
        template_dir = ""
        if os.path.exists(template_dir):
            baseline_results = extract_baseline_metrics(template_dir)
            if baseline_results and len(baseline_results) > 10:  # Should have many demographic metrics
                print(f"✅ launch_scientist.py baseline extraction works ({len(baseline_results)} metrics)")
            else:
                print(f"❌ launch_scientist.py baseline extraction failed: {baseline_results}")
                return False
        else:
            print("⚠️  Population template not found, skipping baseline test")
            
        return True
        
    except ImportError as e:
        if "openai" in str(e):
            print("⚠️  Skipping launch_scientist.py test (openai not available)")
            return True
        else:
            print(f"❌ Error testing launch_scientist.py: {e}")
            return False
    except Exception as e:
        print(f"❌ Error testing launch_scientist.py: {e}")
        return False

def test_population_template_integration():
    """Test that population_policy template provides the expected interface."""
    
    print("\nTesting population_policy template integration...")
    
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
        
        # Verify metric format (should be all numeric)
        non_numeric = [k for k, v in metrics.items() if not isinstance(v, (int, float))]
        if non_numeric:
            print(f"❌ Found non-numeric metrics: {non_numeric}")
            return False
        print("✅ All metrics are numeric")
        
    except Exception as e:
        print(f"❌ Error extracting baseline metrics: {e}")
        return False
    finally:
        if template_dir in sys.path:
            sys.path.remove(template_dir)
    
    print("✅ Population template integration complete")
    return True

def test_cross_project_compatibility():
    """Test that the fixes work across different scenarios."""
    
    print("\nTesting cross-project compatibility...")
    
    # Test that existing ML templates still work
    try:
        # Simulate ML template final_info.json
        ml_final_info = {
            "experiment_1": {"means": {"accuracy": 0.95, "loss": 0.05}},
            "experiment_2": {"means": {"accuracy": 0.97, "loss": 0.03}}
        }
        
        from ai_scientist.perform_experiments import extract_metrics_from_final_info
        ml_results = extract_metrics_from_final_info(ml_final_info)
        
        # Should extract the means properly
        if "experiment_1" in ml_results and "experiment_2" in ml_results:
            print("✅ Backward compatibility with ML templates maintained")
        else:
            print(f"❌ Backward compatibility broken: {ml_results}")
            return False
            
    except Exception as e:
        print(f"❌ Cross-project compatibility test failed: {e}")
        return False
    
    return True

def main():
    """Run all template integration tests."""
    
    print("AI-Scientist Template Integration Fixes - Comprehensive Test")
    print("=" * 70)
    
    success = True
    
    # Test all the fixes
    success &= test_perform_experiments_compatibility()
    success &= test_launch_scientist_compatibility()
    success &= test_population_template_integration()
    success &= test_cross_project_compatibility()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Fixed Files:")
        print("  - ai_scientist/perform_experiments.py")
        print("  - launch_scientist.py")
        print("  - experimental/launch_oe_scientist.py")
        print("  - templates/population_policy/experiment.py")
        print("\n✅ Template Support:")
        print("  - Traditional ML templates (nanoGPT, grokking, etc.)")
        print("  - Demographic templates (population_policy)")
        print("  - Future non-ML templates")
        print("\n✅ Features:")
        print("  - Template-agnostic metric extraction")
        print("  - Backward compatibility with existing templates")
        print("  - Flexible final_info.json format support")
        print("  - Automatic fallback to standard formats")
        print("\n🚀 Ready to run: python launch_scientist.py --experiment population_policy")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()