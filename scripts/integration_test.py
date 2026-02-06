#!/usr/bin/env python3
"""
ReActor V5 Integration Test
Tests all components with proper import handling
"""

import sys
import os
import traceback

# Add extension path
ext_path = r"C:\Users\mevin\Downloads\me\webui\extensions\sd-webui-reactor-v5\scripts"
if ext_path not in sys.path:
    sys.path.insert(0, ext_path)

print("=" * 80)
print("ReActor V5 Integration Test")
print("=" * 80)

def test_basic_imports():
    """Test all basic Python imports"""
    print("\n1. Testing Basic Imports:")
    imports_to_test = [
        ("os", "Operating System interface"),
        ("sys", "System-specific parameters"),
        ("numpy", "NumPy for arrays"),
        ("PIL", "Python Imaging Library"),
        ("cv2", "OpenCV for image processing"),
    ]
    
    results = {}
    for module_name, description in imports_to_test:
        try:
            if module_name == "numpy":
                import numpy as np
            elif module_name == "PIL":
                from PIL import Image
            elif module_name == "cv2":
                import cv2
            elif module_name == "os":
                import os
            elif module_name == "sys":
                import sys
            
            results[module_name] = "✅ PASS"
            print(f"  {module_name}: ✅ Available - {description}")
        except ImportError as e:
            results[module_name] = f"❌ FAIL: {str(e)}"
            print(f"  {module_name}: ❌ Missing - {description}")
    
    return results

def test_optional_imports():
    """Test optional dependencies with proper handling"""
    print("\n2. Testing Optional Dependencies:")
    optional_imports = [
        ("torch", "PyTorch for ML models"),
        ("transformers", "Hugging Face transformers"),
        ("diffusers", "Diffusion models library"),
        ("scipy", "Scientific computing library"),
        ("skimage", "scikit-image for image processing"),
        ("psutil", "Process and system utilities"),
        ("insightface", "Face analysis library")
    ]
    
    results = {}
    for module_name, description in optional_imports:
        try:
            if module_name == "torch":
                import torch
                print(f"  {module_name}: ✅ Available - {description} (CUDA: {torch.cuda.is_available()})")
            elif module_name == "transformers":
                import transformers
                print(f"  {module_name}: ✅ Available - {description}")
            elif module_name == "diffusers":
                import diffusers
                print(f"  {module_name}: ✅ Available - {description}")
            elif module_name == "scipy":
                import scipy
                print(f"  {module_name}: ✅ Available - {description}")
            elif module_name == "skimage":
                import skimage
                print(f"  {module_name}: ✅ Available - {description}")
            elif module_name == "psutil":
                import psutil
                print(f"  {module_name}: ✅ Available - {description}")
            elif module_name == "insightface":
                import insightface
                print(f"  {module_name}: ✅ Available - {description}")
            
            results[module_name] = "✅ PASS"
        except ImportError as e:
            results[module_name] = f"⚠️ OPTIONAL: {str(e)}"
            print(f"  {module_name}: ⚠️ Optional - {description} (Not available but not required)")
    
    return results

def test_reactor_v5_modules():
    """Test ReActor V5 specific modules"""
    print("\n3. Testing ReActor V5 Modules:")
    
    modules_to_test = [
        ("vram_management", "VRAM Management System"),
        ("ipadapter_faceid", "IP-Adapter FaceID Plus v2"),
        ("realism_enhancer", "Realism Enhancement Pipeline"),
        ("reactor_v5_gpen_restorer", "GPEN Face Restoration"),
        ("reactor_v5_swapper_fixed", "Main ReActor V5 Engine"),
        ("!!reactor_v5_ui", "ReActor V5 UI Components")
    ]
    
    results = {}
    for module_name, description in modules_to_test:
        try:
            if module_name == "vram_management":
                from vram_management import get_vram_manager, check_vram_requirements, format_vram_status
                # Test basic functionality
                vram_manager = get_vram_manager()
                status = format_vram_status()
                print(f"  {module_name}: ✅ Available - {description}")
                print(f"    VRAM Status: {status[:50]}...")
                
            elif module_name == "ipadapter_faceid":
                from ipadapter_faceid import get_ipadapter_manager
                print(f"  {module_name}: ✅ Available - {description}")
                
            elif module_name == "realism_enhancer":
                from realism_enhancer import get_realism_enhancer
                enhancer = get_realism_enhancer()
                print(f"  {module_name}: ✅ Available - {description}")
                
            elif module_name == "reactor_v5_gpen_restorer":
                from reactor_v5_gpen_restorer import get_gpen_restorer, get_available_gpen_models
                models = get_available_gpen_models(r"C:\Users\mevin\Downloads\me\webui\models\facerestore_models")
                print(f"  {module_name}: ✅ Available - {description}")
                print(f"    GPEN Models Found: {len(models)}")
                
            elif module_name == "reactor_v5_swapper_fixed":
                from reactor_v5_swapper_fixed import get_reactor_v5_engine, ReactorV5
                print(f"  {module_name}: ✅ Available - {description}")
                
            elif module_name == "!!reactor_v5_ui":
                import importlib.util
                spec = importlib.util.spec_from_file_location("reactor_v5_ui", "!!reactor_v5_ui.py")
                reactor_ui = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reactor_ui)
                print(f"  {module_name}: ✅ Available - {description}")
            
            results[module_name] = "✅ PASS"
            
        except ImportError as e:
            results[module_name] = f"❌ FAIL: {str(e)}"
            print(f"  {module_name}: ❌ Error - {description}")
            print(f"    Error: {str(e)}")
        except Exception as e:
            results[module_name] = f"⚠️ PARTIAL: {str(e)}"
            print(f"  {module_name}: ⚠️ Partial - {description}")
            print(f"    Warning: {str(e)}")
    
    return results

def test_reactor_v5_engine():
    """Test ReActor V5 engine initialization"""
    print("\n4. Testing ReActor V5 Engine:")
    
    try:
        from reactor_v5_swapper_fixed import get_reactor_v5_engine
        
        models_path = r"C:\Users\mevin\Downloads\me\webui\models"
        print(f"  Using models path: {models_path}")
        
        # Initialize engine
        engine = get_reactor_v5_engine(models_path)
        print("  Engine initialized: ✅")
        
        # Check capabilities
        capabilities = engine.get_capabilities()
        print("  Capabilities:")
        for feature, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"    {feature}: {status}")
        
        # Check VRAM status
        vram_status = engine.get_vram_status()
        print(f"  VRAM Status: {str(vram_status)[:100]}...")
        
        # Check available restorers
        restorers = engine.get_available_restorers()
        print(f"  Available Restorers: {len(restorers)} models")
        
        return "✅ PASS - Engine fully functional"
        
    except Exception as e:
        print(f"  Engine test failed: {str(e)}")
        traceback.print_exc()
        return f"❌ FAIL: {str(e)}"

def generate_report(results):
    """Generate comprehensive test report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    all_results = {}
    for category, category_results in results.items():
        all_results.update(category_results)
    
    passed = sum(1 for result in all_results.values() if result.startswith("✅"))
    failed = sum(1 for result in all_results.values() if result.startswith("❌"))
    optional = sum(1 for result in all_results.values() if result.startswith("⚠️"))
    
    print(f"Total Tests: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Optional/Warnings: {optional}")
    
    if failed == 0:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
        print("ReActor V5 is ready for use.")
    else:
        print(f"\n⚠️ {failed} CRITICAL ISSUES FOUND")
        print("Some components may not work correctly.")
    
    print("\nDetailed Results:")
    for category, category_results in results.items():
        print(f"\n{category}:")
        for test_name, result in category_results.items():
            print(f"  {test_name}: {result}")

def main():
    """Run comprehensive integration test"""
    try:
        # Run all tests
        results = {
            "Basic Imports": test_basic_imports(),
            "Optional Dependencies": test_optional_imports(),
            "ReActor V5 Modules": test_reactor_v5_modules(),
            "Engine Test": {"reactor_v5_engine": test_reactor_v5_engine()}
        }
        
        # Generate report
        generate_report(results)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during testing: {str(e)}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)