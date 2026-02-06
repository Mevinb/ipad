"""
Test script to check if basic imports work and identify specific issues.
"""

import sys
import os

# Add extension scripts path
_ext_scripts_path = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts_path not in sys.path:
    sys.path.insert(0, _ext_scripts_path)

def test_imports():
    """Test all imports used in ReActor V5"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
    
    try:
        import cv2
        print("✓ cv2 (opencv)")
    except ImportError as e:
        print(f"✗ cv2 (opencv): {e}")
    
    try:
        from PIL import Image
        print("✓ PIL")
    except ImportError as e:
        print(f"✗ PIL: {e}")
    
    try:
        import insightface
        print("✓ insightface")
    except ImportError as e:
        print(f"✗ insightface: {e}")
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✓ diffusers")
    except ImportError as e:
        print(f"✗ diffusers: {e}")
    
    try:
        from transformers import CLIPVisionModelWithProjection
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")

    print("\nTesting custom module imports...")
    
    try:
        from vram_management import get_vram_manager
        print("✓ vram_management")
    except ImportError as e:
        print(f"✗ vram_management: {e}")
    
    try:
        from ipadapter_faceid import get_ipadapter_manager
        print("✓ ipadapter_faceid")
    except ImportError as e:
        print(f"✗ ipadapter_faceid: {e}")
    
    try:
        from realism_enhancer import get_realism_enhancer
        print("✓ realism_enhancer")
    except ImportError as e:
        print(f"✗ realism_enhancer: {e}")
    
    try:
        from reactor_v5_gpen_restorer import get_gpen_restorer
        print("✓ reactor_v5_gpen_restorer")
    except ImportError as e:
        print(f"✗ reactor_v5_gpen_restorer: {e}")

if __name__ == "__main__":
    test_imports()