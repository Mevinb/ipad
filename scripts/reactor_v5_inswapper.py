"""
ReActor V5 - InSwapper Face Swapper Module

Standard InSwapper (128x128) face swapping using InsightFace.
Uses the inswapper_128.onnx model with emap embedding transformation.

Usage:
    from reactor_v5_inswapper import InSwapperFaceSwapper
    
    swapper = InSwapperFaceSwapper(model_path)
    result = swapper.get(image, target_face, source_face)
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
import sys

# Setup cuDNN path for ONNX Runtime CUDA provider
def setup_cudnn_path():
    """Add cuDNN and cuBLAS to PATH if available"""
    try:
        import site
        site_packages = site.getsitepackages()
        for site_pkg in site_packages:
            cudnn_bin_path = os.path.join(site_pkg, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin_path):
                current_path = os.environ.get('PATH', '')
                if cudnn_bin_path not in current_path:
                    os.environ['PATH'] = cudnn_bin_path + os.pathsep + current_path
            
            cublas_bin_path = os.path.join(site_pkg, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cublas_bin_path):
                current_path = os.environ.get('PATH', '')
                if cublas_bin_path not in current_path:
                    os.environ['PATH'] = cublas_bin_path + os.pathsep + current_path
    except Exception:
        pass

setup_cudnn_path()

try:
    from insightface.model_zoo import model_zoo
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[InSwapper] InsightFace not available")


class InSwapperFaceSwapper:
    """
    InSwapper face swapper using InsightFace model_zoo.
    Uses 128x128 resolution with emap embedding transformation.
    
    This is the standard approach used in ReActor v3 and earlier.
    """
    
    MODEL_SIZE = 128
    
    def __init__(self, model_path: str):
        """
        Initialize InSwapper face swapper.
        
        Args:
            model_path: Path to inswapper_128.onnx model
        """
        self.model_path = model_path
        self.resolution = self.MODEL_SIZE
        self.model = None
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not available - cannot load InSwapper model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"InSwapper model not found: {model_path}")
        
        # Load model using InsightFace model_zoo
        self.model = model_zoo.get_model(model_path)
        
        print(f"[InSwapper] Model loaded: {os.path.basename(model_path)}")
        print(f"[InSwapper] Resolution: {self.resolution}x{self.resolution}")
    
    def get(self, img: np.ndarray, target_face, source_face, paste_back: bool = True) -> np.ndarray:
        """
        Perform face swap using InSwapper model.
        
        The InSwapper model uses:
        - Raw embedding transformed via emap matrix
        - 128x128 face crop
        - Simple normalization (0-1 range)
        
        Args:
            img: Target image (BGR numpy array)
            target_face: Target face object with kps and bbox attributes
            source_face: Source face object with embedding attribute
            paste_back: Whether to paste swapped face back to original image
            
        Returns:
            Swapped image (BGR numpy array)
        """
        if self.model is None:
            print("[InSwapper] Model not loaded")
            return img
        
        try:
            # Use InsightFace's built-in swap method
            result = self.model.get(img, target_face, source_face, paste_back=paste_back)
            return result
        except Exception as e:
            print(f"[InSwapper] Error during face swap: {e}")
            return img
    
    def swap_face(self, img: np.ndarray, target_face, source_face) -> np.ndarray:
        """
        Alternative method name for compatibility.
        """
        return self.get(img, target_face, source_face, paste_back=True)


def get_available_inswapper_models(models_path: str) -> list:
    """
    Get list of available InSwapper models.
    
    Args:
        models_path: Base path to models directory
        
    Returns:
        List of model filenames
    """
    models = []
    
    # Check insightface directory
    insightface_path = os.path.join(models_path, 'insightface')
    if os.path.exists(insightface_path):
        for f in os.listdir(insightface_path):
            if f.endswith('.onnx') and 'inswapper' in f.lower():
                models.append(f)
        
        # Also check models subdirectory
        models_subdir = os.path.join(insightface_path, 'models')
        if os.path.exists(models_subdir):
            for f in os.listdir(models_subdir):
                if f.endswith('.onnx') and 'inswapper' in f.lower():
                    if f not in models:
                        models.append(f)
    
    return sorted(models)


def get_inswapper_model_path(models_path: str, model_name: str) -> Optional[str]:
    """
    Get full path to an InSwapper model.
    
    Args:
        models_path: Base path to models directory
        model_name: Model filename
        
    Returns:
        Full path to model or None if not found
    """
    # Check insightface directory
    insightface_path = os.path.join(models_path, 'insightface')
    model_path = os.path.join(insightface_path, model_name)
    if os.path.exists(model_path):
        return model_path
    
    # Check models subdirectory
    model_path = os.path.join(insightface_path, 'models', model_name)
    if os.path.exists(model_path):
        return model_path
    
    return None


# Singleton instance
_inswapper_instance = None
_inswapper_model_path = None


def get_inswapper(model_path: str) -> Optional[InSwapperFaceSwapper]:
    """
    Get or create InSwapper instance (singleton pattern).
    
    Args:
        model_path: Path to InSwapper model
        
    Returns:
        InSwapperFaceSwapper instance or None
    """
    global _inswapper_instance, _inswapper_model_path
    
    if _inswapper_instance is None or _inswapper_model_path != model_path:
        try:
            _inswapper_instance = InSwapperFaceSwapper(model_path)
            _inswapper_model_path = model_path
        except Exception as e:
            print(f"[InSwapper] Failed to initialize: {e}")
            return None
    
    return _inswapper_instance


def clear_inswapper_cache():
    """Clear InSwapper instance from memory."""
    global _inswapper_instance, _inswapper_model_path
    _inswapper_instance = None
    _inswapper_model_path = None
    
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
