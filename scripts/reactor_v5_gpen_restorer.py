"""
ReActor V5 - GPEN Face Restoration Module

Uses WebUI's FaceRestoreHelper infrastructure exactly like GFPGAN and CodeFormer.
This is the CORRECT way to do face restoration - not custom cropping.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Optional, List
import torch

# Setup cuDNN path for ONNX Runtime CUDA provider
def setup_cudnn_path():
    """Add cuDNN and cuBLAS to PATH if available"""
    try:
        import site
        site_packages = site.getsitepackages()
        paths_added = []
        for site_pkg in site_packages:
            cudnn_bin_path = os.path.join(site_pkg, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin_path):
                current_path = os.environ.get('PATH', '')
                if cudnn_bin_path not in current_path:
                    os.environ['PATH'] = cudnn_bin_path + os.pathsep + current_path
                    paths_added.append(cudnn_bin_path)
            
            cublas_bin_path = os.path.join(site_pkg, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cublas_bin_path):
                current_path = os.environ.get('PATH', '')
                if cublas_bin_path not in current_path:
                    os.environ['PATH'] = cublas_bin_path + os.pathsep + current_path
                    paths_added.append(cublas_bin_path)
        
        if paths_added:
            print(f"[ReActor V5] Added CUDA libraries to PATH: {paths_added}")
        return True
    except Exception as e:
        print(f"[ReActor V5] Error setting up CUDA libraries path: {e}")
        return False

setup_cudnn_path()

# Import WebUI's face restoration infrastructure
import sys
webui_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if webui_path not in sys.path:
    sys.path.insert(0, webui_path)

# Check if FaceRestoreHelper is available
FACE_RESTORE_HELPER_AVAILABLE = False
try:
    from modules import face_restoration_utils, devices
    from facexlib.detection import retinaface
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    FACE_RESTORE_HELPER_AVAILABLE = True
    print("[ReActor V5] FaceRestoreHelper available - using proper restoration")
except ImportError as e:
    print(f"[ReActor V5] FaceRestoreHelper not available: {e}")
    print("[ReActor V5] Will use simple restoration fallback")


class GPENFaceRestorer:
    """
    GPEN face restoration using WebUI's FaceRestoreHelper.
    This works exactly like GFPGAN and CodeFormer - the proper way.
    """
    
    def __init__(self, model_path: str, resolution: int = 512, device: str = 'cuda'):
        self.model_path = model_path
        self.resolution = resolution
        self.device = device
        self.session = None
        self.face_helper = None
        
        self._initialize_model()
        
        if FACE_RESTORE_HELPER_AVAILABLE:
            self._initialize_face_helper()
    
    def _initialize_model(self):
        """Initialize ONNX Runtime session for GPEN"""
        ort.set_default_logger_severity(3)
        
        providers = []
        if self.device == 'cuda':
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }))
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            active_providers = self.session.get_providers()
            print(f"[ReActor V5] GPEN model loaded: {os.path.basename(self.model_path)} @ {self.resolution}x{self.resolution}")
            print(f"[ReActor V5] GPEN active providers: {active_providers}")
        except Exception as e:
            print(f"[ReActor V5] Error loading GPEN model: {e}")
            raise
    
    def _initialize_face_helper(self):
        """Initialize WebUI's FaceRestoreHelper with correct resolution"""
        try:
            device_torch = devices.device_codeformer if self.device == 'cuda' else devices.cpu
            
            if hasattr(retinaface, 'device'):
                retinaface.device = device_torch
            
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=self.resolution,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=device_torch,
            )
            print(f"[ReActor V5] FaceRestoreHelper initialized @ {self.resolution}px")
        except Exception as e:
            print(f"[ReActor V5] Error initializing FaceRestoreHelper: {e}")
            self.face_helper = None
    
    def restore_with_gpen(self, cropped_face_t: torch.Tensor) -> torch.Tensor:
        """
        Restore a single cropped face tensor.
        Called by FaceRestoreHelper for each detected face.
        
        Args:
            cropped_face_t: Normalized face tensor (1, 3, H, W) in range [-1, 1]
            
        Returns:
            Restored face tensor in range [-1, 1]
        """
        # Convert to numpy for ONNX
        face_np = cropped_face_t.cpu().numpy()
        
        # Resize if needed
        if face_np.shape[2] != self.resolution or face_np.shape[3] != self.resolution:
            face_hwc = face_np.transpose(0, 2, 3, 1)[0]
            face_hwc = cv2.resize(face_hwc, (self.resolution, self.resolution))
            face_np = face_hwc.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run GPEN inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        restored_np = self.session.run([output_name], {input_name: face_np.astype(np.float32)})[0]
        
        # CRITICAL: Clamp output to [-1, 1] range
        restored_np = np.clip(restored_np, -1.0, 1.0)
        
        # Convert back to torch tensor
        restored_t = torch.from_numpy(restored_np).to(cropped_face_t.device)
        
        return restored_t
    
    def restore(self, np_image: np.ndarray, strength: float = None, 
                face_analyser=None, faces=None) -> np.ndarray:
        """
        Restore faces in an image.
        
        Uses WebUI's FaceRestoreHelper (like GFPGAN/CodeFormer) if available,
        otherwise uses simple whole-image processing.
        
        Args:
            np_image: Input image in BGR format (H, W, 3) uint8
            strength: Unused (kept for API compatibility)
            face_analyser: Unused (kept for API compatibility)
            faces: Unused (kept for API compatibility)
            
        Returns:
            Restored image in BGR format
        """
        if FACE_RESTORE_HELPER_AVAILABLE and self.face_helper is not None:
            # Use proper WebUI infrastructure
            print("[ReActor V5] Using FaceRestoreHelper for proper restoration")
            result = face_restoration_utils.restore_with_face_helper(
                np_image,
                self.face_helper,
                self.restore_with_gpen
            )
            return result
        else:
            # Fallback: Just return original (no blur is better than bad blur)
            print("[ReActor V5] FaceRestoreHelper unavailable - skipping restoration to avoid blur")
            return np_image


# Cache for loaded models
gpen_models_cache = {}


def get_gpen_restorer(model_path: str, device: str = 'cuda') -> GPENFaceRestorer:
    """
    Get GPEN restorer with caching.
    
    Args:
        model_path: Path to GPEN model
        device: Device to use
        
    Returns:
        GPENFaceRestorer instance
    """
    cache_key = f"{model_path}_{device}"
    
    if cache_key not in gpen_models_cache:
        # Determine resolution from model name
        model_name = os.path.basename(model_path).lower()
        if '1024' in model_name:
            resolution = 1024
        else:
            resolution = 512
        
        gpen_models_cache[cache_key] = GPENFaceRestorer(model_path, resolution, device)
    
    return gpen_models_cache[cache_key]


def get_available_gpen_models(models_path: str) -> List[str]:
    """Get list of available GPEN models"""
    models = []
    if os.path.exists(models_path):
        for f in os.listdir(models_path):
            if f.lower().endswith('.onnx') and 'gpen' in f.lower():
                models.append(f)
    return sorted(models)


def clear_gpen_cache():
    """Clear GPEN model cache to free memory"""
    global gpen_models_cache
    gpen_models_cache.clear()
    print("[ReActor V5] GPEN cache cleared")
