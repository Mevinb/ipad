"""
ReActor V5 - Enhanced Face Swapping Pipeline (Fixed Version)

Integrates all ReActor v3 functionality with:
- IP-Adapter FaceID Plus v2 identity guidance
- Explicit VRAM management
- Advanced realism enhancements
- Frequency-aware processing

Maintains full backward compatibility while adding optional enhancements.
"""

import cv2
import numpy as np
import os
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image
import torch
import gc
import sys

# Add extension scripts path for imports
_ext_scripts_path = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts_path not in sys.path:
    sys.path.insert(0, _ext_scripts_path)

# Setup cuDNN path for ONNX Runtime CUDA provider
def setup_cudnn_path():
    """Add cuDNN and cuBLAS to PATH if available"""
    try:
        import site
        site_packages = site.getsitepackages()
        paths_added = []
        for site_pkg in site_packages:
            # Add cuDNN path
            cudnn_bin_path = os.path.join(site_pkg, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin_path):
                current_path = os.environ.get('PATH', '')
                if cudnn_bin_path not in current_path:
                    os.environ['PATH'] = cudnn_bin_path + os.pathsep + current_path
                    paths_added.append(cudnn_bin_path)
            
            # Add cuBLAS path
            cublas_bin_path = os.path.join(site_pkg, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cublas_bin_path):
                current_path = os.environ.get('PATH', '')
                if cublas_bin_path not in current_path:
                    os.environ['PATH'] = cublas_bin_path + os.pathsep + current_path
                    paths_added.append(cublas_bin_path)
        
        if paths_added:
            print(f"[ReActor V5] Added CUDA libraries to PATH: {paths_added}")
            return True
        else:
            print("[ReActor V5] CUDA libraries (cuDNN/cuBLAS) not found in site-packages")
            return False
    except Exception as e:
        print(f"[ReActor V5] Error setting up CUDA libraries path: {e}")
        return False

# Setup cuDNN path on import
setup_cudnn_path()

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[ReActor V5] ONNX Runtime not available - HyperSwap models won't work")

# Import memory management from WebUI Forge backend
try:
    webui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if webui_path not in sys.path:
        sys.path.insert(0, webui_path)
    from backend import memory_management
    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False
    print("[ReActor V5] WARNING: WebUI memory management not available, using basic cleanup")

# Import ReActor V5 modules with error handling
try:
    from vram_management import get_vram_manager, check_vram_requirements
    VRAM_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] VRAM management not available: {e}")
    VRAM_MANAGEMENT_AVAILABLE = False

try:
    from ipadapter_faceid import get_ipadapter_manager
    IPADAPTER_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] IP-Adapter module not available: {e}")
    IPADAPTER_MODULE_AVAILABLE = False

try:
    from realism_enhancer import get_realism_enhancer
    REALISM_ENHANCER_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] Realism enhancer not available: {e}")
    REALISM_ENHANCER_AVAILABLE = False

try:
    from reactor_v5_gpen_restorer import get_gpen_restorer, get_available_gpen_models, clear_gpen_cache
    GPEN_RESTORER_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] GPEN restorer not available: {e}")
    GPEN_RESTORER_AVAILABLE = False

# Import separate swapper modules
try:
    from reactor_v5_inswapper import InSwapperFaceSwapper, get_inswapper, get_available_inswapper_models, clear_inswapper_cache
    INSWAPPER_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] InSwapper module not available: {e}")
    INSWAPPER_MODULE_AVAILABLE = False

try:
    from reactor_v5_hyperswap import HyperSwapFaceSwapper, get_hyperswap, get_available_hyperswap_models, clear_hyperswap_cache
    HYPERSWAP_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] HyperSwap module not available: {e}")
    HYPERSWAP_MODULE_AVAILABLE = False


# ArcFace 128 template - normalized landmark positions for face alignment
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],  # Left eye
    [0.63696719, 0.40235469],  # Right eye
    [0.50019687, 0.56044219],  # Nose tip
    [0.38710391, 0.72160547],  # Left mouth corner
    [0.61507734, 0.72034453]   # Right mouth corner
], dtype=np.float32)


def norm_crop(img, kps, image_size=256):
    """
    Align and crop face using 5 landmarks.
    Compatible with both inswapper and HyperSwap models.
    """
    if INSIGHTFACE_AVAILABLE:
        M = face_align.estimate_norm(kps, image_size)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped
    return None


def warp_face_by_landmarks(img, landmarks_5, template, crop_size):
    """
    Align face using 5-point landmarks with RANSAC estimation.
    Better suited for HyperSwap models.
    
    Args:
        img: Input image (BGR)
        landmarks_5: 5 facial landmarks (2D array)
        template: Normalized template landmarks
        crop_size: Output size tuple (width, height)
        
    Returns:
        Aligned face image and affine matrix
    """
    # Scale template to target size
    template_scaled = template * np.array(crop_size)
    
    # Estimate affine transform with RANSAC
    affine_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks_5.astype(np.float32), 
        template_scaled,
        method=cv2.RANSAC, 
        ransacReprojThreshold=100
    )
    
    if affine_matrix is None:
        # Fallback to basic estimation
        affine_matrix = cv2.getAffineTransform(
            landmarks_5[:3].astype(np.float32),
            template_scaled[:3]
        )
    
    # Warp face
    aligned = cv2.warpAffine(
        img, affine_matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_AREA
    )
    
    return aligned, affine_matrix


def detect_model_resolution(model_path: str) -> int:
    """
    Detect the resolution of a face swap model from its ONNX structure.
    Returns: 128 (inswapper) or 256 (HyperSwap)
    """
    try:
        import onnx
        model = onnx.load(model_path)
        
        # Check input dimensions
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape.dim
            if len(shape) >= 4:
                # Image input: (batch, channels, height, width)
                h = shape[2].dim_value if shape[2].dim_value else 256
                w = shape[3].dim_value if shape[3].dim_value else 256
                if h == w and h in [128, 256, 512]:
                    return h
        return 256  # Default for HyperSwap
    except Exception as e:
        print(f"[ReActor V5] Could not detect model resolution: {e}")
        # Infer from filename
        if '128' in model_path:
            return 128
        elif '256' in model_path:
            return 256
        return 256


# HyperSwapFaceSwapper class is now in reactor_v5_hyperswap.py
# InSwapperFaceSwapper class is now in reactor_v5_inswapper.py


def get_available_swapper_models(models_path: str) -> List[str]:
    """
    Get list of available face swapper models (inswapper + HyperSwap).
    Uses the separate module functions if available.
    """
    models = []
    
    # Get HyperSwap models
    if HYPERSWAP_MODULE_AVAILABLE:
        models.extend(get_available_hyperswap_models(models_path))
    else:
        # Fallback: check hyperswap directory directly
        hyperswap_path = os.path.join(models_path, 'hyperswap')
        if os.path.exists(hyperswap_path):
            for f in os.listdir(hyperswap_path):
                if f.endswith('.onnx') and 'hyperswap' in f.lower():
                    models.append(f)
    
    # Get InSwapper models
    if INSWAPPER_MODULE_AVAILABLE:
        models.extend(get_available_inswapper_models(models_path))
    else:
        # Fallback: check insightface directory directly
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
    
    # Default fallback
    if not models:
        models = ['inswapper_128.onnx']
    
    return sorted(models)


class ReactorV5:
    """
    ReActor V5 - Advanced face swapping with IP-Adapter FaceID Plus v2.
    
    Features:
    - All ReActor v3 functionality preserved
    - Optional IP-Adapter FaceID Plus v2 integration
    - Explicit VRAM management
    - Advanced realism enhancements
    - Frequency-aware processing
    """
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.shared_models_root = models_path
        self.insightface_path = os.path.join(self.shared_models_root, 'insightface')
        self.facerestore_path = os.path.join(self.shared_models_root, 'facerestore_models')
        
        os.makedirs(self.insightface_path, exist_ok=True)
        os.makedirs(self.facerestore_path, exist_ok=True)
        
        # Core ReActor components (backward compatible)
        self.face_analyser = None
        self.face_swapper = None
        self.current_swapper_name = None  # Track current swapper model
        self.current_swapper_resolution = 128  # Track swapper resolution
        self.current_restorer = None
        self.current_restorer_name = None
        
        # ReActor V5 enhancements (with fallbacks)
        if VRAM_MANAGEMENT_AVAILABLE:
            self.vram_manager = get_vram_manager()
        else:
            self.vram_manager = None
            
        if IPADAPTER_MODULE_AVAILABLE:
            self.ipadapter_manager = get_ipadapter_manager(models_path)
        else:
            self.ipadapter_manager = None
            
        if REALISM_ENHANCER_AVAILABLE:
            self.realism_enhancer = get_realism_enhancer()
        else:
            self.realism_enhancer = None
        
        # Configuration
        self.auto_cleanup = True
        self.aggressive_cleanup = True
        self.enable_ipadapter = False
        
        # V5 processing settings
        self.v5_config = {
            'enable_ipadapter': False,
            'ipadapter_weight': 0.70,
            'texture_preservation': 0.0,  # Disabled - causes blur
            'lighting_match': True,
            'inject_noise': False,  # Disabled - can cause artifacts
            'adaptive_restoration': True,
            'frequency_blending': False,  # DISABLED - main cause of blur
            'skip_restoration': False  # Now ENABLED - uses proper FaceRestoreHelper
        }
        
        print(f"[ReActor V5] Initialized with enhanced capabilities")
        print(f"[ReActor V5] InsightFace path: {self.insightface_path}")
        print(f"[ReActor V5] GPEN models path: {self.facerestore_path}")
        print(f"[ReActor V5] Components available: VRAM={VRAM_MANAGEMENT_AVAILABLE}, IP-Adapter={IPADAPTER_MODULE_AVAILABLE}, Realism={REALISM_ENHANCER_AVAILABLE}")
    
    def initialize_face_analyser(self):
        """Initialize face analyzer (backward compatible with V3)"""
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        if self.face_analyser is None:
            print("[ReActor V5] Initializing face analyzer...")
            self.face_analyser = FaceAnalysis(
                name='buffalo_l',
                root=self.insightface_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            print("[ReActor V5] Face analyzer ready")
    
    def initialize_face_swapper(self, model_name: str = 'inswapper_128.onnx'):
        """
        Initialize face swapper with support for both inswapper and HyperSwap models.
        
        Uses separate modules:
        - reactor_v5_inswapper.py for InSwapper (128x128)
        - reactor_v5_hyperswap.py for HyperSwap (256x256)
        
        HyperSwap models provide higher quality but require correct preprocessing.
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")
        
        # Check if we need to reload (different model requested)
        if self.face_swapper is not None and self.current_swapper_name == model_name:
            return  # Already loaded
        
        # Unload current swapper if switching models
        if self.face_swapper is not None and self.current_swapper_name != model_name:
            print(f"[ReActor V5] Switching swapper from {self.current_swapper_name} to {model_name}")
            self._clear_swapper()
        
        print(f"[ReActor V5] Initializing face swapper: {model_name}")
        
        # Detect if this is a HyperSwap model (256x256) or inswapper (128x128)
        is_hyperswap = 'hyperswap' in model_name.lower()
        
        if is_hyperswap:
            # Use HyperSwap module for 256x256 models
            if not HYPERSWAP_MODULE_AVAILABLE:
                raise ImportError("HyperSwap module not available")
            
            model_path = os.path.join(self.shared_models_root, 'hyperswap', model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"HyperSwap model not found: {model_path}")
            
            print(f"[ReActor V5] Using HyperSwap 256x256 model (higher quality)")
            self.face_swapper = HyperSwapFaceSwapper(model_path)
            self.current_swapper_resolution = 256
        else:
            # Use InSwapper module for 128x128 models
            # Search paths for inswapper
            search_paths = [
                os.path.join(self.insightface_path, model_name),
                os.path.join(self.insightface_path, 'models', model_name),
                os.path.join(self.shared_models_root, 'reactor', model_name),
            ]
            
            model_path = None
            for path in search_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"InSwapper model not found: {model_name}. Searched: {search_paths}")
            
            print(f"[ReActor V5] Using InSwapper 128x128 model")
            
            # Use InSwapper module if available, otherwise fallback to model_zoo
            if INSWAPPER_MODULE_AVAILABLE:
                self.face_swapper = InSwapperFaceSwapper(model_path)
            else:
                self.face_swapper = model_zoo.get_model(
                    model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
            self.current_swapper_resolution = 128
        
        self.current_swapper_name = model_name
        print(f"[ReActor V5] Face swapper ready: {model_path} ({self.current_swapper_resolution}x{self.current_swapper_resolution})")
    
    def _clear_swapper(self):
        """Clear current swapper and free memory."""
        if self.face_swapper is not None:
            del self.face_swapper
            self.face_swapper = None
        self.current_swapper_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_faces(self, img: np.ndarray) -> List:
        """Get detected faces (backward compatible with V3)"""
        if self.face_analyser is None:
            self.initialize_face_analyser()
        return self.face_analyser.get(img)
    
    def get_gender(self, face) -> str:
        """Get gender from face (backward compatible with V3)"""
        try:
            if hasattr(face, 'gender'):
                gender_value = face.gender
                if isinstance(gender_value, (int, float, np.integer, np.floating)):
                    return 'M' if gender_value >= 0.5 else 'F'
            elif hasattr(face, 'sex'):
                sex_value = face.sex
                if isinstance(sex_value, (int, float, np.integer, np.floating)):
                    return 'M' if sex_value >= 0.5 else 'F'
            
            if hasattr(face, '__dict__'):
                face_dict = face.__dict__
                if 'gender' in face_dict:
                    return 'M' if face_dict['gender'] >= 0.5 else 'F'
                if 'sex' in face_dict:
                    return 'M' if face_dict['sex'] >= 0.5 else 'F'
            
            return 'U'
        except Exception as e:
            print(f"[ReActor V5] Warning: Could not determine gender: {e}")
            return 'U'
    
    def filter_faces_by_gender(self, faces: List, target_gender: str) -> List:
        """Filter faces by gender (backward compatible with V3)"""
        if not faces or target_gender == 'A':
            return faces
        
        filtered = []
        for i, face in enumerate(faces):
            gender = self.get_gender(face)
            if gender == target_gender or gender == 'U':
                filtered.append(face)
            print(f"[ReActor V5] Face {i}: Gender={gender} {'✓' if gender == target_gender or gender == 'U' else '✗'}")
        
        return filtered
    
    def set_cleanup_mode(self, aggressive: bool):
        """Set cleanup mode (backward compatible with V3)"""
        self.aggressive_cleanup = aggressive
        print(f"[ReActor V5] Cleanup mode set to aggressive={aggressive}")
    
    def configure_v5_features(self, config: Dict[str, Any]):
        """Configure ReActor V5 enhanced features"""
        self.v5_config.update(config)
        
        # Handle IP-Adapter enable/disable
        if config.get('enable_ipadapter', False) != self.enable_ipadapter:
            self.enable_ipadapter = config['enable_ipadapter']
            
            if self.enable_ipadapter:
                if not IPADAPTER_MODULE_AVAILABLE:
                    print("[ReActor V5] Cannot enable IP-Adapter: Module not available")
                    self.enable_ipadapter = False
                    self.v5_config['enable_ipadapter'] = False
                    return False
                
                # Check VRAM before enabling
                if VRAM_MANAGEMENT_AVAILABLE:
                    can_run, message = check_vram_requirements(
                        enable_ipadapter=True,
                        resolution=(512, 512),
                        batch_size=1
                    )
                    
                    if not can_run:
                        print(f"[ReActor V5] Cannot enable IP-Adapter: {message}")
                        self.enable_ipadapter = False
                        self.v5_config['enable_ipadapter'] = False
                        return False
                
                # Enable IP-Adapter
                success = self.ipadapter_manager.enable(self.v5_config)
                if not success:
                    self.enable_ipadapter = False
                    self.v5_config['enable_ipadapter'] = False
                    return False
            else:
                # Disable IP-Adapter
                if self.ipadapter_manager:
                    self.ipadapter_manager.disable()
        
        print(f"[ReActor V5] V5 features configured: {self.v5_config}")
        return True
    
    def load_restorer(self, model_name: str) -> bool:
        """Load face restoration model (backward compatible with V3)"""
        if not GPEN_RESTORER_AVAILABLE:
            print("[ReActor V5] GPEN restorer not available")
            return False
            
        if model_name == self.current_restorer_name and self.current_restorer is not None:
            return True
        
        if not model_name or model_name.lower() == 'none':
            self.current_restorer = None
            self.current_restorer_name = None
            return True
        
        model_path = os.path.join(self.facerestore_path, model_name)
        if not os.path.exists(model_path):
            print(f"[ReActor V5] Model not found: {model_path}")
            return False
        
        try:
            self.current_restorer = get_gpen_restorer(model_path, device='cuda')
            self.current_restorer_name = model_name
            return True
        except Exception as e:
            print(f"[ReActor V5] Error loading restorer: {e}")
            return False
    
    def process(self,
                source_img: np.ndarray,
                target_img: np.ndarray,
                source_face_index: int = 0,
                target_face_index: int = 0,
                swapper_model: str = None,
                restore_model: str = None,
                gender_match: str = 'A',
                v5_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, str]:
        """
        Enhanced processing pipeline with ReActor V5 features (with fallbacks).
        
        Args:
            swapper_model: Face swap model name (hyperswap_1a_256.onnx, inswapper_128.onnx, etc.)
            restore_model: Face restoration model (GPEN-BFR-512.onnx, etc.)
        """
        try:
            # Configure V5 features if provided
            if v5_config:
                if not self.configure_v5_features(v5_config):
                    print("[ReActor V5] Warning: V5 feature configuration failed, using basic mode")
            
            # Initialize components (backward compatible)
            if self.face_analyser is None:
                self.initialize_face_analyser()
            
            # Initialize face swapper with specified model
            swapper_to_use = swapper_model if swapper_model else 'inswapper_128.onnx'
            if self.face_swapper is None or self.current_swapper_name != swapper_to_use:
                self.initialize_face_swapper(swapper_to_use)
            
            # Face detection (existing ReActor logic)
            print("[ReActor V5] Detecting faces...")
            source_faces = self.get_faces(source_img)
            target_faces = self.get_faces(target_img)
            
            if not source_faces:
                return target_img, "Error: No face in source"
            if not target_faces:
                return target_img, "Error: No face in target"
            
            # Apply gender matching (backward compatible)
            if gender_match == 'S':
                source_face = source_faces[min(source_face_index, len(source_faces)-1)]
                source_gender = self.get_gender(source_face)
                print(f"[ReActor V5] Smart Match: Source gender = {source_gender}")
                
                filtered_target_faces = self.filter_faces_by_gender(target_faces, source_gender)
                if not filtered_target_faces:
                    gender_name = "male" if source_gender == 'M' else "female" if source_gender == 'F' else "matching"
                    return target_img, f"Error: No {gender_name} face in target to match source"
                target_faces = filtered_target_faces
                
            elif gender_match in ['M', 'F']:
                print(f"[ReActor V5] Gender Filter: {gender_match}")
                source_faces = self.filter_faces_by_gender(source_faces, gender_match)
                target_faces = self.filter_faces_by_gender(target_faces, gender_match)
                
                if not source_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} face in source"
                if not target_faces:
                    gender_name = "male" if gender_match == 'M' else "female"
                    return target_img, f"Error: No {gender_name} face in target"
            
            source_face = source_faces[min(source_face_index, len(source_faces)-1)]
            target_face = target_faces[min(target_face_index, len(target_faces)-1)]
            
            # Display gender info
            source_gender = self.get_gender(source_face)
            target_gender = self.get_gender(target_face)
            print(f"[ReActor V5] Swapping: Source ({source_gender}) -> Target ({target_gender})")
            
            # Face swap (identity geometry only)
            print("[ReActor V5] Swapping face...")
            swapped_result = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # V5 enhancements (with fallbacks)
            if REALISM_ENHANCER_AVAILABLE and self.v5_config.get('frequency_blending', False):
                print("[ReActor V5] Applying realism enhancements...")
                # Create simple face mask for processing
                face_mask = self.create_simple_face_mask(swapped_result, target_face)
                
                # IMPORTANT: Pass ORIGINAL TARGET image for texture preservation
                # This preserves the target's skin texture, not the source's face
                enhanced_result = self.realism_enhancer.enhance_realism(
                    original_target=target_img,  # Use target for texture
                    swapped_image=swapped_result,
                    face_mask=face_mask,
                    config=self.v5_config,
                    restorer=self.current_restorer if restore_model else None
                )
                swapped_result = enhanced_result
                status_suffix = " with V5 enhancements"
            else:
                status_suffix = ""
            
            # Face restoration using GPEN with WebUI's FaceRestoreHelper
            # Now uses proper infrastructure like GFPGAN/CodeFormer
            skip_restoration = self.v5_config.get('skip_restoration', False)
            
            if restore_model and not skip_restoration:
                if self.load_restorer(restore_model) and self.current_restorer:
                    print(f"[ReActor V5] Restoring with {restore_model}...")
                    # Just call restore(image) - FaceRestoreHelper handles everything
                    restored_result = self.current_restorer.restore(swapped_result)
                    swapped_result = restored_result
                    status_suffix += f" with GPEN restoration"
            elif restore_model and skip_restoration:
                print("[ReActor V5] Skipping GPEN restoration (disabled in config)")
            
            # Cleanup
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            
            final_status = f"Face swapped successfully{status_suffix}"
            print(f"[ReActor V5] {final_status}")
            
            return swapped_result, final_status
            
        except Exception as e:
            print(f"[ReActor V5] Error: {e}")
            import traceback
            traceback.print_exc()
            if self.auto_cleanup:
                self.cleanup_memory(aggressive=self.aggressive_cleanup)
            return target_img, f"Error: {str(e)}"
    
    def create_simple_face_mask(self, image: np.ndarray, face) -> np.ndarray:
        """Create simple face mask for processing (fallback method)"""
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Expand face area slightly
            expansion = 0.3
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            face_w, face_h = x2 - x1, y2 - y1
            
            expand_w = int(face_w * expansion / 2)
            expand_h = int(face_h * expansion / 2)
            
            x1_exp = max(0, center_x - face_w // 2 - expand_w)
            y1_exp = max(0, center_y - face_h // 2 - expand_h)
            x2_exp = min(w, center_x + face_w // 2 + expand_w)
            y2_exp = min(h, center_y + face_h // 2 + expand_h)
            
            mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255
            
            # Apply Gaussian blur for smooth edges
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            return mask
            
        except Exception as e:
            print(f"[ReActor V5] Error creating face mask: {e}")
            # Return full image mask as fallback
            return np.ones((image.shape[:2]), dtype=np.uint8) * 255
    
    def cleanup_memory(self, aggressive: bool = False):
        """Enhanced memory cleanup with V5 components"""
        print(f"[ReActor V5] Cleaning up memory (aggressive={aggressive})...")
        
        try:
            # V5 component cleanup
            if aggressive:
                if self.enable_ipadapter and self.ipadapter_manager:
                    self.ipadapter_manager.disable()
                    self.enable_ipadapter = False
                
                # Clear restoration models
                if self.current_restorer is not None:
                    del self.current_restorer
                    self.current_restorer = None
                    self.current_restorer_name = None
                    
                if GPEN_RESTORER_AVAILABLE:
                    clear_gpen_cache()
                
                # Unload InsightFace models
                if self.face_analyser is not None:
                    print("[ReActor V5] Unloading InsightFace face analyzer...")
                    del self.face_analyser
                    self.face_analyser = None
                
                if self.face_swapper is not None:
                    print("[ReActor V5] Unloading InsightFace face swapper...")
                    del self.face_swapper
                    self.face_swapper = None
            
            # Use VRAM manager or fallback cleanup
            if VRAM_MANAGEMENT_AVAILABLE and self.vram_manager:
                self.vram_manager.progressive_cleanup('aggressive' if aggressive else 'standard')
            else:
                # Fallback cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if aggressive:
                        torch.cuda.ipc_collect()
                gc.collect()
            
        except Exception as e:
            print(f"[ReActor V5] Warning during cleanup: {e}")
    
    def get_available_restorers(self) -> List[str]:
        """Get available restoration models (backward compatible)"""
        if GPEN_RESTORER_AVAILABLE:
            return ['None'] + get_available_gpen_models(self.facerestore_path)
        else:
            return ['None']
    
    def get_vram_status(self) -> str:
        """Get current VRAM status for UI display"""
        if VRAM_MANAGEMENT_AVAILABLE and self.vram_manager:
            return self.vram_manager.get_status_report()
        else:
            return "VRAM management not available"
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get ReActor V5 capabilities for UI configuration"""
        return {
            'ipadapter_available': IPADAPTER_MODULE_AVAILABLE,
            'vram_management_available': VRAM_MANAGEMENT_AVAILABLE,
            'realism_enhancer_available': REALISM_ENHANCER_AVAILABLE,
            'gpen_restorer_available': GPEN_RESTORER_AVAILABLE,
            'insightface_available': INSIGHTFACE_AVAILABLE,
            'memory_management_available': MEMORY_MANAGEMENT_AVAILABLE,
            'v5_features_enabled': True
        }


# Global instance
reactor_v5_engine: Optional[ReactorV5] = None

def get_reactor_v5_engine(models_path: str) -> ReactorV5:
    """Get global ReActor V5 engine instance"""
    global reactor_v5_engine
    if reactor_v5_engine is None:
        reactor_v5_engine = ReactorV5(models_path)
    return reactor_v5_engine