"""
ReActor V5 - IP-Adapter FaceID Integration Helper

This module provides helper functions to integrate ReActor V5 with
Forge's built-in IP-Adapter extension (sd_forge_ipadapter).

Forge already includes full IP-Adapter FaceID Plus v2 support via ControlNet,
so this helper just assists with configuration and workflow.

USAGE:
1. Enable ControlNet in your generation settings
2. Select "InsightFace+CLIP-H (IPAdapter)" as preprocessor
3. Use IP-Adapter FaceID Plus v2 model (download from HuggingFace if needed)
4. Upload your source face image to ControlNet
5. ReActor V5 will refine the face swap using standard inswapper

The workflow is:
- IP-Adapter provides identity guidance during diffusion
- ReActor V5 performs high-quality face swap on the result
- This gives you the best of both worlds: identity-aware generation + precise face swap
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, Dict, Any, List
import os
from pathlib import Path

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError as e:
    print(f"[ReActor V5] InsightFace not available: {e}")
    INSIGHTFACE_AVAILABLE = False


class ForgeIPAdapterHelper:
    """
    Helper class to work with Forge's built-in IP-Adapter.
    
    This doesn't implement IP-Adapter itself (Forge already has that),
    but provides utilities for face analysis and embedding extraction
    that can be used alongside ControlNet's IP-Adapter.
    """
    
    # Recommended IP-Adapter models for face-focused work
    RECOMMENDED_MODELS = {
        'sd15': {
            'faceid_plus_v2': {
                'name': 'ip-adapter-faceid-plusv2_sd15.bin',
                'url': 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin',
                'preprocessor': 'InsightFace+CLIP-H (IPAdapter)',
                'description': 'Best for face identity - uses InsightFace + CLIP embeddings'
            },
            'faceid_plus': {
                'name': 'ip-adapter-faceid-plus_sd15.bin',
                'url': 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin',
                'preprocessor': 'InsightFace+CLIP-H (IPAdapter)',
                'description': 'Good face identity, slightly less detail than v2'
            },
            'faceid': {
                'name': 'ip-adapter-faceid_sd15.bin',
                'url': 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin',
                'preprocessor': 'InsightFace+CLIP-H (IPAdapter)',
                'description': 'Basic face identity'
            }
        },
        'sdxl': {
            'faceid': {
                'name': 'ip-adapter-faceid_sdxl.bin',
                'url': 'https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin',
                'preprocessor': 'InsightFace+CLIP-H (IPAdapter)',
                'description': 'SDXL face identity'
            }
        }
    }
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.ipadapter_path = os.path.join(models_path, 'ipadapter')
        self.controlnet_path = os.path.join(models_path, 'ControlNet')
        self.face_app = None
        
        print("[ReActor V5] IP-Adapter Helper initialized")
        print(f"[ReActor V5] IP-Adapter models path: {self.ipadapter_path}")
        print(f"[ReActor V5] ControlNet models path: {self.controlnet_path}")
    
    def check_forge_ipadapter_available(self) -> bool:
        """Check if Forge's IP-Adapter extension is available."""
        try:
            from modules_forge.supported_preprocessor import Preprocessor
            from modules_forge.shared import supported_preprocessors
            
            # Check if InsightFace+CLIP-H preprocessor exists
            ipadapter_preprocessors = [
                p for p in supported_preprocessors 
                if 'IPAdapter' in p or 'InsightFace' in p
            ]
            
            if ipadapter_preprocessors:
                print(f"[ReActor V5] Found Forge IP-Adapter preprocessors: {ipadapter_preprocessors}")
                return True
            return False
            
        except Exception as e:
            print(f"[ReActor V5] Forge IP-Adapter check failed: {e}")
            return False
    
    def get_installed_ipadapter_models(self) -> List[str]:
        """Get list of installed IP-Adapter models."""
        models = []
        
        # Check ipadapter directory
        if os.path.exists(self.ipadapter_path):
            for f in os.listdir(self.ipadapter_path):
                if f.endswith('.bin') or f.endswith('.safetensors'):
                    if 'faceid' in f.lower() or 'ip-adapter' in f.lower():
                        models.append(f)
        
        # Check ControlNet directory
        if os.path.exists(self.controlnet_path):
            for f in os.listdir(self.controlnet_path):
                if 'faceid' in f.lower() and ('ip-adapter' in f.lower() or 'ip_adapter' in f.lower()):
                    models.append(f)
        
        return models
    
    def download_recommended_model(self, model_key: str = 'faceid_plus_v2', base_model: str = 'sd15') -> bool:
        """
        Download recommended IP-Adapter FaceID model.
        
        Args:
            model_key: Key from RECOMMENDED_MODELS (faceid_plus_v2, faceid_plus, faceid)
            base_model: 'sd15' or 'sdxl'
            
        Returns:
            bool: True if download successful or model exists
        """
        if base_model not in self.RECOMMENDED_MODELS:
            print(f"[ReActor V5] Unknown base model: {base_model}")
            return False
        
        if model_key not in self.RECOMMENDED_MODELS[base_model]:
            print(f"[ReActor V5] Unknown model key: {model_key}")
            return False
        
        model_info = self.RECOMMENDED_MODELS[base_model][model_key]
        model_path = os.path.join(self.ipadapter_path, model_info['name'])
        
        if os.path.exists(model_path):
            print(f"[ReActor V5] Model already exists: {model_info['name']}")
            return True
        
        try:
            from huggingface_hub import hf_hub_download
            
            os.makedirs(self.ipadapter_path, exist_ok=True)
            
            print(f"[ReActor V5] Downloading {model_info['name']}...")
            
            # Extract repo_id and filename from URL
            # URL format: https://huggingface.co/{repo_id}/resolve/main/{filename}
            parts = model_info['url'].split('/')
            repo_id = f"{parts[3]}/{parts[4]}"
            filename = parts[-1]
            
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.ipadapter_path,
                local_dir_use_symlinks=False
            )
            
            print(f"[ReActor V5] Downloaded: {model_info['name']}")
            return True
            
        except Exception as e:
            print(f"[ReActor V5] Error downloading model: {e}")
            return False
    
    def _init_face_app(self):
        """Initialize InsightFace analyzer."""
        if self.face_app is not None:
            return
        
        if not INSIGHTFACE_AVAILABLE:
            return
        
        try:
            insightface_path = os.path.join(self.models_path, 'insightface')
            self.face_app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root=insightface_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[ReActor V5] Error initializing InsightFace: {e}")
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.
        
        This can be useful for:
        - Comparing similarity between source and result
        - Pre-computing embeddings for batch processing
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Face embedding as numpy array, or None if no face detected
        """
        self._init_face_app()
        
        if self.face_app is None:
            return None
        
        try:
            faces = self.face_app.get(image)
            if not faces:
                return None
            
            # Use the largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Return normed embedding (as used by IP-Adapter FaceID)
            return face.normed_embedding
            
        except Exception as e:
            print(f"[ReActor V5] Error extracting embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def get_usage_instructions(self) -> str:
        """Get usage instructions for IP-Adapter with ReActor V5."""
        return """
## ReActor V5 + IP-Adapter FaceID Workflow

For best results combining IP-Adapter identity guidance with ReActor V5 face swapping:

### Setup
1. Download IP-Adapter FaceID Plus v2 model:
   - `ip-adapter-faceid-plusv2_sd15.bin` from HuggingFace
   - Place in `models/ipadapter/` folder

### Generation Workflow
1. **ControlNet Settings:**
   - Enable ControlNet Unit 0
   - Preprocessor: `InsightFace+CLIP-H (IPAdapter)`
   - Model: `ip-adapter-faceid-plusv2_sd15`
   - Upload your source face image
   - Weight: 0.6-0.8 (adjust for identity strength)
   - Starting/Ending Control Step: 0-0.4 (early guidance only)

2. **ReActor V5 Settings:**
   - Enable ReActor V5
   - Upload same source face image
   - Face Restoration: GPEN-BFR-512
   - Skin Texture: 0.08 (add natural skin detail)

### Why This Works
- IP-Adapter guides the diffusion to create a face with your identity
- ReActor V5 refines the result with precise face geometry
- Skin texture prevents plasticky appearance from restoration
- The combination gives identity-aware generation + clean swap

### Tips
- Use the same source image for both ControlNet and ReActor
- Lower IP-Adapter weight (0.5-0.6) if result looks too stylized
- Higher ReActor restoration if face needs cleanup
"""


# Global singleton
_ipadapter_helper = None


def get_ipadapter_helper(models_path: str) -> ForgeIPAdapterHelper:
    """Get or create IP-Adapter helper singleton."""
    global _ipadapter_helper
    if _ipadapter_helper is None:
        _ipadapter_helper = ForgeIPAdapterHelper(models_path)
    return _ipadapter_helper


class IPAdapterFaceIDPlusV2:
    """
    Legacy compatibility class.
    
    Forge WebUI already has full IP-Adapter support via ControlNet.
    Use the ForgeIPAdapterHelper class instead.
    """
    
    def __init__(self, models_path: str):
        self.helper = get_ipadapter_helper(models_path)
        print("[ReActor V5] IPAdapterFaceIDPlusV2: Using Forge's built-in IP-Adapter integration")
        print("[ReActor V5] Enable ControlNet with 'InsightFace+CLIP-H (IPAdapter)' preprocessor")
    
    def load_models(self) -> bool:
        """Check if Forge IP-Adapter is available."""
        return self.helper.check_forge_ipadapter_available()
    
    def unload_models(self):
        """No-op - Forge handles model management."""
        pass
    
    def extract_face_features(self, image: np.ndarray):
        """Delegate to helper."""
        return self.helper.extract_face_embedding(image)


class IPAdapterManager:
    """
    Manager class for IP-Adapter integration.
    
    Uses Forge's built-in IP-Adapter via ControlNet.
    """
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.helper = get_ipadapter_helper(models_path)
        self.is_enabled = False
        
        # Default configuration
        self.config = {
            'weight': 0.70,
            'start_step': 0.0,
            'end_step': 0.4,  # Early guidance only
        }
    
    def enable(self, config: Optional[Dict] = None) -> bool:
        """
        Check if IP-Adapter can be enabled.
        
        With Forge, IP-Adapter is enabled via ControlNet settings,
        so this just validates the setup.
        """
        if config:
            self.config.update(config)
        
        # Check if Forge IP-Adapter is available
        available = self.helper.check_forge_ipadapter_available()
        
        if available:
            self.is_enabled = True
            print("[ReActor V5] IP-Adapter integration available via ControlNet")
            print("[ReActor V5] Configure ControlNet with 'InsightFace+CLIP-H (IPAdapter)'")
        else:
            print("[ReActor V5] IP-Adapter not available - ensure Forge IP-Adapter extension is installed")
        
        return available
    
    def disable(self):
        """Disable IP-Adapter (ControlNet handles actual unloading)."""
        self.is_enabled = False
    
    def get_instructions(self) -> str:
        """Get setup instructions."""
        return self.helper.get_usage_instructions()


# For backwards compatibility
def is_ipadapter_available() -> bool:
    """Check if IP-Adapter is available."""
    try:
        from modules_forge.supported_preprocessor import Preprocessor
        return True
    except:
        return False


# Global singleton for IPAdapterManager
_ipadapter_manager = None


def get_ipadapter_manager(models_path: str) -> IPAdapterManager:
    """Get or create IP-Adapter manager singleton."""
    global _ipadapter_manager
    if _ipadapter_manager is None:
        _ipadapter_manager = IPAdapterManager(models_path)
    return _ipadapter_manager
