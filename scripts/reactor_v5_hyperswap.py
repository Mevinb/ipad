"""
ReActor V5 - HyperSwap Face Swapper Module

HyperSwap (256x256) face swapping using ONNX Runtime.
Uses hyperswap_1a/1b/1c_256.onnx models with proper preprocessing.

Key differences from InSwapper:
- 256x256 resolution (vs 128x128)
- Input normalization: (x - 0.5) / 0.5 for range [-1, 1]
- Uses L2-normalized embedding directly (normed_embedding)
- arcface_128 template with RANSAC alignment
- Output denormalization: x * 0.5 + 0.5

Usage:
    from reactor_v5_hyperswap import HyperSwapFaceSwapper
    
    swapper = HyperSwapFaceSwapper(model_path)
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
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[HyperSwap] ONNX Runtime not available")

try:
    from insightface.utils import face_align
    from insightface.utils.face_align import norm_crop
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[HyperSwap] InsightFace face_align not available")


# ArcFace 128 template - normalized landmark positions for face alignment
# These are the standard 5-point facial landmarks used by InsightFace
ARCFACE_128_TEMPLATE = np.array([
    [0.36167656, 0.40387734],  # Left eye
    [0.63696719, 0.40235469],  # Right eye
    [0.50019687, 0.56044219],  # Nose tip
    [0.38710391, 0.72160547],  # Left mouth corner
    [0.61507734, 0.72034453]   # Right mouth corner
], dtype=np.float32)


class HyperSwapFaceSwapper:
    """
    HyperSwap face swapper using direct ONNX Runtime.
    Supports 256x256 resolution for higher quality face swaps.
    
    Preprocessing:
    - Input normalization: (x / 255.0 - 0.5) / 0.5 = range [-1, 1]
    - Uses normed_embedding (L2-normalized) from InsightFace
    - arcface_128 alignment template
    
    Postprocessing:
    - Output denormalization: x * 0.5 + 0.5, then * 255
    """
    
    MODEL_SIZE = 256
    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    def __init__(self, model_path: str, providers=None):
        """
        Initialize HyperSwap face swapper.
        
        Args:
            model_path: Path to hyperswap_*.onnx model
            providers: ONNX Runtime execution providers
        """
        self.model_path = model_path
        self.resolution = self.MODEL_SIZE
        self.session = None
        
        if providers is None:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("ONNX Runtime not available - cannot load HyperSwap model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HyperSwap model not found: {model_path}")
        
        # Load the model
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Detect actual input names (may vary between model versions)
        self.source_input_name = self._find_input_name(['source', 'embedding', 'latent'])
        self.target_input_name = self._find_input_name(['target', 'input', 'image'])
        
        print(f"[HyperSwap] Model loaded: {os.path.basename(model_path)}")
        print(f"[HyperSwap] Resolution: {self.resolution}x{self.resolution}")
        print(f"[HyperSwap] Inputs: {self.input_names}, Outputs: {self.output_names}")
    
    def _find_input_name(self, candidates: list) -> str:
        """Find input name from candidates."""
        for name in candidates:
            if name in self.input_names:
                return name
        return self.input_names[0] if len(self.input_names) == 1 else candidates[0]
    
    def _align_face_ransac(self, img: np.ndarray, landmarks_5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face using 5-point landmarks with RANSAC estimation.
        
        Args:
            img: Input image (BGR)
            landmarks_5: 5 facial landmarks array
            
        Returns:
            Tuple of (aligned_face, affine_matrix)
        """
        crop_size = (self.resolution, self.resolution)
        
        # Scale template to target size
        template_scaled = ARCFACE_128_TEMPLATE * np.array(crop_size)
        
        # Estimate affine transform with RANSAC for robustness
        affine_matrix, _ = cv2.estimateAffinePartial2D(
            landmarks_5.astype(np.float32), 
            template_scaled,
            method=cv2.RANSAC, 
            ransacReprojThreshold=100
        )
        
        if affine_matrix is None:
            # Fallback to InsightFace method
            if INSIGHTFACE_AVAILABLE:
                affine_matrix = face_align.estimate_norm(landmarks_5, self.resolution)
            else:
                # Last resort: basic 3-point affine
                affine_matrix = cv2.getAffineTransform(
                    landmarks_5[:3].astype(np.float32),
                    template_scaled[:3]
                )
        
        # Warp face with BORDER_REPLICATE for better edge handling
        aligned = cv2.warpAffine(
            img, affine_matrix, crop_size,
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA
        )
        
        return aligned, affine_matrix
    
    def _preprocess_target(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess target face for HyperSwap model.
        
        Uses cv2.dnn.blobFromImage for consistent preprocessing:
        - Scale: 1/127.5
        - Mean: (127.5, 127.5, 127.5)
        - Result: (pixel - 127.5) / 127.5 = range [-1, 1]
        
        Args:
            face_crop: Aligned face crop (BGR, uint8)
            
        Returns:
            Preprocessed tensor (1, 3, H, W) float32
        """
        # Use cv2.dnn.blobFromImage - same as working V2 implementation
        # This normalizes to [-1, 1] range: (pixel - 127.5) / 127.5
        blob = cv2.dnn.blobFromImage(
            face_crop,
            1.0 / 127.5,  # Scale factor
            (self.resolution, self.resolution),  # Size
            (127.5, 127.5, 127.5),  # Mean subtraction
            swapRB=True  # Convert BGR to RGB
        )
        
        # CPU Float Normalization Fix - ensure float32 for ONNX
        if blob.dtype != np.float32:
            blob = blob.astype(np.float32)
        
        return blob
    
    def _prepare_source_embedding(self, source_face) -> np.ndarray:
        """
        Get normalized embedding for source face.
        
        HyperSwap uses L2-normalized embedding directly,
        NOT the emap-transformed embedding like InSwapper.
        
        Args:
            source_face: Face object with normed_embedding attribute
            
        Returns:
            Embedding array (1, 512) float32
        """
        # HyperSwap uses normed_embedding (L2-normalized)
        if hasattr(source_face, 'normed_embedding'):
            embedding = source_face.normed_embedding
        elif hasattr(source_face, 'embedding'):
            # Fallback: normalize manually
            embedding = source_face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        else:
            # Assume it's already an embedding array
            embedding = source_face
        
        # Ensure proper shape (1, 512)
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding, axis=0)
        
        return embedding.astype(np.float32)
    
    def _postprocess_output(self, output: np.ndarray) -> np.ndarray:
        """
        Denormalize model output.
        
        HyperSwap output is in [-1, 1] range:
        - Denormalize: (x + 1) * 127.5
        - Convert to uint8
        - RGB to BGR
        
        Uses same approach as working V2 implementation.
        
        Args:
            output: Model output (1, 3, H, W) or (3, H, W)
            
        Returns:
            Face image (H, W, 3) BGR uint8
        """
        # Remove batch dimension if present
        if len(output.shape) == 4:
            pred = output[0]  # Shape: (3, H, W)
        else:
            pred = output
        
        # Denormalize from [-1, 1] to [0, 255] - same as V2
        pred = np.clip(pred, -1, 1)
        pred = (pred + 1) * 127.5
        pred = pred.astype(np.uint8)
        
        # CHW to HWC
        pred = pred.transpose(1, 2, 0)
        
        # RGB to BGR
        pred = pred[:, :, ::-1]
        
        return pred
        
        # RGB to BGR
        face = face[:, :, ::-1]
        
        return face
    
    def _create_soft_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Create soft mask for seamless blending.
        
        Args:
            size: Mask size (width, height)
            
        Returns:
            Soft mask array (H, W, 1) float32
        """
        h, w = size[1], size[0]
        mask = np.ones((h, w), dtype=np.float32)
        
        # Apply feathering to edges
        border = max(h, w) // 8
        
        # Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (border * 2 + 1, border * 2 + 1), 0)
        
        # Erode slightly to avoid edge artifacts
        kernel_size = border // 2
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1).astype(np.float32) / 255.0
            mask = cv2.GaussianBlur(mask, (border + 1, border + 1), 0)
        
        return mask[:, :, np.newaxis]
    
    def get(self, img: np.ndarray, target_face, source_face, paste_back: bool = True, debug: bool = False) -> np.ndarray:
        """
        Perform face swap using HyperSwap model.
        
        Args:
            img: Target image (BGR numpy array)
            target_face: Target face object with kps attribute
            source_face: Source face object with normed_embedding attribute
            paste_back: Whether to paste swapped face back to original image
            debug: Print debug information
            
        Returns:
            Swapped image (BGR numpy array)
        """
        if self.session is None:
            print("[HyperSwap] Model not loaded")
            return img
        
        if not INSIGHTFACE_AVAILABLE:
            print("[HyperSwap] InsightFace not available - cannot perform swap")
            return img
        
        try:
            # Get landmarks - must have kps attribute
            if not hasattr(target_face, 'kps'):
                print("[HyperSwap] No 'kps' landmarks found on target face")
                return img
            
            if debug:
                print(f"[HyperSwap] Using kps landmarks, shape: {target_face.kps.shape}")
            
            # Align target face using InsightFace's norm_crop - same as V2
            aimg = norm_crop(img, target_face.kps, image_size=self.resolution)
            
            if debug:
                print(f"[HyperSwap] Aligned face shape: {aimg.shape}")
                print(f"[HyperSwap] Aligned face range: {aimg.min()} to {aimg.max()}")
            
            # Preprocess target face for model using cv2.dnn.blobFromImage - same as V2
            blob = self._preprocess_target(aimg)
            
            if debug:
                print(f"[HyperSwap] Target blob shape: {blob.shape}")
                print(f"[HyperSwap] Target blob range: {blob.min():.3f} to {blob.max():.3f}")
            
            # Prepare source embedding
            source_embedding = self._prepare_source_embedding(source_face)
            
            if debug:
                print(f"[HyperSwap] Source embedding shape: {source_embedding.shape}")
                print(f"[HyperSwap] Source embedding L2 norm: {np.linalg.norm(source_embedding):.3f}")
            
            # Build inputs dict - HyperSwap: 'source' = embedding (1, 512), 'target' = image (1, 3, 256, 256)
            inputs = {
                'source': source_embedding,
                'target': blob
            }
            
            # Run the model - HyperSwap outputs: ['output', 'mask']
            outputs = self.session.run(None, inputs)
            
            if debug:
                print(f"[HyperSwap] Output shape: {outputs[0].shape}")
                print(f"[HyperSwap] Output range: {outputs[0].min():.3f} to {outputs[0].max():.3f}")
            
            # Process output - first output is the swapped face
            pred = self._postprocess_output(outputs[0])
            
            if debug:
                print(f"[HyperSwap] Swapped face shape: {pred.shape}")
                print(f"[HyperSwap] Swapped face range: {pred.min()} to {pred.max()}")
            
            if not paste_back:
                return pred
            
            # Calculate the affine matrix for pasting back - same as V2
            M = face_align.estimate_norm(target_face.kps, self.resolution)
            IM = cv2.invertAffineTransform(M)
            
            # Create output image
            result = img.copy()
            
            # Create mask and warp the swapped face back to original position
            img_white = np.full((self.resolution, self.resolution, 3), 255, dtype=np.uint8)
            mask = cv2.warpAffine(img_white, IM, (result.shape[1], result.shape[0]), flags=cv2.INTER_LINEAR)
            mask = mask.astype(np.float32) / 255.0
            
            pred_warped = cv2.warpAffine(pred, IM, (result.shape[1], result.shape[0]), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # Apply Gaussian blur to mask for smooth blending - same as V2
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT)
            
            # Blend
            result = result * (1 - mask) + pred_warped * mask
            result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"[HyperSwap] Error during face swap: {e}")
            import traceback
            traceback.print_exc()
            return img
    
    def _paste_back(self, target_image: np.ndarray, swapped_face: np.ndarray, 
                    affine_matrix: np.ndarray, model_mask: np.ndarray = None) -> np.ndarray:
        """
        Paste swapped face back to original image with seamless blending.
        
        Uses the model's mask output if available for better blending.
        
        Args:
            target_image: Original image
            swapped_face: Swapped face crop
            affine_matrix: Affine matrix used for alignment
            model_mask: Optional mask from model output [1, 1, H, W]
            
        Returns:
            Result image with face pasted back
        """
        h, w = target_image.shape[:2]
        
        # Invert affine matrix
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        
        # Warp swapped face to original position
        warped_face = cv2.warpAffine(
            swapped_face, inverse_matrix, (w, h),
            borderValue=(0, 0, 0),
            flags=cv2.INTER_LINEAR
        )
        
        # Use model's mask if available, otherwise create one
        if model_mask is not None:
            # Process model mask: [1, 1, H, W] -> [H, W]
            face_mask = model_mask[0, 0]  # Remove batch and channel dims
            face_mask = np.clip(face_mask, 0, 1).astype(np.float32)
        else:
            # Fallback: create simple mask
            face_mask = np.ones((self.resolution, self.resolution), dtype=np.float32)
        
        # Warp mask to original position
        warped_mask = cv2.warpAffine(
            face_mask, inverse_matrix, (w, h),
            borderValue=0,
            flags=cv2.INTER_LINEAR
        )
        
        # Feather mask edges for seamless blending
        blur_size = max(w, h) // 20
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        blur_size = max(3, blur_size)
        
        # Erode slightly to avoid edge artifacts
        erode_size = max(3, blur_size // 4)
        kernel = np.ones((erode_size, erode_size), np.uint8)
        warped_mask = cv2.erode((warped_mask * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
        
        # Gaussian blur for soft edges
        warped_mask = cv2.GaussianBlur(warped_mask, (blur_size, blur_size), 0)
        
        # Expand mask to 3 channels
        warped_mask = warped_mask[:, :, np.newaxis]
        
        # Blend
        result = target_image.astype(np.float32) * (1 - warped_mask) + warped_face.astype(np.float32) * warped_mask
        
        return result.astype(np.uint8)
    
    def swap_face(self, img: np.ndarray, target_face, source_face) -> np.ndarray:
        """
        Alternative method name for compatibility.
        """
        return self.get(img, target_face, source_face, paste_back=True)


def get_available_hyperswap_models(models_path: str) -> list:
    """
    Get list of available HyperSwap models.
    
    Args:
        models_path: Base path to models directory
        
    Returns:
        List of model filenames
    """
    models = []
    
    # Check hyperswap directory
    hyperswap_path = os.path.join(models_path, 'hyperswap')
    if os.path.exists(hyperswap_path):
        for f in os.listdir(hyperswap_path):
            if f.endswith('.onnx') and 'hyperswap' in f.lower():
                models.append(f)
    
    return sorted(models)


def get_hyperswap_model_path(models_path: str, model_name: str) -> Optional[str]:
    """
    Get full path to a HyperSwap model.
    
    Args:
        models_path: Base path to models directory
        model_name: Model filename
        
    Returns:
        Full path to model or None if not found
    """
    hyperswap_path = os.path.join(models_path, 'hyperswap')
    model_path = os.path.join(hyperswap_path, model_name)
    
    if os.path.exists(model_path):
        return model_path
    
    return None


# Singleton instance
_hyperswap_instance = None
_hyperswap_model_path = None


def get_hyperswap(model_path: str) -> Optional[HyperSwapFaceSwapper]:
    """
    Get or create HyperSwap instance (singleton pattern).
    
    Args:
        model_path: Path to HyperSwap model
        
    Returns:
        HyperSwapFaceSwapper instance or None
    """
    global _hyperswap_instance, _hyperswap_model_path
    
    if _hyperswap_instance is None or _hyperswap_model_path != model_path:
        try:
            _hyperswap_instance = HyperSwapFaceSwapper(model_path)
            _hyperswap_model_path = model_path
        except Exception as e:
            print(f"[HyperSwap] Failed to initialize: {e}")
            return None
    
    return _hyperswap_instance


def clear_hyperswap_cache():
    """Clear HyperSwap instance from memory."""
    global _hyperswap_instance, _hyperswap_model_path
    _hyperswap_instance = None
    _hyperswap_model_path = None
    
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
