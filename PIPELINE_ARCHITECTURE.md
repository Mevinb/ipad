# ReActor V5 - Pipeline Architecture

## 🔄 Processing Pipeline (MANDATORY ORDER)

ReActor V5 follows this strict processing sequence for optimal results:

### 1. Face Detection
- **Technology**: InsightFace buffalo_l model
- **Process**: Detect faces in both source and target images
- **Gender Filtering**: Apply smart matching or explicit gender filters
- **Backward Compatible**: Identical to ReActor v3 behavior

### 2. Face Swap (Identity Geometry Only)
- **Technology**: InsightFace inswapper_128.onnx 
- **Process**: Swap face geometry/identity without texture smoothing
- **Key Point**: NO texture blending at this stage
- **Preserve**: Original skin micro-details and texture

### 3. IP-Adapter FaceID Plus v2 Guidance (Optional)
- **Technology**: IP-Adapter FaceID Plus v2 with InsightFace embeddings
- **Application Window**: Early diffusion steps only (~40% of total steps)
- **Fade Out**: Complete fade-out after 40% of steps to prevent over-smoothing
- **Mask Restriction**: Influence limited strictly to face mask regions
- **Memory Requirement**: 10-12GB VRAM minimum

```python
def calculate_step_weight(current_step: int, total_steps: int, base_weight: float = 0.70) -> float:
    fade_step = int(total_steps * 0.4)  # 40% fade point
    
    if current_step >= fade_step:
        return 0.0  # No influence after fade
    
    # Linear fade from base_weight to 0
    fade_progress = current_step / fade_step
    return base_weight * (1.0 - fade_progress)
```

### 4. Identity/Texture Separation
- **Low Frequency**: Identity geometry (shape, proportions)
- **High Frequency**: Skin texture (pores, micro-details, noise)
- **Process**: Apply swap + IP-Adapter to identity layer only
- **Preserve**: Original high-frequency texture completely

```python
def separate_identity_texture(image: np.ndarray, sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    image_float = image.astype(np.float32) / 255.0
    
    # Low frequency: identity/geometry
    low_freq = cv2.GaussianBlur(image_float, (0, 0), sigma)
    
    # High frequency: texture details
    high_freq = image_float - low_freq
    
    return low_freq, high_freq
```

### 5. Adaptive Face Restoration
- **Trigger**: Only when blur is detected (Laplacian variance < threshold)
- **Strength Cap**: Maximum 0.35 to prevent over-smoothing
- **Technology**: GPEN-512/1024 with WebUI's FaceRestoreHelper
- **Key**: This step MUST NOT occur before IP-Adapter guidance

```python
def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[float, bool]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var, laplacian_var < threshold
```

### 6. Controlled Noise Injection
- **Purpose**: Prevent plastic/over-smoothed skin appearance
- **Type**: Gaussian noise (σ ≈ 0.01–0.02)
- **Target**: Skin regions only (using HSV-based skin mask)
- **Resolution Aware**: Noise amplitude scales with image resolution
- **Mandatory**: This step is required for realism

```python
def inject_controlled_noise(image: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    resolution_factor = np.sqrt((w * h) / (512 * 512))
    
    sigma = 0.015 / resolution_factor  # Scale with resolution
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    
    # Apply only to face/skin regions
    masked_noise = noise * (face_mask / 255.0)
    return np.clip((image/255.0 + masked_noise) * 255, 0, 255).astype(np.uint8)
```

### 7. Frequency-Aware Blending
- **Low Frequency**: Blend geometry/shape for continuity
- **High Frequency**: Preserve original skin micro-details
- **Result**: Natural integration with original image texture

```python
def frequency_aware_blending(swapped: np.ndarray, original_texture: np.ndarray, 
                           face_mask: np.ndarray, preservation: float = 0.8) -> np.ndarray:
    swapped_low, swapped_high = separate_identity_texture(swapped)
    blend_mask = face_mask / 255.0
    
    # Blend textures with preservation factor
    texture_blend = (original_texture * preservation * blend_mask + 
                    swapped_high * (1 - preservation * blend_mask))
    
    # Recombine with identity geometry
    return recombine_frequencies(swapped_low, texture_blend)
```

## 🧠 Why This Pipeline Order Matters

### Early-Step IP-Adapter Guidance
- **Problem**: Late-stage guidance can over-smooth skin texture
- **Solution**: Apply guidance only during early diffusion steps
- **Benefit**: Preserves natural skin variation and micro-details
- **Technical**: Fade weight from 0.7 → 0.0 over first 40% of steps

### Face Restoration After IP-Adapter
- **Problem**: Restoration before guidance can interfere with identity transfer
- **Solution**: Apply restoration only after IP-Adapter guidance is complete
- **Benefit**: Maintains both identity accuracy and texture quality

### Mandatory Noise Injection
- **Problem**: Face swapping tends to produce unnaturally smooth skin
- **Solution**: Inject controlled Gaussian noise to restore natural variation
- **Technical**: Resolution-aware noise amplitude prevents over/under-texturing

## 📊 VRAM Estimation Logic

### Base Requirements
```python
def estimate_ipadapter_vram_usage(resolution: Tuple[int, int] = (512, 512), 
                                 batch_size: int = 1) -> float:
    base_usage = 2.1  # Model weights + CLIP Vision + embeddings
    
    # Scale with resolution (quadratic)
    w, h = resolution
    resolution_factor = (w * h) / (512 * 512)
    temp_usage = 0.5 * resolution_factor * batch_size
    
    return base_usage + temp_usage
```

### Memory-Safe Execution Rules
1. **Batch Size**: Force batch_size = 1 when IP-Adapter enabled
2. **Conflict Prevention**: Disable Hi-Res Fix + IP-Adapter combination  
3. **FP16 Enforcement**: All models use FP16 for memory efficiency
4. **Progressive Loading**: Load IP-Adapter only when needed
5. **Immediate Cleanup**: Unload IP-Adapter after generation

### Graceful Failure
```python
def can_run_with_ipadapter(resolution, batch_size) -> Tuple[bool, str]:
    free_vram = get_free_vram()
    required_vram = estimate_usage(resolution, batch_size) + 1.0  # Safety margin
    
    if free_vram < required_vram:
        return False, f"Insufficient VRAM: Need {required_vram:.1f}GB, have {free_vram:.1f}GB"
    
    return True, "VRAM check passed"
```

## 🎯 IP-Adapter Configuration

### Model Specifications
- **Model**: IP-Adapter FaceID Plus v2 (InsightFace-based)
- **Input**: Face embedding from InsightFace buffalo_l
- **Target**: SD 1.5 CLIP Vision encoder
- **Size**: ~1.2GB model weights

### Recommended Settings
```python
DEFAULT_CONFIG = {
    'weight': 0.70,          # Identity strength
    'cfg_scale': 6.0,        # Keep CFG ≤ 6 for stability
    'denoise': 0.35,         # Keep denoise ≤ 0.35
    'batch_size': 1,         # Forced for memory safety
    'fade_steps_ratio': 0.4  # 40% early-step guidance
}
```

### Integration Points
- **U-Net Attention**: Cross-attention injection in SD pipeline
- **CLIP Vision**: Process face embeddings through vision encoder
- **Mask Application**: Restrict influence to detected face regions
- **Step Scheduling**: Apply guidance with linear fade-out

This pipeline ensures maximum realism while maintaining identity accuracy and preventing the plastic skin artifacts common in traditional face swapping approaches.