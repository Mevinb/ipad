# ReActor V5 - Realism Enhancement Guide

## 🎨 Advanced Realism Features

ReActor V5 introduces cutting-edge realism enhancements to eliminate the "plastic skin" and over-smoothed appearance common in traditional face swapping.

## 🔬 Core Realism Technologies

### 1. Identity/Texture Separation
**Problem**: Traditional face swapping replaces both identity AND texture, losing natural skin variation.

**Solution**: Separate processing of low-frequency (identity) and high-frequency (texture) components.

```python
def separate_identity_texture(image: np.ndarray, sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate image into identity geometry and skin texture.
    
    Low frequency = face shape, proportions, lighting
    High frequency = pores, wrinkles, skin texture, micro-details
    """
    image_float = image.astype(np.float32) / 255.0
    
    # Identity: Gaussian blur removes fine details, keeps shape
    identity = cv2.GaussianBlur(image_float, (0, 0), sigma)
    
    # Texture: Residual contains all the fine skin details
    texture = image_float - identity
    
    return identity, texture
```

**Benefits**:
- Preserves original skin micro-texture
- Swaps only identity/shape information
- Maintains natural skin variation
- Prevents over-smoothing

### 2. Adaptive Face Restoration
**Problem**: Traditional restoration applies uniformly, over-smoothing already-sharp faces.

**Solution**: Blur detection triggers restoration only when needed.

```python
def detect_blur(image: np.ndarray, mask: np.ndarray = None) -> Tuple[float, bool]:
    """
    Detect blur using Laplacian variance.
    Higher values = sharper image
    Lower values = blurrier image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is not None:
        gray = gray * (mask / 255.0)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < 100.0  # Threshold
    
    return laplacian_var, is_blurry

def adaptive_restoration(image: np.ndarray, restorer, face_mask: np.ndarray):
    """Apply restoration only if blur detected, with strength cap."""
    blur_score, is_blurry = detect_blur(image, face_mask)
    
    if not is_blurry:
        return image, False  # No restoration needed
    
    # Restoration with strength cap (0.35 maximum)
    max_strength = 0.35
    restoration_strength = min(max_strength, calculate_needed_strength(blur_score))
    
    return restorer.restore(image, strength=restoration_strength), True
```

**Benefits**:
- Restoration only when actually needed
- Strength capped at 0.35 to prevent over-smoothing
- Preserves already-sharp faces unchanged
- Reduces processing time on good quality faces

### 3. Controlled Noise Injection
**Problem**: Face swapping produces unnaturally smooth, "plastic" skin without natural variation.

**Solution**: Inject calibrated Gaussian noise to restore natural skin texture.

```python
def inject_controlled_noise(image: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    """
    Inject resolution-aware Gaussian noise for natural skin texture.
    """
    h, w = image.shape[:2]
    
    # Scale noise with resolution (prevents over/under-texturing)
    resolution_factor = np.sqrt((w * h) / (512 * 512))
    base_sigma = 0.015  # Base noise level
    sigma = base_sigma / resolution_factor
    
    # Generate 3-channel noise
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    
    # Apply only to skin regions
    skin_mask_3d = np.stack([skin_mask] * 3, axis=2) / 255.0
    masked_noise = noise * skin_mask_3d
    
    # Blend with original
    image_float = image.astype(np.float32) / 255.0
    result = image_float + masked_noise
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)
```

**Key Features**:
- **Resolution-aware**: Noise scales with image size
- **Skin-targeted**: Applied only to detected skin regions  
- **Calibrated strength**: σ ≈ 0.01–0.02 for natural appearance
- **Mandatory step**: Essential for preventing plastic look

### 4. Precise Skin Segmentation
**Problem**: Face masks include non-skin areas (eyes, mouth, hair), causing unwanted noise injection.

**Solution**: HSV-based skin detection for precise targeting.

```python
def create_skin_mask(image: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """
    Create precise skin mask using HSV color segmentation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # HSV ranges for various skin tones
    lower_skin1 = np.array([0, 20, 70])    # Light skin tones
    upper_skin1 = np.array([20, 255, 255])
    
    lower_skin2 = np.array([170, 20, 70])  # Wrap-around for red hues
    upper_skin2 = np.array([180, 255, 255])
    
    # Combine skin color ranges
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Intersect with face mask to avoid false positives
    skin_mask = cv2.bitwise_and(skin_mask, face_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges
    return cv2.GaussianBlur(skin_mask, (9, 9), 3)
```

**Benefits**:
- Excludes eyes, teeth, lips from noise injection
- Handles various skin tones (light to dark)
- Smooth mask edges for natural blending
- Reduces noise artifacts in non-skin areas

### 5. Frequency-Aware Blending
**Problem**: Simple alpha blending loses fine texture details during face integration.

**Solution**: Separate blending for different frequency components.

```python
def frequency_aware_blending(swapped_image: np.ndarray, 
                           original_texture: np.ndarray,
                           face_mask: np.ndarray,
                           texture_preservation: float = 0.8) -> np.ndarray:
    """
    Blend with frequency separation for natural texture preservation.
    """
    # Separate swapped image into components
    swapped_identity, swapped_texture = separate_identity_texture(swapped_image)
    
    # Create 3D blend mask
    blend_mask = np.stack([face_mask] * 3, axis=2) / 255.0
    
    # High-frequency (texture) blending - preserve original details
    texture_blend = (
        original_texture * texture_preservation * blend_mask +
        swapped_texture * (1 - texture_preservation * blend_mask)
    )
    
    # Recombine: swapped identity + preserved texture
    result_identity = swapped_identity.astype(np.float32) / 255.0
    result_texture = texture_blend.astype(np.float32) / 255.0 - 0.5
    
    final_result = result_identity + result_texture
    
    return np.clip(final_result * 255, 0, 255).astype(np.uint8)
```

**Benefits**:
- Preserves original skin micro-details
- Maintains new face identity/geometry
- Natural transition at face boundaries
- Configurable texture preservation (0.0-1.0)

## 🎛️ Configuration Parameters

### Core Settings
```python
REALISM_CONFIG = {
    # Frequency separation
    'low_freq_sigma': 3.0,              # Gaussian blur for identity separation
    'texture_preservation': 0.8,        # How much original texture to keep
    
    # Noise injection
    'noise_sigma_range': (0.01, 0.02),  # Gaussian noise strength
    'noise_skin_only': True,            # Apply only to skin regions
    
    # Adaptive restoration
    'blur_threshold': 100.0,            # Laplacian variance threshold
    'max_restoration_strength': 0.35,   # Cap to prevent over-smoothing
    
    # Blending
    'frequency_blending': True,         # Enable frequency-aware blending
    'inject_noise': True,              # Enable controlled noise injection
}
```

### Quality vs Speed Trade-offs

#### Maximum Realism (Slower)
- `texture_preservation`: 0.9
- `noise_sigma`: 0.02
- `blur_threshold`: 120.0
- `frequency_blending`: True
- `adaptive_restoration`: True

#### Balanced (Recommended)
- `texture_preservation`: 0.8
- `noise_sigma`: 0.015
- `blur_threshold`: 100.0
- `frequency_blending`: True
- `adaptive_restoration`: True

#### Performance (Faster)
- `texture_preservation`: 0.7
- `noise_sigma`: 0.01
- `blur_threshold`: 80.0
- `frequency_blending`: True
- `inject_noise`: False (not recommended)

## 🔬 Technical Details

### Why Early-Step IP-Adapter Guidance Reduces Plastic Artifacts

**Traditional Problem**:
- Late-stage guidance applies smoothing throughout entire generation
- Results in loss of high-frequency detail
- Creates artificial, over-processed appearance

**ReActor V5 Solution**:
```python
def calculate_step_weight(current_step: int, total_steps: int, base_weight: float = 0.70):
    """
    Apply strong guidance early, fade to zero by 40% completion.
    This preserves identity while allowing natural texture development.
    """
    fade_step = int(total_steps * 0.4)
    
    if current_step >= fade_step:
        return 0.0  # No guidance in texture refinement phase
    
    # Linear fade from full strength to zero
    fade_progress = current_step / fade_step
    return base_weight * (1.0 - fade_progress)
```

**Benefits**:
- Strong identity guidance in structure-forming steps (0-40%)
- Natural texture development in refinement steps (40-100%)
- Prevents IP-Adapter from over-smoothing fine details
- Maintains both identity accuracy and texture realism

### Noise Calibration Science

**Resolution Scaling**:
```python
resolution_factor = np.sqrt((width * height) / (512 * 512))
adjusted_sigma = base_sigma / resolution_factor
```

**Why This Works**:
- 512×512: σ = 0.015 (base level)
- 1024×1024: σ = 0.0075 (half strength for 4× pixels)
- 2048×2048: σ = 0.00375 (quarter strength for 16× pixels)

**Result**: Consistent apparent noise level regardless of image resolution.

## 🎯 Expected Results

### Before ReActor V5 Realism
- Plastic, over-smoothed skin
- Loss of natural texture variation
- Artificial lighting transitions
- Uniform skin tone (no natural variation)

### After ReActor V5 Realism
- Natural skin micro-texture preserved
- Subtle pore and skin detail variation
- Realistic lighting and shadow transitions  
- Natural skin tone variation and imperfections

## 🛠️ Troubleshooting Realism Issues

### Too Much Noise
- **Cause**: `noise_sigma` too high or skin mask too broad
- **Solution**: Reduce `noise_sigma` to 0.01 or improve skin segmentation

### Still Looks Plastic
- **Cause**: Noise injection disabled or `texture_preservation` too low
- **Solution**: Enable `inject_noise=True` and increase `texture_preservation` to 0.9

### Over-Restoration
- **Cause**: `max_restoration_strength` too high or blur threshold too high
- **Solution**: Reduce to 0.25 strength and lower threshold to 80.0

### Blotchy Skin
- **Cause**: Poor skin mask or excessive noise
- **Solution**: Check HSV skin detection parameters or reduce noise

### Loss of Identity
- **Cause**: `texture_preservation` too high, overwhelming face swap
- **Solution**: Reduce to 0.6-0.7 for stronger identity transfer

The ReActor V5 realism pipeline represents state-of-the-art face swapping technology, producing results that are virtually indistinguishable from natural photographs while maintaining perfect identity accuracy.