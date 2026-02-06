# ReActor v5 - Advanced Face Swapping with IP-Adapter FaceID Plus v2

ReActor v5 is an enhanced face swapping extension that integrates IP-Adapter FaceID Plus v2 for improved identity accuracy while maintaining all functionality from ReActor v3.

## 🚀 Key Features

### Enhanced Identity Accuracy
- **IP-Adapter FaceID Plus v2** integration for superior identity preservation
- **Early-step guidance** that fades out after ~40% of diffusion steps
- **Face-mask restricted** influence to prevent style contamination
- **Frequency separation** between identity geometry and skin texture

### VRAM Management
- **Explicit VRAM detection** and monitoring
- **Progressive model loading** - IP-Adapter loaded only when enabled
- **Memory-safe execution** rules with clear error messages
- **Graceful failure** instead of silent downgrades

### Realism Improvements
- **Adaptive face restoration** - only when blur is detected
- **Controlled noise injection** to prevent plastic skin artifacts
- **High-frequency texture preservation** during identity transfer
- **Frequency-aware blending** back into original image

## 🔄 Processing Pipeline

ReActor v5 follows this **mandatory** processing order:

1. **Face Detection** - Using existing ReActor logic
2. **Face Swap** - Identity geometry only, no texture smoothing
3. **IP-Adapter FaceID Plus v2 Guidance** (optional)
   - Applied only during early diffusion steps (~40% fade-out)
   - Influence restricted to face mask region
4. **Identity/Texture Separation**
   - Preserve original high-frequency skin texture
5. **Adaptive Face Restoration** - Only if blur detected
6. **Frequency-Aware Blending** - Back into original image

## 📊 VRAM Requirements

| Configuration | Minimum VRAM | Recommended |
|---|---|---|
| ReActor v5 Only | 6GB | 8GB |
| + IP-Adapter FaceID Plus v2 | 10GB | 12GB |
| + Hi-Res Fix (disabled when IP-Adapter active) | - | - |

## ⚙️ Configuration

### Identity Guidance (New Section)
- **Enable IP-Adapter FaceID Plus v2**: Toggle for enhanced identity guidance
- **Identity Strength**: 0.65-0.75 (recommended range)
- **Texture Preservation**: Controls high-frequency detail retention
- **Lighting Match**: Normalize luminance between source and target

### VRAM Monitoring (Display Only)
- **GPU Name**: Detected graphics card
- **Total VRAM**: Available video memory
- **Free VRAM**: Available at generation start

## 🎛️ Recommended Settings

For **maximum realism**:
- Identity Strength: 0.70
- CFG Scale: ≤ 6.0
- Denoise: ≤ 0.35
- Batch Size: 1 (enforced with IP-Adapter)
- Texture Preservation: 0.8

## ⚠️ Important Notes

- **No Model Fallbacks**: Only IP-Adapter FaceID Plus v2 is supported
- **Memory Management**: Graceful failure with clear error messages
- **Batch Size**: Forced to 1 when IP-Adapter is enabled
- **Compatibility**: A1111, Forge, and ComfyUI supported

## 🔧 Installation

1. Clone this repository to your `extensions` folder
2. Install requirements: `pip install -r requirements.txt`
3. Download IP-Adapter FaceID Plus v2 model (auto-downloaded on first use)
4. Restart your WebUI

## 📁 Model Locations

- **GPEN Models**: `extensions/sd-webui-reactor-v5/models/facerestore_models/`
- **InsightFace Models**: `models/insightface/`
- **IP-Adapter Models**: `models/ipadapter/` (auto-managed)

## 🆚 Differences from ReActor v3

- ✅ **All ReActor v3 functionality preserved**
- ✅ **IP-Adapter FaceID Plus v2** integration
- ✅ **Explicit VRAM management**
- ✅ **Enhanced realism pipeline**
- ✅ **No behavior changes when IP-Adapter disabled**

---

**License**: Same as original ReActor
**Compatibility**: WebUI Automatic1111, WebUI Forge, ComfyUI