# ReActor V5 - Complete Setup Guide

## 🚀 Quick Start

### 1. Installation
1. Clone/download ReActor V5 to your WebUI `extensions` folder
2. Restart your WebUI 
3. Install will run automatically on first load

### 2. Model Setup
**GPEN Models** (Place in `extensions/sd-webui-reactor-v5/models/facerestore_models/`):
- Download GPEN-BFR-512.onnx and GPEN-BFR-1024.onnx
- **IP-Adapter Models**: Auto-downloaded on first use to `models/ipadapter/`

### 3. First Run
1. Find "ReActor V5 (IP-Adapter + GPEN)" in img2img/txt2img tabs
2. Upload source face image
3. Enable ReActor V5
4. Generate images - faces will be automatically processed

## 📋 Detailed Setup Instructions

### System Requirements

#### Minimum System (Basic Mode)
- **GPU**: GTX 1060 6GB / RTX 2060 
- **VRAM**: 6GB minimum, 8GB recommended
- **RAM**: 16GB system RAM
- **Storage**: 5GB for models
- **CUDA**: 11.0 or later

#### Recommended System (Full Features)
- **GPU**: RTX 3080 / RTX 4070 / RTX 4080 / RTX 4090
- **VRAM**: 12GB or more
- **RAM**: 32GB system RAM  
- **Storage**: 8GB for all models
- **CUDA**: 11.8 or later

### Installation Methods

#### Method 1: Git Clone (Recommended)
```bash
cd /path/to/webui/extensions
git clone https://github.com/your-repo/sd-webui-reactor-v5.git
```

#### Method 2: Manual Download
1. Download ReActor V5 ZIP
2. Extract to `webui/extensions/sd-webui-reactor-v5/`
3. Ensure folder structure is correct

#### Method 3: WebUI Extension Manager
1. Go to Extensions → Available
2. Search for "ReActor V5"  
3. Click Install
4. Apply and restart UI

### Model Downloads

#### Required: GPEN Face Restoration Models
Download and place in `extensions/sd-webui-reactor-v5/models/facerestore_models/`:

**GPEN-BFR-512.onnx**
- Size: ~150MB
- Resolution: 512×512
- Speed: Fast
- Quality: Good
- Use case: General purpose, batch processing

**GPEN-BFR-1024.onnx** 
- Size: ~600MB
- Resolution: 1024×1024  
- Speed: Slower
- Quality: Excellent
- Use case: High-quality portraits, single images

#### Auto-Downloaded: IP-Adapter Models
These download automatically on first IP-Adapter use:

**IP-Adapter FaceID Plus v2**
- Location: `models/ipadapter/ip-adapter-faceid-plus_sd15.bin`
- Size: ~1.2GB
- Purpose: Identity guidance enhancement

**CLIP Vision Model**  
- Location: `models/ipadapter/clip-vit-large-patch14/`
- Size: ~800MB
- Purpose: Vision encoding for IP-Adapter

#### Auto-Downloaded: InsightFace Models
These download automatically on first use:

**Buffalo_L**
- Location: `models/insightface/models/buffalo_l/`
- Size: ~400MB
- Purpose: Face detection and analysis

**InSwapper**
- Location: `models/insightface/inswapper_128.onnx` 
- Size: ~800MB
- Purpose: Face swapping model

### Directory Structure Verification
After setup, verify this structure exists:

```
webui/
├── extensions/
│   └── sd-webui-reactor-v5/
│       ├── scripts/
│       │   ├── !!reactor_v5_ui.py
│       │   ├── reactor_v5_swapper.py
│       │   ├── ipadapter_faceid.py
│       │   ├── realism_enhancer.py
│       │   ├── vram_management.py
│       │   └── reactor_v5_gpen_restorer.py
│       ├── models/
│       │   └── facerestore_models/
│       │       ├── GPEN-BFR-512.onnx     # You download
│       │       └── GPEN-BFR-1024.onnx   # You download
│       ├── README.md
│       ├── requirements.txt
│       └── install.py
└── models/
    ├── insightface/              # Auto-downloaded
    ├── ipadapter/               # Auto-downloaded
    └── facerestore_models/      # Shared location (optional)
```

## ⚙️ Configuration Guide

### Basic Configuration (First Time Users)

#### Step 1: Enable ReActor V5
1. Open WebUI → img2img or txt2img tab
2. Scroll to "ReActor V5 (IP-Adapter + GPEN)" section
3. Check "Enable ReActor V5"
4. Upload source face image

#### Step 2: Basic Settings
- **Source Face Index**: 0 (first detected face)
- **Target Face Index**: 0 (first detected face)  
- **GPEN Model**: Start with "GPEN-BFR-512.onnx"
- **Gender Matching**: "Smart Match (Auto-detect)"

#### Step 3: Generate
- Generate images normally
- ReActor V5 will automatically process faces after generation

### Advanced Configuration

#### Memory Management Settings
```
Aggressive Memory Cleanup: ✓ (if <12GB VRAM)
Auto-Select Resolution: ✓
```

#### IP-Adapter Settings (12GB+ VRAM)
```
Enable IP-Adapter FaceID Plus v2: ✓
Identity Strength: 0.70
Texture Preservation: 0.8
Lighting Match: ✓  
Inject Realistic Noise: ✓
```

#### Advanced Realism
```
Adaptive Face Restoration: ✓
Frequency-Aware Blending: ✓
Max Restoration Strength: 0.35
```

### VRAM-Optimized Configurations

#### 6-8GB VRAM (GTX 1070, RTX 2060)
```python
BASIC_CONFIG = {
    'enable_ipadapter': False,
    'restore_model': 'GPEN-BFR-512.onnx', 
    'aggressive_cleanup': True,
    'frequency_blending': False,
    'inject_noise': False
}
```

#### 8-12GB VRAM (RTX 3070, RTX 4070)
```python
BALANCED_CONFIG = {
    'enable_ipadapter': False,
    'restore_model': 'GPEN-BFR-1024.onnx',
    'aggressive_cleanup': False, 
    'frequency_blending': True,
    'inject_noise': True,
    'texture_preservation': 0.8
}
```

#### 12GB+ VRAM (RTX 4080, RTX 4090)
```python
FULL_CONFIG = {
    'enable_ipadapter': True,
    'ipadapter_weight': 0.70,
    'restore_model': 'GPEN-BFR-1024.onnx',
    'aggressive_cleanup': False,
    'frequency_blending': True,
    'inject_noise': True,
    'texture_preservation': 0.8,
    'adaptive_restoration': True
}
```

## 🎯 Usage Workflows

### Workflow 1: Basic Face Swap
**Goal**: Simple, fast face replacement
**Requirements**: 6GB+ VRAM

1. **Setup**:
   - Enable ReActor V5: ✓
   - IP-Adapter: ✗
   - GPEN Model: 512 or None

2. **Process**:
   - Upload source face
   - Generate images  
   - Faces automatically swapped

3. **Result**: Clean face swap with basic blending

### Workflow 2: High-Quality Portrait  
**Goal**: Maximum quality for single portraits
**Requirements**: 8GB+ VRAM

1. **Setup**:
   - Enable ReActor V5: ✓
   - IP-Adapter: ✗
   - GPEN Model: 1024
   - Advanced features: All enabled

2. **Process**:
   - Use high-quality source face (512×512+)
   - Generate single image (batch size 1)
   - Allow extra processing time

3. **Result**: Ultra-high quality with natural skin texture

### Workflow 3: IP-Adapter Enhanced (Premium)
**Goal**: Best possible identity accuracy
**Requirements**: 12GB+ VRAM

1. **Setup**:
   - Enable ReActor V5: ✓  
   - IP-Adapter: ✓
   - Identity Strength: 0.70
   - All realism features: ✓

2. **Generation Settings**:
   - CFG Scale: ≤ 6.0
   - Denoise: ≤ 0.35
   - Batch Size: 1 (auto-enforced)

3. **Process**:
   - Upload high-quality source face
   - Generate with extra patience (slower)

4. **Result**: Perfect identity match with photorealistic skin

### Workflow 4: Batch Processing
**Goal**: Process many images efficiently  
**Requirements**: Variable

1. **Setup**:
   - Enable ReActor V5: ✓
   - IP-Adapter: ✗ (for speed)
   - GPEN Model: 512 
   - Aggressive Cleanup: ✓

2. **Process**:
   - Generate large batches
   - Each image processed individually
   - Memory cleaned between images

3. **Result**: Consistent quality across many images

## 🛠️ Troubleshooting

### Common Issues

#### "No face detected in source"
- **Cause**: Source image too small, blurry, or face obscured
- **Solution**: Use clear, front-facing source image ≥256×256

#### "Insufficient VRAM for IP-Adapter" 
- **Cause**: GPU has <10GB available VRAM
- **Solutions**:
  - Disable IP-Adapter  
  - Close other applications
  - Enable aggressive cleanup
  - Reduce image resolution

#### "Model failed to load"
- **Cause**: Missing or corrupted model files
- **Solutions**:
  - Re-download GPEN models
  - Check file paths and permissions
  - Clear model cache and restart

#### "CUDA out of memory"
- **Cause**: VRAM exhausted during processing
- **Solutions**:
  - Enable aggressive cleanup
  - Reduce batch size to 1
  - Disable IP-Adapter
  - Restart WebUI

#### Poor face swap quality
- **Cause**: Source/target mismatch or settings
- **Solutions**:
  - Use higher quality source image
  - Try different face indices
  - Enable GPEN restoration
  - Check gender matching settings

#### Plastic/fake looking skin
- **Cause**: Over-restoration or disabled realism features  
- **Solutions**:
  - Reduce restoration strength
  - Enable noise injection
  - Enable frequency blending
  - Lower texture preservation

### Performance Optimization

#### For Speed
- Use GPEN-512 instead of 1024
- Disable IP-Adapter
- Disable frequency blending
- Enable aggressive cleanup

#### For Quality  
- Use GPEN-1024
- Enable all realism features
- Higher texture preservation (0.9)
- Slower but better results

#### For Memory Efficiency
- Aggressive cleanup: ✓
- Batch size: 1
- Basic mode only
- Regular VRAM monitoring

### Debug Information

#### Check VRAM Status
1. Look at VRAM display in ReActor V5 UI
2. Click "Update VRAM Status" for refresh
3. Monitor before/after generation

#### Verify Model Loading
- Check console for model loading messages
- Look for error messages during initialization
- Verify file sizes match expected values

#### Test Basic Functionality
1. Test with IP-Adapter disabled first
2. Try simple face swap without restoration
3. Gradually enable features to isolate issues

## 📞 Support Resources

### Documentation
- `README.md` - Overview and features
- `PIPELINE_ARCHITECTURE.md` - Technical details  
- `VRAM_MANAGEMENT_GUIDE.md` - Memory optimization
- `REALISM_GUIDE.md` - Advanced realism features

### Community
- GitHub Issues - Bug reports and feature requests
- Discord/Reddit - Community support
- Wiki - Additional tutorials and examples

### Model Sources
- **GPEN**: https://github.com/yangxy/GPEN
- **IP-Adapter**: https://github.com/tencent-ailab/IP-Adapter
- **InsightFace**: https://github.com/deepinsight/insightface

Remember: ReActor V5 is designed for **transparency** and **predictability**. If something isn't working as expected, check the console messages for detailed information about what's happening.