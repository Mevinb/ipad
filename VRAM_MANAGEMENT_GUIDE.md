# ReActor V5 - VRAM Management Guide

## 💾 Explicit VRAM Management

ReActor V5 implements transparent VRAM management with **NO silent downgrades** or model fallbacks. 

### Core Principles
1. **Explicit Detection**: Real-time VRAM monitoring and reporting
2. **Clear Error Messages**: Descriptive failures instead of silent changes  
3. **Memory-Safe Rules**: Automatic enforcement of safe configurations
4. **Progressive Loading**: Models loaded only when needed
5. **Immediate Cleanup**: Aggressive memory reclamation after use

## 📊 VRAM Requirements by Configuration

### ReActor V5 Basic Mode
| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| InsightFace (buffalo_l) | ~1.5GB | Face detection + analysis |
| Face Swapper (inswapper) | ~0.8GB | Face swapping model |
| GPEN-512 | ~0.5GB | Face restoration |
| Working Memory | ~1.2GB | Temporary tensors |
| **TOTAL BASIC** | **~4GB** | **Minimum: 6GB, Recommended: 8GB** |

### ReActor V5 + IP-Adapter FaceID Plus v2
| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| Basic Components | ~4GB | As above |
| CLIP Vision | ~0.8GB | Vision encoder |
| IP-Adapter Weights | ~1.2GB | FaceID Plus v2 model |
| Face Embeddings | ~0.1GB | Cached embeddings |
| Additional Working Memory | ~0.9GB | Attention mechanisms |
| **TOTAL WITH IP-ADAPTER** | **~7GB** | **Minimum: 10GB, Recommended: 12GB** |

## ⚙️ Memory-Safe Execution Rules

### Automatic Enforcement
ReActor V5 automatically enforces these rules when IP-Adapter is enabled:

```python
def enforce_memory_safe_settings(settings: Dict) -> Tuple[Dict, List[str]]:
    modified_settings = settings.copy()
    warnings = []
    
    if settings.get('enable_ipadapter', False):
        # Rule 1: Force batch size = 1
        if settings.get('batch_size', 1) > 1:
            modified_settings['batch_size'] = 1
            warnings.append("Batch size forced to 1 for IP-Adapter compatibility")
        
        # Rule 2: Disable Hi-Res Fix
        if settings.get('enable_hr', False):
            modified_settings['enable_hr'] = False
            warnings.append("Hi-Res Fix disabled for IP-Adapter VRAM safety")
        
        # Rule 3: Enforce FP16
        modified_settings['fp16_enabled'] = True
    
    return modified_settings, warnings
```

### Conflict Detection
- **IP-Adapter + Hi-Res Fix**: Automatically disabled
- **IP-Adapter + Multiple ControlNet**: Warning issued
- **Large Batch Sizes**: Automatically reduced to 1
- **FP32 Models**: Automatically converted to FP16

## 🔍 Real-Time VRAM Monitoring

### VRAM Status Display
The UI shows real-time VRAM information:

```
GPU: NVIDIA RTX 4090 | Total: 24.0GB | Free: 18.2GB | IP-Adapter: ✓
```

### Status Components
- **GPU Name**: Detected graphics card
- **Total VRAM**: Maximum available video memory
- **Free VRAM**: Currently available memory
- **IP-Adapter Ready**: ✓/✗ indicator for IP-Adapter compatibility

### Update Triggers
- Manual refresh button
- IP-Adapter enable/disable
- Model loading/unloading
- Generation start/end

## 🚨 Error Handling & Graceful Failure

### VRAM Insufficiency
When VRAM is insufficient, ReActor V5 **fails gracefully** with clear messages:

```python
def check_vram_requirements(enable_ipadapter: bool, resolution: Tuple[int, int], 
                          batch_size: int) -> Tuple[bool, str]:
    if not enable_ipadapter:
        return True, "ReActor V5 basic mode - VRAM check passed"
    
    free_vram = get_free_vram()
    required_vram = estimate_ipadapter_vram_usage(resolution, batch_size) + 1.0
    
    if free_vram < required_vram:
        return False, (
            f"Insufficient VRAM for IP-Adapter FaceID Plus v2. "
            f"Required: {required_vram:.1f}GB, Available: {free_vram:.1f}GB"
        )
    
    return True, f"VRAM check passed. Using {required_vram-1:.1f}GB of {free_vram:.1f}GB available"
```

### Error Message Examples
```
❌ "Insufficient VRAM for IP-Adapter FaceID Plus v2. Required: 8.2GB, Available: 6.1GB"
❌ "IP-Adapter requires CUDA-compatible GPU. CPU mode not supported."
❌ "Failed to load IP-Adapter model. Check model file integrity."
✅ "VRAM check passed. Using 7.1GB of 12.8GB available"
```

### No Silent Behavior Changes
ReActor V5 **NEVER** silently:
- Reduces image resolution
- Decreases sampling steps  
- Changes model quality
- Disables features without notification

## 🔄 Progressive Memory Management

### Loading Strategy
```python
class ProgressiveLoader:
    def load_for_processing(self, enable_ipadapter: bool):
        # Always load basic components
        self.load_insightface()
        self.load_face_swapper()
        
        # Load IP-Adapter only if enabled and VRAM sufficient
        if enable_ipadapter:
            if not self.check_vram_capacity():
                raise VRAMInsufficientError()
            self.load_ipadapter()
        
        # Load restoration model on-demand
        if restore_model_needed:
            self.load_gpen_restorer()
    
    def cleanup_after_processing(self, aggressive: bool = False):
        # Always unload IP-Adapter (biggest memory user)
        if self.ipadapter_loaded:
            self.unload_ipadapter()
        
        # Conditionally unload other components
        if aggressive:
            self.unload_all_models()
        
        # GPU memory cleanup
        self.cleanup_gpu_memory()
```

### Memory Cleanup Levels

#### Standard Cleanup (Default)
- Unload IP-Adapter immediately
- Keep InsightFace models cached
- GPU cache cleanup
- **VRAM Freed**: ~2-3GB

#### Aggressive Cleanup (Low VRAM Systems)
- Unload ALL models
- Clear all caches
- Force garbage collection
- **VRAM Freed**: ~4-5GB

```python
def progressive_cleanup(self, level: str = 'standard'):
    if level in ['standard', 'aggressive']:
        torch.cuda.empty_cache()
        if level == 'aggressive':
            torch.cuda.ipc_collect()
            gc.collect()
    
    self.log_vram_status()
```

## 📈 VRAM Optimization Techniques

### FP16 Enforcement
All models automatically use FP16 precision:
```python
self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = model.to(device=self.device, dtype=self.dtype)
```

### Batch Size Limitation
```python
if self.enable_ipadapter and batch_size > 1:
    batch_size = 1
    self.log_warning("Batch size limited to 1 for IP-Adapter stability")
```

### Memory Pooling
- Reuse tensor allocations where possible
- Pre-allocate common tensor sizes
- Avoid memory fragmentation

### Gradient Disabled
```python
with torch.no_grad():
    # All inference operations
    result = model(inputs)
```

## 🎛️ Configuration Recommendations

### For Different VRAM Capacities

#### < 8GB VRAM (GTX 1070, RTX 2060)
```python
RECOMMENDED_CONFIG_LOW_VRAM = {
    'enable_ipadapter': False,
    'restore_model': 'GPEN-BFR-512.onnx',
    'batch_size': 1,
    'aggressive_cleanup': True,
    'fp16_enabled': True
}
```

#### 8-12GB VRAM (RTX 3070, RTX 4070)
```python
RECOMMENDED_CONFIG_MEDIUM_VRAM = {
    'enable_ipadapter': False,  # Use basic mode
    'restore_model': 'GPEN-BFR-1024.onnx',
    'batch_size': 1,
    'aggressive_cleanup': False,
    'frequency_blending': True
}
```

#### 12GB+ VRAM (RTX 3080Ti, RTX 4080, RTX 4090)
```python
RECOMMENDED_CONFIG_HIGH_VRAM = {
    'enable_ipadapter': True,
    'ipadapter_weight': 0.70,
    'restore_model': 'GPEN-BFR-1024.onnx',
    'batch_size': 1,  # Still limited for stability
    'all_v5_features': True
}
```

## 🔧 Troubleshooting VRAM Issues

### Common Issues & Solutions

#### "CUDA out of memory" Error
1. **Check VRAM Status**: Verify available memory
2. **Disable IP-Adapter**: Reduce memory usage by ~3GB
3. **Enable Aggressive Cleanup**: Free all models after use
4. **Restart WebUI**: Clear any memory leaks

#### IP-Adapter Won't Load
1. **VRAM Check**: Ensure 10GB+ available
2. **Close Other Applications**: Free up GPU memory
3. **Check GPU Compatibility**: Requires CUDA-compatible GPU
4. **Model Download**: Verify IP-Adapter models downloaded correctly

#### Slow Performance
1. **Use GPEN-512**: Instead of GPEN-1024 for faster processing
2. **Disable Advanced Features**: Turn off frequency blending
3. **Standard Cleanup**: Avoid aggressive cleanup if VRAM sufficient

### Debug Commands
```python
# Get detailed VRAM status
status = vram_manager.get_status_report()
print(f"GPU: {status['gpu_name']}")
print(f"Total: {status['total_vram_gb']:.1f}GB")
print(f"Free: {status['free_gb']:.1f}GB")
print(f"IP-Adapter Ready: {status['can_use_ipadapter']}")

# Force cleanup
vram_manager.progressive_cleanup('aggressive')

# Check model memory usage
print(f"Models loaded: {engine.get_loaded_models()}")
```

Remember: ReActor V5 prioritizes **transparency** and **predictability** over automatic optimization. Users always know exactly what's happening with their VRAM.