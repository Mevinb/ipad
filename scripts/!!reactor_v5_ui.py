"""
ReActor V5 - Enhanced WebUI Integration

Enhanced face swapping extension with IP-Adapter FaceID Plus v2, 
explicit VRAM management, and advanced realism features.
"""

import os
import sys
import gradio as gr
import modules.scripts as scripts
from modules import images
from modules.processing import Processed
from modules.shared import opts, state
from PIL import Image
import numpy as np
import cv2

# Add extension scripts path for imports
_ext_scripts_path = os.path.dirname(os.path.abspath(__file__))
if _ext_scripts_path not in sys.path:
    sys.path.insert(0, _ext_scripts_path)

# Import ReActor V5 components
from reactor_v5_swapper_fixed import get_reactor_v5_engine
from vram_management import get_vram_manager, format_vram_status


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format"""
    if pil_img is None:
        return None
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image"""
    if cv2_img is None:
        return None
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_available_models():
    """Get list of available GPEN models"""
    try:
        # Get the WebUI root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extension_dir = os.path.dirname(current_dir)
        extensions_dir = os.path.dirname(extension_dir)
        webui_dir = os.path.dirname(extensions_dir)
        
        # Use shared WebUI models directory
        models_path = os.path.join(webui_dir, 'models')
        engine = get_reactor_v5_engine(models_path)
        return engine.get_available_restorers()
    except Exception as e:
        print(f"[ReActor V5] Error getting models: {e}")
        import traceback
        traceback.print_exc()
        return ['None', 'GPEN-BFR-512.onnx']


def get_default_restore_model():
    """Get the recommended default restoration model"""
    models = get_available_models()
    # Prefer GPEN-BFR-512 for best balance of speed and quality
    for model in models:
        if 'GPEN-BFR-512' in model:
            return model
    # Fall back to any GPEN model
    for model in models:
        if 'GPEN' in model and model != 'None':
            return model
    return 'None'


def refresh_vram_status():
    """Refresh VRAM status display"""
    try:
        return format_vram_status()
    except Exception:
        return "VRAM status unavailable"


class ReactorV5Script(scripts.Script):
    """
    ReActor V5 - Advanced High-Fidelity Face Swapping
    
    Features all ReActor v3 functionality plus:
    - IP-Adapter FaceID Plus v2 identity guidance
    - Explicit VRAM management
    - Advanced realism enhancements
    - Frequency-aware processing
    """
    
    def title(self):
        return "ReActor V5 (IP-Adapter + GPEN)"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        """Create enhanced UI with ReActor V5 features"""
        
        with gr.Accordion("ReActor V5 - Advanced Face Swapping", open=False):
            
            # Header with VRAM status
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable ReActor V5",
                    value=False,
                    info="Advanced face swapping with IP-Adapter FaceID Plus v2"
                )
                vram_status = gr.Textbox(
                    label="VRAM Status",
                    value=refresh_vram_status(),
                    interactive=False,
                    scale=2
                )
            
            gr.Markdown("""
            **🚀 ReActor V5** - Enhanced identity accuracy with IP-Adapter FaceID Plus v2 and advanced realism
            """)
            
            # Basic Face Swapping (backward compatible with V3)
            with gr.Tab("Face Swapping"):
                with gr.Row():
                    with gr.Column():
                        source_image = gr.Image(
                            label="Source Face",
                            type="pil",
                            interactive=True
                        )
                        source_face_index = gr.Slider(
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=0,
                            label="Source Face Index",
                            info="Which face to use if multiple detected"
                        )
                    
                    with gr.Column():
                        target_face_index = gr.Slider(
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=0,
                            label="Target Face Index", 
                            info="Which face to replace in generated image"
                        )
                        
                        restore_model = gr.Dropdown(
                            label="GPEN Restoration Model",
                            choices=get_available_models(),
                            value=get_default_restore_model(),  # Use GPEN-BFR-512 by default for quality
                            info="512 = fast (required to fix blur), 1024 = ultra-quality"
                        )
                
                with gr.Row():
                    gender_match = gr.Radio(
                        label="Gender Matching Mode",
                        choices=[
                            ("All (No Filter)", "A"),
                            ("Smart Match (Auto-detect)", "S"),
                            ("Male Only", "M"),
                            ("Female Only", "F")
                        ],
                        value="S",
                        info="S=Auto match gender, M/F=Filter specific gender"
                    )
                
                with gr.Row():
                    auto_resolution = gr.Checkbox(
                        label="Auto-Select Resolution",
                        value=True,
                        info="Automatically choose 512 or 1024 based on face size"
                    )
                    
                    aggressive_cleanup = gr.Checkbox(
                        label="Aggressive Memory Cleanup",
                        value=True,
                        info="Clear model cache after each image (<12GB VRAM)"
                    )
            
            # Identity Guidance (New V5 Section)
            with gr.Tab("Identity Guidance"):
                gr.Markdown("""
                **🎯 IP-Adapter FaceID Plus v2** - Enhanced identity preservation during generation
                """)
                
                with gr.Row():
                    enable_ipadapter = gr.Checkbox(
                        label="Enable IP-Adapter FaceID Plus v2",
                        value=False,
                        info="Requires additional VRAM - enhances identity accuracy"
                    )
                
                with gr.Row():
                    ipadapter_weight = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.70,
                        label="Identity Strength",
                        info="0.65-0.75 recommended for best results"
                    )
                    
                    texture_preservation = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.0,  # DISABLED by default - causes blur
                        label="Texture Preservation",
                        info="How much original skin texture to preserve (0=disabled)"
                    )
                
                with gr.Row():
                    lighting_match = gr.Checkbox(
                        label="Lighting Match",
                        value=True,
                        info="Normalize luminance between source and target"
                    )
                    
                    skin_texture_strength = gr.Slider(
                        minimum=0.0,
                        maximum=0.3,
                        step=0.01,
                        value=0.08,  # Default: subtle texture to combat plasticky GPEN
                        label="Skin Texture",
                        info="Add realistic skin texture (0=smooth, 0.08=natural, 0.15+=visible)"
                    )
                
                gr.Markdown("""
                **💡 IP-Adapter Integration:**
                - Use Forge's ControlNet with `InsightFace+CLIP-H (IPAdapter)` preprocessor
                - Model: `ip-adapter-faceid-plusv2_sd15` (download from HuggingFace h94/IP-Adapter-FaceID)
                - Weight 0.6-0.8, End Step 0.4 for best results
                - Upload same source face to ControlNet and ReActor for combined effect
                """)
            
            # Advanced Settings (V5 Features)
            with gr.Tab("Advanced"):
                gr.Markdown("""
                **⚙️ Advanced Realism Settings**
                """)
                
                with gr.Row():
                    adaptive_restoration = gr.Checkbox(
                        label="Adaptive Face Restoration",
                        value=True,
                        info="Apply restoration only when blur detected"
                    )
                    
                    frequency_blending = gr.Checkbox(
                        label="Frequency-Aware Blending", 
                        value=False,  # DISABLED by default - main cause of blur
                        info="Experimental: Separate identity and texture (can cause blur)"
                    )
                
                with gr.Row():
                    restoration_strength_limit = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.05,
                        value=0.35,
                        label="Max Restoration Strength",
                        info="Cap restoration to prevent over-smoothing"
                    )
            
            # Controls
            with gr.Row():
                refresh_button = gr.Button("🔄 Refresh Models & VRAM")
                vram_refresh_button = gr.Button("📊 Update VRAM Status")
            
            # Help and Tips
            gr.Markdown("""
            ---
            
            ### 📋 ReActor V5 Processing Pipeline
            
            1. **Face Detection** - InsightFace with gender matching
            2. **Face Swap** - Identity geometry via inswapper_128  
            3. **Face Restoration** - GPEN-BFR-512 for detail recovery
            4. **Skin Texture** - Natural pore/texture overlay (combats plasticky look)
            5. **Blending** - Seamless integration with original image
            
            ### 🎛️ IP-Adapter Integration (via ControlNet)
            
            For best results, use Forge's built-in IP-Adapter via ControlNet:
            
            1. Enable ControlNet Unit 0
            2. Preprocessor: `InsightFace+CLIP-H (IPAdapter)`
            3. Model: `ip-adapter-faceid-plusv2_sd15` 
            4. Upload source face image
            5. Weight: 0.6-0.8, End Step: 0.4
            6. Enable ReActor V5 with same source image
            
            This combines identity-guided generation + precise face swap.
            
            ### 🎨 Recommended Settings
            
            **For Natural Skin:**
            - Face Restoration: GPEN-BFR-512
            - Skin Texture: 0.08 (subtle) to 0.15 (visible)
            - Frequency Blending: OFF
            
            **For Sharp Identity:**
            - Use IP-Adapter via ControlNet + ReActor
            - Identity Strength: 0.70
            - Face Restoration: GPEN-BFR-512
            
            ### 📁 Model Locations
            - **GPEN Models**: `extensions/sd-webui-reactor-v5/models/facerestore_models/`
            - **IP-Adapter**: Auto-downloaded to `models/ipadapter/`
            - **InsightFace**: `models/insightface/`
            """)
            
            # Event handlers
            refresh_button.click(
                fn=lambda: [gr.Dropdown.update(choices=get_available_models()), refresh_vram_status()],
                inputs=[],
                outputs=[restore_model, vram_status]
            )
            
            vram_refresh_button.click(
                fn=refresh_vram_status,
                inputs=[],
                outputs=[vram_status]
            )
            
            # Auto-refresh VRAM when IP-Adapter is toggled
            enable_ipadapter.change(
                fn=refresh_vram_status,
                inputs=[],
                outputs=[vram_status]
            )
        
        return [
            # Basic settings (backward compatible)
            enabled, source_image, source_face_index, target_face_index,
            restore_model, gender_match, auto_resolution, aggressive_cleanup,
            # V5 identity guidance
            enable_ipadapter, ipadapter_weight, texture_preservation, 
            lighting_match, skin_texture_strength,
            # V5 advanced settings
            adaptive_restoration, frequency_blending, restoration_strength_limit
        ]
    
    def postprocess_image(self, p, pp, 
                         # Basic settings
                         enabled, source_image, source_face_index, target_face_index,
                         restore_model, gender_match, auto_resolution, aggressive_cleanup,
                         # V5 identity guidance  
                         enable_ipadapter, ipadapter_weight, texture_preservation,
                         lighting_match, skin_texture_strength,
                         # V5 advanced settings
                         adaptive_restoration, frequency_blending, restoration_strength_limit):
        """
        Enhanced postprocessing with ReActor V5 features.
        """
        if not enabled:
            return
        
        if source_image is None:
            return
        
        try:
            # Get extension path and use shared WebUI models
            script_dir = os.path.dirname(os.path.abspath(__file__))
            extension_dir = os.path.dirname(script_dir)
            extensions_dir = os.path.dirname(extension_dir)
            webui_dir = os.path.dirname(extensions_dir)
            models_path = os.path.join(webui_dir, 'models')
            
            engine = get_reactor_v5_engine(models_path)
            
            # Set cleanup mode
            engine.set_cleanup_mode(aggressive_cleanup)
            
            # Configure V5 features
            v5_config = {
                'enable_ipadapter': enable_ipadapter,
                'ipadapter_weight': ipadapter_weight,
                'texture_preservation': texture_preservation,
                'lighting_match': lighting_match,
                'texture_strength': skin_texture_strength,
                'adaptive_restoration': adaptive_restoration,
                'frequency_blending': frequency_blending,
                'restoration_strength_limit': restoration_strength_limit
            }
            
            # Load restoration model if specified
            if restore_model and restore_model != 'None':
                if not engine.load_restorer(restore_model):
                    print(f"[ReActor V5] Warning: Failed to load restoration model {restore_model}")
                    restore_model = None
            
            # Convert images
            source_cv2 = pil_to_cv2(source_image)
            target_cv2 = pil_to_cv2(pp.image)
            
            if source_cv2 is None or target_cv2 is None:
                return
            
            print(f"[ReActor V5] Processing image with V5 features...")
            print(f"[ReActor V5] IP-Adapter enabled: {enable_ipadapter}")
            
            # Process with ReActor V5 enhanced pipeline
            result_cv2, status = engine.process(
                source_img=source_cv2,
                target_img=target_cv2,
                source_face_index=int(source_face_index),
                target_face_index=int(target_face_index),
                restore_model=restore_model,
                gender_match=gender_match,
                v5_config=v5_config
            )
            
            # Replace the image in-place
            result_pil = cv2_to_pil(result_cv2)
            if result_pil is not None:
                pp.image = result_pil
                print(f"[ReActor V5] {status}")
            
        except Exception as e:
            print(f"[ReActor V5] Error in postprocess_image: {e}")
            import traceback
            traceback.print_exc()


print("[ReActor V5] Enhanced script loaded successfully")
print("[ReActor V5] Features: IP-Adapter FaceID Plus v2, VRAM management, advanced realism")
print("[ReActor V5] Model locations:")
print("[ReActor V5]   GPEN: extensions/sd-webui-reactor-v5/models/facerestore_models/")
print("[ReActor V5]   IP-Adapter: models/ipadapter/ (auto-downloaded)")
print("[ReActor V5]   InsightFace: models/insightface/")