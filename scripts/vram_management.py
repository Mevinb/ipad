"""
ReActor V5 - VRAM Management Utilities

Explicit VRAM detection, monitoring, and memory-safe execution rules.
"""

import torch
import gc
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import os
from typing import Dict, Tuple, Optional, List
import subprocess
import platform


class VRAMManager:
    """
    Handles VRAM detection, monitoring, and memory-safe execution for ReActor V5.
    
    Key principles:
    - Explicit detection and monitoring
    - Clear error messages instead of silent downgrades
    - Progressive model loading/unloading
    - Memory-safe execution rules
    """
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.safety_margin_gb = 1.0  # Reserve 1GB for safety
        
    def _detect_gpu(self) -> Dict[str, any]:
        """Detect GPU information and available VRAM"""
        gpu_info = {
            'name': 'Unknown GPU',
            'total_vram_gb': 0.0,
            'available': False,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            try:
                gpu_info['name'] = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory
                gpu_info['total_vram_gb'] = total_vram / (1024**3)
                gpu_info['available'] = True
            except Exception as e:
                print(f"[ReActor V5] Warning: Could not detect GPU details: {e}")
                
        return gpu_info
    
    def get_current_vram_usage(self) -> Tuple[float, float, float]:
        """
        Get current VRAM usage in GB.
        
        Returns:
            Tuple[allocated_gb, reserved_gb, free_gb]
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
            
        try:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = self.gpu_info['total_vram_gb']
            free = max(0.0, total - reserved)
            return allocated, reserved, free
        except Exception:
            return 0.0, 0.0, 0.0
    
    def estimate_ipadapter_vram_usage(self, 
                                     resolution: Tuple[int, int] = (512, 512),
                                     batch_size: int = 1) -> float:
        """
        Estimate VRAM usage for IP-Adapter FaceID Plus v2.
        
        Conservative estimates based on typical usage:
        - IP-Adapter model weights: ~1.2GB
        - CLIP Vision encoder: ~0.8GB  
        - Face embeddings: ~0.1GB
        - Temporary tensors: ~0.5GB per 512x512 image
        
        Args:
            resolution: Image resolution (width, height)
            batch_size: Number of images processed simultaneously
            
        Returns:
            Estimated VRAM usage in GB
        """
        base_usage = 2.1  # Model weights + CLIP Vision + face embeddings
        
        # Scale with resolution (quadratic scaling)
        w, h = resolution
        resolution_factor = (w * h) / (512 * 512)
        temp_usage = 0.5 * resolution_factor * batch_size
        
        return base_usage + temp_usage
    
    def can_run_with_ipadapter(self, 
                              resolution: Tuple[int, int] = (512, 512),
                              batch_size: int = 1) -> Tuple[bool, str]:
        """
        Check if current VRAM is sufficient for IP-Adapter operation.
        
        Returns:
            Tuple[can_run: bool, message: str]
        """
        if not self.gpu_info['available']:
            return False, "CUDA not available"
        
        _, _, free_vram = self.get_current_vram_usage()
        estimated_usage = self.estimate_ipadapter_vram_usage(resolution, batch_size)
        
        required_vram = estimated_usage + self.safety_margin_gb
        
        if free_vram < required_vram:
            return False, (
                f"Insufficient VRAM for IP-Adapter FaceID Plus v2. "
                f"Required: {required_vram:.1f}GB, Available: {free_vram:.1f}GB"
            )
        
        return True, f"VRAM check passed. Using {estimated_usage:.1f}GB of {free_vram:.1f}GB available"
    
    def enforce_memory_safe_settings(self, settings: Dict) -> Tuple[Dict, List[str]]:
        """
        Enforce memory-safe execution rules.
        
        Rules:
        - Force batch_size = 1 when IP-Adapter enabled
        - Prevent IP-Adapter + Hi-Res Fix
        - Prevent IP-Adapter + stacked ControlNet
        - Enforce FP16 execution
        
        Returns:
            Tuple[modified_settings, warnings]
        """
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
            
            # Rule 3: Check for conflicting ControlNet
            if settings.get('controlnet_enabled', False):
                if isinstance(settings.get('controlnet_units'), list) and len(settings['controlnet_units']) > 1:
                    warnings.append("Warning: Multiple ControlNet units may cause VRAM issues with IP-Adapter")
            
            # Rule 4: Enforce FP16 
            modified_settings['fp16_enabled'] = True
        
        return modified_settings, warnings
    
    def progressive_cleanup(self, level: str = 'standard'):
        """
        Progressive memory cleanup with different levels.
        
        Args:
            level: 'minimal', 'standard', 'aggressive'
        """
        print(f"[ReActor V5] Starting {level} memory cleanup...")
        
        if level in ['standard', 'aggressive']:
            # Standard cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if level == 'aggressive':
                    torch.cuda.ipc_collect()
        
        if level == 'aggressive':
            # Force garbage collection
            gc.collect()
            
            # Additional cleanup if available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                process.memory_full_info()  # Force memory stats update
            except ImportError:
                pass
        
        allocated, reserved, free = self.get_current_vram_usage()
        print(f"[ReActor V5] After cleanup - Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
    
    def get_status_report(self) -> Dict[str, any]:
        """Get comprehensive VRAM status report for UI display"""
        allocated, reserved, free = self.get_current_vram_usage()
        
        return {
            'gpu_name': self.gpu_info['name'],
            'total_vram_gb': self.gpu_info['total_vram_gb'],
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'cuda_available': self.gpu_info['cuda_available'],
            'can_use_ipadapter': self.can_run_with_ipadapter()[0]
        }


# Global VRAM manager instance
vram_manager = VRAMManager()


def get_vram_manager() -> VRAMManager:
    """Get the global VRAM manager instance"""
    return vram_manager


def check_vram_requirements(enable_ipadapter: bool = False,
                           resolution: Tuple[int, int] = (512, 512),
                           batch_size: int = 1) -> Tuple[bool, str]:
    """
    Convenience function to check VRAM requirements.
    
    Returns:
        Tuple[can_run: bool, message: str]
    """
    if not enable_ipadapter:
        return True, "ReActor V5 basic mode - VRAM check passed"
    
    return vram_manager.can_run_with_ipadapter(resolution, batch_size)


def format_vram_status() -> str:
    """Format VRAM status for UI display"""
    status = vram_manager.get_status_report()
    
    if not status['cuda_available']:
        return "CUDA not available"
    
    return (
        f"GPU: {status['gpu_name']} | "
        f"Total: {status['total_vram_gb']:.1f}GB | "
        f"Free: {status['free_gb']:.1f}GB | "
        f"IP-Adapter: {'✓' if status['can_use_ipadapter'] else '✗'}"
    )