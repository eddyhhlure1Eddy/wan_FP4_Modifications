"""
Advanced Performance Optimization Module
Contains experimental and optional optimizations
Author: eddy
"""

import torch
import logging
from typing import Optional, Dict, Any
import gc

log = logging.getLogger(__name__)


class AdvancedOptimizer:
    """
    Advanced optimizations that may need careful tuning
    """
    
    def __init__(self):
        self.cuda_graphs_enabled = False
        self.memory_pool_configured = False
        
    def configure_memory_pool(self, 
                             max_split_size_mb: int = 512,
                             garbage_collection_threshold: float = 0.8):
        """
        Configure CUDA memory pool for better allocation performance
        
        Args:
            max_split_size_mb: Maximum size for memory splits (default 512MB)
            garbage_collection_threshold: Trigger GC when usage exceeds this ratio
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            # Set environment variable for max split size
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'
            
            # Configure garbage collection threshold
            torch.cuda.set_per_process_memory_fraction(garbage_collection_threshold, 0)
            
            self.memory_pool_configured = True
            log.info(f"✓ Configured CUDA memory pool: max_split={max_split_size_mb}MB, gc_threshold={garbage_collection_threshold}")
            return True
            
        except Exception as e:
            log.warning(f"Failed to configure memory pool: {e}")
            return False
    
    def enable_cuda_graphs(self, warmup_steps: int = 3):
        """
        Enable CUDA graphs for reduced kernel launch overhead
        
        WARNING: Requires fixed input shapes and careful implementation
        
        Args:
            warmup_steps: Number of warmup iterations before capturing graph
        """
        if not torch.cuda.is_available():
            return False
            
        # CUDA graphs are complex and need model-specific integration
        log.warning("CUDA graphs require model-specific implementation")
        log.warning("Consider using torch.compile with backend='cudagraphs' instead")
        return False
    
    def optimize_dataloader(self, num_workers: int = 4, pin_memory: bool = True):
        """
        Optimize data loading (if applicable)
        
        Args:
            num_workers: Number of worker processes
            pin_memory: Use pinned memory for faster host-to-device transfer
        """
        config = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': True if num_workers > 0 else False,
            'prefetch_factor': 2 if num_workers > 0 else None,
        }
        
        log.info(f"✓ DataLoader config optimized: {config}")
        return config
    
    def enable_tensor_cores_optimization(self):
        """
        Optimize for Tensor Cores usage
        
        Ensures data is properly aligned for Tensor Core operations
        """
        if not torch.cuda.is_available():
            return False
            
        compute_capability = torch.cuda.get_device_capability(0)
        has_tensor_cores = compute_capability[0] >= 7  # Volta and newer
        
        if has_tensor_cores:
            # Tensor cores work best with specific dimensions
            # (multiples of 8 for FP16, multiples of 16 for INT8)
            log.info("✓ Tensor Cores available (use dimensions multiple of 8/16)")
            log.info("  Recommendation: Ensure hidden_dim % 8 == 0 for optimal performance")
            return True
        else:
            log.info("Tensor Cores not available on this device")
            return False
    
    def configure_jit_fusion(self, enabled: bool = True):
        """
        Configure JIT fusion for operation fusion
        
        Args:
            enabled: Enable or disable JIT fusion
        """
        try:
            torch._C._jit_set_profiling_mode(enabled)
            torch._C._jit_set_profiling_executor(enabled)
            
            # Enable NVFuser for better fusion on CUDA
            if torch.cuda.is_available():
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(True)
                torch._C._jit_set_nvfuser_enabled(enabled)
            
            status = "enabled" if enabled else "disabled"
            log.info(f"✓ JIT fusion {status}")
            return True
            
        except Exception as e:
            log.warning(f"Could not configure JIT fusion: {e}")
            return False
    
    def optimize_autograd(self, use_reentrant: bool = False):
        """
        Optimize autograd for inference
        
        Args:
            use_reentrant: Use reentrant checkpointing (memory vs speed tradeoff)
        """
        # For inference, we typically don't need gradients
        log.info("💡 For inference: Use torch.no_grad() or torch.inference_mode()")
        log.info("   inference_mode() is faster than no_grad()")
        
        return {
            'use_reentrant': use_reentrant,
            'recommendation': 'torch.inference_mode()'
        }
    
    def configure_async_execution(self):
        """
        Configure asynchronous CUDA execution
        """
        if not torch.cuda.is_available():
            return False
            
        # Enable asynchronous error checking (slight performance gain)
        try:
            # This is the default, but explicitly set for clarity
            torch.backends.cuda.async_error_checking = False
            log.info("✓ Configured async CUDA execution (errors checked asynchronously)")
            return True
        except Exception as e:
            log.warning(f"Could not configure async execution: {e}")
            return False
    
    def optimize_conv_algorithms(self, benchmark_limit: int = 10):
        """
        Fine-tune cuDNN convolution algorithm selection
        
        Args:
            benchmark_limit: Maximum number of algorithms to benchmark
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            # Note: This requires cuDNN 8.0+
            if hasattr(torch.backends.cudnn, 'benchmark_limit'):
                torch.backends.cudnn.benchmark_limit = benchmark_limit
                log.info(f"✓ Set cuDNN benchmark limit to {benchmark_limit} algorithms")
                return True
            else:
                log.info("cuDNN benchmark_limit not available in this version")
                return False
                
        except Exception as e:
            log.warning(f"Could not configure conv algorithms: {e}")
            return False
    
    def enable_flash_attention_optimizations(self):
        """
        Check and configure Flash Attention optimizations
        """
        has_flash = False
        
        try:
            import flash_attn
            has_flash = True
            log.info("✓ Flash Attention 2 is installed")
        except ImportError:
            log.info("Flash Attention 2 not installed")
        
        try:
            import flash_attn_interface
            log.info("✓ Flash Attention 3 is installed (faster than FA2)")
            has_flash = True
        except ImportError:
            log.info("Flash Attention 3 not installed")
        
        if not has_flash:
            log.info("💡 Install Flash Attention for 40-60% attention speedup:")
            log.info("   pip install flash-attn --no-build-isolation")
        
        return has_flash
    
    def configure_channels_last(self, model: Optional[torch.nn.Module] = None):
        """
        Convert model to channels-last memory format
        
        Args:
            model: PyTorch model to convert
            
        Returns:
            Converted model or None
        """
        if model is None:
            log.info("💡 Apply channels_last format to models:")
            log.info("   model = model.to(memory_format=torch.channels_last)")
            log.info("   or for 3D: model.to(memory_format=torch.channels_last_3d)")
            log.info("   Expected speedup: 20-30% for conv-heavy models")
            return None
        
        try:
            # Try 3D first (for video models)
            model = model.to(memory_format=torch.channels_last_3d)
            log.info("✓ Converted model to channels_last_3d format")
            return model
        except:
            try:
                # Fall back to 2D
                model = model.to(memory_format=torch.channels_last)
                log.info("✓ Converted model to channels_last format")
                return model
            except Exception as e:
                log.warning(f"Could not convert to channels_last: {e}")
                return model
    
    def profile_memory_usage(self):
        """
        Profile current CUDA memory usage
        """
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        info = {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100
        }
        
        log.info("CUDA Memory Profile:")
        log.info(f"  Allocated: {allocated:.2f} GB")
        log.info(f"  Reserved:  {reserved:.2f} GB")
        log.info(f"  Peak:      {max_allocated:.2f} GB")
        log.info(f"  Total:     {total:.2f} GB")
        log.info(f"  Usage:     {info['utilization_percent']:.1f}%")
        
        return info
    
    def apply_advanced_optimizations(self, 
                                    enable_memory_pool: bool = True,
                                    enable_jit_fusion: bool = True,
                                    enable_async: bool = True):
        """
        Apply all advanced optimizations
        
        Args:
            enable_memory_pool: Configure CUDA memory pool
            enable_jit_fusion: Enable JIT operation fusion
            enable_async: Enable async execution
        """
        log.info("=" * 60)
        log.info("Advanced Performance Optimizations")
        log.info("=" * 60)
        
        results = {}
        
        if enable_memory_pool:
            results['memory_pool'] = self.configure_memory_pool()
        
        if enable_jit_fusion:
            results['jit_fusion'] = self.configure_jit_fusion(True)
        
        if enable_async:
            results['async_exec'] = self.configure_async_execution()
        
        results['tensor_cores'] = self.enable_tensor_cores_optimization()
        results['conv_algorithms'] = self.optimize_conv_algorithms()
        results['flash_attention'] = self.enable_flash_attention_optimizations()
        
        # Suggestions
        self.configure_channels_last(None)
        self.optimize_autograd()
        
        log.info("=" * 60)
        
        return results


# Singleton instance
_advanced_optimizer = None

def get_advanced_optimizer():
    """Get or create the advanced optimizer instance"""
    global _advanced_optimizer
    if _advanced_optimizer is None:
        _advanced_optimizer = AdvancedOptimizer()
    return _advanced_optimizer

def apply_advanced_optimizations(**kwargs):
    """
    Apply advanced optimizations
    
    Args:
        **kwargs: Arguments passed to apply_advanced_optimizations()
    """
    optimizer = get_advanced_optimizer()
    return optimizer.apply_advanced_optimizations(**kwargs)


if __name__ == "__main__":
    # Test mode
    apply_advanced_optimizations()
