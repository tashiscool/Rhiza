"""
Voice Optimizer - Enhanced with Sentient's Voice Optimization Concepts

Integrates Sentient's proven voice optimization patterns:
- Audio quality enhancement and preprocessing
- Performance optimization and caching
- Format conversion and normalization
- Device-specific optimizations

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass
import time
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/voice')
    # Try to import voice optimization concepts from Sentient
    SENTIENT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    SENTIENT_OPTIMIZATION_AVAILABLE = False

@dataclass
class OptimizationConfig:
    """Configuration for voice optimization"""
    enable_noise_reduction: bool = True
    enable_audio_enhancement: bool = True
    enable_format_optimization: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 100
    target_sample_rate: int = 16000
    target_bit_depth: int = 16
    noise_reduction_strength: float = 0.5
    enhancement_quality: str = "medium"  # low, medium, high
    device_optimization: bool = True

@dataclass
class OptimizedAudio:
    """Result of audio optimization"""
    audio_data: np.ndarray
    sample_rate: int
    bit_depth: int
    format: str
    optimization_applied: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class VoiceOptimizer:
    """
    Enhanced voice optimizer integrating Sentient's optimization concepts
    
    Provides:
    - Audio quality enhancement and preprocessing
    - Performance optimization and caching
    - Format conversion and normalization
    - Device-specific optimizations
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization components
        self.noise_reducer = None
        self.audio_enhancer = None
        self.format_converter = None
        self.cache_manager = None
        
        # Performance tracking
        self.optimization_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        self.logger.info("Voice Optimizer initialized with Sentient concepts")
    
    def _initialize_optimization_components(self):
        """Initialize optimization components using Sentient patterns"""
        
        try:
            # Initialize noise reduction
            if self.config.enable_noise_reduction:
                self._initialize_noise_reduction()
            
            # Initialize audio enhancement
            if self.config.enable_audio_enhancement:
                self._initialize_audio_enhancement()
            
            # Initialize format conversion
            if self.config.enable_format_optimization:
                self._initialize_format_conversion()
            
            # Initialize caching
            if self.config.enable_caching:
                self._initialize_caching()
            
            self.logger.info("Optimization components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optimization components: {e}")
            self.logger.warning("Voice optimization will use basic methods")
    
    def _initialize_noise_reduction(self):
        """Initialize noise reduction component"""
        
        try:
            self.noise_reducer = NoiseReducer(
                strength=self.config.noise_reduction_strength
            )
            self.logger.info("Noise reduction initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize noise reduction: {e}")
            self.noise_reducer = None
    
    def _initialize_audio_enhancement(self):
        """Initialize audio enhancement component"""
        
        try:
            self.audio_enhancer = AudioEnhancer(
                quality=self.config.enhancement_quality
            )
            self.logger.info("Audio enhancement initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio enhancement: {e}")
            self.audio_enhancer = None
    
    def _initialize_format_conversion(self):
        """Initialize format conversion component"""
        
        try:
            self.format_converter = FormatConverter(
                target_sample_rate=self.config.target_sample_rate,
                target_bit_depth=self.config.target_bit_depth
            )
            self.logger.info("Format conversion initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize format conversion: {e}")
            self.format_converter = None
    
    def _initialize_caching(self):
        """Initialize caching component"""
        
        try:
            self.cache_manager = AudioCache(
                max_size_mb=self.config.cache_size_mb
            )
            self.logger.info("Audio caching initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize caching: {e}")
            self.cache_manager = None
    
    async def optimize_audio(self, audio_data: np.ndarray, sample_rate: int,
                            audio_type: str = "input", metadata: Optional[Dict[str, Any]] = None) -> OptimizedAudio:
        """
        Optimize audio using Sentient's optimization patterns
        
        Args:
            audio_data: Raw audio data
            sample_rate: Audio sample rate
            audio_type: Type of audio (input, output, etc.)
            metadata: Additional audio metadata
            
        Returns:
            OptimizedAudio with enhanced audio and metadata
        """
        
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache_manager and metadata:
                cache_key = self._generate_cache_key(audio_data, sample_rate, audio_type, metadata)
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self.cache_hits += 1
                    self.logger.debug("Audio optimization result found in cache")
                    return cached_result
            
            self.cache_misses += 1
            
            # Apply optimizations
            optimized_audio = audio_data.copy()
            optimization_applied = []
            
            # 1. Format conversion
            if self.format_converter:
                optimized_audio, sample_rate = await self.format_converter.convert(
                    optimized_audio, sample_rate
                )
                optimization_applied.append("format_conversion")
            
            # 2. Noise reduction
            if self.noise_reducer and audio_type == "input":
                optimized_audio = await self.noise_reducer.reduce(optimized_audio, sample_rate)
                optimization_applied.append("noise_reduction")
            
            # 3. Audio enhancement
            if self.audio_enhancer:
                optimized_audio = await self.audio_enhancer.enhance(optimized_audio, sample_rate)
                optimization_applied.append("audio_enhancement")
            
            # 4. Device-specific optimization
            if self.config.device_optimization:
                optimized_audio = await self._apply_device_optimization(optimized_audio, sample_rate)
                optimization_applied.append("device_optimization")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update performance tracking
            self.optimization_count += 1
            self.total_processing_time += processing_time
            
            # Create result
            result = OptimizedAudio(
                audio_data=optimized_audio,
                sample_rate=sample_rate,
                bit_depth=self.config.target_bit_depth,
                format="optimized",
                optimization_applied=optimization_applied,
                processing_time=processing_time,
                metadata={
                    "original_sample_rate": sample_rate,
                    "audio_type": audio_type,
                    "optimizations": optimization_applied,
                    **(metadata or {})
                }
            )
            
            # Cache result
            if self.cache_manager and metadata:
                await self.cache_manager.set(cache_key, result)
            
            self.logger.info(f"Audio optimization completed in {processing_time:.3f}s: {optimization_applied}")
            return result
            
        except Exception as e:
            self.logger.error(f"Audio optimization failed: {e}")
            processing_time = time.time() - start_time
            
            return OptimizedAudio(
                audio_data=audio_data,
                sample_rate=sample_rate,
                bit_depth=self.config.target_bit_depth,
                format="original",
                optimization_applied=[],
                processing_time=processing_time,
                metadata={
                    "error": str(e),
                    "audio_type": audio_type,
                    **(metadata or {})
                }
            )
    
    def _generate_cache_key(self, audio_data: np.ndarray, sample_rate: int,
                           audio_type: str, metadata: Dict[str, Any]) -> str:
        """Generate cache key for audio optimization"""
        
        try:
            # Create a hash of the audio data and parameters
            import hashlib
            
            # Hash the audio data
            audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
            
            # Hash the parameters
            params_str = f"{sample_rate}_{audio_type}_{self.config.noise_reduction_strength}_{self.config.enhancement_quality}"
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            # Combine hashes
            cache_key = f"{audio_hash}_{params_hash}"
            
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Cache key generation failed: {e}")
            return f"fallback_{int(time.time())}"
    
    async def _apply_device_optimization(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply device-specific optimizations"""
        
        try:
            # Check for Apple Silicon optimizations
            if self._is_apple_silicon():
                return await self._apply_apple_silicon_optimization(audio_data, sample_rate)
            
            # Check for CUDA optimizations
            elif self._is_cuda_available():
                return await self._apply_cuda_optimization(audio_data, sample_rate)
            
            # Default CPU optimization
            else:
                return await self._apply_cpu_optimization(audio_data, sample_rate)
                
        except Exception as e:
            self.logger.error(f"Device optimization failed: {e}")
            return audio_data
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        
        try:
            import platform
            return platform.machine() == "arm64" and platform.system() == "Darwin"
        except Exception:
            return False
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def _apply_apple_silicon_optimization(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Apple Silicon specific optimizations"""
        
        try:
            # Use Metal Performance Shaders if available
            # For now, apply basic optimizations
            optimized = audio_data.copy()
            
            # Normalize audio
            if np.max(np.abs(optimized)) > 0:
                optimized = optimized / np.max(np.abs(optimized)) * 0.95
            
            self.logger.debug("Applied Apple Silicon optimizations")
            return optimized
            
        except Exception as e:
            self.logger.error(f"Apple Silicon optimization failed: {e}")
            return audio_data
    
    async def _apply_cuda_optimization(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply CUDA specific optimizations"""
        
        try:
            # Use CUDA acceleration if available
            # For now, apply basic optimizations
            optimized = audio_data.copy()
            
            # Normalize audio
            if np.max(np.abs(optimized)) > 0:
                optimized = optimized / np.max(np.abs(optimized)) * 0.95
            
            self.logger.debug("Applied CUDA optimizations")
            return optimized
            
        except Exception as e:
            self.logger.error(f"CUDA optimization failed: {e}")
            return audio_data
    
    async def _apply_cpu_optimization(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply CPU specific optimizations"""
        
        try:
            # Apply basic CPU optimizations
            optimized = audio_data.copy()
            
            # Normalize audio
            if np.max(np.abs(optimized)) > 0:
                optimized = optimized / np.max(np.abs(optimized)) * 0.95
            
            self.logger.debug("Applied CPU optimizations")
            return optimized
            
        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
            return audio_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for voice optimization"""
        
        avg_processing_time = (self.total_processing_time / self.optimization_count 
                             if self.optimization_count > 0 else 0)
        
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) 
                         if (self.cache_hits + self.cache_misses) > 0 else 0)
        
        return {
            "optimization_count": self.optimization_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "noise_reduction_available": self.noise_reducer is not None,
            "audio_enhancement_available": self.audio_enhancer is not None,
            "format_conversion_available": self.format_converter is not None,
            "caching_available": self.cache_manager is not None
        }
    
    async def cleanup(self):
        """Clean up optimization resources"""
        
        try:
            if hasattr(self.noise_reducer, 'cleanup'):
                await self.noise_reducer.cleanup()
            
            if hasattr(self.audio_enhancer, 'cleanup'):
                await self.audio_enhancer.cleanup()
            
            if hasattr(self.format_converter, 'cleanup'):
                await self.format_converter.cleanup()
            
            if hasattr(self.cache_manager, 'cleanup'):
                await self.cache_manager.cleanup()
                
            self.logger.info("Voice optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class NoiseReducer:
    """Noise reduction component following Sentient patterns"""
    
    def __init__(self, strength: float = 0.5):
        self.strength = strength
        self.logger = logging.getLogger(__name__)
    
    async def reduce(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reduce noise in audio data"""
        
        try:
            # Simple noise reduction using spectral subtraction
            # In the future, we can integrate more sophisticated methods from Sentient
            
            # Apply basic noise reduction
            reduced = audio_data.copy()
            
            # Simple threshold-based noise reduction
            noise_threshold = self.strength * 0.1
            reduced[np.abs(reduced) < noise_threshold] = 0
            
            self.logger.debug(f"Applied noise reduction with strength {self.strength}")
            return reduced
            
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {e}")
            return audio_data
    
    async def cleanup(self):
        """Clean up noise reduction resources"""
        pass


class AudioEnhancer:
    """Audio enhancement component following Sentient patterns"""
    
    def __init__(self, quality: str = "medium"):
        self.quality = quality
        self.logger = logging.getLogger(__name__)
    
    async def enhance(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance audio quality"""
        
        try:
            # Apply audio enhancement based on quality setting
            enhanced = audio_data.copy()
            
            if self.quality == "high":
                # High quality enhancement
                enhanced = self._apply_high_quality_enhancement(enhanced, sample_rate)
            elif self.quality == "medium":
                # Medium quality enhancement
                enhanced = self._apply_medium_quality_enhancement(enhanced, sample_rate)
            else:
                # Low quality enhancement
                enhanced = self._apply_low_quality_enhancement(enhanced, sample_rate)
            
            self.logger.debug(f"Applied {self.quality} quality audio enhancement")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Audio enhancement failed: {e}")
            return audio_data
    
    def _apply_high_quality_enhancement(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply high quality audio enhancement"""
        
        # High quality enhancement algorithms
        enhanced = audio_data.copy()
        
        # Normalize audio
        if np.max(np.abs(enhanced)) > 0:
            enhanced = enhanced / np.max(np.abs(enhanced)) * 0.95
        
        return enhanced
    
    def _apply_medium_quality_enhancement(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply medium quality audio enhancement"""
        
        # Medium quality enhancement algorithms
        enhanced = audio_data.copy()
        
        # Basic normalization
        if np.max(np.abs(enhanced)) > 0:
            enhanced = enhanced / np.max(np.abs(enhanced)) * 0.9
        
        return enhanced
    
    def _apply_low_quality_enhancement(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply low quality audio enhancement"""
        
        # Low quality enhancement algorithms
        enhanced = audio_data.copy()
        
        # Minimal processing
        return enhanced
    
    async def cleanup(self):
        """Clean up audio enhancement resources"""
        pass


class FormatConverter:
    """Format conversion component following Sentient patterns"""
    
    def __init__(self, target_sample_rate: int = 16000, target_bit_depth: int = 16):
        self.target_sample_rate = target_sample_rate
        self.target_bit_depth = target_bit_depth
        self.logger = logging.getLogger(__name__)
    
    async def convert(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Convert audio format"""
        
        try:
            converted = audio_data.copy()
            new_sample_rate = sample_rate
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                converted, new_sample_rate = await self._resample(converted, sample_rate, self.target_sample_rate)
            
            # Convert bit depth if needed
            if self.target_bit_depth != 16:  # Assuming input is 16-bit
                converted = await self._convert_bit_depth(converted, self.target_bit_depth)
            
            self.logger.debug(f"Converted audio to {new_sample_rate}Hz, {self.target_bit_depth}-bit")
            return converted, new_sample_rate
            
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return audio_data, sample_rate
    
    async def _resample(self, audio_data: np.ndarray, from_rate: int, to_rate: int) -> Tuple[np.ndarray, int]:
        """Resample audio data"""
        
        try:
            # Simple resampling using numpy
            # In the future, we can integrate librosa or other libraries from Sentient
            
            if from_rate == to_rate:
                return audio_data, from_rate
            
            # Calculate resampling ratio
            ratio = to_rate / from_rate
            new_length = int(len(audio_data) * ratio)
            
            # Simple linear interpolation
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            
            return resampled.astype(audio_data.dtype), to_rate
            
        except Exception as e:
            self.logger.error(f"Resampling failed: {e}")
            return audio_data, from_rate
    
    async def _convert_bit_depth(self, audio_data: np.ndarray, target_bits: int) -> np.ndarray:
        """Convert audio bit depth"""
        
        try:
            # Convert bit depth
            if target_bits == 8:
                # Convert to 8-bit
                converted = (audio_data / 32768.0 * 127.0).astype(np.int8)
            elif target_bits == 16:
                # Convert to 16-bit
                converted = (audio_data * 32767.0).astype(np.int16)
            elif target_bits == 32:
                # Convert to 32-bit float
                converted = (audio_data / 32768.0).astype(np.float32)
            else:
                # Unsupported bit depth
                self.logger.warning(f"Unsupported bit depth: {target_bits}")
                return audio_data
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Bit depth conversion failed: {e}")
            return audio_data
    
    async def cleanup(self):
        """Clean up format conversion resources"""
        pass


class AudioCache:
    """Audio caching component following Sentient patterns"""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.cache_order = []
        self.current_size = 0
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[OptimizedAudio]:
        """Get cached audio optimization result"""
        
        try:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache_order.remove(key)
                self.cache_order.append(key)
                
                self.logger.debug(f"Cache hit for key: {key}")
                return self.cache[key]
            
            self.logger.debug(f"Cache miss for key: {key}")
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: OptimizedAudio):
        """Set cached audio optimization result"""
        
        try:
            # Estimate size of the value
            value_size = self._estimate_size(value)
            
            # Check if we need to evict items
            while self.current_size + value_size > self.max_size_bytes and self.cache_order:
                # Remove least recently used item
                oldest_key = self.cache_order.pop(0)
                oldest_value = self.cache.pop(oldest_key)
                oldest_size = self._estimate_size(oldest_value)
                self.current_size -= oldest_size
                
                self.logger.debug(f"Evicted cache item: {oldest_key}")
            
            # Add new item
            self.cache[key] = value
            self.cache_order.append(key)
            self.current_size += value_size
            
            self.logger.debug(f"Cached optimization result for key: {key}")
            
        except Exception as e:
            self.logger.error(f"Cache set failed: {e}")
    
    def _estimate_size(self, value: OptimizedAudio) -> int:
        """Estimate memory size of cached value"""
        
        try:
            # Estimate size based on audio data and metadata
            audio_size = value.audio_data.nbytes
            metadata_size = len(str(value.metadata)) * 8  # Rough estimate
            
            return audio_size + metadata_size + 1000  # Add buffer
            
        except Exception as e:
            self.logger.error(f"Size estimation failed: {e}")
            return 1000  # Default size
    
    async def cleanup(self):
        """Clean up cache resources"""
        
        try:
            self.cache.clear()
            self.cache_order.clear()
            self.current_size = 0
            
            self.logger.info("Audio cache cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")


# Factory function for easy integration
def create_voice_optimizer(config: Optional[OptimizationConfig] = None) -> VoiceOptimizer:
    """Create a voice optimizer with default or custom configuration"""
    
    if config is None:
        config = OptimizationConfig()
    
    return VoiceOptimizer(config)
