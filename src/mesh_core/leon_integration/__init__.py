"""
Leon Integration Module - Enhanced with Sentient Voice Concepts

This module integrates Sentient's proven local voice processing patterns into The Mesh:
- Local STT using Faster Whisper (no external APIs)
- Local TTS using Orpheus (high-quality synthesis)
- Voice activity detection and optimization
- Audio format handling and preprocessing

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

# Import voice processing components
from .voice_processor import (
    LocalVoiceProcessor,
    VoiceProcessingConfig,
    TranscriptionResult,
    SynthesisResult,
    create_voice_processor
)

from .voice_activity import (
    VoiceActivityDetector,
    VADConfig,
    VADResult,
    create_vad_detector
)

from .voice_optimizer import (
    VoiceOptimizer,
    OptimizationConfig,
    OptimizedAudio,
    create_voice_optimizer
)

# Export all voice processing capabilities
__all__ = [
    # Voice Processor
    'LocalVoiceProcessor',
    'VoiceProcessingConfig', 
    'TranscriptionResult',
    'SynthesisResult',
    'create_voice_processor',
    
    # Voice Activity Detection
    'VoiceActivityDetector',
    'VADConfig',
    'VADResult',
    'create_vad_detector',
    
    # Voice Optimization
    'VoiceOptimizer',
    'OptimizationConfig',
    'OptimizedAudio',
    'create_voice_optimizer'
]

# Version information
__version__ = "2.0.0"
__description__ = "Enhanced voice processing with Sentient concepts"
__author__ = "The Mesh Development Team"
