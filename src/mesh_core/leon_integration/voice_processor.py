"""
Voice Processor - Enhanced with Sentient's Local Voice Processing Concepts

Integrates Sentient's proven local voice processing patterns:
- Local STT using Faster Whisper
- Local TTS using Orpheus
- Voice Activity Detection using Silero VAD
- Audio optimization and format handling

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

import asyncio
import logging
import numpy as np
from typing import AsyncGenerator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/voice')
    from stt.faster_whisper import FasterWhisperSTT
    from tts.orpheus import OrpheusTTS, OrpheusTTSOptions
    SENTIENT_VOICE_AVAILABLE = True
except ImportError:
    SENTIENT_VOICE_AVAILABLE = False
    # Mock classes for development/testing
    class FasterWhisperSTT:
        def __init__(self, **kwargs):
            pass
        async def transcribe(self, *args, **kwargs):
            return "Mock transcription - Sentient voice not available"
    
    class OrpheusTTS:
        def __init__(self, **kwargs):
            pass
        async def stream_tts(self, *args, **kwargs):
            yield (16000, np.zeros(1024, dtype=np.float32))

@dataclass
class VoiceProcessingConfig:
    """Configuration for voice processing"""
    stt_model_size: str = "base"
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    stt_compute_type: str = "int8"
    tts_model_path: Optional[str] = None
    tts_n_gpu_layers: Optional[int] = None
    tts_default_voice: str = "tara"
    vad_threshold: float = 0.9
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 3000
    vad_speech_pad_ms: int = 800
    vad_max_speech_s: int = 15

@dataclass
class TranscriptionResult:
    """Result of speech-to-text processing"""
    text: str
    confidence: float
    language: str
    processing_time: float
    audio_metadata: Dict[str, Any]

@dataclass
class SynthesisResult:
    """Result of text-to-speech processing"""
    audio_data: np.ndarray
    sample_rate: int
    voice_used: str
    processing_time: float
    metadata: Dict[str, Any]

class LocalVoiceProcessor:
    """
    Enhanced voice processor integrating Sentient's local voice concepts
    
    Provides:
    - Local STT using Faster Whisper (no external APIs)
    - Local TTS using Orpheus (high-quality synthesis)
    - Voice activity detection and optimization
    - Audio format handling and preprocessing
    """
    
    def __init__(self, config: VoiceProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize voice components
        self.stt_model = None
        self.tts_model = None
        self.vad_model = None
        
        # Performance tracking
        self.transcription_count = 0
        self.synthesis_count = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_voice_components()
        
        self.logger.info("Local Voice Processor initialized with Sentient concepts")
    
    def _initialize_voice_components(self):
        """Initialize voice processing components using Sentient patterns"""
        
        try:
            # Initialize STT (Faster Whisper)
            self._initialize_stt()
            
            # Initialize TTS (Orpheus)
            self._initialize_tts()
            
            # Initialize VAD (Silero)
            self._initialize_vad()
            
            self.logger.info("All voice components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice components: {e}")
            self.logger.warning("Voice processing will use fallback methods")
    
    def _initialize_stt(self):
        """Initialize Speech-to-Text using Sentient's Faster Whisper pattern"""
        
        if not SENTIENT_VOICE_AVAILABLE:
            self.logger.warning("Sentient voice not available, using mock STT")
            return
        
        try:
            # Determine device automatically
            device = self._detect_optimal_device()
            
            self.stt_model = FasterWhisperSTT(
                model_size=self.config.stt_model_size,
                device=device,
                compute_type=self.config.stt_compute_type
            )
            
            self.logger.info(f"STT initialized with Faster Whisper on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize STT: {e}")
            self.stt_model = None
    
    def _initialize_tts(self):
        """Initialize Text-to-Speech using Sentient's Orpheus pattern"""
        
        if not SENTIENT_VOICE_AVAILABLE:
            self.logger.warning("Sentient voice not available, using mock TTS")
            return
        
        try:
            self.tts_model = OrpheusTTS(
                model_path=self.config.tts_model_path,
                n_gpu_layers=self.config.tts_n_gpu_layers,
                default_voice_id=self.config.tts_default_voice
            )
            
            self.logger.info(f"TTS initialized with Orpheus using voice {self.config.tts_default_voice}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            self.tts_model = None
    
    def _initialize_vad(self):
        """Initialize Voice Activity Detection using Sentient's Silero VAD pattern"""
        
        try:
            # For now, we'll use a simple energy-based VAD
            # In the future, we can integrate Silero VAD from Sentient
            self.vad_model = SimpleEnergyVAD(
                threshold=self.config.vad_threshold,
                min_speech_ms=self.config.vad_min_speech_ms,
                min_silence_ms=self.config.vad_min_silence_ms
            )
            
            self.logger.info("VAD initialized with energy-based detection")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}")
            self.vad_model = None
    
    def _detect_optimal_device(self) -> str:
        """Detect optimal device for voice processing (following Sentient pattern)"""
        
        if self.config.stt_device == "auto":
            # Check for CUDA first, then MPS, then CPU
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        else:
            return self.config.stt_device
    
    async def transcribe_audio(self, audio_data: bytes, sample_rate: int, 
                              metadata: Optional[Dict[str, Any]] = None) -> TranscriptionResult:
        """
        Transcribe audio using local STT (following Sentient's pattern)
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            metadata: Additional audio metadata
            
        Returns:
            TranscriptionResult with text and confidence
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.stt_model:
                raise Exception("STT model not initialized")
            
            # Use Sentient's transcription pattern
            transcription = await self.stt_model.transcribe(audio_data, sample_rate)
            
            # Calculate confidence (simplified for now)
            confidence = 0.9 if transcription.strip() else 0.0
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance tracking
            self.transcription_count += 1
            self.total_processing_time += processing_time
            
            result = TranscriptionResult(
                text=transcription,
                confidence=confidence,
                language="en",  # Default to English for now
                processing_time=processing_time,
                audio_metadata=metadata or {}
            )
            
            self.logger.info(f"Transcription completed in {processing_time:.3f}s: '{transcription[:50]}...'")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="en",
                processing_time=processing_time,
                audio_metadata=metadata or {}
            )
    
    async def synthesize_speech(self, text: str, voice: Optional[str] = None,
                               options: Optional[Dict[str, Any]] = None) -> SynthesisResult:
        """
        Synthesize speech using local TTS (following Sentient's pattern)
        
        Args:
            text: Text to synthesize
            voice: Voice to use (if None, uses default)
            options: Additional TTS options
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.tts_model:
                raise Exception("TTS model not initialized")
            
            # Prepare TTS options following Sentient's pattern
            tts_options = OrpheusTTSOptions(
                voice_id=voice or self.config.tts_default_voice,
                max_tokens=options.get("max_tokens", 4096) if options else 4096,
                temperature=options.get("temperature", 0.7) if options else 0.7,
                top_p=options.get("top_p", 0.9) if options else 0.9,
                repetition_penalty=options.get("repetition_penalty", 1.1) if options else 1.1
            )
            
            # Collect audio chunks from streaming TTS
            audio_chunks = []
            sample_rate = None
            
            async for chunk in self.tts_model.stream_tts(text, tts_options):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    chunk_sample_rate, chunk_audio = chunk
                    if sample_rate is None:
                        sample_rate = chunk_sample_rate
                    audio_chunks.append(chunk_audio)
                else:
                    # Handle other chunk formats if needed
                    continue
            
            if not audio_chunks:
                raise Exception("No audio generated from TTS")
            
            # Combine audio chunks
            combined_audio = np.concatenate(audio_chunks)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance tracking
            self.synthesis_count += 1
            self.total_processing_time += processing_time
            
            result = SynthesisResult(
                audio_data=combined_audio,
                sample_rate=sample_rate or 24000,  # Orpheus default
                voice_used=voice or self.config.tts_default_voice,
                processing_time=processing_time,
                metadata={
                    "text_length": len(text),
                    "audio_duration": len(combined_audio) / (sample_rate or 24000),
                    "chunks_generated": len(audio_chunks)
                }
            )
            
            self.logger.info(f"Speech synthesis completed in {processing_time:.3f}s for text: '{text[:50]}...'")
            return result
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return SynthesisResult(
                audio_data=np.zeros(1024, dtype=np.float32),
                sample_rate=24000,
                voice_used=voice or self.config.tts_default_voice,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def detect_voice_activity(self, audio_data: np.ndarray, 
                                   sample_rate: int) -> bool:
        """
        Detect voice activity in audio (following Sentient's VAD pattern)
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            True if voice activity detected, False otherwise
        """
        
        try:
            if self.vad_model:
                return await self.vad_model.detect_activity(audio_data, sample_rate)
            else:
                # Fallback to simple energy-based detection
                return self._simple_energy_vad(audio_data, sample_rate)
                
        except Exception as e:
            self.logger.error(f"Voice activity detection failed: {e}")
            return False
    
    def _simple_energy_vad(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Simple energy-based voice activity detection as fallback"""
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Simple threshold-based detection
        threshold = 0.01  # Adjustable threshold
        return energy > threshold
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for voice processing"""
        
        avg_processing_time = (self.total_processing_time / 
                             (self.transcription_count + self.synthesis_count) 
                             if (self.transcription_count + self.synthesis_count) > 0 else 0)
        
        return {
            "transcription_count": self.transcription_count,
            "synthesis_count": self.synthesis_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "stt_available": self.stt_model is not None,
            "tts_available": self.tts_model is not None,
            "vad_available": self.vad_model is not None
        }
    
    async def cleanup(self):
        """Clean up voice processing resources"""
        
        try:
            # Clean up models if they have cleanup methods
            if hasattr(self.stt_model, 'cleanup'):
                await self.stt_model.cleanup()
            
            if hasattr(self.tts_model, 'cleanup'):
                await self.tts_model.cleanup()
            
            if hasattr(self.vad_model, 'cleanup'):
                await self.vad_model.cleanup()
                
            self.logger.info("Voice processor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class SimpleEnergyVAD:
    """Simple energy-based voice activity detection as fallback"""
    
    def __init__(self, threshold: float = 0.9, min_speech_ms: int = 250, 
                 min_silence_ms: int = 3000):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.logger = logging.getLogger(__name__)
    
    async def detect_activity(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Detect voice activity in audio data"""
        
        try:
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Apply threshold
            is_active = energy > self.threshold
            
            return is_active
            
        except Exception as e:
            self.logger.error(f"VAD detection failed: {e}")
            return False


# Factory function for easy integration
def create_voice_processor(config: Optional[VoiceProcessingConfig] = None) -> LocalVoiceProcessor:
    """Create a voice processor with default or custom configuration"""
    
    if config is None:
        config = VoiceProcessingConfig()
    
    return LocalVoiceProcessor(config)
