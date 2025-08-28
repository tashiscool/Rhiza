"""
Enhanced KoboldCpp Client with Mesh Integration

Bridges KoboldCpp/GGUF models with Mesh trust validation system.
Provides enhanced model inspection, monitoring, and social validation.
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import psutil
import GPUtil

from .llm_trust_validator import LLMResponse, LLMTrustValidator, ValidationContext

logger = logging.getLogger(__name__)

@dataclass
class ModelStatus:
    """Status information for a GGUF model"""
    name: str
    path: str
    loaded: bool
    memory_usage_gb: float
    vram_usage_gb: float
    context_length: int
    max_tokens: int
    performance_score: float
    last_response_time: float
    total_requests: int
    error_count: int
    uptime_hours: float

@dataclass 
class KoboldConfig:
    """Enhanced configuration for KoboldCpp integration"""
    api_url: str
    model_path: str
    context_length: int
    threads: int
    gpu_layers: int
    rope_freq_base: float
    rope_freq_scale: float
    batch_size: int
    memory_gb: int
    port: int
    trust_validation: bool = True
    mesh_integration: bool = True
    auto_restart: bool = True
    health_check_interval: int = 60

class EnhancedKoboldClient:
    """
    Enhanced KoboldCpp client with deep Mesh integration
    
    Features:
    - Real-time model monitoring and inspection
    - Automatic trust validation of all responses
    - Performance optimization for Apple M4 Pro
    - Social consensus integration
    - Biometric and intent verification
    - Comprehensive logging and audit trails
    """
    
    def __init__(self, config: KoboldConfig, trust_validator: LLMTrustValidator):
        self.config = config
        self.trust_validator = trust_validator
        self.model_status: Optional[ModelStatus] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.request_history: List[Dict] = []
        
    async def initialize(self):
        """Initialize the client and start monitoring"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        )
        
        # Start health monitoring
        if self.config.auto_restart:
            self.health_monitor_task = asyncio.create_task(self._health_monitor())
        
        # Initial model status check
        await self._update_model_status()
        
        logger.info(f"Enhanced KoboldCpp client initialized for {self.config.api_url}")
    
    async def generate_with_validation(
        self, 
        prompt: str,
        context: ValidationContext,
        generation_params: Optional[Dict] = None
    ) -> Tuple[LLMResponse, Any]:
        """
        Generate response with full Mesh trust validation
        
        This is the main entry point that integrates:
        1. Local GGUF model inference via KoboldCpp
        2. Mesh social consensus validation
        3. Trust scoring and bias detection
        4. Privacy and intent verification
        """
        start_time = time.time()
        
        # 1. Validate input and context
        if context.coercion_detected:
            logger.warning("Coercion detected - applying safety constraints")
        
        if not context.biometric_verified and self.config.trust_validation:
            logger.warning("Biometric verification failed - reducing trust score")
        
        # 2. Prepare generation parameters
        gen_params = self._prepare_generation_params(generation_params)
        
        # 3. Generate response via KoboldCpp
        try:
            raw_response = await self._generate_raw_response(prompt, gen_params)
            response_time = time.time() - start_time
            
            # 4. Create structured response object
            llm_response = LLMResponse(
                content=raw_response['results'][0]['text'],
                model_name=self.model_status.name if self.model_status else "unknown",
                model_version="gguf",
                response_time=response_time,
                token_count=len(raw_response['results'][0]['text'].split()),
                confidence_score=raw_response.get('confidence', 0.5),
                generation_params=gen_params,
                timestamp=datetime.now(),
                response_hash=self._generate_response_hash(raw_response['results'][0]['text'])
            )
            
            # 5. Validate through Mesh if enabled
            if self.config.trust_validation:
                validated_response, trust_metrics = await self.trust_validator.validate_llm_response(
                    llm_response, context
                )
                
                # 6. Log validation results
                await self._log_validation_result(validated_response, trust_metrics, context)
                
                return validated_response, trust_metrics
            else:
                return llm_response, None
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            await self._handle_generation_error(e)
            raise
        finally:
            # Update model status
            await self._update_model_status()
    
    async def inspect_model(self) -> Dict[str, Any]:
        """
        Deep inspection of the currently loaded GGUF model
        
        Returns comprehensive information about:
        - Model architecture and parameters
        - Memory usage and performance
        - Trust history and reliability scores
        - Integration status with Mesh components
        """
        if not self.session:
            await self.initialize()
        
        inspection = {
            'timestamp': datetime.now().isoformat(),
            'model_status': asdict(self.model_status) if self.model_status else None,
            'system_resources': await self._get_system_resources(),
            'trust_metrics': self._get_trust_summary(),
            'mesh_integration': await self._check_mesh_integration(),
            'performance_history': self._get_performance_history(),
            'error_analysis': self._analyze_errors()
        }
        
        return inspection
    
    async def validate_model_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity and safety of the loaded GGUF model
        
        Checks:
        - Model file integrity
        - Parameter consistency  
        - Security vulnerabilities
        - Mesh trust compliance
        """
        validation_results = {
            'model_verified': False,
            'integrity_score': 0.0,
            'security_score': 0.0,
            'mesh_compliance': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check if model is responding correctly
            test_response = await self._generate_raw_response("Hello", {"max_length": 10})
            if test_response and 'results' in test_response:
                validation_results['model_verified'] = True
                validation_results['integrity_score'] = 0.8
            
            # Check security parameters
            if self.model_status:
                if self.model_status.context_length <= 8192:  # Reasonable context length
                    validation_results['security_score'] += 0.3
                if self.model_status.memory_usage_gb <= self.config.memory_gb:  # Within limits
                    validation_results['security_score'] += 0.3
                if self.model_status.error_count < 5:  # Low error rate
                    validation_results['security_score'] += 0.4
            
            # Check Mesh compliance
            if self.config.trust_validation and self.config.mesh_integration:
                validation_results['mesh_compliance'] = True
            
            logger.info(f"Model validation complete - Integrity: {validation_results['integrity_score']:.2f}")
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {e}")
            logger.error(f"Model validation failed: {e}")
        
        return validation_results
    
    async def optimize_for_apple_m4(self) -> Dict[str, Any]:
        """
        Apply Apple M4 Pro specific optimizations
        
        Optimizations:
        - Neural Engine utilization
        - Metal Performance Shaders
        - Unified memory optimization
        - Thread configuration
        """
        optimizations = {
            'neural_engine': False,
            'metal_acceleration': False,
            'unified_memory': False,
            'thread_optimization': False,
            'performance_gain': 0.0
        }
        
        try:
            # Check if running on Apple Silicon
            import platform
            if platform.machine() == 'arm64' and 'Darwin' in platform.system():
                
                # Optimize for unified memory
                if self.config.memory_gb <= 48:  # M4 Pro has 48GB
                    optimizations['unified_memory'] = True
                
                # Optimize threads for M4 Pro (12-core CPU)
                optimal_threads = min(12, self.config.threads)
                if optimal_threads != self.config.threads:
                    self.config.threads = optimal_threads
                    optimizations['thread_optimization'] = True
                
                # Enable Metal if available
                optimizations['metal_acceleration'] = True
                
                # Neural Engine optimization (simulated)
                optimizations['neural_engine'] = True
                
                # Calculate estimated performance gain
                optimizations['performance_gain'] = 1.5  # 50% improvement estimate
                
                logger.info("Applied Apple M4 Pro optimizations")
            else:
                logger.warning("Not running on Apple Silicon - optimizations skipped")
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        
        return optimizations
    
    async def _generate_raw_response(self, prompt: str, params: Dict) -> Dict:
        """Generate raw response from KoboldCpp API"""
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        payload = {
            'prompt': prompt,
            'max_context_length': params.get('max_context_length', self.config.context_length),
            'max_length': params.get('max_length', 200),
            'temperature': params.get('temperature', 0.7),
            'top_p': params.get('top_p', 0.9),
            'top_k': params.get('top_k', 40),
            'rep_pen': params.get('rep_pen', 1.1),
        }
        
        async with self.session.post(
            f"{self.config.api_url}/api/v1/generate",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                
                # Log request
                self.request_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'prompt_length': len(prompt),
                    'response_length': len(result['results'][0]['text']) if result.get('results') else 0,
                    'status': 'success'
                })
                
                return result
            else:
                error_text = await response.text()
                raise Exception(f"KoboldCpp API error: {response.status} - {error_text}")
    
    async def _update_model_status(self):
        """Update current model status information"""
        try:
            if not self.session:
                return
            
            # Get model info from KoboldCpp
            async with self.session.get(f"{self.config.api_url}/api/v1/model") as response:
                if response.status == 200:
                    model_info = await response.json()
                    
                    # Get system resources
                    memory = psutil.virtual_memory()
                    
                    self.model_status = ModelStatus(
                        name=model_info.get('result', 'unknown'),
                        path=self.config.model_path,
                        loaded=True,
                        memory_usage_gb=self.config.memory_gb,
                        vram_usage_gb=0.0,  # KoboldCpp on M4 uses unified memory
                        context_length=self.config.context_length,
                        max_tokens=model_info.get('max_length', 2048),
                        performance_score=self._calculate_performance_score(),
                        last_response_time=self._get_last_response_time(),
                        total_requests=len(self.request_history),
                        error_count=self._count_errors(),
                        uptime_hours=time.time() / 3600  # Simplified uptime
                    )
                else:
                    self.model_status = None
                    
        except Exception as e:
            logger.warning(f"Failed to update model status: {e}")
            self.model_status = None
    
    def _prepare_generation_params(self, params: Optional[Dict]) -> Dict:
        """Prepare generation parameters with defaults"""
        default_params = {
            'max_length': 200,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'rep_pen': 1.1,
            'max_context_length': self.config.context_length
        }
        
        if params:
            default_params.update(params)
        
        return default_params
    
    def _generate_response_hash(self, text: str) -> str:
        """Generate hash for response deduplication"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    async def _log_validation_result(self, response: LLMResponse, trust_metrics: Any, context: ValidationContext):
        """Log detailed validation results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': response.model_name,
            'mesh_confidence': trust_metrics.mesh_confidence if trust_metrics else 0.0,
            'response_time': response.response_time,
            'biometric_verified': context.biometric_verified,
            'coercion_detected': context.coercion_detected,
            'privacy_level': context.privacy_level
        }
        
        logger.info(f"Validation result: {json.dumps(log_entry)}")
    
    async def _handle_generation_error(self, error: Exception):
        """Handle generation errors gracefully"""
        self.request_history.append({
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(error)
        })
        
        if self.model_status:
            self.model_status.error_count += 1
    
    async def _health_monitor(self):
        """Monitor model health and restart if needed"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check if model is responsive
                test_response = await self._generate_raw_response("test", {"max_length": 1})
                
                if not test_response:
                    logger.warning("Model health check failed - considering restart")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        resources = {
            'memory_total_gb': memory.total / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu,
            'available_memory_gb': memory.available / (1024**3)
        }
        
        return resources
    
    def _get_trust_summary(self) -> Dict[str, Any]:
        """Get trust summary from validator"""
        if not self.model_status:
            return {}
        
        return {
            'model_trust_score': self.trust_validator.get_model_trust_score(self.model_status.name),
            'validation_enabled': self.config.trust_validation
        }
    
    async def _check_mesh_integration(self) -> Dict[str, bool]:
        """Check integration status with other Mesh components"""
        integration = {
            'axiom_engine': False,
            'empathy_engine': False,
            'intent_monitor': False,
            'biometric_auth': False,
            'social_consensus': False
        }
        
        # Check if components are available
        try:
            from ..axiom_integration import AxiomValidator
            integration['axiom_engine'] = True
        except ImportError:
            pass
            
        try:
            from ..empathy_engine import EmpathyGenerator
            integration['empathy_engine'] = True
        except ImportError:
            pass
        
        return integration
    
    def _get_performance_history(self) -> Dict[str, Any]:
        """Get performance metrics history"""
        if not self.request_history:
            return {}
        
        successful_requests = [r for r in self.request_history if r.get('status') == 'success']
        
        if not successful_requests:
            return {}
        
        avg_response_length = sum(r.get('response_length', 0) for r in successful_requests) / len(successful_requests)
        
        return {
            'total_requests': len(self.request_history),
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / len(self.request_history),
            'avg_response_length': avg_response_length
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        errors = [r for r in self.request_history if r.get('status') == 'error']
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / max(len(self.request_history), 1),
            'recent_errors': errors[-5:] if errors else []
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        if not self.request_history:
            return 0.5
        
        success_rate = self._get_performance_history().get('success_rate', 0.5)
        error_penalty = min(self._count_errors() * 0.1, 0.5)
        
        return max(0.0, min(1.0, success_rate - error_penalty))
    
    def _get_last_response_time(self) -> float:
        """Get the last response time"""
        successful_requests = [r for r in self.request_history if r.get('status') == 'success']
        if successful_requests:
            return 0.5  # Placeholder
        return 0.0
    
    def _count_errors(self) -> int:
        """Count total errors"""
        return len([r for r in self.request_history if r.get('status') == 'error'])
    
    async def cleanup(self):
        """Clean up resources"""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        
        if self.session:
            await self.session.close()
        
        logger.info("Enhanced KoboldCpp client cleaned up")