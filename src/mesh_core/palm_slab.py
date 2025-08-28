"""
Palm Slab Interface - Biometric authentication and intention signature verification
Physical device integration for the Mesh's hardware layer
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging


class AuthenticationResult(Enum):
    SUCCESS = "success"
    PALM_MISMATCH = "palm_mismatch"
    INTENTION_DIVERGENCE = "intention_divergence"
    COERCION_DETECTED = "coercion_detected"
    SYSTEM_ERROR = "system_error"


@dataclass
class BiometricData:
    """Biometric data from Palm Slab"""
    palm_hash: str
    intention_signature: str
    pressure_pattern: Optional[List[float]] = None
    temporal_pattern: Optional[List[float]] = None
    confidence: float = 0.0


@dataclass
class AuthResult:
    """Result of authentication attempt"""
    result: AuthenticationResult
    confidence: float
    user_id: Optional[str] = None
    reason: Optional[str] = None
    coercion_indicators: Optional[List[str]] = None
    baseline_deviation: float = 0.0


class PalmSlabInterface:
    """
    Palm Slab Interface for biometric authentication and coercion detection
    
    Each Palm Slab includes:
    - Tensor crystal compute layer for local AI reasoning
    - Privacy ring to shield thought trails unless explicitly shared
    - Biometric tethering (palmprint + intention signature)
    - Soft neural mesh antenna for multi-spectrum peer relaying
    - Solar-microcell recharge film
    
    Core Security Features:
    - Palmprint verification
    - Intention signature analysis
    - Coercion detection through biometric anomalies
    - Local-first processing (no cloud dependencies)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Authentication parameters
        self.palm_match_threshold = config.get('palm_match_threshold', 0.85)
        self.intention_match_threshold = config.get('intention_match_threshold', 0.75)
        self.coercion_sensitivity = config.get('coercion_sensitivity', 0.7)
        
        # Security settings
        self.require_biometric = config.get('biometric_required', True)
        self.require_intention = config.get('intention_required', True)
        self.coercion_detection = config.get('coercion_detection', True)
        
        # Local storage for biometric baselines (encrypted)
        self.storage_path = config.get('storage_path', '/tmp/mesh_biometrics')
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Initialize storage
        self._initialize_secure_storage()
        
        # Load existing baselines
        self._load_user_baselines()
        
        self.logger.info("Palm Slab Interface initialized - Biometric security active")
    
    def _initialize_secure_storage(self):
        """Initialize secure local storage for biometric data"""
        
        try:
            os.makedirs(self.storage_path, mode=0o700, exist_ok=True)
            
            # Create encryption key file if it doesn't exist
            key_file = os.path.join(self.storage_path, '.encryption_key')
            if not os.path.exists(key_file):
                # Generate random encryption key
                import secrets
                encryption_key = secrets.token_bytes(32)
                
                with open(key_file, 'wb') as f:
                    f.write(encryption_key)
                
                # Secure file permissions
                os.chmod(key_file, 0o600)
            
            self.logger.info("Secure storage initialized")
            
        except Exception as e:
            self.logger.error(f"Secure storage initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize secure storage: {e}")
    
    def _load_user_baselines(self):
        """Load existing user biometric baselines"""
        
        try:
            baselines_file = os.path.join(self.storage_path, 'user_baselines.json')
            
            if os.path.exists(baselines_file):
                with open(baselines_file, 'r') as f:
                    encrypted_data = f.read()
                
                # In production, this would be properly encrypted
                # For now, store as JSON for development
                self.user_baselines = json.loads(encrypted_data)
                
                self.logger.info(f"Loaded baselines for {len(self.user_baselines)} users")
            
        except Exception as e:
            self.logger.warning(f"Could not load user baselines: {e}")
            self.user_baselines = {}
    
    def _save_user_baselines(self):
        """Save user biometric baselines securely"""
        
        try:
            baselines_file = os.path.join(self.storage_path, 'user_baselines.json')
            
            # In production, this would be properly encrypted
            with open(baselines_file, 'w') as f:
                json.dump(self.user_baselines, f, indent=2)
            
            # Secure file permissions
            os.chmod(baselines_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save user baselines: {e}")
    
    async def authenticate(
        self,
        user_id: str,
        biometric_data: Optional[Dict[str, Any]]
    ) -> AuthResult:
        """
        Authenticate user using Palm Slab biometric data
        
        Performs multi-factor authentication:
        1. Palmprint verification
        2. Intention signature matching  
        3. Coercion detection through biometric anomalies
        4. Temporal pattern analysis
        """
        
        if not self.require_biometric:
            return AuthResult(
                result=AuthenticationResult.SUCCESS,
                confidence=0.5,
                user_id=user_id,
                reason="Biometric authentication disabled"
            )
        
        if not biometric_data:
            return AuthResult(
                result=AuthenticationResult.SYSTEM_ERROR,
                confidence=0.0,
                reason="No biometric data provided"
            )
        
        try:
            # Parse biometric data
            biometric = self._parse_biometric_data(biometric_data)
            
            # Get user baseline
            baseline = self._get_user_baseline(user_id)
            
            # Authenticate palmprint
            palm_result = await self._authenticate_palmprint(
                biometric.palm_hash,
                baseline.get('palm_baseline')
            )
            
            if palm_result.confidence < self.palm_match_threshold:
                return AuthResult(
                    result=AuthenticationResult.PALM_MISMATCH,
                    confidence=palm_result.confidence,
                    reason="Palmprint does not match baseline"
                )
            
            # Authenticate intention signature
            if self.require_intention:
                intention_result = await self._authenticate_intention(
                    biometric.intention_signature,
                    baseline.get('intention_baseline')
                )
                
                if intention_result.confidence < self.intention_match_threshold:
                    return AuthResult(
                        result=AuthenticationResult.INTENTION_DIVERGENCE,
                        confidence=intention_result.confidence,
                        reason="Intention signature anomaly detected",
                        baseline_deviation=intention_result.deviation
                    )
            
            # Check for coercion indicators
            if self.coercion_detection:
                coercion_result = await self._detect_coercion(
                    biometric,
                    baseline
                )
                
                if coercion_result.coercion_probability > self.coercion_sensitivity:
                    return AuthResult(
                        result=AuthenticationResult.COERCION_DETECTED,
                        confidence=coercion_result.confidence,
                        reason="Possible coercion detected",
                        coercion_indicators=coercion_result.indicators
                    )
            
            # Update baseline with successful authentication
            await self._update_user_baseline(user_id, biometric)
            
            # Calculate overall confidence
            overall_confidence = min(
                palm_result.confidence,
                intention_result.confidence if self.require_intention else 1.0
            )
            
            return AuthResult(
                result=AuthenticationResult.SUCCESS,
                confidence=overall_confidence,
                user_id=user_id,
                reason="Authentication successful"
            )
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return AuthResult(
                result=AuthenticationResult.SYSTEM_ERROR,
                confidence=0.0,
                reason=f"Authentication system error: {e}"
            )
    
    def _parse_biometric_data(self, data: Dict[str, Any]) -> BiometricData:
        """Parse biometric data from Palm Slab"""
        
        return BiometricData(
            palm_hash=data.get('palm_hash', ''),
            intention_signature=data.get('intention_signature', ''),
            pressure_pattern=data.get('pressure_pattern', []),
            temporal_pattern=data.get('temporal_pattern', []),
            confidence=data.get('confidence', 0.0)
        )
    
    def _get_user_baseline(self, user_id: str) -> Dict[str, Any]:
        """Get user's biometric baseline"""
        
        if user_id not in self.user_baselines:
            # Create new baseline
            self.user_baselines[user_id] = {
                'palm_baseline': None,
                'intention_baseline': None,
                'pressure_baseline': None,
                'temporal_baseline': None,
                'creation_date': time.time(),
                'update_count': 0,
                'last_success': None
            }
        
        return self.user_baselines[user_id]
    
    async def _authenticate_palmprint(
        self,
        palm_hash: str,
        baseline: Optional[str]
    ) -> Any:
        """Authenticate palmprint against baseline"""
        
        if not baseline:
            # First-time user - create baseline
            return type('AuthResult', (), {
                'confidence': 1.0,  # Trust first enrollment
                'is_new_user': True
            })
        
        # Calculate palmprint similarity
        # In production, this would use sophisticated biometric matching
        similarity = self._calculate_palm_similarity(palm_hash, baseline)
        
        return type('AuthResult', (), {
            'confidence': similarity,
            'is_new_user': False
        })
    
    def _calculate_palm_similarity(self, current_hash: str, baseline_hash: str) -> float:
        """Calculate similarity between palmprint hashes"""
        
        if not current_hash or not baseline_hash:
            return 0.0
        
        # Simple similarity calculation for development
        # In production, this would use proper biometric matching algorithms
        
        if current_hash == baseline_hash:
            return 1.0
        
        # Calculate Hamming distance for hash comparison
        if len(current_hash) == len(baseline_hash):
            differences = sum(c1 != c2 for c1, c2 in zip(current_hash, baseline_hash))
            similarity = 1.0 - (differences / len(current_hash))
            return max(0.0, similarity)
        
        return 0.0
    
    async def _authenticate_intention(
        self,
        intention_signature: str,
        baseline: Optional[str]
    ) -> Any:
        """Authenticate intention signature against baseline"""
        
        if not baseline:
            return type('AuthResult', (), {
                'confidence': 1.0,
                'deviation': 0.0,
                'is_new_user': True
            })
        
        # Analyze intention signature patterns
        deviation = self._calculate_intention_deviation(intention_signature, baseline)
        confidence = max(0.0, 1.0 - deviation)
        
        return type('AuthResult', (), {
            'confidence': confidence,
            'deviation': deviation,
            'is_new_user': False
        })
    
    def _calculate_intention_deviation(
        self,
        current_signature: str,
        baseline_signature: str
    ) -> float:
        """Calculate deviation in intention signature"""
        
        if not current_signature or not baseline_signature:
            return 1.0  # Maximum deviation
        
        # Simple pattern analysis for development
        # In production, this would analyze biometric intention patterns
        
        # Look for significant changes in intention patterns
        current_features = self._extract_intention_features(current_signature)
        baseline_features = self._extract_intention_features(baseline_signature)
        
        if not current_features or not baseline_features:
            return 0.5  # Moderate deviation for missing features
        
        # Calculate feature deviation
        deviations = []
        for key in set(current_features.keys()) | set(baseline_features.keys()):
            current_val = current_features.get(key, 0)
            baseline_val = baseline_features.get(key, 0)
            
            if baseline_val != 0:
                deviation = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                deviation = abs(current_val)
            
            deviations.append(deviation)
        
        return sum(deviations) / len(deviations) if deviations else 0.0
    
    def _extract_intention_features(self, signature: str) -> Dict[str, float]:
        """Extract features from intention signature for analysis"""
        
        if not signature:
            return {}
        
        # Simple feature extraction for development
        features = {
            'length': len(signature),
            'complexity': len(set(signature)) / len(signature) if signature else 0,
            'pattern_hash': hash(signature) % 1000 / 1000.0  # Normalize to 0-1
        }
        
        return features
    
    async def _detect_coercion(
        self,
        biometric: BiometricData,
        baseline: Dict[str, Any]
    ) -> Any:
        """Detect possible coercion through biometric anomalies"""
        
        indicators = []
        coercion_probability = 0.0
        
        # Analyze pressure patterns for stress indicators
        if biometric.pressure_pattern and baseline.get('pressure_baseline'):
            pressure_anomaly = self._analyze_pressure_anomaly(
                biometric.pressure_pattern,
                baseline['pressure_baseline']
            )
            
            if pressure_anomaly > 0.7:
                indicators.append("Unusual pressure pattern detected")
                coercion_probability += 0.3
        
        # Analyze temporal patterns for hesitation/rushing
        if biometric.temporal_pattern and baseline.get('temporal_baseline'):
            temporal_anomaly = self._analyze_temporal_anomaly(
                biometric.temporal_pattern,
                baseline['temporal_baseline']
            )
            
            if temporal_anomaly > 0.7:
                indicators.append("Abnormal timing pattern detected")
                coercion_probability += 0.2
        
        # Check for multiple rapid authentication attempts
        recent_attempts = self._get_recent_auth_attempts(baseline)
        if len(recent_attempts) > 3:
            indicators.append("Multiple rapid authentication attempts")
            coercion_probability += 0.4
        
        # Analyze intention signature for stress patterns
        if biometric.intention_signature:
            stress_indicators = self._analyze_stress_indicators(biometric.intention_signature)
            if stress_indicators:
                indicators.extend(stress_indicators)
                coercion_probability += len(stress_indicators) * 0.1
        
        return type('CoercionResult', (), {
            'coercion_probability': min(1.0, coercion_probability),
            'confidence': 0.8,  # Confidence in coercion detection
            'indicators': indicators
        })
    
    def _analyze_pressure_anomaly(
        self,
        current_pattern: List[float],
        baseline_pattern: List[float]
    ) -> float:
        """Analyze pressure pattern for anomalies"""
        
        if not current_pattern or not baseline_pattern:
            return 0.0
        
        # Simple pressure analysis
        current_avg = sum(current_pattern) / len(current_pattern)
        baseline_avg = sum(baseline_pattern) / len(baseline_pattern)
        
        if baseline_avg == 0:
            return 0.0
        
        deviation = abs(current_avg - baseline_avg) / baseline_avg
        return min(1.0, deviation)
    
    def _analyze_temporal_anomaly(
        self,
        current_pattern: List[float],
        baseline_pattern: List[float]
    ) -> float:
        """Analyze temporal pattern for anomalies"""
        
        if not current_pattern or not baseline_pattern:
            return 0.0
        
        # Analyze timing patterns
        current_total = sum(current_pattern)
        baseline_total = sum(baseline_pattern)
        
        if baseline_total == 0:
            return 0.0
        
        # Look for significant timing changes
        timing_deviation = abs(current_total - baseline_total) / baseline_total
        
        return min(1.0, timing_deviation)
    
    def _get_recent_auth_attempts(self, baseline: Dict[str, Any]) -> List[float]:
        """Get recent authentication attempts for rate limiting analysis"""
        
        # Simple implementation - in production would track actual attempts
        recent_attempts = baseline.get('recent_attempts', [])
        
        # Filter to last hour
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        return [attempt for attempt in recent_attempts if attempt > one_hour_ago]
    
    def _analyze_stress_indicators(self, intention_signature: str) -> List[str]:
        """Analyze intention signature for stress indicators"""
        
        stress_indicators = []
        
        # Simple stress pattern detection
        if not intention_signature:
            return stress_indicators
        
        # Look for irregular patterns that might indicate stress
        if len(intention_signature) < 10:
            stress_indicators.append("Unusually short intention signature")
        
        # Look for repetitive patterns (stress behavior)
        if len(set(intention_signature)) < len(intention_signature) * 0.3:
            stress_indicators.append("Repetitive intention pattern")
        
        return stress_indicators
    
    async def _update_user_baseline(
        self,
        user_id: str,
        biometric: BiometricData
    ):
        """Update user baseline with successful authentication data"""
        
        baseline = self._get_user_baseline(user_id)
        
        # Update palm baseline
        if not baseline.get('palm_baseline'):
            baseline['palm_baseline'] = biometric.palm_hash
        
        # Update intention baseline  
        if not baseline.get('intention_baseline'):
            baseline['intention_baseline'] = biometric.intention_signature
        
        # Update pattern baselines
        if biometric.pressure_pattern and not baseline.get('pressure_baseline'):
            baseline['pressure_baseline'] = biometric.pressure_pattern
        
        if biometric.temporal_pattern and not baseline.get('temporal_baseline'):
            baseline['temporal_baseline'] = biometric.temporal_pattern
        
        # Update metadata
        baseline['last_success'] = time.time()
        baseline['update_count'] = baseline.get('update_count', 0) + 1
        
        # Track recent attempts for coercion detection
        recent_attempts = baseline.get('recent_attempts', [])
        recent_attempts.append(time.time())
        baseline['recent_attempts'] = recent_attempts[-10:]  # Keep last 10 attempts
        
        # Save updated baselines
        self.user_baselines[user_id] = baseline
        self._save_user_baselines()
    
    async def register_new_user(
        self,
        user_id: str,
        biometric_data: Dict[str, Any]
    ) -> AuthResult:
        """Register new user with biometric baseline"""
        
        try:
            biometric = self._parse_biometric_data(biometric_data)
            
            # Validate biometric data quality
            if not self._validate_biometric_quality(biometric):
                return AuthResult(
                    result=AuthenticationResult.SYSTEM_ERROR,
                    confidence=0.0,
                    reason="Biometric data quality insufficient for enrollment"
                )
            
            # Create user baseline
            baseline = {
                'palm_baseline': biometric.palm_hash,
                'intention_baseline': biometric.intention_signature,
                'pressure_baseline': biometric.pressure_pattern,
                'temporal_baseline': biometric.temporal_pattern,
                'creation_date': time.time(),
                'update_count': 1,
                'last_success': time.time(),
                'recent_attempts': [time.time()]
            }
            
            self.user_baselines[user_id] = baseline
            self._save_user_baselines()
            
            self.logger.info(f"New user registered: {user_id}")
            
            return AuthResult(
                result=AuthenticationResult.SUCCESS,
                confidence=1.0,
                user_id=user_id,
                reason="User registration successful"
            )
            
        except Exception as e:
            self.logger.error(f"User registration error: {e}")
            return AuthResult(
                result=AuthenticationResult.SYSTEM_ERROR,
                confidence=0.0,
                reason=f"Registration failed: {e}"
            )
    
    def _validate_biometric_quality(self, biometric: BiometricData) -> bool:
        """Validate quality of biometric data for enrollment"""
        
        # Basic validation
        if not biometric.palm_hash or len(biometric.palm_hash) < 10:
            return False
        
        if not biometric.intention_signature or len(biometric.intention_signature) < 5:
            return False
        
        if biometric.confidence < 0.5:
            return False
        
        return True
    
    def get_slab_status(self) -> Dict[str, Any]:
        """Get Palm Slab system status"""
        
        return {
            'biometric_required': self.require_biometric,
            'intention_required': self.require_intention,
            'coercion_detection': self.coercion_detection,
            'registered_users': len(self.user_baselines),
            'palm_match_threshold': self.palm_match_threshold,
            'intention_match_threshold': self.intention_match_threshold,
            'coercion_sensitivity': self.coercion_sensitivity,
            'storage_initialized': os.path.exists(self.storage_path),
            'status': 'operational'
        }
    
    async def simulate_palm_interaction(
        self,
        user_id: str,
        simulate_coercion: bool = False
    ) -> Dict[str, Any]:
        """
        Simulate Palm Slab interaction for development/testing
        
        Generates synthetic biometric data that would come from actual hardware
        """
        
        import hashlib
        import random
        
        # Generate synthetic palmprint hash
        palm_data = f"{user_id}_{time.time()}_{random.random()}"
        palm_hash = hashlib.sha256(palm_data.encode()).hexdigest()
        
        # Generate synthetic intention signature
        intention_data = f"intent_{user_id}_{random.randint(100, 999)}"
        intention_signature = hashlib.md5(intention_data.encode()).hexdigest()
        
        # Generate synthetic patterns
        pressure_pattern = [random.uniform(0.3, 0.8) for _ in range(10)]
        temporal_pattern = [random.uniform(0.1, 0.5) for _ in range(5)]
        
        # Modify patterns if simulating coercion
        if simulate_coercion:
            # Increase pressure variation (stress indicator)
            pressure_pattern = [p * random.uniform(1.5, 2.0) for p in pressure_pattern]
            # Speed up temporal pattern (rushing indicator)
            temporal_pattern = [t * 0.5 for t in temporal_pattern]
        
        return {
            'palm_hash': palm_hash,
            'intention_signature': intention_signature,
            'pressure_pattern': pressure_pattern,
            'temporal_pattern': temporal_pattern,
            'confidence': random.uniform(0.8, 0.95) if not simulate_coercion else random.uniform(0.4, 0.7)
        }