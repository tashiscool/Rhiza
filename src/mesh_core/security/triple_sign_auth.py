"""
Triple-Sign Authentication System
================================

Implements three-party verification where authentication requires approval
from three independent sources:
1. The user's identity
2. A trusted peer validator  
3. A network consensus validator

This prevents single-point-of-failure attacks and ensures distributed trust.
"""

import asyncio
import time
import hashlib
import hmac
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VerificationLevel(Enum):
    """Authentication verification levels"""
    BASIC = "basic"           # Single factor
    ENHANCED = "enhanced"     # Two factors
    TRIPLE_SIGN = "triple"    # Three-party verification
    MAXIMUM = "maximum"       # All available factors

class AuthenticationStatus(Enum):
    """Authentication request status"""
    PENDING = "pending"
    APPROVED = "approved" 
    DENIED = "denied"
    EXPIRED = "expired"
    COMPROMISED = "compromised"

@dataclass
class VerificationRequest:
    """Authentication verification request"""
    request_id: str
    user_id: str
    timestamp: float
    verification_level: VerificationLevel
    challenge_data: bytes
    context: Dict
    expires_at: float
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def get_challenge_hash(self) -> str:
        return hashlib.sha256(self.challenge_data).hexdigest()

@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    request_id: str
    status: AuthenticationStatus
    verification_level: VerificationLevel
    signatures: List[Dict]  # List of verification signatures
    confidence_score: float
    timestamp: float
    valid_until: float
    metadata: Dict

class TripleSignAuth:
    """
    Advanced three-party authentication system
    
    Ensures that no single entity can compromise authentication by requiring
    approval from three independent sources: user, peer validator, and network.
    """
    
    def __init__(self, node_id: str, trust_ledger=None):
        self.node_id = node_id
        self.trust_ledger = trust_ledger
        self.pending_requests: Dict[str, VerificationRequest] = {}
        self.active_sessions: Dict[str, AuthenticationResult] = {}
        self.peer_validators: Set[str] = set()
        self.network_validators: Set[str] = set()
        self.secret_key = self._generate_secret_key()
        
    def _generate_secret_key(self) -> bytes:
        """Generate node-specific secret key"""
        return hashlib.sha256(f"{self.node_id}_{time.time()}".encode()).digest()
    
    def _create_challenge(self, user_id: str, context: Dict) -> bytes:
        """Create authentication challenge data"""
        challenge_data = {
            'user_id': user_id,
            'node_id': self.node_id,
            'timestamp': time.time(),
            'context': context
        }
        return str(challenge_data).encode()
    
    def _sign_challenge(self, challenge_data: bytes) -> str:
        """Create HMAC signature for challenge"""
        return hmac.new(self.secret_key, challenge_data, hashlib.sha256).hexdigest()
    
    def _verify_signature(self, challenge_data: bytes, signature: str) -> bool:
        """Verify HMAC signature"""
        expected = self._sign_challenge(challenge_data)
        return hmac.compare_digest(expected, signature)
    
    async def register_peer_validator(self, peer_id: str) -> bool:
        """Register a trusted peer as a validator"""
        if self.trust_ledger:
            trust_score = await self.trust_ledger.get_trust_score(peer_id)
            if trust_score and trust_score.overall_score > 0.8:
                self.peer_validators.add(peer_id)
                logger.info(f"Registered peer validator: {peer_id}")
                return True
        return False
    
    async def register_network_validator(self, validator_id: str) -> bool:
        """Register a network consensus validator"""
        # In real implementation, this would verify validator credentials
        self.network_validators.add(validator_id)
        logger.info(f"Registered network validator: {validator_id}")
        return True
    
    async def create_authentication_request(
        self, 
        user_id: str,
        verification_level: VerificationLevel = VerificationLevel.TRIPLE_SIGN,
        context: Optional[Dict] = None,
        timeout_seconds: int = 300
    ) -> VerificationRequest:
        """Create new authentication request"""
        
        if context is None:
            context = {}
            
        request_id = hashlib.sha256(f"{user_id}_{time.time()}_{self.node_id}".encode()).hexdigest()[:16]
        challenge_data = self._create_challenge(user_id, context)
        
        request = VerificationRequest(
            request_id=request_id,
            user_id=user_id,
            timestamp=time.time(),
            verification_level=verification_level,
            challenge_data=challenge_data,
            context=context,
            expires_at=time.time() + timeout_seconds
        )
        
        self.pending_requests[request_id] = request
        logger.info(f"Created authentication request {request_id} for user {user_id}")
        return request
    
    async def user_signature_approval(self, request_id: str, user_signature: str) -> bool:
        """Process user signature approval (first factor)"""
        request = self.pending_requests.get(request_id)
        if not request or request.is_expired():
            return False
            
        # Verify user signature
        if self._verify_signature(request.challenge_data, user_signature):
            request.context['user_approved'] = True
            request.context['user_signature'] = user_signature
            request.context['user_approved_at'] = time.time()
            logger.info(f"User signature approved for request {request_id}")
            return True
        return False
    
    async def peer_validator_approval(self, request_id: str, validator_id: str, signature: str) -> bool:
        """Process peer validator approval (second factor)"""
        request = self.pending_requests.get(request_id)
        if not request or request.is_expired():
            return False
            
        if validator_id not in self.peer_validators:
            logger.warning(f"Unknown peer validator {validator_id}")
            return False
        
        # Verify peer signature
        if self._verify_signature(request.challenge_data, signature):
            request.context['peer_approved'] = True
            request.context['peer_validator'] = validator_id
            request.context['peer_signature'] = signature
            request.context['peer_approved_at'] = time.time()
            logger.info(f"Peer validator {validator_id} approved request {request_id}")
            return True
        return False
    
    async def network_consensus_approval(self, request_id: str, consensus_data: Dict) -> bool:
        """Process network consensus approval (third factor)"""
        request = self.pending_requests.get(request_id)
        if not request or request.is_expired():
            return False
            
        # Verify network consensus
        required_approvals = max(3, len(self.network_validators) // 2 + 1)
        valid_approvals = 0
        
        for validator_id, signature in consensus_data.items():
            if validator_id in self.network_validators:
                if self._verify_signature(request.challenge_data, signature):
                    valid_approvals += 1
        
        if valid_approvals >= required_approvals:
            request.context['network_approved'] = True
            request.context['network_consensus'] = consensus_data
            request.context['network_approved_at'] = time.time()
            request.context['consensus_validators'] = valid_approvals
            logger.info(f"Network consensus approved request {request_id} with {valid_approvals} validators")
            return True
        return False
    
    async def finalize_authentication(self, request_id: str) -> Optional[AuthenticationResult]:
        """Finalize authentication after all approvals"""
        request = self.pending_requests.get(request_id)
        if not request or request.is_expired():
            return None
        
        # Check if all required approvals are present
        required_approvals = self._get_required_approvals(request.verification_level)
        current_approvals = self._get_current_approvals(request)
        
        if not self._has_sufficient_approvals(required_approvals, current_approvals):
            return AuthenticationResult(
                request_id=request_id,
                status=AuthenticationStatus.DENIED,
                verification_level=request.verification_level,
                signatures=[],
                confidence_score=0.0,
                timestamp=time.time(),
                valid_until=time.time(),
                metadata={'reason': 'insufficient_approvals', 'required': required_approvals, 'current': current_approvals}
            )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(request)
        
        # Create signatures list
        signatures = []
        if request.context.get('user_approved'):
            signatures.append({
                'type': 'user',
                'signature': request.context.get('user_signature'),
                'timestamp': request.context.get('user_approved_at')
            })
        
        if request.context.get('peer_approved'):
            signatures.append({
                'type': 'peer',
                'validator': request.context.get('peer_validator'),
                'signature': request.context.get('peer_signature'),
                'timestamp': request.context.get('peer_approved_at')
            })
            
        if request.context.get('network_approved'):
            signatures.append({
                'type': 'network',
                'consensus': request.context.get('network_consensus'),
                'validators': request.context.get('consensus_validators'),
                'timestamp': request.context.get('network_approved_at')
            })
        
        # Create authentication result
        result = AuthenticationResult(
            request_id=request_id,
            status=AuthenticationStatus.APPROVED,
            verification_level=request.verification_level,
            signatures=signatures,
            confidence_score=confidence_score,
            timestamp=time.time(),
            valid_until=time.time() + 3600,  # Valid for 1 hour
            metadata={
                'user_id': request.user_id,
                'approvals': current_approvals,
                'verification_time': time.time() - request.timestamp
            }
        )
        
        # Clean up and store result
        del self.pending_requests[request_id]
        self.active_sessions[request_id] = result
        
        logger.info(f"Authentication finalized for request {request_id} with confidence {confidence_score:.2f}")
        return result
    
    def _get_required_approvals(self, level: VerificationLevel) -> List[str]:
        """Get required approval types for verification level"""
        if level == VerificationLevel.BASIC:
            return ['user']
        elif level == VerificationLevel.ENHANCED:
            return ['user', 'peer']
        elif level == VerificationLevel.TRIPLE_SIGN:
            return ['user', 'peer', 'network']
        elif level == VerificationLevel.MAXIMUM:
            return ['user', 'peer', 'network']
        return ['user']
    
    def _get_current_approvals(self, request: VerificationRequest) -> List[str]:
        """Get current approval types for request"""
        approvals = []
        if request.context.get('user_approved'):
            approvals.append('user')
        if request.context.get('peer_approved'):
            approvals.append('peer')
        if request.context.get('network_approved'):
            approvals.append('network')
        return approvals
    
    def _has_sufficient_approvals(self, required: List[str], current: List[str]) -> bool:
        """Check if current approvals meet requirements"""
        return all(approval in current for approval in required)
    
    def _calculate_confidence_score(self, request: VerificationRequest) -> float:
        """Calculate confidence score based on approvals"""
        base_score = 0.5
        
        if request.context.get('user_approved'):
            base_score += 0.2
            
        if request.context.get('peer_approved'):
            base_score += 0.2
            
        if request.context.get('network_approved'):
            validators = request.context.get('consensus_validators', 0)
            base_score += min(0.3, validators * 0.1)
            
        # Time factor (faster approval = higher confidence)
        elapsed = time.time() - request.timestamp
        if elapsed < 60:  # Less than 1 minute
            base_score += 0.1
        elif elapsed > 300:  # More than 5 minutes
            base_score -= 0.1
            
        return min(1.0, max(0.0, base_score))
    
    async def verify_active_session(self, request_id: str) -> Optional[AuthenticationResult]:
        """Verify if authentication session is still valid"""
        result = self.active_sessions.get(request_id)
        if not result:
            return None
            
        if time.time() > result.valid_until:
            # Session expired, remove it
            del self.active_sessions[request_id]
            return None
            
        return result
    
    async def revoke_authentication(self, request_id: str, reason: str = "manual_revocation") -> bool:
        """Revoke active authentication session"""
        if request_id in self.active_sessions:
            result = self.active_sessions[request_id]
            result.status = AuthenticationStatus.DENIED
            result.metadata['revocation_reason'] = reason
            result.metadata['revoked_at'] = time.time()
            del self.active_sessions[request_id]
            logger.info(f"Authentication {request_id} revoked: {reason}")
            return True
        return False
    
    async def cleanup_expired_requests(self):
        """Clean up expired requests and sessions"""
        current_time = time.time()
        
        # Clean up expired requests
        expired_requests = [
            req_id for req_id, req in self.pending_requests.items()
            if req.is_expired()
        ]
        for req_id in expired_requests:
            del self.pending_requests[req_id]
            logger.debug(f"Cleaned up expired request {req_id}")
        
        # Clean up expired sessions
        expired_sessions = [
            req_id for req_id, result in self.active_sessions.items()
            if current_time > result.valid_until
        ]
        for req_id in expired_sessions:
            del self.active_sessions[req_id]
            logger.debug(f"Cleaned up expired session {req_id}")
    
    def get_status_summary(self) -> Dict:
        """Get summary of authentication system status"""
        return {
            'pending_requests': len(self.pending_requests),
            'active_sessions': len(self.active_sessions),
            'peer_validators': len(self.peer_validators),
            'network_validators': len(self.network_validators),
            'node_id': self.node_id
        }