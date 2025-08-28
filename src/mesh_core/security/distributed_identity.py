"""
Distributed Identity Management System
=====================================

Network-wide identity verification that doesn't rely on central authority.
Each identity is verified by multiple network participants and maintains
cryptographic proofs of authenticity across the mesh.
"""

import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VerificationLevel(Enum):
    """Identity verification confidence levels"""
    UNVERIFIED = "unverified"     # No verification
    BASIC = "basic"               # Single-node verification  
    PEER_VERIFIED = "peer"        # Multiple peer verification
    NETWORK_CONSENSUS = "network" # Network-wide consensus
    CRYPTOGRAPHIC = "crypto"      # Cryptographic proof verified

class IdentityType(Enum):
    """Types of identities in the mesh"""
    USER = "user"           # Human user identity
    NODE = "node"           # Mesh node identity
    SERVICE = "service"     # Service/application identity
    DEVICE = "device"       # Hardware device identity

@dataclass
class Identity:
    """Core identity information"""
    identity_id: str
    identity_type: IdentityType
    public_key: str
    created_at: float
    last_verified: float
    verification_level: VerificationLevel
    attributes: Dict
    trust_score: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['identity_type'] = self.identity_type.value
        data['verification_level'] = self.verification_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Identity':
        data['identity_type'] = IdentityType(data['identity_type'])
        data['verification_level'] = VerificationLevel(data['verification_level'])
        return cls(**data)

@dataclass
class IdentityProof:
    """Cryptographic proof of identity"""
    proof_id: str
    identity_id: str
    proof_type: str
    proof_data: Dict
    created_at: float
    verified_by: List[str]  # List of verifying node IDs
    signature: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IdentityProof':
        return cls(**data)

class DistributedIdentity:
    """
    Distributed identity management system
    
    Manages identities across the mesh network without central authority,
    using peer verification and cryptographic proofs.
    """
    
    def __init__(self, node_id: str, trust_ledger=None):
        self.node_id = node_id
        self.trust_ledger = trust_ledger
        self.identities: Dict[str, Identity] = {}
        self.identity_proofs: Dict[str, List[IdentityProof]] = {}
        self.verification_requests: Dict[str, Dict] = {}
        self.peer_verifiers: Set[str] = set()
        
    def _generate_identity_id(self, identity_type: IdentityType, public_key: str) -> str:
        """Generate unique identity ID"""
        data = f"{identity_type.value}:{public_key}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def _generate_proof_id(self, identity_id: str, proof_type: str) -> str:
        """Generate unique proof ID"""
        data = f"{identity_id}:{proof_type}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _create_signature(self, data: str) -> str:
        """Create signature for data (simplified)"""
        return hashlib.sha256(f"{self.node_id}:{data}".encode()).hexdigest()
    
    def _verify_signature(self, data: str, signature: str, verifier_id: str) -> bool:
        """Verify signature (simplified)"""
        expected = hashlib.sha256(f"{verifier_id}:{data}".encode()).hexdigest()
        return expected == signature
    
    async def create_identity(
        self,
        identity_type: IdentityType,
        public_key: str,
        attributes: Optional[Dict] = None
    ) -> Identity:
        """Create new identity in the network"""
        
        if attributes is None:
            attributes = {}
            
        identity_id = self._generate_identity_id(identity_type, public_key)
        
        identity = Identity(
            identity_id=identity_id,
            identity_type=identity_type,
            public_key=public_key,
            created_at=time.time(),
            last_verified=time.time(),
            verification_level=VerificationLevel.UNVERIFIED,
            attributes=attributes,
            trust_score=0.5  # Start with neutral trust
        )
        
        self.identities[identity_id] = identity
        self.identity_proofs[identity_id] = []
        
        logger.info(f"Created identity {identity_id} of type {identity_type.value}")
        return identity
    
    async def submit_identity_proof(
        self,
        identity_id: str,
        proof_type: str,
        proof_data: Dict
    ) -> Optional[IdentityProof]:
        """Submit cryptographic proof for identity"""
        
        identity = self.identities.get(identity_id)
        if not identity:
            logger.error(f"Identity {identity_id} not found")
            return None
        
        proof_id = self._generate_proof_id(identity_id, proof_type)
        signature = self._create_signature(json.dumps(proof_data, sort_keys=True))
        
        proof = IdentityProof(
            proof_id=proof_id,
            identity_id=identity_id,
            proof_type=proof_type,
            proof_data=proof_data,
            created_at=time.time(),
            verified_by=[self.node_id],
            signature=signature
        )
        
        self.identity_proofs[identity_id].append(proof)
        logger.info(f"Added proof {proof_id} for identity {identity_id}")
        return proof
    
    async def verify_identity_proof(self, proof_id: str, verifier_id: str) -> bool:
        """Verify identity proof by another node"""
        
        # Find the proof
        proof = None
        identity_id = None
        
        for id_key, proofs in self.identity_proofs.items():
            for p in proofs:
                if p.proof_id == proof_id:
                    proof = p
                    identity_id = id_key
                    break
            if proof:
                break
        
        if not proof:
            logger.error(f"Proof {proof_id} not found")
            return False
        
        # Check if verifier is trusted
        if self.trust_ledger:
            trust_score = await self.trust_ledger.get_trust_score(verifier_id)
            if not trust_score or trust_score.overall_score < 0.6:
                logger.warning(f"Verifier {verifier_id} has insufficient trust score")
                return False
        
        # Verify signature
        data = json.dumps(proof.proof_data, sort_keys=True)
        if not self._verify_signature(data, proof.signature, verifier_id):
            logger.error(f"Invalid signature for proof {proof_id}")
            return False
        
        # Add verifier if not already present
        if verifier_id not in proof.verified_by:
            proof.verified_by.append(verifier_id)
            
            # Update identity verification level based on number of verifiers
            identity = self.identities[identity_id]
            await self._update_verification_level(identity)
            
            logger.info(f"Proof {proof_id} verified by {verifier_id}")
            return True
        
        return True
    
    async def _update_verification_level(self, identity: Identity):
        """Update identity verification level based on proofs and verifications"""
        
        proofs = self.identity_proofs.get(identity.identity_id, [])
        if not proofs:
            identity.verification_level = VerificationLevel.UNVERIFIED
            return
        
        total_verifiers = set()
        crypto_proofs = 0
        
        for proof in proofs:
            total_verifiers.update(proof.verified_by)
            if proof.proof_type in ['cryptographic', 'digital_signature', 'certificate']:
                crypto_proofs += 1
        
        verifier_count = len(total_verifiers)
        
        # Determine verification level
        if crypto_proofs > 0 and verifier_count >= 5:
            identity.verification_level = VerificationLevel.CRYPTOGRAPHIC
        elif verifier_count >= 3:
            identity.verification_level = VerificationLevel.NETWORK_CONSENSUS
        elif verifier_count >= 2:
            identity.verification_level = VerificationLevel.PEER_VERIFIED
        elif verifier_count >= 1:
            identity.verification_level = VerificationLevel.BASIC
        else:
            identity.verification_level = VerificationLevel.UNVERIFIED
        
        # Update trust score based on verification level
        level_scores = {
            VerificationLevel.UNVERIFIED: 0.3,
            VerificationLevel.BASIC: 0.5,
            VerificationLevel.PEER_VERIFIED: 0.7,
            VerificationLevel.NETWORK_CONSENSUS: 0.85,
            VerificationLevel.CRYPTOGRAPHIC: 0.95
        }
        
        identity.trust_score = level_scores[identity.verification_level]
        identity.last_verified = time.time()
        
        logger.info(f"Updated identity {identity.identity_id} to level {identity.verification_level.value}")
    
    async def request_identity_verification(
        self,
        identity_id: str,
        requested_level: VerificationLevel
    ) -> str:
        """Request verification of identity from network"""
        
        identity = self.identities.get(identity_id)
        if not identity:
            logger.error(f"Identity {identity_id} not found")
            return ""
        
        request_id = hashlib.sha256(f"{identity_id}:{time.time()}".encode()).hexdigest()[:16]
        
        request = {
            'request_id': request_id,
            'identity_id': identity_id,
            'requested_level': requested_level,
            'created_at': time.time(),
            'status': 'pending',
            'verifications': []
        }
        
        self.verification_requests[request_id] = request
        logger.info(f"Created verification request {request_id} for identity {identity_id}")
        return request_id
    
    async def process_verification_request(
        self,
        request_id: str,
        verifier_id: str,
        verification_result: Dict
    ) -> bool:
        """Process verification result from another node"""
        
        request = self.verification_requests.get(request_id)
        if not request:
            logger.error(f"Verification request {request_id} not found")
            return False
        
        # Check if verifier is trusted
        if self.trust_ledger:
            trust_score = await self.trust_ledger.get_trust_score(verifier_id)
            if not trust_score or trust_score.overall_score < 0.7:
                logger.warning(f"Verifier {verifier_id} has insufficient trust score")
                return False
        
        # Add verification result
        request['verifications'].append({
            'verifier_id': verifier_id,
            'result': verification_result,
            'timestamp': time.time()
        })
        
        # Check if we have enough verifications
        required_verifications = self._get_required_verifications(request['requested_level'])
        if len(request['verifications']) >= required_verifications:
            await self._finalize_verification_request(request_id)
        
        logger.info(f"Processed verification from {verifier_id} for request {request_id}")
        return True
    
    def _get_required_verifications(self, level: VerificationLevel) -> int:
        """Get number of required verifications for level"""
        requirements = {
            VerificationLevel.BASIC: 1,
            VerificationLevel.PEER_VERIFIED: 2,
            VerificationLevel.NETWORK_CONSENSUS: 3,
            VerificationLevel.CRYPTOGRAPHIC: 5
        }
        return requirements.get(level, 1)
    
    async def _finalize_verification_request(self, request_id: str):
        """Finalize verification request after gathering responses"""
        
        request = self.verification_requests[request_id]
        identity_id = request['identity_id']
        identity = self.identities.get(identity_id)
        
        if not identity:
            return
        
        # Analyze verification results
        positive_verifications = sum(
            1 for v in request['verifications']
            if v['result'].get('verified', False)
        )
        
        total_verifications = len(request['verifications'])
        verification_ratio = positive_verifications / total_verifications if total_verifications > 0 else 0
        
        # Update identity based on verification results
        if verification_ratio >= 0.8:  # 80% positive threshold
            identity.verification_level = request['requested_level']
            identity.trust_score = min(0.95, identity.trust_score + 0.1)
            request['status'] = 'approved'
        elif verification_ratio >= 0.6:  # 60% positive threshold - partial approval
            # Upgrade by one level if possible
            levels = list(VerificationLevel)
            current_index = levels.index(identity.verification_level)
            if current_index < len(levels) - 1:
                identity.verification_level = levels[current_index + 1]
            identity.trust_score = min(0.9, identity.trust_score + 0.05)
            request['status'] = 'partially_approved'
        else:
            # Failed verification
            identity.trust_score = max(0.1, identity.trust_score - 0.1)
            request['status'] = 'denied'
        
        identity.last_verified = time.time()
        request['finalized_at'] = time.time()
        
        logger.info(f"Finalized verification request {request_id}: {request['status']}")
    
    async def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Get identity by ID"""
        return self.identities.get(identity_id)
    
    async def get_identity_proofs(self, identity_id: str) -> List[IdentityProof]:
        """Get all proofs for identity"""
        return self.identity_proofs.get(identity_id, [])
    
    async def search_identities(
        self,
        identity_type: Optional[IdentityType] = None,
        verification_level: Optional[VerificationLevel] = None,
        min_trust_score: Optional[float] = None
    ) -> List[Identity]:
        """Search identities by criteria"""
        
        results = []
        for identity in self.identities.values():
            if identity_type and identity.identity_type != identity_type:
                continue
            if verification_level and identity.verification_level != verification_level:
                continue
            if min_trust_score and identity.trust_score < min_trust_score:
                continue
            results.append(identity)
        
        return results
    
    async def revoke_identity(self, identity_id: str, reason: str) -> bool:
        """Revoke identity and all associated proofs"""
        
        identity = self.identities.get(identity_id)
        if not identity:
            return False
        
        # Mark identity as revoked
        identity.attributes['revoked'] = True
        identity.attributes['revocation_reason'] = reason
        identity.attributes['revoked_at'] = time.time()
        identity.trust_score = 0.0
        identity.verification_level = VerificationLevel.UNVERIFIED
        
        logger.info(f"Revoked identity {identity_id}: {reason}")
        return True
    
    async def export_identity(self, identity_id: str) -> Optional[Dict]:
        """Export identity and proofs for transfer"""
        
        identity = self.identities.get(identity_id)
        if not identity:
            return None
        
        proofs = self.identity_proofs.get(identity_id, [])
        
        return {
            'identity': identity.to_dict(),
            'proofs': [proof.to_dict() for proof in proofs],
            'exported_at': time.time(),
            'exported_by': self.node_id
        }
    
    async def import_identity(self, data: Dict) -> bool:
        """Import identity and proofs from another node"""
        
        try:
            identity_data = data['identity']
            identity = Identity.from_dict(identity_data)
            
            # Verify identity isn't already present
            if identity.identity_id in self.identities:
                logger.warning(f"Identity {identity.identity_id} already exists")
                return False
            
            # Import identity
            self.identities[identity.identity_id] = identity
            
            # Import proofs
            proofs = []
            for proof_data in data.get('proofs', []):
                proof = IdentityProof.from_dict(proof_data)
                proofs.append(proof)
            
            self.identity_proofs[identity.identity_id] = proofs
            
            logger.info(f"Imported identity {identity.identity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import identity: {e}")
            return False
    
    def get_status_summary(self) -> Dict:
        """Get summary of identity system status"""
        
        type_counts = {}
        level_counts = {}
        
        for identity in self.identities.values():
            # Count by type
            type_key = identity.identity_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            
            # Count by verification level
            level_key = identity.verification_level.value
            level_counts[level_key] = level_counts.get(level_key, 0) + 1
        
        return {
            'total_identities': len(self.identities),
            'by_type': type_counts,
            'by_verification_level': level_counts,
            'pending_verifications': len(self.verification_requests),
            'node_id': self.node_id
        }