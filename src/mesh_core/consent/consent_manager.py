"""
Consent Manager
===============

Manages informed consent collection and validation for AI interactions
and data usage within The Mesh network.
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConsentType(Enum):
    """Types of consent"""
    DATA_PROCESSING = "data_processing"
    AI_INTERACTION = "ai_interaction" 
    DATA_SHARING = "data_sharing"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    PERSONALIZATION = "personalization"

class ConsentStatus(Enum):
    """Status of consent"""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"

@dataclass
class ConsentRecord:
    """Record of user consent"""
    consent_id: str
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: Optional[float]
    expires_at: Optional[float]
    details: Dict[str, any]
    revocation_rights: List[str]

class ConsentManager:
    """Manages consent collection, validation, and enforcement"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.consent_templates: Dict[str, Dict] = {}
        
        logger.info(f"ConsentManager initialized for node {node_id}")

    async def request_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        details: Dict[str, any],
        expires_in_days: Optional[int] = None
    ) -> str:
        """Request consent from user"""
        consent_id = self._generate_consent_id(user_id, consent_type)
        
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 24 * 3600)
        
        record = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            status=ConsentStatus.PENDING,
            granted_at=None,
            expires_at=expires_at,
            details=details,
            revocation_rights=["immediate_revocation", "data_deletion", "export_data"]
        )
        
        self.consent_records[consent_id] = record
        
        # Send consent request to user (mock implementation)
        await self._send_consent_request(user_id, record)
        
        logger.info(f"Consent requested: {consent_id} for {user_id}")
        return consent_id

    async def grant_consent(self, consent_id: str, user_confirmation: bool = True) -> bool:
        """Grant consent"""
        if consent_id not in self.consent_records:
            return False
        
        record = self.consent_records[consent_id]
        
        if user_confirmation:
            record.status = ConsentStatus.GRANTED
            record.granted_at = time.time()
            logger.info(f"Consent granted: {consent_id}")
            return True
        else:
            record.status = ConsentStatus.DENIED
            logger.info(f"Consent denied: {consent_id}")
            return False

    async def revoke_consent(self, consent_id: str, user_id: str) -> bool:
        """Revoke existing consent"""
        if consent_id not in self.consent_records:
            return False
        
        record = self.consent_records[consent_id]
        
        if record.user_id != user_id:
            return False
        
        record.status = ConsentStatus.REVOKED
        logger.info(f"Consent revoked: {consent_id}")
        return True

    async def check_consent_valid(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has valid consent for specific type"""
        for record in self.consent_records.values():
            if (record.user_id == user_id and 
                record.consent_type == consent_type and
                record.status == ConsentStatus.GRANTED):
                
                # Check if expired
                if record.expires_at and time.time() > record.expires_at:
                    record.status = ConsentStatus.EXPIRED
                    return False
                
                return True
        
        return False

    def _generate_consent_id(self, user_id: str, consent_type: ConsentType) -> str:
        """Generate unique consent ID"""
        content = f"{user_id}|{consent_type.value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _send_consent_request(self, user_id: str, record: ConsentRecord) -> bool:
        """Send consent request to user (mock implementation)"""
        # In real implementation, would send notification to user
        logger.debug(f"Consent request sent to {user_id}: {record.consent_type.value}")
        return True