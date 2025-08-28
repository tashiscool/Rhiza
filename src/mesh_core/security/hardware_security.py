"""
Hardware Security Integration
============================

Integrates hardware-level security features for The Mesh including:
- Secure element integration
- Hardware token management
- Biometric hardware interfaces
- Tamper detection systems
- Hardware-based key storage
"""

import asyncio
import time
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Hardware security levels"""
    SOFTWARE_ONLY = "software"      # Software-only security
    BASIC_HARDWARE = "basic"        # Basic hardware features
    SECURE_ELEMENT = "secure"       # Secure element integration
    HSM_LEVEL = "hsm"              # Hardware Security Module level
    MILITARY_GRADE = "military"     # Military-grade security

class DeviceType(Enum):
    """Types of security hardware devices"""
    SECURE_ELEMENT = "secure_element"   # Embedded secure element
    TPM = "tpm"                        # Trusted Platform Module
    HARDWARE_TOKEN = "token"           # External hardware token
    BIOMETRIC_READER = "biometric"     # Biometric reading device
    TAMPER_SENSOR = "tamper"          # Tamper detection hardware

class SecurityOperation(Enum):
    """Hardware security operations"""
    KEY_GENERATION = "key_gen"
    KEY_STORAGE = "key_store"
    ENCRYPTION = "encrypt"
    DECRYPTION = "decrypt"
    SIGNING = "sign"
    VERIFICATION = "verify"
    RANDOM_GENERATION = "random"
    BIOMETRIC_CAPTURE = "biometric"

@dataclass
class SecurityDevice:
    """Hardware security device representation"""
    device_id: str
    device_type: DeviceType
    security_level: SecurityLevel
    capabilities: Set[SecurityOperation]
    status: str
    last_verified: float
    metadata: Dict

@dataclass
class HardwareToken:
    """Hardware token for authentication"""
    token_id: str
    device_id: str
    user_id: str
    created_at: float
    expires_at: float
    operations: Set[SecurityOperation]
    security_data: Dict

class HardwareSecurity:
    """
    Hardware security integration system
    
    Manages hardware security devices and provides unified interface
    for hardware-backed security operations.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.available_devices: Dict[str, SecurityDevice] = {}
        self.active_tokens: Dict[str, HardwareToken] = {}
        self.device_capabilities: Dict[DeviceType, Set[SecurityOperation]] = self._init_device_capabilities()
        self.security_policies: Dict[SecurityLevel, Dict] = self._init_security_policies()
        
    def _init_device_capabilities(self) -> Dict[DeviceType, Set[SecurityOperation]]:
        """Initialize device capability mappings"""
        return {
            DeviceType.SECURE_ELEMENT: {
                SecurityOperation.KEY_GENERATION,
                SecurityOperation.KEY_STORAGE,
                SecurityOperation.ENCRYPTION,
                SecurityOperation.DECRYPTION,
                SecurityOperation.SIGNING,
                SecurityOperation.VERIFICATION,
                SecurityOperation.RANDOM_GENERATION
            },
            DeviceType.TPM: {
                SecurityOperation.KEY_GENERATION,
                SecurityOperation.KEY_STORAGE,
                SecurityOperation.ENCRYPTION,
                SecurityOperation.DECRYPTION,
                SecurityOperation.SIGNING,
                SecurityOperation.VERIFICATION,
                SecurityOperation.RANDOM_GENERATION
            },
            DeviceType.HARDWARE_TOKEN: {
                SecurityOperation.KEY_STORAGE,
                SecurityOperation.SIGNING,
                SecurityOperation.VERIFICATION
            },
            DeviceType.BIOMETRIC_READER: {
                SecurityOperation.BIOMETRIC_CAPTURE,
                SecurityOperation.VERIFICATION
            },
            DeviceType.TAMPER_SENSOR: {
                SecurityOperation.VERIFICATION
            }
        }
    
    def _init_security_policies(self) -> Dict[SecurityLevel, Dict]:
        """Initialize security policies for each level"""
        return {
            SecurityLevel.SOFTWARE_ONLY: {
                'required_devices': [],
                'key_length': 2048,
                'encryption_algorithm': 'AES-256',
                'tamper_detection': False
            },
            SecurityLevel.BASIC_HARDWARE: {
                'required_devices': [DeviceType.HARDWARE_TOKEN],
                'key_length': 2048,
                'encryption_algorithm': 'AES-256',
                'tamper_detection': False
            },
            SecurityLevel.SECURE_ELEMENT: {
                'required_devices': [DeviceType.SECURE_ELEMENT],
                'key_length': 3072,
                'encryption_algorithm': 'AES-256',
                'tamper_detection': True
            },
            SecurityLevel.HSM_LEVEL: {
                'required_devices': [DeviceType.SECURE_ELEMENT, DeviceType.TPM],
                'key_length': 4096,
                'encryption_algorithm': 'AES-256',
                'tamper_detection': True
            },
            SecurityLevel.MILITARY_GRADE: {
                'required_devices': [DeviceType.SECURE_ELEMENT, DeviceType.TPM, DeviceType.TAMPER_SENSOR],
                'key_length': 4096,
                'encryption_algorithm': 'AES-256',
                'tamper_detection': True
            }
        }
    
    async def discover_hardware_devices(self) -> List[SecurityDevice]:
        """Discover available hardware security devices"""
        
        discovered_devices = []
        
        # Simulate hardware device discovery
        # In real implementation, would interface with:
        # - Platform security modules
        # - USB hardware tokens  
        # - Embedded secure elements
        # - TPM modules
        # - Biometric readers
        
        # Mock secure element
        if await self._check_secure_element_availability():
            device = SecurityDevice(
                device_id="se_001",
                device_type=DeviceType.SECURE_ELEMENT,
                security_level=SecurityLevel.SECURE_ELEMENT,
                capabilities=self.device_capabilities[DeviceType.SECURE_ELEMENT],
                status="available",
                last_verified=time.time(),
                metadata={
                    'manufacturer': 'SecureChip Corp',
                    'model': 'SC-2024',
                    'firmware_version': '2.1.0'
                }
            )
            discovered_devices.append(device)
            self.available_devices[device.device_id] = device
        
        # Mock TPM
        if await self._check_tpm_availability():
            device = SecurityDevice(
                device_id="tpm_001", 
                device_type=DeviceType.TPM,
                security_level=SecurityLevel.HSM_LEVEL,
                capabilities=self.device_capabilities[DeviceType.TPM],
                status="available",
                last_verified=time.time(),
                metadata={
                    'version': '2.0',
                    'manufacturer': 'TPM Systems'
                }
            )
            discovered_devices.append(device)
            self.available_devices[device.device_id] = device
        
        logger.info(f"Discovered {len(discovered_devices)} hardware security devices")
        return discovered_devices
    
    async def _check_secure_element_availability(self) -> bool:
        """Check if secure element is available"""
        # In real implementation, would check for actual hardware
        # For now, assume available on supported platforms
        return True
    
    async def _check_tpm_availability(self) -> bool:
        """Check if TPM is available"""
        # In real implementation, would check for TPM hardware
        return os.path.exists('/dev/tpm0') or os.path.exists('/sys/class/tpm/tpm0')
    
    async def initialize_device(self, device_id: str, security_level: SecurityLevel) -> bool:
        """Initialize hardware device for specified security level"""
        
        device = self.available_devices.get(device_id)
        if not device:
            logger.error(f"Device {device_id} not found")
            return False
        
        # Check if device supports required security level
        if device.security_level.value < security_level.value:
            logger.error(f"Device {device_id} does not support security level {security_level.value}")
            return False
        
        # Initialize device based on type
        try:
            if device.device_type == DeviceType.SECURE_ELEMENT:
                await self._initialize_secure_element(device, security_level)
            elif device.device_type == DeviceType.TPM:
                await self._initialize_tpm(device, security_level)
            elif device.device_type == DeviceType.HARDWARE_TOKEN:
                await self._initialize_hardware_token(device, security_level)
            
            device.status = "initialized"
            device.last_verified = time.time()
            logger.info(f"Initialized device {device_id} for security level {security_level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize device {device_id}: {e}")
            device.status = "error"
            return False
    
    async def _initialize_secure_element(self, device: SecurityDevice, level: SecurityLevel):
        """Initialize secure element"""
        # In real implementation, would:
        # - Establish secure communication channel
        # - Provision cryptographic keys
        # - Configure security policies
        # - Enable tamper detection
        pass
    
    async def _initialize_tpm(self, device: SecurityDevice, level: SecurityLevel):
        """Initialize TPM"""
        # In real implementation, would:
        # - Take ownership of TPM
        # - Create key hierarchies
        # - Configure PCR policies
        # - Enable attestation
        pass
    
    async def _initialize_hardware_token(self, device: SecurityDevice, level: SecurityLevel):
        """Initialize hardware token"""
        # In real implementation, would:
        # - Authenticate to token
        # - Initialize key slots
        # - Configure access policies
        pass
    
    async def generate_hardware_key(
        self,
        device_id: str,
        key_type: str = "RSA",
        key_size: int = 2048,
        usage: Set[SecurityOperation] = None
    ) -> Optional[str]:
        """Generate cryptographic key in hardware"""
        
        device = self.available_devices.get(device_id)
        if not device or device.status != "initialized":
            logger.error(f"Device {device_id} not available")
            return None
        
        if SecurityOperation.KEY_GENERATION not in device.capabilities:
            logger.error(f"Device {device_id} does not support key generation")
            return None
        
        if usage is None:
            usage = {SecurityOperation.SIGNING, SecurityOperation.ENCRYPTION}
        
        # Generate key in hardware
        key_id = f"hw_key_{device_id}_{int(time.time())}"
        
        try:
            # In real implementation, would interface with hardware API
            await self._hardware_generate_key(device, key_id, key_type, key_size, usage)
            
            logger.info(f"Generated hardware key {key_id} on device {device_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Hardware key generation failed: {e}")
            return None
    
    async def _hardware_generate_key(
        self,
        device: SecurityDevice,
        key_id: str,
        key_type: str,
        key_size: int,
        usage: Set[SecurityOperation]
    ):
        """Perform hardware key generation"""
        # Mock hardware key generation
        # In real implementation, would call device-specific APIs
        pass
    
    async def hardware_sign(self, device_id: str, key_id: str, data: bytes) -> Optional[bytes]:
        """Sign data using hardware-stored key"""
        
        device = self.available_devices.get(device_id)
        if not device or device.status != "initialized":
            logger.error(f"Device {device_id} not available")
            return None
        
        if SecurityOperation.SIGNING not in device.capabilities:
            logger.error(f"Device {device_id} does not support signing")
            return None
        
        try:
            # In real implementation, would use hardware signing
            signature = await self._hardware_sign_operation(device, key_id, data)
            return signature
            
        except Exception as e:
            logger.error(f"Hardware signing failed: {e}")
            return None
    
    async def _hardware_sign_operation(self, device: SecurityDevice, key_id: str, data: bytes) -> bytes:
        """Perform hardware signing operation"""
        # Mock hardware signing
        # In real implementation, would call device-specific signing APIs
        return hashlib.sha256(data + key_id.encode()).digest()
    
    async def hardware_encrypt(self, device_id: str, key_id: str, data: bytes) -> Optional[bytes]:
        """Encrypt data using hardware"""
        
        device = self.available_devices.get(device_id)
        if not device or device.status != "initialized":
            return None
        
        if SecurityOperation.ENCRYPTION not in device.capabilities:
            return None
        
        try:
            encrypted_data = await self._hardware_encrypt_operation(device, key_id, data)
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Hardware encryption failed: {e}")
            return None
    
    async def _hardware_encrypt_operation(self, device: SecurityDevice, key_id: str, data: bytes) -> bytes:
        """Perform hardware encryption operation"""
        # Mock hardware encryption
        # In real implementation, would use device encryption APIs
        return data  # Placeholder
    
    async def create_hardware_token(
        self,
        user_id: str,
        device_id: str,
        operations: Set[SecurityOperation],
        validity_hours: int = 24
    ) -> Optional[HardwareToken]:
        """Create hardware-backed authentication token"""
        
        device = self.available_devices.get(device_id)
        if not device:
            return None
        
        # Check if device supports required operations
        if not operations.issubset(device.capabilities):
            logger.error(f"Device {device_id} does not support all required operations")
            return None
        
        token_id = f"hwtoken_{device_id}_{user_id}_{int(time.time())}"
        
        # Generate hardware-backed token
        security_data = await self._generate_token_security_data(device, user_id)
        
        token = HardwareToken(
            token_id=token_id,
            device_id=device_id,
            user_id=user_id,
            created_at=time.time(),
            expires_at=time.time() + (validity_hours * 3600),
            operations=operations,
            security_data=security_data
        )
        
        self.active_tokens[token_id] = token
        logger.info(f"Created hardware token {token_id} for user {user_id}")
        return token
    
    async def _generate_token_security_data(self, device: SecurityDevice, user_id: str) -> Dict:
        """Generate security data for hardware token"""
        # In real implementation, would generate hardware-backed security data
        return {
            'device_attestation': 'mock_attestation',
            'user_binding': hashlib.sha256(f"{user_id}_{device.device_id}".encode()).hexdigest(),
            'timestamp': time.time()
        }
    
    async def verify_hardware_token(self, token_id: str) -> bool:
        """Verify hardware token validity"""
        
        token = self.active_tokens.get(token_id)
        if not token:
            return False
        
        # Check expiration
        if time.time() > token.expires_at:
            del self.active_tokens[token_id]
            return False
        
        # Verify hardware device is still available
        device = self.available_devices.get(token.device_id)
        if not device or device.status != "initialized":
            return False
        
        # Verify security data with hardware
        try:
            is_valid = await self._verify_token_security_data(device, token)
            return is_valid
            
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False
    
    async def _verify_token_security_data(self, device: SecurityDevice, token: HardwareToken) -> bool:
        """Verify token security data with hardware"""
        # In real implementation, would verify with hardware device
        # For now, basic validation
        return token.security_data.get('device_attestation') is not None
    
    async def capture_biometric(self, device_id: str, biometric_type: str) -> Optional[Dict]:
        """Capture biometric data using hardware"""
        
        device = self.available_devices.get(device_id)
        if not device or device.device_type != DeviceType.BIOMETRIC_READER:
            return None
        
        if SecurityOperation.BIOMETRIC_CAPTURE not in device.capabilities:
            return None
        
        try:
            # In real implementation, would interface with biometric hardware
            biometric_data = await self._hardware_capture_biometric(device, biometric_type)
            return biometric_data
            
        except Exception as e:
            logger.error(f"Biometric capture failed: {e}")
            return None
    
    async def _hardware_capture_biometric(self, device: SecurityDevice, biometric_type: str) -> Dict:
        """Perform hardware biometric capture"""
        # Mock biometric capture
        return {
            'type': biometric_type,
            'data': 'mock_biometric_data',
            'quality_score': 0.95,
            'timestamp': time.time()
        }
    
    async def detect_tampering(self) -> List[Dict]:
        """Detect hardware tampering attempts"""
        
        tampering_events = []
        
        for device_id, device in self.available_devices.items():
            if DeviceType.TAMPER_SENSOR in [device.device_type] or device.security_level in [SecurityLevel.HSM_LEVEL, SecurityLevel.MILITARY_GRADE]:
                
                # Check for tampering indicators
                tamper_status = await self._check_device_tampering(device)
                if tamper_status['tampered']:
                    tampering_events.append({
                        'device_id': device_id,
                        'device_type': device.device_type.value,
                        'tamper_type': tamper_status['tamper_type'],
                        'severity': tamper_status['severity'],
                        'timestamp': time.time()
                    })
        
        if tampering_events:
            logger.warning(f"Detected {len(tampering_events)} tampering events")
        
        return tampering_events
    
    async def _check_device_tampering(self, device: SecurityDevice) -> Dict:
        """Check specific device for tampering"""
        # In real implementation, would query hardware tamper detection
        return {
            'tampered': False,
            'tamper_type': None,
            'severity': 'none'
        }
    
    async def get_security_level_compliance(self, required_level: SecurityLevel) -> Dict:
        """Check if current hardware setup meets security level requirements"""
        
        policy = self.security_policies[required_level]
        required_devices = policy['required_devices']
        
        compliance = {
            'compliant': True,
            'required_level': required_level.value,
            'missing_devices': [],
            'available_devices': list(self.available_devices.keys()),
            'tamper_protection': False
        }
        
        # Check required devices
        available_types = set(device.device_type for device in self.available_devices.values() if device.status == "initialized")
        
        for required_type in required_devices:
            if required_type not in available_types:
                compliance['compliant'] = False
                compliance['missing_devices'].append(required_type.value)
        
        # Check tamper protection
        if policy.get('tamper_detection', False):
            tamper_devices = [
                device for device in self.available_devices.values()
                if DeviceType.TAMPER_SENSOR in [device.device_type] or device.security_level in [SecurityLevel.HSM_LEVEL, SecurityLevel.MILITARY_GRADE]
            ]
            compliance['tamper_protection'] = len(tamper_devices) > 0
            if not compliance['tamper_protection']:
                compliance['compliant'] = False
        
        return compliance
    
    def get_hardware_status(self) -> Dict:
        """Get comprehensive hardware security status"""
        
        device_status = {}
        for device_id, device in self.available_devices.items():
            device_status[device_id] = {
                'type': device.device_type.value,
                'security_level': device.security_level.value,
                'status': device.status,
                'capabilities': [op.value for op in device.capabilities],
                'last_verified': device.last_verified
            }
        
        return {
            'node_id': self.node_id,
            'total_devices': len(self.available_devices),
            'active_tokens': len(self.active_tokens),
            'devices': device_status,
            'supported_levels': [level.value for level in SecurityLevel]
        }