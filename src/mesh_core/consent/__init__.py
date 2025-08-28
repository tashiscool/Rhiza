"""
Consent Framework
================

Manages informed consent for AI interactions and data usage
within The Mesh network, ensuring user agency and privacy.

Components:
- ConsentManager: Manages consent collection and validation
- PrivacyController: Controls privacy settings and data access
- DataUsageTracker: Tracks data usage and consent compliance
- ConsentValidator: Validates consent authenticity and scope
"""

from .consent_manager import ConsentManager
from .privacy_controller import PrivacyController
from .data_usage_tracker import DataUsageTracker
from .consent_validator import ConsentValidator

__all__ = [
    'ConsentManager',
    'PrivacyController',
    'DataUsageTracker',
    'ConsentValidator'
]

__version__ = "1.0.0"