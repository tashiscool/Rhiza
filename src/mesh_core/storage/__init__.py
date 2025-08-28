"""
Storage System - Persistent storage layer for The Mesh

Provides unified storage interface with multiple backend support:
- SQLite for development and single-node deployments
- PostgreSQL for production distributed deployments
- In-memory for testing and caching

Key Features:
- Multi-backend abstraction
- Automatic schema management
- Connection pooling
- Data encryption
- Backup and recovery
- Caching layer
"""

from .storage_manager import (
    StorageManager,
    StorageConfig,
    StorageRecord,
    StorageBackend,
    StorageStatus,
    create_storage_manager
)

__all__ = [
    'StorageManager',
    'StorageConfig', 
    'StorageRecord',
    'StorageBackend',
    'StorageStatus',
    'create_storage_manager'
]

# Version information
__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "Persistent storage layer with multi-backend support"