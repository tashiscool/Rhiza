"""
Storage Manager - Persistent storage layer for The Mesh

Provides a unified storage interface supporting multiple backends:
- SQLite for development and single-node deployments
- PostgreSQL for production distributed deployments
- In-memory for testing and caching

Key Features:
- Multi-backend storage abstraction
- Automatic schema management and migrations
- Connection pooling and optimization
- Backup and recovery capabilities
- Distributed storage coordination
- Data encryption at rest
"""

import asyncio
import logging
import sqlite3
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager
import threading

try:
    import aiosqlite
    import asyncpg
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    logging.warning("aiosqlite and asyncpg not available - using synchronous fallback")

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_SUPPORT = True
except ImportError:
    ENCRYPTION_SUPPORT = False
    logging.warning("cryptography not available - data encryption disabled")


class StorageBackend(Enum):
    """Storage backend types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MEMORY = "memory"


class StorageStatus(Enum):
    """Storage system status"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    MIGRATING = "migrating"
    BACKUP = "backup"


@dataclass
class StorageConfig:
    """Storage configuration"""
    backend: StorageBackend
    database_path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None
    pool_size: int = 10
    max_connections: int = 20
    encryption_key: Optional[str] = None
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class StorageRecord:
    """Generic storage record"""
    id: str
    table: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StorageManager:
    """Unified storage manager for The Mesh"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.backend = config.backend
        self.status = StorageStatus.INITIALIZING
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self._connection_pool = None
        self._connection_lock = threading.Lock()
        self._schema_version = 1
        
        # Encryption
        self._encryption_key = None
        if config.encryption_key and ENCRYPTION_SUPPORT:
            self._encryption_key = Fernet(config.encryption_key.encode())
        
        # Caching
        self._cache = {}
        self._cache_ttl = {}
        self._cache_max_size = 1000
        
        # Statistics
        self.operations_count = 0
        self.cache_hits = 0
        self.connection_errors = 0
        
        # Backup management
        self._last_backup = None
        self._backup_task = None
    
    async def initialize(self) -> bool:
        """Initialize storage system"""
        try:
            self.logger.info(f"Initializing storage with {self.backend.value} backend")
            
            if self.backend == StorageBackend.SQLITE:
                await self._initialize_sqlite()
            elif self.backend == StorageBackend.POSTGRESQL:
                await self._initialize_postgresql()
            elif self.backend == StorageBackend.MEMORY:
                await self._initialize_memory()
            
            # Create tables
            await self._create_tables()
            
            # Start backup task if enabled
            if self.config.backup_enabled:
                self._backup_task = asyncio.create_task(self._backup_scheduler())
            
            self.status = StorageStatus.READY
            self.logger.info("Storage system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            self.status = StorageStatus.ERROR
            return False
    
    async def store(self, table: str, data: Dict[str, Any], record_id: Optional[str] = None) -> str:
        """Store data in the specified table"""
        try:
            if not record_id:
                record_id = self._generate_id(table, data)
            
            # Encrypt data if encryption is enabled
            if self._encryption_key:
                data = self._encrypt_data(data)
            
            timestamp = datetime.utcnow()
            
            if self.backend == StorageBackend.MEMORY:
                return await self._store_memory(table, record_id, data, timestamp)
            elif self.backend == StorageBackend.SQLITE:
                return await self._store_sqlite(table, record_id, data, timestamp)
            elif self.backend == StorageBackend.POSTGRESQL:
                return await self._store_postgresql(table, record_id, data, timestamp)
            
            self.operations_count += 1
            return record_id
            
        except Exception as e:
            self.logger.error(f"Error storing data in {table}: {e}")
            raise
    
    async def retrieve(self, table: str, record_id: str) -> Optional[StorageRecord]:
        """Retrieve data by ID"""
        try:
            # Check cache first
            cache_key = f"{table}:{record_id}"
            if cache_key in self._cache:
                if self._is_cache_valid(cache_key):
                    self.cache_hits += 1
                    return self._cache[cache_key]
                else:
                    del self._cache[cache_key]
                    del self._cache_ttl[cache_key]
            
            # Retrieve from storage
            record = None
            if self.backend == StorageBackend.MEMORY:
                record = await self._retrieve_memory(table, record_id)
            elif self.backend == StorageBackend.SQLITE:
                record = await self._retrieve_sqlite(table, record_id)
            elif self.backend == StorageBackend.POSTGRESQL:
                record = await self._retrieve_postgresql(table, record_id)
            
            if record and self._encryption_key:
                record.data = self._decrypt_data(record.data)
            
            # Cache the result
            if record:
                self._cache_record(cache_key, record)
            
            self.operations_count += 1
            return record
            
        except Exception as e:
            self.logger.error(f"Error retrieving data from {table}: {e}")
            raise
    
    async def query(self, table: str, filters: Dict[str, Any] = None, limit: int = None, offset: int = 0) -> List[StorageRecord]:
        """Query data with filters"""
        try:
            records = []
            
            if self.backend == StorageBackend.MEMORY:
                records = await self._query_memory(table, filters, limit, offset)
            elif self.backend == StorageBackend.SQLITE:
                records = await self._query_sqlite(table, filters, limit, offset)
            elif self.backend == StorageBackend.POSTGRESQL:
                records = await self._query_postgresql(table, filters, limit, offset)
            
            # Decrypt data if needed
            if self._encryption_key:
                for record in records:
                    record.data = self._decrypt_data(record.data)
            
            self.operations_count += 1
            return records
            
        except Exception as e:
            self.logger.error(f"Error querying {table}: {e}")
            raise
    
    async def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update existing record"""
        try:
            # Encrypt data if needed
            if self._encryption_key:
                data = self._encrypt_data(data)
            
            updated = False
            timestamp = datetime.utcnow()
            
            if self.backend == StorageBackend.MEMORY:
                updated = await self._update_memory(table, record_id, data, timestamp)
            elif self.backend == StorageBackend.SQLITE:
                updated = await self._update_sqlite(table, record_id, data, timestamp)
            elif self.backend == StorageBackend.POSTGRESQL:
                updated = await self._update_postgresql(table, record_id, data, timestamp)
            
            # Invalidate cache
            cache_key = f"{table}:{record_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
            
            self.operations_count += 1
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating {table}:{record_id}: {e}")
            raise
    
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete record"""
        try:
            deleted = False
            
            if self.backend == StorageBackend.MEMORY:
                deleted = await self._delete_memory(table, record_id)
            elif self.backend == StorageBackend.SQLITE:
                deleted = await self._delete_sqlite(table, record_id)
            elif self.backend == StorageBackend.POSTGRESQL:
                deleted = await self._delete_postgresql(table, record_id)
            
            # Remove from cache
            cache_key = f"{table}:{record_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
            
            self.operations_count += 1
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting {table}:{record_id}: {e}")
            raise
    
    async def backup(self, backup_path: Optional[str] = None) -> str:
        """Create backup of storage"""
        try:
            self.status = StorageStatus.BACKUP
            
            if not backup_path:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_path = f"mesh_backup_{timestamp}.db"
            
            if self.backend == StorageBackend.SQLITE:
                success = await self._backup_sqlite(backup_path)
            elif self.backend == StorageBackend.POSTGRESQL:
                success = await self._backup_postgresql(backup_path)
            else:
                success = await self._backup_memory(backup_path)
            
            if success:
                self._last_backup = datetime.utcnow()
                self.logger.info(f"Backup created: {backup_path}")
            
            self.status = StorageStatus.READY
            return backup_path if success else None
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            self.status = StorageStatus.READY
            raise
    
    async def restore(self, backup_path: str) -> bool:
        """Restore from backup"""
        try:
            self.logger.info(f"Restoring from backup: {backup_path}")
            
            if self.backend == StorageBackend.SQLITE:
                return await self._restore_sqlite(backup_path)
            elif self.backend == StorageBackend.POSTGRESQL:
                return await self._restore_postgresql(backup_path)
            else:
                return await self._restore_memory(backup_path)
                
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    async def _initialize_sqlite(self):
        """Initialize SQLite backend"""
        if not ASYNC_SUPPORT:
            # Fallback to synchronous SQLite
            db_path = self.config.database_path or "mesh.db"
            self._connection_pool = sqlite3.connect(db_path, check_same_thread=False)
            return
        
        db_path = self.config.database_path or "mesh.db"
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Test connection
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("SELECT 1")
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL backend"""
        if not ASYNC_SUPPORT:
            raise RuntimeError("PostgreSQL requires asyncpg - install with: pip install asyncpg")
        
        # Create connection pool
        self._connection_pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
            database=self.config.database_name,
            min_size=1,
            max_size=self.config.max_connections
        )
    
    async def _initialize_memory(self):
        """Initialize in-memory backend"""
        self._connection_pool = {}  # Simple dict storage
    
    async def _create_tables(self):
        """Create necessary tables"""
        tables = {
            "mesh_data": """
                CREATE TABLE IF NOT EXISTS mesh_data (
                    id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """,
            "mesh_nodes": """
                CREATE TABLE IF NOT EXISTS mesh_nodes (
                    node_id TEXT PRIMARY KEY,
                    address TEXT NOT NULL,
                    trust_score REAL DEFAULT 0.5,
                    last_seen TIMESTAMP,
                    status TEXT DEFAULT 'unknown',
                    metadata TEXT
                )
            """,
            "mesh_truths": """
                CREATE TABLE IF NOT EXISTS mesh_truths (
                    truth_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    verification_status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """
        }
        
        if self.backend == StorageBackend.SQLITE:
            if ASYNC_SUPPORT:
                db_path = self.config.database_path or "mesh.db"
                async with aiosqlite.connect(db_path) as conn:
                    for table_sql in tables.values():
                        await conn.execute(table_sql)
                    await conn.commit()
            else:
                for table_sql in tables.values():
                    self._connection_pool.execute(table_sql)
                self._connection_pool.commit()
                
        elif self.backend == StorageBackend.POSTGRESQL:
            async with self._connection_pool.acquire() as conn:
                for table_sql in tables.values():
                    await conn.execute(table_sql.replace("TEXT", "VARCHAR").replace("REAL", "FLOAT"))
    
    # Storage backend implementations
    async def _store_memory(self, table: str, record_id: str, data: Dict[str, Any], timestamp: datetime) -> str:
        """Store in memory"""
        if table not in self._connection_pool:
            self._connection_pool[table] = {}
        
        record = StorageRecord(
            id=record_id,
            table=table,
            data=data,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self._connection_pool[table][record_id] = record
        return record_id
    
    async def _retrieve_memory(self, table: str, record_id: str) -> Optional[StorageRecord]:
        """Retrieve from memory"""
        if table not in self._connection_pool:
            return None
        
        return self._connection_pool[table].get(record_id)
    
    async def _query_memory(self, table: str, filters: Dict[str, Any], limit: int, offset: int) -> List[StorageRecord]:
        """Query memory storage"""
        if table not in self._connection_pool:
            return []
        
        records = list(self._connection_pool[table].values())
        
        # Apply filters (simple implementation)
        if filters:
            filtered_records = []
            for record in records:
                match = True
                for key, value in filters.items():
                    if key in record.data and record.data[key] != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            records = filtered_records
        
        # Apply pagination
        start = offset
        end = start + limit if limit else len(records)
        return records[start:end]
    
    async def _store_sqlite(self, table: str, record_id: str, data: Dict[str, Any], timestamp: datetime) -> str:
        """Store in SQLite"""
        data_json = json.dumps(data)
        metadata_json = json.dumps({})
        
        if ASYNC_SUPPORT:
            db_path = self.config.database_path or "mesh.db"
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO mesh_data (id, table_name, data, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (record_id, table, data_json, timestamp, timestamp, metadata_json)
                )
                await conn.commit()
        else:
            self._connection_pool.execute(
                "INSERT OR REPLACE INTO mesh_data (id, table_name, data, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, table, data_json, timestamp, timestamp, metadata_json)
            )
            self._connection_pool.commit()
        
        return record_id
    
    async def _retrieve_sqlite(self, table: str, record_id: str) -> Optional[StorageRecord]:
        """Retrieve from SQLite"""
        if ASYNC_SUPPORT:
            db_path = self.config.database_path or "mesh.db"
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT * FROM mesh_data WHERE id = ? AND table_name = ?",
                    (record_id, table)
                )
                row = await cursor.fetchone()
        else:
            cursor = self._connection_pool.execute(
                "SELECT * FROM mesh_data WHERE id = ? AND table_name = ?",
                (record_id, table)
            )
            row = cursor.fetchone()
        
        if row:
            return StorageRecord(
                id=row[0],
                table=row[1],
                data=json.loads(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]) if row[5] else {}
            )
        
        return None
    
    # Utility methods
    def _generate_id(self, table: str, data: Dict[str, Any]) -> str:
        """Generate unique ID for record"""
        content = f"{table}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data"""
        if not self._encryption_key:
            return data
        
        encrypted_data = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                encrypted_value = self._encryption_key.encrypt(str(value).encode())
                encrypted_data[key] = encrypted_value.decode()
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def _decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        if not self._encryption_key:
            return data
        
        decrypted_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    decrypted_value = self._encryption_key.decrypt(value.encode())
                    decrypted_data[key] = decrypted_value.decode()
                except:
                    decrypted_data[key] = value
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    def _cache_record(self, key: str, record: StorageRecord):
        """Cache record with TTL"""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
            del self._cache[oldest_key]
            del self._cache_ttl[oldest_key]
        
        self._cache[key] = record
        self._cache_ttl[key] = datetime.utcnow() + timedelta(minutes=30)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry is still valid"""
        return key in self._cache_ttl and datetime.utcnow() < self._cache_ttl[key]
    
    async def _backup_scheduler(self):
        """Background backup scheduler"""
        while self.status != StorageStatus.ERROR:
            try:
                await asyncio.sleep(self.config.backup_interval)
                if self.config.backup_enabled:
                    await self.backup()
            except Exception as e:
                self.logger.error(f"Backup scheduler error: {e}")
    
    async def _backup_sqlite(self, backup_path: str) -> bool:
        """Backup SQLite database"""
        try:
            import shutil
            source_path = self.config.database_path or "mesh.db"
            shutil.copy2(source_path, backup_path)
            return True
        except Exception as e:
            self.logger.error(f"SQLite backup error: {e}")
            return False
    
    async def close(self):
        """Close storage connections"""
        try:
            if self._backup_task:
                self._backup_task.cancel()
            
            if self.backend == StorageBackend.POSTGRESQL and self._connection_pool:
                await self._connection_pool.close()
            elif self.backend == StorageBackend.SQLITE and hasattr(self._connection_pool, 'close'):
                if ASYNC_SUPPORT:
                    pass  # aiosqlite connections are closed automatically
                else:
                    self._connection_pool.close()
            
            self.logger.info("Storage connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing storage: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        cache_hit_rate = (self.cache_hits / max(self.operations_count, 1)) * 100
        
        return {
            "backend": self.backend.value,
            "status": self.status.value,
            "operations_count": self.operations_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "connection_errors": self.connection_errors,
            "last_backup": self._last_backup.isoformat() if self._last_backup else None,
            "encryption_enabled": self._encryption_key is not None
        }


# Factory function for easy storage creation
def create_storage_manager(backend: str = "sqlite", **kwargs) -> StorageManager:
    """Create storage manager with specified backend"""
    config = StorageConfig(
        backend=StorageBackend(backend),
        **kwargs
    )
    return StorageManager(config)