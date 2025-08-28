"""
Node Isolation System - Quarantine and isolate compromised nodes

Provides mechanisms to isolate nodes that are detected as compromised,
malicious, or corrupted, preventing them from affecting the rest of the network.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    from mesh_core.network.connection_manager import ConnectionManager
    from mesh_core.network.message_router import MessageRouter
    from mesh_core.trust.trust_ledger import TrustLedger
    from mesh_core.config_manager import get_component_config
except ImportError:
    # Fallback to relative imports
    from ..network.connection_manager import ConnectionManager
    from ..network.message_router import MessageRouter
    from ..trust.trust_ledger import TrustLedger
    from ..config_manager import get_component_config
except ImportError:
    # Mock classes for testing
    class ConnectionManager:
        def __init__(self):
            pass
    
    class MessageRouter:
        def __init__(self):
            pass
    
    class TrustLedger:
        def __init__(self):
            pass
    
    def get_component_config(component):
        return {}

logger = logging.getLogger(__name__)

class IsolationLevel(Enum):
    """Levels of node isolation"""
    MONITORING = "monitoring"      # Enhanced monitoring only
    QUARANTINE = "quarantine"      # Restricted communications
    ISOLATION = "isolation"        # Complete network isolation
    BLACKLIST = "blacklist"        # Permanent exclusion

class IsolationReason(Enum):
    """Reasons for node isolation"""
    CORRUPTION_DETECTED = "corruption_detected"
    TRUST_DIVERGENCE = "trust_divergence"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    ATTACK_INDICATOR = "attack_indicator"
    ADMINISTRATIVE = "administrative"
    EMERGENCY = "emergency"

@dataclass
class IsolationRecord:
    """Record of a node isolation action"""
    isolation_id: str
    node_id: str
    isolation_level: IsolationLevel
    reason: IsolationReason
    reason_details: str
    timestamp: datetime
    initiated_by: str
    duration: Optional[timedelta] = None
    lifted_at: Optional[datetime] = None
    lifted_by: Optional[str] = None
    lift_reason: Optional[str] = None

@dataclass
class IsolationStatus:
    """Current isolation status of a node"""
    node_id: str
    is_isolated: bool
    isolation_level: Optional[IsolationLevel]
    isolation_record: Optional[IsolationRecord]
    restrictions: List[str]
    can_communicate: bool
    can_receive_messages: bool
    can_send_messages: bool
    can_access_trust_network: bool

class NodeIsolation:
    """Manages isolation and quarantine of compromised nodes"""
    
    def __init__(self, 
                 connection_manager: ConnectionManager,
                 message_router: MessageRouter,
                 trust_ledger: TrustLedger):
        
        self.connection_manager = connection_manager
        self.message_router = message_router
        self.trust_ledger = trust_ledger
        # Try to get config, fall back to defaults if not available
        try:
            self.config = get_component_config("mesh_immunity")
            self.isolation_enabled = self.config.get("isolation_enabled", True)
            self.quarantine_duration = timedelta(hours=self.config.get("quarantine_duration_hours", 24))
            self.isolation_duration = timedelta(hours=self.config.get("isolation_duration_hours", 72))
            self.max_isolated_nodes = self.config.get("max_isolated_nodes", 10)
        except Exception:
            # Use default values if config is not available
            self.config = {}
            self.isolation_enabled = True
            self.quarantine_duration = timedelta(hours=24)
            self.isolation_duration = timedelta(hours=72)
            self.max_isolated_nodes = 10
        
        # Active isolations
        self.active_isolations: Dict[str, IsolationRecord] = {}
        
        # Isolation history
        self.isolation_history: List[IsolationRecord] = []
        
        # Isolation coordination
        self.isolation_lock = asyncio.Lock()
        
        logger.info("Node isolation system initialized")
    
    async def isolate_node(self, 
                          node_id: str, 
                          isolation_level: IsolationLevel,
                          reason: IsolationReason,
                          reason_details: str,
                          initiated_by: str = "system") -> bool:
        """Isolate a node from the network"""
        try:
            if not self.isolation_enabled:
                logger.warning("Node isolation is disabled")
                return False
            
            async with self.isolation_lock:
                # Check if node is already isolated
                if node_id in self.active_isolations:
                    logger.warning(f"Node {node_id} is already isolated")
                    return False
                
                # Check isolation limits
                if len(self.active_isolations) >= self.max_isolated_nodes:
                    logger.error(f"Cannot isolate node {node_id}: maximum isolated nodes reached")
                    return False
                
                # Create isolation record
                isolation_record = IsolationRecord(
                    isolation_id=f"isolation_{node_id}_{int(time.time())}",
                    node_id=node_id,
                    isolation_level=isolation_level,
                    reason=reason,
                    reason_details=reason_details,
                    timestamp=datetime.now(),
                    initiated_by=initiated_by
                )
                
                # Apply isolation based on level
                isolation_success = await self._apply_isolation(node_id, isolation_level)
                if not isolation_success:
                    logger.error(f"Failed to apply isolation level {isolation_level.value} to node {node_id}")
                    return False
                
                # Record isolation
                self.active_isolations[node_id] = isolation_record
                self.isolation_history.append(isolation_record)
                
                # Update trust ledger
                await self._update_trust_for_isolation(node_id, isolation_level, reason)
                
                # Notify network of isolation
                await self._notify_network_isolation(node_id, isolation_level, reason)
                
                logger.info(f"Node {node_id} isolated at level {isolation_level.value} for reason: {reason.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error isolating node {node_id}: {e}")
            return False
    
    async def quarantine_node(self, 
                            node_id: str,
                            reason: IsolationReason,
                            reason_details: str,
                            initiated_by: str = "system") -> bool:
        """Quarantine a node (restricted communications)"""
        return await self.isolate_node(
            node_id, 
            IsolationLevel.QUARANTINE, 
            reason, 
            reason_details, 
            initiated_by
        )
    
    async def emergency_isolate_node(self, 
                                   node_id: str,
                                   reason: IsolationReason,
                                   reason_details: str,
                                   initiated_by: str = "system") -> bool:
        """Emergency isolation (complete network cut)"""
        return await self.isolate_node(
            node_id, 
            IsolationLevel.ISOLATION, 
            reason, 
            reason_details, 
            initiated_by
        )
    
    async def lift_isolation(self, 
                           node_id: str, 
                           lifted_by: str = "system",
                           lift_reason: str = "Administrative decision") -> bool:
        """Lift isolation from a node"""
        try:
            async with self.isolation_lock:
                if node_id not in self.active_isolations:
                    logger.warning(f"Node {node_id} is not currently isolated")
                    return False
                
                isolation_record = self.active_isolations[node_id]
                
                # Remove isolation restrictions
                removal_success = await self._remove_isolation(node_id, isolation_record.isolation_level)
                if not removal_success:
                    logger.error(f"Failed to remove isolation from node {node_id}")
                    return False
                
                # Update isolation record
                isolation_record.lifted_at = datetime.now()
                isolation_record.lifted_by = lifted_by
                isolation_record.lift_reason = lift_reason
                
                # Remove from active isolations
                del self.active_isolations[node_id]
                
                # Update trust ledger
                await self._update_trust_for_isolation_lift(node_id)
                
                # Notify network of isolation lift
                await self._notify_network_isolation_lift(node_id)
                
                logger.info(f"Isolation lifted from node {node_id} by {lifted_by}")
                return True
                
        except Exception as e:
            logger.error(f"Error lifting isolation from node {node_id}: {e}")
            return False
    
    async def get_isolation_status(self, node_id: str) -> IsolationStatus:
        """Get current isolation status of a node"""
        try:
            isolation_record = self.active_isolations.get(node_id)
            
            if not isolation_record:
                return IsolationStatus(
                    node_id=node_id,
                    is_isolated=False,
                    isolation_level=None,
                    isolation_record=None,
                    restrictions=[],
                    can_communicate=True,
                    can_receive_messages=True,
                    can_send_messages=True,
                    can_access_trust_network=True
                )
            
            # Determine restrictions based on isolation level
            restrictions = self._get_restrictions_for_level(isolation_record.isolation_level)
            
            return IsolationStatus(
                node_id=node_id,
                is_isolated=True,
                isolation_level=isolation_record.isolation_level,
                isolation_record=isolation_record,
                restrictions=restrictions,
                can_communicate=isolation_record.isolation_level == IsolationLevel.MONITORING,
                can_receive_messages=isolation_record.isolation_level in [IsolationLevel.MONITORING, IsolationLevel.QUARANTINE],
                can_send_messages=isolation_record.isolation_level == IsolationLevel.MONITORING,
                can_access_trust_network=isolation_record.isolation_level == IsolationLevel.MONITORING
            )
            
        except Exception as e:
            logger.error(f"Error getting isolation status for node {node_id}: {e}")
            # Return default status
            return IsolationStatus(
                node_id=node_id,
                is_isolated=False,
                isolation_level=None,
                isolation_record=None,
                restrictions=[],
                can_communicate=True,
                can_receive_messages=True,
                can_send_messages=True,
                can_access_trust_network=True
            )
    
    async def is_node_isolated(self, node_id: str) -> bool:
        """Check if a node is currently isolated"""
        return node_id in self.active_isolations
    
    async def get_isolated_nodes(self) -> List[str]:
        """Get list of currently isolated nodes"""
        return list(self.active_isolations.keys())
    
    async def get_isolation_summary(self) -> Dict[str, Any]:
        """Get summary of isolation activity"""
        if not self.isolation_history:
            return {"total_isolations": 0, "active_isolations": 0}
        
        # Count by isolation level
        level_counts = {}
        for isolation in self.isolation_history:
            level = isolation.isolation_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by reason
        reason_counts = {}
        for isolation in self.isolation_history:
            reason = isolation.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Count active isolations
        active_count = len(self.active_isolations)
        
        # Count lifted isolations
        lifted_count = sum(1 for i in self.isolation_history if i.lifted_at)
        
        return {
            "total_isolations": len(self.isolation_history),
            "active_isolations": active_count,
            "lifted_isolations": lifted_count,
            "level_breakdown": level_counts,
            "reason_breakdown": reason_counts
        }
    
    async def is_available(self) -> bool:
        """Check if node isolation system is available"""
        return self.isolation_enabled and self.connection_manager.is_available()
    
    async def _apply_isolation(self, node_id: str, isolation_level: IsolationLevel) -> bool:
        """Apply isolation restrictions to a node"""
        try:
            if isolation_level == IsolationLevel.MONITORING:
                return await self._apply_monitoring_isolation(node_id)
            elif isolation_level == IsolationLevel.QUARANTINE:
                return await self._apply_quarantine_isolation(node_id)
            elif isolation_level == IsolationLevel.ISOLATION:
                return await self._apply_full_isolation(node_id)
            elif isolation_level == IsolationLevel.BLACKLIST:
                return await self._apply_blacklist_isolation(node_id)
            else:
                logger.error(f"Unknown isolation level: {isolation_level}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying isolation level {isolation_level.value}: {e}")
            return False
    
    async def _apply_monitoring_isolation(self, node_id: str) -> bool:
        """Apply monitoring-level isolation (enhanced monitoring only)"""
        try:
            # Increase monitoring frequency
            await self.connection_manager.set_monitoring_level(node_id, "enhanced")
            
            # Log all communications
            await self.message_router.set_logging_level(node_id, "detailed")
            
            logger.info(f"Applied monitoring isolation to node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying monitoring isolation: {e}")
            return False
    
    async def _apply_quarantine_isolation(self, node_id: str) -> bool:
        """Apply quarantine isolation (restricted communications)"""
        try:
            # Apply monitoring isolation first
            await self._apply_monitoring_isolation(node_id)
            
            # Restrict message routing
            await self.message_router.set_node_restrictions(node_id, {
                "max_message_size": 1024,  # 1KB limit
                "allowed_message_types": ["essential", "health_check"],
                "rate_limit": 1,  # 1 message per minute
                "require_approval": True
            })
            
            # Restrict connection types
            await self.connection_manager.set_node_restrictions(node_id, {
                "max_connections": 2,
                "allowed_connection_types": ["health_check", "essential"],
                "connection_timeout": 30  # 30 seconds
            })
            
            logger.info(f"Applied quarantine isolation to node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying quarantine isolation: {e}")
            return False
    
    async def _apply_full_isolation(self, node_id: str) -> bool:
        """Apply full isolation (complete network cut)"""
        try:
            # Cut all connections
            await self.connection_manager.disconnect_node(node_id, reason="isolation")
            
            # Block all message routing
            await self.message_router.block_node(node_id, reason="isolation")
            
            # Remove from trust network
            await self.trust_ledger.suspend_node_trust(node_id, reason="isolation")
            
            logger.info(f"Applied full isolation to node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying full isolation: {e}")
            return False
    
    async def _apply_blacklist_isolation(self, node_id: str) -> bool:
        """Apply blacklist isolation (permanent exclusion)"""
        try:
            # Apply full isolation first
            await self._apply_full_isolation(node_id)
            
            # Add to permanent blacklist
            await self.trust_ledger.blacklist_node(node_id, reason="permanent_exclusion")
            
            # Remove from all routing tables
            await self.message_router.permanently_block_node(node_id)
            
            logger.info(f"Applied blacklist isolation to node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying blacklist isolation: {e}")
            return False
    
    async def _remove_isolation(self, node_id: str, isolation_level: IsolationLevel) -> bool:
        """Remove isolation restrictions from a node"""
        try:
            if isolation_level == IsolationLevel.MONITORING:
                return await self._remove_monitoring_isolation(node_id)
            elif isolation_level == IsolationLevel.QUARANTINE:
                return await self._remove_quarantine_isolation(node_id)
            elif isolation_level == IsolationLevel.ISOLATION:
                return await self._remove_full_isolation(node_id)
            elif isolation_level == IsolationLevel.BLACKLIST:
                return await self._remove_blacklist_isolation(node_id)
            else:
                logger.error(f"Unknown isolation level: {isolation_level}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing isolation level {isolation_level.value}: {e}")
            return False
    
    async def _remove_monitoring_isolation(self, node_id: str) -> bool:
        """Remove monitoring-level isolation"""
        try:
            # Reset monitoring level
            await self.connection_manager.set_monitoring_level(node_id, "normal")
            
            # Reset logging level
            await self.message_router.set_logging_level(node_id, "normal")
            
            logger.info(f"Removed monitoring isolation from node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing monitoring isolation: {e}")
            return False
    
    async def _remove_quarantine_isolation(self, node_id: str) -> bool:
        """Remove quarantine isolation"""
        try:
            # Remove message routing restrictions
            await self.message_router.clear_node_restrictions(node_id)
            
            # Remove connection restrictions
            await self.connection_manager.clear_node_restrictions(node_id)
            
            # Remove monitoring isolation
            await self._remove_monitoring_isolation(node_id)
            
            logger.info(f"Removed quarantine isolation from node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing quarantine isolation: {e}")
            return False
    
    async def _remove_full_isolation(self, node_id: str) -> bool:
        """Remove full isolation"""
        try:
            # Restore connections
            await self.connection_manager.allow_node_connections(node_id)
            
            # Restore message routing
            await self.message_router.unblock_node(node_id)
            
            # Restore trust network access
            await self.trust_ledger.restore_node_trust(node_id)
            
            logger.info(f"Removed full isolation from node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing full isolation: {e}")
            return False
    
    async def _remove_blacklist_isolation(self, node_id: str) -> bool:
        """Remove blacklist isolation (requires special handling)"""
        try:
            # This is a complex operation that requires administrative approval
            # For now, just log the attempt
            logger.warning(f"Attempt to remove blacklist isolation from node {node_id} - requires administrative approval")
            
            # In a real implementation, this would trigger an administrative review process
            return False
            
        except Exception as e:
            logger.error(f"Error removing blacklist isolation: {e}")
            return False
    
    def _get_restrictions_for_level(self, isolation_level: IsolationLevel) -> List[str]:
        """Get list of restrictions for an isolation level"""
        if isolation_level == IsolationLevel.MONITORING:
            return ["enhanced_monitoring", "detailed_logging"]
        elif isolation_level == IsolationLevel.QUARANTINE:
            return ["enhanced_monitoring", "detailed_logging", "restricted_communications", "rate_limiting", "message_approval"]
        elif isolation_level == IsolationLevel.ISOLATION:
            return ["enhanced_monitoring", "detailed_logging", "restricted_communications", "rate_limiting", "message_approval", "connection_restrictions", "trust_network_suspension"]
        elif isolation_level == IsolationLevel.BLACKLIST:
            return ["enhanced_monitoring", "detailed_logging", "restricted_communications", "rate_limiting", "message_approval", "connection_restrictions", "trust_network_suspension", "permanent_exclusion", "routing_block"]
        else:
            return []
    
    async def _update_trust_for_isolation(self, node_id: str, isolation_level: IsolationLevel, reason: IsolationReason):
        """Update trust ledger for node isolation"""
        try:
            # Mark node as isolated in trust system
            await self.trust_ledger.mark_node_isolated(node_id, {
                "isolation_level": isolation_level.value,
                "reason": reason.value,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating trust for isolation: {e}")
    
    async def _update_trust_for_isolation_lift(self, node_id: str):
        """Update trust ledger for isolation lift"""
        try:
            # Mark node as no longer isolated
            await self.trust_ledger.mark_node_not_isolated(node_id)
            
        except Exception as e:
            logger.error(f"Error updating trust for isolation lift: {e}")
    
    async def _notify_network_isolation(self, node_id: str, isolation_level: IsolationLevel, reason: IsolationReason):
        """Notify network of node isolation"""
        try:
            # Broadcast isolation notification to network
            notification = {
                "type": "node_isolation",
                "node_id": node_id,
                "isolation_level": isolation_level.value,
                "reason": reason.value,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.message_router.broadcast_notification(notification)
            
        except Exception as e:
            logger.error(f"Error notifying network of isolation: {e}")
    
    async def _notify_network_isolation_lift(self, node_id: str):
        """Notify network of isolation lift"""
        try:
            # Broadcast isolation lift notification to network
            notification = {
                "type": "node_isolation_lift",
                "node_id": node_id,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.message_router.broadcast_notification(notification)
            
        except Exception as e:
            logger.error(f"Error notifying network of isolation lift: {e}")
    
    async def cleanup_expired_isolations(self):
        """Clean up expired isolations based on duration limits"""
        try:
            current_time = datetime.now()
            expired_nodes = []
            
            for node_id, isolation_record in self.active_isolations.items():
                if isolation_record.isolation_level == IsolationLevel.QUARANTINE:
                    if current_time - isolation_record.timestamp > self.quarantine_duration:
                        expired_nodes.append(node_id)
                elif isolation_record.isolation_level == IsolationLevel.ISOLATION:
                    if current_time - isolation_record.timestamp > self.isolation_duration:
                        expired_nodes.append(node_id)
            
            # Lift expired isolations
            for node_id in expired_nodes:
                await self.lift_isolation(
                    node_id, 
                    lifted_by="system", 
                    lift_reason="Automatic expiration"
                )
                
        except Exception as e:
            logger.error(f"Error cleaning up expired isolations: {e}")
    
    def get_isolation_history(self, node_id: Optional[str] = None) -> List[IsolationRecord]:
        """Get isolation history"""
        if node_id:
            return [i for i in self.isolation_history if i.node_id == node_id]
        return self.isolation_history.copy()
