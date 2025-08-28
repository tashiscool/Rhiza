"""
Degradation Manager
==================

Manages graceful degradation of system components when resources
are limited or components fail within The Mesh network.
"""

import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """Levels of system degradation"""
    NORMAL = "normal"
    MINIMAL = "minimal"
    MODERATE = "moderate"  
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SystemStatus(Enum):
    """Overall system status"""
    OPTIMAL = "optimal"
    STABLE = "stable"
    DEGRADED = "degraded"
    IMPAIRED = "impaired"
    FAILING = "failing"
    OFFLINE = "offline"

class ComponentType(Enum):
    """Types of system components"""
    NETWORK = "network"
    STORAGE = "storage"
    COMPUTE = "compute"
    AI_MODEL = "ai_model"
    CONSENSUS = "consensus"
    TRUTH_VERIFICATION = "truth_verification"
    USER_INTERFACE = "user_interface"
    SECURITY = "security"

@dataclass
class DegradationPolicy:
    """Defines how a component should degrade"""
    component_id: str
    component_type: ComponentType
    priority: int  # 1-10, higher = more critical
    degradation_steps: List[str]
    fallback_options: List[str]
    minimum_function_threshold: float
    recovery_conditions: List[str]

@dataclass  
class SystemHealth:
    """Current system health snapshot"""
    timestamp: float
    overall_status: SystemStatus
    degradation_level: DegradationLevel
    component_health: Dict[str, float]  # component_id -> health_score (0-1)
    active_degradations: Set[str]
    resource_utilization: Dict[str, float]
    predicted_failures: List[str]

class DegradationManager:
    """Manages graceful system degradation and recovery"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.degradation_policies: Dict[str, DegradationPolicy] = {}
        self.component_health: Dict[str, float] = {}
        self.active_degradations: Dict[str, DegradationLevel] = {}
        self.degradation_history: List[SystemHealth] = []
        self.monitoring_active = False
        self.health_callbacks: List[Callable[[SystemHealth], None]] = []
        
    async def register_component(self, policy: DegradationPolicy) -> bool:
        """Register a component for degradation management"""
        try:
            self.degradation_policies[policy.component_id] = policy
            self.component_health[policy.component_id] = 1.0  # Start healthy
            
            logger.info(f"Registered component {policy.component_id} for degradation management")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {policy.component_id}: {e}")
            return False
    
    async def update_component_health(self, component_id: str, health_score: float) -> None:
        """Update health score for a component"""
        try:
            if component_id not in self.degradation_policies:
                logger.warning(f"Component {component_id} not registered")
                return
                
            previous_health = self.component_health.get(component_id, 1.0)
            self.component_health[component_id] = max(0.0, min(1.0, health_score))
            
            # Check if degradation action needed
            policy = self.degradation_policies[component_id]
            if health_score < policy.minimum_function_threshold:
                await self._trigger_degradation(component_id, health_score)
            elif component_id in self.active_degradations and health_score > policy.minimum_function_threshold * 1.2:
                await self._trigger_recovery(component_id, health_score)
                
            logger.debug(f"Updated {component_id} health: {previous_health:.2f} -> {health_score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to update component health for {component_id}: {e}")
    
    async def _trigger_degradation(self, component_id: str, health_score: float) -> None:
        """Trigger degradation for a component"""
        try:
            policy = self.degradation_policies[component_id]
            
            # Determine degradation level based on health
            if health_score > 0.7:
                level = DegradationLevel.MINIMAL
            elif health_score > 0.5:
                level = DegradationLevel.MODERATE
            elif health_score > 0.3:
                level = DegradationLevel.SIGNIFICANT
            elif health_score > 0.1:
                level = DegradationLevel.SEVERE
            else:
                level = DegradationLevel.CRITICAL
                
            # Apply degradation
            self.active_degradations[component_id] = level
            
            # Execute degradation steps
            for step in policy.degradation_steps:
                await self._execute_degradation_step(component_id, step, level)
                
            logger.warning(f"Applied {level.value} degradation to {component_id} (health: {health_score:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to trigger degradation for {component_id}: {e}")
    
    async def _trigger_recovery(self, component_id: str, health_score: float) -> None:
        """Trigger recovery for a component"""
        try:
            if component_id not in self.active_degradations:
                return
                
            previous_level = self.active_degradations[component_id]
            del self.active_degradations[component_id]
            
            # Execute recovery steps
            policy = self.degradation_policies[component_id]
            for condition in policy.recovery_conditions:
                await self._execute_recovery_step(component_id, condition)
                
            logger.info(f"Recovered {component_id} from {previous_level.value} degradation (health: {health_score:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to trigger recovery for {component_id}: {e}")
    
    async def _execute_degradation_step(self, component_id: str, step: str, level: DegradationLevel) -> None:
        """Execute a specific degradation step"""
        try:
            # This would integrate with actual system components
            # For demonstration, log the action
            logger.info(f"Executing degradation step for {component_id}: {step} (level: {level.value})")
            
            # Example degradation actions
            if "reduce_quality" in step:
                await self._reduce_quality(component_id, level)
            elif "disable_features" in step:
                await self._disable_features(component_id, level)
            elif "activate_fallback" in step:
                await self._activate_fallback(component_id)
                
        except Exception as e:
            logger.error(f"Failed to execute degradation step {step} for {component_id}: {e}")
    
    async def _execute_recovery_step(self, component_id: str, condition: str) -> None:
        """Execute a recovery step"""
        try:
            logger.info(f"Executing recovery step for {component_id}: {condition}")
            
            # Example recovery actions
            if "restore_quality" in condition:
                await self._restore_quality(component_id)
            elif "enable_features" in condition:
                await self._enable_features(component_id)
            elif "deactivate_fallback" in condition:
                await self._deactivate_fallback(component_id)
                
        except Exception as e:
            logger.error(f"Failed to execute recovery step {condition} for {component_id}: {e}")
    
    async def _reduce_quality(self, component_id: str, level: DegradationLevel) -> None:
        """Reduce quality/performance of component"""
        # Implementation would depend on component type
        logger.info(f"Reduced quality for {component_id} to {level.value}")
    
    async def _disable_features(self, component_id: str, level: DegradationLevel) -> None:
        """Disable non-essential features"""
        logger.info(f"Disabled features for {component_id} at {level.value}")
    
    async def _activate_fallback(self, component_id: str) -> None:
        """Activate fallback systems"""
        logger.info(f"Activated fallback for {component_id}")
    
    async def _restore_quality(self, component_id: str) -> None:
        """Restore full quality/performance"""
        logger.info(f"Restored quality for {component_id}")
    
    async def _enable_features(self, component_id: str) -> None:
        """Re-enable previously disabled features"""
        logger.info(f"Enabled features for {component_id}")
    
    async def _deactivate_fallback(self, component_id: str) -> None:
        """Deactivate fallback systems"""
        logger.info(f"Deactivated fallback for {component_id}")
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health snapshot"""
        try:
            # Calculate overall health
            if self.component_health:
                avg_health = sum(self.component_health.values()) / len(self.component_health)
            else:
                avg_health = 1.0
            
            # Determine system status
            if avg_health > 0.95:
                system_status = SystemStatus.OPTIMAL
                degradation_level = DegradationLevel.NORMAL
            elif avg_health > 0.85:
                system_status = SystemStatus.STABLE
                degradation_level = DegradationLevel.MINIMAL
            elif avg_health > 0.70:
                system_status = SystemStatus.DEGRADED
                degradation_level = DegradationLevel.MODERATE
            elif avg_health > 0.50:
                system_status = SystemStatus.IMPAIRED
                degradation_level = DegradationLevel.SIGNIFICANT
            elif avg_health > 0.20:
                system_status = SystemStatus.FAILING
                degradation_level = DegradationLevel.SEVERE
            else:
                system_status = SystemStatus.OFFLINE
                degradation_level = DegradationLevel.CRITICAL
            
            # Create health snapshot
            health = SystemHealth(
                timestamp=time.time(),
                overall_status=system_status,
                degradation_level=degradation_level,
                component_health=self.component_health.copy(),
                active_degradations=set(self.active_degradations.keys()),
                resource_utilization=await self._get_resource_utilization(),
                predicted_failures=await self._predict_failures()
            )
            
            # Store in history
            self.degradation_history.append(health)
            if len(self.degradation_history) > 1000:  # Keep last 1000 entries
                self.degradation_history = self.degradation_history[-1000:]
            
            # Notify callbacks
            for callback in self.health_callbacks:
                try:
                    callback(health)
                except Exception as e:
                    logger.error(f"Health callback failed: {e}")
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            raise
    
    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        # This would integrate with actual system monitoring
        return {
            "cpu": 0.45,
            "memory": 0.60,
            "disk": 0.30,
            "network": 0.25
        }
    
    async def _predict_failures(self) -> List[str]:
        """Predict potential component failures"""
        predictions = []
        
        # Simple prediction based on health trends
        for component_id, health in self.component_health.items():
            if health < 0.4:
                predictions.append(f"{component_id}_failure_risk")
                
        return predictions
    
    def add_health_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """Add callback for health updates"""
        self.health_callbacks.append(callback)
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous health monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self.get_system_health()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.monitoring_active = False
