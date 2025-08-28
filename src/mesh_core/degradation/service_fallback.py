"""
Service Fallback
================

Provides fallback mechanisms for critical services when primary
systems fail or are unavailable within The Mesh network.
"""

import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Types of services that can have fallbacks"""
    TRUTH_VERIFICATION = "truth_verification"
    CONSENSUS = "consensus"
    NETWORK_DISCOVERY = "network_discovery"
    DATA_STORAGE = "data_storage"
    AI_INFERENCE = "ai_inference"
    USER_INTERFACE = "user_interface"
    AUTHENTICATION = "authentication"
    COMMUNICATION = "communication"

class FallbackStrategy(Enum):
    """Strategies for service fallback"""
    IMMEDIATE = "immediate"        # Switch immediately on failure
    GRADUAL = "gradual"           # Gradually transition load
    CONDITIONAL = "conditional"    # Switch based on conditions
    HYBRID = "hybrid"             # Mix of primary and fallback
    CIRCUIT_BREAKER = "circuit_breaker"  # With failure threshold

class FallbackStatus(Enum):
    """Status of fallback services"""
    INACTIVE = "inactive"
    STANDBY = "standby"
    PARTIALLY_ACTIVE = "partially_active"
    FULLY_ACTIVE = "fully_active"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class FallbackService:
    """Represents a fallback service configuration"""
    service_id: str
    service_type: ServiceType
    primary_endpoint: str
    fallback_endpoints: List[str]
    strategy: FallbackStrategy
    health_check_interval: int
    failure_threshold: int
    recovery_threshold: int
    timeout_seconds: float
    quality_degradation: float  # 0-1, how much quality is lost
    
@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_id: str
    is_healthy: bool
    response_time: float
    success_rate: float
    last_check: float
    consecutive_failures: int
    consecutive_successes: int

class ServiceFallback:
    """Manages fallback mechanisms for critical services"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.services: Dict[str, FallbackService] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.active_fallbacks: Dict[str, str] = {}  # service_id -> active_endpoint
        self.monitoring_active = False
        self.circuit_breakers: Dict[str, bool] = {}  # service_id -> is_open
        
    async def register_service(self, service: FallbackService) -> bool:
        """Register a service for fallback management"""
        try:
            self.services[service.service_id] = service
            
            # Initialize health tracking
            self.service_health[service.service_id] = ServiceHealth(
                service_id=service.service_id,
                is_healthy=True,
                response_time=0.0,
                success_rate=1.0,
                last_check=time.time(),
                consecutive_failures=0,
                consecutive_successes=0
            )
            
            # Initialize with primary endpoint
            self.active_fallbacks[service.service_id] = service.primary_endpoint
            self.circuit_breakers[service.service_id] = False
            
            logger.info(f"Registered service {service.service_id} for fallback management")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False
    
    async def check_service_health(self, service_id: str) -> bool:
        """Check health of a specific service"""
        if service_id not in self.services:
            logger.warning(f"Service {service_id} not registered")
            return False
            
        try:
            service = self.services[service_id]
            current_endpoint = self.active_fallbacks[service_id]
            
            # Perform health check (placeholder implementation)
            start_time = time.time()
            is_healthy = await self._perform_health_check(current_endpoint, service.timeout_seconds)
            response_time = time.time() - start_time
            
            # Update health status
            health = self.service_health[service_id]
            health.last_check = time.time()
            health.response_time = response_time
            
            if is_healthy:
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.success_rate = min(1.0, health.success_rate + 0.1)
            else:
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.success_rate = max(0.0, health.success_rate - 0.1)
            
            health.is_healthy = is_healthy
            
            # Check if fallback is needed
            if not is_healthy and health.consecutive_failures >= service.failure_threshold:
                await self._trigger_fallback(service_id)
            elif is_healthy and health.consecutive_successes >= service.recovery_threshold:
                await self._trigger_recovery(service_id)
                
            return is_healthy
            
        except Exception as e:
            logger.error(f"Failed to check health for service {service_id}: {e}")
            return False
    
    async def _perform_health_check(self, endpoint: str, timeout: float) -> bool:
        """Perform actual health check on an endpoint"""
        try:
            # This would perform actual service health check
            # For demonstration, simulate based on endpoint
            await asyncio.sleep(0.1)  # Simulate network call
            
            # Simple simulation: endpoints with "backup" in name are less reliable
            if "backup" in endpoint.lower():
                import random
                return random.random() > 0.2  # 80% success rate for backups
            else:
                import random
                return random.random() > 0.1  # 90% success rate for primary
                
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for {endpoint}")
            return False
        except Exception as e:
            logger.error(f"Health check failed for {endpoint}: {e}")
            return False
    
    async def _trigger_fallback(self, service_id: str) -> None:
        """Trigger fallback for a service"""
        try:
            service = self.services[service_id]
            current_endpoint = self.active_fallbacks[service_id]
            
            # Find next available fallback
            available_fallbacks = [
                endpoint for endpoint in service.fallback_endpoints
                if endpoint != current_endpoint
            ]
            
            if not available_fallbacks:
                logger.error(f"No fallback available for service {service_id}")
                self.circuit_breakers[service_id] = True
                return
            
            # Select fallback based on strategy
            new_endpoint = available_fallbacks[0]  # Simple: use first available
            
            # Apply fallback strategy
            if service.strategy == FallbackStrategy.IMMEDIATE:
                await self._switch_endpoint(service_id, new_endpoint)
            elif service.strategy == FallbackStrategy.CIRCUIT_BREAKER:
                self.circuit_breakers[service_id] = True
                await self._switch_endpoint(service_id, new_endpoint)
            else:
                # Other strategies would be implemented here
                await self._switch_endpoint(service_id, new_endpoint)
            
            logger.warning(f"Activated fallback for {service_id}: {current_endpoint} -> {new_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to trigger fallback for service {service_id}: {e}")
    
    async def _trigger_recovery(self, service_id: str) -> None:
        """Trigger recovery to primary service"""
        try:
            service = self.services[service_id]
            current_endpoint = self.active_fallbacks[service_id]
            
            # Only recover if we're on a fallback
            if current_endpoint == service.primary_endpoint:
                return
            
            # Check if primary is healthy
            if await self._perform_health_check(service.primary_endpoint, service.timeout_seconds):
                await self._switch_endpoint(service_id, service.primary_endpoint)
                self.circuit_breakers[service_id] = False
                logger.info(f"Recovered service {service_id} to primary endpoint")
                
        except Exception as e:
            logger.error(f"Failed to trigger recovery for service {service_id}: {e}")
    
    async def _switch_endpoint(self, service_id: str, new_endpoint: str) -> None:
        """Switch service to new endpoint"""
        try:
            old_endpoint = self.active_fallbacks[service_id]
            self.active_fallbacks[service_id] = new_endpoint
            
            # Reset health tracking for new endpoint
            self.service_health[service_id].consecutive_failures = 0
            self.service_health[service_id].consecutive_successes = 0
            
            logger.info(f"Switched {service_id} from {old_endpoint} to {new_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to switch endpoint for service {service_id}: {e}")
    
    async def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """Get current status of a service"""
        if service_id not in self.services:
            return {"error": "Service not registered"}
        
        try:
            service = self.services[service_id]
            health = self.service_health[service_id]
            current_endpoint = self.active_fallbacks[service_id]
            is_on_fallback = current_endpoint != service.primary_endpoint
            
            return {
                "service_id": service_id,
                "service_type": service.service_type.value,
                "current_endpoint": current_endpoint,
                "is_on_fallback": is_on_fallback,
                "is_healthy": health.is_healthy,
                "response_time": health.response_time,
                "success_rate": health.success_rate,
                "consecutive_failures": health.consecutive_failures,
                "consecutive_successes": health.consecutive_successes,
                "circuit_breaker_open": self.circuit_breakers.get(service_id, False),
                "last_check": health.last_check,
                "quality_degradation": service.quality_degradation if is_on_fallback else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for service {service_id}: {e}")
            return {"error": str(e)}
    
    async def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered services"""
        try:
            status = {}
            for service_id in self.services.keys():
                status[service_id] = await self.get_service_status(service_id)
            return status
            
        except Exception as e:
            logger.error(f"Failed to get all services status: {e}")
            return {"error": str(e)}
    
    async def force_fallback(self, service_id: str, fallback_endpoint: Optional[str] = None) -> bool:
        """Manually force fallback for a service"""
        if service_id not in self.services:
            logger.warning(f"Service {service_id} not registered")
            return False
        
        try:
            service = self.services[service_id]
            
            if fallback_endpoint:
                if fallback_endpoint not in service.fallback_endpoints:
                    logger.error(f"Endpoint {fallback_endpoint} not in fallback list for {service_id}")
                    return False
                target_endpoint = fallback_endpoint
            else:
                # Use first available fallback
                if not service.fallback_endpoints:
                    logger.error(f"No fallback endpoints available for {service_id}")
                    return False
                target_endpoint = service.fallback_endpoints[0]
            
            await self._switch_endpoint(service_id, target_endpoint)
            logger.info(f"Manually triggered fallback for {service_id} to {target_endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to force fallback for service {service_id}: {e}")
            return False
    
    async def force_recovery(self, service_id: str) -> bool:
        """Manually force recovery to primary service"""
        if service_id not in self.services:
            logger.warning(f"Service {service_id} not registered")
            return False
        
        try:
            service = self.services[service_id]
            await self._switch_endpoint(service_id, service.primary_endpoint)
            self.circuit_breakers[service_id] = False
            logger.info(f"Manually triggered recovery for {service_id} to primary")
            return True
            
        except Exception as e:
            logger.error(f"Failed to force recovery for service {service_id}: {e}")
            return False
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous service monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Check health of all services
                for service_id in self.services.keys():
                    await self.check_service_health(service_id)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self) -> None:
        """Stop service monitoring"""
        self.monitoring_active = False