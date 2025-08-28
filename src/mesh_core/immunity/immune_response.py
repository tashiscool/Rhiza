"""
Immune Response System - Automatic threat response and recovery

Automatically responds to detected corruption and threats by implementing
appropriate countermeasures while maintaining system integrity and availability.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    from mesh_core.immunity.corruption_detector import CorruptionDetector, CorruptionDetection, CorruptionLevel
    from mesh_core.immunity.trust_divergence import TrustDivergenceMonitor, TrustDivergence, DivergenceSeverity
    from mesh_core.immunity.node_isolation import NodeIsolation
except ImportError:
    # Fallback to relative imports
    from .corruption_detector import CorruptionDetector, CorruptionDetection, CorruptionLevel
    from .trust_divergence import TrustDivergenceMonitor, TrustDivergence, DivergenceSeverity
    from .node_isolation import NodeIsolation
try:
    from ..trust.trust_ledger import TrustLedger
    from ..network.network_health import NetworkHealth as NetworkHealthMonitor
    from ..config_manager import get_component_config
except ImportError:
    # Mock classes for testing
    class TrustLedger:
        def __init__(self):
            pass
    
    class NetworkHealthMonitor:
        def __init__(self):
            pass
    
    def get_component_config(component):
        return {}

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Types of immune responses"""
    MONITORING = "monitoring"
    WARNING = "warning"
    QUARANTINE = "quarantine"
    ISOLATION = "isolation"
    RECOVERY = "recovery"
    PREVENTION = "prevention"

class ResponsePriority(Enum):
    """Priority levels for immune responses"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ImmuneResponse:
    """Result of an immune response action"""
    response_id: str
    node_id: str
    response_type: ResponseType
    priority: ResponsePriority
    description: str
    actions_taken: List[str]
    timestamp: datetime
    duration: Optional[timedelta] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ResponsePlan:
    """Plan for responding to a threat"""
    threat_id: str
    node_id: str
    threat_type: str
    threat_level: str
    response_steps: List[Dict[str, Any]]
    estimated_duration: timedelta
    priority: ResponsePriority
    dependencies: List[str]

class ImmuneResponseSystem:
    """Coordinates automatic responses to detected threats and corruption"""
    
    def __init__(self, 
                 corruption_detector: CorruptionDetector,
                 trust_divergence_monitor: TrustDivergenceMonitor,
                 node_isolation: NodeIsolation,
                 trust_ledger: TrustLedger,
                 network_health: NetworkHealthMonitor):
        
        self.corruption_detector = corruption_detector
        self.trust_divergence_monitor = trust_divergence_monitor
        self.node_isolation = node_isolation
        self.trust_ledger = trust_ledger
        self.network_health = network_health
        
        # Try to get config, fall back to defaults if not available
        try:
            self.config = get_component_config("mesh_immunity")
        except Exception:
            # Use default values if config is not available
            self.config = {}
        
        # Response configuration
        self.response_thresholds = {
            "corruption": {
                CorruptionLevel.LOW: ResponseType.MONITORING,
                CorruptionLevel.MEDIUM: ResponseType.WARNING,
                CorruptionLevel.HIGH: ResponseType.QUARANTINE,
                CorruptionLevel.CRITICAL: ResponseType.ISOLATION
            },
            "divergence": {
                DivergenceSeverity.MINOR: ResponseType.MONITORING,
                DivergenceSeverity.MODERATE: ResponseType.WARNING,
                DivergenceSeverity.MAJOR: ResponseType.QUARANTINE,
                DivergenceSeverity.CRITICAL: ResponseType.ISOLATION
            }
        }
        
        # Active responses
        self.active_responses: Dict[str, ImmuneResponse] = {}
        
        # Response history
        self.response_history: List[ImmuneResponse] = []
        
        # Response coordination
        self.response_queue: List[ResponsePlan] = []
        self.processing_lock = asyncio.Lock()
        
        logger.info("Immune response system initialized")
    
    async def respond_to_threat(self, threat_detection: Any) -> ImmuneResponse:
        """Respond to a detected threat"""
        try:
            # Create response plan
            response_plan = await self._create_response_plan(threat_detection)
            if not response_plan:
                raise ValueError("Could not create response plan")
            
            # Add to response queue
            self.response_queue.append(response_plan)
            
            # Execute response
            response = await self._execute_response(response_plan)
            
            # Record response
            self.response_history.append(response)
            if response.success:
                self.active_responses[response.response_id] = response
            
            logger.info(f"Immune response executed: {response.response_type.value} for node {response.node_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error responding to threat: {e}")
            # Create error response
            error_response = ImmuneResponse(
                response_id=f"error_{int(time.time())}",
                node_id=getattr(threat_detection, 'node_id', 'unknown'),
                response_type=ResponseType.MONITORING,
                priority=ResponsePriority.HIGH,
                description=f"Error in threat response: {str(e)}",
                actions_taken=["error_logging"],
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            self.response_history.append(error_response)
            return error_response
    
    async def _create_response_plan(self, threat_detection: Any) -> Optional[ResponsePlan]:
        """Create a response plan for a detected threat"""
        try:
            # Determine threat type and level
            if isinstance(threat_detection, CorruptionDetection):
                threat_type = "corruption"
                threat_level = threat_detection.corruption_level.value
                node_id = threat_detection.node_id
                threat_details = {
                    "corruption_type": threat_detection.corruption_type.value,
                    "confidence": threat_detection.confidence_score,
                    "evidence": threat_detection.evidence
                }
            elif isinstance(threat_detection, TrustDivergence):
                threat_type = "divergence"
                threat_level = threat_detection.severity.value
                node_id = threat_detection.node_id
                threat_details = {
                    "divergence_type": threat_detection.divergence_type.value,
                    "confidence": threat_detection.confidence_score,
                    "metrics": threat_detection.divergence_metrics
                }
            else:
                logger.warning(f"Unknown threat detection type: {type(threat_detection)}")
                return None
            
            # Determine response type based on threat level
            response_type = self.response_thresholds[threat_type][threat_level]
            
            # Create response steps
            response_steps = await self._create_response_steps(
                threat_type, threat_level, response_type, threat_details
            )
            
            # Determine priority
            priority = self._determine_priority(threat_level, response_type)
            
            # Estimate duration
            estimated_duration = self._estimate_response_duration(response_steps)
            
            # Check dependencies
            dependencies = await self._check_response_dependencies(response_steps)
            
            return ResponsePlan(
                threat_id=f"{threat_type}_{node_id}_{int(time.time())}",
                node_id=node_id,
                threat_type=threat_type,
                threat_level=threat_level,
                response_steps=response_steps,
                estimated_duration=estimated_duration,
                priority=priority,
                dependencies=dependencies
            )
            
        except Exception as e:
            logger.error(f"Error creating response plan: {e}")
            return None
    
    async def _create_response_steps(self, 
                                   threat_type: str, 
                                   threat_level: str, 
                                   response_type: ResponseType,
                                   threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create specific response steps for a threat"""
        steps = []
        
        try:
            if response_type == ResponseType.MONITORING:
                steps = await self._create_monitoring_steps(threat_type, threat_level, threat_details)
            elif response_type == ResponseType.WARNING:
                steps = await self._create_warning_steps(threat_type, threat_level, threat_details)
            elif response_type == ResponseType.QUARANTINE:
                steps = await self._create_quarantine_steps(threat_type, threat_level, threat_details)
            elif response_type == ResponseType.ISOLATION:
                steps = await self._create_isolation_steps(threat_type, threat_level, threat_details)
            elif response_type == ResponseType.RECOVERY:
                steps = await self._create_recovery_steps(threat_type, threat_level, threat_details)
            elif response_type == ResponseType.PREVENTION:
                steps = await self._create_prevention_steps(threat_type, threat_level, threat_details)
            
        except Exception as e:
            logger.error(f"Error creating response steps: {e}")
            # Create basic monitoring steps as fallback
            steps = [
                {
                    "action": "enhanced_monitoring",
                    "description": "Enhanced monitoring due to error in step creation",
                    "duration": timedelta(minutes=5),
                    "dependencies": []
                }
            ]
        
        return steps
    
    async def _create_monitoring_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create monitoring response steps"""
        return [
            {
                "action": "increase_monitoring_frequency",
                "description": f"Increase monitoring frequency for {threat_type} threat",
                "duration": timedelta(minutes=2),
                "dependencies": []
            },
            {
                "action": "log_threat_details",
                "description": "Log detailed threat information for analysis",
                "duration": timedelta(minutes=1),
                "dependencies": []
            },
            {
                "action": "notify_administrators",
                "description": "Notify administrators of potential threat",
                "duration": timedelta(minutes=1),
                "dependencies": []
            }
        ]
    
    async def _create_warning_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create warning response steps"""
        return [
            {
                "action": "issue_warning",
                "description": f"Issue warning for {threat_type} threat",
                "duration": timedelta(minutes=2),
                "dependencies": []
            },
            {
                "action": "enhance_logging",
                "description": "Enhance logging and monitoring",
                "duration": timedelta(minutes=3),
                "dependencies": []
            },
            {
                "action": "prepare_escalation",
                "description": "Prepare escalation procedures if threat persists",
                "duration": timedelta(minutes=5),
                "dependencies": []
            }
        ]
    
    async def _create_quarantine_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create quarantine response steps"""
        return [
            {
                "action": "initiate_quarantine",
                "description": f"Initiating quarantine for {threat_type} threat",
                "duration": timedelta(minutes=5),
                "dependencies": ["node_isolation_available"]
            },
            {
                "action": "restrict_communications",
                "description": "Restrict node communications to essential only",
                "duration": timedelta(minutes=3),
                "dependencies": ["quarantine_initiated"]
            },
            {
                "action": "enhance_monitoring",
                "description": "Enhanced monitoring of quarantined node",
                "duration": timedelta(minutes=2),
                "dependencies": ["quarantine_initiated"]
            },
            {
                "action": "prepare_isolation",
                "description": "Prepare for full isolation if quarantine fails",
                "duration": timedelta(minutes=5),
                "dependencies": ["quarantine_initiated"]
            }
        ]
    
    async def _create_isolation_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create isolation response steps"""
        return [
            {
                "action": "emergency_isolation",
                "description": f"Emergency isolation for {threat_type} threat",
                "duration": timedelta(minutes=3),
                "dependencies": ["node_isolation_available"]
            },
            {
                "action": "cut_communications",
                "description": "Cut all communications with isolated node",
                "duration": timedelta(minutes=2),
                "dependencies": ["isolation_initiated"]
            },
            {
                "action": "notify_network",
                "description": "Notify network of node isolation",
                "duration": timedelta(minutes=2),
                "dependencies": ["isolation_initiated"]
            },
            {
                "action": "prepare_recovery",
                "description": "Prepare recovery procedures",
                "duration": timedelta(minutes=5),
                "dependencies": ["isolation_complete"]
            }
        ]
    
    async def _create_recovery_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create recovery response steps"""
        return [
            {
                "action": "assess_damage",
                "description": "Assess damage from threat",
                "duration": timedelta(minutes=10),
                "dependencies": []
            },
            {
                "action": "restore_services",
                "description": "Restore affected services",
                "duration": timedelta(minutes=15),
                "dependencies": ["damage_assessed"]
            },
            {
                "action": "validate_recovery",
                "description": "Validate recovery success",
                "duration": timedelta(minutes=5),
                "dependencies": ["services_restored"]
            },
            {
                "action": "update_security",
                "description": "Update security measures based on threat",
                "duration": timedelta(minutes=20),
                "dependencies": ["recovery_validated"]
            }
        ]
    
    async def _create_prevention_steps(self, threat_type: str, threat_level: str, threat_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create prevention response steps"""
        return [
            {
                "action": "analyze_threat_pattern",
                "description": "Analyze threat pattern for prevention",
                "duration": timedelta(minutes=10),
                "dependencies": []
            },
            {
                "action": "update_detection_rules",
                "description": "Update detection rules based on analysis",
                "duration": timedelta(minutes=15),
                "dependencies": ["pattern_analyzed"]
            },
            {
                "action": "implement_preventive_measures",
                "description": "Implement preventive measures",
                "duration": timedelta(minutes=20),
                "dependencies": ["rules_updated"]
            },
            {
                "action": "test_preventive_measures",
                "description": "Test preventive measures effectiveness",
                "duration": timedelta(minutes=10),
                "dependencies": ["measures_implemented"]
            }
        ]
    
    def _determine_priority(self, threat_level: str, response_type: ResponseType) -> ResponsePriority:
        """Determine response priority based on threat level and response type"""
        if threat_level in ["critical", "CRITICAL"]:
            return ResponsePriority.CRITICAL
        elif threat_level in ["high", "HIGH", "major", "MAJOR"]:
            return ResponsePriority.HIGH
        elif threat_level in ["medium", "MEDIUM", "moderate", "MODERATE"]:
            return ResponsePriority.MEDIUM
        else:
            return ResponsePriority.LOW
    
    def _estimate_response_duration(self, response_steps: List[Dict[str, Any]]) -> timedelta:
        """Estimate total duration for response steps"""
        total_duration = timedelta()
        for step in response_steps:
            total_duration += step.get("duration", timedelta(minutes=1))
        return total_duration
    
    async def _check_response_dependencies(self, response_steps: List[Dict[str, Any]]) -> List[str]:
        """Check dependencies for response steps"""
        dependencies = []
        for step in response_steps:
            step_deps = step.get("dependencies", [])
            dependencies.extend(step_deps)
        return list(set(dependencies))  # Remove duplicates
    
    async def _execute_response(self, response_plan: ResponsePlan) -> ImmuneResponse:
        """Execute a response plan"""
        try:
            async with self.processing_lock:
                response_id = f"response_{response_plan.threat_id}"
                start_time = datetime.now()
                actions_taken = []
                
                logger.info(f"Executing response plan: {response_plan.threat_type} - {response_plan.threat_level}")
                
                # Execute each response step
                for i, step in enumerate(response_plan.response_steps):
                    try:
                        # Check dependencies
                        if not await self._check_step_dependencies(step, actions_taken):
                            logger.warning(f"Step {i+1} dependencies not met, skipping")
                            continue
                        
                        # Execute step
                        step_result = await self._execute_response_step(step, response_plan)
                        if step_result:
                            actions_taken.append(step["action"])
                            logger.info(f"Step {i+1} completed: {step['action']}")
                        else:
                            logger.warning(f"Step {i+1} failed: {step['action']}")
                    
                    except Exception as e:
                        logger.error(f"Error executing step {i+1}: {e}")
                        # Continue with next step
                
                # Calculate duration
                duration = datetime.now() - start_time
                
                # Determine success
                success = len(actions_taken) >= len(response_plan.response_steps) * 0.7  # 70% success threshold
                
                response = ImmuneResponse(
                    response_id=response_id,
                    node_id=response_plan.node_id,
                    response_type=self._get_response_type_from_plan(response_plan),
                    priority=response_plan.priority,
                    description=f"Response to {response_plan.threat_type} threat (level: {response_plan.threat_level})",
                    actions_taken=actions_taken,
                    timestamp=start_time,
                    duration=duration,
                    success=success
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Error executing response plan: {e}")
            # Return error response
            return ImmuneResponse(
                response_id=f"error_{int(time.time())}",
                node_id=response_plan.node_id,
                response_type=ResponseType.MONITORING,
                priority=ResponsePriority.HIGH,
                description=f"Error executing response plan: {str(e)}",
                actions_taken=["error_logging"],
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    async def _check_step_dependencies(self, step: Dict[str, Any], completed_actions: List[str]) -> bool:
        """Check if step dependencies are met"""
        dependencies = step.get("dependencies", [])
        
        for dependency in dependencies:
            if dependency == "node_isolation_available":
                if not await self.node_isolation.is_available():
                    return False
            elif dependency in completed_actions:
                continue
            else:
                # Check if dependency is a system capability
                if not await self._check_system_capability(dependency):
                    return False
        
        return True
    
    async def _check_system_capability(self, capability: str) -> bool:
        """Check if a system capability is available"""
        try:
            if capability == "node_isolation_available":
                return await self.node_isolation.is_available()
            elif capability == "network_monitoring_available":
                return await self.network_health.is_monitoring_available()
            elif capability == "trust_validation_available":
                return await self.trust_ledger.is_validation_available()
            else:
                # Default to available for unknown capabilities
                return True
        except Exception as e:
            logger.error(f"Error checking system capability {capability}: {e}")
            return False
    
    async def _execute_response_step(self, step: Dict[str, Any], response_plan: ResponsePlan) -> bool:
        """Execute a single response step"""
        try:
            action = step["action"]
            
            if action == "increase_monitoring_frequency":
                return await self._increase_monitoring_frequency(response_plan.node_id)
            elif action == "log_threat_details":
                return await self._log_threat_details(response_plan)
            elif action == "notify_administrators":
                return await self._notify_administrators(response_plan)
            elif action == "issue_warning":
                return await self._issue_warning(response_plan)
            elif action == "enhance_logging":
                return await self._enhance_logging(response_plan.node_id)
            elif action == "prepare_escalation":
                return await self._prepare_escalation(response_plan)
            elif action == "initiate_quarantine":
                return await self._initiate_quarantine(response_plan.node_id)
            elif action == "restrict_communications":
                return await self._restrict_communications(response_plan.node_id)
            elif action == "enhance_monitoring":
                return await self._enhance_monitoring(response_plan.node_id)
            elif action == "prepare_isolation":
                return await self._prepare_isolation(response_plan.node_id)
            elif action == "emergency_isolation":
                return await self._emergency_isolation(response_plan.node_id)
            elif action == "cut_communications":
                return await self._cut_communications(response_plan.node_id)
            elif action == "notify_network":
                return await self._notify_network(response_plan)
            elif action == "prepare_recovery":
                return await self._prepare_recovery(response_plan.node_id)
            else:
                logger.warning(f"Unknown response step action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing response step {step.get('action', 'unknown')}: {e}")
            return False
    
    def _get_response_type_from_plan(self, response_plan: ResponsePlan) -> ResponseType:
        """Get response type from response plan"""
        # This would be determined by the response steps
        # For now, return a default based on threat level
        if response_plan.threat_level in ["critical", "CRITICAL"]:
            return ResponseType.ISOLATION
        elif response_plan.threat_level in ["high", "HIGH", "major", "MAJOR"]:
            return ResponseType.QUARANTINE
        elif response_plan.threat_level in ["medium", "MEDIUM", "moderate", "MODERATE"]:
            return ResponseType.WARNING
        else:
            return ResponseType.MONITORING
    
    # Response step implementations
    async def _increase_monitoring_frequency(self, node_id: str) -> bool:
        """Increase monitoring frequency for a node"""
        try:
            # Implementation would increase monitoring frequency
            logger.info(f"Increased monitoring frequency for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error increasing monitoring frequency: {e}")
            return False
    
    async def _log_threat_details(self, response_plan: ResponsePlan) -> bool:
        """Log detailed threat information"""
        try:
            # Implementation would log threat details
            logger.info(f"Logged threat details for {response_plan.threat_type} threat")
            return True
        except Exception as e:
            logger.error(f"Error logging threat details: {e}")
            return False
    
    async def _notify_administrators(self, response_plan: ResponsePlan) -> bool:
        """Notify administrators of potential threat"""
        try:
            # Implementation would send notifications
            logger.info(f"Notified administrators of {response_plan.threat_type} threat")
            return True
        except Exception as e:
            logger.error(f"Error notifying administrators: {e}")
            return False
    
    async def _issue_warning(self, response_plan: ResponsePlan) -> bool:
        """Issue warning for threat"""
        try:
            # Implementation would issue warnings
            logger.info(f"Issued warning for {response_plan.threat_type} threat")
            return True
        except Exception as e:
            logger.error(f"Error issuing warning: {e}")
            return False
    
    async def _enhance_logging(self, node_id: str) -> bool:
        """Enhance logging and monitoring"""
        try:
            # Implementation would enhance logging
            logger.info(f"Enhanced logging for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error enhancing logging: {e}")
            return False
    
    async def _prepare_escalation(self, response_plan: ResponsePlan) -> bool:
        """Prepare escalation procedures"""
        try:
            # Implementation would prepare escalation
            logger.info(f"Prepared escalation for {response_plan.threat_type} threat")
            return True
        except Exception as e:
            logger.error(f"Error preparing escalation: {e}")
            return False
    
    async def _initiate_quarantine(self, node_id: str) -> bool:
        """Initiate quarantine for a node"""
        try:
            # Implementation would initiate quarantine
            logger.info(f"Initiating quarantine for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error initiating quarantine: {e}")
            return False
    
    async def _restrict_communications(self, node_id: str) -> bool:
        """Restrict node communications"""
        try:
            # Implementation would restrict communications
            logger.info(f"Restricted communications for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error restricting communications: {e}")
            return False
    
    async def _enhance_monitoring(self, node_id: str) -> bool:
        """Enhance monitoring of a node"""
        try:
            # Implementation would enhance monitoring
            logger.info(f"Enhanced monitoring for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error enhancing monitoring: {e}")
            return False
    
    async def _prepare_isolation(self, node_id: str) -> bool:
        """Prepare for full isolation"""
        try:
            # Implementation would prepare isolation
            logger.info(f"Prepared isolation for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error preparing isolation: {e}")
            return False
    
    async def _emergency_isolation(self, node_id: str) -> bool:
        """Emergency isolation of a node"""
        try:
            # Implementation would perform emergency isolation
            logger.info(f"Emergency isolation for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error emergency isolation: {e}")
            return False
    
    async def _cut_communications(self, node_id: str) -> bool:
        """Cut all communications with a node"""
        try:
            # Implementation would cut communications
            logger.info(f"Cut communications for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error cutting communications: {e}")
            return False
    
    async def _notify_network(self, response_plan: ResponsePlan) -> bool:
        """Notify network of node isolation"""
        try:
            # Implementation would notify network
            logger.info(f"Notified network of isolation for {response_plan.threat_type} threat")
            return True
        except Exception as e:
            logger.error(f"Error notifying network: {e}")
            return False
    
    async def _prepare_recovery(self, node_id: str) -> bool:
        """Prepare recovery procedures"""
        try:
            # Implementation would prepare recovery
            logger.info(f"Prepared recovery for node {node_id}")
            return True
        except Exception as e:
            logger.error(f"Error preparing recovery: {e}")
            return False
    
    def get_active_responses(self) -> Dict[str, ImmuneResponse]:
        """Get currently active responses"""
        return self.active_responses.copy()
    
    def get_response_history(self, node_id: Optional[str] = None) -> List[ImmuneResponse]:
        """Get response history"""
        if node_id:
            return [r for r in self.response_history if r.node_id == node_id]
        return self.response_history.copy()
    
    def get_response_summary(self) -> Dict[str, Any]:
        """Get summary of immune response activity"""
        if not self.response_history:
            return {"total_responses": 0, "active_responses": 0}
        
        # Count by response type
        type_counts = {}
        for response in self.response_history:
            response_type = response.response_type.value
            type_counts[response_type] = type_counts.get(response_type, 0) + 1
        
        # Count active responses
        active_count = len(self.active_responses)
        
        # Count successful responses
        successful_count = sum(1 for r in self.response_history if r.success)
        
        return {
            "total_responses": len(self.response_history),
            "active_responses": active_count,
            "successful_responses": successful_count,
            "success_rate": successful_count / len(self.response_history) if self.response_history else 0,
            "type_breakdown": type_counts
        }
