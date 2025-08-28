"""
Workflow Engine - Enhanced with Sentient's Task Execution Concepts

Integrates Sentient's proven task execution patterns:
- Task planning and execution orchestration
- Workflow state management and progress tracking
- Context-aware task execution and refinement
- Intelligent error handling and recovery

Following the same integration pattern used for Leon, Empathy, AxiomEngine, and Memory.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/tasks')
    from models import TaskStep
    SENTIENT_TASKS_AVAILABLE = True
except ImportError:
    SENTIENT_TASKS_AVAILABLE = False

class TaskStatus(Enum):
    """Task execution status"""
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStepStatus(Enum):
    """Task step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    """A step in a workflow execution"""
    id: str
    name: str
    description: str
    tool: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStepStatus = TaskStepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """A workflow execution instance"""
    id: str
    task_id: str
    status: TaskStatus
    steps: List[WorkflowStep]
    current_step: Optional[str] = None
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    execution_id: str
    task_id: str
    status: TaskStatus
    total_steps: int
    completed_steps: int
    failed_steps: int
    execution_time: float
    result: Optional[Any]
    error: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class WorkflowEngineConfig:
    """Configuration for workflow engine"""
    enable_parallel_execution: bool = True
    enable_error_recovery: bool = True
    enable_progress_tracking: bool = True
    enable_context_integration: bool = True
    max_parallel_steps: int = 5
    step_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    enable_debug_logging: bool = False

class WorkflowEngine:
    """
    Enhanced workflow engine integrating Sentient's task execution concepts
    
    Provides:
    - Task planning and execution orchestration
    - Workflow state management and progress tracking
    - Context-aware task execution and refinement
    - Intelligent error handling and recovery
    """
    
    def __init__(self, config: WorkflowEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize workflow components
        self.task_planner = None
        self.execution_orchestrator = None
        self.progress_tracker = None
        self.error_handler = None
        
        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        # Performance tracking
        self.execution_count = 0
        self.total_execution_time = 0.0
        
        # Initialize components
        self._initialize_workflow_components()
        
        self.logger.info("Workflow Engine initialized with Sentient concepts")
    
    def _initialize_workflow_components(self):
        """Initialize workflow components using Sentient patterns"""
        
        try:
            # Initialize task planner
            self._initialize_task_planner()
            
            # Initialize execution orchestrator
            self._initialize_execution_orchestrator()
            
            # Initialize progress tracker
            if self.config.enable_progress_tracking:
                self._initialize_progress_tracker()
            
            # Initialize error handler
            if self.config.enable_error_recovery:
                self._initialize_error_handler()
            
            self.logger.info("Workflow components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow components: {e}")
            self.logger.warning("Workflow engine will use basic methods")
    
    def _initialize_task_planner(self):
        """Initialize task planning component"""
        
        try:
            self.task_planner = TaskPlanner()
            self.logger.info("Task planner initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize task planner: {e}")
            self.task_planner = None
    
    def _initialize_execution_orchestrator(self):
        """Initialize execution orchestration component"""
        
        try:
            self.execution_orchestrator = ExecutionOrchestrator(self.config)
            self.logger.info("Execution orchestrator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution orchestrator: {e}")
            self.execution_orchestrator = None
    
    def _initialize_progress_tracker(self):
        """Initialize progress tracking component"""
        
        try:
            self.progress_tracker = ProgressTracker()
            self.logger.info("Progress tracker initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize progress tracker: {e}")
            self.progress_tracker = None
    
    def _initialize_error_handler(self):
        """Initialize error handling component"""
        
        try:
            self.error_handler = ErrorHandler(self.config)
            self.logger.info("Error handler initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error handler: {e}")
            self.error_handler = None
    
    async def create_workflow(self, task_id: str, steps: List[Dict[str, Any]]) -> str:
        """
        Create a new workflow for task execution
        
        Args:
            task_id: ID of the task to execute
            steps: List of workflow steps
            
        Returns:
            Workflow execution ID
        """
        
        try:
            # Generate execution ID
            execution_id = f"workflow_{task_id}_{int(time.time())}"
            
            # Create workflow steps
            workflow_steps = []
            for i, step_data in enumerate(steps):
                step = WorkflowStep(
                    id=f"step_{i}",
                    name=step_data.get("name", f"Step {i}"),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", "unknown"),
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", [])
                )
                workflow_steps.append(step)
            
            # Create workflow execution
            execution = WorkflowExecution(
                id=execution_id,
                task_id=task_id,
                status=TaskStatus.PLANNING,
                steps=workflow_steps
            )
            
            # Store execution
            self.active_executions[execution_id] = execution
            
            self.logger.info(f"Created workflow {execution_id} with {len(steps)} steps")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def execute_workflow(self, execution_id: str) -> WorkflowResult:
        """
        Execute a workflow
        
        Args:
            execution_id: ID of the workflow to execute
            
        Returns:
            WorkflowResult with execution details
        """
        
        start_time = time.time()
        
        try:
            if execution_id not in self.active_executions:
                raise ValueError(f"Workflow {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            
            # Update status to running
            execution.status = TaskStatus.RUNNING
            execution.start_time = time.time()
            
            # Plan execution if planner is available
            if self.task_planner:
                await self.task_planner.plan_execution(execution)
            
            # Execute workflow
            if self.execution_orchestrator:
                result = await self.execution_orchestrator.execute(execution)
            else:
                result = await self._basic_execution(execution)
            
            # Update execution
            execution.end_time = time.time()
            execution.result = result
            
            # Update performance tracking
            execution_time = execution.end_time - execution.start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            # Create result
            workflow_result = WorkflowResult(
                execution_id=execution_id,
                task_id=execution.task_id,
                status=execution.status,
                total_steps=len(execution.steps),
                completed_steps=len([s for s in execution.steps if s.status == TaskStepStatus.COMPLETED]),
                failed_steps=len([s for s in execution.steps if s.status == TaskStepStatus.FAILED]),
                execution_time=execution_time,
                result=execution.result,
                error=execution.error,
                metadata={"method": "orchestrated" if self.execution_orchestrator else "basic"}
            )
            
            self.logger.info(f"Workflow {execution_id} executed successfully in {execution_time:.2f}s")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            execution_time = time.time() - start_time
            
            # Update execution with error
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = TaskStatus.FAILED
                execution.error = str(e)
                execution.end_time = time.time()
            
            # Return error result
            return WorkflowResult(
                execution_id=execution_id,
                task_id=execution.task_id if execution_id in self.active_executions else "unknown",
                status=TaskStatus.FAILED,
                total_steps=0,
                completed_steps=0,
                failed_steps=1,
                execution_time=execution_time,
                result=None,
                error=str(e),
                metadata={"error": str(e)}
            )
    
    async def _basic_execution(self, execution: WorkflowExecution) -> Any:
        """Basic workflow execution without orchestration"""
        
        try:
            self.logger.info(f"Executing workflow {execution.id} with basic method")
            
            # Execute steps sequentially
            for step in execution.steps:
                step.status = TaskStepStatus.RUNNING
                step.start_time = time.time()
                
                try:
                    # Simulate step execution
                    await asyncio.sleep(0.1)  # Simulate work
                    
                    # Mark step as completed
                    step.status = TaskStepStatus.COMPLETED
                    step.end_time = time.time()
                    step.result = {"status": "completed", "step_id": step.id}
                    
                except Exception as e:
                    step.status = TaskStepStatus.FAILED
                    step.end_time = time.time()
                    step.error = str(e)
                    self.logger.error(f"Step {step.id} failed: {e}")
            
            # Check if all steps completed
            failed_steps = [s for s in execution.steps if s.status == TaskStepStatus.FAILED]
            if failed_steps:
                execution.status = TaskStatus.FAILED
                execution.error = f"{len(failed_steps)} steps failed"
            else:
                execution.status = TaskStatus.COMPLETED
            
            return {"status": "completed", "steps": len(execution.steps)}
            
        except Exception as e:
            self.logger.error(f"Basic execution failed: {e}")
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            raise
    
    async def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get current status of a workflow execution"""
        
        try:
            return self.active_executions.get(execution_id)
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            return None
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        
        try:
            if execution_id not in self.active_executions:
                return False
            
            execution = self.active_executions[execution_id]
            
            if execution.status == TaskStatus.RUNNING:
                execution.status = TaskStatus.CANCELLED
                execution.end_time = time.time()
                
                # Cancel running steps
                for step in execution.steps:
                    if step.status == TaskStepStatus.RUNNING:
                        step.status = TaskStepStatus.SKIPPED
                        step.end_time = time.time()
                
                self.logger.info(f"Workflow {execution_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow: {e}")
            return False
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow"""
        
        try:
            if execution_id not in self.active_executions:
                return False
            
            execution = self.active_executions[execution_id]
            
            if execution.status == TaskStatus.RUNNING:
                execution.status = TaskStatus.PAUSED
                self.logger.info(f"Workflow {execution_id} paused")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to pause workflow: {e}")
            return False
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow"""
        
        try:
            if execution_id not in self.active_executions:
                return False
            
            execution = self.active_executions[execution_id]
            
            if execution.status == TaskStatus.PAUSED:
                execution.status = TaskStatus.RUNNING
                self.logger.info(f"Workflow {execution_id} resumed")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resume workflow: {e}")
            return False
    
    def get_active_workflows(self) -> List[WorkflowExecution]:
        """Get list of active workflow executions"""
        
        try:
            return list(self.active_executions.values())
        except Exception as e:
            self.logger.error(f"Failed to get active workflows: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for workflow engine"""
        
        avg_execution_time = (self.total_execution_time / self.execution_count 
                            if self.execution_count > 0 else 0)
        
        active_count = len(self.active_executions)
        completed_count = len([w for w in self.active_executions.values() 
                             if w.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]])
        
        return {
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "active_workflows": active_count,
            "completed_workflows": completed_count,
            "task_planner_available": self.task_planner is not None,
            "execution_orchestrator_available": self.execution_orchestrator is not None,
            "progress_tracker_available": self.progress_tracker is not None,
            "error_handler_available": self.error_handler is not None
        }
    
    async def cleanup(self):
        """Clean up workflow engine resources"""
        
        try:
            # Cancel all running workflows
            for execution_id in list(self.active_executions.keys()):
                await self.cancel_workflow(execution_id)
            
            # Clean up components
            if hasattr(self.task_planner, 'cleanup'):
                await self.task_planner.cleanup()
            
            if hasattr(self.execution_orchestrator, 'cleanup'):
                await self.execution_orchestrator.cleanup()
            
            if hasattr(self.progress_tracker, 'cleanup'):
                await self.progress_tracker.cleanup()
            
            if hasattr(self.error_handler, 'cleanup'):
                await self.error_handler.cleanup()
                
            self.logger.info("Workflow engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class TaskPlanner:
    """Task planning component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def plan_execution(self, execution: WorkflowExecution):
        """Plan the execution of a workflow"""
        
        try:
            # Analyze step dependencies
            self._analyze_dependencies(execution)
            
            # Optimize step order
            self._optimize_step_order(execution)
            
            # Set initial step
            if execution.steps:
                execution.current_step = execution.steps[0].id
            
            self.logger.debug(f"Planned execution for workflow {execution.id}")
            
        except Exception as e:
            self.logger.error(f"Task planning failed: {e}")
    
    def _analyze_dependencies(self, execution: WorkflowExecution):
        """Analyze dependencies between workflow steps"""
        
        try:
            # Create dependency graph
            dependency_graph = {}
            
            for step in execution.steps:
                dependency_graph[step.id] = step.dependencies
            
            # Validate dependencies
            for step in execution.steps:
                for dep_id in step.dependencies:
                    if dep_id not in dependency_graph:
                        self.logger.warning(f"Step {step.id} depends on unknown step {dep_id}")
            
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
    
    def _optimize_step_order(self, execution: WorkflowExecution):
        """Optimize the order of workflow steps"""
        
        try:
            # Simple optimization: sort by dependencies
            # In a real implementation, we'd use topological sorting
            
            execution.steps.sort(key=lambda x: len(x.dependencies))
            
        except Exception as e:
            self.logger.error(f"Step order optimization failed: {e}")
    
    async def cleanup(self):
        """Clean up task planner resources"""
        pass


class ExecutionOrchestrator:
    """Execution orchestration component following Sentient patterns"""
    
    def __init__(self, config: WorkflowEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, execution: WorkflowExecution) -> Any:
        """Execute a workflow with orchestration"""
        
        try:
            self.logger.info(f"Executing workflow {execution.id} with orchestration")
            
            # Execute steps with dependency management
            completed_steps = set()
            
            while len(completed_steps) < len(execution.steps):
                # Find ready steps (dependencies satisfied)
                ready_steps = [
                    step for step in execution.steps
                    if step.id not in completed_steps and
                    all(dep in completed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    # Check for circular dependencies
                    remaining_steps = [s for s in execution.steps if s.id not in completed_steps]
                    if remaining_steps:
                        self.logger.error(f"Circular dependency detected in steps: {[s.id for s in remaining_steps]}")
                        execution.status = TaskStatus.FAILED
                        execution.error = "Circular dependency detected"
                        return None
                    break
                
                # Execute ready steps
                if self.config.enable_parallel_execution:
                    # Execute in parallel (limited by max_parallel_steps)
                    parallel_steps = ready_steps[:self.config.max_parallel_steps]
                    tasks = [self._execute_step(step) for step in parallel_steps]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for step, result in zip(parallel_steps, results):
                        if isinstance(result, Exception):
                            step.status = TaskStepStatus.FAILED
                            step.error = str(result)
                        else:
                            step.status = TaskStepStatus.COMPLETED
                            step.result = result
                        completed_steps.add(step.id)
                else:
                    # Execute sequentially
                    for step in ready_steps:
                        try:
                            result = await self._execute_step(step)
                            step.status = TaskStepStatus.COMPLETED
                            step.result = result
                        except Exception as e:
                            step.status = TaskStepStatus.FAILED
                            step.error = str(e)
                        completed_steps.add(step.id)
            
            # Check final status
            failed_steps = [s for s in execution.steps if s.status == TaskStepStatus.FAILED]
            if failed_steps:
                execution.status = TaskStatus.FAILED
                execution.error = f"{len(failed_steps)} steps failed"
                return None
            else:
                execution.status = TaskStatus.COMPLETED
                return {"status": "completed", "steps": len(execution.steps)}
            
        except Exception as e:
            self.logger.error(f"Execution orchestration failed: {e}")
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            raise
    
    async def _execute_step(self, step: WorkflowStep) -> Any:
        """Execute a single workflow step"""
        
        try:
            step.status = TaskStepStatus.RUNNING
            step.start_time = time.time()
            
            # Simulate step execution based on tool
            if step.tool == "delay":
                await asyncio.sleep(step.parameters.get("duration", 1))
                result = {"status": "delayed", "duration": step.parameters.get("duration", 1)}
            elif step.tool == "process":
                # Simulate processing
                await asyncio.sleep(0.1)
                result = {"status": "processed", "input": step.parameters.get("input", "")}
            else:
                # Generic step execution
                await asyncio.sleep(0.1)
                result = {"status": "executed", "tool": step.tool, "parameters": step.parameters}
            
            step.end_time = time.time()
            return result
            
        except Exception as e:
            step.end_time = time.time()
            step.error = str(e)
            raise
    
    async def cleanup(self):
        """Clean up execution orchestrator resources"""
        pass


class ProgressTracker:
    """Progress tracking component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def track_progress(self, execution: WorkflowExecution) -> float:
        """Track progress of workflow execution"""
        
        try:
            total_steps = len(execution.steps)
            if total_steps == 0:
                return 0.0
            
            completed_steps = len([s for s in execution.steps if s.status == TaskStepStatus.COMPLETED])
            failed_steps = len([s for s in execution.steps if s.status == TaskStepStatus.FAILED])
            
            # Calculate progress percentage
            progress = (completed_steps + failed_steps) / total_steps
            execution.progress = progress
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Progress tracking failed: {e}")
            return 0.0
    
    async def cleanup(self):
        """Clean up progress tracker resources"""
        pass


class ErrorHandler:
    """Error handling component following Sentient patterns"""
    
    def __init__(self, config: WorkflowEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def handle_error(self, execution: WorkflowExecution, step: WorkflowStep, error: Exception):
        """Handle errors during workflow execution"""
        
        try:
            self.logger.error(f"Error in step {step.id}: {error}")
            
            # Update step status
            step.status = TaskStepStatus.FAILED
            step.error = str(error)
            step.end_time = time.time()
            
            # Check if we should retry
            retry_count = step.metadata.get("retry_count", 0)
            if retry_count < self.config.retry_attempts:
                step.metadata["retry_count"] = retry_count + 1
                step.status = TaskStepStatus.PENDING
                step.start_time = None
                step.end_time = None
                step.error = None
                
                self.logger.info(f"Retrying step {step.id} (attempt {retry_count + 1})")
            else:
                self.logger.error(f"Step {step.id} failed after {retry_count} retries")
                
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
    
    async def cleanup(self):
        """Clean up error handler resources"""
        pass


# Factory function for easy integration
def create_workflow_engine(config: Optional[WorkflowEngineConfig] = None) -> WorkflowEngine:
    """Create a workflow engine with default or custom configuration"""
    
    if config is None:
        config = WorkflowEngineConfig()
    
    return WorkflowEngine(config)
