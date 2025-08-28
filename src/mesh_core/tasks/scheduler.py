"""
Task Scheduler - Enhanced with Sentient's Task Scheduling Concepts

Integrates Sentient's proven task scheduling patterns:
- Intelligent task scheduling and prioritization
- Timezone-aware scheduling and execution
- Recurring and triggered task management
- Context-aware scheduling optimization

Following the same integration pattern used for Leon, Empathy, AxiomEngine, and Memory.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/tasks')
    from models import TaskStep
    SENTIENT_TASKS_AVAILABLE = True
except ImportError:
    SENTIENT_TASKS_AVAILABLE = False

class ScheduleType(Enum):
    """Schedule types for tasks"""
    ONCE = "once"
    RECURRING = "recurring"
    TRIGGERED = "triggered"

class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2

@dataclass
class ScheduledTask:
    """A scheduled task with execution details"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    next_execution: Optional[datetime] = None
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    max_executions: Optional[int] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionContext:
    """Context for task execution"""
    task_id: str
    execution_time: datetime
    user_id: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchedulingResult:
    """Result of task scheduling operation"""
    task_id: str
    scheduled: bool
    next_execution: Optional[datetime]
    message: str
    metadata: Dict[str, Any]

@dataclass
class SchedulerConfig:
    """Configuration for task scheduler"""
    enable_timezone_support: bool = True
    enable_priority_queuing: bool = True
    enable_parallel_execution: bool = True
    enable_context_optimization: bool = True
    default_timezone: str = "UTC"
    max_parallel_tasks: int = 10
    execution_timeout: float = 300.0  # 5 minutes
    enable_debug_logging: bool = False

class TaskScheduler:
    """
    Enhanced task scheduler integrating Sentient's task scheduling concepts
    
    Provides:
    - Intelligent task scheduling and prioritization
    - Timezone-aware scheduling and execution
    - Recurring and triggered task management
    - Context-aware scheduling optimization
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize scheduling components
        self.priority_queue = None
        self.timezone_manager = None
        self.execution_optimizer = None
        self.trigger_manager = None
        
        # Scheduled tasks storage
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        
        # Execution queue
        self.execution_queue: List[ScheduledTask] = []
        
        # Performance tracking
        self.scheduling_count = 0
        self.execution_count = 0
        
        # Initialize components
        self._initialize_scheduling_components()
        
        self.logger.info("Task Scheduler initialized with Sentient concepts")
    
    def _initialize_scheduling_components(self):
        """Initialize scheduling components using Sentient patterns"""
        
        try:
            # Initialize priority queue
            if self.config.enable_priority_queuing:
                self._initialize_priority_queue()
            
            # Initialize timezone manager
            if self.config.enable_timezone_support:
                self._initialize_timezone_manager()
            
            # Initialize execution optimizer
            if self.config.enable_context_optimization:
                self._initialize_execution_optimizer()
            
            # Initialize trigger manager
            self._initialize_trigger_manager()
            
            self.logger.info("Scheduling components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduling components: {e}")
            self.logger.warning("Task scheduler will use basic methods")
    
    def _initialize_priority_queue(self):
        """Initialize priority queue component"""
        
        try:
            self.priority_queue = PriorityQueue()
            self.logger.info("Priority queue initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize priority queue: {e}")
            self.priority_queue = None
    
    def _initialize_timezone_manager(self):
        """Initialize timezone management component"""
        
        try:
            self.timezone_manager = TimezoneManager(self.config.default_timezone)
            self.logger.info("Timezone manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize timezone manager: {e}")
            self.timezone_manager = None
    
    def _initialize_execution_optimizer(self):
        """Initialize execution optimization component"""
        
        try:
            self.execution_optimizer = ExecutionOptimizer()
            self.logger.info("Execution optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution optimizer: {e}")
            self.execution_optimizer = None
    
    def _initialize_trigger_manager(self):
        """Initialize trigger management component"""
        
        try:
            self.trigger_manager = TriggerManager()
            self.logger.info("Trigger manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trigger manager: {e}")
            self.trigger_manager = None
    
    async def schedule_task(self, task_data: Dict[str, Any], user_id: str = "user") -> SchedulingResult:
        """
        Schedule a new task
        
        Args:
            task_data: Task information including schedule
            user_id: ID of the user scheduling the task
            
        Returns:
            SchedulingResult with scheduling details
        """
        
        try:
            # Generate task ID
            task_id = f"task_{user_id}_{int(time.time())}"
            
            # Parse schedule configuration
            schedule_type, schedule_config = await self._parse_schedule(task_data.get("schedule", {}))
            
            # Create scheduled task
            task = ScheduledTask(
                id=task_id,
                name=task_data.get("name", "New Task"),
                description=task_data.get("description", ""),
                priority=TaskPriority(task_data.get("priority", 1)),
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                metadata={"user_id": user_id, "created_at": datetime.now()}
            )
            
            # Calculate next execution time
            next_execution = await self._calculate_next_execution(task)
            task.next_execution = next_execution
            
            # Store task
            self.scheduled_tasks[task_id] = task
            
            # Add to execution queue
            if next_execution:
                await self._add_to_execution_queue(task)
            
            # Update performance tracking
            self.scheduling_count += 1
            
            self.logger.info(f"Scheduled task {task_id} for {next_execution}")
            
            return SchedulingResult(
                task_id=task_id,
                scheduled=True,
                next_execution=next_execution,
                message="Task scheduled successfully",
                metadata={"schedule_type": schedule_type.value}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to schedule task: {e}")
            return SchedulingResult(
                task_id="unknown",
                scheduled=False,
                next_execution=None,
                message=f"Failed to schedule task: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _parse_schedule(self, schedule_data: Dict[str, Any]) -> Tuple[ScheduleType, Dict[str, Any]]:
        """Parse schedule configuration from task data"""
        
        try:
            schedule_type_str = schedule_data.get("type", "once")
            
            if schedule_type_str == "once":
                schedule_type = ScheduleType.ONCE
                schedule_config = {
                    "run_at": schedule_data.get("run_at"),
                    "immediate": schedule_data.get("run_at") is None
                }
            
            elif schedule_type_str == "recurring":
                schedule_type = ScheduleType.RECURRING
                schedule_config = {
                    "frequency": schedule_data.get("frequency", "daily"),
                    "days": schedule_data.get("days", ["Monday"]),
                    "time": schedule_data.get("time", "09:00"),
                    "timezone": schedule_data.get("timezone", self.config.default_timezone)
                }
            
            elif schedule_type_str == "triggered":
                schedule_type = ScheduleType.TRIGGERED
                schedule_config = {
                    "source": schedule_data.get("source", "unknown"),
                    "event": schedule_data.get("event", "unknown"),
                    "filter": schedule_data.get("filter", {})
                }
            
            else:
                # Default to once
                schedule_type = ScheduleType.ONCE
                schedule_config = {"immediate": True}
            
            return schedule_type, schedule_config
            
        except Exception as e:
            self.logger.error(f"Schedule parsing failed: {e}")
            return ScheduleType.ONCE, {"immediate": True}
    
    async def _calculate_next_execution(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next execution time for a task"""
        
        try:
            if task.schedule_type == ScheduleType.ONCE:
                return await self._calculate_once_execution(task)
            
            elif task.schedule_type == ScheduleType.RECURRING:
                return await self._calculate_recurring_execution(task)
            
            elif task.schedule_type == ScheduleType.TRIGGERED:
                return None  # Triggered tasks don't have fixed execution times
            
            return None
            
        except Exception as e:
            self.logger.error(f"Next execution calculation failed: {e}")
            return None
    
    async def _calculate_once_execution(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate execution time for one-time tasks"""
        
        try:
            schedule_config = task.schedule_config
            
            if schedule_config.get("immediate", False):
                return datetime.now()
            
            run_at = schedule_config.get("run_at")
            if run_at:
                # Parse ISO format datetime string
                if isinstance(run_at, str):
                    return datetime.fromisoformat(run_at.replace('Z', '+00:00'))
                elif isinstance(run_at, datetime):
                    return run_at
            
            return datetime.now()
            
        except Exception as e:
            self.logger.error(f"Once execution calculation failed: {e}")
            return datetime.now()
    
    async def _calculate_recurring_execution(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next execution time for recurring tasks"""
        
        try:
            schedule_config = task.schedule_config
            frequency = schedule_config.get("frequency", "daily")
            time_str = schedule_config.get("time", "09:00")
            days = schedule_config.get("days", ["Monday"])
            timezone_str = schedule_config.get("timezone", self.config.default_timezone)
            
            # Parse time
            hour, minute = map(int, time_str.split(":"))
            
            # Get current time in task timezone
            current_time = datetime.now()
            if self.timezone_manager:
                current_time = await self.timezone_manager.convert_to_timezone(current_time, timezone_str)
            
            if frequency == "daily":
                # Next execution is today or tomorrow at specified time
                next_execution = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_execution <= current_time:
                    next_execution += timedelta(days=1)
                return next_execution
            
            elif frequency == "weekly":
                # Find next occurrence of specified days
                current_weekday = current_time.weekday()
                weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                
                for day_name in days:
                    day_index = weekday_names.index(day_name)
                    days_ahead = (day_index - current_weekday) % 7
                    if days_ahead == 0 and current_time.time() < datetime.strptime(time_str, "%H:%M").time():
                        days_ahead = 7
                    
                    next_execution = current_time + timedelta(days=days_ahead)
                    next_execution = next_execution.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return next_execution
            
            return None
            
        except Exception as e:
            self.logger.error(f"Recurring execution calculation failed: {e}")
            return None
    
    async def _add_to_execution_queue(self, task: ScheduledTask):
        """Add task to execution queue"""
        
        try:
            if self.priority_queue:
                await self.priority_queue.add_task(task)
            else:
                # Basic queue management
                self.execution_queue.append(task)
                self.execution_queue.sort(key=lambda t: (t.priority.value, t.next_execution or datetime.max))
            
        except Exception as e:
            self.logger.error(f"Failed to add task to execution queue: {e}")
    
    async def get_next_tasks(self, limit: int = 10) -> List[ScheduledTask]:
        """Get next tasks ready for execution"""
        
        try:
            if self.priority_queue:
                return await self.priority_queue.get_next_tasks(limit)
            else:
                # Basic queue management
                current_time = datetime.now()
                ready_tasks = [
                    task for task in self.execution_queue
                    if task.enabled and task.next_execution and task.next_execution <= current_time
                ]
                return ready_tasks[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get next tasks: {e}")
            return []
    
    async def execute_task(self, task: ScheduledTask, context: Optional[ExecutionContext] = None) -> bool:
        """
        Execute a scheduled task
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            True if execution was successful, False otherwise
        """
        
        try:
            self.logger.info(f"Executing task {task.id}: {task.name}")
            
            # Update execution tracking
            task.last_execution = datetime.now()
            task.execution_count += 1
            
            # Create execution context if not provided
            if not context:
                context = ExecutionContext(
                    task_id=task.id,
                    execution_time=datetime.now(),
                    user_id=task.metadata.get("user_id", "unknown"),
                    parameters={}
                )
            
            # Execute task (simulated for now)
            success = await self._execute_task_logic(task, context)
            
            if success:
                # Calculate next execution for recurring tasks
                if task.schedule_type == ScheduleType.RECURRING:
                    next_execution = await self._calculate_recurring_execution(task)
                    task.next_execution = next_execution
                    
                    # Re-add to queue if there's a next execution
                    if next_execution:
                        await self._add_to_execution_queue(task)
                
                # Check if task has reached max executions
                if task.max_executions and task.execution_count >= task.max_executions:
                    task.enabled = False
                    self.logger.info(f"Task {task.id} reached max executions, disabling")
            
            # Update performance tracking
            self.execution_count += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return False
    
    async def _execute_task_logic(self, task: ScheduledTask, context: ExecutionContext) -> bool:
        """Execute the actual task logic"""
        
        try:
            # Simulate task execution
            await asyncio.sleep(0.1)
            
            # Log execution
            self.logger.info(f"Task {task.id} executed successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Task logic execution failed: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        
        try:
            if task_id not in self.scheduled_tasks:
                return False
            
            task = self.scheduled_tasks[task_id]
            task.enabled = False
            
            # Remove from execution queue
            if self.priority_queue:
                await self.priority_queue.remove_task(task_id)
            else:
                self.execution_queue = [t for t in self.execution_queue if t.id != task_id]
            
            self.logger.info(f"Task {task_id} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task: {e}")
            return False
    
    async def update_task_schedule(self, task_id: str, new_schedule: Dict[str, Any]) -> bool:
        """Update schedule for an existing task"""
        
        try:
            if task_id not in self.scheduled_tasks:
                return False
            
            task = self.scheduled_tasks[task_id]
            
            # Parse new schedule
            schedule_type, schedule_config = await self._parse_schedule(new_schedule)
            
            # Update task schedule
            task.schedule_type = schedule_type
            task.schedule_config = schedule_config
            
            # Calculate new next execution
            next_execution = await self._calculate_next_execution(task)
            task.next_execution = next_execution
            
            # Update execution queue
            if self.priority_queue:
                await self.priority_queue.update_task(task)
            else:
                # Remove and re-add to basic queue
                self.execution_queue = [t for t in self.execution_queue if t.id != task_id]
                if next_execution:
                    await self._add_to_execution_queue(task)
            
            self.logger.info(f"Task {task_id} schedule updated")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update task schedule: {e}")
            return False
    
    def get_scheduled_tasks(self, user_id: Optional[str] = None) -> List[ScheduledTask]:
        """Get all scheduled tasks, optionally filtered by user"""
        
        try:
            if user_id:
                return [task for task in self.scheduled_tasks.values() if task.metadata.get("user_id") == user_id]
            else:
                return list(self.scheduled_tasks.values())
            
        except Exception as e:
            self.logger.error(f"Failed to get scheduled tasks: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for task scheduler"""
        
        active_tasks = len([t for t in self.scheduled_tasks.values() if t.enabled])
        queued_tasks = len(self.execution_queue) if not self.priority_queue else 0
        
        return {
            "scheduling_count": self.scheduling_count,
            "execution_count": self.execution_count,
            "active_tasks": active_tasks,
            "queued_tasks": queued_tasks,
            "priority_queue_available": self.priority_queue is not None,
            "timezone_manager_available": self.timezone_manager is not None,
            "execution_optimizer_available": self.execution_optimizer is not None,
            "trigger_manager_available": self.trigger_manager is not None
        }
    
    async def cleanup(self):
        """Clean up task scheduler resources"""
        
        try:
            # Clean up components
            if hasattr(self.priority_queue, 'cleanup'):
                await self.priority_queue.cleanup()
            
            if hasattr(self.timezone_manager, 'cleanup'):
                await self.timezone_manager.cleanup()
            
            if hasattr(self.execution_optimizer, 'cleanup'):
                await self.execution_optimizer.cleanup()
            
            if hasattr(self.trigger_manager, 'cleanup'):
                await self.trigger_manager.cleanup()
                
            self.logger.info("Task scheduler cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class PriorityQueue:
    """Priority queue component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tasks: List[ScheduledTask] = []
    
    async def add_task(self, task: ScheduledTask):
        """Add task to priority queue"""
        
        try:
            self.tasks.append(task)
            self._sort_queue()
            
        except Exception as e:
            self.logger.error(f"Failed to add task to priority queue: {e}")
    
    async def get_next_tasks(self, limit: int) -> List[ScheduledTask]:
        """Get next tasks from priority queue"""
        
        try:
            current_time = datetime.now()
            ready_tasks = [
                task for task in self.tasks
                if task.enabled and task.next_execution and task.next_execution <= current_time
            ]
            return ready_tasks[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get next tasks from priority queue: {e}")
            return []
    
    async def remove_task(self, task_id: str):
        """Remove task from priority queue"""
        
        try:
            self.tasks = [task for task in self.tasks if task.id != task_id]
            
        except Exception as e:
            self.logger.error(f"Failed to remove task from priority queue: {e}")
    
    async def update_task(self, task: ScheduledTask):
        """Update task in priority queue"""
        
        try:
            # Remove old version
            await self.remove_task(task.id)
            # Add updated version
            await self.add_task(task)
            
        except Exception as e:
            self.logger.error(f"Failed to update task in priority queue: {e}")
    
    def _sort_queue(self):
        """Sort queue by priority and execution time"""
        
        try:
            self.tasks.sort(key=lambda t: (t.priority.value, t.next_execution or datetime.max))
            
        except Exception as e:
            self.logger.error(f"Queue sorting failed: {e}")
    
    async def cleanup(self):
        """Clean up priority queue resources"""
        pass


class TimezoneManager:
    """Timezone management component following Sentient patterns"""
    
    def __init__(self, default_timezone: str):
        self.default_timezone = default_timezone
        self.logger = logging.getLogger(__name__)
    
    async def convert_to_timezone(self, dt: datetime, target_timezone: str) -> datetime:
        """Convert datetime to target timezone"""
        
        try:
            # Simplified timezone conversion
            # In a real implementation, we'd use proper timezone libraries
            
            # For now, just return the datetime as-is
            # This is a placeholder for proper timezone handling
            
            return dt
            
        except Exception as e:
            self.logger.error(f"Timezone conversion failed: {e}")
            return dt
    
    async def get_current_time(self, timezone_str: str) -> datetime:
        """Get current time in specified timezone"""
        
        try:
            # Simplified timezone handling
            # In a real implementation, we'd use proper timezone libraries
            
            return datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to get current time in timezone: {e}")
            return datetime.now()
    
    async def cleanup(self):
        """Clean up timezone manager resources"""
        pass


class ExecutionOptimizer:
    """Execution optimization component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def optimize_execution_order(self, tasks: List[ScheduledTask]) -> List[ScheduledTask]:
        """Optimize execution order of tasks"""
        
        try:
            # Simple optimization: sort by priority and execution time
            # In a real implementation, we'd use more sophisticated algorithms
            
            optimized_tasks = sorted(tasks, key=lambda t: (t.priority.value, t.next_execution or datetime.max))
            return optimized_tasks
            
        except Exception as e:
            self.logger.error(f"Execution optimization failed: {e}")
            return tasks
    
    async def cleanup(self):
        """Clean up execution optimizer resources"""
        pass


class TriggerManager:
    """Trigger management component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_triggers: Dict[str, List[ScheduledTask]] = {}
    
    async def register_trigger(self, task: ScheduledTask):
        """Register a triggered task"""
        
        try:
            if task.schedule_type == ScheduleType.TRIGGERED:
                source = task.schedule_config.get("source", "unknown")
                event = task.schedule_config.get("event", "unknown")
                
                trigger_key = f"{source}:{event}"
                
                if trigger_key not in self.active_triggers:
                    self.active_triggers[trigger_key] = []
                
                self.active_triggers[trigger_key].append(task)
                self.logger.info(f"Registered trigger {trigger_key} for task {task.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to register trigger: {e}")
    
    async def fire_trigger(self, source: str, event: str, data: Dict[str, Any]) -> List[ScheduledTask]:
        """Fire a trigger and return tasks to execute"""
        
        try:
            trigger_key = f"{source}:{event}"
            
            if trigger_key not in self.active_triggers:
                return []
            
            triggered_tasks = []
            
            for task in self.active_triggers[trigger_key]:
                if task.enabled and await self._should_execute_triggered_task(task, data):
                    triggered_tasks.append(task)
            
            self.logger.info(f"Trigger {trigger_key} fired, {len(triggered_tasks)} tasks ready")
            return triggered_tasks
            
        except Exception as e:
            self.logger.error(f"Failed to fire trigger: {e}")
            return []
    
    async def _should_execute_triggered_task(self, task: ScheduledTask, data: Dict[str, Any]) -> bool:
        """Check if a triggered task should execute based on data"""
        
        try:
            filter_config = task.schedule_config.get("filter", {})
            
            # Simple filter matching
            # In a real implementation, we'd use more sophisticated filtering
            
            for key, value in filter_config.items():
                if key in data and data[key] != value:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trigger filter check failed: {e}")
            return True
    
    async def cleanup(self):
        """Clean up trigger manager resources"""
        pass


# Factory function for easy integration
def create_task_scheduler(config: Optional[SchedulerConfig] = None) -> TaskScheduler:
    """Create a task scheduler with default or custom configuration"""
    
    if config is None:
        config = SchedulerConfig()
    
    return TaskScheduler(config)
