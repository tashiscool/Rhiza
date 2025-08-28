"""
Tasks Module - Enhanced with Sentient Task Management Concepts

This module integrates Sentient's proven task management patterns into The Mesh:
- Natural language task parsing and understanding
- Workflow execution and orchestration
- Intelligent task scheduling and prioritization
- Context-aware task management and optimization

Following the same integration pattern used for Leon, Empathy, AxiomEngine, and Memory.
"""

# Import task components
from .task_parser import (
    TaskParser,
    TaskParserConfig,
    ParsedTask,
    TaskSchedule,
    TaskParsingResult,
    create_task_parser
)

from .workflow_engine import (
    WorkflowEngine,
    WorkflowEngineConfig,
    WorkflowStep,
    WorkflowExecution,
    WorkflowResult,
    TaskStatus,
    TaskStepStatus,
    create_workflow_engine
)

from .scheduler import (
    TaskScheduler,
    SchedulerConfig,
    ScheduledTask,
    ExecutionContext,
    SchedulingResult,
    ScheduleType,
    TaskPriority,
    create_task_scheduler
)

# Export all task capabilities
__all__ = [
    # Task Parsing
    'TaskParser',
    'TaskParserConfig',
    'ParsedTask',
    'TaskSchedule',
    'TaskParsingResult',
    'create_task_parser',
    
    # Workflow Engine
    'WorkflowEngine',
    'WorkflowEngineConfig',
    'WorkflowStep',
    'WorkflowExecution',
    'WorkflowResult',
    'TaskStatus',
    'TaskStepStatus',
    'create_workflow_engine',
    
    # Task Scheduler
    'TaskScheduler',
    'SchedulerConfig',
    'ScheduledTask',
    'ExecutionContext',
    'SchedulingResult',
    'ScheduleType',
    'TaskPriority',
    'create_task_scheduler'
]

# Version information
__version__ = "3.0.0"
__description__ = "Enhanced task management with Sentient concepts"
__author__ = "The Mesh Development Team"
