"""
Task Parser - Enhanced with Sentient's Task Management Concepts

Integrates Sentient's proven task management patterns:
- Natural language task parsing and understanding
- Priority classification and scheduling detection
- Context-aware task creation and refinement
- Intelligent workflow planning and orchestration

Following the same integration pattern used for Leon, Empathy, AxiomEngine, and Memory.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/tasks')
    from prompts import TASK_CREATION_PROMPT
    SENTIENT_TASKS_AVAILABLE = True
except ImportError:
    SENTIENT_TASKS_AVAILABLE = False

@dataclass
class TaskSchedule:
    """Task scheduling configuration"""
    type: str  # "once", "recurring", "triggered"
    run_at: Optional[str] = None  # YYYY-MM-DDTHH:MM format for one-time
    frequency: Optional[str] = None  # "daily" or "weekly" for recurring
    days: Optional[List[str]] = None  # ["Monday", "Wednesday"] for weekly
    time: Optional[str] = None  # "HH:MM" 24-hour format
    source: Optional[str] = None  # Service source for triggered workflows
    event: Optional[str] = None  # Event type for triggered workflows
    filter: Optional[Dict[str, Any]] = None  # Filter conditions for triggered workflows

@dataclass
class ParsedTask:
    """Parsed task with all extracted information"""
    name: str
    description: str
    priority: int  # 0=High, 1=Medium, 2=Low
    schedule: TaskSchedule
    original_prompt: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    clarifying_questions: List[str] = field(default_factory=list)

@dataclass
class TaskParsingResult:
    """Result of task parsing process"""
    task: ParsedTask
    processing_time: float
    parsing_method: str
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class TaskParserConfig:
    """Configuration for task parsing"""
    enable_ai_parsing: bool = True
    enable_rule_based_parsing: bool = True
    enable_priority_detection: bool = True
    enable_schedule_detection: bool = True
    enable_context_integration: bool = True
    default_priority: int = 1
    default_time: str = "09:00"
    enable_debug_logging: bool = False
    timezone: str = "UTC"

class TaskParser:
    """
    Enhanced task parser integrating Sentient's task management concepts
    
    Provides:
    - Natural language task parsing and understanding
    - Priority classification and scheduling detection
    - Context-aware task creation and refinement
    - Intelligent workflow planning and orchestration
    """
    
    def __init__(self, config: TaskParserConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsing components
        self.ai_parser = None
        self.rule_parser = None
        self.priority_detector = None
        self.schedule_detector = None
        
        # Performance tracking
        self.parsing_count = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_parsing_components()
        
        self.logger.info("Task Parser initialized with Sentient concepts")
    
    def _initialize_parsing_components(self):
        """Initialize task parsing components using Sentient patterns"""
        
        try:
            # Initialize AI parser
            if self.config.enable_ai_parsing:
                self._initialize_ai_parser()
            
            # Initialize rule-based parser
            if self.config.enable_rule_based_parsing:
                self._initialize_rule_parser()
            
            # Initialize priority detector
            if self.config.enable_priority_detection:
                self._initialize_priority_detector()
            
            # Initialize schedule detector
            if self.config.enable_schedule_detection:
                self._initialize_schedule_detector()
            
            self.logger.info("Task parsing components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parsing components: {e}")
            self.logger.warning("Task parsing will use basic methods")
    
    def _initialize_ai_parser(self):
        """Initialize AI-based parsing component"""
        
        try:
            self.ai_parser = AITaskParser(self.config)
            self.logger.info("AI task parser initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI parser: {e}")
            self.ai_parser = None
    
    def _initialize_rule_parser(self):
        """Initialize rule-based parsing component"""
        
        try:
            self.rule_parser = RuleBasedTaskParser(self.config)
            self.logger.info("Rule-based task parser initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rule parser: {e}")
            self.rule_parser = None
    
    def _initialize_priority_detector(self):
        """Initialize priority detection component"""
        
        try:
            self.priority_detector = PriorityDetector()
            self.logger.info("Priority detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize priority detector: {e}")
            self.priority_detector = None
    
    def _initialize_schedule_detector(self):
        """Initialize schedule detection component"""
        
        try:
            self.schedule_detector = ScheduleDetector(self.config)
            self.logger.info("Schedule detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize schedule detector: {e}")
            self.schedule_detector = None
    
    async def parse_task(self, prompt: str, username: str = "user", 
                         context: Optional[Dict[str, Any]] = None) -> TaskParsingResult:
        """
        Parse natural language prompt into structured task data
        
        Args:
            prompt: Natural language task description
            username: Username for context
            context: Additional context information
            
        Returns:
            TaskParsingResult with parsed task and metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Try AI parsing first
            if self.ai_parser:
                try:
                    result = await self.ai_parser.parse(prompt, username, context)
                    if result and result.confidence > 0.8:
                        return self._create_parsing_result(result, start_time, "ai")
                except Exception as e:
                    self.logger.warning(f"AI parsing failed, falling back to rule-based: {e}")
            
            # Fall back to rule-based parsing
            if self.rule_parser:
                result = await self.rule_parser.parse(prompt, username, context)
                if result:
                    return self._create_parsing_result(result, start_time, "rule_based")
            
            # Final fallback to basic parsing
            result = await self._basic_parsing(prompt, username, context)
            return self._create_parsing_result(result, start_time, "basic")
            
        except Exception as e:
            self.logger.error(f"Task parsing failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return error result
            error_task = ParsedTask(
                name="Error: Task parsing failed",
                description=f"Failed to parse task: {str(e)}",
                priority=self.config.default_priority,
                schedule=TaskSchedule(type="once", run_at=None),
                original_prompt=prompt,
                confidence=0.0
            )
            
            return TaskParsingResult(
                task=error_task,
                processing_time=processing_time,
                parsing_method="error",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    def _create_parsing_result(self, task: ParsedTask, start_time: float, 
                              method: str) -> TaskParsingResult:
        """Create parsing result with timing and metadata"""
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Update performance tracking
        self.parsing_count += 1
        self.total_processing_time += processing_time
        
        return TaskParsingResult(
            task=task,
            processing_time=processing_time,
            parsing_method=method,
            confidence_score=task.confidence,
            metadata={
                "username": getattr(task, 'username', 'unknown'),
                "method": method,
                "components_used": self._get_used_components()
            }
        )
    
    def _get_used_components(self) -> List[str]:
        """Get list of parsing components currently in use"""
        
        components = []
        
        if self.ai_parser:
            components.append("ai_parser")
        if self.rule_parser:
            components.append("rule_parser")
        if self.priority_detector:
            components.append("priority_detector")
        if self.schedule_detector:
            components.append("schedule_detector")
        
        return components
    
    async def _basic_parsing(self, prompt: str, username: str, 
                            context: Optional[Dict[str, Any]]) -> ParsedTask:
        """Basic task parsing as final fallback"""
        
        try:
            # Extract basic task information
            name = self._extract_task_name(prompt)
            description = prompt
            
            # Detect priority
            priority = await self._detect_priority(prompt)
            
            # Detect schedule
            schedule = await self._detect_schedule(prompt)
            
            return ParsedTask(
                name=name,
                description=description,
                priority=priority,
                schedule=schedule,
                original_prompt=prompt,
                confidence=0.5,  # Lower confidence for basic parsing
                metadata={"parsing_method": "basic"}
            )
            
        except Exception as e:
            self.logger.error(f"Basic parsing failed: {e}")
            raise
    
    def _extract_task_name(self, prompt: str) -> str:
        """Extract a concise task name from prompt"""
        
        try:
            # Simple extraction: first sentence or first 50 characters
            sentences = re.split(r'[.!?]+', prompt)
            first_sentence = sentences[0].strip()
            
            if len(first_sentence) <= 50:
                return first_sentence
            
            # Truncate and add ellipsis
            return first_sentence[:47] + "..."
            
        except Exception as e:
            self.logger.error(f"Task name extraction failed: {e}")
            return "New Task"
    
    async def _detect_priority(self, prompt: str) -> int:
        """Detect task priority from prompt"""
        
        try:
            if not self.priority_detector:
                return self.config.default_priority
            
            priority = await self.priority_detector.detect(prompt)
            return priority
            
        except Exception as e:
            self.logger.error(f"Priority detection failed: {e}")
            return self.config.default_priority
    
    async def _detect_schedule(self, prompt: str) -> TaskSchedule:
        """Detect task schedule from prompt"""
        
        try:
            if not self.schedule_detector:
                return TaskSchedule(type="once", run_at=None)
            
            schedule = await self.schedule_detector.detect(prompt)
            return schedule
            
        except Exception as e:
            self.logger.error(f"Schedule detection failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for task parsing"""
        
        avg_processing_time = (self.total_processing_time / self.parsing_count 
                             if self.parsing_count > 0 else 0)
        
        return {
            "parsing_count": self.parsing_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "ai_parser_available": self.ai_parser is not None,
            "rule_parser_available": self.rule_parser is not None,
            "priority_detector_available": self.priority_detector is not None,
            "schedule_detector_available": self.schedule_detector is not None
        }
    
    async def cleanup(self):
        """Clean up task parsing resources"""
        
        try:
            if hasattr(self.ai_parser, 'cleanup'):
                await self.ai_parser.cleanup()
            
            if hasattr(self.rule_parser, 'cleanup'):
                await self.rule_parser.cleanup()
            
            if hasattr(self.priority_detector, 'cleanup'):
                await self.priority_detector.cleanup()
            
            if hasattr(self.schedule_detector, 'cleanup'):
                await self.schedule_detector.cleanup()
                
            self.logger.info("Task parser cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class AITaskParser:
    """AI-based task parsing component following Sentient patterns"""
    
    def __init__(self, config: TaskParserConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def parse(self, prompt: str, username: str, 
                    context: Optional[Dict[str, Any]]) -> Optional[ParsedTask]:
        """Parse task using AI-based approach"""
        
        try:
            # For now, we'll use a simplified AI parsing approach
            # In the future, we can integrate with an LLM for full parsing
            
            # Use Sentient's task creation prompt as inspiration
            if SENTIENT_TASKS_AVAILABLE:
                self.logger.debug("Using Sentient task creation patterns")
            
            # Parse using rule-based approach with AI-inspired heuristics
            parsed_data = await self._parse_with_heuristics(prompt, username)
            
            if parsed_data:
                return self._create_parsed_task(parsed_data, prompt)
            
            return None
            
        except Exception as e:
            self.logger.error(f"AI parsing failed: {e}")
            return None
    
    async def _parse_with_heuristics(self, prompt: str, username: str) -> Optional[Dict[str, Any]]:
        """Parse task using AI-inspired heuristics"""
        
        try:
            # Extract task name and description
            name = self._extract_ai_task_name(prompt)
            description = prompt
            
            # Detect priority using AI-inspired patterns
            priority = await self._detect_ai_priority(prompt)
            
            # Detect schedule using AI-inspired patterns
            schedule = await self._detect_ai_schedule(prompt)
            
            return {
                "name": name,
                "description": description,
                "priority": priority,
                "schedule": schedule
            }
            
        except Exception as e:
            self.logger.error(f"AI heuristic parsing failed: {e}")
            return None
    
    def _extract_ai_task_name(self, prompt: str) -> str:
        """Extract task name using AI-inspired patterns"""
        
        try:
            # Look for action verbs and key nouns
            action_patterns = [
                r'(?:remind me to|i need to|please|can you|help me)\s+([^.]+)',
                r'([A-Z][^.!?]*?)(?:\s+(?:tomorrow|today|next|every|daily|weekly))',
                r'([A-Z][^.!?]*?)(?:\s+(?:at|on|in|by|before|after))'
            ]
            
            for pattern in action_patterns:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Clean up the name
                    name = re.sub(r'^(?:to|about|regarding)\s+', '', name, flags=re.IGNORECASE)
                    return name.capitalize()
            
            # Fallback to first sentence
            sentences = re.split(r'[.!?]+', prompt)
            first_sentence = sentences[0].strip()
            return first_sentence[:50] if len(first_sentence) <= 50 else first_sentence[:47] + "..."
            
        except Exception as e:
            self.logger.error(f"AI task name extraction failed: {e}")
            return "New Task"
    
    async def _detect_ai_priority(self, prompt: str) -> int:
        """Detect priority using AI-inspired patterns"""
        
        try:
            prompt_lower = prompt.lower()
            
            # High priority indicators
            high_priority_words = [
                "urgent", "asap", "immediately", "now", "emergency", "critical",
                "deadline", "due", "important", "priority", "urgently"
            ]
            
            if any(word in prompt_lower for word in high_priority_words):
                return 0
            
            # Low priority indicators
            low_priority_words = [
                "sometime", "when convenient", "low priority", "not urgent",
                "can wait", "whenever", "no rush"
            ]
            
            if any(word in prompt_lower for word in low_priority_words):
                return 2
            
            # Default to medium priority
            return 1
            
        except Exception as e:
            self.logger.error(f"AI priority detection failed: {e}")
            return 1
    
    async def _detect_ai_schedule(self, prompt: str) -> TaskSchedule:
        """Detect schedule using AI-inspired patterns"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Check for recurring patterns
            if any(word in prompt_lower for word in ["every", "daily", "weekly", "monthly", "yearly"]):
                return await self._parse_recurring_schedule(prompt)
            
            # Check for triggered workflows
            if any(word in prompt_lower for word in ["when", "every time", "on", "triggered"]):
                return await self._parse_triggered_schedule(prompt)
            
            # Check for specific times
            if any(word in prompt_lower for word in ["tomorrow", "today", "next", "at", "on"]):
                return await self._parse_one_time_schedule(prompt)
            
            # Default to immediate execution
            return TaskSchedule(type="once", run_at=None)
            
        except Exception as e:
            self.logger.error(f"AI schedule detection failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    async def _parse_recurring_schedule(self, prompt: str) -> TaskSchedule:
        """Parse recurring schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Determine frequency
            if "daily" in prompt_lower:
                frequency = "daily"
            elif "weekly" in prompt_lower:
                frequency = "weekly"
            else:
                frequency = "daily"  # Default
            
            # Extract time
            time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', prompt_lower)
            time_str = "09:00"  # Default time
            
            if time_match:
                hour = int(time_match.group(1))
                minute = time_match.group(2) or "00"
                ampm = time_match.group(3)
                
                # Convert to 24-hour format
                if ampm == "pm" and hour != 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
                
                time_str = f"{hour:02d}:{minute}"
            
            # Extract days for weekly frequency
            days = ["Monday"]  # Default
            if frequency == "weekly":
                day_patterns = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                found_days = [day.title() for day in day_patterns if day in prompt_lower]
                if found_days:
                    days = found_days
            
            return TaskSchedule(
                type="recurring",
                frequency=frequency,
                days=days,
                time=time_str
            )
            
        except Exception as e:
            self.logger.error(f"Recurring schedule parsing failed: {e}")
            return TaskSchedule(type="recurring", frequency="daily", time="09:00")
    
    async def _parse_triggered_schedule(self, prompt: str) -> TaskSchedule:
        """Parse triggered schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Determine source and event
            source = "unknown"
            event = "unknown"
            
            if "email" in prompt_lower:
                source = "email"
                event = "new_email"
            elif "calendar" in prompt_lower or "event" in prompt_lower:
                source = "calendar"
                event = "new_event"
            
            # Extract filter conditions
            filter_conditions = {}
            
            # Look for specific email addresses
            email_match = re.search(r'from\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', prompt_lower)
            if email_match:
                filter_conditions["from"] = email_match.group(1)
            
            return TaskSchedule(
                type="triggered",
                source=source,
                event=event,
                filter=filter_conditions
            )
            
        except Exception as e:
            self.logger.error(f"Triggered schedule parsing failed: {e}")
            return TaskSchedule(type="triggered", source="unknown", event="unknown")
    
    async def _parse_one_time_schedule(self, prompt: str) -> TaskSchedule:
        """Parse one-time schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Check if it's immediate execution
            if any(word in prompt_lower for word in ["now", "immediately", "asap"]):
                return TaskSchedule(type="once", run_at=None)
            
            # Extract date and time
            # This is a simplified implementation
            # In a real system, we'd use more sophisticated date parsing
            
            return TaskSchedule(type="once", run_at=None)
            
        except Exception as e:
            self.logger.error(f"One-time schedule parsing failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    def _create_parsed_task(self, parsed_data: Dict[str, Any], original_prompt: str) -> ParsedTask:
        """Create ParsedTask from parsed data"""
        
        return ParsedTask(
            name=parsed_data["name"],
            description=parsed_data["description"],
            priority=parsed_data["priority"],
            schedule=parsed_data["schedule"],
            original_prompt=original_prompt,
            confidence=0.9,  # High confidence for AI parsing
            metadata={"parsing_method": "ai_heuristics"}
        )
    
    async def cleanup(self):
        """Clean up AI parser resources"""
        pass


class RuleBasedTaskParser:
    """Rule-based task parsing component following Sentient patterns"""
    
    def __init__(self, config: TaskParserConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def parse(self, prompt: str, username: str, 
                    context: Optional[Dict[str, Any]]) -> Optional[ParsedTask]:
        """Parse task using rule-based approach"""
        
        try:
            # Extract basic information using rules
            name = self._extract_task_name(prompt)
            description = prompt
            
            # Apply priority rules
            priority = self._apply_priority_rules(prompt)
            
            # Apply schedule rules
            schedule = self._apply_schedule_rules(prompt)
            
            return ParsedTask(
                name=name,
                description=description,
                priority=priority,
                schedule=schedule,
                original_prompt=prompt,
                confidence=0.7,  # Medium confidence for rule-based parsing
                metadata={"parsing_method": "rule_based"}
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based parsing failed: {e}")
            return None
    
    def _extract_task_name(self, prompt: str) -> str:
        """Extract task name using rule-based approach"""
        
        try:
            # Simple rule: first sentence or first 50 characters
            sentences = re.split(r'[.!?]+', prompt)
            first_sentence = sentences[0].strip()
            
            if len(first_sentence) <= 50:
                return first_sentence
            
            return first_sentence[:47] + "..."
            
        except Exception as e:
            self.logger.error(f"Task name extraction failed: {e}")
            return "New Task"
    
    def _apply_priority_rules(self, prompt: str) -> int:
        """Apply priority rules to determine task priority"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Rule 1: Check for urgent keywords
            urgent_keywords = ["urgent", "asap", "emergency", "critical", "deadline"]
            if any(keyword in prompt_lower for keyword in urgent_keywords):
                return 0
            
            # Rule 2: Check for low priority keywords
            low_priority_keywords = ["sometime", "when convenient", "low priority"]
            if any(keyword in prompt_lower for keyword in low_priority_keywords):
                return 2
            
            # Rule 3: Default to medium priority
            return 1
            
        except Exception as e:
            self.logger.error(f"Priority rules application failed: {e}")
            return 1
    
    def _apply_schedule_rules(self, prompt: str) -> TaskSchedule:
        """Apply schedule rules to determine task schedule"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Rule 1: Check for recurring patterns
            if any(word in prompt_lower for word in ["every", "daily", "weekly"]):
                return TaskSchedule(type="recurring", frequency="daily", time="09:00")
            
            # Rule 2: Check for immediate execution
            if any(word in prompt_lower for word in ["now", "immediately"]):
                return TaskSchedule(type="once", run_at=None)
            
            # Rule 3: Default to immediate execution
            return TaskSchedule(type="once", run_at=None)
            
        except Exception as e:
            self.logger.error(f"Schedule rules application failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    async def cleanup(self):
        """Clean up rule-based parser resources"""
        pass


class PriorityDetector:
    """Priority detection component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def detect(self, prompt: str) -> int:
        """Detect task priority from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # High priority (0) - Urgent, important, deadlines
            high_priority_indicators = [
                "urgent", "asap", "immediately", "now", "emergency", "critical",
                "deadline", "due", "important", "priority", "urgently", "rush"
            ]
            
            if any(indicator in prompt_lower for indicator in high_priority_indicators):
                return 0
            
            # Low priority (2) - Can be done anytime, not urgent
            low_priority_indicators = [
                "sometime", "when convenient", "low priority", "not urgent",
                "can wait", "whenever", "no rush", "backlog", "nice to have"
            ]
            
            if any(indicator in prompt_lower for indicator in low_priority_indicators):
                return 2
            
            # Medium priority (1) - Default, standard tasks
            return 1
            
        except Exception as e:
            self.logger.error(f"Priority detection failed: {e}")
            return 1
    
    async def cleanup(self):
        """Clean up priority detector resources"""
        pass


class ScheduleDetector:
    """Schedule detection component following Sentient patterns"""
    
    def __init__(self, config: TaskParserConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def detect(self, prompt: str) -> TaskSchedule:
        """Detect task schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Check for recurring patterns
            if any(word in prompt_lower for word in ["every", "daily", "weekly", "monthly"]):
                return await self._detect_recurring_schedule(prompt)
            
            # Check for triggered workflows
            if any(word in prompt_lower for word in ["when", "every time", "on", "triggered"]):
                return await self._detect_triggered_schedule(prompt)
            
            # Check for specific times
            if any(word in prompt_lower for word in ["tomorrow", "today", "next", "at", "on"]):
                return await self._detect_one_time_schedule(prompt)
            
            # Default to immediate execution
            return TaskSchedule(type="once", run_at=None)
            
        except Exception as e:
            self.logger.error(f"Schedule detection failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    async def _detect_recurring_schedule(self, prompt: str) -> TaskSchedule:
        """Detect recurring schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Determine frequency
            if "daily" in prompt_lower:
                frequency = "daily"
            elif "weekly" in prompt_lower:
                frequency = "weekly"
            else:
                frequency = "daily"  # Default
            
            # Extract time (simplified)
            time_str = self.config.default_time
            
            # Extract days for weekly frequency
            days = ["Monday"]  # Default
            if frequency == "weekly":
                day_patterns = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                found_days = [day.title() for day in day_patterns if day in prompt_lower]
                if found_days:
                    days = found_days
            
            return TaskSchedule(
                type="recurring",
                frequency=frequency,
                days=days,
                time=time_str
            )
            
        except Exception as e:
            self.logger.error(f"Recurring schedule detection failed: {e}")
            return TaskSchedule(type="recurring", frequency="daily", time="09:00")
    
    async def _detect_triggered_schedule(self, prompt: str) -> TaskSchedule:
        """Detect triggered schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Determine source and event
            source = "unknown"
            event = "unknown"
            
            if "email" in prompt_lower:
                source = "email"
                event = "new_email"
            elif "calendar" in prompt_lower or "event" in prompt_lower:
                source = "calendar"
                event = "new_event"
            
            return TaskSchedule(
                type="triggered",
                source=source,
                event=event,
                filter={}
            )
            
        except Exception as e:
            self.logger.error(f"Triggered schedule detection failed: {e}")
            return TaskSchedule(type="triggered", source="unknown", event="unknown")
    
    async def _detect_one_time_schedule(self, prompt: str) -> TaskSchedule:
        """Detect one-time schedule from prompt"""
        
        try:
            prompt_lower = prompt.lower()
            
            # Check if it's immediate execution
            if any(word in prompt_lower for word in ["now", "immediately", "asap"]):
                return TaskSchedule(type="once", run_at=None)
            
            # For now, default to immediate execution
            # In a real implementation, we'd parse specific dates and times
            
            return TaskSchedule(type="once", run_at=None)
            
        except Exception as e:
            self.logger.error(f"One-time schedule detection failed: {e}")
            return TaskSchedule(type="once", run_at=None)
    
    async def cleanup(self):
        """Clean up schedule detector resources"""
        pass


# Factory function for easy integration
def create_task_parser(config: Optional[TaskParserConfig] = None) -> TaskParser:
    """Create a task parser with default or custom configuration"""
    
    if config is None:
        config = TaskParserConfig()
    
    return TaskParser(config)
