"""
Mesh Choice Rehearser
====================

Allows users to privately rehearse decisions and responses
to scenarios, building confidence and improving decision-making skills.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random

from .scenario_generator import Scenario, ScenarioType, ScenarioComplexity

logger = logging.getLogger(__name__)


class RehearsalType(Enum):
    """Types of rehearsal sessions"""
    DECISION_REHEARSAL = "decision_rehearsal"      # Practice making decisions
    CONVERSATION_REHEARSAL = "conversation_rehearsal"  # Practice conversations
    RESPONSE_REHEARSAL = "response_rehearsal"      # Practice responses
    ACTION_REHEARSAL = "action_rehearsal"          # Practice taking actions


class RehearsalStatus(Enum):
    """Status of a rehearsal session"""
    ACTIVE = "active"                    # Currently being rehearsed
    PAUSED = "paused"                    # Temporarily paused
    COMPLETED = "completed"               # Rehearsal finished
    ABANDONED = "abandoned"               # Rehearsal abandoned


@dataclass
class RehearsalSession:
    """A rehearsal session for a specific scenario"""
    session_id: str
    scenario_id: str
    user_id: str
    rehearsal_type: RehearsalType
    status: RehearsalStatus
    created_at: datetime
    
    # Rehearsal progress
    current_step: int = 0
    total_steps: int = 0
    confidence_score: float = 0.0  # 0.0 to 1.0
    comfort_level: float = 0.0     # 0.0 to 1.0
    
    # User responses and choices
    user_choices: List[Dict[str, Any]] = field(default_factory=list)
    user_responses: List[str] = field(default_factory=list)
    alternative_paths: List[Dict[str, Any]] = field(default_factory=list)
    
    # Feedback and learning
    self_assessment: Optional[str] = None
    areas_for_improvement: List[str] = field(default_factory=list)
    strengths_identified: List[str] = field(default_factory=list)
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        content = f"{self.scenario_id}{self.user_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "user_id": self.user_id,
            "rehearsal_type": self.rehearsal_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "confidence_score": self.confidence_score,
            "comfort_level": self.comfort_level,
            "user_choices": self.user_choices,
            "user_responses": self.user_responses,
            "alternative_paths": self.alternative_paths,
            "self_assessment": self.self_assessment,
            "areas_for_improvement": self.areas_for_improvement,
            "strengths_identified": self.strengths_identified
        }


@dataclass
class RehearsalStep:
    """A single step in a rehearsal session"""
    step_id: str
    session_id: str
    step_number: int
    step_type: str
    description: str
    instructions: str
    options: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    
    # User interaction
    user_response: Optional[str] = None
    user_confidence: Optional[float] = None
    feedback: Optional[str] = None
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = self._generate_step_id()
    
    def _generate_step_id(self) -> str:
        """Generate unique step ID"""
        content = f"{self.session_id}{self.step_number}{self.step_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ChoiceRehearser:
    """
    Manages rehearsal sessions for decision-making and response practice
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.sessions: Dict[str, RehearsalSession] = {}
        self.steps: Dict[str, List[RehearsalStep]] = {}  # session_id -> steps
        self.rehearsal_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, session_id
        
        # Rehearsal templates for different scenario types
        self.rehearsal_templates = self._initialize_rehearsal_templates()
    
    def _initialize_rehearsal_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rehearsal templates for different scenario types"""
        return {
            ScenarioType.CONFLICT_RESOLUTION.value: {
                "steps": [
                    {
                        "type": "situation_analysis",
                        "description": "Analyze the conflict situation",
                        "instructions": "Take time to understand the conflict from all perspectives. What are the underlying issues?",
                        "options": ["Focus on facts", "Consider emotions", "Identify stakeholders", "Assess urgency"]
                    },
                    {
                        "type": "perspective_taking",
                        "description": "Consider different perspectives",
                        "instructions": "Put yourself in each participant's shoes. What are their concerns and motivations?",
                        "options": ["Team member A's view", "Team member B's view", "Organizational impact", "Personal relationships"]
                    },
                    {
                        "type": "solution_brainstorming",
                        "description": "Brainstorm potential solutions",
                        "instructions": "Generate multiple approaches to resolving the conflict. Think creatively.",
                        "options": ["Compromise", "Collaboration", "Accommodation", "Avoidance", "Competition"]
                    },
                    {
                        "type": "implementation_planning",
                        "description": "Plan the implementation",
                        "instructions": "How will you implement your chosen solution? What steps are needed?",
                        "options": ["Set clear expectations", "Establish timeline", "Define success metrics", "Plan follow-up"]
                    }
                ]
            },
            ScenarioType.DECISION_MAKING.value: {
                "steps": [
                    {
                        "type": "problem_definition",
                        "description": "Define the decision problem",
                        "instructions": "Clearly articulate what decision needs to be made and why it's important.",
                        "options": ["Identify core issue", "Assess urgency", "Define scope", "Consider stakeholders"]
                    },
                    {
                        "type": "option_generation",
                        "description": "Generate decision options",
                        "instructions": "Brainstorm all possible options, even unconventional ones.",
                        "options": ["Traditional approach", "Innovative solution", "Hybrid approach", "Status quo", "Radical change"]
                    },
                    {
                        "type": "criteria_definition",
                        "description": "Define decision criteria",
                        "instructions": "What factors will you use to evaluate the options?",
                        "options": ["Cost", "Time", "Risk", "Impact", "Feasibility", "Ethics"]
                    },
                    {
                        "type": "option_evaluation",
                        "description": "Evaluate each option",
                        "instructions": "Systematically evaluate each option against your criteria.",
                        "options": ["Score each option", "Consider trade-offs", "Assess risks", "Evaluate impact"]
                    },
                    {
                        "type": "decision_making",
                        "description": "Make the decision",
                        "instructions": "Based on your analysis, make your decision and commit to it.",
                        "options": ["Choose best option", "Document reasoning", "Plan communication", "Prepare for implementation"]
                    }
                ]
            },
            ScenarioType.EMPATHY_TRAINING.value: {
                "steps": [
                    {
                        "type": "emotional_recognition",
                        "description": "Recognize emotional states",
                        "instructions": "What emotions is the person experiencing? How can you tell?",
                        "options": ["Facial expressions", "Body language", "Tone of voice", "Word choice", "Behavior changes"]
                    },
                    {
                        "type": "perspective_understanding",
                        "description": "Understand their perspective",
                        "instructions": "What might be causing these emotions? What's their experience?",
                        "options": ["Personal situation", "Work stress", "Health issues", "Family concerns", "Career worries"]
                    },
                    {
                        "type": "empathic_response",
                        "description": "Practice empathic responses",
                        "instructions": "How can you respond with genuine empathy and understanding?",
                        "options": ["Validate feelings", "Show understanding", "Offer support", "Maintain boundaries", "Ask questions"]
                    },
                    {
                        "type": "support_planning",
                        "description": "Plan ongoing support",
                        "instructions": "What ongoing support might be helpful? How can you provide it appropriately?",
                        "options": ["Check in regularly", "Offer resources", "Connect with others", "Respect boundaries", "Follow up"]
                    }
                ]
            }
        }
    
    def start_rehearsal(self, scenario: Scenario, user_id: str, 
                        rehearsal_type: RehearsalType = RehearsalType.DECISION_REHEARSAL) -> RehearsalSession:
        """Start a new rehearsal session for a scenario"""
        try:
            # Create rehearsal session
            session = RehearsalSession(
                session_id="",
                scenario_id=scenario.scenario_id,
                user_id=user_id,
                rehearsal_type=rehearsal_type,
                status=RehearsalStatus.ACTIVE,
                created_at=datetime.utcnow(),
                total_steps=self._get_step_count(scenario.scenario_type),
                started_at=datetime.utcnow()
            )
            
            # Create rehearsal steps
            steps = self._create_rehearsal_steps(session, scenario)
            
            # Store session and steps
            self.sessions[session.session_id] = session
            self.steps[session.session_id] = steps
            
            # Track rehearsal start
            self.rehearsal_history.append((datetime.utcnow(), "started", session.session_id))
            
            logger.info(f"Started {rehearsal_type.value} rehearsal for scenario: {scenario.title}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start rehearsal: {e}")
            raise
    
    def _get_step_count(self, scenario_type: ScenarioType) -> int:
        """Get number of steps for a scenario type"""
        template = self.rehearsal_templates.get(scenario_type.value, {})
        return len(template.get("steps", []))
    
    def _create_rehearsal_steps(self, session: RehearsalSession, scenario: Scenario) -> List[RehearsalStep]:
        """Create rehearsal steps for a session"""
        template = self.rehearsal_templates.get(scenario.scenario_type.value, {})
        steps = []
        
        for i, step_template in enumerate(template.get("steps", [])):
            step = RehearsalStep(
                step_id="",
                session_id=session.session_id,
                step_number=i + 1,
                step_type=step_template["type"],
                description=step_template["description"],
                instructions=step_template["instructions"],
                options=step_template.get("options", []),
                expected_outcome=step_template.get("expected_outcome", "")
            )
            steps.append(step)
        
        return steps
    
    def get_current_step(self, session_id: str) -> Optional[RehearsalStep]:
        """Get the current step for a rehearsal session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if session.current_step >= len(self.steps.get(session_id, [])):
            return None
        
        steps = self.steps.get(session_id, [])
        return steps[session.current_step] if steps else None
    
    def advance_step(self, session_id: str, user_response: str, 
                    user_confidence: float = 0.5) -> Optional[RehearsalStep]:
        """Advance to the next step in a rehearsal session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if session.status != RehearsalStatus.ACTIVE:
            logger.warning(f"Session {session_id} is not active")
            return None
        
        # Update current step with user response
        current_step = self.get_current_step(session_id)
        if current_step:
            current_step.user_response = user_response
            current_step.user_confidence = user_confidence
            
            # Store user choice
            session.user_choices.append({
                "step": current_step.step_number,
                "response": user_response,
                "confidence": user_confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update session progress
            session.current_step += 1
            
            # Check if session is complete
            if session.current_step >= session.total_steps:
                self._complete_rehearsal(session_id)
                return None
            
            # Return next step
            return self.get_current_step(session_id)
        
        return None
    
    def _complete_rehearsal(self, session_id: str):
        """Mark a rehearsal session as complete"""
        session = self.sessions[session_id]
        session.status = RehearsalStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        
        # Calculate final scores
        session.confidence_score = self._calculate_confidence_score(session)
        session.comfort_level = self._calculate_comfort_level(session)
        
        # Track completion
        self.rehearsal_history.append((datetime.utcnow(), "completed", session_id))
        
        logger.info(f"Completed rehearsal session: {session_id}")
    
    def _calculate_confidence_score(self, session: RehearsalSession) -> float:
        """Calculate overall confidence score for the session"""
        if not session.user_choices:
            return 0.0
        
        total_confidence = sum(choice["confidence"] for choice in session.user_choices)
        return total_confidence / len(session.user_choices)
    
    def _calculate_comfort_level(self, session: RehearsalSession) -> float:
        """Calculate overall comfort level for the session"""
        if not session.user_choices:
            return 0.0
        
        # Comfort level increases with practice and positive responses
        base_comfort = 0.5
        confidence_bonus = session.confidence_score * 0.3
        practice_bonus = min(len(session.user_choices) * 0.1, 0.2)
        
        return min(base_comfort + confidence_bonus + practice_bonus, 1.0)
    
    def pause_rehearsal(self, session_id: str) -> bool:
        """Pause a rehearsal session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.status == RehearsalStatus.ACTIVE:
            session.status = RehearsalStatus.PAUSED
            self.rehearsal_history.append((datetime.utcnow(), "paused", session_id))
            logger.info(f"Paused rehearsal session: {session_id}")
            return True
        
        return False
    
    def resume_rehearsal(self, session_id: str) -> bool:
        """Resume a paused rehearsal session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.status == RehearsalStatus.PAUSED:
            session.status = RehearsalStatus.ACTIVE
            self.rehearsal_history.append((datetime.utcnow(), "resumed", session_id))
            logger.info(f"Resumed rehearsal session: {session_id}")
            return True
        
        return False
    
    def abandon_rehearsal(self, session_id: str, reason: str = "User choice") -> bool:
        """Abandon a rehearsal session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = RehearsalStatus.ABANDONED
        session.self_assessment = f"Abandoned: {reason}"
        
        self.rehearsal_history.append((datetime.utcnow(), "abandoned", session_id))
        logger.info(f"Abandoned rehearsal session: {session_id}: {reason}")
        return True
    
    def add_self_assessment(self, session_id: str, assessment: str, 
                           areas_for_improvement: List[str], strengths_identified: List[str]) -> bool:
        """Add self-assessment to a completed rehearsal session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session.status != RehearsalStatus.COMPLETED:
            logger.warning(f"Cannot add assessment to incomplete session: {session_id}")
            return False
        
        session.self_assessment = assessment
        session.areas_for_improvement = areas_for_improvement
        session.strengths_identified = strengths_identified
        
        logger.info(f"Added self-assessment to session: {session_id}")
        return True
    
    def explore_alternative_path(self, session_id: str, step_number: int, 
                                alternative_response: str) -> Dict[str, Any]:
        """Explore an alternative path from a specific step"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        if step_number < 1 or step_number > session.total_steps:
            return {"error": "Invalid step number"}
        
        # Create alternative path analysis
        alternative_path = {
            "step_number": step_number,
            "original_response": None,
            "alternative_response": alternative_response,
            "potential_outcomes": self._generate_alternative_outcomes(session, step_number, alternative_response),
            "explored_at": datetime.utcnow().isoformat()
        }
        
        # Get original response if available
        if step_number <= len(session.user_choices):
            original_choice = session.user_choices[step_number - 1]
            alternative_path["original_response"] = original_choice["response"]
        
        # Store alternative path
        session.alternative_paths.append(alternative_path)
        
        return alternative_path
    
    def _generate_alternative_outcomes(self, session: RehearsalSession, step_number: int, 
                                     alternative_response: str) -> List[str]:
        """Generate potential outcomes for an alternative response"""
        # This is a simplified version - in practice, this could use more sophisticated analysis
        outcomes = [
            "Different emotional response from participants",
            "Alternative solution path",
            "Modified timeline or approach",
            "Different stakeholder reactions",
            "Alternative long-term consequences"
        ]
        
        # Add some randomness to make it interesting
        random.shuffle(outcomes)
        return outcomes[:3]  # Return 3 outcomes
    
    def get_rehearsal_summary(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of rehearsal sessions"""
        sessions_to_analyze = self.sessions.values()
        if user_id:
            sessions_to_analyze = [s for s in sessions_to_analyze if s.user_id == user_id]
        
        total_sessions = len(sessions_to_analyze)
        completed_sessions = len([s for s in sessions_to_analyze if s.status == RehearsalStatus.COMPLETED])
        active_sessions = len([s for s in sessions_to_analyze if s.status == RehearsalStatus.ACTIVE])
        
        # Calculate average scores
        confidence_scores = [s.confidence_score for s in sessions_to_analyze if s.confidence_score > 0]
        comfort_scores = [s.comfort_level for s in sessions_to_analyze if s.comfort_level > 0]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        avg_comfort = sum(comfort_scores) / len(comfort_scores) if comfort_scores else 0.0
        
        # Sessions by type
        sessions_by_type = {}
        for session in sessions_to_analyze:
            session_type = session.rehearsal_type.value
            sessions_by_type[session_type] = sessions_by_type.get(session_type, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "active_sessions": active_sessions,
            "average_confidence": avg_confidence,
            "average_comfort": avg_comfort,
            "sessions_by_type": sessions_by_type,
            "recent_sessions": self._get_recent_sessions(user_id, 10)
        }
    
    def _get_recent_sessions(self, user_id: Optional[str], count: int) -> List[Dict[str, Any]]:
        """Get recent rehearsal sessions"""
        sessions_to_analyze = self.sessions.values()
        if user_id:
            sessions_to_analyze = [s for s in sessions_to_analyze if s.user_id == user_id]
        
        sorted_sessions = sorted(sessions_to_analyze, key=lambda s: s.created_at, reverse=True)
        recent_sessions = sorted_sessions[:count]
        
        return [
            {
                "session_id": s.session_id,
                "scenario_id": s.scenario_id,
                "type": s.rehearsal_type.value,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
                "confidence_score": s.confidence_score,
                "comfort_level": s.comfort_level
            }
            for s in recent_sessions
        ]
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a rehearsal session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        steps = self.steps.get(session_id, [])
        
        return {
            "session": session.to_dict(),
            "steps": [
                {
                    "step_id": step.step_id,
                    "step_number": step.step_number,
                    "type": step.step_type,
                    "description": step.description,
                    "instructions": step.instructions,
                    "options": step.options,
                    "expected_outcome": step.expected_outcome,
                    "user_response": step.user_response,
                    "user_confidence": step.user_confidence,
                    "feedback": step.feedback
                }
                for step in steps
            ],
            "progress": {
                "current_step": session.current_step,
                "total_steps": session.total_steps,
                "completion_percentage": (session.current_step / session.total_steps * 100) if session.total_steps > 0 else 0
            }
        }
