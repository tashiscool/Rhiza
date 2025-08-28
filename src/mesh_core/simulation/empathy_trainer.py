"""
Mesh Empathy Trainer
====================

Builds empathy through simulation, emotional intelligence training,
and perspective-taking exercises.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib

from .scenario_generator import Scenario, Persona, ScenarioType, ScenarioComplexity

logger = logging.getLogger(__name__)


class EmpathySkill(Enum):
    """Types of empathy skills to train"""
    EMOTIONAL_RECOGNITION = "emotional_recognition"      # Recognize emotions in others
    PERSPECTIVE_TAKING = "perspective_taking"            # See from others' viewpoints
    ACTIVE_LISTENING = "active_listening"                # Listen with full attention
    EMOTIONAL_VALIDATION = "emotional_validation"        # Validate others' feelings
    COMPASSIONATE_RESPONSE = "compassionate_response"    # Respond with compassion
    BOUNDARY_MAINTENANCE = "boundary_maintenance"        # Maintain healthy boundaries


class TrainingLevel(Enum):
    """Levels of empathy training"""
    BEGINNER = "beginner"                # Basic empathy concepts
    INTERMEDIATE = "intermediate"        # Applied empathy skills
    ADVANCED = "advanced"                # Complex empathy situations
    EXPERT = "expert"                    # Expert-level empathy mastery


@dataclass
class EmpathyExercise:
    """An empathy training exercise"""
    exercise_id: str
    skill_focus: EmpathySkill
    training_level: TrainingLevel
    title: str
    description: str
    instructions: str
    scenario_context: str
    personas_involved: List[Persona]
    
    # Exercise details
    duration_minutes: int
    difficulty_score: float  # 0.0 to 1.0
    learning_objectives: List[str]
    
    # Success criteria
    success_indicators: List[str]
    common_pitfalls: List[str]
    
    # Metadata
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.exercise_id:
            self.exercise_id = self._generate_exercise_id()
    
    def _generate_exercise_id(self) -> str:
        """Generate unique exercise ID"""
        content = f"{self.title}{self.skill_focus.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exercise to dictionary"""
        return {
            "exercise_id": self.exercise_id,
            "skill_focus": self.skill_focus.value,
            "training_level": self.training_level.value,
            "title": self.title,
            "description": self.description,
            "instructions": self.instructions,
            "scenario_context": self.scenario_context,
            "personas_involved": [p.to_dict() for p in self.personas_involved],
            "duration_minutes": self.duration_minutes,
            "difficulty_score": self.difficulty_score,
            "learning_objectives": self.learning_objectives,
            "success_indicators": self.success_indicators,
            "common_pitfalls": self.common_pitfalls,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags
        }


@dataclass
class TrainingSession:
    """A training session for empathy development"""
    session_id: str
    exercise_id: str
    user_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # User performance
    responses: List[Dict[str, Any]] = field(default_factory=list)
    empathy_score: float = 0.0  # 0.0 to 1.0
    skill_scores: Dict[str, float] = field(default_factory=dict)  # skill -> score
    
    # Feedback and learning
    self_reflection: Optional[str] = None
    areas_for_improvement: List[str] = field(default_factory=list)
    strengths_demonstrated: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        content = f"{self.exercise_id}{self.user_id}{self.started_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "exercise_id": self.exercise_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "responses": self.responses,
            "empathy_score": self.empathy_score,
            "skill_scores": self.skill_scores,
            "self_reflection": self.self_reflection,
            "areas_for_improvement": self.areas_for_improvement,
            "strengths_demonstrated": self.strengths_demonstrated
        }


class EmpathyTrainer:
    """
    Provides empathy training through structured exercises and feedback
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.exercises: Dict[str, EmpathyExercise] = {}
        self.training_sessions: Dict[str, TrainingSession] = {}
        self.user_progress: Dict[str, Dict[str, float]] = {}  # user_id -> skill -> score
        self.training_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, session_id
        
        # Initialize with default empathy exercises
        self._initialize_default_exercises()
    
    def _initialize_default_exercises(self):
        """Initialize with default empathy training exercises"""
        default_exercises = [
            EmpathyExercise(
                exercise_id="",
                skill_focus=EmpathySkill.EMOTIONAL_RECOGNITION,
                training_level=TrainingLevel.BEGINNER,
                title="Reading Facial Expressions",
                description="Practice recognizing emotions through facial expressions and body language",
                instructions="Observe the persona's expressions and body language. What emotions are they experiencing? How can you tell?",
                scenario_context="A colleague is having a difficult day at work",
                personas_involved=[],  # Will be populated with personas
                duration_minutes=15,
                difficulty_score=0.3,
                learning_objectives=[
                    "Identify basic emotions (happy, sad, angry, surprised, fearful, disgusted)",
                    "Recognize subtle emotional cues",
                    "Understand context's role in emotional expression"
                ],
                success_indicators=[
                    "Correctly identifies primary emotion",
                    "Notices secondary emotional cues",
                    "Considers situational context"
                ],
                common_pitfalls=[
                    "Focusing only on obvious expressions",
                    "Ignoring body language",
                    "Making assumptions without context"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                tags=["beginner", "emotional_recognition", "facial_expressions"]
            ),
            EmpathyExercise(
                exercise_id="",
                skill_focus=EmpathySkill.PERSPECTIVE_TAKING,
                training_level=TrainingLevel.INTERMEDIATE,
                title="Walking in Their Shoes",
                description="Practice understanding situations from another person's perspective",
                instructions="Imagine yourself in the persona's situation. What would you think, feel, and need?",
                scenario_context="A team member is struggling with a project deadline",
                personas_involved=[],
                duration_minutes=20,
                difficulty_score=0.5,
                learning_objectives=[
                    "Understand others' viewpoints",
                    "Recognize different needs and priorities",
                    "Develop cognitive empathy"
                ],
                success_indicators=[
                    "Accurately describes persona's perspective",
                    "Identifies underlying needs and concerns",
                    "Shows understanding of different priorities"
                ],
                common_pitfalls=[
                    "Projecting own feelings onto others",
                    "Making assumptions about motivations",
                    "Failing to consider background and context"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                tags=["intermediate", "perspective_taking", "understanding"]
            ),
            EmpathyExercise(
                exercise_id="",
                skill_focus=EmpathySkill.ACTIVE_LISTENING,
                training_level=TrainingLevel.INTERMEDIATE,
                title="Deep Listening Practice",
                description="Practice active listening skills in emotionally charged situations",
                instructions="Listen to what the persona is saying, both verbally and non-verbally. Focus on understanding, not responding.",
                scenario_context="A friend is sharing a personal challenge",
                personas_involved=[],
                duration_minutes=25,
                difficulty_score=0.6,
                learning_objectives=[
                    "Practice full attention listening",
                    "Recognize emotional undertones",
                    "Avoid interrupting or judging"
                ],
                success_indicators=[
                    "Demonstrates full attention",
                    "Recognizes emotional content",
                    "Shows understanding through body language"
                ],
                common_pitfalls=[
                    "Thinking about what to say next",
                    "Interrupting with advice",
                    "Minimizing the person's feelings"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                tags=["intermediate", "active_listening", "communication"]
            ),
            EmpathyExercise(
                exercise_id="",
                skill_focus=EmpathySkill.COMPASSIONATE_RESPONSE,
                training_level=TrainingLevel.ADVANCED,
                title="Responding with Compassion",
                description="Practice responding to others' emotional needs with genuine compassion",
                instructions="Respond to the persona's emotional state with compassion and understanding. What would be most helpful?",
                scenario_context="A colleague is experiencing a personal crisis",
                personas_involved=[],
                duration_minutes=30,
                difficulty_score=0.7,
                learning_objectives=[
                    "Respond with genuine compassion",
                    "Provide appropriate emotional support",
                    "Maintain professional boundaries"
                ],
                success_indicators=[
                    "Shows genuine concern and care",
                    "Offers appropriate support",
                    "Respects boundaries and limitations"
                ],
                common_pitfalls=[
                    "Being overly emotional or dramatic",
                    "Offering unsolicited advice",
                    "Crossing professional boundaries"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                tags=["advanced", "compassion", "emotional_support"]
            )
        ]
        
        for exercise in default_exercises:
            self.exercises[exercise.exercise_id] = exercise
    
    def start_training_session(self, exercise_id: str, user_id: str) -> TrainingSession:
        """Start a new empathy training session"""
        try:
            if exercise_id not in self.exercises:
                raise ValueError(f"Exercise {exercise_id} not found")
            
            exercise = self.exercises[exercise_id]
            
            # Create training session
            session = TrainingSession(
                session_id="",
                exercise_id=exercise_id,
                user_id=user_id,
                started_at=datetime.utcnow()
            )
            
            # Initialize skill scores
            session.skill_scores = {skill.value: 0.0 for skill in EmpathySkill}
            
            # Store session
            self.training_sessions[session.session_id] = session
            
            # Track session start
            self.training_history.append((datetime.utcnow(), "started", session.session_id))
            
            logger.info(f"Started empathy training session: {exercise.title}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start training session: {e}")
            raise
    
    def submit_response(self, session_id: str, response_data: Dict[str, Any]) -> bool:
        """Submit a response for a training session"""
        try:
            if session_id not in self.training_sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.training_sessions[session_id]
            exercise = self.exercises[session.exercise_id]
            
            # Store response
            response_data["timestamp"] = datetime.utcnow().isoformat()
            session.responses.append(response_data)
            
            # Evaluate response and update scores
            self._evaluate_response(session, exercise, response_data)
            
            logger.info(f"Submitted response for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit response: {e}")
            return False
    
    def _evaluate_response(self, session: TrainingSession, exercise: EmpathyExercise, 
                          response_data: Dict[str, Any]):
        """Evaluate a user response and update scores"""
        try:
            # Get the skill being trained
            skill = exercise.skill_focus.value
            
            # Simple scoring based on response quality
            # In practice, this could use more sophisticated NLP analysis
            base_score = 0.5
            
            # Adjust score based on response characteristics
            if "understanding" in response_data.get("response_text", "").lower():
                base_score += 0.2
            if "emotion" in response_data.get("response_text", "").lower():
                base_score += 0.1
            if "support" in response_data.get("response_text", "").lower():
                base_score += 0.1
            if "boundary" in response_data.get("response_text", "").lower():
                base_score += 0.1
            
            # Cap score at 1.0
            final_score = min(base_score, 1.0)
            
            # Update skill score
            session.skill_scores[skill] = final_score
            
            # Update overall empathy score
            self._update_empathy_score(session)
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
    
    def _update_empathy_score(self, session: TrainingSession):
        """Update the overall empathy score for a session"""
        try:
            if not session.skill_scores:
                session.empathy_score = 0.0
                return
            
            # Calculate average of all skill scores
            total_score = sum(session.skill_scores.values())
            session.empathy_score = total_score / len(session.skill_scores)
            
        except Exception as e:
            logger.error(f"Error updating empathy score: {e}")
    
    def complete_training_session(self, session_id: str, self_reflection: str,
                                areas_for_improvement: List[str], 
                                strengths_demonstrated: List[str]) -> bool:
        """Complete a training session with self-reflection"""
        try:
            if session_id not in self.training_sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.training_sessions[session_id]
            
            # Mark session as completed
            session.completed_at = datetime.utcnow()
            session.self_reflection = self_reflection
            session.areas_for_improvement = areas_for_improvement
            session.strengths_demonstrated = strengths_demonstrated
            
            # Update user progress
            self._update_user_progress(session)
            
            # Track completion
            self.training_history.append((datetime.utcnow(), "completed", session_id))
            
            logger.info(f"Completed training session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete training session: {e}")
            return False
    
    def _update_user_progress(self, session: TrainingSession):
        """Update user progress tracking"""
        try:
            user_id = session.user_id
            
            if user_id not in self.user_progress:
                self.user_progress[user_id] = {skill.value: 0.0 for skill in EmpathySkill}
            
            # Update skill scores (take the higher score)
            for skill, score in session.skill_scores.items():
                current_score = self.user_progress[user_id].get(skill, 0.0)
                self.user_progress[user_id][skill] = max(current_score, score)
            
        except Exception as e:
            logger.error(f"Error updating user progress: {e}")
    
    def get_exercises_by_skill(self, skill: EmpathySkill) -> List[EmpathyExercise]:
        """Get exercises focused on a specific empathy skill"""
        return [ex for ex in self.exercises.values() if ex.skill_focus == skill]
    
    def get_exercises_by_level(self, level: TrainingLevel) -> List[EmpathyExercise]:
        """Get exercises at a specific training level"""
        return [ex for ex in self.exercises.values() if ex.training_level == level]
    
    def get_recommended_exercise(self, user_id: str, skill_focus: Optional[EmpathySkill] = None) -> Optional[EmpathyExercise]:
        """Get a recommended exercise based on user progress"""
        try:
            if user_id not in self.user_progress:
                # New user - recommend beginner exercises
                beginner_exercises = self.get_exercises_by_level(TrainingLevel.BEGINNER)
                if beginner_exercises:
                    return random.choice(beginner_exercises)
                return None
            
            user_progress = self.user_progress[user_id]
            
            # If specific skill requested, find appropriate level
            if skill_focus:
                skill_score = user_progress.get(skill_focus.value, 0.0)
                if skill_score < 0.3:
                    level = TrainingLevel.BEGINNER
                elif skill_score < 0.6:
                    level = TrainingLevel.INTERMEDIATE
                elif skill_score < 0.8:
                    level = TrainingLevel.ADVANCED
                else:
                    level = TrainingLevel.EXPERT
                
                exercises = [ex for ex in self.exercises.values() 
                           if ex.skill_focus == skill_focus and ex.training_level == level]
                if exercises:
                    return random.choice(exercises)
            
            # Find weakest skill and recommend appropriate exercise
            weakest_skill = min(user_progress.items(), key=lambda x: x[1])
            skill_name = weakest_skill[0]
            
            # Find exercises for this skill at appropriate level
            skill_score = weakest_skill[1]
            if skill_score < 0.3:
                level = TrainingLevel.BEGINNER
            elif skill_score < 0.6:
                level = TrainingLevel.INTERMEDIATE
            elif skill_score < 0.8:
                level = TrainingLevel.ADVANCED
            else:
                level = TrainingLevel.EXPERT
            
            exercises = [ex for ex in self.exercises.values() 
                       if ex.skill_focus.value == skill_name and ex.training_level == level]
            
            if exercises:
                return random.choice(exercises)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recommended exercise: {e}")
            return None
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get progress summary for a specific user"""
        if user_id not in self.user_progress:
            return {
                "user_id": user_id,
                "overall_empathy_score": 0.0,
                "skill_scores": {skill.value: 0.0 for skill in EmpathySkill},
                "training_sessions_completed": 0,
                "recommended_next_skill": "emotional_recognition"
            }
        
        user_progress = self.user_progress[user_id]
        
        # Calculate overall score
        overall_score = sum(user_progress.values()) / len(user_progress) if user_progress else 0.0
        
        # Count completed sessions
        completed_sessions = len([s for s in self.training_sessions.values() 
                                if s.user_id == user_id and s.completed_at])
        
        # Find weakest skill for recommendation
        weakest_skill = min(user_progress.items(), key=lambda x: x[1])[0] if user_progress else "emotional_recognition"
        
        return {
            "user_id": user_id,
            "overall_empathy_score": overall_score,
            "skill_scores": user_progress.copy(),
            "training_sessions_completed": completed_sessions,
            "recommended_next_skill": weakest_skill
        }
    
    def get_empathy_trainer_summary(self) -> Dict[str, Any]:
        """Get summary of the empathy training system"""
        total_exercises = len(self.exercises)
        total_sessions = len(self.training_sessions)
        completed_sessions = len([s for s in self.training_sessions.values() if s.completed_at])
        
        # Exercises by skill
        exercises_by_skill = {}
        for exercise in self.exercises.values():
            skill = exercise.skill_focus.value
            exercises_by_skill[skill] = exercises_by_skill.get(skill, 0) + 1
        
        # Exercises by level
        exercises_by_level = {}
        for exercise in self.exercises.values():
            level = exercise.training_level.value
            exercises_by_level[level] = exercises_by_level.get(level, 0) + 1
        
        return {
            "total_exercises": total_exercises,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "exercises_by_skill": exercises_by_skill,
            "exercises_by_level": exercises_by_level,
            "active_users": len(self.user_progress),
            "recent_sessions": self._get_recent_sessions(10)
        }
    
    def _get_recent_sessions(self, count: int) -> List[Dict[str, Any]]:
        """Get recent training sessions"""
        sorted_sessions = sorted(self.training_sessions.values(), key=lambda s: s.started_at, reverse=True)
        recent_sessions = sorted_sessions[:count]
        
        return [
            {
                "session_id": s.session_id,
                "exercise_id": s.exercise_id,
                "user_id": s.user_id,
                "started_at": s.started_at.isoformat(),
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "empathy_score": s.empathy_score
            }
            for s in recent_sessions
        ]
