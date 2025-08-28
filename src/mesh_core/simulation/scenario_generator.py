"""
Mesh Scenario Generator
======================

Creates counterfactual scenarios for behavioral coaching,
empathy training, and decision rehearsal.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios that can be generated"""
    CONFLICT_RESOLUTION = "conflict_resolution"      # Interpersonal conflicts
    DECISION_MAKING = "decision_making"              # Difficult choices
    EMPATHY_TRAINING = "empathy_training"            # Understanding others
    COMMUNICATION = "communication"                  # Communication challenges
    TRUST_BUILDING = "trust_building"                # Building trust
    JUSTICE_TESTING = "justice_testing"              # Testing justice models
    SOCIAL_FRICTION = "social_friction"              # Social conflict modeling


class ScenarioComplexity(Enum):
    """Complexity levels for scenarios"""
    BASIC = "basic"                  # Simple, single-issue scenarios
    INTERMEDIATE = "intermediate"    # Multi-issue scenarios
    ADVANCED = "advanced"            # Complex, multi-stakeholder scenarios
    EXPERT = "expert"                # Expert-level, nuanced scenarios


@dataclass
class Persona:
    """A persona for scenario participants"""
    persona_id: str
    name: str
    age: int
    background: str
    personality_traits: List[str]
    goals: List[str]
    fears: List[str]
    communication_style: str
    emotional_state: str
    relationship_to_user: str
    
    # Behavioral patterns
    decision_making_style: str
    conflict_approach: str
    trust_level: float  # 0.0 to 1.0
    empathy_capacity: float  # 0.0 to 1.0
    
    def __post_init__(self):
        if not self.persona_id:
            self.persona_id = self._generate_persona_id()
    
    def _generate_persona_id(self) -> str:
        """Generate unique persona ID"""
        content = f"{self.name}{self.background}{self.personality_traits}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "age": self.age,
            "background": self.background,
            "personality_traits": self.personality_traits,
            "goals": self.goals,
            "fears": self.fears,
            "communication_style": self.communication_style,
            "emotional_state": self.emotional_state,
            "relationship_to_user": self.relationship_to_user,
            "decision_making_style": self.decision_making_style,
            "conflict_approach": self.conflict_approach,
            "trust_level": self.trust_level,
            "empathy_capacity": self.empathy_capacity
        }


@dataclass
class Scenario:
    """A generated scenario for behavioral coaching"""
    scenario_id: str
    scenario_type: ScenarioType
    complexity: ScenarioComplexity
    title: str
    description: str
    context: str
    participants: List[Persona]
    user_role: str
    user_objectives: List[str]
    challenges: List[str]
    ethical_considerations: List[str]
    
    # Scenario details
    setting: str
    timeline: str
    stakes: str
    constraints: List[str]
    
    # Generated content
    initial_situation: str
    potential_actions: List[str]
    expected_outcomes: List[str]
    learning_objectives: List[str]
    
    # Metadata
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    difficulty_score: float = 0.5  # 0.0 to 1.0
    
    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = self._generate_scenario_id()
    
    def _generate_scenario_id(self) -> str:
        """Generate unique scenario ID"""
        content = f"{self.title}{self.scenario_type.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary"""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "complexity": self.complexity.value,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "participants": [p.to_dict() for p in self.participants],
            "user_role": self.user_role,
            "user_objectives": self.user_objectives,
            "challenges": self.challenges,
            "ethical_considerations": self.ethical_considerations,
            "setting": self.setting,
            "timeline": self.timeline,
            "stakes": self.stakes,
            "constraints": self.constraints,
            "initial_situation": self.initial_situation,
            "potential_actions": self.potential_actions,
            "expected_outcomes": self.expected_outcomes,
            "learning_objectives": self.learning_objectives,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "difficulty_score": self.difficulty_score
        }


class ScenarioGenerator:
    """
    Generates realistic scenarios for behavioral coaching and empathy training
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.scenarios: Dict[str, Scenario] = {}
        self.personas: Dict[str, Persona] = {}
        self.scenario_templates: Dict[str, Dict[str, Any]] = {}
        self.generation_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, scenario_id
        
        # Initialize with default personas and templates
        self._initialize_default_personas()
        self._initialize_scenario_templates()
    
    def _initialize_default_personas(self):
        """Initialize with default personas for scenario generation"""
        default_personas = [
            Persona(
                persona_id="",
                name="Alex",
                age=28,
                background="Software engineer, recently promoted to team lead",
                personality_traits=["analytical", "introverted", "detail-oriented"],
                goals=["Lead team effectively", "Maintain work-life balance"],
                fears=["Making wrong decisions", "Team conflict"],
                communication_style="Direct and factual",
                emotional_state="Stressed but determined",
                relationship_to_user="Team member",
                decision_making_style="Data-driven",
                conflict_approach="Avoidance",
                trust_level=0.7,
                empathy_capacity=0.6
            ),
            Persona(
                persona_id="",
                name="Sarah",
                age=35,
                background="Marketing manager, experienced team leader",
                personality_traits=["extroverted", "empathetic", "creative"],
                goals=["Foster team collaboration", "Drive innovation"],
                fears=["Team underperformance", "Losing team trust"],
                communication_style="Warm and encouraging",
                emotional_state="Optimistic and caring",
                relationship_to_user="Peer manager",
                decision_making_style="Intuitive",
                conflict_approach="Collaboration",
                trust_level=0.9,
                empathy_capacity=0.9
            ),
            Persona(
                persona_id="",
                name="Marcus",
                age=42,
                background="Senior executive, results-focused leader",
                personality_traits=["assertive", "results-driven", "pragmatic"],
                goals=["Achieve business objectives", "Develop future leaders"],
                fears=["Missing targets", "Team underperformance"],
                communication_style="Clear and directive",
                emotional_state="Focused and determined",
                relationship_to_user="Senior manager",
                decision_making_style="Efficient",
                conflict_approach="Direct resolution",
                trust_level=0.8,
                empathy_capacity=0.5
            )
        ]
        
        for persona in default_personas:
            self.personas[persona.persona_id] = persona
    
    def _initialize_scenario_templates(self):
        """Initialize scenario templates for different types"""
        self.scenario_templates = {
            ScenarioType.CONFLICT_RESOLUTION.value: {
                "title_patterns": [
                    "Team Conflict: {conflict_type}",
                    "Resolving {conflict_type} Dispute",
                    "Mediating {conflict_type} Conflict"
                ],
                "conflict_types": [
                    "Resource Allocation",
                    "Project Priorities",
                    "Communication Styles",
                    "Work Methods",
                    "Performance Expectations"
                ],
                "learning_objectives": [
                    "Practice active listening",
                    "Identify underlying concerns",
                    "Find common ground",
                    "Develop win-win solutions"
                ]
            },
            ScenarioType.DECISION_MAKING.value: {
                "title_patterns": [
                    "Difficult Decision: {decision_type}",
                    "Choosing Between {option_a} and {option_b}",
                    "Ethical Decision: {ethical_issue}"
                ],
                "decision_types": [
                    "Team Restructuring",
                    "Budget Cuts",
                    "Project Cancellation",
                    "Hiring Decisions",
                    "Strategic Direction"
                ],
                "learning_objectives": [
                    "Weigh pros and cons",
                    "Consider stakeholder impact",
                    "Apply ethical frameworks",
                    "Make difficult choices"
                ]
            },
            ScenarioType.EMPATHY_TRAINING.value: {
                "title_patterns": [
                    "Understanding {persona_name}'s Perspective",
                    "Walking in {persona_name}'s Shoes",
                    "Empathetic Response to {situation}"
                ],
                "situations": [
                    "Personal Crisis",
                    "Work Stress",
                    "Family Issues",
                    "Health Problems",
                    "Career Concerns"
                ],
                "learning_objectives": [
                    "Recognize emotional states",
                    "Understand different perspectives",
                    "Practice empathetic responses",
                    "Build emotional intelligence"
                ]
            }
        }
    
    def generate_scenario(self, scenario_type: ScenarioType, complexity: ScenarioComplexity,
                         user_preferences: Dict[str, Any] = None) -> Scenario:
        """Generate a new scenario based on type and complexity"""
        try:
            # Get template for scenario type
            template = self.scenario_templates.get(scenario_type.value, {})
            
            # Generate scenario content based on type and complexity
            if scenario_type == ScenarioType.CONFLICT_RESOLUTION:
                scenario = self._generate_conflict_scenario(complexity, template, user_preferences)
            elif scenario_type == ScenarioType.DECISION_MAKING:
                scenario = self._generate_decision_scenario(complexity, template, user_preferences)
            elif scenario_type == ScenarioType.EMPATHY_TRAINING:
                scenario = self._generate_empathy_scenario(complexity, template, user_preferences)
            else:
                scenario = self._generate_generic_scenario(complexity, template, user_preferences)
            
            # Store and track the scenario
            self.scenarios[scenario.scenario_id] = scenario
            self.generation_history.append((datetime.utcnow(), "generated", scenario.scenario_id))
            
            logger.info(f"Generated {scenario_type.value} scenario: {scenario.title}")
            return scenario
            
        except Exception as e:
            logger.error(f"Failed to generate scenario: {e}")
            raise
    
    def _generate_conflict_scenario(self, complexity: ScenarioComplexity, template: Dict[str, Any], 
                                  user_preferences: Dict[str, Any]) -> Scenario:
        """Generate a conflict resolution scenario"""
        # Select conflict type
        conflict_type = random.choice(template.get("conflict_types", ["Team Dispute"]))
        
        # Select participants based on complexity
        num_participants = self._get_participant_count(complexity)
        participants = random.sample(list(self.personas.values()), min(num_participants, len(self.personas)))
        
        # Generate scenario content
        title = f"Team Conflict: {conflict_type}"
        context = f"A conflict has arisen in your team regarding {conflict_type.lower()}. "
        context += f"Multiple team members have different perspectives and the situation is affecting productivity."
        
        initial_situation = self._generate_conflict_situation(conflict_type, participants)
        challenges = self._generate_conflict_challenges(complexity)
        ethical_considerations = ["Fairness", "Transparency", "Respect for all parties"]
        
        return Scenario(
            scenario_id="",
            scenario_type=ScenarioType.CONFLICT_RESOLUTION,
            complexity=complexity,
            title=title,
            description=f"Navigate a complex team conflict involving {conflict_type.lower()}",
            context=context,
            participants=participants,
            user_role="Team Leader/Mediator",
            user_objectives=["Resolve conflict", "Maintain team cohesion", "Find sustainable solution"],
            challenges=challenges,
            ethical_considerations=ethical_considerations,
            setting="Office meeting room",
            timeline="2-3 hours",
            stakes="Team productivity and morale",
            constraints=["Time pressure", "Conflicting interests", "Limited resources"],
            initial_situation=initial_situation,
            potential_actions=self._generate_potential_actions("conflict_resolution"),
            expected_outcomes=self._generate_expected_outcomes("conflict_resolution"),
            learning_objectives=template.get("learning_objectives", []),
            created_at=datetime.utcnow(),
            created_by=self.node_id,
            tags=["conflict", "team", "resolution", "leadership"],
            difficulty_score=self._calculate_difficulty_score(complexity)
        )
    
    def _generate_decision_scenario(self, complexity: ScenarioComplexity, template: Dict[str, Any],
                                   user_preferences: Dict[str, Any]) -> Scenario:
        """Generate a decision-making scenario"""
        # Select decision type
        decision_type = random.choice(template.get("decision_types", ["Strategic Choice"]))
        
        # Select participants
        num_participants = self._get_participant_count(complexity)
        participants = random.sample(list(self.personas.values()), min(num_participants, len(self.personas)))
        
        # Generate scenario content
        title = f"Difficult Decision: {decision_type}"
        context = f"You must make a critical decision about {decision_type.lower()}. "
        context += f"This decision will impact multiple stakeholders and has significant consequences."
        
        initial_situation = self._generate_decision_situation(decision_type, participants)
        challenges = self._generate_decision_challenges(complexity)
        ethical_considerations = ["Stakeholder impact", "Long-term consequences", "Ethical implications"]
        
        return Scenario(
            scenario_id="",
            scenario_type=ScenarioType.DECISION_MAKING,
            complexity=complexity,
            title=title,
            description=f"Make a difficult decision about {decision_type.lower()}",
            context=context,
            participants=participants,
            user_role="Decision Maker",
            user_objectives=["Make informed decision", "Consider all stakeholders", "Minimize negative impact"],
            challenges=challenges,
            ethical_considerations=ethical_considerations,
            setting="Executive boardroom",
            timeline="1-2 weeks",
            stakes="Company direction and stakeholder welfare",
            constraints=["Time pressure", "Incomplete information", "Conflicting stakeholder needs"],
            initial_situation=initial_situation,
            potential_actions=self._generate_potential_actions("decision_making"),
            expected_outcomes=self._generate_expected_outcomes("decision_making"),
            learning_objectives=template.get("learning_objectives", []),
            created_at=datetime.utcnow(),
            created_by=self.node_id,
            tags=["decision", "leadership", "ethics", "stakeholders"],
            difficulty_score=self._calculate_difficulty_score(complexity)
        )
    
    def _generate_empathy_scenario(self, complexity: ScenarioComplexity, template: Dict[str, Any],
                                  user_preferences: Dict[str, Any]) -> Scenario:
        """Generate an empathy training scenario"""
        # Select persona and situation
        persona = random.choice(list(self.personas.values()))
        situation = random.choice(template.get("situations", ["Personal Challenge"]))
        
        # Generate scenario content
        title = f"Understanding {persona.name}'s Perspective"
        context = f"{persona.name} is experiencing {situation.lower()}. "
        context += f"You need to respond with empathy and understanding."
        
        initial_situation = self._generate_empathy_situation(persona, situation)
        challenges = self._generate_empathy_challenges(complexity)
        ethical_considerations = ["Respect for privacy", "Genuine concern", "Appropriate boundaries"]
        
        return Scenario(
            scenario_id="",
            scenario_type=ScenarioType.EMPATHY_TRAINING,
            complexity=complexity,
            title=title,
            description=f"Practice empathy with {persona.name} during {situation.lower()}",
            context=context,
            participants=[persona],
            user_role="Supportive Colleague/Friend",
            user_objectives=["Show genuine empathy", "Provide appropriate support", "Maintain boundaries"],
            challenges=challenges,
            ethical_considerations=ethical_considerations,
            setting="Private office or coffee shop",
            timeline="30-60 minutes",
            stakes="Person's emotional well-being and trust",
            constraints=["Professional boundaries", "Limited time", "Emotional complexity"],
            initial_situation=initial_situation,
            potential_actions=self._generate_potential_actions("empathy_training"),
            expected_outcomes=self._generate_expected_outcomes("empathy_training"),
            learning_objectives=template.get("learning_objectives", []),
            created_at=datetime.utcnow(),
            created_by=self.node_id,
            tags=["empathy", "emotional intelligence", "support", "communication"],
            difficulty_score=self._calculate_difficulty_score(complexity)
        )
    
    def _generate_generic_scenario(self, complexity: ScenarioComplexity, template: Dict[str, Any],
                                  user_preferences: Dict[str, Any]) -> Scenario:
        """Generate a generic scenario when type is not specifically handled"""
        return Scenario(
            scenario_id="",
            scenario_type=ScenarioType.COMMUNICATION,
            complexity=complexity,
            title="Communication Challenge",
            description="Navigate a complex communication situation",
            context="You face a challenging communication scenario that requires careful handling.",
            participants=random.sample(list(self.personas.values()), 2),
            user_role="Communicator",
            user_objectives=["Clear communication", "Understanding", "Positive outcome"],
            challenges=["Misunderstandings", "Emotional barriers", "Cultural differences"],
            ethical_considerations=["Honesty", "Respect", "Cultural sensitivity"],
            setting="Various locations",
            timeline="Variable",
            stakes="Relationship quality and outcomes",
            constraints=["Communication barriers", "Time limitations", "Emotional complexity"],
            initial_situation="A communication breakdown has occurred.",
            potential_actions=["Active listening", "Clarification", "Empathetic response"],
            expected_outcomes=["Improved understanding", "Better relationship", "Resolved issues"],
            learning_objectives=["Communication skills", "Empathy", "Problem-solving"],
            created_at=datetime.utcnow(),
            created_by=self.node_id,
            tags=["communication", "general", "skills"],
            difficulty_score=self._calculate_difficulty_score(complexity)
        )
    
    def _get_participant_count(self, complexity: ScenarioComplexity) -> int:
        """Get number of participants based on complexity"""
        complexity_map = {
            ScenarioComplexity.BASIC: 2,
            ScenarioComplexity.INTERMEDIATE: 3,
            ScenarioComplexity.ADVANCED: 4,
            ScenarioComplexity.EXPERT: 5
        }
        return complexity_map.get(complexity, 3)
    
    def _calculate_difficulty_score(self, complexity: ScenarioComplexity) -> float:
        """Calculate difficulty score based on complexity"""
        complexity_scores = {
            ScenarioComplexity.BASIC: 0.3,
            ScenarioComplexity.INTERMEDIATE: 0.5,
            ScenarioComplexity.ADVANCED: 0.7,
            ScenarioComplexity.EXPERT: 0.9
        }
        return complexity_scores.get(complexity, 0.5)
    
    def _generate_conflict_situation(self, conflict_type: str, participants: List[Persona]) -> str:
        """Generate initial conflict situation"""
        if conflict_type == "Resource Allocation":
            return f"{participants[0].name} believes the team should focus on Project A, while {participants[1].name} argues for Project B. Both have valid points and limited resources make it impossible to pursue both fully."
        elif conflict_type == "Communication Styles":
            return f"{participants[0].name} prefers detailed written communication, while {participants[1].name} favors quick verbal updates. This has led to misunderstandings and missed deadlines."
        else:
            return f"A disagreement has arisen between team members about {conflict_type.lower()}. Tensions are high and productivity is suffering."
    
    def _generate_decision_situation(self, decision_type: str, participants: List[Persona]) -> str:
        """Generate initial decision situation"""
        if decision_type == "Team Restructuring":
            return f"Due to budget constraints, you must restructure the team. This will involve role changes, potential layoffs, and significant disruption to ongoing projects."
        elif decision_type == "Project Cancellation":
            return f"A major project that several team members have invested months in may need to be cancelled due to changing business priorities."
        else:
            return f"You face a critical decision about {decision_type.lower()} that will impact multiple stakeholders and have long-term consequences."
    
    def _generate_empathy_situation(self, persona: Persona, situation: str) -> str:
        """Generate initial empathy situation"""
        if situation == "Personal Crisis":
            return f"{persona.name} has just learned that a family member is seriously ill. They're trying to maintain professionalism at work but are clearly struggling emotionally."
        elif situation == "Work Stress":
            return f"{persona.name} has been working overtime for weeks and is showing signs of burnout. They're normally enthusiastic but now seem withdrawn and irritable."
        else:
            return f"{persona.name} is experiencing {situation.lower()} and needs support and understanding from colleagues."
    
    def _generate_conflict_challenges(self, complexity: ScenarioComplexity) -> List[str]:
        """Generate challenges for conflict scenarios"""
        basic_challenges = ["Emotional tension", "Communication barriers"]
        intermediate_challenges = basic_challenges + ["Conflicting interests", "Time pressure"]
        advanced_challenges = intermediate_challenges + ["Multiple stakeholders", "Historical tensions"]
        expert_challenges = advanced_challenges + ["Cultural differences", "Power dynamics"]
        
        complexity_map = {
            ScenarioComplexity.BASIC: basic_challenges,
            ScenarioComplexity.INTERMEDIATE: intermediate_challenges,
            ScenarioComplexity.ADVANCED: advanced_challenges,
            ScenarioComplexity.EXPERT: expert_challenges
        }
        return complexity_map.get(complexity, intermediate_challenges)
    
    def _generate_decision_challenges(self, complexity: ScenarioComplexity) -> List[str]:
        """Generate challenges for decision scenarios"""
        basic_challenges = ["Limited information", "Time pressure"]
        intermediate_challenges = basic_challenges + ["Multiple options", "Stakeholder concerns"]
        advanced_challenges = intermediate_challenges + ["Ethical implications", "Long-term consequences"]
        expert_challenges = advanced_challenges + ["Conflicting values", "Uncertain outcomes"]
        
        complexity_map = {
            ScenarioComplexity.BASIC: basic_challenges,
            ScenarioComplexity.INTERMEDIATE: intermediate_challenges,
            ScenarioComplexity.ADVANCED: advanced_challenges,
            ScenarioComplexity.EXPERT: expert_challenges
        }
        return complexity_map.get(complexity, intermediate_challenges)
    
    def _generate_empathy_challenges(self, complexity: ScenarioComplexity) -> List[str]:
        """Generate challenges for empathy scenarios"""
        basic_challenges = ["Emotional complexity", "Professional boundaries"]
        intermediate_challenges = basic_challenges + ["Cultural differences", "Personal biases"]
        advanced_challenges = intermediate_challenges + ["Trauma sensitivity", "Support limitations"]
        expert_challenges = advanced_challenges + ["Ethical dilemmas", "Long-term support needs"]
        
        complexity_map = {
            ScenarioComplexity.BASIC: basic_challenges,
            ScenarioComplexity.INTERMEDIATE: intermediate_challenges,
            ScenarioComplexity.ADVANCED: advanced_challenges,
            ScenarioComplexity.EXPERT: expert_challenges
        }
        return complexity_map.get(complexity, intermediate_challenges)
    
    def _generate_potential_actions(self, action_type: str) -> List[str]:
        """Generate potential actions for scenarios"""
        action_templates = {
            "conflict_resolution": [
                "Facilitate open discussion",
                "Identify common ground",
                "Propose compromise solutions",
                "Establish clear communication protocols"
            ],
            "decision_making": [
                "Gather additional information",
                "Consult with stakeholders",
                "Analyze pros and cons",
                "Consider ethical implications"
            ],
            "empathy_training": [
                "Practice active listening",
                "Validate emotions",
                "Offer appropriate support",
                "Maintain professional boundaries"
            ]
        }
        return action_templates.get(action_type, ["Analyze situation", "Consider options", "Take action"])
    
    def _generate_expected_outcomes(self, outcome_type: str) -> List[str]:
        """Generate expected outcomes for scenarios"""
        outcome_templates = {
            "conflict_resolution": [
                "Improved team communication",
                "Resolved underlying issues",
                "Enhanced team collaboration",
                "Prevented future conflicts"
            ],
            "decision_making": [
                "Informed decision made",
                "Stakeholder buy-in achieved",
                "Long-term benefits realized",
                "Ethical standards maintained"
            ],
            "empathy_training": [
                "Stronger relationships built",
                "Emotional intelligence improved",
                "Support provided appropriately",
                "Professional boundaries maintained"
            ]
        }
        return outcome_templates.get(outcome_type, ["Positive outcome", "Learning achieved", "Growth experienced"])
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get a specific scenario by ID"""
        return self.scenarios.get(scenario_id)
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[Scenario]:
        """Get all scenarios of a specific type"""
        return [s for s in self.scenarios.values() if s.scenario_type == scenario_type]
    
    def get_scenarios_by_complexity(self, complexity: ScenarioComplexity) -> List[Scenario]:
        """Get all scenarios of a specific complexity"""
        return [s for s in self.scenarios.values() if s.complexity == complexity]
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of all generated scenarios"""
        total_scenarios = len(self.scenarios)
        scenarios_by_type = {}
        scenarios_by_complexity = {}
        
        for scenario in self.scenarios.values():
            # Count by type
            scenario_type = scenario.scenario_type.value
            scenarios_by_type[scenario_type] = scenarios_by_type.get(scenario_type, 0) + 1
            
            # Count by complexity
            complexity = scenario.complexity.value
            scenarios_by_complexity[complexity] = scenarios_by_complexity.get(complexity, 0) + 1
        
        return {
            "total_scenarios": total_scenarios,
            "scenarios_by_type": scenarios_by_type,
            "scenarios_by_complexity": scenarios_by_complexity,
            "recent_scenarios": self._get_recent_scenarios(10)
        }
    
    def _get_recent_scenarios(self, count: int) -> List[Dict[str, Any]]:
        """Get recent scenarios"""
        sorted_scenarios = sorted(self.scenarios.values(), key=lambda s: s.created_at, reverse=True)
        recent_scenarios = sorted_scenarios[:count]
        
        return [
            {
                "scenario_id": s.scenario_id,
                "title": s.title,
                "type": s.scenario_type.value,
                "complexity": s.complexity.value,
                "created_at": s.created_at.isoformat(),
                "difficulty_score": s.difficulty_score
            }
            for s in recent_scenarios
        ]
