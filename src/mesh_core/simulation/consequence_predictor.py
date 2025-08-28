"""
Mesh Consequence Predictor
==========================

Predicts outcomes of choices and decisions using scenario modeling,
causal reasoning, and impact analysis.
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


class ImpactLevel(Enum):
    """Levels of impact for consequences"""
    NEGLIGIBLE = "negligible"            # Minimal or no impact
    LOW = "low"                          # Small, manageable impact
    MODERATE = "moderate"                # Noticeable but manageable impact
    HIGH = "high"                        # Significant impact requiring attention
    CRITICAL = "critical"                # Major impact with serious consequences
    CATASTROPHIC = "catastrophic"        # Severe, potentially irreversible impact


class Timeframe(Enum):
    """Timeframes for consequences to manifest"""
    IMMEDIATE = "immediate"              # Within minutes to hours
    SHORT_TERM = "short_term"            # Within days to weeks
    MEDIUM_TERM = "medium_term"          # Within months to a year
    LONG_TERM = "long_term"              # Within years to decades
    GENERATIONAL = "generational"        # Affects future generations


class ConsequenceType(Enum):
    """Types of consequences"""
    PERSONAL = "personal"                # Affects the individual
    INTERPERSONAL = "interpersonal"      # Affects relationships
    ORGANIZATIONAL = "organizational"    # Affects organizations/teams
    SOCIETAL = "societal"                # Affects society at large
    ENVIRONMENTAL = "environmental"      # Affects the environment
    ECONOMIC = "economic"                # Affects financial/economic systems


@dataclass
class Consequence:
    """A predicted consequence of an action or decision"""
    consequence_id: str
    action_description: str
    impact_level: ImpactLevel
    timeframe: Timeframe
    consequence_type: ConsequenceType
    description: str
    probability: float  # 0.0 to 1.0
    
    # Impact details
    affected_parties: List[str]
    severity_description: str
    reversibility: str  # "easily_reversible", "reversible_with_effort", "difficult_to_reverse", "irreversible"
    
    # Mitigation and prevention
    mitigation_strategies: List[str]
    prevention_methods: List[str]
    
    # Cascading effects
    cascading_consequences: List[str]
    ripple_effects: List[str]
    
    # Metadata
    created_at: datetime
    created_by: str
    confidence_score: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.consequence_id:
            self.consequence_id = self._generate_consequence_id()
    
    def _generate_consequence_id(self) -> str:
        """Generate unique consequence ID"""
        content = f"{self.action_description}{self.impact_level.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert consequence to dictionary"""
        return {
            "consequence_id": self.consequence_id,
            "action_description": self.action_description,
            "impact_level": self.impact_level.value,
            "timeframe": self.timeframe.value,
            "consequence_type": self.consequence_type.value,
            "description": self.description,
            "probability": self.probability,
            "affected_parties": self.affected_parties,
            "severity_description": self.severity_description,
            "reversibility": self.reversibility,
            "mitigation_strategies": self.mitigation_strategies,
            "prevention_methods": self.prevention_methods,
            "cascading_consequences": self.cascading_consequences,
            "ripple_effects": self.ripple_effects,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "confidence_score": self.confidence_score,
            "tags": self.tags
        }


@dataclass
class PredictionModel:
    """A model for predicting consequences"""
    model_id: str
    model_name: str
    description: str
    applicable_scenarios: List[ScenarioType]
    
    # Model parameters
    parameters: Dict[str, Any]
    training_data_size: int
    last_updated: datetime
    
    # Performance metrics
    precision: float
    recall: float
    f1_score: float
    
    # Metadata
    created_at: datetime
    created_by: str
    accuracy_score: float = 0.5  # 0.0 to 1.0
    version: str = "1.0.0"
    is_active: bool = True
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = self._generate_model_id()
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID"""
        content = f"{self.model_name}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "description": self.description,
            "applicable_scenarios": [s.value for s in self.applicable_scenarios],
            "accuracy_score": self.accuracy_score,
            "parameters": self.parameters,
            "training_data_size": self.training_data_size,
            "last_updated": self.last_updated.isoformat(),
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "is_active": self.is_active
        }


class ConsequencePredictor:
    """
    Predicts consequences of actions and decisions using scenario modeling
    and causal reasoning
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.consequence_history: Dict[str, List[Consequence]] = {}  # scenario_id -> consequences
        self.prediction_accuracy: Dict[str, List[Tuple[datetime, float]]] = {}  # model_id -> accuracy_tracking
        self.scenario_outcomes: Dict[str, Dict[str, Any]] = {}  # scenario_id -> actual_outcomes
        
        # Initialize with default prediction models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default consequence prediction models"""
        default_models = [
            PredictionModel(
                model_id="",
                model_name="Interpersonal Impact Model",
                description="Predicts consequences of interpersonal actions and decisions",
                applicable_scenarios=[ScenarioType.CONFLICT_RESOLUTION, ScenarioType.COMMUNICATION],
                parameters={
                    "relationship_strength_weight": 0.3,
                    "communication_style_weight": 0.25,
                    "emotional_state_weight": 0.25,
                    "context_importance_weight": 0.2
                },
                training_data_size=1000,
                last_updated=datetime.utcnow(),
                precision=0.72,
                recall=0.78,
                f1_score=0.75,
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                accuracy_score=0.75
            ),
            PredictionModel(
                model_id="",
                model_name="Organizational Decision Model",
                description="Predicts consequences of organizational decisions and policy changes",
                applicable_scenarios=[ScenarioType.DECISION_MAKING, ScenarioType.TRUST_BUILDING],
                parameters={
                    "stakeholder_impact_weight": 0.35,
                    "resource_availability_weight": 0.25,
                    "timeline_constraints_weight": 0.2,
                    "organizational_culture_weight": 0.2
                },
                training_data_size=500,
                last_updated=datetime.utcnow(),
                precision=0.65,
                recall=0.71,
                f1_score=0.68,
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                accuracy_score=0.68
            ),
            PredictionModel(
                model_id="",
                model_name="Ethical Consequence Model",
                description="Predicts ethical implications and moral consequences of actions",
                applicable_scenarios=[ScenarioType.DECISION_MAKING, ScenarioType.JUSTICE_TESTING],
                parameters={
                    "moral_framework_weight": 0.4,
                    "stakeholder_rights_weight": 0.3,
                    "long_term_impact_weight": 0.2,
                    "cultural_context_weight": 0.1
                },
                training_data_size=800,
                last_updated=datetime.utcnow(),
                precision=0.79,
                recall=0.85,
                f1_score=0.82,
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                accuracy_score=0.82
            )
        ]
        
        for model in default_models:
            self.prediction_models[model.model_id] = model
    
    def predict_consequences(self, scenario: Scenario, action: str, 
                           context: Dict[str, Any]) -> List[Consequence]:
        """Predict consequences of a specific action in a scenario"""
        try:
            consequences = []
            
            # Get applicable models for this scenario type
            applicable_models = self._get_applicable_models(scenario.scenario_type)
            
            for model in applicable_models:
                # Generate predictions using the model
                model_consequences = self._apply_prediction_model(
                    model, scenario, action, context
                )
                consequences.extend(model_consequences)
            
            # If no models applicable, use fallback prediction
            if not consequences:
                consequences = self._generate_fallback_predictions(scenario, action, context)
            
            # Store predictions for accuracy tracking
            scenario_id = scenario.scenario_id
            if scenario_id not in self.consequence_history:
                self.consequence_history[scenario_id] = []
            self.consequence_history[scenario_id].extend(consequences)
            
            logger.info(f"Generated {len(consequences)} consequence predictions for action: {action}")
            return consequences
            
        except Exception as e:
            logger.error(f"Error predicting consequences: {e}")
            return []
    
    def _get_applicable_models(self, scenario_type: ScenarioType) -> List[PredictionModel]:
        """Get prediction models applicable to a scenario type"""
        applicable = []
        for model in self.prediction_models.values():
            if not model.is_active:
                continue
            if scenario_type in model.applicable_scenarios:
                applicable.append(model)
        return applicable
    
    def _apply_prediction_model(self, model: PredictionModel, scenario: Scenario, 
                               action: str, context: Dict[str, Any]) -> List[Consequence]:
        """Apply a specific prediction model to generate consequences"""
        try:
            consequences = []
            
            # Generate consequences based on model type and parameters
            if "Interpersonal" in model.model_name:
                consequences = self._generate_interpersonal_consequences(model, scenario, action, context)
            elif "Organizational" in model.model_name:
                consequences = self._generate_organizational_consequences(model, scenario, action, context)
            elif "Ethical" in model.model_name:
                consequences = self._generate_ethical_consequences(model, scenario, action, context)
            else:
                # Generic consequence generation
                consequences = self._generate_generic_consequences(model, scenario, action, context)
            
            return consequences
            
        except Exception as e:
            logger.error(f"Error applying prediction model {model.model_id}: {e}")
            return []
    
    def _generate_interpersonal_consequences(self, model: PredictionModel, scenario: Scenario,
                                           action: str, context: Dict[str, Any]) -> List[Consequence]:
        """Generate interpersonal consequences"""
        consequences = []
        
        # Analyze relationship impact
        if "conflict" in action.lower() or "confront" in action.lower():
            consequences.append(Consequence(
                consequence_id="",
                action_description=action,
                impact_level=ImpactLevel.MODERATE,
                timeframe=Timeframe.IMMEDIATE,
                consequence_type=ConsequenceType.INTERPERSONAL,
                description="This action may create tension in the relationship and require repair efforts",
                probability=0.7,
                affected_parties=[p.name for p in scenario.participants],
                severity_description="Moderate relationship strain",
                reversibility="reversible_with_effort",
                mitigation_strategies=[
                    "Follow up with a clarifying conversation",
                    "Acknowledge the other person's perspective",
                    "Offer to discuss concerns openly"
                ],
                prevention_methods=[
                    "Choose words carefully",
                    "Consider timing and setting",
                    "Prepare for potential emotional reactions"
                ],
                cascading_consequences=[
                    "May affect future communication patterns",
                    "Could influence team dynamics if in work context"
                ],
                ripple_effects=[
                    "Other team members may become more cautious",
                    "Trust levels may be temporarily reduced"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                confidence_score=0.75,
                tags=["interpersonal", "conflict", "relationship"]
            ))
        
        # Analyze communication impact
        if "listen" in action.lower() or "understand" in action.lower():
            consequences.append(Consequence(
                consequence_id="",
                action_description=action,
                impact_level=ImpactLevel.LOW,
                timeframe=Timeframe.IMMEDIATE,
                consequence_type=ConsequenceType.INTERPERSONAL,
                description="This action demonstrates empathy and may strengthen the relationship",
                probability=0.8,
                affected_parties=[p.name for p in scenario.participants],
                severity_description="Positive relationship building",
                reversibility="easily_reversible",
                mitigation_strategies=[],
                prevention_methods=[],
                cascading_consequences=[
                    "May improve future communication",
                    "Could set positive example for others"
                ],
                ripple_effects=[
                    "Team communication may become more open",
                    "Trust levels may increase"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                confidence_score=0.8,
                tags=["interpersonal", "empathy", "positive"]
            ))
        
        return consequences
    
    def _generate_organizational_consequences(self, model: PredictionModel, scenario: Scenario,
                                            action: str, context: Dict[str, Any]) -> List[Consequence]:
        """Generate organizational consequences"""
        consequences = []
        
        # Analyze decision impact
        if "implement" in action.lower() or "change" in action.lower():
            consequences.append(Consequence(
                consequence_id="",
                action_description=action,
                impact_level=ImpactLevel.MODERATE,
                timeframe=Timeframe.SHORT_TERM,
                consequence_type=ConsequenceType.ORGANIZATIONAL,
                description="This action will require resources and may affect team workflows",
                probability=0.9,
                affected_parties=["Team members", "Stakeholders", "Organization"],
                severity_description="Moderate organizational change",
                reversibility="reversible_with_effort",
                mitigation_strategies=[
                    "Provide clear communication about changes",
                    "Offer training and support",
                    "Monitor implementation progress"
                ],
                prevention_methods=[
                    "Conduct thorough impact assessment",
                    "Involve key stakeholders in planning",
                    "Prepare contingency plans"
                ],
                cascading_consequences=[
                    "May require additional training",
                    "Could affect performance metrics"
                ],
                ripple_effects=[
                    "Other teams may need to adjust",
                    "Client expectations may change"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                confidence_score=0.7,
                tags=["organizational", "change", "implementation"]
            ))
        
        return consequences
    
    def _generate_ethical_consequences(self, model: PredictionModel, scenario: Scenario,
                                     action: str, context: Dict[str, Any]) -> List[Consequence]:
        """Generate ethical consequences"""
        consequences = []
        
        # Analyze moral implications
        if "lie" in action.lower() or "deceive" in action.lower():
            consequences.append(Consequence(
                consequence_id="",
                action_description=action,
                impact_level=ImpactLevel.HIGH,
                timeframe=Timeframe.LONG_TERM,
                consequence_type=ConsequenceType.PERSONAL,
                description="This action may damage trust and have long-term ethical implications",
                probability=0.8,
                affected_parties=[p.name for p in scenario.participants],
                severity_description="Significant trust damage",
                reversibility="difficult_to_reverse",
                mitigation_strategies=[
                    "Admit the deception immediately",
                    "Take full responsibility",
                    "Work to rebuild trust over time"
                ],
                prevention_methods=[
                    "Consider long-term consequences",
                    "Evaluate against personal values",
                    "Seek ethical guidance when uncertain"
                ],
                cascading_consequences=[
                    "May affect future credibility",
                    "Could influence others' behavior"
                ],
                ripple_effects=[
                    "Organizational culture may be affected",
                    "Personal reputation may suffer"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                confidence_score=0.85,
                tags=["ethical", "trust", "long_term"]
            ))
        
        return consequences
    
    def _generate_fallback_predictions(self, scenario: Scenario, action: str, 
                                     context: Dict[str, Any]) -> List[Consequence]:
        """Generate fallback predictions when no models are applicable"""
        consequences = []
        
        # Generic consequence based on action characteristics
        if any(word in action.lower() for word in ["help", "support", "assist"]):
            consequences.append(Consequence(
                consequence_id="",
                action_description=action,
                impact_level=ImpactLevel.LOW,
                timeframe=Timeframe.IMMEDIATE,
                consequence_type=ConsequenceType.INTERPERSONAL,
                description="This supportive action may strengthen relationships and build trust",
                probability=0.7,
                affected_parties=[p.name for p in scenario.participants],
                severity_description="Positive relationship impact",
                reversibility="easily_reversible",
                mitigation_strategies=[],
                prevention_methods=[],
                cascading_consequences=[
                    "May encourage reciprocal support",
                    "Could improve team morale"
                ],
                ripple_effects=[
                    "Positive example for others",
                    "Enhanced reputation for helpfulness"
                ],
                created_at=datetime.utcnow(),
                created_by=self.node_id,
                confidence_score=0.6,
                tags=["fallback", "positive", "support"]
            ))
        
        return consequences
    
    def record_actual_outcome(self, scenario_id: str, action: str, 
                             actual_consequences: List[str], outcome_rating: float):
        """Record actual outcomes for accuracy tracking"""
        try:
            if scenario_id not in self.scenario_outcomes:
                self.scenario_outcomes[scenario_id] = {}
            
            self.scenario_outcomes[scenario_id][action] = {
                "actual_consequences": actual_consequences,
                "outcome_rating": outcome_rating,
                "recorded_at": datetime.utcnow().isoformat()
            }
            
            # Update model accuracy if we have predictions to compare
            if scenario_id in self.consequence_history:
                self._update_model_accuracy(scenario_id, action, outcome_rating)
            
            logger.info(f"Recorded actual outcome for scenario {scenario_id}, action: {action}")
            
        except Exception as e:
            logger.error(f"Error recording actual outcome: {e}")
    
    def _update_model_accuracy(self, scenario_id: str, action: str, outcome_rating: float):
        """Update prediction model accuracy based on actual outcomes"""
        try:
            # Find predictions for this scenario and action
            predictions = self.consequence_history.get(scenario_id, [])
            relevant_predictions = [p for p in predictions if p.action_description == action]
            
            if not relevant_predictions:
                return
            
            # Calculate prediction accuracy (simplified)
            # In practice, this would be more sophisticated
            avg_confidence = sum(p.confidence_score for p in relevant_predictions) / len(relevant_predictions)
            accuracy_delta = outcome_rating - avg_confidence
            
            # Update accuracy for models that made predictions
            for prediction in relevant_predictions:
                # Find which model made this prediction (simplified)
                # In practice, we'd track which model generated each prediction
                for model in self.prediction_models.values():
                    if model.model_id not in self.prediction_accuracy:
                        self.prediction_accuracy[model.model_id] = []
                    
                    self.prediction_accuracy[model.model_id].append(
                        (datetime.utcnow(), accuracy_delta)
                    )
                    
                    # Update model accuracy score
                    recent_accuracy = self.prediction_accuracy[model.model_id][-10:]  # Last 10 predictions
                    if recent_accuracy:
                        avg_accuracy = sum(acc for _, acc in recent_accuracy) / len(recent_accuracy)
                        model.accuracy_score = max(0.0, min(1.0, model.accuracy_score + avg_accuracy * 0.1))
            
        except Exception as e:
            logger.error(f"Error updating model accuracy: {e}")
    
    def get_consequence_summary(self, scenario_id: str) -> Dict[str, Any]:
        """Get summary of consequences for a scenario"""
        if scenario_id not in self.consequence_history:
            return {
                "scenario_id": scenario_id,
                "total_predictions": 0,
                "consequences_by_impact": {},
                "consequences_by_timeframe": {},
                "consequences_by_type": {}
            }
        
        consequences = self.consequence_history[scenario_id]
        
        # Group by impact level
        by_impact = {}
        for consequence in consequences:
            impact = consequence.impact_level.value
            by_impact[impact] = by_impact.get(impact, 0) + 1
        
        # Group by timeframe
        by_timeframe = {}
        for consequence in consequences:
            timeframe = consequence.timeframe.value
            by_timeframe[timeframe] = by_timeframe.get(timeframe, 0) + 1
        
        # Group by type
        by_type = {}
        for consequence in consequences:
            ctype = consequence.consequence_type.value
            by_type[ctype] = by_type.get(ctype, 0) + 1
        
        return {
            "scenario_id": scenario_id,
            "total_predictions": len(consequences),
            "consequences_by_impact": by_impact,
            "consequences_by_timeframe": by_timeframe,
            "consequences_by_type": by_type,
            "high_impact_consequences": [
                c for c in consequences 
                if c.impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL, ImpactLevel.CATASTROPHIC]
            ]
        }
    
    def get_prediction_models_summary(self) -> Dict[str, Any]:
        """Get summary of all prediction models"""
        total_models = len(self.prediction_models)
        active_models = len([m for m in self.prediction_models.values() if m.is_active])
        
        # Average accuracy by model type
        model_performance = {}
        for model in self.prediction_models.values():
            model_type = model.model_name.split()[0]  # First word of model name
            if model_type not in model_performance:
                model_performance[model_type] = []
            model_performance[model_type].append(model.accuracy_score)
        
        # Calculate averages
        for model_type in model_performance:
            model_performance[model_type] = sum(model_performance[model_type]) / len(model_performance[model_type])
        
        return {
            "total_models": total_models,
            "active_models": active_models,
            "model_performance": model_performance,
            "models": [m.to_dict() for m in self.prediction_models.values()]
        }
