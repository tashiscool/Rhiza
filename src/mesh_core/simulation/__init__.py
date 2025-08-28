"""
Mesh Simulation & Scenario Engine
================================

Provides counterfactual reasoning and empathic rehearsal capabilities:
- Scenario generation for behavioral coaching
- Choice rehearsal mechanisms
- Empathy training through simulation
- Consequence prediction and outcome tracking
- Multi-agent coordination and justice modeling
"""

from .scenario_generator import ScenarioGenerator
from .choice_rehearser import ChoiceRehearser
from .empathy_trainer import EmpathyTrainer
from .consequence_predictor import ConsequencePredictor
from .scenario_sharer import ScenarioSharer

__all__ = [
    "ScenarioGenerator",
    "ChoiceRehearser",
    "EmpathyTrainer",
    "ConsequencePredictor",
    "ScenarioSharer"
]

