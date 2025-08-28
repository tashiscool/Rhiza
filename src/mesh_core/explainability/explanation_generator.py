"""
Explanation Generator
====================

Generates comprehensive explanations for AI decisions and behaviors
within The Mesh network, ensuring transparency and accountability.
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations"""
    DECISION_RATIONALE = "decision_rationale"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    PROCESS_FLOW = "process_flow"
    CONFIDENCE_BREAKDOWN = "confidence_breakdown"
    ALTERNATIVE_ANALYSIS = "alternative_analysis"

@dataclass
class Explanation:
    """Generated explanation"""
    explanation_id: str
    explanation_type: ExplanationType
    subject: str
    explanation_text: str
    confidence_score: float
    supporting_evidence: List[Dict]
    generated_at: float
    detail_level: str = "medium"
    
class ExplanationGenerator:
    """Generates explanations for decisions and behaviors"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.explanations: Dict[str, Explanation] = {}
        
    async def generate_explanation(
        self,
        subject: str,
        explanation_type: ExplanationType,
        context: Dict[str, Any],
        detail_level: str = "medium"
    ) -> str:
        """Generate explanation for a subject"""
        explanation_id = self._generate_id(subject, explanation_type)
        
        explanation_text = await self._create_explanation_text(
            subject, explanation_type, context, detail_level
        )
        
        explanation = Explanation(
            explanation_id=explanation_id,
            explanation_type=explanation_type,
            subject=subject,
            explanation_text=explanation_text,
            confidence_score=0.85,
            supporting_evidence=context.get("evidence", []),
            generated_at=time.time(),
            detail_level=detail_level
        )
        
        self.explanations[explanation_id] = explanation
        return explanation_id
    
    def _generate_id(self, subject: str, explanation_type: ExplanationType) -> str:
        """Generate unique explanation ID"""
        content = f"{subject}|{explanation_type.value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _create_explanation_text(
        self,
        subject: str,
        explanation_type: ExplanationType,
        context: Dict[str, Any],
        detail_level: str
    ) -> str:
        """Create explanation text based on type and context"""
        if explanation_type == ExplanationType.DECISION_RATIONALE:
            return f"Decision for {subject} was made based on {len(context.get('factors', []))} key factors."
        elif explanation_type == ExplanationType.BEHAVIOR_ANALYSIS:
            return f"Behavior analysis for {subject} shows consistent patterns in {context.get('domain', 'general')} domain."
        else:
            return f"Explanation for {subject} generated with {detail_level} detail level."