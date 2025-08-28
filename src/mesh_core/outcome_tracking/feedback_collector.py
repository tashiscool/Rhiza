"""
Feedback Collector
==================

Collects and manages feedback on outcomes within The Mesh network.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class FeedbackSource(Enum):
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"

@dataclass
class FeedbackEntry:
    entry_id: str
    source: FeedbackSource
    feedback_type: FeedbackType
    content: str
    confidence: float

class FeedbackCollector:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.feedback_entries: List[FeedbackEntry] = []
        logger.info(f"FeedbackCollector initialized for node {node_id}")
        
    async def collect_feedback(self, feedback: FeedbackEntry) -> bool:
        self.feedback_entries.append(feedback)
        return True