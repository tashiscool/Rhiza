"""
Fact Extractor - Enhanced with Sentient's Memory Concepts

Integrates Sentient's proven memory patterns:
- Atomic fact decomposition and extraction
- Personalization and user-specific fact creation
- Noise filtering and content cleaning
- Structured fact classification and analysis

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/memories')
    from prompts import (
        fact_extraction_system_prompt_template,
        fact_extraction_user_prompt_template
    )
    from formats import fact_analysis_required_format
    from constants import TOPICS
    SENTIENT_MEMORY_AVAILABLE = True
except ImportError:
    SENTIENT_MEMORY_AVAILABLE = False
    # Mock constants for development/testing
    TOPICS = [
        {"name": "Personal Identity", "description": "Core traits, personality, beliefs, values, ethics, and preferences"},
        {"name": "Interests & Lifestyle", "description": "Hobbies, recreational activities, habits, routines, daily behavior"},
        {"name": "Work & Learning", "description": "Career, jobs, professional achievements, academic background, skills, certifications"},
        {"name": "Health & Wellbeing", "description": "Mental and physical health, self-care practices"},
        {"name": "Relationships & Social Life", "description": "Family, friends, romantic connections, social interactions, social media"},
        {"name": "Financial", "description": "Income, expenses, investments, financial goals"},
        {"name": "Goals & Challenges", "description": "Aspirations, objectives, obstacles, and difficulties faced"},
        {"name": "Miscellaneous", "description": "Anything that doesn't clearly fit into the above"}
    ]

@dataclass
class ExtractedFact:
    """A single extracted fact with metadata"""
    content: str
    topics: List[str]
    memory_type: str  # "long-term" or "short-term"
    duration: Optional[str] = None
    confidence: float = 1.0
    source: str = "extraction"
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class FactExtractionResult:
    """Result of fact extraction process"""
    facts: List[ExtractedFact]
    total_facts: int
    processing_time: float
    source_length: int
    extraction_ratio: float
    metadata: Dict[str, Any]

@dataclass
class FactExtractionConfig:
    """Configuration for fact extraction"""
    enable_personalization: bool = True
    enable_noise_filtering: bool = True
    enable_topic_classification: bool = True
    enable_duration_estimation: bool = True
    min_confidence_threshold: float = 0.7
    max_facts_per_source: int = 50
    enable_debug_logging: bool = False

class FactExtractor:
    """
    Enhanced fact extractor integrating Sentient's memory concepts
    
    Provides:
    - Atomic fact decomposition and extraction
    - Personalization and user-specific fact creation
    - Noise filtering and content cleaning
    - Structured fact classification and analysis
    """
    
    def __init__(self, config: FactExtractionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize extraction components
        self.topic_classifier = None
        self.duration_estimator = None
        self.personalizer = None
        
        # Performance tracking
        self.extraction_count = 0
        self.total_facts_extracted = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_extraction_components()
        
        self.logger.info("Fact Extractor initialized with Sentient concepts")
    
    def _initialize_extraction_components(self):
        """Initialize fact extraction components using Sentient patterns"""
        
        try:
            # Initialize topic classifier
            if self.config.enable_topic_classification:
                self._initialize_topic_classifier()
            
            # Initialize duration estimator
            if self.config.enable_duration_estimation:
                self._initialize_duration_estimator()
            
            # Initialize personalizer
            if self.config.enable_personalization:
                self._initialize_personalizer()
            
            self.logger.info("Fact extraction components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize extraction components: {e}")
            self.logger.warning("Fact extraction will use basic methods")
    
    def _initialize_topic_classifier(self):
        """Initialize topic classification component"""
        
        try:
            self.topic_classifier = TopicClassifier(TOPICS)
            self.logger.info("Topic classifier initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize topic classifier: {e}")
            self.topic_classifier = None
    
    def _initialize_duration_estimator(self):
        """Initialize duration estimation component"""
        
        try:
            self.duration_estimator = DurationEstimator()
            self.logger.info("Duration estimator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize duration estimator: {e}")
            self.duration_estimator = None
    
    def _initialize_personalizer(self):
        """Initialize personalization component"""
        
        try:
            self.personalizer = FactPersonalizer()
            self.logger.info("Fact personalizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fact personalizer: {e}")
            self.personalizer = None
    
    async def extract_facts(self, text: str, username: str = "user", 
                           source: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> FactExtractionResult:
        """
        Extract facts from text using Sentient's extraction patterns
        
        Args:
            text: Text to extract facts from
            username: Username for personalization
            source: Source of the text
            metadata: Additional metadata
            
        Returns:
            FactExtractionResult with extracted facts and metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Extract atomic facts
            raw_facts = await self._extract_atomic_facts(cleaned_text, username)
            
            # Filter and validate facts
            valid_facts = await self._filter_facts(raw_facts, username)
            
            # Classify and enhance facts
            enhanced_facts = await self._enhance_facts(valid_facts, source, metadata)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance tracking
            self.extraction_count += 1
            self.total_facts_extracted += len(enhanced_facts)
            self.total_processing_time += processing_time
            
            # Calculate extraction ratio
            extraction_ratio = len(enhanced_facts) / max(len(text.split()), 1)
            
            result = FactExtractionResult(
                facts=enhanced_facts,
                total_facts=len(enhanced_facts),
                processing_time=processing_time,
                source_length=len(text),
                extraction_ratio=extraction_ratio,
                metadata={
                    "username": username,
                    "source": source,
                    "cleaned_length": len(cleaned_text),
                    "raw_facts_count": len(raw_facts),
                    "valid_facts_count": len(valid_facts),
                    **(metadata or {})
                }
            )
            
            self.logger.info(f"Fact extraction completed in {processing_time:.3f}s: {len(enhanced_facts)} facts from {len(text)} chars")
            return result
            
        except Exception as e:
            self.logger.error(f"Fact extraction failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return FactExtractionResult(
                facts=[],
                total_facts=0,
                processing_time=processing_time,
                source_length=len(text),
                extraction_ratio=0.0,
                metadata={
                    "error": str(e),
                    "username": username,
                    "source": source,
                    **(metadata or {})
                }
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for fact extraction"""
        
        try:
            cleaned = text.strip()
            
            # Remove common noise patterns
            if self.config.enable_noise_filtering:
                # Remove email signatures
                cleaned = re.sub(r'--\s*\n.*', '', cleaned, flags=re.DOTALL)
                
                # Remove common UI text
                ui_patterns = [
                    r'Click here to view',
                    r'Reply to this email',
                    r'Unsubscribe',
                    r'View in browser',
                    r'Notification from',
                    r'Avatar of',
                    r'Subject:',
                    r'Fwd:',
                    r'Re:'
                ]
                
                for pattern in ui_patterns:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                
                # Remove multiple whitespace
                cleaned = re.sub(r'\s+', ' ', cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {e}")
            return text
    
    async def _extract_atomic_facts(self, text: str, username: str) -> List[str]:
        """Extract atomic facts from text using Sentient's patterns"""
        
        try:
            if not SENTIENT_MEMORY_AVAILABLE:
                # Fallback to basic extraction
                return self._basic_fact_extraction(text, username)
            
            # Use Sentient's extraction prompt
            system_prompt = fact_extraction_system_prompt_template
            user_prompt = fact_extraction_user_prompt_template.format(
                username=username,
                paragraph=text
            )
            
            # For now, we'll use a simplified extraction approach
            # In the future, we can integrate with an LLM for full extraction
            facts = self._extract_facts_with_patterns(text, username)
            
            if self.config.enable_debug_logging:
                self.logger.debug(f"Extracted {len(facts)} facts using Sentient patterns")
            
            return facts
            
        except Exception as e:
            self.logger.error(f"Atomic fact extraction failed: {e}")
            return self._basic_fact_extraction(text, username)
    
    def _extract_facts_with_patterns(self, text: str, username: str) -> List[str]:
        """Extract facts using pattern-based approach (Sentient-inspired)"""
        
        facts = []
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue
                
                # Personalize the sentence
                personalized = self._personalize_text(sentence, username)
                
                # Check if it contains extractable information
                if self._is_extractable_fact(personalized):
                    facts.append(personalized)
                
                # Split compound sentences
                compound_facts = self._split_compound_sentences(personalized, username)
                facts.extend(compound_facts)
            
            # Remove duplicates and filter
            unique_facts = list(set(facts))
            filtered_facts = [f for f in unique_facts if self._is_valid_fact(f)]
            
            return filtered_facts[:self.config.max_facts_per_source]
            
        except Exception as e:
            self.logger.error(f"Pattern-based extraction failed: {e}")
            return []
    
    def _personalize_text(self, text: str, username: str) -> str:
        """Personalize text by replacing generic references with username"""
        
        try:
            if not self.config.enable_personalization:
                return text
            
            # Replace common patterns
            replacements = [
                (r'\bmy\b', f"{username}'s"),
                (r'\bI\b', username),
                (r'\bme\b', username),
                (r'\bthe user\b', username),
                (r'\buser\b', username)
            ]
            
            personalized = text
            for pattern, replacement in replacements:
                personalized = re.sub(pattern, replacement, personalized, flags=re.IGNORECASE)
            
            return personalized
            
        except Exception as e:
            self.logger.error(f"Text personalization failed: {e}")
            return text
    
    def _is_extractable_fact(self, text: str) -> bool:
        """Check if text contains extractable factual information"""
        
        try:
            # Skip if too short
            if len(text) < 10:
                return False
            
            # Skip if it's a question
            if text.strip().endswith('?'):
                return False
            
            # Skip if it's a command
            command_indicators = ['please', 'could you', 'would you', 'let me know', 'tell me']
            if any(indicator in text.lower() for indicator in command_indicators):
                return False
            
            # Skip if it's just punctuation or whitespace
            if not re.search(r'[a-zA-Z]', text):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fact validation failed: {e}")
            return False
    
    def _split_compound_sentences(self, text: str, username: str) -> List[str]:
        """Split compound sentences into atomic facts"""
        
        facts = []
        
        try:
            # Split on common conjunctions
            conjunctions = [' and ', ' but ', ' while ', ' however ', ' although ', ' though ']
            
            for conjunction in conjunctions:
                if conjunction in text:
                    parts = text.split(conjunction)
                    for part in parts:
                        part = part.strip()
                        if part and self._is_extractable_fact(part):
                            facts.append(part)
                    return facts
            
            # If no conjunctions, return the original as a single fact
            if self._is_extractable_fact(text):
                facts.append(text)
            
            return facts
            
        except Exception as e:
            self.logger.error(f"Compound sentence splitting failed: {e}")
            return [text] if self._is_extractable_fact(text) else []
    
    def _is_valid_fact(self, fact: str) -> bool:
        """Validate if a fact is valid and meaningful"""
        
        try:
            # Skip if too short
            if len(fact) < 10:
                return False
            
            # Skip if it's just noise
            noise_patterns = [
                r'^\s*$',  # Empty or whitespace only
                r'^[^\w]*$',  # No word characters
                r'^[A-Z\s]+$',  # All caps (likely headers)
            ]
            
            for pattern in noise_patterns:
                if re.match(pattern, fact):
                    return False
            
            # Must contain some meaningful content
            if not re.search(r'\b\w+\b', fact):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fact validation failed: {e}")
            return False
    
    def _basic_fact_extraction(self, text: str, username: str) -> List[str]:
        """Basic fact extraction as fallback"""
        
        try:
            # Simple sentence-based extraction
            sentences = re.split(r'[.!?]+', text)
            facts = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    personalized = self._personalize_text(sentence, username)
                    if self._is_valid_fact(personalized):
                        facts.append(personalized)
            
            return facts[:self.config.max_facts_per_source]
            
        except Exception as e:
            self.logger.error(f"Basic fact extraction failed: {e}")
            return []
    
    async def _filter_facts(self, facts: List[str], username: str) -> List[str]:
        """Filter and validate extracted facts"""
        
        try:
            filtered_facts = []
            
            for fact in facts:
                if self._is_valid_fact(fact):
                    # Additional filtering based on Sentient patterns
                    if self._passes_sentient_filters(fact):
                        filtered_facts.append(fact)
            
            return filtered_facts
            
        except Exception as e:
            self.logger.error(f"Fact filtering failed: {e}")
            return facts
    
    def _passes_sentient_filters(self, fact: str) -> bool:
        """Check if fact passes Sentient's filtering criteria"""
        
        try:
            # Skip temporary information
            temporary_patterns = [
                r'\b(today|tomorrow|yesterday)\b',
                r'\b(now|currently|at the moment)\b',
                r'\b(meeting|appointment|call)\s+(is|at|on)\b'
            ]
            
            for pattern in temporary_patterns:
                if re.search(pattern, fact, re.IGNORECASE):
                    return False
            
            # Skip vague statements
            vague_patterns = [
                r'\b(see below|here is|let me know)\b',
                r'\b(details|information|thoughts)\b',
                r'\b(completed|finished|done)\b'
            ]
            
            for pattern in vague_patterns:
                if re.search(pattern, fact, re.IGNORECASE):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sentient filtering failed: {e}")
            return True
    
    async def _enhance_facts(self, facts: List[str], source: str, 
                            metadata: Optional[Dict[str, Any]]) -> List[ExtractedFact]:
        """Enhance facts with classification and metadata"""
        
        try:
            enhanced_facts = []
            
            for fact in facts:
                # Classify topics
                topics = await self._classify_topics(fact)
                
                # Determine memory type and duration
                memory_type, duration = await self._classify_memory_type(fact)
                
                # Create enhanced fact
                enhanced_fact = ExtractedFact(
                    content=fact,
                    topics=topics,
                    memory_type=memory_type,
                    duration=duration,
                    confidence=1.0,  # Default confidence
                    source=source,
                    timestamp=asyncio.get_event_loop().time(),
                    metadata=metadata or {}
                )
                
                enhanced_facts.append(enhanced_fact)
            
            return enhanced_facts
            
        except Exception as e:
            self.logger.error(f"Fact enhancement failed: {e}")
            return []
    
    async def _classify_topics(self, fact: str) -> List[str]:
        """Classify fact into relevant topics"""
        
        try:
            if not self.topic_classifier:
                return ["Miscellaneous"]
            
            topics = await self.topic_classifier.classify(fact)
            return topics if topics else ["Miscellaneous"]
            
        except Exception as e:
            self.logger.error(f"Topic classification failed: {e}")
            return ["Miscellaneous"]
    
    async def _classify_memory_type(self, fact: str) -> Tuple[str, Optional[str]]:
        """Classify fact as long-term or short-term memory"""
        
        try:
            if not self.duration_estimator:
                return "long-term", None
            
            memory_type, duration = await self.duration_estimator.estimate(fact)
            return memory_type, duration
            
        except Exception as e:
            self.logger.error(f"Memory type classification failed: {e}")
            return "long-term", None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for fact extraction"""
        
        avg_processing_time = (self.total_processing_time / self.extraction_count 
                             if self.extraction_count > 0 else 0)
        
        avg_facts_per_extraction = (self.total_facts_extracted / self.extraction_count 
                                  if self.extraction_count > 0 else 0)
        
        return {
            "extraction_count": self.extraction_count,
            "total_facts_extracted": self.total_facts_extracted,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "average_facts_per_extraction": avg_facts_per_extraction,
            "topic_classifier_available": self.topic_classifier is not None,
            "duration_estimator_available": self.duration_estimator is not None,
            "personalizer_available": self.personalizer is not None
        }
    
    async def cleanup(self):
        """Clean up extraction resources"""
        
        try:
            if hasattr(self.topic_classifier, 'cleanup'):
                await self.topic_classifier.cleanup()
            
            if hasattr(self.duration_estimator, 'cleanup'):
                await self.duration_estimator.cleanup()
            
            if hasattr(self.personalizer, 'cleanup'):
                await self.personalizer.cleanup()
                
            self.logger.info("Fact extractor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class TopicClassifier:
    """Topic classification component following Sentient patterns"""
    
    def __init__(self, topics: List[Dict[str, str]]):
        self.topics = topics
        self.topic_names = [topic["name"] for topic in topics]
        self.logger = logging.getLogger(__name__)
    
    async def classify(self, fact: str) -> List[str]:
        """Classify fact into relevant topics"""
        
        try:
            # Simple keyword-based classification
            # In the future, we can integrate with an LLM for better classification
            
            relevant_topics = []
            
            # Check each topic for relevant keywords
            for topic in self.topics:
                if self._topic_matches(fact, topic):
                    relevant_topics.append(topic["name"])
            
            # If no topics match, use Miscellaneous
            if not relevant_topics:
                relevant_topics = ["Miscellaneous"]
            
            return relevant_topics
            
        except Exception as e:
            self.logger.error(f"Topic classification failed: {e}")
            return ["Miscellaneous"]
    
    def _topic_matches(self, fact: str, topic: Dict[str, str]) -> bool:
        """Check if fact matches a topic based on keywords"""
        
        try:
            fact_lower = fact.lower()
            
            # Define keywords for each topic
            topic_keywords = {
                "Personal Identity": ["name", "age", "birthday", "personality", "beliefs", "values", "preferences"],
                "Interests & Lifestyle": ["hobby", "interest", "like", "enjoy", "favorite", "routine", "habit"],
                "Work & Learning": ["job", "work", "career", "project", "skill", "certification", "degree"],
                "Health & Wellbeing": ["health", "medical", "exercise", "diet", "mental", "physical", "wellness"],
                "Relationships & Social Life": ["family", "friend", "relationship", "spouse", "partner", "social"],
                "Financial": ["money", "income", "expense", "investment", "budget", "financial", "salary"],
                "Goals & Challenges": ["goal", "objective", "challenge", "aspiration", "dream", "target"]
            }
            
            topic_name = topic["name"]
            if topic_name in topic_keywords:
                keywords = topic_keywords[topic_name]
                return any(keyword in fact_lower for keyword in keywords)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Topic matching failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up topic classifier resources"""
        pass


class DurationEstimator:
    """Duration estimation component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def estimate(self, fact: str) -> Tuple[str, Optional[str]]:
        """Estimate memory type and duration for a fact"""
        
        try:
            fact_lower = fact.lower()
            
            # Check for short-term indicators
            short_term_indicators = [
                "today", "tomorrow", "yesterday", "this week", "next week",
                "meeting", "appointment", "deadline", "reminder", "schedule",
                "temporary", "for now", "currently", "at the moment"
            ]
            
            is_short_term = any(indicator in fact_lower for indicator in short_term_indicators)
            
            if is_short_term:
                duration = self._estimate_short_term_duration(fact)
                return "short-term", duration
            else:
                return "long-term", None
                
        except Exception as e:
            self.logger.error(f"Duration estimation failed: {e}")
            return "long-term", None
    
    def _estimate_short_term_duration(self, fact: str) -> str:
        """Estimate duration for short-term facts"""
        
        try:
            fact_lower = fact.lower()
            
            # Time-based patterns
            if "today" in fact_lower:
                return "1 day"
            elif "tomorrow" in fact_lower:
                return "1 day"
            elif "this week" in fact_lower:
                return "1 week"
            elif "next week" in fact_lower:
                return "1 week"
            elif "meeting" in fact_lower:
                return "2 hours"
            elif "appointment" in fact_lower:
                return "1 hour"
            elif "deadline" in fact_lower:
                return "1 day"
            else:
                return "1 day"  # Default duration
                
        except Exception as e:
            self.logger.error(f"Short-term duration estimation failed: {e}")
            return "1 day"
    
    async def cleanup(self):
        """Clean up duration estimator resources"""
        pass


class FactPersonalizer:
    """Fact personalization component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def personalize(self, fact: str, username: str) -> str:
        """Personalize fact by replacing generic references"""
        
        try:
            # This is handled in the main extractor for now
            # Can be extended for more complex personalization
            return fact
            
        except Exception as e:
            self.logger.error(f"Fact personalization failed: {e}")
            return fact
    
    async def cleanup(self):
        """Clean up personalizer resources"""
        pass


# Factory function for easy integration
def create_fact_extractor(config: Optional[FactExtractionConfig] = None) -> FactExtractor:
    """Create a fact extractor with default or custom configuration"""
    
    if config is None:
        config = FactExtractionConfig()
    
    return FactExtractor(config)
