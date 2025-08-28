"""
Context Framing System
======================

Provides cultural, regional, and temporal context framing for information
flowing through The Mesh. Helps identify biases and provide appropriate
context for different communities and time periods.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context framing"""
    CULTURAL = "cultural"       # Cultural background context
    REGIONAL = "regional"       # Geographic/regional context
    TEMPORAL = "temporal"       # Time-based context
    LINGUISTIC = "linguistic"  # Language and communication context
    SOCIAL = "social"          # Social group context
    ECONOMIC = "economic"      # Economic context
    POLITICAL = "political"    # Political context
    TECHNOLOGICAL = "tech"     # Technology context

class BiasType(Enum):
    """Types of identified biases"""
    CULTURAL_BIAS = "cultural"
    SELECTION_BIAS = "selection"
    CONFIRMATION_BIAS = "confirmation"
    TEMPORAL_BIAS = "temporal"
    LINGUISTIC_BIAS = "linguistic"
    DEMOGRAPHIC_BIAS = "demographic"
    SOURCE_BIAS = "source"

@dataclass
class ContextFrame:
    """Context frame for information"""
    frame_id: str
    context_type: ContextType
    context_value: str          # e.g., "Western", "2024", "English"
    confidence: float          # Confidence in context assignment
    source_indicators: List[str]  # What indicated this context
    created_at: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['context_type'] = self.context_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContextFrame':
        data['context_type'] = ContextType(data['context_type'])
        return cls(**data)

@dataclass
class BiasIndicator:
    """Identified bias in information"""
    bias_id: str
    bias_type: BiasType
    strength: float            # 0.0 to 1.0
    description: str
    evidence: List[str]        # Evidence for the bias
    mitigation: Optional[str]  # Suggested mitigation
    detected_at: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['bias_type'] = self.bias_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BiasIndicator':
        data['bias_type'] = BiasType(data['bias_type'])
        return cls(**data)

class ContextFraming:
    """
    Context framing and bias detection system
    
    Analyzes information to identify cultural, temporal, and other contextual
    frames, as well as potential biases that should be disclosed.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.context_frames: Dict[str, List[ContextFrame]] = {}  # item_id -> frames
        self.bias_indicators: Dict[str, List[BiasIndicator]] = {}  # item_id -> biases
        self.context_patterns: Dict[ContextType, Dict] = self._init_context_patterns()
        self.bias_detection_rules: Dict[BiasType, Dict] = self._init_bias_rules()
        
    def _init_context_patterns(self) -> Dict[ContextType, Dict]:
        """Initialize context detection patterns"""
        return {
            ContextType.CULTURAL: {
                'keywords': {
                    'Western': ['democracy', 'individual rights', 'capitalism', 'secularism'],
                    'Eastern': ['harmony', 'collective', 'hierarchy', 'tradition'],
                    'Islamic': ['sharia', 'ummah', 'halal', 'mosque'],
                    'Chinese': ['guanxi', 'face', 'harmony', 'collective'],
                    'Indigenous': ['tribal', 'traditional knowledge', 'ancestral', 'sacred']
                },
                'language_indicators': {
                    'English': ['the', 'and', 'of', 'to', 'a'],
                    'Spanish': ['el', 'la', 'de', 'que', 'y'],
                    'Chinese': ['的', '是', '了', '在', '我'],
                    'Arabic': ['في', 'من', 'إلى', 'على', 'أن']
                }
            },
            ContextType.TEMPORAL: {
                'time_markers': {
                    '2020s': ['covid', 'pandemic', 'remote work', 'climate change'],
                    '2010s': ['social media', 'smartphone', 'arab spring', 'brexit'],
                    '2000s': ['9/11', 'iraq war', 'social network', 'google'],
                    '1990s': ['internet', 'cold war end', 'clinton', 'windows']
                }
            },
            ContextType.REGIONAL: {
                'geographic_markers': {
                    'US': ['states', 'congress', 'constitution', 'federal'],
                    'EU': ['european union', 'brussels', 'euro', 'brexit'],
                    'Asia': ['asia-pacific', 'asean', 'monsoon', 'rice'],
                    'Africa': ['sahara', 'sahel', 'ubuntu', 'african union']
                }
            },
            ContextType.ECONOMIC: {
                'economic_systems': {
                    'Capitalist': ['free market', 'private property', 'competition'],
                    'Socialist': ['collective ownership', 'public goods', 'redistribution'],
                    'Mixed': ['regulated market', 'social safety net', 'public-private']
                }
            }
        }
    
    def _init_bias_rules(self) -> Dict[BiasType, Dict]:
        """Initialize bias detection rules"""
        return {
            BiasType.CULTURAL_BIAS: {
                'indicators': [
                    'assumes Western values',
                    'ignores non-Western perspectives',
                    'cultural stereotyping'
                ],
                'patterns': ['obviously', 'naturally', 'of course', 'everyone knows']
            },
            BiasType.SELECTION_BIAS: {
                'indicators': [
                    'cherry-picked examples',
                    'unrepresentative sample',
                    'excluded contradictory evidence'
                ],
                'patterns': ['only shows', 'ignores', 'fails to mention']
            },
            BiasType.CONFIRMATION_BIAS: {
                'indicators': [
                    'seeks confirming evidence only',
                    'dismisses contradictory views',
                    'motivated reasoning'
                ],
                'patterns': ['proves that', 'confirms', 'as expected']
            },
            BiasType.TEMPORAL_BIAS: {
                'indicators': [
                    'presentism',
                    'outdated assumptions',
                    'ignores historical context'
                ],
                'patterns': ['has always been', 'never changes', 'timeless']
            }
        }
    
    async def analyze_context(self, item_id: str, content: str, metadata: Dict) -> List[ContextFrame]:
        """Analyze content to identify contextual frames"""
        
        detected_frames = []
        content_lower = content.lower()
        
        for context_type, patterns in self.context_patterns.items():
            frames = await self._detect_context_type(item_id, content_lower, context_type, patterns, metadata)
            detected_frames.extend(frames)
        
        # Store detected frames
        self.context_frames[item_id] = detected_frames
        
        logger.info(f"Detected {len(detected_frames)} context frames for item {item_id}")
        return detected_frames
    
    async def _detect_context_type(
        self,
        item_id: str,
        content: str,
        context_type: ContextType,
        patterns: Dict,
        metadata: Dict
    ) -> List[ContextFrame]:
        """Detect specific type of context"""
        
        frames = []
        
        if context_type == ContextType.CULTURAL:
            frames.extend(await self._detect_cultural_context(item_id, content, patterns))
        elif context_type == ContextType.TEMPORAL:
            frames.extend(await self._detect_temporal_context(item_id, content, patterns, metadata))
        elif context_type == ContextType.REGIONAL:
            frames.extend(await self._detect_regional_context(item_id, content, patterns))
        elif context_type == ContextType.ECONOMIC:
            frames.extend(await self._detect_economic_context(item_id, content, patterns))
        
        return frames
    
    async def _detect_cultural_context(self, item_id: str, content: str, patterns: Dict) -> List[ContextFrame]:
        """Detect cultural context frames"""
        
        frames = []
        
        # Check for cultural keywords
        for culture, keywords in patterns.get('keywords', {}).items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in content:
                    matches.append(keyword)
            
            if matches:
                confidence = min(1.0, len(matches) / len(keywords))
                frame = ContextFrame(
                    frame_id=f"{item_id}_cultural_{culture.lower()}",
                    context_type=ContextType.CULTURAL,
                    context_value=culture,
                    confidence=confidence,
                    source_indicators=matches,
                    created_at=time.time(),
                    metadata={'detection_method': 'keyword_matching'}
                )
                frames.append(frame)
        
        # Check for language indicators
        for language, indicators in patterns.get('language_indicators', {}).items():
            matches = sum(1 for indicator in indicators if indicator in content)
            if matches >= 2:  # Require at least 2 matches
                confidence = min(1.0, matches / len(indicators))
                frame = ContextFrame(
                    frame_id=f"{item_id}_linguistic_{language.lower()}",
                    context_type=ContextType.LINGUISTIC,
                    context_value=language,
                    confidence=confidence,
                    source_indicators=indicators[:matches],
                    created_at=time.time(),
                    metadata={'detection_method': 'language_indicators'}
                )
                frames.append(frame)
        
        return frames
    
    async def _detect_temporal_context(self, item_id: str, content: str, patterns: Dict, metadata: Dict) -> List[ContextFrame]:
        """Detect temporal context frames"""
        
        frames = []
        
        # Check explicit date in metadata
        if 'timestamp' in metadata:
            timestamp = metadata['timestamp']
            year = time.gmtime(timestamp).tm_year
            decade = f"{year // 10 * 10}s"
            
            frame = ContextFrame(
                frame_id=f"{item_id}_temporal_explicit",
                context_type=ContextType.TEMPORAL,
                context_value=str(year),
                confidence=1.0,
                source_indicators=['explicit_timestamp'],
                created_at=time.time(),
                metadata={'year': year, 'decade': decade}
            )
            frames.append(frame)
        
        # Check for time period markers
        for period, markers in patterns.get('time_markers', {}).items():
            matches = []
            for marker in markers:
                if marker.lower() in content:
                    matches.append(marker)
            
            if matches:
                confidence = min(1.0, len(matches) / len(markers))
                frame = ContextFrame(
                    frame_id=f"{item_id}_temporal_{period}",
                    context_type=ContextType.TEMPORAL,
                    context_value=period,
                    confidence=confidence,
                    source_indicators=matches,
                    created_at=time.time(),
                    metadata={'detection_method': 'temporal_markers'}
                )
                frames.append(frame)
        
        return frames
    
    async def _detect_regional_context(self, item_id: str, content: str, patterns: Dict) -> List[ContextFrame]:
        """Detect regional context frames"""
        
        frames = []
        
        for region, markers in patterns.get('geographic_markers', {}).items():
            matches = []
            for marker in markers:
                if marker.lower() in content:
                    matches.append(marker)
            
            if matches:
                confidence = min(1.0, len(matches) / len(markers))
                frame = ContextFrame(
                    frame_id=f"{item_id}_regional_{region.lower()}",
                    context_type=ContextType.REGIONAL,
                    context_value=region,
                    confidence=confidence,
                    source_indicators=matches,
                    created_at=time.time(),
                    metadata={'detection_method': 'geographic_markers'}
                )
                frames.append(frame)
        
        return frames
    
    async def _detect_economic_context(self, item_id: str, content: str, patterns: Dict) -> List[ContextFrame]:
        """Detect economic context frames"""
        
        frames = []
        
        for system, keywords in patterns.get('economic_systems', {}).items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in content:
                    matches.append(keyword)
            
            if matches:
                confidence = min(1.0, len(matches) / len(keywords))
                frame = ContextFrame(
                    frame_id=f"{item_id}_economic_{system.lower()}",
                    context_type=ContextType.ECONOMIC,
                    context_value=system,
                    confidence=confidence,
                    source_indicators=matches,
                    created_at=time.time(),
                    metadata={'detection_method': 'economic_indicators'}
                )
                frames.append(frame)
        
        return frames
    
    async def detect_biases(self, item_id: str, content: str, context_frames: List[ContextFrame]) -> List[BiasIndicator]:
        """Detect potential biases in content"""
        
        detected_biases = []
        content_lower = content.lower()
        
        for bias_type, rules in self.bias_detection_rules.items():
            biases = await self._detect_bias_type(item_id, content_lower, bias_type, rules, context_frames)
            detected_biases.extend(biases)
        
        # Store detected biases
        self.bias_indicators[item_id] = detected_biases
        
        logger.info(f"Detected {len(detected_biases)} bias indicators for item {item_id}")
        return detected_biases
    
    async def _detect_bias_type(
        self,
        item_id: str,
        content: str,
        bias_type: BiasType,
        rules: Dict,
        context_frames: List[ContextFrame]
    ) -> List[BiasIndicator]:
        """Detect specific type of bias"""
        
        biases = []
        patterns = rules.get('patterns', [])
        indicators = rules.get('indicators', [])
        
        # Check for linguistic patterns
        pattern_matches = []
        for pattern in patterns:
            if pattern.lower() in content:
                pattern_matches.append(pattern)
        
        if pattern_matches:
            # Calculate strength based on number of patterns matched
            strength = min(1.0, len(pattern_matches) / len(patterns))
            
            # Generate description based on bias type
            description = self._generate_bias_description(bias_type, pattern_matches, context_frames)
            
            bias = BiasIndicator(
                bias_id=f"{item_id}_{bias_type.value}_{int(time.time())}",
                bias_type=bias_type,
                strength=strength,
                description=description,
                evidence=pattern_matches,
                mitigation=self._suggest_bias_mitigation(bias_type),
                detected_at=time.time()
            )
            biases.append(bias)
        
        return biases
    
    def _generate_bias_description(self, bias_type: BiasType, patterns: List[str], contexts: List[ContextFrame]) -> str:
        """Generate human-readable bias description"""
        
        descriptions = {
            BiasType.CULTURAL_BIAS: f"Potential cultural bias detected - content may assume specific cultural values. Patterns: {', '.join(patterns[:3])}",
            BiasType.SELECTION_BIAS: f"Possible selection bias - content may cherry-pick examples. Patterns: {', '.join(patterns[:3])}",
            BiasType.CONFIRMATION_BIAS: f"Confirmation bias indicators found - content may seek only supporting evidence. Patterns: {', '.join(patterns[:3])}",
            BiasType.TEMPORAL_BIAS: f"Temporal bias detected - content may not account for historical context. Patterns: {', '.join(patterns[:3])}"
        }
        
        base_description = descriptions.get(bias_type, f"Bias of type {bias_type.value} detected")
        
        # Add context information if available
        if contexts:
            context_info = ', '.join([f"{c.context_type.value}:{c.context_value}" for c in contexts[:2]])
            base_description += f" (Context: {context_info})"
        
        return base_description
    
    def _suggest_bias_mitigation(self, bias_type: BiasType) -> str:
        """Suggest mitigation strategies for detected bias"""
        
        mitigations = {
            BiasType.CULTURAL_BIAS: "Consider multiple cultural perspectives and avoid assuming universal values",
            BiasType.SELECTION_BIAS: "Include contradictory evidence and representative examples",
            BiasType.CONFIRMATION_BIAS: "Actively seek disconfirming evidence and alternative viewpoints",
            BiasType.TEMPORAL_BIAS: "Provide historical context and acknowledge changing perspectives over time",
            BiasType.LINGUISTIC_BIAS: "Consider how language choices may exclude or include different groups",
            BiasType.DEMOGRAPHIC_BIAS: "Ensure representation across different demographic groups"
        }
        
        return mitigations.get(bias_type, "Consider alternative perspectives and examine assumptions")
    
    def get_context_frames(self, item_id: str) -> List[ContextFrame]:
        """Get context frames for item"""
        return self.context_frames.get(item_id, [])
    
    def get_bias_indicators(self, item_id: str) -> List[BiasIndicator]:
        """Get bias indicators for item"""
        return self.bias_indicators.get(item_id, [])
    
    def get_contextual_summary(self, item_id: str) -> Dict:
        """Get summary of contextual analysis for item"""
        
        frames = self.get_context_frames(item_id)
        biases = self.get_bias_indicators(item_id)
        
        # Group frames by type
        frames_by_type = {}
        for frame in frames:
            context_type = frame.context_type.value
            if context_type not in frames_by_type:
                frames_by_type[context_type] = []
            frames_by_type[context_type].append({
                'value': frame.context_value,
                'confidence': frame.confidence
            })
        
        # Group biases by type
        biases_by_type = {}
        for bias in biases:
            bias_type = bias.bias_type.value
            if bias_type not in biases_by_type:
                biases_by_type[bias_type] = []
            biases_by_type[bias_type].append({
                'strength': bias.strength,
                'description': bias.description
            })
        
        return {
            'item_id': item_id,
            'context_frames': frames_by_type,
            'bias_indicators': biases_by_type,
            'total_frames': len(frames),
            'total_biases': len(biases),
            'analysis_timestamp': time.time()
        }
    
    def export_analysis(self, item_id: str) -> Dict:
        """Export complete contextual analysis for item"""
        
        frames = self.get_context_frames(item_id)
        biases = self.get_bias_indicators(item_id)
        
        return {
            'item_id': item_id,
            'context_frames': [frame.to_dict() for frame in frames],
            'bias_indicators': [bias.to_dict() for bias in biases],
            'node_id': self.node_id,
            'exported_at': time.time()
        }