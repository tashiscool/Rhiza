"""
Conflict Resolver - Conflict Resolution Mechanisms

Implements sophisticated conflict resolution for distributed data
synchronization in The Mesh network, with multiple resolution
strategies and consensus-based decision making.

Key Features:
- Multi-strategy conflict resolution
- Vector clock-based conflict detection
- Trust-weighted consensus resolution  
- Semantic conflict analysis
- Automated and manual resolution modes
"""

import asyncio
import time
import json
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from enum import Enum
from collections import defaultdict
import logging
import difflib

from .sync_manager import VectorClock
from .data_chunker import DataChunk, ChunkType, PrivacyLevel
from ..trust.trust_ledger import TrustLedger, TrustScore


class ConflictType(Enum):
    """Types of data conflicts"""
    CONCURRENT_UPDATE = "concurrent_update"    # Multiple simultaneous updates
    ORDERING_CONFLICT = "ordering_conflict"    # Different ordering of operations
    SEMANTIC_CONFLICT = "semantic_conflict"    # Semantic inconsistency
    TRUST_CONFLICT = "trust_conflict"         # Trust-based disagreement
    PRIVACY_CONFLICT = "privacy_conflict"      # Privacy level mismatch
    DEPENDENCY_CONFLICT = "dependency_conflict" # Dependency chain conflict


class ResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    LAST_WRITER_WINS = "last_writer_wins"      # Newest timestamp wins
    TRUST_WEIGHTED = "trust_weighted"          # Trust score-based resolution
    CONSENSUS = "consensus"                    # Majority consensus
    SEMANTIC_MERGE = "semantic_merge"          # Intelligent semantic merging
    MANUAL_RESOLUTION = "manual_resolution"    # Human intervention required
    THREE_WAY_MERGE = "three_way_merge"       # Git-style three-way merge


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = 1      # Minor inconsistencies
    MEDIUM = 2   # Moderate conflicts
    HIGH = 3     # Significant conflicts
    CRITICAL = 4 # Data integrity at risk


@dataclass
class ConflictingData:
    """Represents conflicting versions of data"""
    version_id: str
    data: Any
    vector_clock: VectorClock
    originator_peer: str
    trust_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConflict:
    """Represents a detected data conflict"""
    conflict_id: str
    data_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    conflicting_versions: List[ConflictingData]
    detected_at: float
    resolution_deadline: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_trust_weighted_order(self) -> List[ConflictingData]:
        """Get versions ordered by trust score"""
        return sorted(self.conflicting_versions, 
                     key=lambda v: v.trust_score, 
                     reverse=True)
    
    def get_chronological_order(self) -> List[ConflictingData]:
        """Get versions ordered by timestamp"""
        return sorted(self.conflicting_versions, 
                     key=lambda v: v.timestamp)


@dataclass
class ResolutionResult:
    """Result of conflict resolution"""
    conflict_id: str
    strategy_used: ResolutionStrategy
    resolved_data: Any
    confidence: float  # 0.0 to 1.0
    resolution_metadata: Dict[str, Any]
    requires_verification: bool = False
    manual_review_needed: bool = False
    resolved_at: float = field(default_factory=time.time)


class SemanticAnalyzer:
    """Analyzes semantic content of conflicting data"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        self.semantic_patterns = {}
        
    def analyze_similarity(self, data1: Any, data2: Any) -> float:
        """Calculate semantic similarity between two data items"""
        try:
            # Convert data to comparable strings
            str1 = self._normalize_data(data1)
            str2 = self._normalize_data(data2)
            
            # Use sequence matching for basic similarity
            matcher = difflib.SequenceMatcher(None, str1, str2)
            return matcher.ratio()
            
        except Exception:
            return 0.0
            
    def detect_semantic_conflicts(self, versions: List[ConflictingData]) -> List[Tuple[int, int, str]]:
        """Detect semantic conflicts between versions"""
        conflicts = []
        
        for i, version1 in enumerate(versions):
            for j, version2 in enumerate(versions[i+1:], i+1):
                similarity = self.analyze_similarity(version1.data, version2.data)
                
                if similarity < self.similarity_threshold:
                    conflict_description = self._describe_semantic_conflict(
                        version1.data, version2.data
                    )
                    conflicts.append((i, j, conflict_description))
                    
        return conflicts
        
    def suggest_merge(self, versions: List[ConflictingData]) -> Optional[Any]:
        """Suggest a semantic merge of conflicting versions"""
        if len(versions) < 2:
            return versions[0].data if versions else None
            
        # For now, implement simple merge logic
        # In production, this would be much more sophisticated
        base_data = versions[0].data
        
        for version in versions[1:]:
            base_data = self._merge_data_semantically(base_data, version.data)
            
        return base_data
        
    def _normalize_data(self, data: Any) -> str:
        """Normalize data for comparison"""
        if isinstance(data, str):
            return data.lower().strip()
        elif isinstance(data, dict):
            return json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            return str(sorted(str(item) for item in data))
        else:
            return str(data)
            
    def _describe_semantic_conflict(self, data1: Any, data2: Any) -> str:
        """Describe the nature of semantic conflict"""
        type1, type2 = type(data1).__name__, type(data2).__name__
        
        if type1 != type2:
            return f"Type mismatch: {type1} vs {type2}"
        elif isinstance(data1, dict) and isinstance(data2, dict):
            diff_keys = set(data1.keys()) ^ set(data2.keys())
            if diff_keys:
                return f"Different keys: {diff_keys}"
            else:
                return "Value differences in dictionary"
        else:
            return "Content differences"
            
    def _merge_data_semantically(self, data1: Any, data2: Any) -> Any:
        """Perform semantic merge of two data items"""
        # Simple merge strategy - in production would be much more sophisticated
        if isinstance(data1, dict) and isinstance(data2, dict):
            merged = data1.copy()
            for key, value in data2.items():
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    # Conflict in values - keep both with suffixes
                    merged[f"{key}_v1"] = merged[key]
                    merged[f"{key}_v2"] = value
                    del merged[key]
            return merged
        elif isinstance(data1, list) and isinstance(data2, list):
            # Merge lists by combining unique elements
            return list(set(data1) | set(data2))
        else:
            # For other types, return the newer one
            return data2


class TrustBasedResolver:
    """Resolves conflicts using trust-based mechanisms"""
    
    def __init__(self, trust_ledger: TrustLedger):
        self.trust_ledger = trust_ledger
        self.trust_threshold = 0.6
        
    def resolve_by_trust(self, conflict: DataConflict) -> ResolutionResult:
        """Resolve conflict based on trust scores"""
        # Get trust-weighted order
        ordered_versions = conflict.get_trust_weighted_order()
        
        if not ordered_versions:
            raise ValueError("No versions available for resolution")
            
        # Select highest trust version if above threshold
        best_version = ordered_versions[0]
        
        if best_version.trust_score >= self.trust_threshold:
            confidence = best_version.trust_score
            requires_verification = confidence < 0.8
        else:
            # No sufficiently trusted version - require manual review
            confidence = 0.5
            requires_verification = True
            
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.TRUST_WEIGHTED,
            resolved_data=best_version.data,
            confidence=confidence,
            resolution_metadata={
                'selected_version': best_version.version_id,
                'trust_score': best_version.trust_score,
                'originator': best_version.originator_peer
            },
            requires_verification=requires_verification,
            manual_review_needed=best_version.trust_score < self.trust_threshold
        )
        
    def resolve_by_consensus(self, conflict: DataConflict, peer_votes: Dict[str, str]) -> ResolutionResult:
        """Resolve conflict using trust-weighted consensus"""
        vote_weights = defaultdict(float)
        
        # Weight votes by trust scores
        for peer_id, version_id in peer_votes.items():
            trust_score = self.trust_ledger.get_trust_score(peer_id)
            weight = trust_score.composite_score if trust_score else 0.5
            vote_weights[version_id] += weight
            
        if not vote_weights:
            raise ValueError("No votes received for consensus resolution")
            
        # Find version with highest weighted votes
        winning_version_id = max(vote_weights.keys(), key=lambda v: vote_weights[v])
        winning_version = next(v for v in conflict.conflicting_versions 
                             if v.version_id == winning_version_id)
        
        total_weight = sum(vote_weights.values())
        confidence = vote_weights[winning_version_id] / total_weight
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.CONSENSUS,
            resolved_data=winning_version.data,
            confidence=confidence,
            resolution_metadata={
                'vote_weights': dict(vote_weights),
                'total_votes': len(peer_votes),
                'winning_version': winning_version_id
            },
            requires_verification=confidence < 0.7
        )


class ConflictResolver:
    """Main conflict resolution system"""
    
    def __init__(self, trust_ledger: TrustLedger, local_peer_id: str):
        self.trust_ledger = trust_ledger
        self.local_peer_id = local_peer_id
        self.semantic_analyzer = SemanticAnalyzer()
        self.trust_resolver = TrustBasedResolver(trust_ledger)
        
        # Active conflicts and resolutions
        self.active_conflicts: Dict[str, DataConflict] = {}
        self.resolution_history: List[ResolutionResult] = []
        self.manual_queue: List[str] = []  # Conflicts requiring manual resolution
        
        # Configuration
        self.default_strategy = ResolutionStrategy.TRUST_WEIGHTED
        self.auto_resolve_threshold = 0.8  # Auto-resolve if confidence >= this
        self.max_resolution_time = 300  # 5 minutes
        
        # Callbacks
        self.on_conflict_detected: List[Callable] = []
        self.on_conflict_resolved: List[Callable] = []
        self.on_manual_resolution_needed: List[Callable] = []
        
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start the conflict resolution system"""
        self.logger.info("Starting conflict resolution system")
        
        # Start background tasks
        asyncio.create_task(self._resolution_processor())
        asyncio.create_task(self._conflict_monitor())
        
    async def detect_conflict(self, 
                       data_id: str,
                       versions: List[ConflictingData],
                       context: Optional[Dict[str, Any]] = None) -> Optional[DataConflict]:
        """Detect and classify data conflicts"""
        
        if len(versions) < 2:
            return None  # No conflict with less than 2 versions
            
        # Determine conflict type and severity
        conflict_type = self._classify_conflict(versions)
        severity = self._assess_severity(versions, conflict_type)
        
        # Create conflict record
        conflict = DataConflict(
            conflict_id=secrets.token_hex(16),
            data_id=data_id,
            conflict_type=conflict_type,
            severity=severity,
            conflicting_versions=versions,
            detected_at=time.time(),
            resolution_deadline=time.time() + self.max_resolution_time,
            context=context or {}
        )
        
        # Store active conflict
        self.active_conflicts[conflict.conflict_id] = conflict
        
        # Notify callbacks
        for callback in self.on_conflict_detected:
            try:
                await callback(conflict)
            except Exception as e:
                self.logger.error(f"Conflict detection callback error: {e}")
                
        self.logger.warning(f"Detected {conflict_type.value} conflict for data {data_id}")
        return conflict
        
    async def resolve_conflict(self, 
                             conflict_id: str,
                             strategy: Optional[ResolutionStrategy] = None,
                             additional_context: Optional[Dict[str, Any]] = None) -> Optional[ResolutionResult]:
        """Resolve a specific conflict"""
        
        conflict = self.active_conflicts.get(conflict_id)
        if not conflict:
            return None
            
        resolution_strategy = strategy or self._select_resolution_strategy(conflict)
        
        try:
            if resolution_strategy == ResolutionStrategy.LAST_WRITER_WINS:
                result = self._resolve_last_writer_wins(conflict)
            elif resolution_strategy == ResolutionStrategy.TRUST_WEIGHTED:
                result = self.trust_resolver.resolve_by_trust(conflict)
            elif resolution_strategy == ResolutionStrategy.SEMANTIC_MERGE:
                result = await self._resolve_semantic_merge(conflict)
            elif resolution_strategy == ResolutionStrategy.THREE_WAY_MERGE:
                result = await self._resolve_three_way_merge(conflict)
            elif resolution_strategy == ResolutionStrategy.MANUAL_RESOLUTION:
                result = self._queue_manual_resolution(conflict)
            else:
                raise ValueError(f"Unsupported resolution strategy: {resolution_strategy}")
                
            # Store resolution result
            if result:
                self.resolution_history.append(result)
                
                # Remove from active conflicts if resolved
                if not result.manual_review_needed:
                    del self.active_conflicts[conflict_id]
                    
                # Notify callbacks
                for callback in self.on_conflict_resolved:
                    try:
                        await callback(result)
                    except Exception as e:
                        self.logger.error(f"Conflict resolution callback error: {e}")
                        
                self.logger.info(f"Resolved conflict {conflict_id} using {resolution_strategy.value}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return None
            
    def _classify_conflict(self, versions: List[ConflictingData]) -> ConflictType:
        """Classify the type of conflict"""
        
        # Check for concurrent updates (overlapping timestamps)
        timestamps = [v.timestamp for v in versions]
        if max(timestamps) - min(timestamps) < 10:  # Within 10 seconds
            return ConflictType.CONCURRENT_UPDATE
            
        # Check for trust conflicts (high trust disagreement)
        trust_scores = [v.trust_score for v in versions]
        if all(score > 0.8 for score in trust_scores):
            return ConflictType.TRUST_CONFLICT
            
        # Check for semantic conflicts
        semantic_conflicts = self.semantic_analyzer.detect_semantic_conflicts(versions)
        if semantic_conflicts:
            return ConflictType.SEMANTIC_CONFLICT
            
        # Default to concurrent update
        return ConflictType.CONCURRENT_UPDATE
        
    def _assess_severity(self, versions: List[ConflictingData], conflict_type: ConflictType) -> ConflictSeverity:
        """Assess the severity of a conflict"""
        
        # High severity for trust conflicts
        if conflict_type == ConflictType.TRUST_CONFLICT:
            return ConflictSeverity.HIGH
            
        # Check data similarity for semantic conflicts
        if conflict_type == ConflictType.SEMANTIC_CONFLICT:
            similarities = []
            for i, v1 in enumerate(versions):
                for v2 in versions[i+1:]:
                    sim = self.semantic_analyzer.analyze_similarity(v1.data, v2.data)
                    similarities.append(sim)
                    
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            
            if avg_similarity < 0.3:
                return ConflictSeverity.CRITICAL
            elif avg_similarity < 0.6:
                return ConflictSeverity.HIGH
            else:
                return ConflictSeverity.MEDIUM
                
        # Default severity
        return ConflictSeverity.MEDIUM
        
    def _select_resolution_strategy(self, conflict: DataConflict) -> ResolutionStrategy:
        """Select appropriate resolution strategy for a conflict"""
        
        if conflict.severity == ConflictSeverity.CRITICAL:
            return ResolutionStrategy.MANUAL_RESOLUTION
            
        if conflict.conflict_type == ConflictType.TRUST_CONFLICT:
            return ResolutionStrategy.CONSENSUS
            
        if conflict.conflict_type == ConflictType.SEMANTIC_CONFLICT:
            return ResolutionStrategy.SEMANTIC_MERGE
            
        # Default strategy
        return self.default_strategy
        
    def _resolve_last_writer_wins(self, conflict: DataConflict) -> ResolutionResult:
        """Resolve using last-writer-wins strategy"""
        latest_version = max(conflict.conflicting_versions, key=lambda v: v.timestamp)
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.LAST_WRITER_WINS,
            resolved_data=latest_version.data,
            confidence=0.7,  # Moderate confidence
            resolution_metadata={
                'selected_version': latest_version.version_id,
                'timestamp': latest_version.timestamp,
                'originator': latest_version.originator_peer
            },
            requires_verification=True
        )
        
    async def _resolve_semantic_merge(self, conflict: DataConflict) -> ResolutionResult:
        """Resolve using semantic merge strategy"""
        merged_data = self.semantic_analyzer.suggest_merge(conflict.conflicting_versions)
        
        if merged_data is None:
            # Fallback to trust-based resolution
            return self.trust_resolver.resolve_by_trust(conflict)
            
        # Calculate confidence based on semantic similarity
        similarities = []
        for version in conflict.conflicting_versions:
            sim = self.semantic_analyzer.analyze_similarity(merged_data, version.data)
            similarities.append(sim)
            
        avg_similarity = sum(similarities) / len(similarities)
        confidence = min(avg_similarity + 0.2, 1.0)  # Bonus for successful merge
        
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.SEMANTIC_MERGE,
            resolved_data=merged_data,
            confidence=confidence,
            resolution_metadata={
                'merge_similarity': avg_similarity,
                'versions_merged': len(conflict.conflicting_versions)
            },
            requires_verification=confidence < 0.8
        )
        
    async def _resolve_three_way_merge(self, conflict: DataConflict) -> ResolutionResult:
        """Resolve using three-way merge strategy (Git-style)"""
        if len(conflict.conflicting_versions) != 2:
            # Fallback for non-binary conflicts
            return await self._resolve_semantic_merge(conflict)
            
        version1, version2 = conflict.conflicting_versions
        
        # This would implement actual three-way merge logic
        # For now, use semantic merge as approximation
        return await self._resolve_semantic_merge(conflict)
        
    def _queue_manual_resolution(self, conflict: DataConflict) -> ResolutionResult:
        """Queue conflict for manual resolution"""
        if conflict.conflict_id not in self.manual_queue:
            self.manual_queue.append(conflict.conflict_id)
            
        # Notify callbacks
        for callback in self.on_manual_resolution_needed:
            try:
                asyncio.create_task(callback(conflict))
            except Exception as e:
                self.logger.error(f"Manual resolution callback error: {e}")
                
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            strategy_used=ResolutionStrategy.MANUAL_RESOLUTION,
            resolved_data=None,
            confidence=0.0,
            resolution_metadata={'queued_at': time.time()},
            manual_review_needed=True
        )
        
    async def submit_manual_resolution(self, 
                                     conflict_id: str, 
                                     resolved_data: Any,
                                     resolution_notes: str = "") -> bool:
        """Submit manual resolution for a conflict"""
        
        conflict = self.active_conflicts.get(conflict_id)
        if not conflict:
            return False
            
        # Create resolution result
        result = ResolutionResult(
            conflict_id=conflict_id,
            strategy_used=ResolutionStrategy.MANUAL_RESOLUTION,
            resolved_data=resolved_data,
            confidence=1.0,  # Manual resolution is fully confident
            resolution_metadata={
                'resolution_notes': resolution_notes,
                'resolved_by': self.local_peer_id
            },
            requires_verification=False,
            manual_review_needed=False
        )
        
        # Store and cleanup
        self.resolution_history.append(result)
        del self.active_conflicts[conflict_id]
        
        if conflict_id in self.manual_queue:
            self.manual_queue.remove(conflict_id)
            
        # Notify callbacks
        for callback in self.on_conflict_resolved:
            try:
                await callback(result)
            except Exception as e:
                self.logger.error(f"Manual resolution callback error: {e}")
                
        self.logger.info(f"Manual resolution submitted for conflict {conflict_id}")
        return True
        
    async def _resolution_processor(self):
        """Background processor for automatic conflict resolution"""
        while True:
            try:
                # Process conflicts that can be auto-resolved
                for conflict_id, conflict in list(self.active_conflicts.items()):
                    if (conflict.severity != ConflictSeverity.CRITICAL and 
                        conflict.conflict_id not in self.manual_queue):
                        
                        # Attempt automatic resolution
                        result = await self.resolve_conflict(conflict_id)
                        
                        if (result and 
                            result.confidence >= self.auto_resolve_threshold and 
                            not result.manual_review_needed):
                            
                            self.logger.info(f"Auto-resolved conflict {conflict_id}")
                            
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resolution processor: {e}")
                await asyncio.sleep(5)
                
    async def _conflict_monitor(self):
        """Monitor conflicts for deadlines and escalation"""
        while True:
            try:
                current_time = time.time()
                
                # Check for expired conflicts
                for conflict_id, conflict in list(self.active_conflicts.items()):
                    if (conflict.resolution_deadline and 
                        current_time > conflict.resolution_deadline):
                        
                        self.logger.warning(f"Conflict {conflict_id} exceeded deadline")
                        
                        # Escalate to manual resolution
                        if conflict_id not in self.manual_queue:
                            self._queue_manual_resolution(conflict)
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in conflict monitor: {e}")
                await asyncio.sleep(30)
                
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """Get comprehensive conflict resolution statistics"""
        total_conflicts = len(self.resolution_history)
        if not total_conflicts:
            return {'total_conflicts': 0}
            
        # Calculate statistics
        by_strategy = defaultdict(int)
        by_severity = defaultdict(int) 
        by_type = defaultdict(int)
        confidence_sum = 0
        
        for result in self.resolution_history:
            by_strategy[result.strategy_used.value] += 1
            confidence_sum += result.confidence
            
        for conflict in self.active_conflicts.values():
            by_severity[conflict.severity.name] += 1
            by_type[conflict.conflict_type.value] += 1
            
        return {
            'total_conflicts': total_conflicts,
            'active_conflicts': len(self.active_conflicts),
            'manual_queue_size': len(self.manual_queue),
            'resolution_strategies': dict(by_strategy),
            'conflict_severity': dict(by_severity),
            'conflict_types': dict(by_type),
            'avg_confidence': confidence_sum / total_conflicts,
            'auto_resolution_rate': sum(1 for r in self.resolution_history 
                                      if not r.manual_review_needed) / total_conflicts
        }