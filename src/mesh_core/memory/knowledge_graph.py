"""
Knowledge Graph - Enhanced with Sentient's Memory Knowledge Organization Concepts

Integrates Sentient's proven memory knowledge patterns:
- Structured information organization and relationships
- Topic-based knowledge categorization
- Memory type classification and management
- Intelligent knowledge retrieval and navigation

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/memories')
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
class KnowledgeNode:
    """A node in the knowledge graph representing a fact or concept"""
    id: str
    content: str
    topics: List[str]
    memory_type: str  # "long-term" or "short-term"
    duration: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # IDs of related nodes

@dataclass
class KnowledgeRelationship:
    """A relationship between two knowledge nodes"""
    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # "similar", "related", "contradicts", "supports", etc.
    strength: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class KnowledgeGraphResult:
    """Result of knowledge graph operations"""
    nodes: List[KnowledgeNode]
    relationships: List[KnowledgeRelationship]
    total_nodes: int
    total_relationships: int
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph"""
    enable_auto_relationships: bool = True
    enable_topic_clustering: bool = True
    enable_memory_management: bool = True
    enable_relationship_scoring: bool = True
    max_nodes_per_topic: int = 1000
    max_relationships_per_node: int = 50
    relationship_strength_threshold: float = 0.3
    enable_debug_logging: bool = False

class KnowledgeGraph:
    """
    Enhanced knowledge graph integrating Sentient's memory knowledge concepts
    
    Provides:
    - Structured information organization and relationships
    - Topic-based knowledge categorization
    - Memory type classification and management
    - Intelligent knowledge retrieval and navigation
    """
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Graph storage
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relationships: Dict[str, KnowledgeRelationship] = {}
        
        # Indexes for efficient retrieval
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)  # topic -> node_ids
        self.memory_type_index: Dict[str, Set[str]] = defaultdict(set)  # memory_type -> node_ids
        self.content_index: Dict[str, Set[str]] = defaultdict(set)  # word -> node_ids
        
        # Initialize graph components
        self.relationship_builder = None
        self.topic_clusterer = None
        self.memory_manager = None
        
        # Performance tracking
        self.operation_count = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_graph_components()
        
        self.logger.info("Knowledge Graph initialized with Sentient concepts")
    
    def _initialize_graph_components(self):
        """Initialize knowledge graph components using Sentient patterns"""
        
        try:
            # Initialize relationship builder
            if self.config.enable_auto_relationships:
                self._initialize_relationship_builder()
            
            # Initialize topic clusterer
            if self.config.enable_topic_clustering:
                self._initialize_topic_clusterer()
            
            # Initialize memory manager
            if self.config.enable_memory_management:
                self._initialize_memory_manager()
            
            self.logger.info("Knowledge graph components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize graph components: {e}")
            self.logger.warning("Knowledge graph will use basic methods")
    
    def _initialize_relationship_builder(self):
        """Initialize relationship builder component"""
        
        try:
            self.relationship_builder = RelationshipBuilder(self.config)
            self.logger.info("Relationship builder initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize relationship builder: {e}")
            self.relationship_builder = None
    
    def _initialize_topic_clusterer(self):
        """Initialize topic clustering component"""
        
        try:
            self.topic_clusterer = TopicClusterer(TOPICS)
            self.logger.info("Topic clusterer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize topic clusterer: {e}")
            self.topic_clusterer = None
    
    def _initialize_memory_manager(self):
        """Initialize memory management component"""
        
        try:
            self.memory_manager = MemoryManager()
            self.logger.info("Memory manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}")
            self.memory_manager = None
    
    async def add_node(self, node: KnowledgeNode) -> bool:
        """
        Add a knowledge node to the graph
        
        Args:
            node: Knowledge node to add
            
        Returns:
            True if node was added successfully, False otherwise
        """
        
        try:
            # Validate node
            if not self._validate_node(node):
                self.logger.warning(f"Invalid node rejected: {node.id}")
                return False
            
            # Check if node already exists
            if node.id in self.nodes:
                self.logger.warning(f"Node already exists: {node.id}")
                return False
            
            # Add node to storage
            self.nodes[node.id] = node
            
            # Update indexes
            self._update_node_indexes(node)
            
            # Build automatic relationships
            if self.config.enable_auto_relationships and self.relationship_builder:
                await self._build_auto_relationships(node)
            
            # Update topic clustering
            if self.config.enable_topic_clustering and self.topic_clusterer:
                await self._update_topic_clustering(node)
            
            # Update memory management
            if self.config.enable_memory_management and self.memory_manager:
                await self._update_memory_management(node)
            
            self.logger.info(f"Node added successfully: {node.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add node: {e}")
            return False
    
    def _validate_node(self, node: KnowledgeNode) -> bool:
        """Validate a knowledge node"""
        
        try:
            # Check required fields
            if not node.id or not node.content:
                return False
            
            # Check topics
            if not node.topics or not isinstance(node.topics, list):
                return False
            
            # Check memory type
            if node.memory_type not in ["long-term", "short-term"]:
                return False
            
            # Check confidence
            if not 0.0 <= node.confidence <= 1.0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Node validation failed: {e}")
            return False
    
    def _update_node_indexes(self, node: KnowledgeNode):
        """Update all indexes for a node"""
        
        try:
            # Topic index
            for topic in node.topics:
                self.topic_index[topic].add(node.id)
            
            # Memory type index
            self.memory_type_index[node.memory_type].add(node.id)
            
            # Content index (simple word-based indexing)
            words = set(node.content.lower().split())
            for word in words:
                if len(word) > 2:  # Only index words longer than 2 characters
                    self.content_index[word].add(node.id)
                    
        except Exception as e:
            self.logger.error(f"Index update failed: {e}")
    
    async def _build_auto_relationships(self, node: KnowledgeNode):
        """Build automatic relationships for a new node"""
        
        try:
            if not self.relationship_builder:
                return
            
            # Find potential relationships
            relationships = await self.relationship_builder.find_relationships(node, self.nodes)
            
            # Add relationships
            for relationship in relationships:
                await self.add_relationship(relationship)
                
        except Exception as e:
            self.logger.error(f"Auto-relationship building failed: {e}")
    
    async def _update_topic_clustering(self, node: KnowledgeNode):
        """Update topic clustering for a new node"""
        
        try:
            if not self.topic_clusterer:
                return
            
            # Update clustering
            await self.topic_clusterer.update_clustering(node, self.nodes)
            
        except Exception as e:
            self.logger.error(f"Topic clustering update failed: {e}")
    
    async def _update_memory_management(self, node: KnowledgeNode):
        """Update memory management for a new node"""
        
        try:
            if not self.memory_manager:
                return
            
            # Update memory management
            await self.memory_manager.update_memory(node, self.nodes)
            
        except Exception as e:
            self.logger.error(f"Memory management update failed: {e}")
    
    async def add_relationship(self, relationship: KnowledgeRelationship) -> bool:
        """
        Add a relationship between two nodes
        
        Args:
            relationship: Relationship to add
            
        Returns:
            True if relationship was added successfully, False otherwise
        """
        
        try:
            # Validate relationship
            if not self._validate_relationship(relationship):
                self.logger.warning(f"Invalid relationship rejected: {relationship.id}")
                return False
            
            # Check if relationship already exists
            if relationship.id in self.relationships:
                self.logger.warning(f"Relationship already exists: {relationship.id}")
                return False
            
            # Check if both nodes exist
            if (relationship.source_node_id not in self.nodes or 
                relationship.target_node_id not in self.nodes):
                self.logger.warning(f"Relationship references non-existent nodes: {relationship.id}")
                return False
            
            # Add relationship
            self.relationships[relationship.id] = relationship
            
            # Update node relationships
            source_node = self.nodes[relationship.source_node_id]
            target_node = self.nodes[relationship.target_node_id]
            
            source_node.relationships.append(relationship.id)
            target_node.relationships.append(relationship.id)
            
            self.logger.info(f"Relationship added successfully: {relationship.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add relationship: {e}")
            return False
    
    def _validate_relationship(self, relationship: KnowledgeRelationship) -> bool:
        """Validate a knowledge relationship"""
        
        try:
            # Check required fields
            if not relationship.id or not relationship.source_node_id or not relationship.target_node_id:
                return False
            
            # Check relationship type
            valid_types = ["similar", "related", "contradicts", "supports", "depends_on", "part_of"]
            if relationship.relationship_type not in valid_types:
                return False
            
            # Check strength
            if not 0.0 <= relationship.strength <= 1.0:
                return False
            
            # Check self-relationship
            if relationship.source_node_id == relationship.target_node_id:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Relationship validation failed: {e}")
            return False
    
    async def get_nodes_by_topic(self, topic: str, limit: Optional[int] = None) -> List[KnowledgeNode]:
        """Get nodes by topic"""
        
        try:
            node_ids = self.topic_index.get(topic, set())
            nodes = [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
            
            # Sort by timestamp (newest first)
            nodes.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit is not None:
                nodes = nodes[:limit]
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get nodes by topic: {e}")
            return []
    
    async def get_nodes_by_memory_type(self, memory_type: str, limit: Optional[int] = None) -> List[KnowledgeNode]:
        """Get nodes by memory type"""
        
        try:
            node_ids = self.memory_type_index.get(memory_type, set())
            nodes = [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
            
            # Sort by timestamp (newest first)
            nodes.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit is not None:
                nodes = nodes[:limit]
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get nodes by memory type: {e}")
            return []
    
    async def search_nodes(self, query: str, limit: Optional[int] = None) -> List[KnowledgeNode]:
        """Search nodes by content"""
        
        try:
            query_words = set(query.lower().split())
            matching_nodes = set()
            
            # Find nodes containing query words
            for word in query_words:
                if word in self.content_index:
                    matching_nodes.update(self.content_index[word])
            
            # Get matching nodes
            nodes = [self.nodes[node_id] for node_id in matching_nodes if node_id in self.nodes]
            
            # Sort by relevance (simple word count for now)
            def relevance_score(node):
                node_words = set(node.content.lower().split())
                return len(query_words.intersection(node_words))
            
            nodes.sort(key=relevance_score, reverse=True)
            
            # Apply limit
            if limit is not None:
                nodes = nodes[:limit]
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Node search failed: {e}")
            return []
    
    async def get_related_nodes(self, node_id: str, limit: Optional[int] = None) -> List[KnowledgeNode]:
        """Get nodes related to a specific node"""
        
        try:
            if node_id not in self.nodes:
                return []
            
            node = self.nodes[node_id]
            related_nodes = []
            
            # Get nodes through relationships
            for relationship_id in node.relationships:
                if relationship_id in self.relationships:
                    relationship = self.relationships[relationship_id]
                    
                    # Get the other node in the relationship
                    other_node_id = (relationship.target_node_id 
                                   if relationship.source_node_id == node_id 
                                   else relationship.source_node_id)
                    
                    if other_node_id in self.nodes:
                        related_nodes.append(self.nodes[other_node_id])
            
            # Sort by relationship strength
            related_nodes.sort(key=lambda x: self._get_relationship_strength(node_id, x.id), reverse=True)
            
            # Apply limit
            if limit is not None:
                related_nodes = related_nodes[:limit]
            
            return related_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get related nodes: {e}")
            return []
    
    def _get_relationship_strength(self, node1_id: str, node2_id: str) -> float:
        """Get the strength of relationship between two nodes"""
        
        try:
            for relationship in self.relationships.values():
                if ((relationship.source_node_id == node1_id and relationship.target_node_id == node2_id) or
                    (relationship.source_node_id == node2_id and relationship.target_node_id == node1_id)):
                    return relationship.strength
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get relationship strength: {e}")
            return 0.0
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        try:
            # Basic counts
            total_nodes = len(self.nodes)
            total_relationships = len(self.relationships)
            
            # Topic distribution
            topic_distribution = {}
            for topic, node_ids in self.topic_index.items():
                topic_distribution[topic] = len(node_ids)
            
            # Memory type distribution
            memory_type_distribution = {}
            for memory_type, node_ids in self.memory_type_index.items():
                memory_type_distribution[memory_type] = len(node_ids)
            
            # Relationship type distribution
            relationship_type_distribution = defaultdict(int)
            for relationship in self.relationships.values():
                relationship_type_distribution[relationship.relationship_type] += 1
            
            # Average relationships per node
            avg_relationships_per_node = (total_relationships / total_nodes 
                                       if total_nodes > 0 else 0.0)
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "topic_distribution": dict(topic_distribution),
                "memory_type_distribution": dict(memory_type_distribution),
                "relationship_type_distribution": dict(relationship_type_distribution),
                "average_relationships_per_node": avg_relationships_per_node,
                "index_sizes": {
                    "topic_index": len(self.topic_index),
                    "memory_type_index": len(self.memory_type_index),
                    "content_index": len(self.content_index)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    async def cleanup_expired_nodes(self) -> int:
        """Clean up expired short-term memory nodes"""
        
        try:
            if not self.memory_manager:
                return 0
            
            expired_nodes = await self.memory_manager.get_expired_nodes(self.nodes)
            cleaned_count = 0
            
            for node_id in expired_nodes:
                if await self.remove_node(node_id):
                    cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} expired nodes")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired nodes: {e}")
            return 0
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its relationships"""
        
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # Remove relationships
            relationships_to_remove = []
            for relationship_id in node.relationships:
                if relationship_id in self.relationships:
                    relationships_to_remove.append(relationship_id)
            
            for relationship_id in relationships_to_remove:
                await self.remove_relationship(relationship_id)
            
            # Remove from indexes
            self._remove_node_from_indexes(node)
            
            # Remove node
            del self.nodes[node_id]
            
            self.logger.info(f"Node removed successfully: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove node: {e}")
            return False
    
    def _remove_node_from_indexes(self, node: KnowledgeNode):
        """Remove a node from all indexes"""
        
        try:
            # Topic index
            for topic in node.topics:
                if topic in self.topic_index:
                    self.topic_index[topic].discard(node.id)
            
            # Memory type index
            if node.memory_type in self.memory_type_index:
                self.memory_type_index[node.memory_type].discard(node.id)
            
            # Content index
            words = set(node.content.lower().split())
            for word in words:
                if word in self.content_index:
                    self.content_index[word].discard(node.id)
                    
        except Exception as e:
            self.logger.error(f"Failed to remove node from indexes: {e}")
    
    async def remove_relationship(self, relationship_id: str) -> bool:
        """Remove a relationship"""
        
        try:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            
            # Remove from node relationships
            if relationship.source_node_id in self.nodes:
                source_node = self.nodes[relationship.source_node_id]
                if relationship_id in source_node.relationships:
                    source_node.relationships.remove(relationship_id)
            
            if relationship.target_node_id in self.nodes:
                target_node = self.nodes[relationship.target_node_id]
                if relationship_id in target_node.relationships:
                    target_node.relationships.remove(relationship_id)
            
            # Remove relationship
            del self.relationships[relationship_id]
            
            self.logger.info(f"Relationship removed successfully: {relationship_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove relationship: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for knowledge graph"""
        
        return {
            "operation_count": self.operation_count,
            "total_processing_time": self.total_processing_time,
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "relationship_builder_available": self.relationship_builder is not None,
            "topic_clusterer_available": self.topic_clusterer is not None,
            "memory_manager_available": self.memory_manager is not None
        }
    
    async def cleanup(self):
        """Clean up knowledge graph resources"""
        
        try:
            if hasattr(self.relationship_builder, 'cleanup'):
                await self.relationship_builder.cleanup()
            
            if hasattr(self.topic_clusterer, 'cleanup'):
                await self.topic_clusterer.cleanup()
            
            if hasattr(self.memory_manager, 'cleanup'):
                await self.memory_manager.cleanup()
                
            self.logger.info("Knowledge graph cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class RelationshipBuilder:
    """Relationship builder component following Sentient patterns"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def find_relationships(self, node: KnowledgeNode, 
                               all_nodes: Dict[str, KnowledgeNode]) -> List[KnowledgeRelationship]:
        """Find potential relationships for a new node"""
        
        try:
            relationships = []
            
            for existing_node in all_nodes.values():
                if existing_node.id == node.id:
                    continue
                
                # Calculate similarity
                similarity = await self._calculate_similarity(node, existing_node)
                
                if similarity >= self.config.relationship_strength_threshold:
                    relationship = KnowledgeRelationship(
                        id=f"rel_{node.id}_{existing_node.id}",
                        source_node_id=node.id,
                        target_node_id=existing_node.id,
                        relationship_type="similar",
                        strength=similarity,
                        metadata={"auto_generated": True}
                    )
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship finding failed: {e}")
            return []
    
    async def _calculate_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Calculate similarity between two nodes"""
        
        try:
            # Topic similarity
            topic_overlap = len(set(node1.topics).intersection(set(node2.topics)))
            topic_similarity = topic_overlap / max(len(set(node1.topics).union(set(node2.topics))), 1)
            
            # Content similarity (simple word overlap)
            words1 = set(node1.content.lower().split())
            words2 = set(node2.content.lower().split())
            
            if not words1 or not words2:
                content_similarity = 0.0
            else:
                content_overlap = len(words1.intersection(words2))
                content_similarity = content_overlap / len(words1.union(words2))
            
            # Weighted similarity
            similarity = (topic_similarity * 0.6) + (content_similarity * 0.4)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def cleanup(self):
        """Clean up relationship builder resources"""
        pass


class TopicClusterer:
    """Topic clustering component following Sentient patterns"""
    
    def __init__(self, topics: List[Dict[str, str]]):
        self.topics = topics
        self.logger = logging.getLogger(__name__)
    
    async def update_clustering(self, node: KnowledgeNode, 
                               all_nodes: Dict[str, KnowledgeNode]):
        """Update topic clustering for a new node"""
        
        try:
            # For now, we'll use simple topic-based clustering
            # In the future, we can implement more sophisticated clustering algorithms
            
            self.logger.debug(f"Updated topic clustering for node: {node.id}")
            
        except Exception as e:
            self.logger.error(f"Topic clustering update failed: {e}")
    
    async def cleanup(self):
        """Clean up topic clusterer resources"""
        pass


class MemoryManager:
    """Memory management component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def update_memory(self, node: KnowledgeNode, 
                           all_nodes: Dict[str, KnowledgeNode]):
        """Update memory management for a new node"""
        
        try:
            # For now, we'll use simple memory management
            # In the future, we can implement more sophisticated memory management
            
            self.logger.debug(f"Updated memory management for node: {node.id}")
            
        except Exception as e:
            self.logger.error(f"Memory management update failed: {e}")
    
    async def get_expired_nodes(self, all_nodes: Dict[str, KnowledgeNode]) -> List[str]:
        """Get list of expired short-term memory nodes"""
        
        try:
            expired_nodes = []
            current_time = time.time()
            
            for node_id, node in all_nodes.items():
                if node.memory_type == "short-term" and node.duration:
                    # Simple expiration check (in a real implementation, we'd parse duration)
                    # For now, we'll use a simple time-based approach
                    if current_time - node.timestamp > 86400:  # 24 hours
                        expired_nodes.append(node_id)
            
            return expired_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get expired nodes: {e}")
            return []
    
    async def cleanup(self):
        """Clean up memory manager resources"""
        pass


# Factory function for easy integration
def create_knowledge_graph(config: Optional[KnowledgeGraphConfig] = None) -> KnowledgeGraph:
    """Create a knowledge graph with default or custom configuration"""
    
    if config is None:
        config = KnowledgeGraphConfig()
    
    return KnowledgeGraph(config)
