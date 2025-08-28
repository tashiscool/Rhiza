"""
Mesh Scenario Sharer
====================

Enables sharing scenarios between users and communities,
facilitating collaborative learning and scenario development.
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


class SharingPermission(Enum):
    """Permission levels for scenario sharing"""
    PRIVATE = "private"                  # Only visible to creator
    PERSONAL = "personal"                # Visible to creator and selected individuals
    COMMUNITY = "community"              # Visible to specific communities
    PUBLIC = "public"                    # Visible to all users
    OPEN_SOURCE = "open_source"          # Public with contribution rights


class SharingStatus(Enum):
    """Status of a shared scenario"""
    DRAFT = "draft"                      # Work in progress
    PUBLISHED = "published"              # Available for use
    ARCHIVED = "archived"                # No longer active
    MODERATED = "moderated"              # Under review
    FLAGGED = "flagged"                  # Reported for review


class ContributionType(Enum):
    """Types of contributions to shared scenarios"""
    SCENARIO_CREATION = "scenario_creation"      # Original scenario
    SCENARIO_MODIFICATION = "scenario_modification"  # Changes to existing scenario
    PERSONA_ADDITION = "persona_addition"        # New personas
    FEEDBACK = "feedback"                        # User feedback and ratings
    USE_CASE = "use_case"                        # Example usage
    IMPROVEMENT = "improvement"                  # Suggested improvements


@dataclass
class SharedScenario:
    """A scenario that has been shared with the community"""
    share_id: str
    original_scenario: Scenario
    creator_id: str
    sharing_permission: SharingPermission
    sharing_status: SharingStatus
    
    # Sharing metadata
    title: str
    description: str
    tags: List[str]
    category: str
    
    # Timestamps
    created_at: datetime
    
    # Community interaction
    rating: float = 0.0  # 0.0 to 5.0
    rating_count: int = 0
    use_count: int = 0
    favorite_count: int = 0
    
    # Contribution tracking
    contributors: List[str] = field(default_factory=list)
    contribution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Moderation
    moderation_notes: List[str] = field(default_factory=list)
    flagged_reasons: List[str] = field(default_factory=list)
    
    # Optional timestamps
    published_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.share_id:
            self.share_id = self._generate_share_id()
    
    def _generate_share_id(self) -> str:
        """Generate unique share ID"""
        content = f"{self.original_scenario.scenario_id}{self.creator_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert shared scenario to dictionary"""
        return {
            "share_id": self.share_id,
            "original_scenario": self.original_scenario.to_dict(),
            "creator_id": self.creator_id,
            "sharing_permission": self.sharing_permission.value,
            "sharing_status": self.sharing_status.value,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "category": self.category,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "use_count": self.use_count,
            "favorite_count": self.favorite_count,
            "contributors": self.contributors,
            "contribution_history": self.contribution_history,
            "moderation_notes": self.moderation_notes,
            "flagged_reasons": self.flagged_reasons,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class ScenarioContribution:
    """A contribution to a shared scenario"""
    contribution_id: str
    share_id: str
    contributor_id: str
    contribution_type: ContributionType
    description: str
    content: Dict[str, Any]
    
    # Timestamps
    created_at: datetime
    
    # Contribution metadata
    impact_score: float = 0.0  # 0.0 to 1.0
    community_rating: float = 0.0  # 0.0 to 5.0
    review_status: str = "pending"  # pending, approved, rejected
    
    # Optional timestamps
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    
    def __post_init__(self):
        if not self.contribution_id:
            self.contribution_id = self._generate_contribution_id()
    
    def _generate_contribution_id(self) -> str:
        """Generate unique contribution ID"""
        content = f"{self.share_id}{self.contributor_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contribution to dictionary"""
        return {
            "contribution_id": self.contribution_id,
            "share_id": self.share_id,
            "contributor_id": self.contributor_id,
            "contribution_type": self.contribution_type.value,
            "description": self.description,
            "content": self.content,
            "impact_score": self.impact_score,
            "community_rating": self.impact_score,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by
        }


@dataclass
class CommunityProfile:
    """Profile for a community that shares scenarios"""
    community_id: str
    name: str
    description: str
    creator_id: str
    
    # Community settings
    sharing_policy: SharingPermission
    moderation_enabled: bool = True
    contribution_guidelines: List[str] = field(default_factory=list)
    
    # Community statistics
    member_count: int = 0
    scenario_count: int = 0
    total_contributions: int = 0
    
    # Community metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.community_id:
            self.community_id = self._generate_community_id()
    
    def _generate_community_id(self) -> str:
        """Generate unique community ID"""
        content = f"{self.name}{self.creator_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert community profile to dictionary"""
        return {
            "community_id": self.community_id,
            "name": self.name,
            "description": self.description,
            "creator_id": self.creator_id,
            "sharing_policy": self.sharing_policy.value,
            "moderation_enabled": self.moderation_enabled,
            "contribution_guidelines": self.contribution_guidelines,
            "member_count": self.member_count,
            "scenario_count": self.scenario_count,
            "total_contributions": self.total_contributions,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


class ScenarioSharer:
    """
    Manages sharing of scenarios between users and communities
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.shared_scenarios: Dict[str, SharedScenario] = {}
        self.contributions: Dict[str, ScenarioContribution] = {}
        self.communities: Dict[str, CommunityProfile] = {}
        self.user_favorites: Dict[str, Set[str]] = {}  # user_id -> set of share_ids
        self.user_ratings: Dict[str, Dict[str, float]] = {}  # user_id -> share_id -> rating
        
        # Initialize with default communities
        self._initialize_default_communities()
    
    def _initialize_default_communities(self):
        """Initialize with default communities"""
        default_communities = [
            CommunityProfile(
                community_id="",
                name="Empathy Training Community",
                description="Community focused on empathy and emotional intelligence training scenarios",
                creator_id=self.node_id,
                sharing_policy=SharingPermission.COMMUNITY,
                contribution_guidelines=[
                    "Scenarios should promote empathy and understanding",
                    "Include diverse perspectives and backgrounds",
                    "Focus on real-world applications"
                ],
                tags=["empathy", "training", "emotional_intelligence"]
            ),
            CommunityProfile(
                community_id="",
                name="Conflict Resolution Network",
                description="Community for conflict resolution and communication scenarios",
                creator_id=self.node_id,
                sharing_policy=SharingPermission.COMMUNITY,
                contribution_guidelines=[
                    "Scenarios should address real conflict situations",
                    "Include multiple resolution approaches",
                    "Consider cultural and contextual factors"
                ],
                tags=["conflict_resolution", "communication", "mediation"]
            ),
            CommunityProfile(
                community_id="",
                name="Leadership Development Hub",
                description="Community for leadership and organizational decision-making scenarios",
                creator_id=self.node_id,
                sharing_policy=SharingPermission.COMMUNITY,
                contribution_guidelines=[
                    "Scenarios should challenge leadership thinking",
                    "Include ethical considerations",
                    "Address both individual and team leadership"
                ],
                tags=["leadership", "decision_making", "organizational"]
            )
        ]
        
        for community in default_communities:
            self.communities[community.community_id] = community
    
    def share_scenario(self, scenario: Scenario, creator_id: str, title: str, 
                      description: str, tags: List[str], category: str,
                      sharing_permission: SharingPermission = SharingPermission.COMMUNITY) -> str:
        """Share a scenario with the community"""
        try:
            # Create shared scenario
            shared_scenario = SharedScenario(
                share_id="",
                original_scenario=scenario,
                creator_id=creator_id,
                sharing_permission=sharing_permission,
                sharing_status=SharingStatus.DRAFT,
                title=title,
                description=description,
                tags=tags,
                category=category,
                created_at=datetime.utcnow()
            )
            
            # Store the shared scenario
            self.shared_scenarios[shared_scenario.share_id] = shared_scenario
            
            # Add creator as contributor
            shared_scenario.contributors.append(creator_id)
            
            # Add to community if applicable
            if sharing_permission == SharingPermission.COMMUNITY:
                self._add_to_community(shared_scenario.share_id, category)
            
            logger.info(f"Shared scenario: {title} (ID: {shared_scenario.share_id})")
            return shared_scenario.share_id
            
        except Exception as e:
            logger.error(f"Error sharing scenario: {e}")
            raise
    
    def _add_to_community(self, share_id: str, category: str):
        """Add a shared scenario to appropriate communities"""
        for community in self.communities.values():
            if category.lower() in [tag.lower() for tag in community.tags]:
                community.scenario_count += 1
                community.last_activity = datetime.utcnow()
                break
    
    def publish_scenario(self, share_id: str, publisher_id: str) -> bool:
        """Publish a draft scenario"""
        try:
            if share_id not in self.shared_scenarios:
                logger.error(f"Shared scenario {share_id} not found")
                return False
            
            shared_scenario = self.shared_scenarios[share_id]
            
            # Check permissions
            if shared_scenario.creator_id != publisher_id:
                logger.error(f"User {publisher_id} not authorized to publish scenario {share_id}")
                return False
            
            # Update status
            shared_scenario.sharing_status = SharingStatus.PUBLISHED
            shared_scenario.published_at = datetime.utcnow()
            shared_scenario.last_modified = datetime.utcnow()
            
            logger.info(f"Published scenario: {shared_scenario.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing scenario: {e}")
            return False
    
    def rate_scenario(self, user_id: str, share_id: str, rating: float) -> bool:
        """Rate a shared scenario"""
        try:
            if share_id not in self.shared_scenarios:
                logger.error(f"Shared scenario {share_id} not found")
                return False
            
            # Validate rating
            if not 0.0 <= rating <= 5.0:
                logger.error(f"Invalid rating: {rating}. Must be between 0.0 and 5.0")
                return False
            
            shared_scenario = self.shared_scenarios[share_id]
            
            # Store user rating
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            self.user_ratings[user_id][share_id] = rating
            
            # Update scenario rating
            if shared_scenario.rating_count == 0:
                shared_scenario.rating = rating
            else:
                # Calculate new average
                total_rating = shared_scenario.rating * shared_scenario.rating_count + rating
                shared_scenario.rating = total_rating / (shared_scenario.rating_count + 1)
            
            shared_scenario.rating_count += 1
            
            logger.info(f"User {user_id} rated scenario {share_id}: {rating}")
            return True
            
        except Exception as e:
            logger.error(f"Error rating scenario: {e}")
            return False
    
    def favorite_scenario(self, user_id: str, share_id: str) -> bool:
        """Add a scenario to user's favorites"""
        try:
            if share_id not in self.shared_scenarios:
                logger.error(f"Shared scenario {share_id} not found")
                return False
            
            if user_id not in self.user_favorites:
                self.user_favorites[user_id] = set()
            
            if share_id in self.user_favorites[user_id]:
                # Remove from favorites
                self.user_favorites[user_id].remove(share_id)
                self.shared_scenarios[share_id].favorite_count -= 1
                logger.info(f"Removed scenario {share_id} from favorites for user {user_id}")
            else:
                # Add to favorites
                self.user_favorites[user_id].add(share_id)
                self.shared_scenarios[share_id].favorite_count += 1
                logger.info(f"Added scenario {share_id} to favorites for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing favorites: {e}")
            return False
    
    def contribute_to_scenario(self, share_id: str, contributor_id: str,
                             contribution_type: ContributionType, description: str,
                             content: Dict[str, Any]) -> str:
        """Contribute to a shared scenario"""
        try:
            if share_id not in self.shared_scenarios:
                logger.error(f"Shared scenario {share_id} not found")
                return False
            
            shared_scenario = self.shared_scenarios[share_id]
            
            # Create contribution
            contribution = ScenarioContribution(
                contribution_id="",
                share_id=share_id,
                contributor_id=contributor_id,
                contribution_type=contribution_type,
                description=description,
                content=content,
                created_at=datetime.utcnow()
            )
            
            # Store contribution
            self.contributions[contribution.contribution_id] = contribution
            
            # Add contributor to scenario if not already present
            if contributor_id not in shared_scenario.contributors:
                shared_scenario.contributors.append(contributor_id)
            
            # Add to contribution history
            shared_scenario.contribution_history.append({
                "contribution_id": contribution.contribution_id,
                "contributor_id": contributor_id,
                "type": contribution_type.value,
                "description": description,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update community statistics
            for community in self.communities.values():
                if community.community_id in [c.get("community_id") for c in shared_scenario.contribution_history]:
                    community.total_contributions += 1
                    community.last_activity = datetime.utcnow()
                    break
            
            logger.info(f"Contribution added to scenario {share_id}: {description}")
            return contribution.contribution_id
            
        except Exception as e:
            logger.error(f"Error contributing to scenario: {e}")
            raise
    
    def search_scenarios(self, query: str = "", tags: List[str] = None, 
                        category: str = None, min_rating: float = 0.0,
                        permission_level: SharingPermission = None) -> List[SharedScenario]:
        """Search for shared scenarios"""
        try:
            results = []
            
            for scenario in self.shared_scenarios.values():
                # Check if scenario matches search criteria
                if not self._matches_search_criteria(scenario, query, tags, category, min_rating, permission_level):
                    continue
                
                results.append(scenario)
            
            # Sort by relevance (rating, use count, recency)
            results.sort(key=lambda s: (
                s.rating * 0.4 + 
                s.use_count * 0.3 + 
                (datetime.utcnow() - s.created_at).days * 0.3
            ), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching scenarios: {e}")
            return []
    
    def _matches_search_criteria(self, scenario: SharedScenario, query: str, tags: List[str],
                               category: str, min_rating: float, permission_level: SharingPermission) -> bool:
        """Check if a scenario matches search criteria"""
        # Query match
        if query and query.lower() not in scenario.title.lower() and query.lower() not in scenario.description.lower():
            return False
        
        # Tags match
        if tags and not any(tag.lower() in [t.lower() for t in scenario.tags]):
            return False
        
        # Category match
        if category and category.lower() != scenario.category.lower():
            return False
        
        # Rating match
        if scenario.rating < min_rating:
            return False
        
        # Permission level match
        if permission_level and scenario.sharing_permission != permission_level:
            return False
        
        return True
    
    def get_user_scenarios(self, user_id: str) -> List[SharedScenario]:
        """Get scenarios created by a specific user"""
        return [s for s in self.shared_scenarios.values() if s.creator_id == user_id]
    
    def get_user_favorites(self, user_id: str) -> List[SharedScenario]:
        """Get scenarios favorited by a specific user"""
        if user_id not in self.user_favorites:
            return []
        
        return [self.shared_scenarios[share_id] for share_id in self.user_favorites[user_id]
                if share_id in self.shared_scenarios]
    
    def get_community_scenarios(self, community_id: str) -> List[SharedScenario]:
        """Get scenarios associated with a specific community"""
        if community_id not in self.communities:
            return []
        
        community = self.communities[community_id]
        return [s for s in self.shared_scenarios.values() 
                if s.category.lower() in [tag.lower() for tag in community.tags]]
    
    def get_scenario_contributions(self, share_id: str) -> List[ScenarioContribution]:
        """Get all contributions for a specific scenario"""
        return [c for c in self.contributions.values() if c.share_id == share_id]
    
    def get_scenario_sharer_summary(self) -> Dict[str, Any]:
        """Get summary of the scenario sharing system"""
        total_scenarios = len(self.shared_scenarios)
        published_scenarios = len([s for s in self.shared_scenarios.values() 
                                if s.sharing_status == SharingStatus.PUBLISHED])
        total_contributions = len(self.contributions)
        total_communities = len(self.communities)
        
        # Scenarios by category
        scenarios_by_category = {}
        for scenario in self.shared_scenarios.values():
            category = scenario.category
            scenarios_by_category[category] = scenarios_by_category.get(category, 0) + 1
        
        # Scenarios by permission level
        scenarios_by_permission = {}
        for scenario in self.shared_scenarios.values():
            permission = scenario.sharing_permission.value
            scenarios_by_permission[permission] = scenarios_by_permission.get(permission, 0) + 1
        
        return {
            "total_scenarios": total_scenarios,
            "published_scenarios": published_scenarios,
            "total_contributions": total_contributions,
            "total_communities": total_communities,
            "scenarios_by_category": scenarios_by_category,
            "scenarios_by_permission": scenarios_by_permission,
            "top_rated_scenarios": self._get_top_rated_scenarios(5),
            "recent_contributions": self._get_recent_contributions(10)
        }
    
    def _get_top_rated_scenarios(self, count: int) -> List[Dict[str, Any]]:
        """Get top rated scenarios"""
        sorted_scenarios = sorted(self.shared_scenarios.values(), key=lambda s: s.rating, reverse=True)
        top_scenarios = sorted_scenarios[:count]
        
        return [
            {
                "share_id": s.share_id,
                "title": s.title,
                "rating": s.rating,
                "rating_count": s.rating_count,
                "category": s.category
            }
            for s in top_scenarios
        ]
    
    def _get_recent_contributions(self, count: int) -> List[Dict[str, Any]]:
        """Get recent contributions"""
        sorted_contributions = sorted(self.contributions.values(), key=lambda c: c.created_at, reverse=True)
        recent_contributions = sorted_contributions[:count]
        
        return [
            {
                "contribution_id": c.contribution_id,
                "share_id": c.share_id,
                "contributor_id": c.contributor_id,
                "type": c.contribution_type.value,
                "description": c.description,
                "created_at": c.created_at.isoformat()
            }
            for c in recent_contributions
        ]
