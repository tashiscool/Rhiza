"""
Hierarchical Message Router for Nested Communication Channels
============================================================

Intelligent message routing across the nested communication hierarchy:
family → village → region → world → chosen circles

Handles message escalation, cross-scope communication, and privacy-preserving routing.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .nested_channels import (
    CommunicationScope, MessagePriority, NestedMessage, 
    CommunicationChannel, NestedChannelManager
)

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Different strategies for routing messages across scopes"""
    DIRECT = "direct"                    # Direct to specified scope only
    ESCALATE = "escalate"               # Start local, escalate if needed
    BROADCAST = "broadcast"             # Send to all appropriate scopes
    PERCOLATE = "percolate"             # Bubble up through hierarchy
    AFFINITY = "affinity"               # Route to chosen circles only

class MessageFlow(Enum):
    """Direction of message flow through hierarchy"""
    UPWARD = "upward"                   # Family → Village → Region → World
    DOWNWARD = "downward"               # World → Region → Village → Family  
    LATERAL = "lateral"                 # Within same scope level
    CROSS_SCOPE = "cross_scope"         # Between chosen circles

@dataclass
class RoutingRule:
    """Rules for routing messages based on content and context"""
    rule_id: str
    name: str
    condition: Dict[str, Any]           # Conditions that trigger this rule
    routing_strategy: RoutingStrategy
    target_scopes: List[CommunicationScope]
    priority_boost: int = 0             # Boost message priority if rule matches
    privacy_override: Optional[str] = None  # Override privacy level
    trust_requirement: Optional[float] = None  # Override trust requirements

class HierarchicalMessageRouter:
    """
    Intelligent router for the nested communication hierarchy
    
    Routes messages through the social layers:
    - Whispers at the hearth (family)
    - Songs of the village (community) 
    - Councils of regions (broader geography)
    - The great chorus of the world (global)
    - Secret circles of shared obsession (chosen circles)
    """
    
    def __init__(self, channel_manager: NestedChannelManager):
        self.channel_manager = channel_manager
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.escalation_thresholds: Dict[CommunicationScope, Dict[str, float]] = {}
        
        # Initialize default routing rules
        self._initialize_default_rules()
        self._initialize_escalation_thresholds()
    
    def _initialize_default_rules(self):
        """Initialize default routing rules for common scenarios"""
        
        # Emergency messages - immediate escalation
        emergency_rule = RoutingRule(
            rule_id="emergency_escalation",
            name="Emergency Escalation",
            condition={"priority": "emergency", "keywords": ["help", "emergency", "urgent"]},
            routing_strategy=RoutingStrategy.ESCALATE,
            target_scopes=[CommunicationScope.FAMILY, CommunicationScope.VILLAGE, CommunicationScope.REGION],
            priority_boost=2,
            trust_requirement=0.5  # Lower trust barrier for emergencies
        )
        
        # Local questions - keep in village
        local_rule = RoutingRule(
            rule_id="local_questions",
            name="Local Community Questions",
            condition={"keywords": ["local", "neighborhood", "nearby", "community"]},
            routing_strategy=RoutingStrategy.DIRECT,
            target_scopes=[CommunicationScope.VILLAGE],
            trust_requirement=0.8
        )
        
        # Specialized knowledge - route to chosen circles
        specialty_rule = RoutingRule(
            rule_id="specialized_knowledge",
            name="Specialized Knowledge Sharing",
            condition={"tags": ["technical", "expertise", "specialized"]},
            routing_strategy=RoutingStrategy.AFFINITY,
            target_scopes=[CommunicationScope.CHOSEN],
            trust_requirement=0.7
        )
        
        # Global discussions - world scope
        global_rule = RoutingRule(
            rule_id="global_discussions",
            name="Global Discussions",
            condition={"keywords": ["global", "world", "international", "humanity"]},
            routing_strategy=RoutingStrategy.DIRECT,
            target_scopes=[CommunicationScope.WORLD],
            trust_requirement=0.6
        )
        
        # Family matters - intimate scope only
        family_rule = RoutingRule(
            rule_id="family_matters",
            name="Family and Personal Matters",
            condition={"keywords": ["family", "personal", "private", "intimate"]},
            routing_strategy=RoutingStrategy.DIRECT,
            target_scopes=[CommunicationScope.FAMILY],
            privacy_override="private",
            trust_requirement=0.9
        )
        
        # Register rules
        rules = [emergency_rule, local_rule, specialty_rule, global_rule, family_rule]
        for rule in rules:
            self.routing_rules[rule.rule_id] = rule
        
        logger.info(f"Initialized {len(rules)} default routing rules")
    
    def _initialize_escalation_thresholds(self):
        """Initialize thresholds for message escalation"""
        
        self.escalation_thresholds = {
            CommunicationScope.FAMILY: {
                "response_time_hours": 2,      # Escalate if no response in 2 hours
                "engagement_threshold": 0.5,   # Escalate if <50% engagement
                "consensus_threshold": 0.8     # Escalate if can't reach 80% consensus
            },
            CommunicationScope.VILLAGE: {
                "response_time_hours": 8,
                "engagement_threshold": 0.3,
                "consensus_threshold": 0.7
            },
            CommunicationScope.REGION: {
                "response_time_hours": 24,
                "engagement_threshold": 0.2,
                "consensus_threshold": 0.6
            },
            CommunicationScope.WORLD: {
                "response_time_hours": 72,
                "engagement_threshold": 0.1,
                "consensus_threshold": 0.5
            },
            CommunicationScope.CHOSEN: {
                "response_time_hours": 12,
                "engagement_threshold": 0.4,
                "consensus_threshold": 0.7
            }
        }
    
    async def route_message(self, message: NestedMessage) -> Dict[str, Any]:
        """
        Route a message through the nested communication hierarchy
        
        Analyzes content, applies routing rules, and determines optimal
        communication path through family → village → region → world → chosen circles
        """
        
        routing_result = {
            'message_id': message.message_id,
            'original_scopes': message.target_scopes.copy(),
            'applied_rules': [],
            'final_scopes': [],
            'routing_strategy': RoutingStrategy.DIRECT,
            'privacy_level': message.privacy_level,
            'trust_adjustments': {},
            'routing_path': [],
            'estimated_reach': 0
        }
        
        # 1. Analyze message content and context
        content_analysis = await self._analyze_message_content(message)
        
        # 2. Apply routing rules
        applicable_rules = self._find_applicable_rules(message, content_analysis)
        routing_result['applied_rules'] = [rule.rule_id for rule in applicable_rules]
        
        # 3. Determine routing strategy
        primary_strategy = self._determine_routing_strategy(applicable_rules, message)
        routing_result['routing_strategy'] = primary_strategy
        
        # 4. Calculate target scopes based on strategy
        target_scopes = await self._calculate_target_scopes(
            message, applicable_rules, primary_strategy
        )
        routing_result['final_scopes'] = [scope.value for scope in target_scopes]
        
        # 5. Apply privacy and trust adjustments
        privacy_level, trust_adjustments = self._apply_privacy_trust_rules(
            applicable_rules, message
        )
        routing_result['privacy_level'] = privacy_level
        routing_result['trust_adjustments'] = trust_adjustments
        
        # 6. Calculate routing path through hierarchy
        routing_path = await self._calculate_routing_path(target_scopes, primary_strategy)
        routing_result['routing_path'] = routing_path
        
        # 7. Estimate message reach
        estimated_reach = await self._estimate_message_reach(target_scopes, message)
        routing_result['estimated_reach'] = estimated_reach
        
        # 8. Execute routing
        await self._execute_routing(message, routing_result)
        
        # 9. Log routing decision
        self.routing_history.append({
            'timestamp': datetime.now().isoformat(),
            'routing_result': routing_result,
            'content_analysis': content_analysis
        })
        
        logger.info(f"Routed message {message.message_id} using {primary_strategy.value} strategy to {len(target_scopes)} scopes")
        return routing_result
    
    async def _analyze_message_content(self, message: NestedMessage) -> Dict[str, Any]:
        """Analyze message content for routing decisions"""
        
        content = message.content.lower()
        words = content.split()
        
        analysis = {
            'word_count': len(words),
            'keywords': [],
            'urgency_indicators': [],
            'scope_hints': [],
            'topic_categories': [],
            'emotional_tone': 'neutral',
            'requires_expertise': False
        }
        
        # Keyword extraction (simplified)
        urgency_keywords = ['urgent', 'emergency', 'help', 'crisis', 'immediate']
        local_keywords = ['local', 'neighborhood', 'nearby', 'community', 'village']
        global_keywords = ['global', 'world', 'international', 'everyone', 'humanity']
        family_keywords = ['family', 'personal', 'private', 'intimate', 'close']
        expertise_keywords = ['technical', 'specialized', 'expert', 'advanced', 'complex']
        
        # Check for urgency
        analysis['urgency_indicators'] = [word for word in urgency_keywords if word in content]
        
        # Check for scope hints  
        if any(word in content for word in local_keywords):
            analysis['scope_hints'].append('local')
        if any(word in content for word in global_keywords):
            analysis['scope_hints'].append('global')
        if any(word in content for word in family_keywords):
            analysis['scope_hints'].append('family')
        
        # Check for expertise requirements
        analysis['requires_expertise'] = any(word in content for word in expertise_keywords)
        
        # Emotional tone analysis (simplified)
        positive_words = ['happy', 'great', 'wonderful', 'excited']
        negative_words = ['sad', 'angry', 'frustrated', 'worried', 'concerned']
        
        if any(word in content for word in positive_words):
            analysis['emotional_tone'] = 'positive'
        elif any(word in content for word in negative_words):
            analysis['emotional_tone'] = 'negative'
        
        return analysis
    
    def _find_applicable_rules(
        self, 
        message: NestedMessage, 
        content_analysis: Dict[str, Any]
    ) -> List[RoutingRule]:
        """Find routing rules that apply to this message"""
        
        applicable_rules = []
        
        for rule in self.routing_rules.values():
            if self._rule_matches_message(rule, message, content_analysis):
                applicable_rules.append(rule)
        
        # Sort by specificity (more specific conditions first)
        applicable_rules.sort(key=lambda r: len(r.condition), reverse=True)
        
        return applicable_rules
    
    def _rule_matches_message(
        self, 
        rule: RoutingRule, 
        message: NestedMessage, 
        content_analysis: Dict[str, Any]
    ) -> bool:
        """Check if a routing rule matches the message"""
        
        conditions = rule.condition
        
        # Check priority condition
        if 'priority' in conditions:
            if message.priority.value != conditions['priority']:
                return False
        
        # Check keyword conditions
        if 'keywords' in conditions:
            message_content = message.content.lower()
            if not any(keyword in message_content for keyword in conditions['keywords']):
                return False
        
        # Check tag conditions
        if 'tags' in conditions:
            message_tags = getattr(message, 'tags', [])
            if not any(tag in message_tags for tag in conditions['tags']):
                return False
        
        # Check scope hints from content analysis
        if 'scope_hints' in conditions:
            if not any(hint in content_analysis['scope_hints'] for hint in conditions['scope_hints']):
                return False
        
        return True
    
    def _determine_routing_strategy(
        self, 
        applicable_rules: List[RoutingRule], 
        message: NestedMessage
    ) -> RoutingStrategy:
        """Determine the primary routing strategy to use"""
        
        if not applicable_rules:
            # Default strategy based on message priority
            if message.priority in [MessagePriority.EMERGENCY, MessagePriority.CALL]:
                return RoutingStrategy.ESCALATE
            elif message.priority == MessagePriority.CHORUS:
                return RoutingStrategy.BROADCAST
            else:
                return RoutingStrategy.DIRECT
        
        # Use strategy from highest priority rule
        return applicable_rules[0].routing_strategy
    
    async def _calculate_target_scopes(
        self,
        message: NestedMessage,
        applicable_rules: List[RoutingRule], 
        strategy: RoutingStrategy
    ) -> List[CommunicationScope]:
        """Calculate target communication scopes based on routing strategy"""
        
        if strategy == RoutingStrategy.DIRECT:
            # Use scopes from rules or original message
            if applicable_rules:
                return applicable_rules[0].target_scopes
            return message.target_scopes
        
        elif strategy == RoutingStrategy.ESCALATE:
            # Start with lowest scope and add higher ones
            scopes = [CommunicationScope.FAMILY]
            if message.priority in [MessagePriority.CALL, MessagePriority.EMERGENCY]:
                scopes.extend([CommunicationScope.VILLAGE, CommunicationScope.REGION])
            if message.priority == MessagePriority.EMERGENCY:
                scopes.append(CommunicationScope.WORLD)
            return scopes
        
        elif strategy == RoutingStrategy.BROADCAST:
            # Send to all appropriate scopes
            return [CommunicationScope.VILLAGE, CommunicationScope.REGION, CommunicationScope.WORLD]
        
        elif strategy == RoutingStrategy.PERCOLATE:
            # Start local and bubble up
            return [CommunicationScope.FAMILY, CommunicationScope.VILLAGE, CommunicationScope.REGION]
        
        elif strategy == RoutingStrategy.AFFINITY:
            # Route to chosen circles
            return [CommunicationScope.CHOSEN]
        
        return message.target_scopes
    
    def _apply_privacy_trust_rules(
        self, 
        applicable_rules: List[RoutingRule], 
        message: NestedMessage
    ) -> Tuple[str, Dict[str, float]]:
        """Apply privacy and trust rule overrides"""
        
        privacy_level = message.privacy_level
        trust_adjustments = {}
        
        for rule in applicable_rules:
            # Override privacy level if specified
            if rule.privacy_override:
                privacy_level = rule.privacy_override
            
            # Apply trust requirement overrides
            if rule.trust_requirement is not None:
                for scope in rule.target_scopes:
                    trust_adjustments[scope.value] = rule.trust_requirement
        
        return privacy_level, trust_adjustments
    
    async def _calculate_routing_path(
        self, 
        target_scopes: List[CommunicationScope], 
        strategy: RoutingStrategy
    ) -> List[str]:
        """Calculate the specific routing path through the hierarchy"""
        
        if strategy == RoutingStrategy.ESCALATE:
            # Hierarchical path: family → village → region → world
            scope_order = [
                CommunicationScope.FAMILY,
                CommunicationScope.VILLAGE, 
                CommunicationScope.REGION,
                CommunicationScope.WORLD
            ]
            
            return [
                scope.value for scope in scope_order 
                if scope in target_scopes
            ]
        
        elif strategy == RoutingStrategy.BROADCAST:
            # Parallel delivery to all scopes
            return [f"parallel_{scope.value}" for scope in target_scopes]
        
        else:
            # Direct routing
            return [scope.value for scope in target_scopes]
    
    async def _estimate_message_reach(
        self, 
        target_scopes: List[CommunicationScope], 
        message: NestedMessage
    ) -> int:
        """Estimate how many nodes the message will reach"""
        
        total_reach = 0
        
        for scope in target_scopes:
            channels = self.channel_manager.get_channels_for_scope(scope)
            
            for channel in channels:
                # Filter members by trust level
                qualified_members = 0
                for member_id in channel.members:
                    member_trust = self.channel_manager.trust_scores.get(member_id, 0.5)
                    if member_trust >= channel.min_trust_level:
                        qualified_members += 1
                
                total_reach += qualified_members
        
        return total_reach
    
    async def _execute_routing(self, message: NestedMessage, routing_result: Dict[str, Any]):
        """Execute the routing decision"""
        
        # Update message with routing results
        final_scopes = [CommunicationScope(scope) for scope in routing_result['final_scopes']]
        message.target_scopes = final_scopes
        message.privacy_level = routing_result['privacy_level']
        
        # Apply trust adjustments
        trust_adjustments = routing_result['trust_adjustments']
        
        # Route through channel manager with adjustments
        for scope in final_scopes:
            channels = self.channel_manager.get_channels_for_scope(scope)
            
            for channel in channels:
                # Apply trust adjustment if specified
                if scope.value in trust_adjustments:
                    original_trust = channel.min_trust_level
                    channel.min_trust_level = trust_adjustments[scope.value]
                    
                    # Route message
                    await self._route_to_channel(message, channel)
                    
                    # Restore original trust level
                    channel.min_trust_level = original_trust
                else:
                    await self._route_to_channel(message, channel)
    
    async def _route_to_channel(self, message: NestedMessage, channel: CommunicationChannel):
        """Route message to a specific channel"""
        
        # This would integrate with the channel manager's routing
        # For now, just update the message's channel list
        if channel.channel_id not in message.channel_ids:
            message.channel_ids.append(channel.channel_id)
        
        logger.debug(f"Routed message {message.message_id} to channel {channel.name}")
    
    async def check_escalation_needed(self, message_id: str) -> Optional[List[CommunicationScope]]:
        """Check if a message needs to be escalated to higher scopes"""
        
        if message_id not in self.channel_manager.message_history:
            return None
        
        message = self.channel_manager.message_history[message_id]
        
        # Check if enough time has passed for escalation
        time_since_sent = datetime.now() - message.timestamp
        
        for scope in message.target_scopes:
            thresholds = self.escalation_thresholds.get(scope, {})
            max_response_time = thresholds.get('response_time_hours', 24)
            
            if time_since_sent.total_seconds() / 3600 > max_response_time:
                # Message should be escalated
                return await self._get_escalation_scopes(scope)
        
        return None
    
    async def _get_escalation_scopes(self, current_scope: CommunicationScope) -> List[CommunicationScope]:
        """Get the next level scopes for escalation"""
        
        escalation_map = {
            CommunicationScope.FAMILY: [CommunicationScope.VILLAGE],
            CommunicationScope.VILLAGE: [CommunicationScope.REGION],
            CommunicationScope.REGION: [CommunicationScope.WORLD],
            CommunicationScope.WORLD: [],  # Already at highest level
            CommunicationScope.CHOSEN: [CommunicationScope.REGION, CommunicationScope.WORLD]
        }
        
        return escalation_map.get(current_scope, [])
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule"""
        self.routing_rules[rule.rule_id] = rule
        logger.info(f"Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule"""
        if rule_id in self.routing_rules:
            del self.routing_rules[rule_id]
            logger.info(f"Removed routing rule: {rule_id}")
            return True
        return False
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        
        if not self.routing_history:
            return {'error': 'No routing history available'}
        
        stats = {
            'total_messages_routed': len(self.routing_history),
            'routing_strategies': {},
            'scope_distribution': {},
            'rule_usage': {},
            'average_reach': 0
        }
        
        total_reach = 0
        
        for entry in self.routing_history:
            result = entry['routing_result']
            
            # Count routing strategies
            strategy = result['routing_strategy'].value if hasattr(result['routing_strategy'], 'value') else str(result['routing_strategy'])
            stats['routing_strategies'][strategy] = stats['routing_strategies'].get(strategy, 0) + 1
            
            # Count scope distribution
            for scope in result['final_scopes']:
                stats['scope_distribution'][scope] = stats['scope_distribution'].get(scope, 0) + 1
            
            # Count rule usage
            for rule_id in result['applied_rules']:
                stats['rule_usage'][rule_id] = stats['rule_usage'].get(rule_id, 0) + 1
            
            # Track reach
            total_reach += result.get('estimated_reach', 0)
        
        stats['average_reach'] = total_reach / len(self.routing_history) if self.routing_history else 0
        
        return stats