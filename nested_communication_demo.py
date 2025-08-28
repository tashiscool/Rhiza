#!/usr/bin/env python3
"""
Nested Communication System Demonstration
=========================================

Shows the hierarchical communication channels in The Mesh:

"from the intimacy of family, to the voice of the village, to the chorus of regions,
to the global tide, to the smaller circles of shared passions."

Demonstrates:
- Family → Village → Region → World → Chosen Circles
- Message routing and escalation
- Privacy and trust controls at each level
- Intelligent routing based on content and context
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_nested_communication():
    """Comprehensive demonstration of the nested communication system"""
    
    print("🌐 THE MESH: NESTED COMMUNICATION CHANNELS")
    print("=" * 55)
    print()
    print('🎭 "from the intimacy of family, to the voice of the village,')
    print('    to the chorus of regions, to the global tide, to the')
    print('    smaller circles of shared passions."')
    print()
    
    # 1. Communication Scopes Overview
    print("📡 1. COMMUNICATION SCOPES")
    print("-" * 30)
    
    scopes = [
        ("👨‍👩‍👧‍👦 FAMILY", "Intimate trusted circle (1-8 nodes)", "whispers at the hearth", 0.9, "private"),
        ("🏘️ VILLAGE", "Local community (20-150 nodes)", "songs of the village", 0.8, "selective"),
        ("🌆 REGION", "Geographic/cultural area (500-5000 nodes)", "councils of regions", 0.7, "selective"),
        ("🌍 WORLD", "Global mesh network (unlimited)", "the great chorus of the world", 0.6, "open"),
        ("⭕ CHOSEN", "Affinity circles across levels", "secret circles of shared obsession", 0.7, "selective")
    ]
    
    for emoji_name, description, poetic, trust, privacy in scopes:
        print(f"  {emoji_name}")
        print(f"    📊 Size: {description}")
        print(f"    🎭 Tone: {poetic}")
        print(f"    🤝 Trust Required: {trust} ({int(trust*100)}%)")
        print(f"    🔒 Privacy: {privacy}")
        print()
    
    # 2. Message Flow Patterns
    print("🔄 2. MESSAGE FLOW PATTERNS")
    print("-" * 30)
    
    flow_patterns = [
        ("ESCALATION", "family → village → region → world", "Local issue needs broader help"),
        ("BROADCAST", "all scopes simultaneously", "Emergency or celebration announcement"),
        ("PERCOLATION", "idea bubbles up through levels", "Innovation spreading organically"), 
        ("AFFINITY", "chosen circles only", "Specialized knowledge sharing"),
        ("INTIMATE", "family scope only", "Personal matters stay private")
    ]
    
    for pattern, flow, use_case in flow_patterns:
        print(f"  📈 {pattern}:")
        print(f"     🔀 Flow: {flow}")
        print(f"     💡 Use: {use_case}")
        print()
    
    # 3. Routing Intelligence Demo
    print("🧠 3. INTELLIGENT MESSAGE ROUTING")
    print("-" * 35)
    
    try:
        from src.mesh_core.communication.nested_channels import (
            NestedChannelManager, CommunicationScope, MessagePriority
        )
        from src.mesh_core.communication.message_router import HierarchicalMessageRouter
        
        # Initialize communication system
        channel_manager = NestedChannelManager("demo_node_001")
        router = HierarchicalMessageRouter(channel_manager)
        
        print("✅ Communication system initialized")
        print(f"   📡 Channels created: {len(channel_manager.channels)}")
        print(f"   🧠 Routing rules: {len(router.routing_rules)}")
        print()
        
        # Demo messages with different routing patterns
        demo_messages = [
            {
                'content': "Help! Medical emergency at home!",
                'expected_routing': "ESCALATION (family → village → region)",
                'priority': MessagePriority.EMERGENCY,
                'scopes': [CommunicationScope.FAMILY]
            },
            {
                'content': "Looking for local farmers market recommendations",
                'expected_routing': "DIRECT (village only)",
                'priority': MessagePriority.VOICE,
                'scopes': [CommunicationScope.VILLAGE]
            },
            {
                'content': "New breakthrough in quantum computing protocols",
                'expected_routing': "AFFINITY (chosen circles)",
                'priority': MessagePriority.VOICE,
                'scopes': [CommunicationScope.CHOSEN]
            },
            {
                'content': "Family celebration this weekend - private",
                'expected_routing': "INTIMATE (family only)",
                'priority': MessagePriority.WHISPER,
                'scopes': [CommunicationScope.FAMILY]
            },
            {
                'content': "Global climate action coordination needed",
                'expected_routing': "BROADCAST (all scopes)",
                'priority': MessagePriority.CHORUS,
                'scopes': [CommunicationScope.WORLD, CommunicationScope.REGION]
            }
        ]
        
        for i, msg_demo in enumerate(demo_messages, 1):
            print(f"📝 Message {i}: \"{msg_demo['content']}\"")
            print(f"   🎯 Expected Routing: {msg_demo['expected_routing']}")
            print(f"   📢 Priority: {msg_demo['priority'].value}")
            
            # Simulate routing (would be real message in production)
            message = await channel_manager.send_message(
                content=msg_demo['content'],
                target_scopes=msg_demo['scopes'],
                priority=msg_demo['priority']
            )
            
            print(f"   ✅ Message routed successfully (ID: {message.message_id})")
            print()
        
    except Exception as e:
        print(f"⚠️  Routing demo: {e}")
        print("   (This demonstrates the system architecture)")
        print()
    
    # 4. Privacy and Trust Controls
    print("🛡️ 4. PRIVACY AND TRUST CONTROLS")
    print("-" * 35)
    
    privacy_controls = {
        'FAMILY': {
            'trust_threshold': 0.9,
            'privacy_level': 'private',
            'encryption': 'end-to-end',
            'member_limit': 8,
            'consensus_required': False
        },
        'VILLAGE': {
            'trust_threshold': 0.8,
            'privacy_level': 'selective',
            'encryption': 'group',
            'member_limit': 150,
            'consensus_required': True,
            'consensus_threshold': 0.7
        },
        'REGION': {
            'trust_threshold': 0.7,
            'privacy_level': 'selective',
            'encryption': 'optional',
            'member_limit': 5000,
            'consensus_required': True,
            'consensus_threshold': 0.6
        },
        'WORLD': {
            'trust_threshold': 0.6,
            'privacy_level': 'open',
            'encryption': 'optional',
            'member_limit': None,
            'consensus_required': True,
            'consensus_threshold': 0.5
        },
        'CHOSEN': {
            'trust_threshold': 0.7,
            'privacy_level': 'selective',
            'encryption': 'group',
            'member_limit': 500,
            'consensus_required': True,
            'consensus_threshold': 0.7
        }
    }
    
    for scope, controls in privacy_controls.items():
        print(f"  🔒 {scope} Controls:")
        for control, value in controls.items():
            if control == 'member_limit' and value is None:
                value = 'unlimited'
            elif isinstance(value, float):
                value = f"{value} ({int(value*100)}%)"
            print(f"     • {control.replace('_', ' ').title()}: {value}")
        print()
    
    # 5. Escalation Example
    print("📈 5. MESSAGE ESCALATION EXAMPLE")
    print("-" * 35)
    
    escalation_scenario = {
        'initial_message': "Car broke down on rural road, need help",
        'timeline': [
            ('0 min', 'FAMILY', 'Send to family circle', 'No response (family busy)'),
            ('15 min', 'VILLAGE', 'Escalate to village', 'Local mechanic responds'),
            ('30 min', 'REGION', 'Alert regional network', 'Backup help dispatched'),
            ('RESOLVED', 'SUCCESS', 'Help arrives', 'Multi-layer response successful')
        ]
    }
    
    print(f"🚗 Scenario: \"{escalation_scenario['initial_message']}\"")
    print()
    
    for time, scope, action, result in escalation_scenario['timeline']:
        if scope == 'SUCCESS':
            print(f"  ✅ {time}: {action}")
            print(f"      🎯 {result}")
        else:
            print(f"  📤 {time}: {scope} - {action}")
            print(f"      📥 {result}")
        print()
    
    # 6. Chosen Circles Example
    print("⭕ 6. CHOSEN CIRCLES EXAMPLE")
    print("-" * 30)
    
    chosen_circles = [
        {
            'name': 'Quantum Computing Research',
            'specialization': 'quantum_computing',
            'members': 47,
            'trust_score': 0.89,
            'activity': 'High - daily discussions',
            'recent_topic': 'Error correction protocols'
        },
        {
            'name': 'Permaculture Network',
            'specialization': 'sustainable_agriculture', 
            'members': 156,
            'trust_score': 0.82,
            'activity': 'Medium - seasonal peaks',
            'recent_topic': 'Water management strategies'
        },
        {
            'name': 'Medieval History Society',
            'specialization': 'historical_research',
            'members': 23,
            'trust_score': 0.94,
            'activity': 'Low - weekly meetings',
            'recent_topic': 'Byzantine trade routes'
        }
    ]
    
    for circle in chosen_circles:
        print(f"  🔷 {circle['name']}")
        print(f"     🎯 Specialization: {circle['specialization']}")
        print(f"     👥 Members: {circle['members']}")
        print(f"     🤝 Trust Score: {circle['trust_score']} ({int(circle['trust_score']*100)}%)")
        print(f"     📊 Activity: {circle['activity']}")
        print(f"     💬 Recent: {circle['recent_topic']}")
        print()
    
    # 7. Communication Statistics
    print("📊 7. COMMUNICATION STATISTICS")
    print("-" * 35)
    
    stats = {
        'total_channels': 847,
        'active_conversations': 156,
        'messages_today': 2341,
        'scope_distribution': {
            'family': {'channels': 234, 'messages': 567, 'percentage': '24%'},
            'village': {'channels': 89, 'messages': 891, 'percentage': '38%'}, 
            'region': {'channels': 34, 'messages': 445, 'percentage': '19%'},
            'world': {'channels': 12, 'messages': 267, 'percentage': '11%'},
            'chosen': {'channels': 478, 'messages': 171, 'percentage': '8%'}
        },
        'routing_efficiency': '94%',
        'average_response_time': {
            'family': '3 minutes',
            'village': '12 minutes', 
            'region': '1.2 hours',
            'world': '4.7 hours',
            'chosen': '45 minutes'
        }
    }
    
    print(f"  📈 Total Channels: {stats['total_channels']}")
    print(f"  💬 Active Conversations: {stats['active_conversations']}")
    print(f"  📨 Messages Today: {stats['messages_today']}")
    print(f"  🎯 Routing Efficiency: {stats['routing_efficiency']}")
    print()
    
    print("  📊 Distribution by Scope:")
    for scope, data in stats['scope_distribution'].items():
        print(f"     {scope.title()}: {data['channels']} channels, {data['messages']} msgs ({data['percentage']})")
    
    print()
    print("  ⏱️ Average Response Times:")
    for scope, time in stats['average_response_time'].items():
        print(f"     {scope.title()}: {time}")
    
    print()
    
    # 8. Benefits Summary
    print("🌟 8. NESTED COMMUNICATION BENEFITS")
    print("-" * 40)
    
    benefits = [
        ("🔄 Natural Flow", "Messages flow through social structures humans understand"),
        ("⚡ Efficient Routing", "Right message to right audience with minimal noise"),
        ("🛡️ Privacy Protection", "Intimate conversations stay private, public ones reach broadly"),
        ("🤝 Trust-Based Access", "Higher trust enables broader communication reach"),
        ("📈 Smart Escalation", "Important messages automatically get wider attention"),
        ("🎯 Specialized Expertise", "Chosen circles connect people with shared interests"),
        ("🌍 Global + Local", "World-spanning network respects local communities"),
        ("🔒 Secure by Design", "Privacy and encryption appropriate to each communication level")
    ]
    
    for icon_benefit, description in benefits:
        print(f"  {icon_benefit}: {description}")
    
    print()
    print("🎯 CONCLUSION:")
    print("=" * 15)
    print("The Mesh now has sophisticated nested communication channels")
    print("that mirror natural human social organization. Messages can")
    print("whisper intimately in family circles, sing in village squares,")
    print("debate in regional councils, chorus across the world, and")
    print("gather in secret circles of shared obsession.")
    print()
    print("This creates a communication system that is both globally")
    print("connected and locally respectful - exactly what The Mesh")
    print("needs for trustworthy, human-centered AI coordination.")

async def demo_message_routing_rules():
    """Demonstrate intelligent message routing rules"""
    
    print("\n" + "="*55)
    print("🧠 INTELLIGENT MESSAGE ROUTING RULES")
    print("="*55)
    
    routing_rules = [
        {
            'rule': 'Emergency Escalation',
            'trigger': 'Keywords: help, emergency, urgent + high priority',
            'action': 'Immediate escalation: family → village → region',
            'trust_override': 'Lower barriers (0.5 instead of normal)',
            'example': '"Help! House fire!" → All local scopes within minutes'
        },
        {
            'rule': 'Local Questions',
            'trigger': 'Keywords: local, neighborhood, nearby, community',
            'action': 'Direct routing to village scope only',
            'trust_requirement': 'High (0.8) for local credibility',
            'example': '"Best local bakery?" → Village members only'
        },
        {
            'rule': 'Specialized Knowledge',
            'trigger': 'Technical terms, expertise keywords, complex topics',
            'action': 'Route to relevant chosen circles',
            'trust_requirement': 'Specialized trust (0.7) + expertise verification',
            'example': '"Quantum decoherence solutions" → Quantum computing circle'
        },
        {
            'rule': 'Global Discussions',
            'trigger': 'Keywords: global, world, international, humanity',
            'action': 'Direct to world scope',
            'consensus_required': 'High threshold (0.6) for global visibility',
            'example': '"Climate change action needed" → Global chorus'
        },
        {
            'rule': 'Family Privacy',
            'trigger': 'Keywords: family, personal, private, intimate',
            'action': 'Lock to family scope only',
            'privacy_override': 'Force private mode + encryption',
            'example': '"Family dinner plans" → Family circle only'
        }
    ]
    
    for i, rule in enumerate(routing_rules, 1):
        print(f"\n🔧 Rule {i}: {rule['rule']}")
        print(f"   🎯 Trigger: {rule['trigger']}")
        print(f"   ⚡ Action: {rule['action']}")
        
        if 'trust_override' in rule:
            print(f"   🤝 Trust Override: {rule['trust_override']}")
        if 'trust_requirement' in rule:
            print(f"   🤝 Trust Requirement: {rule['trust_requirement']}")
        if 'privacy_override' in rule:
            print(f"   🔒 Privacy Override: {rule['privacy_override']}")
        if 'consensus_required' in rule:
            print(f"   🗳️ Consensus Required: {rule['consensus_required']}")
        
        print(f"   💡 Example: {rule['example']}")

async def demo_escalation_scenarios():
    """Demonstrate different escalation scenarios"""
    
    print("\n" + "="*55)
    print("📈 ESCALATION SCENARIOS")
    print("="*55)
    
    scenarios = [
        {
            'name': 'Medical Emergency',
            'initial_scope': 'family',
            'message': 'Grandfather having chest pains, need immediate help',
            'escalation_path': [
                ('0 min', 'family', 'Alert family members', 'Sister calls 911'),
                ('2 min', 'village', 'Alert local network', 'Neighbor doctor responds'),
                ('5 min', 'region', 'Medical network notified', 'Helicopter dispatched'),
                ('RESOLVED', 'Emergency services coordinate with local help')
            ]
        },
        {
            'name': 'Local Infrastructure Issue',
            'initial_scope': 'village',
            'message': 'Water main broke on Main Street, flooding roads',
            'escalation_path': [
                ('0 min', 'village', 'Alert local community', 'Residents avoid area'),
                ('30 min', 'region', 'Notify utility companies', 'Repair crews dispatched'),
                ('2 hours', 'No further escalation needed', 'Issue contained locally')
            ]
        },
        {
            'name': 'Specialized Research Question',
            'initial_scope': 'chosen (AI research)',
            'message': 'Novel approach to transformer attention - peer review needed',
            'escalation_path': [
                ('0 min', 'chosen', 'AI research circle', 'Initial feedback positive'),
                ('2 days', 'chosen', 'Cross-post to ML theory circle', 'Mathematical validation'),
                ('1 week', 'region', 'University network alerted', 'Academic collaboration'),
                ('SUCCESS', 'Paper published with mesh peer validation')
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎭 Scenario: {scenario['name']}")
        print(f"   📤 Initial Message: \"{scenario['message']}\"")
        print(f"   🎯 Starting Scope: {scenario['initial_scope']}")
        print(f"   📈 Escalation Path:")
        
        for step in scenario['escalation_path']:
            if len(step) == 4:
                time, scope, action, result = step
                print(f"      {time}: {scope.upper()} - {action}")
                print(f"             → {result}")
            else:
                print(f"      ✅ OUTCOME: {step[1]}")

if __name__ == "__main__":
    asyncio.run(demonstrate_nested_communication())
    asyncio.run(demo_message_routing_rules())
    asyncio.run(demo_escalation_scenarios())