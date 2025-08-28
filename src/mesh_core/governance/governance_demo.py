"""
Mesh Governance System Demo
===========================

Comprehensive demonstration of The Mesh's governance and constitutional layer,
showing how constitutional rules, enforcement, amendments, local customization,
and rights protection work together.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleType, RulePriority
from .protocol_enforcer import ProtocolEnforcer, EnforcementAction
from .amendment_system import AmendmentSystem, ProposalStatus, VoteType, VotingMechanism
from .local_customizer import LocalCustomizer, CustomizationScope, CustomizationType
from .rights_framework import RightsFramework, RightType, RightStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GovernanceDemo:
    """Demonstrates the complete governance system"""
    
    def __init__(self, node_id: str = "demo_node"):
        self.node_id = node_id
        
        # Initialize all governance components
        logger.info("🚀 Initializing Mesh Governance System...")
        
        self.constitution = ConstitutionEngine(node_id)
        self.enforcer = ProtocolEnforcer(self.constitution, node_id)
        self.amendment_system = AmendmentSystem(self.constitution, node_id)
        self.customizer = LocalCustomizer(self.constitution, node_id)
        self.rights_framework = RightsFramework(self.constitution, node_id)
        
        logger.info("✅ Governance System initialized successfully!")
    
    async def run_complete_demo(self):
        """Run the complete governance system demonstration"""
        logger.info("\n" + "="*80)
        logger.info("🏛️  MESH GOVERNANCE SYSTEM COMPLETE DEMONSTRATION")
        logger.info("="*80)
        
        try:
            # 1. Constitutional Rules Demonstration
            await self._demo_constitutional_rules()
            
            # 2. Protocol Enforcement Demonstration
            await self._demo_protocol_enforcement()
            
            # 3. Amendment System Demonstration
            await self._demo_amendment_system()
            
            # 4. Local Customization Demonstration
            await self._demo_local_customization()
            
            # 5. Rights Framework Demonstration
            await self._demo_rights_framework()
            
            # 6. Integration Demonstration
            await self._demo_integration()
            
            # 7. System Summary
            self._show_system_summary()
            
            logger.info("\n🎉 Governance System Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def _demo_constitutional_rules(self):
        """Demonstrate constitutional rules management"""
        logger.info("\n📜 PHASE 1: Constitutional Rules Management")
        logger.info("-" * 50)
        
        # Show initial constitution
        logger.info("📋 Initial Constitution State:")
        summary = self.constitution.get_constitution_summary()
        logger.info(f"   • Total Rules: {summary['total_rules']}")
        logger.info(f"   • Active Rules: {summary['active_rules']}")
        logger.info(f"   • Rule Types: {summary['rule_types']}")
        
        # Add a new rule
        logger.info("\n➕ Adding New Constitutional Rule...")
        new_rule = ConstitutionalRule(
            rule_id="",
            rule_type=RuleType.BEHAVIORAL,
            priority=RulePriority.HIGH,
            title="Content Quality Standards",
            description="All shared content must meet minimum quality standards",
            constraints={"min_quality_score": 0.7, "moderation_required": True},
            enforcement_mechanism="quality_gate",
            created_at=datetime.utcnow(),
            created_by=self.node_id
        )
        
        success = self.constitution.add_rule(new_rule)
        if success:
            logger.info(f"   ✅ Added rule: {new_rule.title}")
        else:
            logger.info(f"   ❌ Failed to add rule: {new_rule.title}")
        
        # Show updated constitution
        updated_summary = self.constitution.get_constitution_summary()
        logger.info(f"\n📊 Updated Constitution: {updated_summary['active_rules']} active rules")
        
        # Test compliance checking
        logger.info("\n🔍 Testing Compliance Checking...")
        test_context = {"quality_score": 0.8, "moderated": True}
        is_compliant, violations = self.constitution.check_compliance("test_user", "share_content", test_context)
        
        if is_compliant:
            logger.info("   ✅ Test action is compliant with constitution")
        else:
            logger.info(f"   ❌ Test action violates constitution: {violations}")
    
    async def _demo_protocol_enforcement(self):
        """Demonstrate protocol enforcement"""
        logger.info("\n⚖️  PHASE 2: Protocol Enforcement")
        logger.info("-" * 50)
        
        # Test enforcement on compliant action
        logger.info("✅ Testing Compliant Action...")
        compliant_context = {"verified": True, "confidence": 0.9}
        is_compliant, actions = await self.enforcer.enforce_compliance("good_user", "share_verified_info", compliant_context)
        
        if is_compliant:
            logger.info("   ✅ Action compliant - no enforcement needed")
        else:
            logger.info(f"   ❌ Action non-compliant - enforcement actions: {actions}")
        
        # Test enforcement on non-compliant action
        logger.info("\n❌ Testing Non-Compliant Action...")
        non_compliant_context = {"verified": False, "confidence": 0.3}
        is_compliant, actions = await self.enforcer.enforce_compliance("bad_user", "share_unverified_info", non_compliant_context)
        
        if is_compliant:
            logger.info("   ✅ Action compliant - no enforcement needed")
        else:
            logger.info(f"   ❌ Action non-compliant - enforcement actions: {actions}")
        
        # Show enforcement summary
        enforcement_summary = self.enforcer.get_enforcement_summary()
        logger.info(f"\n📊 Enforcement Summary:")
        logger.info(f"   • Total Enforcements: {enforcement_summary['total_enforcements']}")
        logger.info(f"   • Active Enforcements: {enforcement_summary['active_enforcements']}")
        logger.info(f"   • Enforcement Types: {enforcement_summary['enforcement_by_type']}")
    
    async def _demo_amendment_system(self):
        """Demonstrate the amendment system"""
        logger.info("\n🗳️  PHASE 3: Constitutional Amendment System")
        logger.info("-" * 50)
        
        # Create a proposal
        logger.info("📝 Creating Amendment Proposal...")
        proposed_changes = {
            "add_rule": {
                "rule_type": "behavioral",
                "priority": "medium",
                "title": "Community Contribution Requirements",
                "description": "Nodes must contribute to community knowledge base",
                "constraints": {"min_contributions": 1, "contribution_quality": 0.6},
                "enforcement_mechanism": "reputation_tracking"
            }
        }
        
        proposal_id = self.amendment_system.create_proposal(
            title="Add Community Contribution Rule",
            description="Require nodes to contribute to community knowledge",
            proposed_changes=proposed_changes,
            proposer_id=self.node_id,
            category="behavioral",
            urgency="normal"
        )
        
        if proposal_id:
            logger.info(f"   ✅ Created proposal: {proposal_id}")
            
            # Submit proposal
            if self.amendment_system.submit_proposal(proposal_id):
                logger.info("   ✅ Proposal submitted for review")
                
                # Start review
                if self.amendment_system.start_review(proposal_id):
                    logger.info("   ✅ Review period started")
                    
                    # Start voting
                    if self.amendment_system.start_voting(proposal_id):
                        logger.info("   ✅ Voting period started")
                        
                        # Cast votes
                        logger.info("   🗳️  Casting votes...")
                        
                        # Vote 1: Approve
                        self.amendment_system.cast_vote(
                            proposal_id, "voter_1", VoteType.APPROVE,
                            reasoning="Good for community engagement"
                        )
                        
                        # Vote 2: Approve
                        self.amendment_system.cast_vote(
                            proposal_id, "voter_2", VoteType.APPROVE,
                            reasoning="Encourages participation"
                        )
                        
                        # Vote 3: Abstain
                        self.amendment_system.cast_vote(
                            proposal_id, "voter_3", VoteType.ABSTAIN,
                            reasoning="Need more information"
                        )
                        
                        # Check voting results
                        results = self.amendment_system.check_voting_results(proposal_id)
                        if results:
                            logger.info(f"   📊 Voting finalized: {results['outcome']}")
                        else:
                            logger.info("   ⏳ Voting still in progress")
                    else:
                        logger.info("   ❌ Failed to start voting")
                else:
                    logger.info("   ❌ Failed to start review")
            else:
                logger.info("   ❌ Failed to submit proposal")
        else:
            logger.info("   ❌ Failed to create proposal")
        
        # Show proposal summary
        proposal_summary = self.amendment_system.get_proposal_summary()
        logger.info(f"\n📊 Amendment System Summary:")
        logger.info(f"   • Total Proposals: {proposal_summary['total_proposals']}")
        logger.info(f"   • Active Voting: {proposal_summary['active_voting']}")
        logger.info(f"   • Proposals by Status: {proposal_summary['proposals_by_status']}")
    
    async def _demo_local_customization(self):
        """Demonstrate local customization"""
        logger.info("\n🏘️  PHASE 4: Local Customization")
        logger.info("-" * 50)
        
        # Create a community profile
        logger.info("🏘️  Creating Community Profile...")
        community_id = self.customizer.create_community_profile(
            name="Academic Research Community",
            description="Community focused on academic research and peer review",
            governance_style="consensus",
            compatibility_level="full"
        )
        
        if community_id:
            logger.info(f"   ✅ Created community: {community_id}")
            
            # Create a local customization
            logger.info("\n🔧 Creating Local Customization...")
            custom_content = {
                "rule_type": "behavioral",
                "priority": "high",
                "title": "Academic Citation Standards",
                "description": "All claims must include academic citations",
                "constraints": {"citation_required": True, "min_citations": 2},
                "enforcement_mechanism": "citation_verification"
            }
            
            customization_id = self.customizer.create_customization(
                scope=CustomizationScope.COMMUNITY,
                customization_type=CustomizationType.RULE_ADDITION,
                title="Academic Citation Standards",
                description="Require academic citations for all claims",
                custom_content=custom_content,
                community_id=community_id,
                tags=["academic", "research", "citations"]
            )
            
            if customization_id:
                logger.info(f"   ✅ Created customization: {customization_id}")
                
                # Apply the customization
                if self.customizer.apply_customization(customization_id):
                    logger.info("   ✅ Customization applied to constitution")
                else:
                    logger.info("   ❌ Failed to apply customization")
            else:
                logger.info("   ❌ Failed to create customization")
        else:
            logger.info("   ❌ Failed to create community profile")
        
        # Show customization summary
        customization_summary = self.customizer.get_customization_summary()
        logger.info(f"\n📊 Local Customization Summary:")
        logger.info(f"   • Total Customizations: {customization_summary['total_customizations']}")
        logger.info(f"   • Active Customizations: {customization_summary['active_customizations']}")
        logger.info(f"   • Compatible Customizations: {customization_summary['compatible_customizations']}")
        logger.info(f"   • Communities: {customization_summary['communities']}")
    
    async def _demo_rights_framework(self):
        """Demonstrate the rights framework"""
        logger.info("\n⚖️  PHASE 5: Rights Framework")
        logger.info("-" * 50)
        
        # Grant additional rights
        logger.info("🎁 Granting Additional Rights...")
        from .rights_framework import UserRight
        
        expression_right = UserRight(
            right_id="",
            right_type=RightType.EXPRESSION,
            user_id="demo_user",
            status=RightStatus.ACTIVE,
            granted_at=datetime.utcnow(),
            granted_by=self.node_id,
            scope="full"
        )
        
        if self.rights_framework.grant_right(expression_right):
            logger.info(f"   ✅ Granted {expression_right.right_type.value} right to demo_user")
        else:
            logger.info(f"   ❌ Failed to grant {expression_right.right_type.value} right")
        
        # Test rights verification
        logger.info("\n🔍 Testing Rights Verification...")
        can_exercise, reason = self.rights_framework.verify_right("demo_user", RightType.EXPRESSION)
        
        if can_exercise:
            logger.info("   ✅ User can exercise expression right")
        else:
            logger.info(f"   ❌ User cannot exercise expression right: {reason}")
        
        # Test rights verification with conditions
        logger.info("\n⏰ Testing Rights Verification with Time Conditions...")
        time_restricted_right = UserRight(
            right_id="",
            right_type=RightType.ASSOCIATION,
            user_id="demo_user",
            status=RightStatus.ACTIVE,
            granted_at=datetime.utcnow(),
            granted_by=self.node_id,
            scope="conditional",
            conditions={
                "time_restrictions": [
                    {"start": "09:00", "end": "17:00"},
                    {"start": "19:00", "end": "22:00"}
                ]
            }
        )
        
        if self.rights_framework.grant_right(time_restricted_right):
            logger.info("   ✅ Granted time-restricted association right")
            
            # Test during allowed time
            current_time = datetime.utcnow().time()
            context = {"location": "office"}
            can_exercise, reason = self.rights_framework.verify_right("demo_user", RightType.ASSOCIATION, context)
            
            if can_exercise:
                logger.info("   ✅ User can exercise association right during current time")
            else:
                logger.info(f"   ❌ User cannot exercise association right: {reason}")
        else:
            logger.info("   ❌ Failed to grant time-restricted right")
        
        # Submit a rights claim
        logger.info("\n📋 Submitting Rights Claim...")
        claim_id = self.rights_framework.submit_rights_claim(
            right_id="new_right",
            user_id="claimant_user",
            claim_type="access_request",
            description="Request access to advanced features",
            evidence={"user_level": "advanced", "contribution_score": 0.9}
        )
        
        if claim_id:
            logger.info(f"   ✅ Submitted rights claim: {claim_id}")
            
            # Review the claim
            if self.rights_framework.review_rights_claim(claim_id, "approved", "reviewer_admin", "User meets requirements"):
                logger.info("   ✅ Rights claim approved")
            else:
                logger.info("   ❌ Failed to review rights claim")
        else:
            logger.info("   ❌ Failed to submit rights claim")
        
        # Show rights summary
        rights_summary = self.rights_framework.get_rights_framework_summary()
        logger.info(f"\n📊 Rights Framework Summary:")
        logger.info(f"   • Total Users: {rights_summary['total_users']}")
        logger.info(f"   • Total Rights: {rights_summary['total_rights']}")
        logger.info(f"   • Total Violations: {rights_summary['total_violations']}")
        logger.info(f"   • Total Claims: {rights_summary['total_claims']}")
    
    async def _demo_integration(self):
        """Demonstrate integration between all components"""
        logger.info("\n🔗 PHASE 6: System Integration")
        logger.info("-" * 50)
        
        # Test constitutional compliance with enforcement
        logger.info("🔍 Testing Constitutional Compliance with Enforcement...")
        test_context = {"quality_score": 0.5, "moderated": False}  # Non-compliant
        
        # Check compliance
        is_compliant, violations = self.constitution.check_compliance("integration_test_user", "share_content", test_context)
        
        if not is_compliant:
            logger.info(f"   ❌ Action non-compliant: {violations}")
            
            # Enforce compliance
            enforcement_result, actions = await self.enforcer.enforce_compliance("integration_test_user", "share_content", test_context)
            
            if not enforcement_result:
                logger.info(f"   ⚖️  Enforcement actions taken: {actions}")
            else:
                logger.info("   ✅ Action became compliant after enforcement")
        else:
            logger.info("   ✅ Action compliant")
        
        # Test rights verification with constitutional rules
        logger.info("\n⚖️  Testing Rights with Constitutional Rules...")
        can_exercise, reason = self.rights_framework.verify_right("demo_user", RightType.EXPRESSION)
        
        if can_exercise:
            # Check if exercising the right complies with constitution
            exercise_context = {"content_type": "opinion", "moderated": True}
            is_constitutional, rule_violations = self.constitution.check_compliance("demo_user", "exercise_expression", exercise_context)
            
            if is_constitutional:
                logger.info("   ✅ Right exercise is constitutional")
            else:
                logger.info(f"   ❌ Right exercise violates constitution: {rule_violations}")
        else:
            logger.info(f"   ❌ User cannot exercise right: {reason}")
        
        # Test local customization with rights
        logger.info("\n🏘️  Testing Local Customization with Rights...")
        if hasattr(self.customizer, 'customizations') and self.customizer.customizations:
            # Get first customization
            first_customization = list(self.customizer.customizations.values())[0]
            
            # Check if it affects user rights
            if first_customization.customization_type == CustomizationType.RULE_ADDITION:
                logger.info(f"   🔧 Local customization: {first_customization.title}")
                logger.info(f"   📋 Type: {first_customization.customization_type.value}")
                logger.info(f"   ✅ Mesh compatible: {first_customization.mesh_compatibility}")
        
        logger.info("   🔗 All governance components integrated and working together!")
    
    def _show_system_summary(self):
        """Show comprehensive system summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 GOVERNANCE SYSTEM COMPREHENSIVE SUMMARY")
        logger.info("="*80)
        
        # Constitution summary
        constitution_summary = self.constitution.get_constitution_summary()
        logger.info(f"📜 Constitution Engine:")
        logger.info(f"   • Total Rules: {constitution_summary['total_rules']}")
        logger.info(f"   • Active Rules: {constitution_summary['active_rules']}")
        logger.info(f"   • Constitution Hash: {constitution_summary['constitution_hash'][:16]}...")
        
        # Enforcement summary
        enforcement_summary = self.enforcer.get_enforcement_summary()
        logger.info(f"\n⚖️  Protocol Enforcer:")
        logger.info(f"   • Total Enforcements: {enforcement_summary['total_enforcements']}")
        logger.info(f"   • Active Enforcements: {enforcement_summary['active_enforcements']}")
        logger.info(f"   • Enforcement Types: {enforcement_summary['enforcement_by_type']}")
        
        # Amendment summary
        amendment_summary = self.amendment_system.get_proposal_summary()
        logger.info(f"\n🗳️  Amendment System:")
        logger.info(f"   • Total Proposals: {amendment_summary['total_proposals']}")
        logger.info(f"   • Active Voting: {amendment_summary['active_voting']}")
        logger.info(f"   • Proposals by Status: {amendment_summary['proposals_by_status']}")
        
        # Customization summary
        customization_summary = self.customizer.get_customization_summary()
        logger.info(f"\n🏘️  Local Customizer:")
        logger.info(f"   • Total Customizations: {customization_summary['total_customizations']}")
        logger.info(f"   • Active Customizations: {customization_summary['active_customizations']}")
        logger.info(f"   • Communities: {customization_summary['communities']}")
        
        # Rights summary
        rights_summary = self.rights_framework.get_rights_framework_summary()
        logger.info(f"\n⚖️  Rights Framework:")
        logger.info(f"   • Total Users: {rights_summary['total_users']}")
        logger.info(f"   • Total Rights: {rights_summary['total_rights']}")
        logger.info(f"   • Total Violations: {rights_summary['total_violations']}")
        logger.info(f"   • Total Claims: {rights_summary['total_claims']}")
        
        logger.info("\n🎯 GOVERNANCE SYSTEM STATUS: FULLY OPERATIONAL")
        logger.info("="*80)


async def main():
    """Main demo function"""
    try:
        # Create and run the demo
        demo = GovernanceDemo("demo_governance_node")
        await demo.run_complete_demo()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())

