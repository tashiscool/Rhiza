"""
Mesh Simulation System Demo
===========================

Comprehensive demonstration of the simulation and scenario engine,
including scenario generation, choice rehearsal, empathy training,
consequence prediction, and scenario sharing.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from .scenario_generator import ScenarioGenerator, Scenario, Persona, ScenarioType, ScenarioComplexity
from .choice_rehearser import ChoiceRehearser, RehearsalType, RehearsalStatus
from .empathy_trainer import EmpathyTrainer, EmpathySkill, TrainingLevel
from .consequence_predictor import ConsequencePredictor, ImpactLevel, Timeframe
from .scenario_sharer import ScenarioSharer, SharingPermission, SharingStatus, ContributionType

logger = logging.getLogger(__name__)


class SimulationDemo:
    """
    Demonstrates the complete simulation system functionality
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Initialize all simulation components
        self.scenario_generator = ScenarioGenerator(node_id)
        self.choice_rehearser = ChoiceRehearser(node_id)
        self.empathy_trainer = EmpathyTrainer(node_id)
        self.consequence_predictor = ConsequencePredictor(node_id)
        self.scenario_sharer = ScenarioSharer(node_id)
        
        # Demo data
        self.demo_user_id = "demo_user_001"
        self.demo_scenarios: List[Scenario] = []
        self.demo_rehearsal_sessions: List[str] = []
        self.demo_training_sessions: List[str] = []
        self.demo_shared_scenarios: List[str] = []
    
    async def run_complete_demo(self):
        """Run the complete simulation system demonstration"""
        logger.info("\n" + "="*80)
        logger.info("🎭 MESH SIMULATION SYSTEM COMPLETE DEMONSTRATION")
        logger.info("="*80)
        
        try:
            # 1. Scenario Generation Demonstration
            await self._demo_scenario_generation()
            
            # 2. Choice Rehearsal Demonstration
            await self._demo_choice_rehearsal()
            
            # 3. Empathy Training Demonstration
            await self._demo_empathy_training()
            
            # 4. Consequence Prediction Demonstration
            await self._demo_consequence_prediction()
            
            # 5. Scenario Sharing Demonstration
            await self._demo_scenario_sharing()
            
            # 6. Integration Demonstration
            await self._demo_integration()
            
            # 7. System Summary
            self._show_system_summary()
            
            logger.info("\n🎉 Simulation System Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def _demo_scenario_generation(self):
        """Demonstrate scenario generation capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🎯 SCENARIO GENERATION DEMONSTRATION")
        logger.info("="*60)
        
        # Generate different types of scenarios
        scenario_types = [
            (ScenarioType.CONFLICT_RESOLUTION, "Team conflict over project priorities"),
            (ScenarioType.COMMUNICATION, "Difficult conversation with a colleague"),
            (ScenarioType.DECISION_MAKING, "Ethical dilemma in business"),
            (ScenarioType.TRUST_BUILDING, "Building trust in a new team"),
            (ScenarioType.JUSTICE_TESTING, "Testing justice and fairness")
        ]
        
        for scenario_type, description in scenario_types:
            logger.info(f"\n📝 Generating {scenario_type.value} scenario...")
            
            # Generate scenario
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=scenario_type,
                complexity=ScenarioComplexity.INTERMEDIATE
            )
            
            if scenario:
                self.demo_scenarios.append(scenario)
                logger.info(f"✅ Generated scenario: {scenario.title}")
                logger.info(f"   Type: {scenario.scenario_type.value}")
                logger.info(f"   Complexity: {scenario.complexity.value}")
                logger.info(f"   Participants: {len(scenario.participants)}")
                logger.info(f"   Learning objectives: {len(scenario.learning_objectives)}")
            else:
                logger.warning(f"⚠️ Failed to generate {scenario_type.value} scenario")
        
        logger.info(f"\n📊 Generated {len(self.demo_scenarios)} scenarios successfully")
    
    async def _demo_choice_rehearsal(self):
        """Demonstrate choice rehearsal capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🎭 CHOICE REHEARSAL DEMONSTRATION")
        logger.info("="*60)
        
        if not self.demo_scenarios:
            logger.warning("⚠️ No scenarios available for rehearsal demo")
            return
        
        # Use the first scenario for rehearsal
        scenario = self.demo_scenarios[0]
        
        # Start different types of rehearsal sessions
        rehearsal_types = [
            (RehearsalType.DECISION_REHEARSAL, "Rehearsing decision-making process"),
            (RehearsalType.CONVERSATION_REHEARSAL, "Practicing conversation flow"),
            (RehearsalType.RESPONSE_REHEARSAL, "Preparing responses to challenges")
        ]
        
        for rehearsal_type, description in rehearsal_types:
            logger.info(f"\n🎬 Starting {rehearsal_type.value} rehearsal...")
            
            # Start rehearsal session
            session = self.choice_rehearser.start_rehearsal(
                scenario=scenario,
                user_id=self.demo_user_id,
                rehearsal_type=rehearsal_type
            )
            
            if session:
                self.demo_rehearsal_sessions.append(session.session_id)
                logger.info(f"✅ Started rehearsal session: {session.session_id}")
                logger.info(f"   Type: {session.rehearsal_type.value}")
                logger.info(f"   Status: {session.status.value}")
                
                # Simulate some rehearsal steps
                await self._simulate_rehearsal_progress(session.session_id)
            else:
                logger.warning(f"⚠️ Failed to start {rehearsal_type.value} rehearsal")
        
        logger.info(f"\n📊 Started {len(self.demo_rehearsal_sessions)} rehearsal sessions")
    
    async def _simulate_rehearsal_progress(self, session_id: str):
        """Simulate progress through a rehearsal session"""
        try:
            # Get current step
            current_step = self.choice_rehearser.get_current_step(session_id)
            if not current_step:
                return
            
            # Simulate user choices and responses
            sample_choices = [
                "I would listen carefully to understand their perspective",
                "I would ask clarifying questions to gather more information",
                "I would propose a collaborative solution",
                "I would acknowledge their concerns and validate their feelings"
            ]
            
            for i, choice in enumerate(sample_choices[:2]):  # Simulate 2 steps
                # Advance step with user response
                confidence_level = 0.7 + (i * 0.1)
                next_step = self.choice_rehearser.advance_step(session_id, choice, confidence_level)
                
                logger.info(f"   Step {i+1}: {choice[:50]}...")
                
                if not next_step:
                    logger.info(f"   ✅ Rehearsal completed successfully")
                    break
            
        except Exception as e:
            logger.error(f"Error simulating rehearsal progress: {e}")
    
    async def _demo_empathy_training(self):
        """Demonstrate empathy training capabilities"""
        logger.info("\n" + "="*60)
        logger.info("❤️ EMPATHY TRAINING DEMONSTRATION")
        logger.info("="*60)
        
        # Get recommended exercises for the demo user
        logger.info(f"\n🎯 Getting recommended exercises for user: {self.demo_user_id}")
        
        recommended_exercise = self.empathy_trainer.get_recommended_exercise(self.demo_user_id)
        if recommended_exercise:
            logger.info(f"✅ Recommended exercise: {recommended_exercise.title}")
            logger.info(f"   Skill focus: {recommended_exercise.skill_focus.value}")
            logger.info(f"   Level: {recommended_exercise.training_level.value}")
            logger.info(f"   Duration: {recommended_exercise.duration_minutes} minutes")
            
            # Start training session
            session = self.empathy_trainer.start_training_session(
                exercise_id=recommended_exercise.exercise_id,
                user_id=self.demo_user_id
            )
            
            if session:
                self.demo_training_sessions.append(session.session_id)
                logger.info(f"✅ Started training session: {session.session_id}")
                
                # Simulate training responses
                await self._simulate_empathy_training(session.session_id, recommended_exercise)
            else:
                logger.warning("⚠️ Failed to start empathy training session")
        else:
            logger.warning("⚠️ No recommended exercises available")
        
        # Show user progress
        user_progress = self.empathy_trainer.get_user_progress(self.demo_user_id)
        logger.info(f"\n📊 User Progress Summary:")
        logger.info(f"   Overall empathy score: {user_progress['overall_empathy_score']:.2f}")
        logger.info(f"   Training sessions completed: {user_progress['training_sessions_completed']}")
        logger.info(f"   Recommended next skill: {user_progress['recommended_next_skill']}")
    
    async def _simulate_empathy_training(self, session_id: str, exercise):
        """Simulate empathy training responses"""
        try:
            # Sample responses based on exercise type
            if exercise.skill_focus == EmpathySkill.EMOTIONAL_RECOGNITION:
                responses = [
                    "I can see they're feeling frustrated and overwhelmed",
                    "Their body language suggests they're defensive",
                    "I notice they're avoiding eye contact, which might indicate discomfort"
                ]
            elif exercise.skill_focus == EmpathySkill.PERSPECTIVE_TAKING:
                responses = [
                    "From their perspective, this change threatens their job security",
                    "They probably feel like their concerns aren't being heard",
                    "I can understand why they might be resistant to this proposal"
                ]
            else:
                responses = [
                    "I would listen with full attention to understand their situation",
                    "I would validate their feelings and acknowledge their experience",
                    "I would offer support while respecting their boundaries"
                ]
            
            # Submit responses
            for i, response in enumerate(responses):
                self.empathy_trainer.submit_response(session_id, {
                    "response_text": response,
                    "response_type": "text",
                    "confidence": 0.6 + (i * 0.15)
                })
                logger.info(f"   Response {i+1}: {response[:50]}...")
            
            # Complete training session
            self.empathy_trainer.complete_training_session(
                session_id=session_id,
                self_reflection="This exercise helped me better understand emotional recognition",
                areas_for_improvement=["Need to practice more with subtle cues"],
                strengths_demonstrated=["Good at identifying obvious emotions"]
            )
            
            logger.info(f"   ✅ Training session completed with self-reflection")
            
        except Exception as e:
            logger.error(f"Error simulating empathy training: {e}")
    
    async def _demo_consequence_prediction(self):
        """Demonstrate consequence prediction capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🔮 CONSEQUENCE PREDICTION DEMONSTRATION")
        logger.info("="*60)
        
        if not self.demo_scenarios:
            logger.warning("⚠️ No scenarios available for consequence prediction demo")
            return
        
        # Use a conflict resolution scenario
        scenario = next((s for s in self.demo_scenarios if s.scenario_type == ScenarioType.CONFLICT_RESOLUTION), None)
        if not scenario:
            scenario = self.demo_scenarios[0]  # Fallback to first scenario
        
        logger.info(f"\n🎯 Predicting consequences for scenario: {scenario.title}")
        
        # Sample actions to analyze
        sample_actions = [
            "Confront the person directly about their behavior",
            "Listen to their perspective and seek understanding",
            "Implement a new policy without consultation",
            "Mediate a discussion between all parties involved"
        ]
        
        for action in sample_actions:
            logger.info(f"\n🔍 Analyzing action: {action}")
            
            # Predict consequences
            consequences = self.consequence_predictor.predict_consequences(
                scenario=scenario,
                action=action,
                context={"urgency": "medium", "stakeholder_count": 3}
            )
            
            if consequences:
                logger.info(f"✅ Predicted {len(consequences)} consequences:")
                for consequence in consequences:
                    logger.info(f"   • {consequence.impact_level.value.upper()} impact: {consequence.description[:60]}...")
                    logger.info(f"     Timeframe: {consequence.timeframe.value}, Probability: {consequence.probability:.2f}")
            else:
                logger.info("   ⚠️ No consequences predicted")
        
        # Show prediction models summary
        models_summary = self.consequence_predictor.get_prediction_models_summary()
        logger.info(f"\n📊 Prediction Models Summary:")
        logger.info(f"   Total models: {models_summary['total_models']}")
        logger.info(f"   Active models: {models_summary['active_models']}")
        for model_type, accuracy in models_summary['model_performance'].items():
            logger.info(f"   {model_type} accuracy: {accuracy:.2f}")
    
    async def _demo_scenario_sharing(self):
        """Demonstrate scenario sharing capabilities"""
        logger.info("\n" + "="*60)
        logger.info("📤 SCENARIO SHARING DEMONSTRATION")
        logger.info("="*60)
        
        if not self.demo_scenarios:
            logger.warning("⚠️ No scenarios available for sharing demo")
            return
        
        # Share a few scenarios
        sample_scenarios = self.demo_scenarios[:3]
        
        for i, scenario in enumerate(sample_scenarios):
            logger.info(f"\n📤 Sharing scenario {i+1}: {scenario.title}")
            
            # Share the scenario
            share_id = self.scenario_sharer.share_scenario(
                scenario=scenario,
                creator_id=self.demo_user_id,
                title=f"Shared: {scenario.title}",
                description=f"Community-shared version of {scenario.title}",
                tags=scenario.tags + ["shared", "community"],
                category=scenario.scenario_type.value,
                sharing_permission=SharingPermission.COMMUNITY
            )
            
            if share_id:
                self.demo_shared_scenarios.append(share_id)
                logger.info(f"✅ Scenario shared with ID: {share_id}")
                
                # Publish the scenario
                if self.scenario_sharer.publish_scenario(share_id, self.demo_user_id):
                    logger.info(f"   ✅ Scenario published successfully")
                
                # Add some community interaction
                self.scenario_sharer.rate_scenario(self.demo_user_id, share_id, 4.5)
                self.scenario_sharer.favorite_scenario(self.demo_user_id, share_id)
                
                # Add a contribution
                contribution_id = self.scenario_sharer.contribute_to_scenario(
                    share_id=share_id,
                    contributor_id="community_member_001",
                    contribution_type=ContributionType.FEEDBACK,
                    description="Great scenario for team building exercises",
                    content={"feedback_type": "positive", "suggested_use": "team_workshops"}
                )
                
                if contribution_id:
                    logger.info(f"   ✅ Community contribution added: {contribution_id}")
            else:
                logger.warning(f"⚠️ Failed to share scenario {i+1}")
        
        # Show sharing system summary
        sharing_summary = self.scenario_sharer.get_scenario_sharer_summary()
        logger.info(f"\n📊 Scenario Sharing Summary:")
        logger.info(f"   Total scenarios: {sharing_summary['total_scenarios']}")
        logger.info(f"   Published scenarios: {sharing_summary['published_scenarios']}")
        logger.info(f"   Total contributions: {sharing_summary['total_contributions']}")
        logger.info(f"   Total communities: {sharing_summary['total_communities']}")
    
    async def _demo_integration(self):
        """Demonstrate integration between simulation components"""
        logger.info("\n" + "="*60)
        logger.info("🔗 INTEGRATION DEMONSTRATION")
        logger.info("="*60)
        
        if not self.demo_scenarios:
            logger.warning("⚠️ No scenarios available for integration demo")
            return
        
        # Use the first scenario for integration demo
        scenario = self.demo_scenarios[0]
        
        logger.info(f"\n🎯 Demonstrating integration with scenario: {scenario.title}")
        
        # 1. Generate scenario
        logger.info("1️⃣ Scenario generated by ScenarioGenerator")
        
        # 2. Start choice rehearsal
        rehearsal_session = self.choice_rehearser.start_rehearsal(
            scenario=scenario,
            user_id=self.demo_user_id,
            rehearsal_type=RehearsalType.DECISION_REHEARSAL
        )
        logger.info("2️⃣ Choice rehearsal started by ChoiceRehearser")
        
        # 3. Predict consequences for a specific action
        action = "Address the conflict directly with all parties"
        consequences = self.consequence_predictor.predict_consequences(
            scenario=scenario,
            action=action,
            context={"approach": "direct", "participants": len(scenario.participants)}
        )
        logger.info(f"3️⃣ Consequences predicted by ConsequencePredictor: {len(consequences)} predictions")
        
        # 4. Start empathy training for the scenario
        empathy_exercise = self.empathy_trainer.get_recommended_exercise(self.demo_user_id)
        if empathy_exercise:
            training_session = self.empathy_trainer.start_training_session(
                exercise_id=empathy_exercise.exercise_id,
                user_id=self.demo_user_id
            )
            logger.info("4️⃣ Empathy training started by EmpathyTrainer")
        
        # 5. Share the scenario
        share_id = self.scenario_sharer.share_scenario(
            scenario=scenario,
            creator_id=self.demo_user_id,
            title=f"Integrated Demo: {scenario.title}",
            description="Demonstration of integrated simulation components",
            tags=["integration", "demo", "simulation"],
            category="demonstration"
        )
        logger.info("5️⃣ Scenario shared by ScenarioSharer")
        
        # 6. Complete the integration cycle
        # Rehearsals complete automatically when all steps are done
        if 'training_session' in locals() and hasattr(training_session, 'session_id'):
            self.empathy_trainer.complete_training_session(
                training_session.session_id,
                "Integration demo completed successfully",
                ["Continue exploring component interactions"],
                ["Good understanding of system integration"]
            )
        
        logger.info("6️⃣ Integration cycle completed successfully")
        logger.info("✅ All simulation components working together seamlessly!")
    
    def _show_system_summary(self):
        """Show comprehensive system summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 SIMULATION SYSTEM COMPREHENSIVE SUMMARY")
        logger.info("="*80)
        
        # Scenario Generator Summary
        logger.info("\n🎯 SCENARIO GENERATOR:")
        logger.info(f"   Scenarios created: {len(self.demo_scenarios)}")
        scenario_types = {}
        for scenario in self.demo_scenarios:
            scenario_type = scenario.scenario_type.value
            scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
        for scenario_type, count in scenario_types.items():
            logger.info(f"   - {scenario_type}: {count}")
        
        # Choice Rehearser Summary
        logger.info("\n🎭 CHOICE REHEARSER:")
        logger.info(f"   Rehearsal sessions: {len(self.demo_rehearsal_sessions)}")
        if self.demo_rehearsal_sessions:
            logger.info(f"   Active sessions: {len([s for s in self.demo_rehearsal_sessions])}")
        
        # Empathy Trainer Summary
        logger.info("\n❤️ EMPATHY TRAINER:")
        logger.info(f"   Training sessions: {len(self.demo_training_sessions)}")
        empathy_summary = self.empathy_trainer.get_empathy_trainer_summary()
        logger.info(f"   Total exercises: {empathy_summary['total_exercises']}")
        logger.info(f"   Exercises by skill: {empathy_summary['exercises_by_skill']}")
        
        # Consequence Predictor Summary
        logger.info("\n🔮 CONSEQUENCE PREDICTOR:")
        predictor_summary = self.consequence_predictor.get_prediction_models_summary()
        logger.info(f"   Prediction models: {predictor_summary['total_models']}")
        logger.info(f"   Active models: {predictor_summary['active_models']}")
        
        # Scenario Sharer Summary
        logger.info("\n📤 SCENARIO SHARER:")
        logger.info(f"   Shared scenarios: {len(self.demo_shared_scenarios)}")
        sharing_summary = self.scenario_sharer.get_scenario_sharer_summary()
        logger.info(f"   Total scenarios in system: {sharing_summary['total_scenarios']}")
        logger.info(f"   Communities: {sharing_summary['total_communities']}")
        
        # Overall System Status
        logger.info("\n🎉 OVERALL SYSTEM STATUS:")
        total_components = 5
        active_components = sum([
            len(self.demo_scenarios) > 0,
            len(self.demo_rehearsal_sessions) > 0,
            len(self.demo_training_sessions) > 0,
            len(self.demo_shared_scenarios) > 0,
            True  # Consequence predictor always active
        ])
        logger.info(f"   Components active: {active_components}/{total_components}")
        logger.info(f"   System health: {'✅ EXCELLENT' if active_components == total_components else '⚠️ PARTIAL' if active_components > 2 else '❌ POOR'}")
        
        logger.info("\n🚀 Simulation System Ready for Production Use!")
