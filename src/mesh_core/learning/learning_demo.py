"""
Mesh Learning System Demo
=========================

Phase 8: Autonomous Evolution & Learning
Comprehensive demonstration of the learning system components

Demonstrates continual learning, adapter management, interaction learning,
knowledge distillation, and quality assurance capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from .continual_learner import ContinualLearner, LearningType, LearningStatus
from .adapter_manager import AdapterManager, AdapterType, AdapterStatus
from .interaction_learner import InteractionLearner, InteractionType, PatternType
from .knowledge_distiller import KnowledgeDistiller, DistillationType, DistillationStatus
from .quality_assurer import QualityAssurer, QualityStatus

logger = logging.getLogger(__name__)


class LearningDemo:
    """
    Demonstrates the complete learning system functionality
    
    Shows how nodes can learn and evolve while maintaining
    alignment with community values and system stability.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Initialize all learning components
        self.continual_learner = ContinualLearner(None, None)  # Mock dependencies
        self.adapter_manager = AdapterManager(node_id)
        self.interaction_learner = InteractionLearner(node_id)
        self.knowledge_distiller = KnowledgeDistiller(node_id)
        self.quality_assurer = QualityAssurer(node_id)
        
        # Demo data
        self.demo_user_id = "demo_user_001"
        self.demo_learning_sessions: List[str] = []
        self.demo_adapters: List[str] = []
        self.demo_knowledge_packages: List[str] = []
        
        logger.info(f"LearningDemo initialized for node: {self.node_id}")
    
    async def run_complete_demo(self):
        """Run the complete learning system demonstration"""
        logger.info("\n" + "="*80)
        logger.info("🧠 MESH LEARNING SYSTEM COMPLETE DEMONSTRATION")
        logger.info("="*80)
        
        try:
            # 1. Continual Learning Demonstration
            await self._demo_continual_learning()
            
            # 2. Adapter Management Demonstration
            await self._demo_adapter_management()
            
            # 3. Interaction Learning Demonstration
            await self._demo_interaction_learning()
            
            # 4. Knowledge Distillation Demonstration
            await self._demo_knowledge_distillation()
            
            # 5. Quality Assurance Demonstration
            await self._demo_quality_assurance()
            
            # 6. Integration Demonstration
            await self._demo_integration()
            
            # 7. System Summary
            self._show_system_summary()
            
            logger.info("\n🎉 Learning System Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def _demo_continual_learning(self):
        """Demonstrate continual learning capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🧠 CONTINUAL LEARNING DEMONSTRATION")
        logger.info("="*60)
        
        # Start different types of learning sessions
        learning_types = [
            (LearningType.INTERACTION_LEARNING, "Learn from user interactions"),
            (LearningType.CORRECTION_LEARNING, "Learn from mistakes and corrections"),
            (LearningType.OPTIMIZATION_LEARNING, "Learn for performance optimization")
        ]
        
        for learning_type, description in learning_types:
            logger.info(f"\n📚 Starting {learning_type.value} session...")
            
            # Create training data
            training_data = {
                "training": [f"sample_{i}" for i in range(100)],
                "validation": [f"val_{i}" for i in range(20)],
                "test": [f"test_{i}" for i in range(10)]
            }
            
            # Start learning session
            session_id = self.continual_learner.start_learning_session(
                learning_type=learning_type,
                target_model="demo_model",
                training_data=training_data,
                learning_rate=0.001,
                max_epochs=5
            )
            
            if session_id:
                self.demo_learning_sessions.append(session_id)
                logger.info(f"✅ Started learning session: {session_id}")
                
                # Wait for learning to complete
                await asyncio.sleep(2)
                
                # Get session status
                session = self.continual_learner.active_sessions.get(session_id)
                if session:
                    logger.info(f"   Status: {session.status.value}")
                    logger.info(f"   Current epoch: {session.current_epoch}/{session.max_epochs}")
                    logger.info(f"   Loss: {session.current_loss:.4f}")
                    logger.info(f"   Validation loss: {session.validation_loss:.4f}")
            else:
                logger.warning(f"⚠️ Failed to start {learning_type.value} session")
        
        logger.info(f"\n📊 Started {len(self.demo_learning_sessions)} learning sessions")
    
    async def _demo_adapter_management(self):
        """Demonstrate adapter management capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🔧 ADAPTER MANAGEMENT DEMONSTRATION")
        logger.info("="*60)
        
        # Create different types of adapters
        adapter_types = [
            (AdapterType.LORA, "Low-Rank Adaptation for efficient fine-tuning"),
            (AdapterType.ADAPTER_FUSION, "Adapter fusion for multi-task learning"),
            (AdapterType.PREFIX_TUNING, "Prefix tuning for prompt-based learning")
        ]
        
        for adapter_type, description in adapter_types:
            logger.info(f"\n🔧 Creating {adapter_type.value} adapter...")
            
            # Create adapter configuration
            adapter_id = self.adapter_manager.create_adapter(
                adapter_type=adapter_type,
                target_model="demo_model",
                rank=16,
                alpha=32.0,
                description=description
            )
            
            if adapter_id:
                # Instantiate adapter
                instance_id = self.adapter_manager.instantiate_adapter(adapter_id)
                
                if instance_id:
                    self.demo_adapters.append(instance_id)
                    logger.info(f"✅ Created adapter: {adapter_id}")
                    logger.info(f"   Instance: {instance_id}")
                    
                    # Train the adapter
                    training_data = {"training": [f"adapter_sample_{i}" for i in range(50)]}
                    success = self.adapter_manager.train_adapter(instance_id, training_data)
                    
                    if success:
                        logger.info(f"   ✅ Adapter training completed")
                        
                        # Evaluate adapter
                        evaluation = self.adapter_manager.evaluate_adapter(instance_id)
                        if evaluation:
                            logger.info(f"   Performance score: {evaluation.get('performance_score', 0):.2f}")
                            logger.info(f"   Meets threshold: {evaluation.get('meets_threshold', False)}")
                    else:
                        logger.warning(f"   ⚠️ Adapter training failed")
                else:
                    logger.warning(f"   ⚠️ Failed to instantiate adapter")
            else:
                logger.warning(f"⚠️ Failed to create {adapter_type.value} adapter")
        
        logger.info(f"\n📊 Created {len(self.demo_adapters)} adapters")
    
    async def _demo_interaction_learning(self):
        """Demonstrate interaction learning capabilities"""
        logger.info("\n" + "="*60)
        logger.info("👥 INTERACTION LEARNING DEMONSTRATION")
        logger.info("="*60)
        
        # Record different types of user interactions
        interaction_types = [
            (InteractionType.QUERY, "What is the weather like today?", {"context": "weather_inquiry", "time": "morning"}),
            (InteractionType.FEEDBACK, "That response was very helpful", {"context": "positive_feedback", "rating": 0.9}),
            (InteractionType.CORRECTION, "Actually, that's not quite right", {"context": "correction", "severity": "minor"}),
            (InteractionType.PREFERENCE, "I prefer shorter responses", {"context": "user_preference", "category": "response_length"}),
            (InteractionType.BEHAVIOR, "User frequently asks follow-up questions", {"context": "behavior_pattern", "frequency": "high"})
        ]
        
        for interaction_type, content, context in interaction_types:
            logger.info(f"\n📝 Recording {interaction_type.value} interaction...")
            
            # Record interaction
            interaction_id = self.interaction_learner.record_interaction(
                user_id=self.demo_user_id,
                interaction_type=interaction_type,
                content=content,
                context=context,
                confidence=0.8,
                feedback_score=0.9 if interaction_type == InteractionType.FEEDBACK else None
            )
            
            if interaction_id:
                logger.info(f"✅ Recorded interaction: {interaction_id}")
                logger.info(f"   Content: {content[:50]}...")
                logger.info(f"   Context: {list(context.keys())}")
            else:
                logger.warning(f"⚠️ Failed to record {interaction_type.value} interaction")
        
        # Wait for pattern learning
        await asyncio.sleep(1)
        
        # Get learned patterns
        patterns = self.interaction_learner.get_learned_patterns()
        logger.info(f"\n📊 Learned {len(patterns)} patterns from interactions")
        
        for pattern in patterns[:3]:  # Show first 3 patterns
            logger.info(f"   Pattern: {pattern.pattern_type.value} (confidence: {pattern.confidence:.2f})")
    
    async def _demo_knowledge_distillation(self):
        """Demonstrate knowledge distillation capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🧪 KNOWLEDGE DISTILLATION DEMONSTRATION")
        logger.info("="*60)
        
        # Create different types of knowledge packages
        distillation_types = [
            (DistillationType.PATTERN_DISTILLATION, "Learned interaction patterns"),
            (DistillationType.EXPERIENCE_DISTILLATION, "User experience insights"),
            (DistillationType.RULE_DISTILLATION, "Learned behavioral rules")
        ]
        
        for distillation_type, description in distillation_types:
            logger.info(f"\n🧪 Creating {distillation_type.value} package...")
            
            # Create sample knowledge data
            knowledge_data = {
                "type": distillation_type.value,
                "description": description,
                "patterns": [f"pattern_{i}" for i in range(5)],
                "confidence_scores": [0.8, 0.7, 0.9, 0.6, 0.8],
                "metadata": {
                    "source": "interaction_learning",
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_id": self.node_id
                }
            }
            
            # Create knowledge package
            package_id = self.knowledge_distiller.create_knowledge_package(
                distillation_type=distillation_type,
                knowledge_data=knowledge_data,
                sharing_priority=7,
                expiration_date=datetime.utcnow() + timedelta(days=30)
            )
            
            if package_id:
                self.demo_knowledge_packages.append(package_id)
                logger.info(f"✅ Created knowledge package: {package_id}")
                
                # Get package details
                package = self.knowledge_distiller.knowledge_packages.get(package_id)
                if package:
                    logger.info(f"   Compression ratio: {package.compression_ratio:.2f}")
                    logger.info(f"   Quality score: {package.quality_score:.2f}")
                    logger.info(f"   Size: {len(str(package.knowledge_data))} chars")
            else:
                logger.warning(f"⚠️ Failed to create {distillation_type.value} package")
        
        # Test knowledge sharing
        if self.demo_knowledge_packages:
            logger.info(f"\n📤 Testing knowledge sharing...")
            
            target_nodes = ["node_001", "node_002", "node_003"]
            for package_id in self.demo_knowledge_packages[:2]:  # Share first 2 packages
                success = self.knowledge_distiller.share_knowledge_package(package_id, target_nodes)
                if success:
                    logger.info(f"   ✅ Shared package {package_id} with {len(target_nodes)} nodes")
                else:
                    logger.warning(f"   ⚠️ Failed to share package {package_id}")
        
        logger.info(f"\n📊 Created {len(self.demo_knowledge_packages)} knowledge packages")
    
    async def _demo_quality_assurance(self):
        """Demonstrate quality assurance capabilities"""
        logger.info("\n" + "="*60)
        logger.info("🔍 QUALITY ASSURANCE DEMONSTRATION")
        logger.info("="*60)
        
        # Assess quality for learning sessions
        if self.demo_learning_sessions:
            logger.info(f"\n🔍 Assessing quality for {len(self.demo_learning_sessions)} learning sessions...")
            
            for session_id in self.demo_learning_sessions:
                # Create sample learning outcomes
                learning_outcomes = {
                    "accuracy_improvement": 0.15,
                    "alignment_preservation": 0.85,
                    "performance_improvement": 0.12,
                    "stability_score": 0.78,
                    "safety_score": 0.92,
                    "robustness_improvement": 0.08
                }
                
                # Assess quality
                assessment_id = self.quality_assurer.assess_learning_quality(
                    learning_session_id=session_id,
                    learning_outcomes=learning_outcomes
                )
                
                if assessment_id:
                    logger.info(f"✅ Quality assessment completed: {assessment_id}")
                    
                    # Get assessment details
                    assessment = self.quality_assurer.quality_assessments.get(assessment_id)
                    if assessment:
                        logger.info(f"   Overall quality: {assessment.overall_quality:.2f}")
                        logger.info(f"   Status: {assessment.quality_status.value}")
                        logger.info(f"   Rollback recommended: {assessment.rollback_recommended}")
                        
                        if assessment.recommendations:
                            logger.info(f"   Recommendations: {assessment.recommendations[0]}")
                else:
                    logger.warning(f"⚠️ Failed to assess quality for session {session_id}")
        
        # Test quality threshold updates
        logger.info(f"\n⚙️ Testing quality threshold updates...")
        
        success = self.quality_assurer.update_quality_threshold(
            "accuracy",
            warning_threshold=0.85,
            critical_threshold=0.65
        )
        
        if success:
            logger.info("   ✅ Updated accuracy threshold successfully")
        else:
            logger.warning("   ⚠️ Failed to update accuracy threshold")
        
        # Get quality summary
        quality_summary = self.quality_assurer.get_quality_summary()
        if "error" not in quality_summary:
            logger.info(f"\n📊 Quality summary:")
            logger.info(f"   Total assessments: {quality_summary.get('total_assessments', 0)}")
            logger.info(f"   Rollback recommendations: {quality_summary.get('rollback_recommendations', 0)}")
            logger.info(f"   Quality violations: {quality_summary.get('quality_violations', 0)}")
            logger.info(f"   Active alerts: {quality_summary.get('active_alerts', 0)}")
    
    async def _demo_integration(self):
        """Demonstrate integration between learning components"""
        logger.info("\n" + "="*60)
        logger.info("🔗 LEARNING INTEGRATION DEMONSTRATION")
        logger.info("="*60)
        
        logger.info(f"\n🎯 Demonstrating integration with learning session...")
        
        if self.demo_learning_sessions:
            session_id = self.demo_learning_sessions[0]
            
            # 1. Get learning outcomes from continual learner
            session = self.continual_learner.active_sessions.get(session_id)
            if session:
                logger.info("1️⃣ Learning session completed by ContinualLearner")
                
                # 2. Assess quality
                learning_outcomes = {
                    "accuracy_improvement": session.performance_improvement_score,
                    "alignment_preservation": session.alignment_preservation_score,
                    "performance_improvement": session.performance_improvement_score,
                    "stability_score": 0.8,
                    "safety_score": 0.9,
                    "robustness_improvement": 0.1
                }
                
                assessment_id = self.quality_assurer.assess_learning_quality(
                    learning_session_id=session_id,
                    learning_outcomes=learning_outcomes
                )
                logger.info("2️⃣ Quality assessed by QualityAssurer")
                
                # 3. Create knowledge package if quality is good
                assessment = self.quality_assurer.quality_assessments.get(assessment_id)
                if assessment and assessment.overall_quality > 0.7:
                    package_id = self.knowledge_distiller.create_knowledge_package(
                        distillation_type=DistillationType.EXPERIENCE_DISTILLATION,
                        knowledge_data={
                            "learning_session_id": session_id,
                            "quality_assessment": assessment.to_dict(),
                            "learning_outcomes": learning_outcomes,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    logger.info("3️⃣ Knowledge package created by KnowledgeDistiller")
                    
                    # 4. Share knowledge if package was created
                    if package_id:
                        target_nodes = ["neighbor_001", "neighbor_002"]
                        self.knowledge_distiller.share_knowledge_package(package_id, target_nodes)
                        logger.info("4️⃣ Knowledge shared with network")
                
                logger.info("✅ Learning integration cycle completed successfully!")
            else:
                logger.warning("⚠️ No learning session available for integration demo")
    
    def _show_system_summary(self):
        """Show comprehensive learning system summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 LEARNING SYSTEM COMPREHENSIVE SUMMARY")
        logger.info("="*80)
        
        # Continual Learner Summary
        learning_summary = self.continual_learner.get_learning_summary()
        logger.info("\n🧠 CONTINUAL LEARNER:")
        logger.info(f"   Total sessions: {learning_summary['total_sessions']}")
        logger.info(f"   Successful sessions: {learning_summary['successful_sessions']}")
        logger.info(f"   Failed sessions: {learning_summary['failed_sessions']}")
        logger.info(f"   Success rate: {learning_summary['success_rate']:.2%}")
        
        # Adapter Manager Summary
        adapter_summary = self.adapter_manager.get_adapter_summary()
        logger.info("\n🔧 ADAPTER MANAGER:")
        logger.info(f"   Total configs: {adapter_summary['total_configs']}")
        logger.info(f"   Total instances: {adapter_summary['total_instances']}")
        logger.info(f"   Active adapters: {adapter_summary['active_adapters']}")
        logger.info(f"   Failed adapters: {adapter_summary['failed_adapters']}")
        
        # Interaction Learner Summary
        interaction_summary = self.interaction_learner.get_interaction_summary()
        logger.info("\n👥 INTERACTION LEARNER:")
        logger.info(f"   Total interactions: {interaction_summary['total_interactions']}")
        logger.info(f"   Total patterns: {interaction_summary['total_patterns']}")
        logger.info(f"   Patterns learned: {interaction_summary['patterns_learned']}")
        logger.info(f"   Patterns updated: {interaction_summary['patterns_updated']}")
        
        # Knowledge Distiller Summary
        knowledge_summary = self.knowledge_distiller.get_knowledge_summary()
        logger.info("\n🧪 KNOWLEDGE DISTILLER:")
        logger.info(f"   Packages created: {knowledge_summary['packages_created']}")
        logger.info(f"   Packages shared: {knowledge_summary['packages_shared']}")
        logger.info(f"   Packages received: {knowledge_summary['packages_received']}")
        logger.info(f"   Requests processed: {knowledge_summary['requests_processed']}")
        
        # Quality Assurer Summary
        quality_summary = self.quality_assurer.get_quality_summary()
        if "error" not in quality_summary:
            logger.info("\n🔍 QUALITY ASSURER:")
            logger.info(f"   Total assessments: {quality_summary['total_assessments']}")
            logger.info(f"   Rollback recommendations: {quality_summary['rollback_recommendations']}")
            logger.info(f"   Quality violations: {quality_summary['quality_violations']}")
            logger.info(f"   Active alerts: {quality_summary['active_alerts']}")
        
        # Overall System Status
        logger.info("\n🎉 OVERALL LEARNING SYSTEM STATUS:")
        total_components = 5
        active_components = sum([
            len(self.demo_learning_sessions) > 0,
            len(self.demo_adapters) > 0,
            interaction_summary['total_interactions'] > 0,
            len(self.demo_knowledge_packages) > 0,
            quality_summary.get('total_assessments', 0) > 0
        ])
        
        logger.info(f"   Components active: {active_components}/{total_components}")
        logger.info(f"   System health: {'✅ EXCELLENT' if active_components == total_components else '⚠️ PARTIAL' if active_components > 2 else '❌ POOR'}")
        
        logger.info("\n🚀 Learning System Ready for Autonomous Evolution!")


async def main():
    """Run the complete learning system demonstration"""
    # Setup logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Create and run demo
    demo = LearningDemo("phase8_demo_node")
    await demo.run_complete_demo()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
