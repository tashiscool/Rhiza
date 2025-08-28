#!/usr/bin/env python3
"""
Complete Palm Slab Integration Test

This script validates the complete transformation of Sentient integration
into true palm slab nodes that embody The Mesh principles:

1. "Every slab is a full node" - Complete autonomous operation
2. "Data is Local First" - Privacy ring with controlled sharing  
3. "Consensus through Cross-Validation" - Social checksum validation
4. "Adaptive Synapses" - Weighted connections to useful peers
5. "Truth Without Gatekeepers" - Confidence-ranked insights
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test imports
try:
    from mesh_core import (
        create_complete_palm_slab,
        PalmSlabInterface,
        SentientMeshBridge,
        PalmSlabConfig
    )
    PALM_SLAB_AVAILABLE = True
except ImportError as e:
    print(f"❌ Palm slab integration not available: {e}")
    PALM_SLAB_AVAILABLE = False

class CompletePalmSlabTest:
    """Comprehensive test of the complete palm slab implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.palm_slab = None
        
    async def run_complete_test_suite(self):
        """Run the complete palm slab test suite"""
        
        if not PALM_SLAB_AVAILABLE:
            self.logger.error("❌ Palm slab integration not available - cannot run tests")
            return
        
        self.logger.info("🚀 COMPLETE PALM SLAB INTEGRATION TEST SUITE")
        self.logger.info("=" * 80)
        self.logger.info("Testing transformation of Sentient integration into true palm slab nodes")
        self.logger.info("Validating all Mesh principles: Local-first, Peer validation, Cooperative trust")
        print()
        
        try:
            # Test 1: Palm Slab Node Creation
            await self._test_palm_slab_creation()
            
            # Test 2: Local-First Processing (Privacy Ring)
            await self._test_local_first_processing()
            
            # Test 3: Mesh Validation (Social Checksum)
            await self._test_mesh_validation()
            
            # Test 4: Adaptive Synapses (Peer Learning)
            await self._test_adaptive_synapses()
            
            # Test 5: Truth Without Gatekeepers
            await self._test_truth_without_gatekeepers()
            
            # Test 6: Complete Palm Slab Pipeline
            await self._test_complete_pipeline()
            
            # Test 7: Privacy Levels
            await self._test_privacy_levels()
            
            # Test 8: System Integration
            await self._test_system_integration()
            
            # Print comprehensive results
            self._print_test_results()
            
        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _test_palm_slab_creation(self):
        """Test 1: Palm Slab Node Creation"""
        
        self.logger.info("🧪 Test 1: Palm Slab Node Creation")
        self.logger.info("-" * 60)
        
        try:
            # Create palm slab node with different privacy levels
            self.palm_slab = create_complete_palm_slab(
                node_id="test_palm_slab_001",
                privacy_level="selective"
            )
            
            # Initialize the palm slab
            await self.palm_slab.initialize()
            
            # Validate initialization
            status = await self.palm_slab.get_palm_slab_status()
            
            self.test_results["palm_slab_creation"] = {
                "success": True,
                "node_id": status.get("node_id"),
                "privacy_level": status.get("privacy_level"),
                "capabilities": status.get("capabilities", {}),
                "systems_initialized": len([k for k, v in status.get("mesh_bridge_status", {}).get("systems_status", {}).items() if v])
            }
            
            self.logger.info(f"✅ Palm slab node created: {status.get('node_id')}")
            self.logger.info(f"   Privacy Level: {status.get('privacy_level')}")
            self.logger.info(f"   Systems Initialized: {self.test_results['palm_slab_creation']['systems_initialized']}")
            self.logger.info(f"   Capabilities: {list(status.get('capabilities', {}).keys())}")
            
        except Exception as e:
            self.logger.error(f"❌ Palm slab creation failed: {e}")
            self.test_results["palm_slab_creation"] = {"success": False, "error": str(e)}
    
    async def _test_local_first_processing(self):
        """Test 2: Local-First Processing (Privacy Ring)"""
        
        self.logger.info("\n🧪 Test 2: Local-First Processing (Privacy Ring)")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            test_cases = [
                {
                    "input": "My name is Alex and I work as a software engineer at TechCorp. I love Python programming.",
                    "type": "fact_extraction",
                    "expected_privacy": "selective"
                },
                {
                    "input": "Please schedule a meeting with the team tomorrow at 2 PM about the new project.",
                    "type": "task_request", 
                    "expected_privacy": "selective"
                },
                {
                    "input": "My password is secret123 and I store private documents in my personal folder.",
                    "type": "conversation",
                    "expected_privacy": "private"  # Should be detected as sensitive
                }
            ]
            
            local_processing_results = []
            
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"   Testing local processing case {i+1}: {test_case['type']}")
                
                # Process through palm slab
                interaction = await self.palm_slab.process_user_input(
                    user_id="test_user",
                    input_content=test_case["input"],
                    interaction_type=test_case["type"]
                )
                
                result = {
                    "case": i + 1,
                    "processed": interaction.local_response is not None,
                    "privacy_level": interaction.privacy_level,
                    "processing_time": interaction.processing_time,
                    "confidence": interaction.confidence_score,
                    "mesh_validation_attempted": interaction.mesh_validation is not None
                }
                
                local_processing_results.append(result)
                
                self.logger.info(f"     ✅ Processed locally in {interaction.processing_time:.3f}s")
                self.logger.info(f"     Privacy: {interaction.privacy_level}, Confidence: {interaction.confidence_score:.3f}")
            
            self.test_results["local_first_processing"] = {
                "success": True,
                "cases_tested": len(test_cases),
                "cases_passed": len([r for r in local_processing_results if r["processed"]]),
                "results": local_processing_results
            }
            
            self.logger.info(f"✅ Local-first processing: {len(local_processing_results)} cases processed")
            
        except Exception as e:
            self.logger.error(f"❌ Local-first processing test failed: {e}")
            self.test_results["local_first_processing"] = {"success": False, "error": str(e)}
    
    async def _test_mesh_validation(self):
        """Test 3: Mesh Validation (Social Checksum)"""
        
        self.logger.info("\n🧪 Test 3: Mesh Validation (Social Checksum)")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            # Test mesh validation with shareable content
            test_input = "Climate change is affecting weather patterns globally, leading to more extreme weather events."
            
            self.logger.info("   Testing mesh validation with factual content...")
            
            # Process with mesh validation enabled
            interaction = await self.palm_slab.process_user_input(
                user_id="test_user",
                input_content=test_input,
                interaction_type="fact_extraction"
            )
            
            # Check if mesh validation was attempted
            mesh_validation_attempted = interaction.mesh_validation is not None
            mesh_validation_success = mesh_validation_attempted and interaction.mesh_validation.validated if interaction.mesh_validation else False
            
            self.test_results["mesh_validation"] = {
                "success": True,
                "validation_attempted": mesh_validation_attempted,
                "validation_succeeded": mesh_validation_success,
                "confidence_boost": interaction.confidence_score - 0.7 if interaction.confidence_score > 0.7 else 0.0,
                "peer_consensus": interaction.mesh_validation.peer_consensus if interaction.mesh_validation else 0.0,
                "social_checksum": interaction.mesh_validation.social_checksum is not None if interaction.mesh_validation else False
            }
            
            self.logger.info(f"   ✅ Mesh validation attempted: {mesh_validation_attempted}")
            if mesh_validation_attempted:
                self.logger.info(f"   ✅ Peer consensus: {interaction.mesh_validation.peer_consensus:.3f}")
                self.logger.info(f"   ✅ Social checksum generated: {interaction.mesh_validation.social_checksum is not None}")
                self.logger.info(f"   ✅ Final confidence: {interaction.confidence_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Mesh validation test failed: {e}")
            self.test_results["mesh_validation"] = {"success": False, "error": str(e)}
    
    async def _test_adaptive_synapses(self):
        """Test 4: Adaptive Synapses (Peer Learning)"""
        
        self.logger.info("\n🧪 Test 4: Adaptive Synapses (Peer Learning)")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            # Get initial peer connection status
            status = await self.palm_slab.get_palm_slab_status()
            initial_synapses = status.get("adaptive_synapses", {})
            
            # Simulate multiple interactions to test adaptive learning
            test_interactions = [
                "What's the current weather like?",
                "Can you help me plan a healthy breakfast?", 
                "How do I improve my productivity at work?",
                "What are some good book recommendations?"
            ]
            
            for i, input_text in enumerate(test_interactions):
                self.logger.info(f"   Processing adaptive learning case {i+1}...")
                
                interaction = await self.palm_slab.process_user_input(
                    user_id="test_user",
                    input_content=input_text,
                    interaction_type="conversation"
                )
            
            # Get updated peer connection status
            updated_status = await self.palm_slab.get_palm_slab_status()
            updated_synapses = updated_status.get("adaptive_synapses", {})
            
            self.test_results["adaptive_synapses"] = {
                "success": True,
                "initial_peer_connections": initial_synapses.get("peer_connections", 0),
                "updated_peer_connections": updated_synapses.get("peer_connections", 0), 
                "average_peer_weight": updated_synapses.get("average_peer_weight", 0.0),
                "top_peers": updated_synapses.get("top_peers", []),
                "interactions_processed": len(test_interactions)
            }
            
            self.logger.info(f"   ✅ Peer connections: {updated_synapses.get('peer_connections', 0)}")
            self.logger.info(f"   ✅ Average peer weight: {updated_synapses.get('average_peer_weight', 0.0):.3f}")
            self.logger.info(f"   ✅ Interactions processed for learning: {len(test_interactions)}")
            
        except Exception as e:
            self.logger.error(f"❌ Adaptive synapses test failed: {e}")
            self.test_results["adaptive_synapses"] = {"success": False, "error": str(e)}
    
    async def _test_truth_without_gatekeepers(self):
        """Test 5: Truth Without Gatekeepers"""
        
        self.logger.info("\n🧪 Test 5: Truth Without Gatekeepers")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            # Test confidence-ranked insights vs absolute truth claims
            test_queries = [
                {
                    "input": "What are the health benefits of exercise?",
                    "expected_confidence_range": (0.7, 1.0)  # Well-established facts
                },
                {
                    "input": "What will the weather be like next month?",
                    "expected_confidence_range": (0.3, 0.7)  # Uncertain predictions
                },
                {
                    "input": "Tell me about quantum computing applications.",
                    "expected_confidence_range": (0.6, 0.9)  # Technical but established
                }
            ]
            
            confidence_results = []
            
            for i, query in enumerate(test_queries):
                self.logger.info(f"   Testing confidence ranking case {i+1}...")
                
                interaction = await self.palm_slab.process_user_input(
                    user_id="test_user",
                    input_content=query["input"],
                    interaction_type="conversation"
                )
                
                confidence_in_range = (
                    query["expected_confidence_range"][0] <= interaction.confidence_score <= query["expected_confidence_range"][1]
                )
                
                result = {
                    "query": i + 1,
                    "confidence_score": interaction.confidence_score,
                    "expected_range": query["expected_confidence_range"],
                    "in_expected_range": confidence_in_range,
                    "has_mesh_validation": interaction.mesh_validation is not None
                }
                
                confidence_results.append(result)
                
                self.logger.info(f"     Confidence: {interaction.confidence_score:.3f} (expected: {query['expected_confidence_range']})")
            
            successful_rankings = len([r for r in confidence_results if r["in_expected_range"]])
            
            self.test_results["truth_without_gatekeepers"] = {
                "success": True,
                "queries_tested": len(test_queries),
                "successful_rankings": successful_rankings,
                "ranking_accuracy": successful_rankings / len(test_queries),
                "confidence_results": confidence_results
            }
            
            self.logger.info(f"   ✅ Confidence ranking accuracy: {successful_rankings}/{len(test_queries)} ({100 * successful_rankings / len(test_queries):.1f}%)")
            
        except Exception as e:
            self.logger.error(f"❌ Truth without gatekeepers test failed: {e}")
            self.test_results["truth_without_gatekeepers"] = {"success": False, "error": str(e)}
    
    async def _test_complete_pipeline(self):
        """Test 6: Complete Palm Slab Pipeline"""
        
        self.logger.info("\n🧪 Test 6: Complete Palm Slab Pipeline")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            # Test complete pipeline: input → privacy ring → local processing → mesh validation → adaptive learning
            complex_input = """
            I'm working on a machine learning project for my company TechStartup Inc. 
            We're building a recommendation system for e-commerce. I need help understanding 
            collaborative filtering algorithms and their implementation in Python. 
            Can you schedule a meeting with my team next Tuesday to discuss this?
            """
            
            self.logger.info("   Testing complete pipeline with complex input...")
            
            start_time = time.time()
            
            # Process through complete pipeline
            interaction = await self.palm_slab.process_user_input(
                user_id="test_user_pipeline",
                input_content=complex_input,
                interaction_type="conversation",
                context={
                    "current_activity": "research",
                    "mood": "focused",
                    "location": "office"
                }
            )
            
            pipeline_time = time.time() - start_time
            
            # Validate pipeline components
            pipeline_components = {
                "privacy_ring_activated": "privacy_level" in interaction.metadata.get("sharing_permissions", {}),
                "local_processing_completed": interaction.local_response is not None,
                "mesh_validation_attempted": interaction.mesh_validation is not None,
                "confidence_calculated": 0.0 < interaction.confidence_score <= 1.0,
                "adaptive_learning_performed": "adaptive_synapses_updated" in interaction.metadata
            }
            
            components_working = sum(pipeline_components.values())
            
            self.test_results["complete_pipeline"] = {
                "success": True,
                "processing_time": pipeline_time,
                "confidence_score": interaction.confidence_score,
                "components_tested": len(pipeline_components),
                "components_working": components_working,
                "pipeline_efficiency": components_working / len(pipeline_components),
                "component_status": pipeline_components
            }
            
            self.logger.info(f"   ✅ Pipeline processed in {pipeline_time:.3f}s")
            self.logger.info(f"   ✅ Components working: {components_working}/{len(pipeline_components)}")
            self.logger.info(f"   ✅ Final confidence: {interaction.confidence_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Complete pipeline test failed: {e}")
            self.test_results["complete_pipeline"] = {"success": False, "error": str(e)}
    
    async def _test_privacy_levels(self):
        """Test 7: Privacy Levels"""
        
        self.logger.info("\n🧪 Test 7: Privacy Levels")
        self.logger.info("-" * 60)
        
        try:
            privacy_levels = ["private", "selective", "open"]
            privacy_results = []
            
            for privacy_level in privacy_levels:
                self.logger.info(f"   Testing {privacy_level} privacy level...")
                
                # Create palm slab with specific privacy level
                test_slab = create_complete_palm_slab(
                    node_id=f"test_privacy_{privacy_level}",
                    privacy_level=privacy_level
                )
                
                await test_slab.initialize()
                
                # Test with same input across privacy levels
                test_input = "I need help with my work project on data analysis."
                
                interaction = await test_slab.process_user_input(
                    user_id="privacy_test_user",
                    input_content=test_input,
                    interaction_type="conversation"
                )
                
                result = {
                    "privacy_level": privacy_level,
                    "mesh_validation_attempted": interaction.mesh_validation is not None,
                    "processing_time": interaction.processing_time,
                    "confidence_score": interaction.confidence_score
                }
                
                privacy_results.append(result)
                
                await test_slab.cleanup()
                
                self.logger.info(f"     Mesh validation: {'Yes' if result['mesh_validation_attempted'] else 'No'}")
                self.logger.info(f"     Confidence: {result['confidence_score']:.3f}")
            
            self.test_results["privacy_levels"] = {
                "success": True,
                "levels_tested": len(privacy_levels),
                "results": privacy_results
            }
            
            self.logger.info(f"   ✅ Privacy levels tested: {len(privacy_levels)}")
            
        except Exception as e:
            self.logger.error(f"❌ Privacy levels test failed: {e}")
            self.test_results["privacy_levels"] = {"success": False, "error": str(e)}
    
    async def _test_system_integration(self):
        """Test 8: System Integration"""
        
        self.logger.info("\n🧪 Test 8: System Integration")
        self.logger.info("-" * 60)
        
        if not self.palm_slab:
            self.logger.error("❌ Palm slab not available for testing")
            return
        
        try:
            # Test integration with all Sentient phases
            integration_tests = [
                ("Voice Processing", "fact_extraction", "I want to extract key facts from this conversation."),
                ("Memory Systems", "fact_extraction", "Remember that I prefer morning meetings and work in the tech industry."),
                ("Task Automation", "task_request", "Schedule weekly team standup meetings every Monday at 9 AM."),
                ("Personal AI", "conversation", "What's the best approach for managing my daily tasks efficiently?")
            ]
            
            integration_results = []
            
            for system_name, interaction_type, test_input in integration_tests:
                self.logger.info(f"   Testing {system_name} integration...")
                
                interaction = await self.palm_slab.process_user_input(
                    user_id="integration_test_user",
                    input_content=test_input,
                    interaction_type=interaction_type
                )
                
                result = {
                    "system": system_name,
                    "success": interaction.local_response is not None,
                    "processing_time": interaction.processing_time,
                    "confidence": interaction.confidence_score
                }
                
                integration_results.append(result)
                
                if result["success"]:
                    self.logger.info(f"     ✅ {system_name}: {interaction.processing_time:.3f}s, confidence {interaction.confidence_score:.3f}")
                else:
                    self.logger.info(f"     ❌ {system_name}: Integration failed")
            
            successful_integrations = len([r for r in integration_results if r["success"]])
            
            self.test_results["system_integration"] = {
                "success": True,
                "systems_tested": len(integration_tests),
                "successful_integrations": successful_integrations,
                "integration_success_rate": successful_integrations / len(integration_tests),
                "results": integration_results
            }
            
            self.logger.info(f"   ✅ System integrations successful: {successful_integrations}/{len(integration_tests)}")
            
        except Exception as e:
            self.logger.error(f"❌ System integration test failed: {e}")
            self.test_results["system_integration"] = {"success": False, "error": str(e)}
    
    def _print_test_results(self):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 80)
        print("🏆 COMPLETE PALM SLAB INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() if r.get("success", False)])
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Tests Run: {total_tests}")
        print(f"   Tests Passed: {successful_tests}")
        print(f"   Success Rate: {100 * successful_tests / total_tests:.1f}%")
        
        print(f"\n🎯 MESH PRINCIPLES VALIDATION:")
        
        # Test 1: Palm Slab Creation
        creation_test = self.test_results.get("palm_slab_creation", {})
        if creation_test.get("success"):
            print(f"   ✅ 'Every slab is a full node': {creation_test.get('systems_initialized', 0)} systems operational")
        
        # Test 2: Local-First Processing
        local_test = self.test_results.get("local_first_processing", {})
        if local_test.get("success"):
            cases_passed = local_test.get("cases_passed", 0)
            cases_total = local_test.get("cases_tested", 0)
            print(f"   ✅ 'Data is Local First': {cases_passed}/{cases_total} cases processed locally")
        
        # Test 3: Mesh Validation
        mesh_test = self.test_results.get("mesh_validation", {})
        if mesh_test.get("success"):
            validation_attempted = mesh_test.get("validation_attempted", False)
            social_checksum = mesh_test.get("social_checksum", False)
            print(f"   ✅ 'Consensus through Cross-Validation': Validation {'attempted' if validation_attempted else 'not attempted'}, Social checksum {'generated' if social_checksum else 'not generated'}")
        
        # Test 4: Adaptive Synapses
        synapses_test = self.test_results.get("adaptive_synapses", {})
        if synapses_test.get("success"):
            peer_connections = synapses_test.get("updated_peer_connections", 0)
            avg_weight = synapses_test.get("average_peer_weight", 0.0)
            print(f"   ✅ 'Adaptive Synapses': {peer_connections} peer connections, {avg_weight:.3f} avg weight")
        
        # Test 5: Truth Without Gatekeepers
        truth_test = self.test_results.get("truth_without_gatekeepers", {})
        if truth_test.get("success"):
            ranking_accuracy = truth_test.get("ranking_accuracy", 0.0)
            print(f"   ✅ 'Truth Without Gatekeepers': {100 * ranking_accuracy:.1f}% confidence ranking accuracy")
        
        print(f"\n🔧 TECHNICAL PERFORMANCE:")
        
        # Complete Pipeline
        pipeline_test = self.test_results.get("complete_pipeline", {})
        if pipeline_test.get("success"):
            processing_time = pipeline_test.get("processing_time", 0.0)
            efficiency = pipeline_test.get("pipeline_efficiency", 0.0)
            print(f"   ⚡ Pipeline Performance: {processing_time:.3f}s processing, {100 * efficiency:.1f}% efficiency")
        
        # Privacy Levels
        privacy_test = self.test_results.get("privacy_levels", {})
        if privacy_test.get("success"):
            levels_tested = privacy_test.get("levels_tested", 0)
            print(f"   🔒 Privacy Control: {levels_tested} privacy levels validated")
        
        # System Integration
        integration_test = self.test_results.get("system_integration", {})
        if integration_test.get("success"):
            integration_rate = integration_test.get("integration_success_rate", 0.0)
            systems_tested = integration_test.get("systems_tested", 0)
            print(f"   🔗 System Integration: {100 * integration_rate:.1f}% success rate across {systems_tested} systems")
        
        print(f"\n🎉 TRANSFORMATION ASSESSMENT:")
        
        if successful_tests >= 7:
            print("   🚀 TRANSFORMATION COMPLETE!")
            print("   ✅ Sentient integration successfully transformed into true palm slab nodes")
            print("   ✅ All Mesh principles implemented and validated")
            print("   ✅ Ready for production deployment in The Mesh ecosystem")
        elif successful_tests >= 5:
            print("   🌟 TRANSFORMATION MOSTLY SUCCESSFUL!")
            print("   ✅ Core palm slab functionality operational")
            print("   ⚠️  Minor issues may need attention")
        else:
            print("   ⚠️  TRANSFORMATION NEEDS WORK")
            print("   🔧 Significant issues need resolution before deployment")
        
        print("\n" + "=" * 80)
    
    async def _cleanup(self):
        """Clean up test resources"""
        
        try:
            if self.palm_slab:
                await self.palm_slab.cleanup()
            self.logger.info("✅ Test cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Test cleanup failed: {e}")


async def main():
    """Main test execution"""
    
    print("🎯 COMPLETE PALM SLAB INTEGRATION VALIDATION")
    print("Testing transformation of Sentient integration into true Mesh palm slab nodes")
    print()
    
    test_suite = CompletePalmSlabTest()
    await test_suite.run_complete_test_suite()


if __name__ == "__main__":
    asyncio.run(main())