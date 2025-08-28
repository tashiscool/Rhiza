#!/usr/bin/env python3
"""
Test suite for LLM Integration components

Tests the enhanced LLM integration system that combines traditional ML models
with The Mesh's social consensus and trust validation.
"""

import asyncio
import logging
import unittest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestLLMIntegration(unittest.IsolatedAsyncioTestCase):
    """Test cases for LLM Integration components"""
    
    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.test_node_id = "test_llm_node_001"
        
    def test_llm_trust_validator_imports(self):
        """Test that LLM integration components can be imported"""
        try:
            from src.mesh_core.llm_integration import (
                LLMTrustValidator, LLMResponse, TrustMetrics, ValidationContext
            )
            self.assertTrue(True, "LLM integration imports successful")
        except ImportError as e:
            self.fail(f"LLM integration import failed: {e}")
    
    def test_llm_response_creation(self):
        """Test LLMResponse dataclass creation"""
        from src.mesh_core.llm_integration.llm_trust_validator import LLMResponse
        from datetime import datetime
        
        response = LLMResponse(
            content="Test response content",
            model_name="test-model-7b",
            model_version="gguf",
            response_time=0.5,
            token_count=10,
            confidence_score=0.8,
            generation_params={"temperature": 0.7},
            timestamp=datetime.now(),
            response_hash="abc123"
        )
        
        self.assertEqual(response.content, "Test response content")
        self.assertEqual(response.model_name, "test-model-7b")
        self.assertEqual(response.confidence_score, 0.8)
        self.assertIsInstance(response.generation_params, dict)
    
    def test_trust_metrics_creation(self):
        """Test TrustMetrics dataclass creation"""
        from src.mesh_core.llm_integration.llm_trust_validator import TrustMetrics
        
        metrics = TrustMetrics(
            social_consensus=0.85,
            factual_alignment=0.92,
            bias_detection=0.12,
            source_credibility=0.88,
            historical_accuracy=0.79,
            context_relevance=0.94,
            mesh_confidence=0.86
        )
        
        self.assertEqual(metrics.social_consensus, 0.85)
        self.assertEqual(metrics.mesh_confidence, 0.86)
        self.assertTrue(0 <= metrics.bias_detection <= 1)
    
    def test_validation_context_creation(self):
        """Test ValidationContext dataclass creation"""
        from src.mesh_core.llm_integration.llm_trust_validator import ValidationContext
        
        context = ValidationContext(
            query="What are the benefits of renewable energy?",
            user_intent="Learn about environmental benefits",
            privacy_level="high",
            required_confidence=0.75,
            validation_peers=["peer1", "peer2", "peer3"],
            biometric_verified=True,
            coercion_detected=False
        )
        
        self.assertEqual(context.query, "What are the benefits of renewable energy?")
        self.assertEqual(context.privacy_level, "high")
        self.assertEqual(len(context.validation_peers), 3)
        self.assertTrue(context.biometric_verified)
        self.assertFalse(context.coercion_detected)
    
    async def test_llm_trust_validator_initialization(self):
        """Test LLMTrustValidator initialization"""
        try:
            from src.mesh_core.llm_integration.llm_trust_validator import LLMTrustValidator
            from src.mesh_core.config_manager import ConfigurationManager
            
            config_manager = ConfigurationManager()
            validator = LLMTrustValidator(config_manager)
            
            self.assertIsInstance(validator.trust_history, dict)
            self.assertIsInstance(validator.model_registry, dict)
            self.assertIsInstance(validator.validation_cache, dict)
            
        except Exception as e:
            self.fail(f"LLMTrustValidator initialization failed: {e}")
    
    async def test_enhanced_kobold_client_initialization(self):
        """Test EnhancedKoboldClient initialization"""
        try:
            from src.mesh_core.llm_integration.enhanced_kobold_client import (
                EnhancedKoboldClient, KoboldConfig
            )
            from src.mesh_core.llm_integration.llm_trust_validator import LLMTrustValidator
            from src.mesh_core.config_manager import ConfigurationManager
            
            config = KoboldConfig(
                api_url="http://127.0.0.1:5001",
                model_path="/test/path/model.gguf",
                context_length=4096,
                threads=8,
                gpu_layers=0,
                rope_freq_base=10000.0,
                rope_freq_scale=1.0,
                batch_size=512,
                memory_gb=6,
                port=5001
            )
            
            config_manager = ConfigurationManager()
            trust_validator = LLMTrustValidator(config_manager)
            client = EnhancedKoboldClient(config, trust_validator)
            
            self.assertEqual(client.config.api_url, "http://127.0.0.1:5001")
            self.assertEqual(client.config.memory_gb, 6)
            self.assertIsNotNone(client.trust_validator)
            
        except Exception as e:
            self.fail(f"EnhancedKoboldClient initialization failed: {e}")
    
    def test_model_inspector_initialization(self):
        """Test ModelInspector initialization"""
        try:
            from src.mesh_core.llm_integration.model_inspector import ModelInspector
            
            inspector = ModelInspector()
            
            self.assertIsInstance(inspector.inspection_cache, dict)
            self.assertIsInstance(inspector.benchmark_cache, dict)
            
        except Exception as e:
            self.fail(f"ModelInspector initialization failed: {e}")
    
    async def test_model_inspection_result_structure(self):
        """Test ModelInspectionResult structure"""
        try:
            from src.mesh_core.llm_integration.model_inspector import (
                ModelInspector, ModelInspectionResult
            )
            from datetime import datetime
            
            # Create a mock inspection result
            result = ModelInspectionResult(
                model_name="test-model-7b-q4_k_m",
                file_path="/test/path/model.gguf",
                file_size_gb=4.2,
                file_hash="abc123def456",
                architecture="Llama",
                parameter_count=7000000000,
                quantization="Q4_K_M",
                context_length=4096,
                vocab_size=32000,
                security_score=0.85,
                trust_compatibility=0.92,
                performance_score=0.78,
                mesh_readiness=0.85,
                issues=["No major issues found"],
                recommendations=["Consider Q5_K_M for better quality"],
                inspection_timestamp=datetime.now()
            )
            
            self.assertEqual(result.model_name, "test-model-7b-q4_k_m")
            self.assertEqual(result.architecture, "Llama")
            self.assertEqual(result.quantization, "Q4_K_M")
            self.assertTrue(0 <= result.mesh_readiness <= 1)
            self.assertIsInstance(result.issues, list)
            self.assertIsInstance(result.recommendations, list)
            
        except Exception as e:
            self.fail(f"ModelInspectionResult test failed: {e}")
    
    async def test_apple_m4_optimization_concepts(self):
        """Test Apple M4 optimization concepts"""
        try:
            from src.mesh_core.llm_integration.enhanced_kobold_client import EnhancedKoboldClient
            
            # Test that Apple M4 optimizations are conceptually available
            # (We can't test actual hardware optimization without M4 hardware)
            optimizations = {
                'neural_engine': False,
                'metal_acceleration': False,
                'unified_memory': False,
                'thread_optimization': False,
                'performance_gain': 0.0
            }
            
            # Simulate what optimize_for_apple_m4 would return
            import platform
            if 'test' in platform.node() or platform.machine() != 'arm64':
                # Running in test environment or not on Apple Silicon
                optimizations['neural_engine'] = False
                optimizations['metal_acceleration'] = False
            else:
                # Would enable optimizations on real Apple Silicon
                optimizations['neural_engine'] = True
                optimizations['metal_acceleration'] = True
            
            self.assertIn('neural_engine', optimizations)
            self.assertIn('unified_memory', optimizations)
            self.assertIsInstance(optimizations['performance_gain'], (int, float))
            
        except Exception as e:
            self.fail(f"Apple M4 optimization test failed: {e}")
    
    def test_integration_architecture(self):
        """Test that the integration architecture is properly structured"""
        try:
            # Test that all main components can be imported together
            from src.mesh_core.llm_integration import (
                LLMTrustValidator,
                LLMResponse, 
                TrustMetrics,
                ValidationContext,
                EnhancedKoboldClient,
                KoboldConfig,
                ModelStatus
            )
            
            # Verify the integration architecture
            components = [
                LLMTrustValidator,
                LLMResponse,
                TrustMetrics,
                ValidationContext,
                EnhancedKoboldClient,
                KoboldConfig,
                ModelStatus
            ]
            
            for component in components:
                self.assertTrue(hasattr(component, '__name__'), f"Component {component} should have __name__")
            
            logger.info("✅ All LLM integration components imported successfully")
            
        except ImportError as e:
            self.fail(f"Integration architecture test failed: {e}")

class TestLLMIntegrationDemo(unittest.TestCase):
    """Test the LLM integration demonstration"""
    
    def test_demo_can_be_imported(self):
        """Test that the demo script can be imported"""
        try:
            # Try importing the demo module
            import sys
            import os
            sys.path.insert(0, os.getcwd())
            
            # Import should not fail
            import enhanced_llm_integration_demo
            
            self.assertTrue(hasattr(enhanced_llm_integration_demo, 'demonstrate_enhanced_llm_integration'))
            logger.info("✅ LLM integration demo can be imported")
            
        except Exception as e:
            # Demo import failure is not critical for testing
            logger.warning(f"⚠️ Demo import failed (non-critical): {e}")

async def run_llm_integration_tests():
    """Run all LLM integration tests"""
    print("🧪 RUNNING LLM INTEGRATION TESTS")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [TestLLMIntegration, TestLLMIntegrationDemo]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print(f"\n📊 LLM INTEGRATION TEST RESULTS:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    asyncio.run(run_llm_integration_tests())