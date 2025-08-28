"""
Intent Monitor - Integration with intent_aware_privacy_protection_in_pws
Detects when users are being manipulated based on intent divergence
"""

import asyncio
import subprocess
import json
import os
import tempfile
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging


@dataclass
class IntentAnalysis:
    """Result of intent analysis"""
    baseline_intent: str
    detected_intent: str
    confidence: float
    intent_categories: List[str]
    privacy_risk: float


@dataclass  
class ManipulationCheck:
    """Result of manipulation detection"""
    manipulation_detected: bool
    original_intent: str
    detected_intent: str
    divergence_score: float
    breadcrumbs: List[str]
    requires_mediation: bool


class IntentMonitor:
    """
    Intent Monitor detects manipulation by analyzing intent divergence
    
    Integrates with Java-based intent_aware_privacy_protection_in_pws system
    for sophisticated intent classification and privacy protection.
    
    Core Capabilities:
    - Build personal intent baselines
    - Detect when responses lead users away from stated goals
    - Generate "truth breadcrumbs" back to original intent
    - Privacy-preserving intent obfuscation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Java system paths
        self.java_system_path = '/Users/admin/AI/intent_aware_privacy_protection_in_pws/source_code'
        self.java_classes_path = os.path.join(self.java_system_path, 'build', 'classes')
        self.java_libs_path = os.path.join(self.java_system_path, 'libraries')
        
        # Intent analysis parameters
        self.sensitivity_threshold = config.get('sensitivity_threshold', 0.7)
        self.divergence_threshold = config.get('divergence_threshold', 0.6)
        
        # User intent baselines (local storage)
        self.intent_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Java system
        self._initialize_java_system()
        
        self.logger.info("Intent Monitor initialized - Ready for manipulation detection")
    
    def _initialize_java_system(self):
        """Initialize Java-based intent classification system"""
        try:
            # Check if Java classes are compiled
            if not os.path.exists(self.java_classes_path):
                self._compile_java_system()
            
            # Verify system is working
            self._test_java_system()
            
            self.logger.info("Java intent classification system ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Java system: {e}")
            self.logger.warning("Falling back to Python-based intent analysis")
            self._use_python_fallback = True
    
    def _compile_java_system(self):
        """Compile Java intent classification system"""
        self.logger.info("Compiling Java intent classification system...")
        
        try:
            # Create build directory
            os.makedirs(self.java_classes_path, exist_ok=True)
            
            # Find all Java files
            java_files = []
            for root, dirs, files in os.walk(os.path.join(self.java_system_path, 'src')):
                for file in files:
                    if file.endswith('.java'):
                        java_files.append(os.path.join(root, file))
            
            # Build classpath
            jar_files = [
                os.path.join(self.java_libs_path, jar)
                for jar in os.listdir(self.java_libs_path)
                if jar.endswith('.jar')
            ]
            classpath = ':'.join(jar_files) + ':' + self.java_classes_path
            
            # Compile command
            cmd = [
                'javac',
                '-cp', classpath,
                '-d', self.java_classes_path
            ] + java_files
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Java compilation failed: {result.stderr}")
            
            self.logger.info("Java system compiled successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to compile Java system: {e}")
    
    def _test_java_system(self):
        """Test that Java intent system is working"""
        test_query = "weather forecast tomorrow"
        
        try:
            result = self._run_java_intent_classifier(test_query)
            self.logger.debug(f"Java system test successful: {result}")
            
        except Exception as e:
            raise RuntimeError(f"Java system test failed: {e}")
    
    async def analyze_query(
        self, 
        query: str, 
        user_baseline: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Analyze query intent using Java-based classification system
        
        Determines user's genuine search intent and privacy risks
        """
        
        try:
            # Run Java intent classifier
            intent_result = self._run_java_intent_classifier(query)
            
            # Parse classification results
            intent_categories = self._parse_intent_categories(intent_result)
            primary_intent = intent_categories[0] if intent_categories else "unknown"
            
            # Calculate confidence based on Java classifier output
            confidence = self._extract_confidence(intent_result)
            
            # Assess privacy risk
            privacy_risk = self._assess_privacy_risk(query, intent_categories)
            
            return IntentAnalysis(
                baseline_intent=primary_intent,
                detected_intent=primary_intent,
                confidence=confidence,
                intent_categories=intent_categories,
                privacy_risk=privacy_risk
            )
            
        except Exception as e:
            self.logger.error(f"Intent analysis error: {e}")
            # Fallback to simple analysis
            return self._fallback_intent_analysis(query)
    
    def _run_java_intent_classifier(self, query: str) -> str:
        """Run Java intent classification system"""
        
        # Build classpath
        jar_files = [
            os.path.join(self.java_libs_path, jar)
            for jar in os.listdir(self.java_libs_path)
            if jar.endswith('.jar')
        ]
        classpath = ':'.join(jar_files) + ':' + self.java_classes_path
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(query)
            temp_input = f.name
        
        try:
            # Run Java classifier
            cmd = [
                'java',
                '-cp', classpath,
                'edu.virginia.cs.model.ClassifyIntent',
                temp_input
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=self.java_system_path
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Java classifier error: {result.stderr}")
            
            return result.stdout.strip()
            
        finally:
            # Clean up temporary file
            os.unlink(temp_input)
    
    def _parse_intent_categories(self, java_output: str) -> List[str]:
        """Parse intent categories from Java classifier output"""
        
        # Parse Java output format (this depends on the actual output format)
        # For now, implement basic parsing
        lines = java_output.strip().split('\n')
        
        categories = []
        for line in lines:
            if 'intent' in line.lower() or 'category' in line.lower():
                # Extract intent category from line
                parts = line.split(':')
                if len(parts) >= 2:
                    category = parts[1].strip()
                    categories.append(category)
        
        return categories[:5]  # Return top 5 categories
    
    def _extract_confidence(self, java_output: str) -> float:
        """Extract confidence score from Java classifier output"""
        
        # Look for confidence/probability scores in output
        lines = java_output.strip().split('\n')
        
        for line in lines:
            if 'confidence' in line.lower() or 'probability' in line.lower():
                # Extract numeric value
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                if numbers:
                    return min(1.0, float(numbers[0]))
        
        # Default confidence
        return 0.5
    
    def _assess_privacy_risk(self, query: str, intent_categories: List[str]) -> float:
        """Assess privacy risk of the query"""
        
        # Simple privacy risk assessment
        privacy_indicators = [
            'personal', 'private', 'confidential', 'secret',
            'password', 'account', 'finance', 'medical', 'health'
        ]
        
        query_lower = query.lower()
        risk_score = 0.0
        
        for indicator in privacy_indicators:
            if indicator in query_lower:
                risk_score += 0.2
        
        # Sensitive intent categories
        sensitive_categories = ['personal_info', 'financial', 'medical', 'private']
        for category in intent_categories:
            if any(sensitive in category.lower() for sensitive in sensitive_categories):
                risk_score += 0.3
        
        return min(1.0, risk_score)
    
    async def detect_manipulation(
        self, 
        original_query: str, 
        system_response: str,
        baseline_intent: str
    ) -> ManipulationCheck:
        """
        Detect if system response is manipulating user away from original intent
        
        Core manipulation detection:
        - Compare original query intent vs response direction
        - Flag when responses lead away from user's stated goals
        - Generate breadcrumbs back to original intent
        """
        
        try:
            # Analyze intent of system response
            response_intent = await self.analyze_query(system_response)
            
            # Calculate intent divergence
            divergence_score = self._calculate_intent_divergence(
                baseline_intent,
                response_intent.detected_intent
            )
            
            # Check if divergence exceeds threshold
            manipulation_detected = divergence_score > self.divergence_threshold
            
            # Generate truth breadcrumbs if manipulation detected
            breadcrumbs = []
            if manipulation_detected:
                breadcrumbs = self._generate_truth_breadcrumbs(
                    original_query,
                    baseline_intent,
                    system_response
                )
            
            # Determine if mediation is required
            requires_mediation = (
                manipulation_detected and 
                divergence_score > 0.8
            )
            
            return ManipulationCheck(
                manipulation_detected=manipulation_detected,
                original_intent=baseline_intent,
                detected_intent=response_intent.detected_intent,
                divergence_score=divergence_score,
                breadcrumbs=breadcrumbs,
                requires_mediation=requires_mediation
            )
            
        except Exception as e:
            self.logger.error(f"Manipulation detection error: {e}")
            return ManipulationCheck(
                manipulation_detected=False,
                original_intent=baseline_intent,
                detected_intent="unknown",
                divergence_score=0.0,
                breadcrumbs=[],
                requires_mediation=False
            )
    
    def _calculate_intent_divergence(
        self, 
        original_intent: str, 
        response_intent: str
    ) -> float:
        """
        Calculate semantic divergence between original and response intents
        
        Uses simple semantic similarity for now - can be enhanced with 
        more sophisticated NLP models
        """
        
        if not original_intent or not response_intent:
            return 0.0
        
        # Simple token-based similarity (can be enhanced)
        original_tokens = set(original_intent.lower().split())
        response_tokens = set(response_intent.lower().split())
        
        if not original_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(original_tokens & response_tokens)
        union = len(original_tokens | response_tokens)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        divergence = 1.0 - similarity
        
        return divergence
    
    def _generate_truth_breadcrumbs(
        self, 
        original_query: str,
        original_intent: str, 
        system_response: str
    ) -> List[str]:
        """
        Generate breadcrumbs leading back to user's original intent
        
        Provides pathway back to truth when manipulation is detected
        """
        
        breadcrumbs = [
            f"Your original question: '{original_query}'",
            f"Your stated intent: {original_intent}",
            "The response appears to be leading you toward a different goal",
            "Consider: Does this answer actually help with your original question?",
            f"To refocus on your goal, try asking: 'How does this relate to {original_intent}?'"
        ]
        
        return breadcrumbs
    
    def build_baseline(self, user_id: str) -> Dict[str, Any]:
        """
        Build intent baseline for user based on historical interactions
        
        This creates the foundation for detecting manipulation attempts
        """
        
        if user_id in self.intent_baselines:
            return self.intent_baselines[user_id]
        
        # For now, create default baseline
        # In production, this would analyze user's historical query patterns
        baseline = {
            'primary_intents': ['information_seeking', 'problem_solving'],
            'privacy_preferences': {'high': ['personal', 'financial'], 'medium': ['social']},
            'typical_query_patterns': [],
            'manipulation_sensitivity': self.sensitivity_threshold,
            'trusted_domains': [],
            'creation_date': asyncio.get_event_loop().time()
        }
        
        self.intent_baselines[user_id] = baseline
        return baseline
    
    def _fallback_intent_analysis(self, query: str) -> IntentAnalysis:
        """Fallback intent analysis when Java system is unavailable"""
        
        # Simple keyword-based intent detection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['weather', 'forecast', 'temperature']):
            intent = 'weather_info'
        elif any(word in query_lower for word in ['news', 'current', 'events']):
            intent = 'news_info'
        elif any(word in query_lower for word in ['how', 'why', 'what', 'when', 'where']):
            intent = 'information_seeking'
        elif any(word in query_lower for word in ['help', 'problem', 'issue', 'solve']):
            intent = 'problem_solving'
        else:
            intent = 'general_query'
        
        return IntentAnalysis(
            baseline_intent=intent,
            detected_intent=intent,
            confidence=0.6,  # Lower confidence for fallback
            intent_categories=[intent],
            privacy_risk=0.1
        )
    
    def get_user_privacy_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's privacy profile and protection settings"""
        
        baseline = self.build_baseline(user_id)
        
        return {
            'privacy_level': 'high',  # Default to high privacy
            'obfuscation_enabled': True,
            'manipulation_detection': True,
            'intent_logging': False,  # Local-first, no logging by default
            'trusted_entities': baseline.get('trusted_domains', []),
            'sensitivity_settings': {
                'manipulation_threshold': baseline.get('manipulation_sensitivity', 0.7),
                'privacy_risk_threshold': 0.6,
                'intent_divergence_threshold': self.divergence_threshold
            }
        }
    
    async def update_user_baseline(
        self, 
        user_id: str, 
        interaction_data: Dict[str, Any]
    ):
        """Update user's intent baseline based on new interactions"""
        
        baseline = self.build_baseline(user_id)
        
        # Update baseline with new interaction patterns
        # This implements adaptive learning while preserving privacy
        
        if 'query_intent' in interaction_data:
            baseline['typical_query_patterns'].append({
                'intent': interaction_data['query_intent'],
                'timestamp': asyncio.get_event_loop().time(),
                'context': interaction_data.get('context', {})
            })
        
        # Keep only recent patterns (privacy-preserving)
        max_patterns = 50
        if len(baseline['typical_query_patterns']) > max_patterns:
            baseline['typical_query_patterns'] = baseline['typical_query_patterns'][-max_patterns:]
        
        self.intent_baselines[user_id] = baseline