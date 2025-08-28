"""
Mesh Watchdogs Demo
==================

Component 10.1: Mesh Degeneration Watchdogs
Comprehensive demonstration of all watchdog components

Demonstrates entropy monitoring, manipulation detection,
feedback analysis, drift warning, and health monitoring.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import watchdog components
from .entropy_monitor import EntropyMonitor, EntropyType
from .manipulation_detector import ManipulationDetector, ManipulationType
from .feedback_analyzer import FeedbackAnalyzer, FeedbackType
from .drift_warner import DriftWarner, DriftType
from .health_monitor import HealthMonitor, HealthDimension

logger = logging.getLogger(__name__)


class WatchdogsDemo:
    """
    Comprehensive demonstration of all Phase 10 watchdog components
    
    Shows how the watchdogs work together to monitor system health,
    detect degeneration, and provide early warnings.
    """
    
    def __init__(self, node_id: str = "demo_node"):
        self.node_id = node_id
        
        # Initialize all watchdog components
        self.entropy_monitor = EntropyMonitor(node_id)
        self.manipulation_detector = ManipulationDetector(node_id)
        self.feedback_analyzer = FeedbackAnalyzer(node_id)
        self.drift_warner = DriftWarner(node_id)
        self.health_monitor = HealthMonitor(node_id)
        
        logger.info(f"WatchdogsDemo initialized for node: {self.node_id}")
    
    async def run_complete_demo(self):
        """Run complete demonstration of all watchdog components"""
        logger.info("🚀 Starting Phase 10: Mesh Degeneration Watchdogs Demo")
        logger.info("=" * 60)
        
        try:
            # Demo each component individually
            await self._demo_entropy_monitoring()
            await self._demo_manipulation_detection()
            await self._demo_feedback_analysis()
            await self._demo_drift_warning()
            await self._demo_health_monitoring()
            
            # Demo integration and collaboration
            await self._demo_watchdog_integration()
            
            # Generate comprehensive report
            await self._generate_comprehensive_report()
            
            logger.info("✅ Phase 10 Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _demo_entropy_monitoring(self):
        """Demonstrate entropy monitoring capabilities"""
        logger.info("\n🔍 Demo: Entropy Monitoring")
        logger.info("-" * 40)
        
        # Measure different types of entropy
        entropy_data = {
            EntropyType.INFORMATION: {
                "information_distribution": {
                    "news": 150, "social_media": 300, "academic": 50,
                    "entertainment": 200, "technical": 100
                }
            },
            EntropyType.BEHAVIORAL: {
                "behavior_patterns": {
                    "collaborative": 80, "competitive": 20, "neutral": 30
                }
            },
            EntropyType.STRUCTURAL: {
                "structural_metrics": {
                    "connectivity": 0.7, "coherence": 0.8, "complexity": 6.0
                }
            },
            EntropyType.CULTURAL: {
                "cultural_metrics": {
                    "value_consistency": 0.7, "norm_adherence": 0.8, "cultural_stability": 0.6
                }
            },
            EntropyType.TECHNICAL: {
                "technical_metrics": {
                    "error_rate": 0.05, "response_time": 200, "resource_utilization": 0.6
                }
            }
        }
        
        # Measure entropy for each type
        for entropy_type, data in entropy_data.items():
            measurement = self.entropy_monitor.measure_entropy(entropy_type, data)
            logger.info(f"  {entropy_type.value}: {measurement.entropy_value:.3f} ({measurement.entropy_level.value})")
        
        # Analyze trends
        trends = self.entropy_monitor.analyze_trends()
        logger.info(f"  Trends analyzed: {len(trends)}")
        
        # Get summary
        summary = self.entropy_monitor.get_entropy_summary()
        logger.info(f"  Total measurements: {summary['total_measurements']}")
        logger.info(f"  Alerts generated: {summary['total_alerts']}")
    
    async def _demo_manipulation_detection(self):
        """Demonstrate manipulation detection capabilities"""
        logger.info("\n🕵️ Demo: Manipulation Detection")
        logger.info("-" * 40)
        
        # Test communication analysis
        test_messages = [
            {
                "content": "That never happened, you're imagining things. You're being too sensitive.",
                "metadata": {"gaslighting_count": 4, "sent_at_odd_hours": True}
            },
            {
                "content": "As an expert, I can tell you that everyone else agrees. Act now, don't wait!",
                "metadata": {"urgency_level": "high"}
            },
            {
                "content": "I can't tell you the details, but some people say this is classified information.",
                "metadata": {"information_restriction": True}
            }
        ]
        
        # Analyze each message
        for i, message in enumerate(test_messages):
            pattern = self.manipulation_detector.analyze_communication(
                message, f"sender_{i}", f"receiver_{i}"
            )
            if pattern:
                logger.info(f"  Message {i+1}: {pattern.manipulation_type.value} detected (confidence: {pattern.confidence_score:.3f})")
            else:
                logger.info(f"  Message {i+1}: No manipulation detected")
        
        # Test behavioral analysis
        current_behavior = {"response_time": 500, "error_rate": 0.15, "collaboration_level": 0.3}
        historical_data = [
            {"response_time": 200, "error_rate": 0.05, "collaboration_level": 0.8},
            {"response_time": 250, "error_rate": 0.08, "collaboration_level": 0.7},
            {"response_time": 300, "error_rate": 0.10, "collaboration_level": 0.6}
        ]
        
        anomaly = self.manipulation_detector.analyze_behavioral_changes(
            "test_node", current_behavior, historical_data
        )
        
        if anomaly:
            logger.info(f"  Behavioral anomaly detected: {anomaly.description}")
        else:
            logger.info("  No behavioral anomalies detected")
        
        # Test trust relationship analysis
        trust_data = {
            "trust_history": [0.8, 0.7, 0.6, 0.5],
            "broken_commitments": 3,
            "communication_frequency": 0.2,
            "involved_nodes": ["node1", "node2"]
        }
        
        trust_patterns = self.manipulation_detector.analyze_trust_relationships(trust_data)
        logger.info(f"  Trust patterns detected: {len(trust_patterns)}")
        
        # Get summary
        summary = self.manipulation_detector.get_manipulation_summary()
        logger.info(f"  Total patterns detected: {summary['total_patterns_detected']}")
        logger.info(f"  Total anomalies detected: {summary['total_anomalies_detected']}")
    
    async def _demo_feedback_analysis(self):
        """Demonstrate feedback analysis capabilities"""
        logger.info("\n🔄 Demo: Feedback Analysis")
        logger.info("-" * 40)
        
        # Test network feedback analysis
        network_data = {
            "nodes": [
                {"id": "node1", "type": "core", "importance": 0.9},
                {"id": "node2", "type": "edge", "importance": 0.6},
                {"id": "node3", "type": "edge", "importance": 0.6}
            ],
            "connections": [
                {"source": "node1", "target": "node2", "weight": 0.8},
                {"source": "node2", "target": "node3", "weight": 0.7},
                {"source": "node3", "target": "node1", "weight": 0.6}
            ]
        }
        
        feedback_loops = self.feedback_analyzer.analyze_network_feedback(network_data)
        logger.info(f"  Feedback loops detected: {len(feedback_loops)}")
        
        for loop in feedback_loops:
            logger.info(f"    Loop: {loop.feedback_type.value} (strength: {loop.loop_strength:.3f}, distortion: {loop.distortion_level.value})")
        
        # Test echo chamber detection
        communication_data = {
            "communications": [
                {"sender": "user1", "content": "Great idea!", "sentiment": "positive"},
                {"sender": "user1", "content": "I agree completely", "sentiment": "positive"},
                {"sender": "user1", "content": "This is the way forward", "sentiment": "positive"},
                {"sender": "user2", "content": "Interesting perspective", "sentiment": "neutral"},
                {"sender": "user2", "content": "I see your point", "sentiment": "neutral"},
                {"sender": "user2", "content": "Let's discuss this", "sentiment": "neutral"}
            ]
        }
        
        echo_chambers = self.feedback_analyzer.detect_echo_chambers(communication_data)
        logger.info(f"  Echo chambers detected: {len(echo_chambers)}")
        
        for chamber in echo_chambers:
            logger.info(f"    Chamber: {len(chamber.core_nodes)} core nodes, isolation: {chamber.isolation_score:.3f}")
        
        # Get summary
        summary = self.feedback_analyzer.get_feedback_summary()
        logger.info(f"  Total loops detected: {summary['total_loops_detected']}")
        logger.info(f"  Total chambers detected: {summary['total_chambers_detected']}")
    
    async def _demo_drift_warning(self):
        """Demonstrate drift warning capabilities"""
        logger.info("\n⚠️ Demo: Drift Warning")
        logger.info("-" * 40)
        
        # Test cultural baseline update
        cultural_data = {
            "core_values": {
                "collaboration": 0.8, "innovation": 0.7, "sustainability": 0.9,
                "equality": 0.8, "transparency": 0.7
            },
            "behavioral_norms": {
                "respect": 0.9, "honesty": 0.8, "responsibility": 0.7,
                "empathy": 0.8, "curiosity": 0.7
            },
            "cultural_practices": {
                "knowledge_sharing": 0.8, "mentoring": 0.7, "celebration": 0.6,
                "reflection": 0.7, "adaptation": 0.6
            }
        }
        
        baseline = self.drift_warner.update_cultural_baseline("community_1", cultural_data)
        logger.info(f"  Cultural baseline updated for community_1")
        logger.info(f"    Value stability: {baseline.value_stability:.3f}")
        logger.info(f"    Norm consistency: {baseline.norm_consistency:.3f}")
        logger.info(f"    Practice continuity: {baseline.practice_continuity:.3f}")
        
        # Test drift detection with modified data
        current_data = {
            "core_values": {
                "collaboration": 0.6, "innovation": 0.5, "sustainability": 0.7,
                "equality": 0.6, "transparency": 0.5
            },
            "behavioral_norms": {
                "respect": 0.7, "honesty": 0.6, "responsibility": 0.5,
                "empathy": 0.6, "curiosity": 0.5
            },
            "cultural_practices": {
                "knowledge_sharing": 0.6, "mentoring": 0.5, "celebration": 0.4,
                "reflection": 0.5, "adaptation": 0.4
            }
        }
        
        drift_alert = self.drift_warner.detect_cultural_drift("community_1", current_data)
        
        if drift_alert:
            logger.info(f"  Cultural drift detected: {drift_alert.drift_type.value}")
            logger.info(f"    Severity: {drift_alert.severity.value}")
            logger.info(f"    Magnitude: {drift_alert.drift_magnitude:.3f}")
            logger.info(f"    Urgency: {drift_alert.urgency_level}")
        else:
            logger.info("  No cultural drift detected")
        
        # Get summary
        summary = self.drift_warner.get_drift_summary()
        logger.info(f"  Total alerts generated: {summary['total_alerts_generated']}")
        logger.info(f"  Total baselines updated: {summary['total_baselines_updated']}")
    
    async def _demo_health_monitoring(self):
        """Demonstrate health monitoring capabilities"""
        logger.info("\n🏥 Demo: Health Monitoring")
        logger.info("-" * 40)
        
        # Test health measurements across dimensions
        health_metrics = [
            (HealthDimension.SOCIAL, "social_cohesion", {"cohesion_score": 0.7}),
            (HealthDimension.SOCIAL, "community_trust", {"trust_score": 0.6}),
            (HealthDimension.ECONOMIC, "economic_stability", {"stability_score": 0.5}),
            (HealthDimension.ECONOMIC, "wealth_distribution", {"gini_coefficient": 0.6}),
            (HealthDimension.ENVIRONMENTAL, "air_quality", {"air_quality_index": 75}),
            (HealthDimension.ENVIRONMENTAL, "water_quality", {"water_quality_score": 0.8}),
            (HealthDimension.TECHNOLOGICAL, "innovation_rate", {"innovation_score": 0.6}),
            (HealthDimension.TECHNOLOGICAL, "technology_access", {"access_score": 0.7}),
            (HealthDimension.CULTURAL, "cultural_diversity", {"diversity_score": 0.8}),
            (HealthDimension.CULTURAL, "cultural_preservation", {"preservation_score": 0.7}),
            (HealthDimension.POLITICAL, "democratic_health", {"democracy_score": 0.5}),
            (HealthDimension.POLITICAL, "political_stability", {"stability_score": 0.6}),
            (HealthDimension.ETHICAL, "moral_coherence", {"coherence_score": 0.7}),
            (HealthDimension.ETHICAL, "ethical_education", {"education_score": 0.6}),
            (HealthDimension.RESILIENCE, "system_resilience", {"resilience_score": 0.7}),
            (HealthDimension.RESILIENCE, "adaptability", {"adaptability_score": 0.6})
        ]
        
        # Measure health for each metric
        for dimension, metric_name, data in health_metrics:
            metric = self.health_monitor.measure_health(dimension, metric_name, data)
            logger.info(f"  {dimension.value}.{metric_name}: {metric.metric_value:.3f} ({metric.health_status.value})")
        
        # Generate comprehensive health assessment
        assessment = self.health_monitor.generate_health_assessment()
        logger.info(f"\n  Overall Health Assessment:")
        logger.info(f"    Score: {assessment.overall_health_score:.3f}")
        logger.info(f"    Status: {assessment.overall_health_status.value}")
        logger.info(f"    Trend: {assessment.health_trend}")
        logger.info(f"    Priority Actions: {len(assessment.priority_actions)}")
        logger.info(f"    Risk Factors: {len(assessment.risk_factors)}")
        
        # Get summary
        summary = self.health_monitor.get_health_summary()
        logger.info(f"  Total metrics: {summary['total_metrics']}")
        logger.info(f"  Total assessments: {summary['total_assessments']}")
        logger.info(f"  Total alerts: {summary['total_alerts']}")
    
    async def _demo_watchdog_integration(self):
        """Demonstrate how watchdogs work together"""
        logger.info("\n🤝 Demo: Watchdog Integration")
        logger.info("-" * 40)
        
        # Simulate a scenario where multiple watchdogs detect issues
        logger.info("  Simulating integrated monitoring scenario...")
        
        # Create a synthetic crisis scenario
        crisis_data = {
            "entropy_spike": True,
            "manipulation_detected": True,
            "feedback_loops": True,
            "cultural_drift": True,
            "health_degradation": True
        }
        
        # Simulate coordinated response
        responses = []
        
        if crisis_data["entropy_spike"]:
            entropy_summary = self.entropy_monitor.get_entropy_summary()
            if entropy_summary.get("total_alerts", 0) > 0:
                responses.append("Entropy Monitor: High entropy levels detected")
        
        if crisis_data["manipulation_detected"]:
            manipulation_summary = self.manipulation_detector.get_manipulation_summary()
            if manipulation_summary.get("total_patterns_detected", 0) > 0:
                responses.append("Manipulation Detector: Manipulation patterns identified")
        
        if crisis_data["feedback_loops"]:
            feedback_summary = self.feedback_analyzer.get_feedback_summary()
            if feedback_summary.get("total_loops_detected", 0) > 0:
                responses.append("Feedback Analyzer: Unhealthy feedback loops detected")
        
        if crisis_data["cultural_drift"]:
            drift_summary = self.drift_warner.get_drift_summary()
            if drift_summary.get("total_alerts_generated", 0) > 0:
                responses.append("Drift Warner: Cultural drift alerts active")
        
        if crisis_data["health_degradation"]:
            health_summary = self.health_monitor.get_health_summary()
            if health_summary.get("total_alerts", 0) > 0:
                responses.append("Health Monitor: Health degradation detected")
        
        # Show coordinated response
        logger.info(f"  Coordinated Response Generated:")
        for response in responses:
            logger.info(f"    • {response}")
        
        logger.info(f"  Total Watchdog Alerts: {len(responses)}")
        
        if len(responses) >= 3:
            logger.info("  🚨 CRISIS LEVEL: Multiple watchdogs detecting issues")
            logger.info("  Recommended: Immediate system-wide intervention")
        elif len(responses) >= 2:
            logger.info("  ⚠️ WARNING LEVEL: Multiple watchdogs detecting issues")
            logger.info("  Recommended: Targeted intervention and monitoring")
        elif len(responses) >= 1:
            logger.info("  📊 MONITORING LEVEL: Some watchdogs detecting issues")
            logger.info("  Recommended: Enhanced monitoring and analysis")
        else:
            logger.info("  ✅ STABLE LEVEL: No significant issues detected")
            logger.info("  Recommended: Continue normal monitoring")
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive watchdog report"""
        logger.info("\n📊 Comprehensive Watchdog Report")
        logger.info("=" * 60)
        
        # Collect summaries from all watchdogs
        entropy_summary = self.entropy_monitor.get_entropy_summary()
        manipulation_summary = self.manipulation_detector.get_manipulation_summary()
        feedback_summary = self.feedback_analyzer.get_feedback_summary()
        drift_summary = self.drift_warner.get_drift_summary()
        health_summary = self.health_monitor.get_health_summary()
        
        # Overall statistics
        total_alerts = (
            entropy_summary.get("total_alerts", 0) +
            manipulation_summary.get("total_patterns_detected", 0) +
            feedback_summary.get("total_loops_detected", 0) +
            drift_summary.get("total_alerts_generated", 0) +
            health_summary.get("total_alerts", 0)
        )
        
        total_measurements = (
            entropy_summary.get("total_measurements", 0) +
            health_summary.get("total_metrics", 0)
        )
        
        # System health assessment
        if total_alerts == 0:
            system_status = "EXCELLENT"
            status_emoji = "🟢"
        elif total_alerts <= 2:
            system_status = "GOOD"
            status_emoji = "🟡"
        elif total_alerts <= 5:
            system_status = "FAIR"
            status_emoji = "🟠"
        elif total_alerts <= 10:
            system_status = "POOR"
            status_emoji = "🔴"
        else:
            system_status = "CRITICAL"
            status_emoji = "⚫"
        
        # Generate report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": self.node_id,
            "system_status": system_status,
            "total_alerts": total_alerts,
            "total_measurements": total_measurements,
            "watchdog_summaries": {
                "entropy_monitor": entropy_summary,
                "manipulation_detector": manipulation_summary,
                "feedback_analyzer": feedback_summary,
                "drift_warner": drift_summary,
                "health_monitor": health_summary
            }
        }
        
        # Display report
        logger.info(f"  System Status: {status_emoji} {system_status}")
        logger.info(f"  Total Alerts: {total_alerts}")
        logger.info(f"  Total Measurements: {total_measurements}")
        logger.info(f"  Report Generated: {report['timestamp']}")
        
        # Save report to file
        try:
            with open("watchdog_report.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info("  📄 Report saved to: watchdog_report.json")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not save report: {e}")
        
        return report


async def main():
    """Main function to run the watchdog demo"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run demo
    demo = WatchdogsDemo("demo_node")
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())

