#!/usr/bin/env python3
"""
Complete Parameters Guide for The Mesh System
============================================

Shows all parameters for KoboldCpp, GGUF models, and Mesh components
"""

import json
from pathlib import Path

def show_all_parameters():
    """Display all system parameters"""
    
    print("🔧 THE MESH SYSTEM - COMPLETE PARAMETERS GUIDE")
    print("=" * 55)
    print()
    
    # 1. KoboldCpp Configuration Parameters
    print("🤖 1. KOBOLDCPP CONFIGURATION PARAMETERS")
    print("-" * 45)
    
    kobold_params = {
        "api_url": "http://127.0.0.1:5001",
        "api_key": "null (optional authentication)",
        "model_name": "intent-classification-7b-q4_k_m",
        "max_tokens": "2048 (max generation length)",
        "temperature": "0.7 (randomness control 0.0-2.0)",
        "top_p": "0.9 (nucleus sampling 0.0-1.0)",
        "top_k": "40 (top-k sampling)",
        "repeat_penalty": "1.1 (repetition control 1.0+)",
        "context_length": "2048 (context window size)",
        "stop_sequences": "[] (optional stop tokens)",
        "frequency_penalty": "0.0 (frequency penalty)",
        "presence_penalty": "0.0 (presence penalty)"
    }
    
    for param, description in kobold_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 2. GGUF Model Parameters
    print("📦 2. GGUF MODEL PARAMETERS")
    print("-" * 30)
    
    gguf_params = {
        "name": "Model identifier",
        "model_type": "intent_classification|empathy_generation|personal_assistant",
        "path": "Full path to .gguf file",
        "context_length": "2048-8192 (model context window)",
        "threads": "6-8 (CPU threads for inference)",
        "memory_gb": "6-12 (RAM allocation)",
        "port": "5001-5003 (API port)",
        "gpu_layers": "0+ (layers offloaded to GPU)",
        "rope_freq_base": "10000.0 (RoPE frequency base)",
        "rope_freq_scale": "1.0 (RoPE frequency scale)",
        "batch_size": "1-512 (batch processing size)",
        "n_predict": "-1 (prediction length, -1=unlimited)",
        "mmap": "true (memory mapping)",
        "mlock": "false (memory locking)",
        "numa": "false (NUMA optimization)"
    }
    
    for param, description in gguf_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 3. Apple M4 Optimization Parameters
    print("🍎 3. APPLE M4 OPTIMIZATION PARAMETERS")
    print("-" * 40)
    
    m4_params = {
        "memory_limit_gb": "48 (total system memory)",
        "neural_engine": "true (use Neural Engine)",
        "metal_acceleration": "true (use Metal GPU)",
        "core_ml": "true (use Core ML)",
        "concurrent_models": "5 (max simultaneous models)",
        "cache_strategy": "adaptive (memory caching)",
        "unified_memory": "true (shared CPU/GPU memory)",
        "total_memory_gb": "48.0 (available memory)",
        "reserved_memory_gb": "4.0 (system reserved)",
        "max_concurrent_models": "5 (hard limit)",
        "use_neural_engine": "true (18-core Neural Engine)",
        "use_metal_compute": "true (20-core GPU)",
        "dynamic_scaling": "true (adaptive resource allocation)"
    }
    
    for param, description in m4_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 4. Mesh Network Parameters
    print("🌐 4. MESH NETWORK PARAMETERS")
    print("-" * 30)
    
    mesh_params = {
        "node_id": "unique identifier for mesh node",
        "privacy_level": "private|selective|open",
        "trust_threshold": "0.5-1.0 (minimum trust for peers)",
        "consensus_threshold": "0.6-0.9 (agreement required)",
        "reputation_decay": "0.95-0.99 (reputation aging)",
        "max_peers": "10-100 (maximum peer connections)",
        "discovery_interval": "30-300 (seconds between discovery)",
        "heartbeat_interval": "10-60 (seconds between heartbeats)",
        "message_timeout": "5-30 (seconds for message timeout)",
        "retry_attempts": "2-5 (retry failed operations)",
        "cache_ttl": "60-3600 (cache time-to-live seconds)"
    }
    
    for param, description in mesh_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 5. Truth Validation Parameters
    print("⚖️ 5. TRUTH VALIDATION PARAMETERS")
    print("-" * 35)
    
    truth_params = {
        "min_confidence": "0.65 (minimum confidence threshold)",
        "evidence_weight_axiom": "0.4 (axiom validation weight)",
        "evidence_weight_knowledge": "0.3 (knowledge validation weight)",
        "evidence_weight_trust": "0.2 (trust network weight)",
        "evidence_weight_consensus": "0.1 (peer consensus weight)",
        "temporal_decay": "0.95 (aging factor for old evidence)",
        "cross_validation_peers": "3-7 (peers for validation)",
        "confidence_threshold": "0.7 (accept/reject threshold)",
        "manipulation_threshold": "0.8 (manipulation detection)",
        "consensus_quorum": "0.6 (required agreement ratio)"
    }
    
    for param, description in truth_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 6. Privacy & Security Parameters
    print("🔒 6. PRIVACY & SECURITY PARAMETERS")
    print("-" * 35)
    
    security_params = {
        "biometric_enabled": "true (palm slab biometrics)",
        "intention_verification": "true (intent verification)",
        "coercion_detection": "true (detect coercion)",
        "palm_match_threshold": "0.85 (biometric match threshold)",
        "intention_match_threshold": "0.75 (intent match threshold)",
        "coercion_sensitivity": "0.7 (coercion detection sensitivity)",
        "encryption_key": "null (optional encryption)",
        "local_storage": "true (store data locally)",
        "anonymity_enabled": "true (anonymous sharing)",
        "isolation_enabled": "true (node isolation capability)"
    }
    
    for param, description in security_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 7. Personal AI Parameters
    print("🤖 7. PERSONAL AI PARAMETERS")
    print("-" * 30)
    
    personal_params = {
        "character_system": "victoria_steel (character profile)",
        "personalization_level": "0.7 (adaptation level 0.0-1.0)",
        "learning_enabled": "true (continuous learning)",
        "privacy_level": "high|medium|low",
        "intellectual": "0.9 (intellectual trait weight)",
        "empathetic": "0.7 (empathy trait weight)",
        "assertive": "0.8 (assertiveness trait weight)",
        "analytical": "0.9 (analytical trait weight)",
        "supportive": "0.8 (supportiveness trait weight)",
        "proactive_suggestions": "true (offer suggestions)",
        "context_memory_depth": "100 (interaction history depth)"
    }
    
    for param, description in personal_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 8. Resource Management Parameters
    print("⚡ 8. RESOURCE MANAGEMENT PARAMETERS")
    print("-" * 40)
    
    resource_params = {
        "truth_processing_memory": "8.0 GB (truth validation)",
        "intent_classification_memory": "6.0 GB (intent processing)",
        "empathy_generation_memory": "8.0 GB (empathy responses)",
        "personal_assistant_memory": "12.0 GB (personal AI)",
        "content_generation_memory": "10.0 GB (content creation)",
        "truth_processing_priority": "9 (highest priority)",
        "intent_classification_priority": "8 (high priority)",
        "empathy_generation_priority": "7 (medium-high priority)",
        "personal_assistant_priority": "6 (medium priority)",
        "concurrent_sessions_truth": "3 (parallel sessions)",
        "concurrent_sessions_intent": "4 (parallel sessions)",
        "health_check_interval": "60 (seconds)",
        "auto_restart": "true (automatic recovery)"
    }
    
    for param, description in resource_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 9. Advanced System Parameters
    print("🔬 9. ADVANCED SYSTEM PARAMETERS")
    print("-" * 35)
    
    advanced_params = {
        "trust_divergence_threshold": "0.3 (trust anomaly detection)",
        "behavioral_anomaly_threshold": "2.0 (behavior change detection)",
        "consensus_manipulation_threshold": "0.4 (manipulation detection)",
        "isolation_duration_hours": "72 (quarantine time)",
        "retention_days": "365 (data retention period)",
        "max_records_per_info": "1000 (record limit)",
        "context_tracking": "true (track information flow)",
        "confidence_evolution": "true (track confidence changes)",
        "audit_trail_enabled": "true (maintain audit logs)",
        "pattern_matching_enabled": "true (detect patterns)",
        "flow_tracking": "true (track information flow)"
    }
    
    for param, description in advanced_params.items():
        print(f"  • {param}: {description}")
    
    print()
    
    # 10. Integration Parameters
    print("🔗 10. INTEGRATION PARAMETERS")
    print("-" * 30)
    
    integration_params = {
        "leon_voice_enabled": "true (Leon voice integration)",
        "axiom_engine_enabled": "true (AxiomEngine truth validation)",
        "sentient_phases": "4 (all phases: voice, memory, tasks, AI)",
        "empathy_engine_enabled": "true (focused-empathy integration)",
        "privacy_protection_enabled": "true (intent-aware privacy)",
        "java_classpath": "path to intent classifier JAR",
        "main_class": "edu.virginia.cs.main.IntentClassifier",
        "api_nodes": "['http://127.0.0.1:8001', 'http://127.0.0.1:8002']",
        "fallback_to_api": "true (fallback to API if local fails)",
        "direct_db": "true (direct database access)"
    }
    
    for param, description in integration_params.items():
        print(f"  • {param}: {description}")
    
    print()
    print("📊 PARAMETER CATEGORIES SUMMARY:")
    print("=" * 35)
    print("• KoboldCpp: 12 parameters")
    print("• GGUF Models: 15 parameters") 
    print("• Apple M4: 13 parameters")
    print("• Mesh Network: 11 parameters")
    print("• Truth Validation: 10 parameters")
    print("• Privacy/Security: 10 parameters")
    print("• Personal AI: 11 parameters")
    print("• Resource Management: 13 parameters")
    print("• Advanced Systems: 11 parameters")
    print("• Integrations: 10 parameters")
    print()
    print("🎯 TOTAL: 116+ configurable parameters")
    print("🔧 All parameters tunable via config/mesh_config.json")

def show_actual_config():
    """Show current configuration values"""
    
    print("\n" + "="*55)
    print("📄 CURRENT CONFIGURATION VALUES")
    print("="*55)
    
    try:
        config_path = Path("/Users/admin/AI/the-mesh/config/mesh_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Show key sections
            sections = [
                ("KoboldCpp Config", "kobold_config"),
                ("M4 Resource Config", "m4_resource_manager"), 
                ("Model Allocations", "m4_resource_manager", "model_allocations"),
                ("KoboldCpp Models", "kobold_client", "models")
            ]
            
            for section_name, *keys in sections:
                print(f"\n🔧 {section_name.upper()}:")
                print("-" * (len(section_name) + 5))
                
                data = config
                for key in keys:
                    data = data.get(key, {})
                
                if isinstance(data, dict):
                    for param, value in data.items():
                        if isinstance(value, list) and len(value) > 2:
                            print(f"  {param}: [{len(value)} items]")
                        elif isinstance(value, dict):
                            print(f"  {param}: {{{len(value)} settings}}")
                        else:
                            print(f"  {param}: {value}")
                elif isinstance(data, list):
                    print(f"  [{len(data)} items configured]")
                    for i, item in enumerate(data[:2]):  # Show first 2 items
                        if isinstance(item, dict):
                            print(f"    Item {i+1}: {item.get('name', 'Unknown')}")
        else:
            print("❌ Config file not found")
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")

if __name__ == "__main__":
    show_all_parameters()
    show_actual_config()