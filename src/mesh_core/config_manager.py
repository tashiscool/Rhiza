"""
Configuration Manager - Centralized configuration system for the Mesh
Handles loading, validation, and runtime configuration management
"""
import json
import os
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class AxiomConfig:
    """Configuration for AxiomEngine integration"""
    api_nodes: List[str]
    direct_db: bool
    fallback_to_api: bool
    min_confidence: float
    min_status: str
    max_facts: int
    query_timeout: int
    retry_attempts: int
    cache_ttl: int

@dataclass
class IntentConfig:
    """Configuration for intent monitoring system"""
    java_classpath: str
    main_class: str
    confidence_threshold: float
    manipulation_threshold: float
    intent_cache_size: int
    baseline_update_frequency: int

@dataclass
class EmpathyConfig:
    """Configuration for empathy engine"""
    model_path: str
    device: str
    batch_size: int
    max_length: int
    temperature: float
    top_p: float
    cache_models: bool

@dataclass
class KoboldConfig:
    """Configuration for KoboldCpp integration"""
    api_url: str
    api_key: Optional[str]
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    context_length: int

@dataclass
class M4Config:
    """Configuration for M4 Pro optimization"""
    memory_limit_gb: int
    neural_engine: bool
    metal_acceleration: bool
    core_ml: bool
    concurrent_models: int
    cache_strategy: str

@dataclass
class PalmSlabConfig:
    """Configuration for Palm Slab interface"""
    biometric_enabled: bool
    intention_verification: bool
    coercion_detection: bool
    local_storage: bool
    encryption_key: Optional[str]

@dataclass
class MeshConfig:
    """Complete Mesh configuration"""
    axiom: AxiomConfig
    intent: IntentConfig
    empathy: EmpathyConfig
    kobold: KoboldConfig
    m4: M4Config
    palm_slab: PalmSlabConfig
    logging_level: str
    data_dir: str
    temp_dir: str

class ConfigurationManager:
    """Manages Mesh configuration loading, validation, and runtime updates"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_default_config()
        self.config: Optional[MeshConfig] = None
        self._load_config()
    
    def _find_default_config(self) -> str:
        """Find the default configuration file"""
        possible_paths = [
            "mesh_config.json",
            "config/mesh_config.json",
            "../mesh_config.json",
            os.path.expanduser("~/.mesh/config.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Create default config if none exists
        default_path = "mesh_config.json"
        self._create_default_config(default_path)
        return default_path
    
    def _create_default_config(self, path: str):
        """Create a default configuration file"""
        default_config = {
            "axiom_config": {
                "api_nodes": ["http://127.0.0.1:8001"],
                "direct_db": True,
                "fallback_to_api": True,
                "min_confidence": 0.65,
                "min_status": "corroborated",
                "max_facts": 10,
                "query_timeout": 30,
                "retry_attempts": 3,
                "cache_ttl": 3600
            },
            "intent_config": {
                "java_classpath": "./intent_aware_privacy_protection_in_pws/source_code",
                "main_class": "edu.virginia.cs.main.IntentClassifier",
                "confidence_threshold": 0.7,
                "manipulation_threshold": 0.8,
                "intent_cache_size": 1000,
                "baseline_update_frequency": 3600
            },
            "empathy_config": {
                "model_path": "./focused-empathy/models",
                "device": "auto",
                "batch_size": 8,
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "cache_models": True
            },
            "kobold_config": {
                "api_url": "http://127.0.0.1:5001",
                "api_key": None,
                "model_name": "qwen-7b",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "context_length": 4096
            },
            "m4_config": {
                "memory_limit_gb": 40,
                "neural_engine": True,
                "metal_acceleration": True,
                "core_ml": True,
                "concurrent_models": 4,
                "cache_strategy": "adaptive"
            },
            "palm_slab_config": {
                "biometric_enabled": True,
                "intention_verification": True,
                "coercion_detection": True,
                "local_storage": True,
                "encryption_key": None
            },
            "logging_level": "INFO",
            "data_dir": "./mesh_data",
            "temp_dir": "./mesh_temp"
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {path}")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Convert to dataclass structure
            self.config = MeshConfig(
                axiom=AxiomConfig(**config_data["axiom_config"]),
                intent=IntentConfig(**config_data["intent_config"]),
                empathy=EmpathyConfig(**config_data["empathy_config"]),
                kobold=KoboldConfig(**config_data["kobold_config"]),
                m4=M4Config(**config_data["m4_config"]),
                palm_slab=PalmSlabConfig(**config_data["palm_slab_config"]),
                logging_level=config_data.get("logging_level", "INFO"),
                data_dir=config_data.get("data_dir", "./mesh_data"),
                temp_dir=config_data.get("temp_dir", "./mesh_temp")
            )
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> MeshConfig:
        """Get the current configuration"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration at runtime"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        # Update nested configurations
        for section, values in updates.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        # Save updated configuration
        self._save_config()
        logger.info("Configuration updated and saved")
    
    def _save_config(self):
        """Save current configuration to file"""
        if self.config is None:
            return
        
        config_data = {
            "axiom_config": asdict(self.config.axiom),
            "intent_config": asdict(self.config.intent),
            "empathy_config": asdict(self.config.empathy),
            "kobold_config": asdict(self.config.kobold),
            "m4_config": asdict(self.config.m4),
            "palm_slab_config": asdict(self.config.palm_slab),
            "logging_level": self.config.logging_level,
            "data_dir": self.config.data_dir,
            "temp_dir": self.config.temp_dir
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.config is None:
            issues.append("Configuration not loaded")
            return issues
        
        # Validate paths exist
        if not os.path.exists(self.config.intent.java_classpath):
            issues.append(f"Java classpath does not exist: {self.config.intent.java_classpath}")
        
        if not os.path.exists(self.config.empathy.model_path):
            issues.append(f"Empathy model path does not exist: {self.config.empathy.model_path}")
        
        # Validate numeric ranges
        if not 0.0 <= self.config.axiom.min_confidence <= 1.0:
            issues.append("min_confidence must be between 0.0 and 1.0")
        
        if not 0.0 <= self.config.intent.confidence_threshold <= 1.0:
            issues.append("confidence_threshold must be between 0.0 and 1.0")
        
        if self.config.m4.memory_limit_gb > 48:
            issues.append("M4 memory limit cannot exceed 48GB")
        
        return issues
    
    def get_component_config(self, component: str) -> Any:
        """Get configuration for a specific component"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        component_map = {
            "axiom": self.config.axiom,
            "intent": self.config.intent,
            "empathy": self.config.empathy,
            "kobold": self.config.kobold,
            "m4": self.config.m4,
            "palm_slab": self.config.palm_slab
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")
        
        return component_map[component]
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration in specified format"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        if format.lower() == "json":
            return json.dumps(asdict(self.config), indent=2)
        elif format.lower() == "yaml":
            return yaml.dump(asdict(self.config), default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global configuration instance
config_manager = ConfigurationManager()

def get_config() -> MeshConfig:
    """Get the global configuration instance"""
    return config_manager.get_config()

def get_component_config(component: str) -> Any:
    """Get configuration for a specific component"""
    return config_manager.get_component_config(component)