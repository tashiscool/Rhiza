"""
Mock zeroconf module for development/testing
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

class ServiceInfo:
    """Mock service info"""
    def __init__(self, service_type=None, service_name=None, addresses=None, port=None, properties=None, server=None, **kwargs):
        self.type_ = service_type
        self.name = service_name
        self.addresses = addresses or []
        self.port = port or 0
        self.properties = properties or {}
        self.server = server
        
        # Handle any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class ServiceBrowser:
    """Mock service browser"""
    def __init__(self, zeroconf, type_, listener):
        self.zeroconf = zeroconf
        self.type_ = type_
        self.listener = listener
        self.logger = logger
    
    def start(self):
        """Mock start browsing"""
        self.logger.info("Mock service browser started")
    
    def stop(self):
        """Mock stop browsing"""
        self.logger.info("Mock service browser stopped")
        
    def cancel(self):
        """Mock cancel browsing"""
        self.logger.info("Mock service browser cancelled")

class ServiceListener:
    """Mock service listener"""
    def __init__(self):
        self.logger = logger
    
    def add_service(self, zeroconf, type_, name):
        """Mock service added"""
        self.logger.info(f"Mock service added: {name}")
    
    def remove_service(self, zeroconf, type_, name):
        """Mock service removed"""
        self.logger.info(f"Mock service removed: {name}")
    
    def update_service(self, zeroconf, type_, name):
        """Mock service updated"""
        self.logger.info(f"Mock service updated: {name}")

class Zeroconf:
    """Mock zeroconf service"""
    def __init__(self):
        self.services = {}
        self.logger = logger
    
    def register_service(self, info):
        """Mock service registration"""
        self.services[info.name] = info
        self.logger.info(f"Mock registered service: {info.name}")
        return True
    
    def unregister_service(self, info):
        """Mock service unregistration"""
        if info.name in self.services:
            del self.services[info.name]
            self.logger.info(f"Mock unregistered service: {info.name}")
        return True
        
    def get_service_info(self, type_, name):
        """Mock get service info"""
        return self.services.get(name)
    
    def close(self):
        """Mock close"""
        self.logger.info("Mock zeroconf closed")
