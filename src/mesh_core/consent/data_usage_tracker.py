"""Data Usage Tracker for consent compliance"""
from typing import Dict, List
import time

class DataUsageTracker:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.usage_log: List[Dict] = []
    
    async def log_data_usage(self, user_id: str, data_type: str, purpose: str) -> bool:
        self.usage_log.append({
            "user_id": user_id,
            "data_type": data_type, 
            "purpose": purpose,
            "timestamp": time.time()
        })
        return True