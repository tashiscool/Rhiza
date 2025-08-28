"""User Session Manager"""
class UserSessionManager:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.sessions = {}