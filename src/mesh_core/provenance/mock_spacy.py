"""
Mock spacy module for development/testing
"""
import logging

logger = logging.getLogger(__name__)

class MockNLP:
    """Mock NLP model"""
    def __init__(self):
        self.logger = logger
    
    def __call__(self, text):
        """Mock text processing"""
        return MockDoc(text)
    
    def pipe(self, texts):
        """Mock batch processing"""
        for text in texts:
            yield self(text)

class MockDoc:
    """Mock processed document"""
    def __init__(self, text):
        self.text = text
        self.ents = []
        self.vector = [0.0] * 300  # Mock 300-dim vector
    
    def __len__(self):
        return len(self.text.split())

# Mock spacy module
nlp = MockNLP()
load = lambda model: MockNLP()
