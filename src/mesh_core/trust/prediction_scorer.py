"""
Prediction Scoring System
========================

Tracks and scores the accuracy of predictions made by nodes,
building a track record of prediction performance to assess
reliability and expertise in specific domains.
"""

import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class PredictionStatus(Enum):
    """Status of predictions"""
    PENDING = "pending"           # Awaiting verification
    VERIFIED = "verified"         # Outcome verified
    EXPIRED = "expired"          # Prediction expired
    CANCELLED = "cancelled"      # Prediction cancelled

class OutcomeType(Enum):
    """Types of prediction outcomes"""
    BINARY = "binary"            # True/False outcome
    NUMERIC = "numeric"          # Numeric value outcome  
    CATEGORICAL = "categorical"   # Category outcome
    RANKING = "ranking"          # Ranking outcome

class AlignmentMetrics(Enum):
    """Metrics for measuring prediction alignment"""
    ACCURACY = "accuracy"         # Overall accuracy
    PRECISION = "precision"       # Precision for positive predictions
    RECALL = "recall"            # Recall for positive predictions
    F1_SCORE = "f1"              # F1 score
    CALIBRATION = "calibration"   # Confidence calibration
    DISCRIMINATION = "discrimination"  # Ability to discriminate

@dataclass
class PredictionRecord:
    """Record of a prediction made by a node"""
    prediction_id: str
    predictor_id: str
    prediction_text: str
    predicted_outcome: Any
    confidence: float
    outcome_type: OutcomeType
    domain: str
    made_at: float
    resolution_time: Optional[float]
    actual_outcome: Optional[Any]
    status: PredictionStatus
    verification_source: Optional[str]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['outcome_type'] = self.outcome_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionRecord':
        data['outcome_type'] = OutcomeType(data['outcome_type'])
        data['status'] = PredictionStatus(data['status'])
        return cls(**data)

@dataclass
class PredictionScore:
    """Prediction scoring results for a node"""
    predictor_id: str
    domain: str
    total_predictions: int
    verified_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    calibration_score: float
    confidence_alignment: float
    last_updated: float
    time_period_days: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

class PredictionScorer:
    """
    Prediction scoring and tracking system
    
    Maintains records of predictions made by nodes and scores
    their accuracy when outcomes are verified.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.prediction_records: Dict[str, PredictionRecord] = {}
        self.predictor_scores: Dict[Tuple[str, str], PredictionScore] = {}  # (predictor_id, domain) -> score
        self.domain_statistics: Dict[str, Dict] = {}
        self.scoring_parameters: Dict[str, float] = self._init_scoring_parameters()
        
    def _init_scoring_parameters(self) -> Dict[str, float]:
        """Initialize scoring parameters"""
        return {
            'min_predictions_for_score': 5,      # Minimum predictions needed for scoring
            'confidence_tolerance': 0.1,         # Tolerance for calibration scoring
            'recent_weight': 0.7,               # Weight for recent predictions
            'expertise_threshold': 0.8,          # Threshold for expert status
            'calibration_bins': 10,             # Number of bins for calibration
            'temporal_decay_rate': 0.1          # Daily decay rate for old predictions
        }
    
    def _generate_prediction_id(self, predictor_id: str) -> str:
        """Generate unique prediction ID"""
        import hashlib
        data = f"{predictor_id}_{time.time()}_{self.node_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def record_prediction(
        self,
        predictor_id: str,
        prediction_text: str,
        predicted_outcome: Any,
        confidence: float,
        outcome_type: OutcomeType,
        domain: str = "general",
        resolution_time: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> PredictionRecord:
        """Record a new prediction"""
        
        if metadata is None:
            metadata = {}
            
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Create prediction record
        prediction = PredictionRecord(
            prediction_id=self._generate_prediction_id(predictor_id),
            predictor_id=predictor_id,
            prediction_text=prediction_text,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            outcome_type=outcome_type,
            domain=domain,
            made_at=time.time(),
            resolution_time=resolution_time,
            actual_outcome=None,
            status=PredictionStatus.PENDING,
            verification_source=None,
            metadata=metadata
        )
        
        # Store prediction
        self.prediction_records[prediction.prediction_id] = prediction
        
        logger.info(f"Recorded prediction {prediction.prediction_id} by {predictor_id} in domain {domain}")
        return prediction
    
    async def verify_prediction(
        self,
        prediction_id: str,
        actual_outcome: Any,
        verification_source: str
    ) -> bool:
        """Verify a prediction with actual outcome"""
        
        prediction = self.prediction_records.get(prediction_id)
        if not prediction:
            logger.error(f"Prediction {prediction_id} not found")
            return False
        
        if prediction.status != PredictionStatus.PENDING:
            logger.warning(f"Prediction {prediction_id} already verified or expired")
            return False
        
        # Update prediction with actual outcome
        prediction.actual_outcome = actual_outcome
        prediction.status = PredictionStatus.VERIFIED
        prediction.verification_source = verification_source
        
        # Update scores for the predictor
        await self._update_predictor_scores(prediction)
        
        logger.info(f"Verified prediction {prediction_id}: predicted {prediction.predicted_outcome}, actual {actual_outcome}")
        return True
    
    async def _update_predictor_scores(self, verified_prediction: PredictionRecord):
        """Update predictor scores based on verified prediction"""
        
        predictor_id = verified_prediction.predictor_id
        domain = verified_prediction.domain
        key = (predictor_id, domain)
        
        # Get all verified predictions for this predictor in this domain
        verified_predictions = [
            p for p in self.prediction_records.values()
            if (p.predictor_id == predictor_id and 
                p.domain == domain and 
                p.status == PredictionStatus.VERIFIED)
        ]
        
        # Need minimum predictions for scoring
        if len(verified_predictions) < self.scoring_parameters['min_predictions_for_score']:
            return
        
        # Calculate scores
        scores = await self._calculate_prediction_scores(verified_predictions)
        
        # Store scores
        self.predictor_scores[key] = PredictionScore(
            predictor_id=predictor_id,
            domain=domain,
            total_predictions=len([p for p in self.prediction_records.values() 
                                 if p.predictor_id == predictor_id and p.domain == domain]),
            verified_predictions=len(verified_predictions),
            accuracy=scores['accuracy'],
            precision=scores['precision'],
            recall=scores['recall'],
            f1_score=scores['f1_score'],
            calibration_score=scores['calibration'],
            confidence_alignment=scores['confidence_alignment'],
            last_updated=time.time(),
            time_period_days=scores['time_period_days']
        )
        
        logger.info(f"Updated scores for {predictor_id} in {domain}: accuracy={scores['accuracy']:.3f}")
    
    async def _calculate_prediction_scores(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate various prediction performance scores"""
        
        if not predictions:
            return self._empty_scores()
        
        # Time-weighted analysis
        current_time = time.time()
        weights = []
        outcomes_correct = []
        confidences = []
        
        for pred in predictions:
            # Calculate time weight (more recent = higher weight)
            days_ago = (current_time - pred.made_at) / 86400
            weight = math.exp(-self.scoring_parameters['temporal_decay_rate'] * days_ago)
            weights.append(weight)
            
            # Check if prediction was correct
            correct = self._is_prediction_correct(pred)
            outcomes_correct.append(correct)
            confidences.append(pred.confidence)
        
        # Calculate weighted accuracy
        weighted_accuracy = sum(w * c for w, c in zip(weights, outcomes_correct)) / sum(weights)
        
        # Calculate precision, recall for binary outcomes
        if predictions[0].outcome_type == OutcomeType.BINARY:
            true_positives = sum(1 for p in predictions if p.predicted_outcome == True and p.actual_outcome == True)
            false_positives = sum(1 for p in predictions if p.predicted_outcome == True and p.actual_outcome == False)
            false_negatives = sum(1 for p in predictions if p.predicted_outcome == False and p.actual_outcome == True)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = weighted_accuracy  # For non-binary, use accuracy
            recall = weighted_accuracy
            f1_score = weighted_accuracy
        
        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(predictions)
        
        # Calculate confidence alignment
        confidence_alignment = self._calculate_confidence_alignment(predictions)
        
        # Time period covered
        if len(predictions) > 1:
            time_span = max(p.made_at for p in predictions) - min(p.made_at for p in predictions)
            time_period_days = time_span / 86400
        else:
            time_period_days = 0
        
        return {
            'accuracy': weighted_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'calibration': calibration_score,
            'confidence_alignment': confidence_alignment,
            'time_period_days': time_period_days
        }
    
    def _is_prediction_correct(self, prediction: PredictionRecord) -> bool:
        """Check if a prediction was correct"""
        
        if prediction.outcome_type == OutcomeType.BINARY:
            return prediction.predicted_outcome == prediction.actual_outcome
        elif prediction.outcome_type == OutcomeType.NUMERIC:
            # For numeric, consider correct if within reasonable tolerance
            if isinstance(prediction.actual_outcome, (int, float)) and isinstance(prediction.predicted_outcome, (int, float)):
                tolerance = abs(prediction.actual_outcome) * 0.1  # 10% tolerance
                return abs(prediction.predicted_outcome - prediction.actual_outcome) <= tolerance
        elif prediction.outcome_type == OutcomeType.CATEGORICAL:
            return prediction.predicted_outcome == prediction.actual_outcome
        
        return False
    
    def _calculate_calibration_score(self, predictions: List[PredictionRecord]) -> float:
        """Calculate calibration score (how well confidence matches accuracy)"""
        
        if not predictions:
            return 0.5
        
        # Group predictions by confidence bins
        num_bins = int(self.scoring_parameters['calibration_bins'])
        bins = [[] for _ in range(num_bins)]
        
        for pred in predictions:
            bin_index = min(num_bins - 1, int(pred.confidence * num_bins))
            bins[bin_index].append(pred)
        
        # Calculate calibration error
        calibration_errors = []
        
        for i, bin_predictions in enumerate(bins):
            if not bin_predictions:
                continue
            
            # Expected confidence for this bin
            bin_center = (i + 0.5) / num_bins
            
            # Actual accuracy in this bin
            correct_count = sum(1 for p in bin_predictions if self._is_prediction_correct(p))
            bin_accuracy = correct_count / len(bin_predictions)
            
            # Calibration error for this bin
            error = abs(bin_center - bin_accuracy)
            calibration_errors.append(error)
        
        if not calibration_errors:
            return 0.5
        
        # Return calibration score (1 - average error)
        avg_error = statistics.mean(calibration_errors)
        return max(0.0, 1.0 - avg_error)
    
    def _calculate_confidence_alignment(self, predictions: List[PredictionRecord]) -> float:
        """Calculate how well confidence aligns with actual performance"""
        
        if len(predictions) < 2:
            return 0.5
        
        # Calculate correlation between confidence and correctness
        confidences = [p.confidence for p in predictions]
        correctness = [1.0 if self._is_prediction_correct(p) else 0.0 for p in predictions]
        
        # Simple correlation calculation
        mean_confidence = statistics.mean(confidences)
        mean_correctness = statistics.mean(correctness)
        
        numerator = sum((c - mean_confidence) * (correct - mean_correctness) 
                       for c, correct in zip(confidences, correctness))
        
        conf_variance = sum((c - mean_confidence) ** 2 for c in confidences)
        corr_variance = sum((correct - mean_correctness) ** 2 for correct in correctness)
        
        if conf_variance == 0 or corr_variance == 0:
            return 0.5
        
        correlation = numerator / math.sqrt(conf_variance * corr_variance)
        
        # Convert correlation to 0-1 score
        return (correlation + 1.0) / 2.0
    
    def _empty_scores(self) -> Dict[str, float]:
        """Return empty scores structure"""
        return {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'calibration': 0.5,
            'confidence_alignment': 0.5,
            'time_period_days': 0
        }
    
    async def get_predictor_score(self, predictor_id: str, domain: str = "general") -> Optional[PredictionScore]:
        """Get prediction score for predictor in domain"""
        return self.predictor_scores.get((predictor_id, domain))
    
    async def get_predictor_predictions(self, predictor_id: str, domain: Optional[str] = None, limit: Optional[int] = None) -> List[PredictionRecord]:
        """Get predictions made by a predictor"""
        
        predictions = [
            p for p in self.prediction_records.values()
            if p.predictor_id == predictor_id and (domain is None or p.domain == domain)
        ]
        
        # Sort by timestamp (most recent first)
        predictions.sort(key=lambda x: x.made_at, reverse=True)
        
        if limit:
            return predictions[:limit]
        return predictions
    
    async def get_domain_experts(self, domain: str, min_accuracy: float = 0.8, limit: int = 10) -> List[Tuple[str, PredictionScore]]:
        """Get top experts in a domain"""
        
        domain_scores = [
            (predictor_id, score) for (predictor_id, d), score in self.predictor_scores.items()
            if d == domain and score.accuracy >= min_accuracy
        ]
        
        # Sort by accuracy (highest first)
        domain_scores.sort(key=lambda x: x[1].accuracy, reverse=True)
        
        return domain_scores[:limit]
    
    async def get_prediction_statistics(self, domain: Optional[str] = None) -> Dict:
        """Get prediction system statistics"""
        
        # Filter predictions by domain if specified
        if domain:
            predictions = [p for p in self.prediction_records.values() if p.domain == domain]
        else:
            predictions = list(self.prediction_records.values())
        
        if not predictions:
            return {'total_predictions': 0, 'domain': domain}
        
        # Count by status
        status_counts = {}
        for pred in predictions:
            status = pred.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by outcome type
        outcome_type_counts = {}
        for pred in predictions:
            outcome_type = pred.outcome_type.value
            outcome_type_counts[outcome_type] = outcome_type_counts.get(outcome_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = statistics.mean(p.confidence for p in predictions) if predictions else 0.0
        
        # Calculate verification rate
        verified_count = status_counts.get('verified', 0)
        verification_rate = verified_count / len(predictions) if predictions else 0.0
        
        return {
            'total_predictions': len(predictions),
            'by_status': status_counts,
            'by_outcome_type': outcome_type_counts,
            'average_confidence': avg_confidence,
            'verification_rate': verification_rate,
            'unique_predictors': len(set(p.predictor_id for p in predictions)),
            'domains': len(set(p.domain for p in predictions)),
            'observer_id': self.node_id,
            'domain_filter': domain
        }
    
    async def expire_old_predictions(self, max_age_days: int = 365):
        """Mark old unverified predictions as expired"""
        
        cutoff_time = time.time() - (max_age_days * 86400)
        expired_count = 0
        
        for prediction in self.prediction_records.values():
            if (prediction.status == PredictionStatus.PENDING and 
                prediction.made_at < cutoff_time):
                prediction.status = PredictionStatus.EXPIRED
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Marked {expired_count} old predictions as expired")
        
        return expired_count
    
    def export_prediction_data(self) -> Dict:
        """Export prediction data for backup or analysis"""
        
        return {
            'node_id': self.node_id,
            'predictions': [pred.to_dict() for pred in self.prediction_records.values()],
            'scores': {
                f"{predictor_id}_{domain}": score.to_dict()
                for (predictor_id, domain), score in self.predictor_scores.items()
            },
            'exported_at': time.time()
        }