"""
Mesh Amendment System
=====================

Democratic proposal and voting mechanisms for constitutional changes,
enabling the community to evolve governance rules collaboratively.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

from .constitution_engine import ConstitutionEngine, ConstitutionalRule, RuleType, RulePriority

logger = logging.getLogger(__name__)


@dataclass 
class Vote:
    """A vote on an amendment proposal"""
    vote_id: str
    proposal_id: str
    voter_id: str
    vote_type: 'VoteType'
    weight: float  # Voting weight (could be based on trust/reputation)
    timestamp: datetime
    reasoning: Optional[str] = None
    amendments_suggested: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.vote_id:
            self.vote_id = self._generate_vote_id()
    
    def _generate_vote_id(self) -> str:
        """Generate unique vote ID"""
        content = f"{self.proposal_id}{self.voter_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class ProposalStatus(Enum):
    """Status of a constitutional amendment proposal"""
    DRAFT = "draft"                    # Proposal being drafted
    SUBMITTED = "submitted"            # Submitted for review
    UNDER_REVIEW = "under_review"      # Under community review
    VOTING = "voting"                  # Open for voting
    APPROVED = "approved"              # Approved by community
    REJECTED = "rejected"              # Rejected by community
    EXPIRED = "expired"                # Expired without reaching threshold
    WITHDRAWN = "withdrawn"            # Withdrawn by proposer


class VoteType(Enum):
    """Types of votes on proposals"""
    APPROVE = "approve"                # Approve the proposal
    REJECT = "reject"                  # Reject the proposal
    ABSTAIN = "abstain"                # Abstain from voting
    AMEND = "amend"                    # Suggest amendments


class VotingMechanism(Enum):
    """Voting mechanisms for proposals"""
    SIMPLE_MAJORITY = "simple_majority"        # 50% + 1
    SUPER_MAJORITY = "super_majority"          # 2/3 majority
    CONSENSUS = "consensus"                    # 90% agreement
    QUALIFIED_MAJORITY = "qualified_majority"  # 75% agreement


@dataclass
class AmendmentProposal:
    """A proposal to amend the constitution"""
    proposal_id: str
    title: str
    description: str
    proposed_changes: Dict[str, Any]
    proposer_id: str
    status: ProposalStatus
    created_at: datetime
    voting_mechanism: VotingMechanism
    voting_duration: timedelta
    minimum_participation: float  # Minimum % of network that must vote
    approval_threshold: float     # Minimum % of votes needed for approval
    
    # Metadata
    category: str = "general"
    urgency: str = "normal"
    estimated_impact: str = "low"
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Timestamps
    submitted_at: Optional[datetime] = None
    review_started_at: Optional[datetime] = None
    voting_started_at: Optional[datetime] = None
    voting_ended_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Results
    total_votes: int = 0
    approve_votes: int = 0
    reject_votes: int = 0
    abstain_votes: int = 0
    amend_votes: int = 0
    
    def __post_init__(self):
        if not self.proposal_id:
            self.proposal_id = self._generate_proposal_id()
    
    def _generate_proposal_id(self) -> str:
        """Generate unique proposal ID"""
        content = f"{self.title}{self.proposer_id}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary"""
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "proposed_changes": self.proposed_changes,
            "proposer_id": self.proposer_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "voting_mechanism": self.voting_mechanism.value,
            "voting_duration": str(self.voting_duration),
            "minimum_participation": self.minimum_participation,
            "approval_threshold": self.approval_threshold,
            "category": self.category,
            "urgency": self.urgency,
            "estimated_impact": self.estimated_impact,
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "review_started_at": self.review_started_at.isoformat() if self.review_started_at else None,
            "voting_started_at": self.voting_started_at.isoformat() if self.voting_started_at else None,
            "voting_ended_at": self.voting_ended_at.isoformat() if self.voting_ended_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "total_votes": self.total_votes,
            "approve_votes": self.approve_votes,
            "reject_votes": self.reject_votes,
            "abstain_votes": self.abstain_votes,
            "amend_votes": self.amend_votes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AmendmentProposal':
        """Create proposal from dictionary"""
        data = data.copy()
        data['status'] = ProposalStatus(data['status'])
        data['voting_mechanism'] = VotingMechanism(data['voting_mechanism'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['voting_duration'] = timedelta(seconds=int(data['voting_duration'].split(':')[-1]))
        
        # Parse optional timestamps
        for field in ['submitted_at', 'review_started_at', 'voting_started_at', 'voting_ended_at', 'resolved_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class ProposalVote:
    """A single vote on a proposal"""
    vote_id: str
    proposal_id: str
    voter_id: str
    vote_type: VoteType
    timestamp: datetime
    reasoning: Optional[str] = None
    amendments_suggested: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.vote_id:
            self.vote_id = self._generate_vote_id()
    
    def _generate_vote_id(self) -> str:
        """Generate unique vote ID"""
        content = f"{self.proposal_id}{self.voter_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class AmendmentSystem:
    """
    Manages constitutional amendment proposals and voting
    """
    
    def __init__(self, constitution_engine: ConstitutionEngine, node_id: str):
        self.constitution = constitution_engine
        self.node_id = node_id
        self.proposals: Dict[str, AmendmentProposal] = {}
        self.votes: Dict[str, List[ProposalVote]] = {}  # proposal_id -> votes
        self.proposal_history: List[Tuple[datetime, str, str]] = []  # timestamp, action, proposal_id
        
        # Default voting parameters
        self.default_voting_params = {
            "voting_duration": timedelta(days=7),
            "minimum_participation": 0.1,  # 10% of network
            "approval_threshold": 0.75,    # 75% approval needed
            "review_period": timedelta(days=3)
        }
        
        # Proposal categories and their specific rules
        self.category_rules = {
            "critical_security": {
                "voting_duration": timedelta(days=14),
                "minimum_participation": 0.2,
                "approval_threshold": 0.9,
                "review_period": timedelta(days=7)
            },
            "rights_protection": {
                "voting_duration": timedelta(days=10),
                "minimum_participation": 0.15,
                "approval_threshold": 0.8,
                "review_period": timedelta(days=5)
            },
            "structural_change": {
                "voting_duration": timedelta(days=21),
                "minimum_participation": 0.25,
                "approval_threshold": 0.85,
                "review_period": timedelta(days=10)
            }
        }
    
    def create_proposal(self, title: str, description: str, proposed_changes: Dict[str, Any], 
                       proposer_id: str, category: str = "general", urgency: str = "normal") -> str:
        """Create a new amendment proposal"""
        try:
            # Get voting parameters for category
            voting_params = self.category_rules.get(category, self.default_voting_params)
            
            proposal = AmendmentProposal(
                proposal_id="",
                title=title,
                description=description,
                proposed_changes=proposed_changes,
                proposer_id=proposer_id,
                status=ProposalStatus.DRAFT,
                created_at=datetime.utcnow(),
                voting_mechanism=VotingMechanism.QUALIFIED_MAJORITY,
                voting_duration=voting_params["voting_duration"],
                minimum_participation=voting_params["minimum_participation"],
                approval_threshold=voting_params["approval_threshold"],
                category=category,
                urgency=urgency
            )
            
            self.proposals[proposal.proposal_id] = proposal
            self.votes[proposal.proposal_id] = []
            self.proposal_history.append((datetime.utcnow(), "created", proposal.proposal_id))
            
            logger.info(f"Created amendment proposal: {title} by {proposer_id}")
            return proposal.proposal_id
            
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            return ""
    
    def submit_proposal(self, proposal_id: str) -> bool:
        """Submit a proposal for community review"""
        if proposal_id not in self.proposals:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.DRAFT:
            logger.warning(f"Proposal {proposal_id} is not in draft status")
            return False
        
        # Validate proposal
        if not self._validate_proposal(proposal):
            logger.error(f"Proposal {proposal_id} validation failed")
            return False
        
        # Change status to submitted
        proposal.status = ProposalStatus.SUBMITTED
        proposal.submitted_at = datetime.utcnow()
        
        self.proposal_history.append((datetime.utcnow(), "submitted", proposal_id))
        logger.info(f"Proposal {proposal_id} submitted for review")
        return True
    
    def start_review(self, proposal_id: str) -> bool:
        """Start the community review period"""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.SUBMITTED:
            return False
        
        proposal.status = ProposalStatus.UNDER_REVIEW
        proposal.review_started_at = datetime.utcnow()
        
        self.proposal_history.append((datetime.utcnow(), "review_started", proposal_id))
        logger.info(f"Review started for proposal {proposal_id}")
        return True
    
    def start_voting(self, proposal_id: str) -> bool:
        """Start the voting period for a proposal"""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.UNDER_REVIEW:
            return False
        
        proposal.status = ProposalStatus.VOTING
        proposal.voting_started_at = datetime.utcnow()
        
        self.proposal_history.append((datetime.utcnow(), "voting_started", proposal_id))
        logger.info(f"Voting started for proposal {proposal_id}")
        return True
    
    def cast_vote(self, proposal_id: str, voter_id: str, vote_type: VoteType, 
                  reasoning: Optional[str] = None, amendments_suggested: Optional[List[str]] = None) -> bool:
        """Cast a vote on a proposal"""
        if proposal_id not in self.proposals:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.VOTING:
            logger.error(f"Proposal {proposal_id} is not open for voting")
            return False
        
        # Check if voter has already voted
        existing_vote = self._get_existing_vote(proposal_id, voter_id)
        if existing_vote:
            logger.warning(f"Voter {voter_id} has already voted on proposal {proposal_id}")
            return False
        
        # Create vote
        vote = ProposalVote(
            vote_id="",
            proposal_id=proposal_id,
            voter_id=voter_id,
            vote_type=vote_type,
            timestamp=datetime.utcnow(),
            reasoning=reasoning,
            amendments_suggested=amendments_suggested
        )
        
        # Add vote to proposal
        self.votes[proposal_id].append(vote)
        
        # Update proposal vote counts
        self._update_vote_counts(proposal_id)
        
        logger.info(f"Vote cast on proposal {proposal_id}: {voter_id} voted {vote_type.value}")
        return True
    
    def _get_existing_vote(self, proposal_id: str, voter_id: str) -> Optional[ProposalVote]:
        """Get existing vote by a voter on a proposal"""
        for vote in self.votes.get(proposal_id, []):
            if vote.voter_id == voter_id:
                return vote
        return None
    
    def _update_vote_counts(self, proposal_id: str):
        """Update vote counts for a proposal"""
        proposal = self.proposals[proposal_id]
        votes = self.votes.get(proposal_id, [])
        
        proposal.total_votes = len(votes)
        proposal.approve_votes = len([v for v in votes if v.vote_type == VoteType.APPROVE])
        proposal.reject_votes = len([v for v in votes if v.vote_type == VoteType.REJECT])
        proposal.abstain_votes = len([v for v in votes if v.vote_type == VoteType.ABSTAIN])
        proposal.amend_votes = len([v for v in votes if v.vote_type == VoteType.AMEND])
    
    def check_voting_results(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Check if voting has concluded and determine results"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.VOTING:
            return None
        
        # Check if voting period has ended
        if proposal.voting_started_at and datetime.utcnow() > proposal.voting_started_at + proposal.voting_duration:
            return self._finalize_voting(proposal_id)
        
        # Check if minimum participation and approval threshold are met
        if self._check_voting_success(proposal):
            return self._finalize_voting(proposal_id)
        
        return None
    
    def _check_voting_success(self, proposal: AmendmentProposal) -> bool:
        """Check if voting has succeeded based on thresholds"""
        # Check minimum participation
        if proposal.total_votes < proposal.minimum_participation:
            return False
        
        # Check approval threshold
        if proposal.approve_votes < proposal.total_votes * proposal.approval_threshold:
            return False
        
        return True
    
    def _finalize_voting(self, proposal_id: str) -> Dict[str, Any]:
        """Finalize voting and determine outcome"""
        proposal = self.proposals[proposal_id]
        proposal.voting_ended_at = datetime.utcnow()
        
        # Determine outcome
        if self._check_voting_success(proposal):
            proposal.status = ProposalStatus.APPROVED
            outcome = "approved"
            self._implement_approved_proposal(proposal)
        else:
            proposal.status = ProposalStatus.REJECTED
            outcome = "rejected"
        
        proposal.resolved_at = datetime.utcnow()
        
        self.proposal_history.append((datetime.utcnow(), f"voting_{outcome}", proposal_id))
        logger.info(f"Proposal {proposal_id} voting finalized: {outcome}")
        
        return {
            "proposal_id": proposal_id,
            "outcome": outcome,
            "final_vote_counts": {
                "total": proposal.total_votes,
                "approve": proposal.approve_votes,
                "reject": proposal.reject_votes,
                "abstain": proposal.abstain_votes,
                "amend": proposal.amend_votes
            },
            "thresholds_met": {
                "participation": proposal.total_votes >= proposal.minimum_participation,
                "approval": proposal.approve_votes >= proposal.total_votes * proposal.approval_threshold
            }
        }
    
    def _implement_approved_proposal(self, proposal: AmendmentProposal):
        """Implement an approved constitutional amendment"""
        try:
            changes = proposal.proposed_changes
            
            if "add_rule" in changes:
                # Add new constitutional rule
                rule_data = changes["add_rule"]
                rule = ConstitutionalRule.from_dict(rule_data)
                self.constitution.add_rule(rule)
                logger.info(f"Added new constitutional rule: {rule.title}")
            
            if "modify_rule" in changes:
                # Modify existing rule
                rule_id = changes["modify_rule"]["rule_id"]
                updates = changes["modify_rule"]["updates"]
                self.constitution.update_rule(rule_id, updates, proposal.proposer_id)
                logger.info(f"Modified constitutional rule: {rule_id}")
            
            if "remove_rule" in changes:
                # Remove rule
                rule_id = changes["remove_rule"]["rule_id"]
                reason = changes["remove_rule"].get("reason", "Approved by community vote")
                self.constitution.remove_rule(rule_id, reason)
                logger.info(f"Removed constitutional rule: {rule_id}")
            
            logger.info(f"Successfully implemented approved proposal: {proposal.title}")
            
        except Exception as e:
            logger.error(f"Failed to implement approved proposal {proposal.proposal_id}: {e}")
    
    def _validate_proposal(self, proposal: AmendmentProposal) -> bool:
        """Validate a proposal before submission"""
        # Check for conflicts with existing rules
        if self._check_proposal_conflicts(proposal):
            return False
        
        # Check dependencies
        if not self._check_proposal_dependencies(proposal):
            return False
        
        # Validate proposed changes format
        if not self._validate_proposed_changes(proposal.proposed_changes):
            return False
        
        return True
    
    def _check_proposal_conflicts(self, proposal: AmendmentProposal) -> bool:
        """Check if proposal conflicts with existing rules"""
        # TODO: Implement conflict detection logic
        return False
    
    def _check_proposal_dependencies(self, proposal: AmendmentProposal) -> bool:
        """Check if proposal dependencies are met"""
        # TODO: Implement dependency checking logic
        return True
    
    def _validate_proposed_changes(self, changes: Dict[str, Any]) -> bool:
        """Validate the format of proposed changes"""
        required_fields = {
            "add_rule": ["rule_type", "priority", "title", "description", "constraints", "enforcement_mechanism"],
            "modify_rule": ["rule_id", "updates"],
            "remove_rule": ["rule_id"]
        }
        
        for change_type, fields in required_fields.items():
            if change_type in changes:
                if not all(field in changes[change_type] for field in fields):
                    return False
        
        return True
    
    def get_proposal_summary(self) -> Dict[str, Any]:
        """Get summary of all proposals"""
        total_proposals = len(self.proposals)
        proposals_by_status = {}
        
        for proposal in self.proposals.values():
            status = proposal.status.value
            proposals_by_status[status] = proposals_by_status.get(status, 0) + 1
        
        return {
            "total_proposals": total_proposals,
            "proposals_by_status": proposals_by_status,
            "recent_proposals": self._get_recent_proposals(10),
            "active_voting": len([p for p in self.proposals.values() if p.status == ProposalStatus.VOTING])
        }
    
    def _get_recent_proposals(self, count: int) -> List[Dict[str, Any]]:
        """Get recent proposals"""
        sorted_proposals = sorted(self.proposals.values(), key=lambda p: p.created_at, reverse=True)
        recent_proposals = sorted_proposals[:count]
        
        return [
            {
                "proposal_id": p.proposal_id,
                "title": p.title,
                "status": p.status.value,
                "created_at": p.created_at.isoformat(),
                "proposer_id": p.proposer_id,
                "category": p.category
            }
            for p in recent_proposals
        ]
    
    def get_proposal_details(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific proposal"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        votes = self.votes.get(proposal_id, [])
        
        return {
            "proposal": proposal.to_dict(),
            "votes": [
                {
                    "voter_id": v.voter_id,
                    "vote_type": v.vote_type.value,
                    "timestamp": v.timestamp.isoformat(),
                    "reasoning": v.reasoning,
                    "amendments_suggested": v.amendments_suggested
                }
                for v in votes
            ],
            "voting_progress": {
                "total_votes": proposal.total_votes,
                "approve_votes": proposal.approve_votes,
                "reject_votes": proposal.reject_votes,
                "abstain_votes": proposal.abstain_votes,
                "amend_votes": proposal.amend_votes
            }
        }

