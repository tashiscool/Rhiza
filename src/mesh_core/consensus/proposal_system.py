"""
Consensus Proposal System
========================

Creates and manages governance proposals for The Mesh constitutional democracy.
Handles proposal lifecycle from creation through voting to implementation.
"""

import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ProposalType(Enum):
    """Types of governance proposals"""
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    PROTOCOL_CHANGE = "protocol_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMMUNITY_STANDARD = "community_standard"
    EMERGENCY_ACTION = "emergency_action"
    MEMBERSHIP_DECISION = "membership_decision"
    TECHNICAL_UPGRADE = "technical_upgrade"
    POLICY_CHANGE = "policy_change"

class ProposalStatus(Enum):
    """Proposal lifecycle status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VOTING_ACTIVE = "voting_active"
    VOTING_CLOSED = "voting_closed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

class VotingMethod(Enum):
    """Voting methodologies"""
    SIMPLE_MAJORITY = "simple_majority"
    SUPERMAJORITY_60 = "supermajority_60"
    SUPERMAJORITY_66 = "supermajority_66"
    SUPERMAJORITY_75 = "supermajority_75"
    UNANIMOUS = "unanimous"
    RANKED_CHOICE = "ranked_choice"
    DELEGATED_VOTING = "delegated_voting"
    QUADRATIC_VOTING = "quadratic_voting"

@dataclass
class Vote:
    """Individual vote record"""
    voter_id: str
    proposal_id: str
    vote_value: str  # "yes", "no", "abstain", or ranked choices
    voting_power: float
    timestamp: float
    reasoning: Optional[str] = None
    delegation_chain: Optional[List[str]] = None
    signature: Optional[str] = None

@dataclass
class Proposal:
    """Governance proposal"""
    proposal_id: str
    title: str
    description: str
    proposal_type: ProposalType
    proposer_id: str
    created_at: float
    status: ProposalStatus
    voting_method: VotingMethod
    voting_starts_at: float
    voting_ends_at: float
    required_threshold: float
    content_hash: str
    amendments: List[Dict] = None
    supporting_evidence: List[Dict] = None
    impact_assessment: Optional[Dict] = None
    implementation_plan: Optional[Dict] = None
    
    def __post_init__(self):
        if self.amendments is None:
            self.amendments = []
        if self.supporting_evidence is None:
            self.supporting_evidence = []

    def is_voting_active(self) -> bool:
        """Check if voting is currently active"""
        now = time.time()
        return (self.voting_starts_at <= now <= self.voting_ends_at and 
                self.status == ProposalStatus.VOTING_ACTIVE)

    def get_content_hash(self) -> str:
        """Generate content hash for integrity verification"""
        content = f"{self.title}|{self.description}|{self.proposal_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()

class ProposalSystem:
    """Manages the proposal creation and lifecycle system"""
    
    def __init__(self, node_id: str, governance_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = governance_config or {}
        self.proposals: Dict[str, Proposal] = {}
        self.proposal_votes: Dict[str, List[Vote]] = {}
        self.user_reputations: Dict[str, float] = {}
        self.delegation_chains: Dict[str, str] = {}  # delegator -> delegate
        
        logger.info(f"ProposalSystem initialized for node {node_id}")

    async def create_proposal(
        self,
        title: str,
        description: str,
        proposal_type: ProposalType,
        proposer_id: str,
        voting_duration_hours: int = 168,  # 7 days default
        voting_method: VotingMethod = VotingMethod.SIMPLE_MAJORITY,
        supporting_evidence: Optional[List[Dict]] = None,
        implementation_plan: Optional[Dict] = None
    ) -> str:
        """Create a new governance proposal"""
        try:
            proposal_id = self._generate_proposal_id(title, proposer_id)
            
            # Determine required threshold based on proposal type and method
            required_threshold = self._get_required_threshold(proposal_type, voting_method)
            
            now = time.time()
            voting_starts_at = now + (24 * 3600)  # 24 hour review period
            voting_ends_at = voting_starts_at + (voting_duration_hours * 3600)
            
            proposal = Proposal(
                proposal_id=proposal_id,
                title=title,
                description=description,
                proposal_type=proposal_type,
                proposer_id=proposer_id,
                created_at=now,
                status=ProposalStatus.SUBMITTED,
                voting_method=voting_method,
                voting_starts_at=voting_starts_at,
                voting_ends_at=voting_ends_at,
                required_threshold=required_threshold,
                content_hash=hashlib.sha256(f"{title}|{description}|{proposal_type.value}".encode()).hexdigest(),
                supporting_evidence=supporting_evidence or [],
                implementation_plan=implementation_plan
            )
            
            # Validate proposal meets requirements
            validation_result = await self._validate_proposal(proposal, proposer_id)
            if not validation_result["valid"]:
                raise ValueError(f"Proposal validation failed: {validation_result['reason']}")
            
            self.proposals[proposal_id] = proposal
            self.proposal_votes[proposal_id] = []
            
            # Generate impact assessment
            proposal.impact_assessment = await self._generate_impact_assessment(proposal)
            
            logger.info(f"Proposal created: {proposal_id} by {proposer_id}")
            return proposal_id
            
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            raise

    async def amend_proposal(
        self,
        proposal_id: str,
        amendment: Dict,
        proposer_id: str
    ) -> bool:
        """Add an amendment to an existing proposal"""
        try:
            if proposal_id not in self.proposals:
                return False
                
            proposal = self.proposals[proposal_id]
            
            # Only allow amendments during draft or review phase
            if proposal.status not in [ProposalStatus.DRAFT, ProposalStatus.UNDER_REVIEW]:
                return False
            
            # Only original proposer can amend initially
            if proposal.proposer_id != proposer_id:
                return False
            
            amendment_record = {
                "amendment_id": hashlib.sha256(f"{proposal_id}{time.time()}".encode()).hexdigest()[:16],
                "proposer_id": proposer_id,
                "timestamp": time.time(),
                "content": amendment,
                "rationale": amendment.get("rationale", "")
            }
            
            proposal.amendments.append(amendment_record)
            
            # Update content hash to reflect changes
            content = f"{proposal.title}|{proposal.description}|{json.dumps(proposal.amendments)}"
            proposal.content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            logger.info(f"Amendment added to proposal {proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to amend proposal {proposal_id}: {e}")
            return False

    async def submit_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_value: str,
        reasoning: Optional[str] = None,
        delegation_source: Optional[str] = None
    ) -> bool:
        """Submit a vote on a proposal"""
        try:
            if proposal_id not in self.proposals:
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Check if voting is active
            if not proposal.is_voting_active():
                return False
            
            # Calculate voting power (reputation-weighted)
            voting_power = self._calculate_voting_power(voter_id, proposal_id)
            
            # Handle delegation if applicable
            delegation_chain = None
            if delegation_source:
                delegation_chain = self._resolve_delegation_chain(delegation_source, voter_id)
            
            vote = Vote(
                voter_id=voter_id,
                proposal_id=proposal_id,
                vote_value=vote_value,
                voting_power=voting_power,
                timestamp=time.time(),
                reasoning=reasoning,
                delegation_chain=delegation_chain,
                signature=self._sign_vote(voter_id, proposal_id, vote_value)
            )
            
            # Remove any existing vote from this voter
            self.proposal_votes[proposal_id] = [
                v for v in self.proposal_votes[proposal_id] 
                if v.voter_id != voter_id
            ]
            
            # Add the new vote
            self.proposal_votes[proposal_id].append(vote)
            
            logger.info(f"Vote submitted: {voter_id} on {proposal_id} = {vote_value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit vote: {e}")
            return False

    async def get_proposal_results(self, proposal_id: str) -> Optional[Dict]:
        """Get current results for a proposal"""
        try:
            if proposal_id not in self.proposals:
                return None
            
            proposal = self.proposals[proposal_id]
            votes = self.proposal_votes[proposal_id]
            
            # Calculate results based on voting method
            results = self._calculate_results(proposal, votes)
            
            return {
                "proposal_id": proposal_id,
                "status": proposal.status.value,
                "voting_active": proposal.is_voting_active(),
                "total_votes": len(votes),
                "results": results,
                "threshold_met": results.get("threshold_met", False),
                "voting_ends_at": proposal.voting_ends_at,
                "time_remaining": max(0, proposal.voting_ends_at - time.time()) if proposal.is_voting_active() else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get proposal results: {e}")
            return None

    async def finalize_proposal(self, proposal_id: str) -> bool:
        """Finalize a proposal after voting period ends"""
        try:
            if proposal_id not in self.proposals:
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Only finalize if voting has ended
            if time.time() < proposal.voting_ends_at:
                return False
            
            results = await self.get_proposal_results(proposal_id)
            if not results:
                return False
            
            # Update proposal status based on results
            if results["results"]["threshold_met"]:
                proposal.status = ProposalStatus.APPROVED
                logger.info(f"Proposal {proposal_id} approved")
            else:
                proposal.status = ProposalStatus.REJECTED
                logger.info(f"Proposal {proposal_id} rejected")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to finalize proposal {proposal_id}: {e}")
            return False

    def _generate_proposal_id(self, title: str, proposer_id: str) -> str:
        """Generate unique proposal ID"""
        content = f"{title}{proposer_id}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_required_threshold(self, proposal_type: ProposalType, voting_method: VotingMethod) -> float:
        """Determine required threshold for proposal passage"""
        # Constitutional amendments require supermajority
        if proposal_type == ProposalType.CONSTITUTIONAL_AMENDMENT:
            return 0.75
        
        # Emergency actions require quick consensus
        if proposal_type == ProposalType.EMERGENCY_ACTION:
            return 0.60
        
        # Map voting methods to thresholds
        method_thresholds = {
            VotingMethod.SIMPLE_MAJORITY: 0.51,
            VotingMethod.SUPERMAJORITY_60: 0.60,
            VotingMethod.SUPERMAJORITY_66: 0.66,
            VotingMethod.SUPERMAJORITY_75: 0.75,
            VotingMethod.UNANIMOUS: 1.0
        }
        
        return method_thresholds.get(voting_method, 0.51)

    async def _validate_proposal(self, proposal: Proposal, proposer_id: str) -> Dict:
        """Validate proposal meets all requirements"""
        try:
            # Check proposer reputation
            reputation = self.user_reputations.get(proposer_id, 0.0)
            if reputation < 0.1:  # Minimum reputation threshold
                return {"valid": False, "reason": "Insufficient reputation"}
            
            # Check proposal content completeness
            if len(proposal.title) < 10:
                return {"valid": False, "reason": "Title too short"}
            
            if len(proposal.description) < 100:
                return {"valid": False, "reason": "Description too brief"}
            
            # Check for duplicate proposals
            for existing_id, existing in self.proposals.items():
                if existing.content_hash == proposal.content_hash:
                    return {"valid": False, "reason": "Duplicate proposal"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "reason": str(e)}

    async def _generate_impact_assessment(self, proposal: Proposal) -> Dict:
        """Generate automated impact assessment"""
        return {
            "complexity_score": min(1.0, len(proposal.description) / 1000),
            "governance_impact": self._assess_governance_impact(proposal),
            "technical_impact": self._assess_technical_impact(proposal),
            "social_impact": self._assess_social_impact(proposal),
            "risk_level": self._assess_risk_level(proposal),
            "estimated_implementation_time": self._estimate_implementation_time(proposal)
        }

    def _assess_governance_impact(self, proposal: Proposal) -> str:
        """Assess impact on governance structure"""
        if proposal.proposal_type == ProposalType.CONSTITUTIONAL_AMENDMENT:
            return "HIGH"
        elif proposal.proposal_type in [ProposalType.PROTOCOL_CHANGE, ProposalType.POLICY_CHANGE]:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_technical_impact(self, proposal: Proposal) -> str:
        """Assess technical implementation impact"""
        if proposal.proposal_type == ProposalType.TECHNICAL_UPGRADE:
            return "HIGH"
        elif proposal.proposal_type == ProposalType.PROTOCOL_CHANGE:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_social_impact(self, proposal: Proposal) -> str:
        """Assess social/community impact"""
        if proposal.proposal_type in [ProposalType.COMMUNITY_STANDARD, ProposalType.MEMBERSHIP_DECISION]:
            return "HIGH"
        else:
            return "MEDIUM"

    def _assess_risk_level(self, proposal: Proposal) -> str:
        """Assess overall risk level"""
        high_risk_types = [
            ProposalType.CONSTITUTIONAL_AMENDMENT,
            ProposalType.EMERGENCY_ACTION,
            ProposalType.TECHNICAL_UPGRADE
        ]
        return "HIGH" if proposal.proposal_type in high_risk_types else "MEDIUM"

    def _estimate_implementation_time(self, proposal: Proposal) -> str:
        """Estimate implementation timeframe"""
        if proposal.proposal_type == ProposalType.EMERGENCY_ACTION:
            return "IMMEDIATE"
        elif proposal.proposal_type in [ProposalType.TECHNICAL_UPGRADE, ProposalType.CONSTITUTIONAL_AMENDMENT]:
            return "3-6 MONTHS"
        else:
            return "1-3 MONTHS"

    def _calculate_voting_power(self, voter_id: str, proposal_id: str) -> float:
        """Calculate voter's power based on reputation and context"""
        base_power = 1.0
        reputation = self.user_reputations.get(voter_id, 0.5)
        
        # Reputation-weighted voting power
        reputation_multiplier = min(2.0, max(0.1, reputation))
        
        return base_power * reputation_multiplier

    def _resolve_delegation_chain(self, source: str, target: str) -> List[str]:
        """Resolve chain of vote delegation"""
        chain = [source]
        current = source
        
        # Follow delegation chain (prevent cycles)
        while current in self.delegation_chains and len(chain) < 10:
            next_delegate = self.delegation_chains[current]
            if next_delegate in chain:  # Cycle detected
                break
            chain.append(next_delegate)
            current = next_delegate
        
        return chain

    def _sign_vote(self, voter_id: str, proposal_id: str, vote_value: str) -> str:
        """Generate cryptographic signature for vote"""
        content = f"{voter_id}|{proposal_id}|{vote_value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_results(self, proposal: Proposal, votes: List[Vote]) -> Dict:
        """Calculate voting results based on method"""
        if proposal.voting_method == VotingMethod.SIMPLE_MAJORITY:
            return self._calculate_simple_majority(proposal, votes)
        elif proposal.voting_method == VotingMethod.RANKED_CHOICE:
            return self._calculate_ranked_choice(proposal, votes)
        elif proposal.voting_method == VotingMethod.QUADRATIC_VOTING:
            return self._calculate_quadratic_voting(proposal, votes)
        else:
            return self._calculate_simple_majority(proposal, votes)

    def _calculate_simple_majority(self, proposal: Proposal, votes: List[Vote]) -> Dict:
        """Calculate simple majority results"""
        yes_power = sum(v.voting_power for v in votes if v.vote_value.lower() == "yes")
        no_power = sum(v.voting_power for v in votes if v.vote_value.lower() == "no")
        abstain_power = sum(v.voting_power for v in votes if v.vote_value.lower() == "abstain")
        total_power = yes_power + no_power
        
        yes_percentage = yes_power / total_power if total_power > 0 else 0
        
        return {
            "yes_votes": len([v for v in votes if v.vote_value.lower() == "yes"]),
            "no_votes": len([v for v in votes if v.vote_value.lower() == "no"]),
            "abstain_votes": len([v for v in votes if v.vote_value.lower() == "abstain"]),
            "yes_power": yes_power,
            "no_power": no_power,
            "abstain_power": abstain_power,
            "yes_percentage": yes_percentage,
            "threshold_met": yes_percentage >= proposal.required_threshold
        }

    def _calculate_ranked_choice(self, proposal: Proposal, votes: List[Vote]) -> Dict:
        """Calculate ranked choice voting results"""
        # Simplified ranked choice - just return most preferred option
        vote_counts = {}
        for vote in votes:
            choices = vote.vote_value.split("|")  # Pipe-separated ranked choices
            if choices:
                first_choice = choices[0]
                vote_counts[first_choice] = vote_counts.get(first_choice, 0) + vote.voting_power
        
        if not vote_counts:
            return {"winner": None, "threshold_met": False}
        
        winner = max(vote_counts.items(), key=lambda x: x[1])
        total_power = sum(vote_counts.values())
        winner_percentage = winner[1] / total_power if total_power > 0 else 0
        
        return {
            "winner": winner[0],
            "vote_counts": vote_counts,
            "winner_percentage": winner_percentage,
            "threshold_met": winner_percentage >= proposal.required_threshold
        }

    def _calculate_quadratic_voting(self, proposal: Proposal, votes: List[Vote]) -> Dict:
        """Calculate quadratic voting results"""
        # Simplified quadratic voting implementation
        yes_credits = 0
        no_credits = 0
        
        for vote in votes:
            if vote.vote_value.lower() == "yes":
                # Square root of voting power for quadratic effect
                yes_credits += vote.voting_power ** 0.5
            elif vote.vote_value.lower() == "no":
                no_credits += vote.voting_power ** 0.5
        
        total_credits = yes_credits + no_credits
        yes_percentage = yes_credits / total_credits if total_credits > 0 else 0
        
        return {
            "yes_credits": yes_credits,
            "no_credits": no_credits,
            "yes_percentage": yes_percentage,
            "threshold_met": yes_percentage >= proposal.required_threshold
        }

    async def get_active_proposals(self) -> List[Dict]:
        """Get all currently active proposals"""
        active = []
        for proposal_id, proposal in self.proposals.items():
            if proposal.is_voting_active():
                results = await self.get_proposal_results(proposal_id)
                active.append({
                    "proposal": asdict(proposal),
                    "results": results
                })
        return active

    async def get_proposal_history(self, user_id: str) -> List[Dict]:
        """Get proposal history for a user"""
        history = []
        for proposal_id, proposal in self.proposals.items():
            if proposal.proposer_id == user_id:
                results = await self.get_proposal_results(proposal_id)
                history.append({
                    "proposal": asdict(proposal),
                    "results": results
                })
        return history

    async def export_governance_data(self) -> Dict:
        """Export complete governance data"""
        return {
            "node_id": self.node_id,
            "proposals": {pid: asdict(p) for pid, p in self.proposals.items()},
            "votes": {pid: [asdict(v) for v in votes] for pid, votes in self.proposal_votes.items()},
            "reputations": self.user_reputations,
            "delegations": self.delegation_chains,
            "export_timestamp": time.time()
        }

    async def import_governance_data(self, data: Dict) -> bool:
        """Import governance data from another node"""
        try:
            # Validate data structure
            if not all(key in data for key in ["proposals", "votes"]):
                return False
            
            # Import proposals
            for pid, proposal_data in data["proposals"].items():
                if pid not in self.proposals:  # Don't overwrite existing
                    proposal = Proposal(**proposal_data)
                    self.proposals[pid] = proposal
            
            # Import votes
            for pid, vote_data_list in data["votes"].items():
                if pid not in self.proposal_votes:
                    self.proposal_votes[pid] = []
                
                for vote_data in vote_data_list:
                    vote = Vote(**vote_data)
                    # Check for duplicates
                    existing_voter_ids = [v.voter_id for v in self.proposal_votes[pid]]
                    if vote.voter_id not in existing_voter_ids:
                        self.proposal_votes[pid].append(vote)
            
            logger.info("Governance data imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import governance data: {e}")
            return False