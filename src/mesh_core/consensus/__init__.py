"""
Consensus Module
===============

Democratic consensus mechanisms for The Mesh constitutional governance system.
Provides proposal creation, voting, and consensus resolution systems.

Components:
- ProposalSystem: Creates and manages governance proposals
- VotingEngine: Handles democratic voting processes
- ConsensusResolver: Resolves consensus and implements decisions
"""

from .proposal_system import (
    ProposalSystem,
    Proposal,
    ProposalType,
    ProposalStatus,
    Vote,
    VotingMethod
)

from .voting_engine import (
    VotingEngine,
    BallotBox,
    VotingSession,
    VotingResult
)

from .consensus_resolver import (
    ConsensusResolver,
    ConsensusDecision,
    ConsensusThreshold,
    DecisionImplementor
)

__all__ = [
    'ProposalSystem',
    'Proposal', 
    'ProposalType',
    'ProposalStatus',
    'Vote',
    'VotingMethod',
    'VotingEngine',
    'BallotBox',
    'VotingSession', 
    'VotingResult',
    'ConsensusResolver',
    'ConsensusDecision',
    'ConsensusThreshold',
    'DecisionImplementor'
]

__version__ = "1.0.0"