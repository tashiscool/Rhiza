"""
Voting Engine
=============

Advanced voting mechanisms for The Mesh democratic governance.
Handles ballot management, vote aggregation, and result calculation.
"""

import asyncio
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VotingSessionStatus(Enum):
    """Voting session status"""
    PENDING = "pending"
    ACTIVE = "active"
    EXTENDED = "extended"
    CLOSED = "closed"
    FINALIZED = "finalized"

@dataclass
class BallotBox:
    """Secure ballot collection system"""
    session_id: str
    proposal_id: str
    encrypted_ballots: List[Dict]
    ballot_hashes: Set[str]
    tally_started: bool = False
    results_sealed: bool = False

@dataclass
class VotingSession:
    """Voting session management"""
    session_id: str
    proposal_id: str
    status: VotingSessionStatus
    start_time: float
    end_time: float
    eligible_voters: Set[str]
    participation_rate: float = 0.0
    turnout_threshold: float = 0.25
    extension_count: int = 0
    max_extensions: int = 2

@dataclass
class VotingResult:
    """Comprehensive voting results"""
    session_id: str
    proposal_id: str
    total_eligible: int
    total_votes_cast: int
    participation_rate: float
    results_breakdown: Dict
    consensus_achieved: bool
    margin_of_victory: Optional[float]
    statistical_confidence: float
    audit_trail: List[Dict]

class VotingEngine:
    """Advanced democratic voting engine"""
    
    def __init__(self, node_id: str, crypto_config: Optional[Dict] = None):
        self.node_id = node_id
        self.crypto_config = crypto_config or {}
        self.active_sessions: Dict[str, VotingSession] = {}
        self.ballot_boxes: Dict[str, BallotBox] = {}
        self.voting_results: Dict[str, VotingResult] = {}
        self.voter_registry: Dict[str, Dict] = {}  # Eligible voters and their credentials
        self.vote_encryption_keys: Dict[str, str] = {}
        
        logger.info(f"VotingEngine initialized for node {node_id}")

    async def create_voting_session(
        self,
        proposal_id: str,
        eligible_voters: List[str],
        duration_hours: int = 168,
        turnout_threshold: float = 0.25
    ) -> str:
        """Create a new voting session"""
        try:
            session_id = self._generate_session_id(proposal_id)
            
            now = time.time()
            end_time = now + (duration_hours * 3600)
            
            session = VotingSession(
                session_id=session_id,
                proposal_id=proposal_id,
                status=VotingSessionStatus.PENDING,
                start_time=now,
                end_time=end_time,
                eligible_voters=set(eligible_voters),
                turnout_threshold=turnout_threshold
            )
            
            # Create secure ballot box
            ballot_box = BallotBox(
                session_id=session_id,
                proposal_id=proposal_id,
                encrypted_ballots=[],
                ballot_hashes=set()
            )
            
            self.active_sessions[session_id] = session
            self.ballot_boxes[session_id] = ballot_box
            
            # Generate encryption keys for this session
            await self._setup_session_encryption(session_id)
            
            logger.info(f"Voting session created: {session_id} for proposal {proposal_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create voting session: {e}")
            raise

    async def activate_session(self, session_id: str) -> bool:
        """Activate a voting session to begin accepting votes"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.status = VotingSessionStatus.ACTIVE
            
            # Notify eligible voters
            await self._notify_eligible_voters(session)
            
            logger.info(f"Voting session activated: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate session {session_id}: {e}")
            return False

    async def cast_encrypted_vote(
        self,
        session_id: str,
        voter_id: str,
        encrypted_ballot: str,
        ballot_signature: str
    ) -> bool:
        """Cast an encrypted vote in a session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            ballot_box = self.ballot_boxes[session_id]
            
            # Verify session is active
            if session.status != VotingSessionStatus.ACTIVE:
                return False
            
            # Verify voter eligibility
            if voter_id not in session.eligible_voters:
                logger.warning(f"Ineligible voter attempted to vote: {voter_id}")
                return False
            
            # Verify ballot signature
            if not await self._verify_ballot_signature(voter_id, encrypted_ballot, ballot_signature):
                logger.warning(f"Invalid ballot signature from {voter_id}")
                return False
            
            # Create ballot hash for duplicate prevention
            ballot_hash = hashlib.sha256(f"{voter_id}|{session_id}".encode()).hexdigest()
            
            # Check for duplicate votes
            if ballot_hash in ballot_box.ballot_hashes:
                logger.warning(f"Duplicate vote attempt from {voter_id}")
                return False
            
            # Store encrypted ballot
            ballot_record = {
                "ballot_hash": ballot_hash,
                "encrypted_ballot": encrypted_ballot,
                "signature": ballot_signature,
                "timestamp": time.time(),
                "verification_code": self._generate_verification_code(voter_id, session_id)
            }
            
            ballot_box.encrypted_ballots.append(ballot_record)
            ballot_box.ballot_hashes.add(ballot_hash)
            
            # Update participation rate
            session.participation_rate = len(ballot_box.ballot_hashes) / len(session.eligible_voters)
            
            logger.info(f"Encrypted vote cast: {voter_id} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cast vote: {e}")
            return False

    async def extend_voting_session(
        self,
        session_id: str,
        additional_hours: int = 24,
        reason: str = "low_turnout"
    ) -> bool:
        """Extend a voting session if conditions are met"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check if extension is allowed
            if session.extension_count >= session.max_extensions:
                return False
            
            # Check if extension conditions are met
            if not await self._check_extension_conditions(session, reason):
                return False
            
            # Extend the session
            session.end_time += additional_hours * 3600
            session.extension_count += 1
            session.status = VotingSessionStatus.EXTENDED
            
            # Notify participants of extension
            await self._notify_session_extension(session, additional_hours, reason)
            
            logger.info(f"Voting session extended: {session_id} by {additional_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extend session {session_id}: {e}")
            return False

    async def close_voting_session(self, session_id: str) -> bool:
        """Close a voting session and begin tallying"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            ballot_box = self.ballot_boxes[session_id]
            
            # Check if session can be closed
            now = time.time()
            if now < session.end_time and session.participation_rate >= session.turnout_threshold:
                # Session still active and has sufficient turnout
                return False
            
            session.status = VotingSessionStatus.CLOSED
            
            # Begin secure tallying process
            await self._begin_tally_process(session_id)
            
            logger.info(f"Voting session closed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
            return False

    async def _begin_tally_process(self, session_id: str) -> bool:
        """Begin secure ballot tallying process"""
        try:
            session = self.active_sessions[session_id]
            ballot_box = self.ballot_boxes[session_id]
            
            ballot_box.tally_started = True
            
            # Decrypt and tally ballots
            decrypted_votes = []
            audit_trail = []
            
            for ballot_record in ballot_box.encrypted_ballots:
                try:
                    # Decrypt ballot using session keys
                    decrypted_vote = await self._decrypt_ballot(
                        session_id,
                        ballot_record["encrypted_ballot"]
                    )
                    
                    if decrypted_vote:
                        decrypted_votes.append(decrypted_vote)
                        
                        # Add to audit trail (without revealing voter identity)
                        audit_trail.append({
                            "ballot_hash": ballot_record["ballot_hash"][:8],  # Truncated for privacy
                            "timestamp": ballot_record["timestamp"],
                            "verification_code": ballot_record["verification_code"],
                            "vote_valid": True
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to decrypt ballot: {e}")
                    audit_trail.append({
                        "ballot_hash": ballot_record["ballot_hash"][:8],
                        "timestamp": ballot_record["timestamp"],
                        "vote_valid": False,
                        "error": str(e)
                    })
            
            # Calculate results
            results_breakdown = self._calculate_tally_results(decrypted_votes)
            
            # Determine consensus
            consensus_achieved = self._determine_consensus(results_breakdown, session)
            
            # Calculate statistical measures
            margin_of_victory = self._calculate_margin_of_victory(results_breakdown)
            statistical_confidence = self._calculate_statistical_confidence(
                len(decrypted_votes),
                len(session.eligible_voters),
                results_breakdown
            )
            
            # Create final results
            result = VotingResult(
                session_id=session_id,
                proposal_id=session.proposal_id,
                total_eligible=len(session.eligible_voters),
                total_votes_cast=len(decrypted_votes),
                participation_rate=session.participation_rate,
                results_breakdown=results_breakdown,
                consensus_achieved=consensus_achieved,
                margin_of_victory=margin_of_victory,
                statistical_confidence=statistical_confidence,
                audit_trail=audit_trail
            )
            
            self.voting_results[session_id] = result
            session.status = VotingSessionStatus.FINALIZED
            ballot_box.results_sealed = True
            
            logger.info(f"Tally completed for session {session_id}: consensus={consensus_achieved}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to tally session {session_id}: {e}")
            return False

    async def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a voting session"""
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            ballot_box = self.ballot_boxes[session_id]
            
            now = time.time()
            time_remaining = max(0, session.end_time - now) if session.status == VotingSessionStatus.ACTIVE else 0
            
            status = {
                "session_id": session_id,
                "proposal_id": session.proposal_id,
                "status": session.status.value,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "time_remaining_seconds": time_remaining,
                "eligible_voters": len(session.eligible_voters),
                "votes_cast": len(ballot_box.ballot_hashes),
                "participation_rate": session.participation_rate,
                "turnout_threshold": session.turnout_threshold,
                "extension_count": session.extension_count,
                "can_extend": session.extension_count < session.max_extensions,
                "results_available": session_id in self.voting_results
            }
            
            # Include results if available
            if session_id in self.voting_results:
                status["results"] = asdict(self.voting_results[session_id])
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return None

    async def verify_vote_receipt(self, session_id: str, verification_code: str) -> Optional[Dict]:
        """Verify that a vote was received and counted"""
        try:
            if session_id not in self.ballot_boxes:
                return None
            
            ballot_box = self.ballot_boxes[session_id]
            
            for ballot_record in ballot_box.encrypted_ballots:
                if ballot_record["verification_code"] == verification_code:
                    return {
                        "verified": True,
                        "session_id": session_id,
                        "timestamp": ballot_record["timestamp"],
                        "ballot_hash": ballot_record["ballot_hash"][:8],  # Truncated for privacy
                        "counted": ballot_box.tally_started
                    }
            
            return {"verified": False}
            
        except Exception as e:
            logger.error(f"Failed to verify vote receipt: {e}")
            return None

    def _generate_session_id(self, proposal_id: str) -> str:
        """Generate unique session ID"""
        content = f"{proposal_id}|{self.node_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _setup_session_encryption(self, session_id: str) -> bool:
        """Setup encryption keys for voting session"""
        try:
            # Generate session-specific encryption key
            key_material = f"{session_id}|{self.node_id}|{time.time()}"
            encryption_key = hashlib.sha256(key_material.encode()).hexdigest()
            
            self.vote_encryption_keys[session_id] = encryption_key
            
            logger.info(f"Encryption setup completed for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup encryption for session {session_id}: {e}")
            return False

    async def _notify_eligible_voters(self, session: VotingSession) -> bool:
        """Notify eligible voters that voting has begun"""
        try:
            # Implementation would send notifications through mesh network
            logger.info(f"Notified {len(session.eligible_voters)} eligible voters for session {session.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to notify voters: {e}")
            return False

    async def _verify_ballot_signature(self, voter_id: str, ballot: str, signature: str) -> bool:
        """Verify cryptographic signature on ballot"""
        try:
            # Simplified signature verification
            expected_content = f"{voter_id}|{ballot}"
            expected_signature = hashlib.sha256(expected_content.encode()).hexdigest()
            return signature == expected_signature
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def _generate_verification_code(self, voter_id: str, session_id: str) -> str:
        """Generate verification code for vote receipt"""
        content = f"{voter_id}|{session_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def _check_extension_conditions(self, session: VotingSession, reason: str) -> bool:
        """Check if session extension conditions are met"""
        if reason == "low_turnout":
            return session.participation_rate < session.turnout_threshold
        elif reason == "technical_issues":
            return True  # Allow extension for technical problems
        elif reason == "community_request":
            return session.participation_rate > 0.1  # Some minimum engagement
        else:
            return False

    async def _notify_session_extension(self, session: VotingSession, hours: int, reason: str) -> bool:
        """Notify participants of session extension"""
        try:
            logger.info(f"Session {session.session_id} extended by {hours} hours: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to notify extension: {e}")
            return False

    async def _decrypt_ballot(self, session_id: str, encrypted_ballot: str) -> Optional[Dict]:
        """Decrypt an encrypted ballot"""
        try:
            # Simplified decryption - in reality would use proper crypto
            encryption_key = self.vote_encryption_keys.get(session_id)
            if not encryption_key:
                return None
            
            # Mock decryption process
            ballot_data = {
                "vote": "yes",  # Simplified - would be actual decrypted vote
                "weight": 1.0,
                "preferences": {}
            }
            
            return ballot_data
            
        except Exception as e:
            logger.error(f"Failed to decrypt ballot: {e}")
            return None

    def _calculate_tally_results(self, decrypted_votes: List[Dict]) -> Dict:
        """Calculate final tally results"""
        try:
            yes_votes = sum(1 for vote in decrypted_votes if vote.get("vote") == "yes")
            no_votes = sum(1 for vote in decrypted_votes if vote.get("vote") == "no")
            abstain_votes = sum(1 for vote in decrypted_votes if vote.get("vote") == "abstain")
            
            total_votes = len(decrypted_votes)
            
            return {
                "yes_votes": yes_votes,
                "no_votes": no_votes,
                "abstain_votes": abstain_votes,
                "total_votes": total_votes,
                "yes_percentage": yes_votes / total_votes if total_votes > 0 else 0,
                "no_percentage": no_votes / total_votes if total_votes > 0 else 0,
                "abstain_percentage": abstain_votes / total_votes if total_votes > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate tally results: {e}")
            return {}

    def _determine_consensus(self, results: Dict, session: VotingSession) -> bool:
        """Determine if consensus was achieved"""
        try:
            yes_percentage = results.get("yes_percentage", 0)
            participation_rate = session.participation_rate
            
            # Require both majority support and minimum participation
            return yes_percentage > 0.5 and participation_rate >= session.turnout_threshold
            
        except Exception as e:
            logger.error(f"Failed to determine consensus: {e}")
            return False

    def _calculate_margin_of_victory(self, results: Dict) -> Optional[float]:
        """Calculate margin of victory"""
        try:
            yes_percentage = results.get("yes_percentage", 0)
            no_percentage = results.get("no_percentage", 0)
            return abs(yes_percentage - no_percentage)
        except Exception as e:
            logger.error(f"Failed to calculate margin: {e}")
            return None

    def _calculate_statistical_confidence(
        self,
        votes_cast: int,
        total_eligible: int,
        results: Dict
    ) -> float:
        """Calculate statistical confidence in results"""
        try:
            participation_rate = votes_cast / total_eligible if total_eligible > 0 else 0
            margin = self._calculate_margin_of_victory(results) or 0
            
            # Simplified confidence calculation
            base_confidence = min(0.95, participation_rate + 0.1)
            margin_boost = min(0.05, margin * 0.1)
            
            return min(0.99, base_confidence + margin_boost)
            
        except Exception as e:
            logger.error(f"Failed to calculate statistical confidence: {e}")
            return 0.0

    async def get_voting_history(self, voter_id: str) -> List[Dict]:
        """Get voting history for a voter (privacy-preserving)"""
        history = []
        
        for session_id, session in self.active_sessions.items():
            if voter_id in session.eligible_voters:
                ballot_box = self.ballot_boxes[session_id]
                
                # Check if voter participated (without revealing vote)
                voter_hash = hashlib.sha256(f"{voter_id}|{session_id}".encode()).hexdigest()
                participated = voter_hash in ballot_box.ballot_hashes
                
                history.append({
                    "session_id": session_id,
                    "proposal_id": session.proposal_id,
                    "participated": participated,
                    "session_status": session.status.value
                })
        
        return history

    async def export_session_data(self, session_id: str, include_ballots: bool = False) -> Optional[Dict]:
        """Export session data for backup or analysis"""
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            data = {
                "session": asdict(session),
                "ballot_box_summary": {
                    "session_id": session_id,
                    "total_ballots": len(self.ballot_boxes[session_id].encrypted_ballots),
                    "tally_started": self.ballot_boxes[session_id].tally_started,
                    "results_sealed": self.ballot_boxes[session_id].results_sealed
                }
            }
            
            if session_id in self.voting_results:
                data["results"] = asdict(self.voting_results[session_id])
            
            # Only include encrypted ballots if explicitly requested (for auditing)
            if include_ballots and session.status == VotingSessionStatus.FINALIZED:
                data["encrypted_ballots"] = self.ballot_boxes[session_id].encrypted_ballots
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return None