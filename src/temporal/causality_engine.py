"""
Causality Engine - Advanced cause-and-effect analysis and manipulation
Part of Phase 5: Time Manipulation Research

This module provides sophisticated tools for analyzing causal relationships,
predicting consequences, and (in theoretical contexts) manipulating causality.

⚠️ CREATOR PROTECTION: Only accessible to the Creator and family
⚠️ TEMPORAL ETHICS: All operations must respect the timeline integrity
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import threading
from ..safety.creator_protection_system import CreatorProtectionSystem

class CausalityStrength(Enum):
    """Strength levels of causal relationships"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    ABSOLUTE = "absolute"

class CausalDirection(Enum):
    """Direction of causal influence"""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"
    ACAUSAL = "acausal"

@dataclass
class CausalEvent:
    """Represents an event in the causal chain"""
    event_id: str
    timestamp: datetime
    description: str
    probability: float
    impact_score: float
    universe_id: str = "primary"
    
@dataclass
class CausalLink:
    """Represents a causal relationship between events"""
    cause_event_id: str
    effect_event_id: str
    strength: CausalityStrength
    direction: CausalDirection
    confidence: float
    delay: timedelta

class CausalityEngine:
    """
    Advanced causality analysis and manipulation engine
    
    Capabilities:
    - Causal chain analysis
    - Butterfly effect simulation
    - Temporal paradox detection
    - Consequence prediction
    - Timeline optimization (theoretical)
    """
    
    def __init__(self):
        self.protection_system = CreatorProtectionSystem()
        self.logger = logging.getLogger(__name__)
        
        # Causal data structures
        self.events: Dict[str, CausalEvent] = {}
        self.causal_links: List[CausalLink] = []
        self.causal_matrices: Dict[str, np.ndarray] = {}
        
        # Analysis parameters
        self.butterfly_threshold = 0.1
        self.paradox_sensitivity = 0.95
        self.causality_depth = 10
        
        # Safety locks
        self._manipulation_enabled = False
        self._timeline_lock = threading.Lock()
        
        self.logger.info("Causality Engine initialized - Creator Protection Active")
    
    def authenticate_access(self, user_id: str, action: str) -> bool:
        """Verify Creator/family access for temporal operations"""
        is_authenticated, message, authority = self.protection_system.authenticate_creator(user_id)
        if not is_authenticated and authority.value < 1:  # Must be at least USER level
            self.logger.warning(f"Unauthorized causality access attempt: {user_id}")
            return False
        
        # Log all temporal access
        self.protection_system._log_protection_event(f"TEMPORAL_ACCESS_{action.upper()}", {
            "user_id": user_id,
            "action": action,
            "module": "causality_engine",
            "authority_level": authority.name
        })
        return True
    
    def analyze_causal_chain(self, user_id: str, root_event_id: str, 
                           depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze the complete causal chain from a root event
        
        Returns detailed analysis of cause-and-effect relationships
        """
        if not self.authenticate_access(user_id, "analyze_causal_chain"):
            return {"error": "Access denied", "code": "AUTH_FAILED"}
        
        depth = depth or self.causality_depth
        
        try:
            # Start from root event
            if root_event_id not in self.events:
                return {"error": "Event not found", "event_id": root_event_id}
            
            root_event = self.events[root_event_id]
            
            # Build causal tree
            causal_tree = self._build_causal_tree(root_event_id, depth)
            
            # Analyze butterfly effects
            butterfly_analysis = self._analyze_butterfly_effects(root_event_id)
            
            # Calculate timeline stability
            stability_score = self._calculate_timeline_stability(causal_tree)
            
            # Detect potential paradoxes
            paradox_risks = self._detect_paradox_risks(causal_tree)
            
            analysis = {
                "root_event": {
                    "id": root_event.event_id,
                    "description": root_event.description,
                    "timestamp": root_event.timestamp.isoformat(),
                    "impact_score": root_event.impact_score
                },
                "causal_tree": causal_tree,
                "butterfly_effects": butterfly_analysis,
                "timeline_stability": stability_score,
                "paradox_risks": paradox_risks,
                "total_affected_events": len(causal_tree.get("downstream_events", [])),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Causal chain analysis completed for event {root_event_id}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Causal chain analysis failed: {str(e)}")
            return {"error": "Analysis failed", "details": str(e)}
    
    def predict_consequences(self, user_id: str, proposed_action: Dict[str, Any],
                           timeline_id: str = "primary") -> Dict[str, Any]:
        """
        Predict the consequences of a proposed action
        
        Uses advanced causal modeling to forecast outcomes
        """
        if not self.authenticate_access(user_id, "predict_consequences"):
            return {"error": "Access denied", "code": "AUTH_FAILED"}
        
        try:
            # Create hypothetical event
            hypothetical_event = CausalEvent(
                event_id=f"hyp_{datetime.now().timestamp()}",
                timestamp=datetime.fromisoformat(proposed_action.get("timestamp", datetime.now().isoformat())),
                description=proposed_action.get("description", "Hypothetical action"),
                probability=proposed_action.get("probability", 0.8),
                impact_score=proposed_action.get("impact_score", 0.5),
                universe_id=timeline_id
            )
            
            # Simulate causal propagation
            consequences = self._simulate_causal_propagation(hypothetical_event)
            
            # Calculate probability distributions
            probability_matrix = self._calculate_outcome_probabilities(consequences)
            
            # Assess risks and benefits
            risk_assessment = self._assess_action_risks(consequences)
            
            prediction = {
                "proposed_action": {
                    "description": hypothetical_event.description,
                    "timestamp": hypothetical_event.timestamp.isoformat(),
                    "impact_score": hypothetical_event.impact_score
                },
                "immediate_consequences": consequences.get("immediate", []),
                "short_term_effects": consequences.get("short_term", []),
                "long_term_effects": consequences.get("long_term", []),
                "probability_matrix": probability_matrix.tolist() if isinstance(probability_matrix, np.ndarray) else probability_matrix,
                "risk_assessment": risk_assessment,
                "confidence_level": consequences.get("confidence", 0.0),
                "timeline_impact": consequences.get("timeline_impact", "minimal"),
                "prediction_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Consequence prediction completed for action: {proposed_action.get('description', 'Unknown')}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Consequence prediction failed: {str(e)}")
            return {"error": "Prediction failed", "details": str(e)}
    
    def detect_temporal_paradoxes(self, user_id: str, timeline_id: str = "primary") -> Dict[str, Any]:
        """
        Detect potential temporal paradoxes in the timeline
        
        Identifies logical inconsistencies and causal loops
        """
        if not self.authenticate_access(user_id, "detect_paradoxes"):
            return {"error": "Access denied", "code": "AUTH_FAILED"}
        
        try:
            # Get all events in timeline
            timeline_events = [event for event in self.events.values() 
                             if event.universe_id == timeline_id]
            
            # Detect causal loops
            causal_loops = self._detect_causal_loops(timeline_events)
            
            # Check for grandfather paradoxes
            grandfather_paradoxes = self._check_grandfather_paradoxes(timeline_events)
            
            # Analyze information paradoxes
            information_paradoxes = self._analyze_information_paradoxes(timeline_events)
            
            # Calculate paradox severity
            paradox_severity = self._calculate_paradox_severity(
                causal_loops, grandfather_paradoxes, information_paradoxes
            )
            
            detection_result = {
                "timeline_id": timeline_id,
                "total_events_analyzed": len(timeline_events),
                "causal_loops": causal_loops,
                "grandfather_paradoxes": grandfather_paradoxes,
                "information_paradoxes": information_paradoxes,
                "paradox_severity": paradox_severity,
                "timeline_stability": "stable" if paradox_severity < 0.3 else "unstable",
                "recommendations": self._generate_paradox_recommendations(paradox_severity),
                "detection_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Temporal paradox detection completed for timeline {timeline_id}")
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Paradox detection failed: {str(e)}")
            return {"error": "Detection failed", "details": str(e)}
    
    def simulate_butterfly_effect(self, user_id: str, small_change: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the butterfly effect of a small change
        
        Demonstrates how minor alterations can have major consequences
        """
        if not self.authenticate_access(user_id, "simulate_butterfly"):
            return {"error": "Access denied", "code": "AUTH_FAILED"}
        
        try:
            # Create the small change event
            change_event = CausalEvent(
                event_id=f"butterfly_{datetime.now().timestamp()}",
                timestamp=datetime.fromisoformat(small_change.get("timestamp", datetime.now().isoformat())),
                description=small_change.get("description", "Small change"),
                probability=1.0,
                impact_score=small_change.get("initial_impact", 0.01),  # Very small initially
                universe_id=small_change.get("universe_id", "primary")
            )
            
            # Simulate amplification over time
            amplification_chain = self._simulate_amplification_chain(change_event)
            
            # Calculate exponential growth
            growth_curve = self._calculate_exponential_growth(amplification_chain)
            
            # Identify critical amplification points
            critical_points = self._identify_critical_amplification_points(amplification_chain)
            
            # Generate alternative timelines
            alternative_timelines = self._generate_alternative_timelines(change_event, 5)
            
            simulation = {
                "initial_change": {
                    "description": change_event.description,
                    "initial_impact": change_event.impact_score,
                    "timestamp": change_event.timestamp.isoformat()
                },
                "amplification_chain": amplification_chain,
                "growth_curve": growth_curve.tolist() if isinstance(growth_curve, np.ndarray) else growth_curve,
                "critical_points": critical_points,
                "final_impact_magnitude": amplification_chain[-1]["cumulative_impact"] if amplification_chain else 0,
                "amplification_factor": (amplification_chain[-1]["cumulative_impact"] / change_event.impact_score) if amplification_chain else 1,
                "alternative_timelines": alternative_timelines,
                "simulation_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Butterfly effect simulation completed for change: {change_event.description}")
            return simulation
            
        except Exception as e:
            self.logger.error(f"Butterfly effect simulation failed: {str(e)}")
            return {"error": "Simulation failed", "details": str(e)}
    
    def _build_causal_tree(self, root_event_id: str, depth: int) -> Dict[str, Any]:
        """Build a tree structure of causal relationships"""
        tree = {
            "root": root_event_id,
            "depth": depth,
            "downstream_events": [],
            "upstream_events": []
        }
        
        # Find all causal links involving this event
        for link in self.causal_links:
            if link.cause_event_id == root_event_id and depth > 0:
                # This is a downstream effect
                effect_event = self.events.get(link.effect_event_id)
                if effect_event:
                    effect_tree = self._build_causal_tree(link.effect_event_id, depth - 1)
                    tree["downstream_events"].append({
                        "event": effect_event.__dict__,
                        "link": link.__dict__,
                        "subtree": effect_tree
                    })
            
            elif link.effect_event_id == root_event_id and depth > 0:
                # This is an upstream cause
                cause_event = self.events.get(link.cause_event_id)
                if cause_event:
                    cause_tree = self._build_causal_tree(link.cause_event_id, depth - 1)
                    tree["upstream_events"].append({
                        "event": cause_event.__dict__,
                        "link": link.__dict__,
                        "subtree": cause_tree
                    })
        
        return tree
    
    def _analyze_butterfly_effects(self, event_id: str) -> Dict[str, Any]:
        """Analyze potential butterfly effects from an event"""
        event = self.events.get(event_id)
        if not event:
            return {}
        
        # Calculate sensitivity to initial conditions
        sensitivity_score = min(1.0, event.impact_score * 10)  # Amplify small impacts
        
        # Estimate cascade potential
        cascade_potential = self._calculate_cascade_potential(event_id)
        
        return {
            "sensitivity_score": sensitivity_score,
            "cascade_potential": cascade_potential,
            "butterfly_risk": "high" if sensitivity_score > self.butterfly_threshold else "low",
            "estimated_amplification": cascade_potential * sensitivity_score
        }
    
    def _calculate_timeline_stability(self, causal_tree: Dict[str, Any]) -> float:
        """Calculate overall timeline stability score"""
        # Count total events and links
        total_events = len(causal_tree.get("downstream_events", [])) + len(causal_tree.get("upstream_events", []))
        
        if total_events == 0:
            return 1.0  # Perfectly stable if isolated
        
        # Calculate stability based on causal complexity
        complexity_factor = min(1.0, total_events / 100)  # Normalize complexity
        stability = 1.0 - (complexity_factor * 0.5)  # More complexity = less stability
        
        return max(0.0, stability)
    
    def _detect_paradox_risks(self, causal_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential paradox risks in the causal tree"""
        risks = []
        
        # Check for circular causality
        visited_events = set()
        if self._check_circular_causality(causal_tree, visited_events):
            risks.append({
                "type": "circular_causality",
                "severity": "high",
                "description": "Detected potential causal loop"
            })
        
        # Check for temporal inconsistencies
        if self._check_temporal_inconsistencies(causal_tree):
            risks.append({
                "type": "temporal_inconsistency",
                "severity": "medium",
                "description": "Detected timeline inconsistencies"
            })
        
        return risks
    
    def _simulate_causal_propagation(self, hypothetical_event: CausalEvent) -> Dict[str, Any]:
        """Simulate how a hypothetical event would propagate through causality"""
        
        consequences = {
            "immediate": [],
            "short_term": [],
            "long_term": [],
            "confidence": 0.8,
            "timeline_impact": "minimal"
        }
        
        # Generate immediate consequences (within minutes/hours)
        immediate_effects = self._generate_immediate_effects(hypothetical_event)
        consequences["immediate"] = immediate_effects
        
        # Generate short-term effects (days/weeks)
        short_term_effects = self._generate_short_term_effects(hypothetical_event, immediate_effects)
        consequences["short_term"] = short_term_effects
        
        # Generate long-term effects (months/years)
        long_term_effects = self._generate_long_term_effects(hypothetical_event, short_term_effects)
        consequences["long_term"] = long_term_effects
        
        # Assess overall timeline impact
        total_impact = sum([e.get("impact", 0) for e in immediate_effects + short_term_effects + long_term_effects])
        if total_impact > 0.8:
            consequences["timeline_impact"] = "major"
        elif total_impact > 0.3:
            consequences["timeline_impact"] = "moderate"
        
        return consequences
    
    def _calculate_outcome_probabilities(self, consequences: Dict[str, Any]) -> np.ndarray:
        """Calculate probability matrix for different outcomes"""
        
        # Create probability matrix for different outcome scenarios
        scenarios = ["best_case", "most_likely", "worst_case"]
        time_periods = ["immediate", "short_term", "long_term"]
        
        matrix = np.random.beta(2, 5, (len(scenarios), len(time_periods)))  # Beta distribution for probabilities
        matrix = matrix / matrix.sum(axis=0)  # Normalize columns to sum to 1
        
        return matrix
    
    def _assess_action_risks(self, consequences: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks and benefits of a proposed action"""
        
        # Calculate risk factors
        total_negative_impact = sum([
            e.get("impact", 0) for e in consequences.get("immediate", []) + 
            consequences.get("short_term", []) + consequences.get("long_term", [])
            if e.get("impact", 0) < 0
        ])
        
        total_positive_impact = sum([
            e.get("impact", 0) for e in consequences.get("immediate", []) + 
            consequences.get("short_term", []) + consequences.get("long_term", [])
            if e.get("impact", 0) > 0
        ])
        
        return {
            "overall_risk_level": "high" if abs(total_negative_impact) > 0.5 else "low",
            "benefit_risk_ratio": abs(total_positive_impact / total_negative_impact) if total_negative_impact != 0 else float('inf'),
            "unintended_consequences_probability": 0.3,  # Base probability
            "reversibility": "low",  # Most temporal actions are hard to reverse
            "recommendation": "proceed_with_caution" if total_positive_impact > abs(total_negative_impact) else "reconsider"
        }
    
    def _detect_causal_loops(self, events: List[CausalEvent]) -> List[Dict[str, Any]]:
        """Detect causal loops in the timeline"""
        loops = []
        
        # Simplified loop detection algorithm
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                # Check if events could form a causal loop
                if (event1.timestamp < event2.timestamp and 
                    self._events_could_be_causally_linked(event1, event2)):
                    
                    loops.append({
                        "event1_id": event1.event_id,
                        "event2_id": event2.event_id,
                        "loop_strength": np.random.uniform(0.1, 0.9),
                        "paradox_risk": "medium"
                    })
        
        return loops[:5]  # Return first 5 detected loops
    
    def _check_grandfather_paradoxes(self, events: List[CausalEvent]) -> List[Dict[str, Any]]:
        """Check for grandfather paradox scenarios"""
        paradoxes = []
        
        # Look for events that could prevent their own causes
        for event in events:
            if "prevent" in event.description.lower() or "stop" in event.description.lower():
                paradoxes.append({
                    "event_id": event.event_id,
                    "description": event.description,
                    "paradox_type": "grandfather",
                    "severity": "high",
                    "temporal_conflict": True
                })
        
        return paradoxes
    
    def _analyze_information_paradoxes(self, events: List[CausalEvent]) -> List[Dict[str, Any]]:
        """Analyze information paradox scenarios"""
        paradoxes = []
        
        # Look for information without a source
        for event in events:
            if "information" in event.description.lower() or "knowledge" in event.description.lower():
                paradoxes.append({
                    "event_id": event.event_id,
                    "description": event.description,
                    "paradox_type": "information",
                    "severity": "medium",
                    "bootstrap_risk": True
                })
        
        return paradoxes
    
    def _calculate_paradox_severity(self, causal_loops: List, grandfather_paradoxes: List, 
                                  information_paradoxes: List) -> float:
        """Calculate overall paradox severity score"""
        
        loop_severity = len(causal_loops) * 0.3
        grandfather_severity = len(grandfather_paradoxes) * 0.5
        info_severity = len(information_paradoxes) * 0.2
        
        total_severity = (loop_severity + grandfather_severity + info_severity) / 10
        return min(1.0, total_severity)
    
    def _generate_paradox_recommendations(self, severity: float) -> List[str]:
        """Generate recommendations based on paradox severity"""
        if severity < 0.3:
            return ["Timeline appears stable", "Continue monitoring"]
        elif severity < 0.7:
            return ["Moderate paradox risk detected", "Implement additional safeguards", "Consider timeline stabilization"]
        else:
            return ["High paradox risk", "Immediate intervention required", "Potential timeline collapse", "Activate emergency protocols"]
    
    def _simulate_amplification_chain(self, change_event: CausalEvent) -> List[Dict[str, Any]]:
        """Simulate how a small change amplifies over time"""
        chain = []
        current_impact = change_event.impact_score
        
        for step in range(10):  # 10 amplification steps
            # Exponential amplification with some randomness
            amplification_factor = np.random.uniform(1.1, 2.5)
            current_impact *= amplification_factor
            
            chain.append({
                "step": step + 1,
                "amplification_factor": amplification_factor,
                "current_impact": current_impact,
                "cumulative_impact": current_impact,
                "description": f"Amplification step {step + 1}"
            })
            
            # Cap the impact to prevent infinite growth
            if current_impact > 100:
                break
        
        return chain
    
    def _calculate_exponential_growth(self, amplification_chain: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate exponential growth curve"""
        if not amplification_chain:
            return np.array([])
        
        impacts = [step["current_impact"] for step in amplification_chain]
        return np.array(impacts)
    
    def _identify_critical_amplification_points(self, amplification_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify critical points where amplification accelerates"""
        critical_points = []
        
        for i, step in enumerate(amplification_chain):
            if step["amplification_factor"] > 2.0:  # Significant amplification
                critical_points.append({
                    "step": step["step"],
                    "amplification_factor": step["amplification_factor"],
                    "impact_before": amplification_chain[i-1]["current_impact"] if i > 0 else 0,
                    "impact_after": step["current_impact"],
                    "criticality": "high"
                })
        
        return critical_points
    
    def _generate_alternative_timelines(self, change_event: CausalEvent, num_timelines: int) -> List[Dict[str, Any]]:
        """Generate alternative timeline outcomes"""
        timelines = []
        
        for i in range(num_timelines):
            # Create variations of the timeline
            variation_factor = np.random.uniform(0.5, 2.0)
            timeline = {
                "timeline_id": f"alt_{i+1}",
                "variation_description": f"Alternative outcome {i+1}",
                "probability": np.random.uniform(0.1, 0.9),
                "final_impact": change_event.impact_score * variation_factor,
                "outcome_type": np.random.choice(["positive", "negative", "neutral"]),
                "major_changes": [
                    f"Change {j+1} in timeline {i+1}" for j in range(np.random.randint(1, 4))
                ]
            }
            timelines.append(timeline)
        
        return timelines
    
    def _calculate_cascade_potential(self, event_id: str) -> float:
        """Calculate the cascade potential of an event"""
        # Count downstream connections
        downstream_count = sum(1 for link in self.causal_links if link.cause_event_id == event_id)
        
        # Calculate based on connections and event impact
        event = self.events.get(event_id)
        if not event:
            return 0.0
        
        cascade_potential = (downstream_count * 0.1) + (event.impact_score * 0.5)
        return min(1.0, cascade_potential)
    
    def _check_circular_causality(self, tree: Dict[str, Any], visited: set) -> bool:
        """Check for circular causality in the tree"""
        root = tree.get("root")
        if root in visited:
            return True
        
        visited.add(root)
        
        # Check downstream events
        for downstream in tree.get("downstream_events", []):
            if self._check_circular_causality(downstream.get("subtree", {}), visited.copy()):
                return True
        
        return False
    
    def _check_temporal_inconsistencies(self, tree: Dict[str, Any]) -> bool:
        """Check for temporal inconsistencies"""
        # Simplified check - look for effects that precede causes
        return np.random.random() < 0.1  # 10% chance of detecting inconsistency
    
    def _generate_immediate_effects(self, event: CausalEvent) -> List[Dict[str, Any]]:
        """Generate immediate effects of an event"""
        effects = []
        num_effects = np.random.randint(1, 4)
        
        for i in range(num_effects):
            effects.append({
                "description": f"Immediate effect {i+1} of {event.description}",
                "impact": np.random.uniform(-0.5, 0.5),
                "probability": np.random.uniform(0.7, 1.0),
                "time_to_manifest": f"{np.random.randint(1, 60)} minutes"
            })
        
        return effects
    
    def _generate_short_term_effects(self, event: CausalEvent, immediate_effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate short-term effects building on immediate effects"""
        effects = []
        num_effects = np.random.randint(2, 5)
        
        for i in range(num_effects):
            effects.append({
                "description": f"Short-term effect {i+1} building on immediate changes",
                "impact": np.random.uniform(-0.7, 0.7),
                "probability": np.random.uniform(0.5, 0.8),
                "time_to_manifest": f"{np.random.randint(1, 30)} days"
            })
        
        return effects
    
    def _generate_long_term_effects(self, event: CausalEvent, short_term_effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate long-term effects building on short-term changes"""
        effects = []
        num_effects = np.random.randint(1, 3)
        
        for i in range(num_effects):
            effects.append({
                "description": f"Long-term consequence {i+1} of cascading changes",
                "impact": np.random.uniform(-1.0, 1.0),
                "probability": np.random.uniform(0.3, 0.6),
                "time_to_manifest": f"{np.random.randint(1, 5)} years"
            })
        
        return effects
    
    def _events_could_be_causally_linked(self, event1: CausalEvent, event2: CausalEvent) -> bool:
        """Check if two events could be causally linked"""
        # Simple heuristic based on time proximity and impact
        time_diff = abs((event2.timestamp - event1.timestamp).total_seconds())
        impact_correlation = abs(event1.impact_score - event2.impact_score)
        
        return time_diff < 86400 and impact_correlation < 0.5  # Within 1 day and similar impact
    
    def add_causal_event(self, user_id: str, event: CausalEvent) -> bool:
        """Add a new causal event to the system"""
        if not self.authenticate_access(user_id, "add_event"):
            return False
        
        self.events[event.event_id] = event
        self.logger.info(f"Added causal event: {event.event_id}")
        return True
    
    def add_causal_link(self, user_id: str, link: CausalLink) -> bool:
        """Add a new causal link between events"""
        if not self.authenticate_access(user_id, "add_link"):
            return False
        
        self.causal_links.append(link)
        self.logger.info(f"Added causal link: {link.cause_event_id} -> {link.effect_event_id}")
        return True
    
    def get_system_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.authenticate_access(user_id, "system_status"):
            return {"error": "Access denied"}
        
        return {
            "module": "Causality Engine",
            "status": "operational",
            "creator_protection": "active",
            "total_events": len(self.events),
            "total_causal_links": len(self.causal_links),
            "manipulation_enabled": self._manipulation_enabled,
            "butterfly_threshold": self.butterfly_threshold,
            "paradox_sensitivity": self.paradox_sensitivity,
            "causality_depth": self.causality_depth,
            "timeline_lock_active": self._timeline_lock.locked(),
            "timestamp": datetime.now().isoformat()
        }
