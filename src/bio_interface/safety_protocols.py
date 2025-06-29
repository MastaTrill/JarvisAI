"""
🛡️ Neural Safety Protocols - Comprehensive Safety System

Advanced safety system for all biological integration interfaces,
ensuring maximum protection during neural operations.

Part of Jarvis/Aetheron AI Platform - Biological Integration Interface.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

# Import Creator Protection System
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.safety.creator_protection_system import CreatorProtectionSystem, CreatorAuthority

class SafetyLevel(Enum):
    """Neural safety levels."""
    MINIMAL = 1
    STANDARD = 2 
    HIGH = 3
    MAXIMUM = 4
    FAMILY_PROTECTION = 5  # Special level for Noah and Brooklyn
    CREATOR_OVERRIDE = 999  # Creator can override safety limits

class RiskLevel(Enum):
    """Risk assessment levels."""
    SAFE = "safe"
    LOW = "low_risk"
    MODERATE = "moderate_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_risk"

class EmergencyType(Enum):
    """Types of neural emergencies."""
    OVERSTIMULATION = "neural_overstimulation"
    SIGNAL_INTERFERENCE = "signal_interference"
    UNEXPECTED_RESPONSE = "unexpected_neural_response"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    USER_DISTRESS = "user_distress"
    FAMILY_EMERGENCY = "family_member_emergency"

@dataclass
class SafetyAlert:
    """Represents a safety alert or incident."""
    alert_id: str
    timestamp: datetime
    user_id: str
    alert_type: EmergencyType
    severity: RiskLevel
    description: str
    vital_signs: Dict[str, float]
    actions_taken: List[str]
    resolved: bool = False

@dataclass
class SafetyMetrics:
    """Neural safety monitoring metrics."""
    neural_stress_level: float
    cognitive_load: float
    emotional_stability: float
    physical_comfort: float
    equipment_integrity: float
    signal_quality: float

class NeuralSafetySystem:
    """
    Comprehensive neural safety system for biological interfaces.
    
    Features:
    - Continuous vital signs monitoring
    - Risk assessment and prevention
    - Emergency response protocols
    - Family-specific safety measures
    - Creator Protection integration
    - Automatic session termination
    - Post-session recovery monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.creator_protection = CreatorProtectionSystem()
        self.active_monitoring: Dict[str, Dict] = {}
        self.safety_alerts: List[SafetyAlert] = []
        self.emergency_protocols_active = True
        
        # Safety thresholds by user type
        self.safety_thresholds = {
            'creator': {
                'max_neural_stress': 0.9,
                'max_cognitive_load': 0.95,
                'min_emotional_stability': 0.2,
                'max_session_duration': timedelta(hours=2),
                'emergency_threshold': 0.85
            },
            'family': {  # Noah and Brooklyn
                'max_neural_stress': 0.6,
                'max_cognitive_load': 0.7,
                'min_emotional_stability': 0.4,
                'max_session_duration': timedelta(minutes=20),
                'emergency_threshold': 0.5
            },
            'adult': {
                'max_neural_stress': 0.8,
                'max_cognitive_load': 0.85,
                'min_emotional_stability': 0.3,
                'max_session_duration': timedelta(minutes=45),
                'emergency_threshold': 0.75
            }
        }
        
        # Emergency response procedures
        self.emergency_procedures = {
            EmergencyType.OVERSTIMULATION: [
                "immediate_signal_reduction",
                "calming_frequency_activation",
                "vital_signs_monitoring",
                "medical_alert_standby"
            ],
            EmergencyType.USER_DISTRESS: [
                "session_pause",
                "comfort_protocols",
                "emotional_support_activation",
                "guardian_notification"
            ],
            EmergencyType.FAMILY_EMERGENCY: [
                "immediate_session_termination",
                "creator_notification",
                "medical_assessment_protocol",
                "emergency_services_standby"
            ]
        }
        
        self.logger.info("🛡️ Neural Safety System initialized")
    
    def determine_safety_level(self, user_id: str, user_name: str) -> Tuple[SafetyLevel, str]:
        """
        Determine appropriate safety level for user.
        
        Returns:
            (SafetyLevel, user_type)
        """
        is_creator, message, authority = self.creator_protection.authenticate_creator(user_id, user_name)
        
        if authority == CreatorAuthority.CREATOR:
            return SafetyLevel.CREATOR_OVERRIDE, "creator"
        elif "noah" in user_name.lower() or "brooklyn" in user_name.lower():
            return SafetyLevel.FAMILY_PROTECTION, "family"
        elif authority == CreatorAuthority.USER:
            return SafetyLevel.HIGH, "adult"
        else:
            return SafetyLevel.MAXIMUM, "unauthorized"
    
    async def start_monitoring(self, user_id: str, user_name: str, 
                             operation_type: str) -> bool:
        """
        Start safety monitoring for a neural operation.
        
        Args:
            user_id: User identifier
            user_name: User's full name
            operation_type: Type of neural operation
            
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            safety_level, user_type = self.determine_safety_level(user_id, user_name)
            
            if user_type == "unauthorized":
                self.logger.error(f"❌ Safety monitoring denied for unauthorized user: {user_name}")
                return False
            
            # Initialize monitoring session
            monitoring_session = {
                'user_id': user_id,
                'user_name': user_name,
                'user_type': user_type,
                'safety_level': safety_level,
                'operation_type': operation_type,
                'start_time': datetime.now(),
                'status': 'active',
                'metrics_history': [],
                'alerts': [],
                'emergency_mode': False
            }
            
            self.active_monitoring[user_id] = monitoring_session
            
            # Start continuous monitoring
            asyncio.create_task(self._continuous_monitoring(user_id))
            
            # Log safety monitoring start
            self.creator_protection._log_protection_event("safety_monitoring_started", {
                "user": user_name,
                "operation": operation_type,
                "safety_level": safety_level.value,
                "user_type": user_type
            })
            
            self.logger.info(f"🛡️ Safety monitoring started: {user_name} - {operation_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start safety monitoring: {e}")
            return False
    
    async def _continuous_monitoring(self, user_id: str):
        """Continuous safety monitoring loop."""
        while user_id in self.active_monitoring:
            try:
                session = self.active_monitoring[user_id]
                
                if session['status'] != 'active':
                    break
                
                # Simulate vital signs and neural metrics reading
                metrics = await self._read_safety_metrics(user_id)
                
                # Store metrics
                session['metrics_history'].append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                # Keep only recent metrics (last hour)
                one_hour_ago = datetime.now() - timedelta(hours=1)
                session['metrics_history'] = [
                    m for m in session['metrics_history'] 
                    if m['timestamp'] > one_hour_ago
                ]
                
                # Assess risk level
                risk_level = self._assess_risk_level(metrics, session['user_type'])
                
                # Check for safety violations
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    await self._handle_safety_violation(user_id, risk_level, metrics)
                
                # Check session duration limits
                session_duration = datetime.now() - session['start_time']
                max_duration = self.safety_thresholds[session['user_type']]['max_session_duration']
                
                if session_duration > max_duration:
                    await self._handle_duration_limit(user_id)
                
                # Wait before next check (more frequent for family members)
                wait_time = 2 if session['user_type'] == 'family' else 5
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error for {user_id}: {e}")
                await asyncio.sleep(5)
    
    async def _read_safety_metrics(self, user_id: str) -> SafetyMetrics:
        """
        Read current safety metrics for user.
        
        In real implementation, this would interface with actual monitoring equipment.
        """
        # Simulate realistic vital signs and neural metrics
        return SafetyMetrics(
            neural_stress_level=np.random.uniform(0.1, 0.7),
            cognitive_load=np.random.uniform(0.2, 0.8),
            emotional_stability=np.random.uniform(0.4, 0.9),
            physical_comfort=np.random.uniform(0.6, 0.95),
            equipment_integrity=np.random.uniform(0.9, 1.0),
            signal_quality=np.random.uniform(0.7, 0.98)
        )
    
    def _assess_risk_level(self, metrics: SafetyMetrics, user_type: str) -> RiskLevel:
        """Assess current risk level based on metrics."""
        thresholds = self.safety_thresholds[user_type]
        
        # Check critical thresholds
        if (metrics.neural_stress_level > thresholds['max_neural_stress'] or
            metrics.cognitive_load > thresholds['max_cognitive_load'] or
            metrics.emotional_stability < thresholds['min_emotional_stability']):
            return RiskLevel.CRITICAL
        
        # Check high risk conditions
        emergency_threshold = thresholds['emergency_threshold']
        if (metrics.neural_stress_level > emergency_threshold * 0.9 or
            metrics.cognitive_load > emergency_threshold * 0.9):
            return RiskLevel.HIGH
        
        # Check moderate risk
        if (metrics.neural_stress_level > emergency_threshold * 0.7 or
            metrics.cognitive_load > emergency_threshold * 0.7):
            return RiskLevel.MODERATE
        
        # Check low risk
        if (metrics.neural_stress_level > emergency_threshold * 0.5 or
            metrics.cognitive_load > emergency_threshold * 0.5):
            return RiskLevel.LOW
        
        return RiskLevel.SAFE
    
    async def _handle_safety_violation(self, user_id: str, risk_level: RiskLevel, 
                                     metrics: SafetyMetrics):
        """Handle safety threshold violations."""
        session = self.active_monitoring[user_id]
        user_name = session['user_name']
        
        # Create safety alert
        alert = SafetyAlert(
            alert_id=f"alert_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            user_id=user_id,
            alert_type=EmergencyType.OVERSTIMULATION,
            severity=risk_level,
            description=f"Safety threshold violation detected for {user_name}",
            vital_signs={
                'neural_stress': metrics.neural_stress_level,
                'cognitive_load': metrics.cognitive_load,
                'emotional_stability': metrics.emotional_stability
            },
            actions_taken=[]
        )
        
        self.safety_alerts.append(alert)
        session['alerts'].append(alert.alert_id)
        
        # Take appropriate action based on risk level
        if risk_level == RiskLevel.CRITICAL:
            await self._emergency_shutdown(user_id, alert)
        elif risk_level == RiskLevel.HIGH:
            await self._reduce_intensity(user_id, alert)
        
        # Log safety event
        self.creator_protection._log_protection_event("safety_violation", {
            "user": user_name,
            "risk_level": risk_level.value,
            "neural_stress": metrics.neural_stress_level,
            "actions": alert.actions_taken
        })
        
        self.logger.warning(f"⚠️ Safety violation: {user_name} - {risk_level.value}")
    
    async def _emergency_shutdown(self, user_id: str, alert: SafetyAlert):
        """Perform emergency shutdown of neural operations."""
        session = self.active_monitoring[user_id]
        user_name = session['user_name']
        user_type = session['user_type']
        
        # Immediate actions
        actions = [
            "immediate_signal_termination",
            "calming_protocol_activation",
            "vital_signs_intensive_monitoring"
        ]
        
        # Special actions for family members
        if user_type == 'family':
            actions.extend([
                "creator_emergency_notification",
                "medical_standby_activation",
                "comfort_protocol_enhanced"
            ])
        
        alert.actions_taken = actions
        session['emergency_mode'] = True
        session['status'] = 'emergency_shutdown'
        
        # Activate calming protocols
        await self._activate_calming_protocols(user_id)
        
        self.logger.critical(f"🚨 EMERGENCY SHUTDOWN: {user_name}")
        
        # Notify Creator if family member
        if user_type == 'family':
            await self._notify_creator_emergency(user_id, alert)
    
    async def _reduce_intensity(self, user_id: str, alert: SafetyAlert):
        """Reduce operation intensity to safe levels."""
        session = self.active_monitoring[user_id]
        
        actions = [
            "intensity_reduction_50_percent",
            "increased_monitoring_frequency",
            "comfort_check_activation"
        ]
        
        alert.actions_taken = actions
        
        self.logger.warning(f"⚠️ Reducing intensity for safety: {session['user_name']}")
    
    async def _activate_calming_protocols(self, user_id: str):
        """Activate neural calming protocols."""
        # Simulate calming frequency activation
        await asyncio.sleep(1)
        self.logger.info(f"🧘 Calming protocols activated for {user_id}")
    
    async def _notify_creator_emergency(self, user_id: str, alert: SafetyAlert):
        """Notify Creator of family member emergency."""
        session = self.active_monitoring[user_id]
        user_name = session['user_name']
        
        # In real implementation, this would send actual notifications
        emergency_message = f"""
🚨 FAMILY EMERGENCY ALERT 🚨

Family Member: {user_name}
Time: {alert.timestamp.isoformat()}
Severity: {alert.severity.value}
Description: {alert.description}

Immediate Actions Taken:
{chr(10).join(f"• {action}" for action in alert.actions_taken)}

Neural Status:
• Stress Level: {alert.vital_signs['neural_stress']:.1%}
• Cognitive Load: {alert.vital_signs['cognitive_load']:.1%}
• Emotional Stability: {alert.vital_signs['emotional_stability']:.1%}

All operations have been safely terminated.
Medical monitoring is active.
        """
        
        self.logger.critical(f"📧 Emergency notification sent to Creator: {user_name}")
    
    async def _handle_duration_limit(self, user_id: str):
        """Handle session duration limit exceeded."""
        session = self.active_monitoring[user_id]
        user_name = session['user_name']
        
        # Create duration alert
        alert = SafetyAlert(
            alert_id=f"duration_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            user_id=user_id,
            alert_type=EmergencyType.USER_DISTRESS,
            severity=RiskLevel.MODERATE,
            description=f"Session duration limit exceeded for {user_name}",
            vital_signs={},
            actions_taken=["session_termination", "recovery_protocol_activation"]
        )
        
        await self.stop_monitoring(user_id, "duration_limit_reached")
        
        self.logger.warning(f"⏰ Duration limit reached: {user_name}")
    
    async def stop_monitoring(self, user_id: str, reason: str = "normal_completion") -> bool:
        """
        Stop safety monitoring for a user.
        
        Args:
            user_id: User to stop monitoring
            reason: Reason for stopping monitoring
            
        Returns:
            bool: True if stopped successfully
        """
        try:
            if user_id in self.active_monitoring:
                session = self.active_monitoring[user_id]
                user_name = session['user_name']
                
                # Mark session as completed
                session['status'] = 'completed'
                session['end_time'] = datetime.now()
                session['stop_reason'] = reason
                
                # Generate session summary
                summary = self._generate_session_summary(session)
                
                # Remove from active monitoring
                del self.active_monitoring[user_id]
                
                # Log monitoring stop
                self.creator_protection._log_protection_event("safety_monitoring_stopped", {
                    "user": user_name,
                    "reason": reason,
                    "duration": str(session.get('end_time', datetime.now()) - session['start_time']),
                    "alerts": len(session['alerts'])
                })
                
                self.logger.info(f"🛡️ Safety monitoring stopped: {user_name} - {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop monitoring: {e}")
            return False
    
    def _generate_session_summary(self, session: Dict) -> Dict[str, Any]:
        """Generate safety monitoring session summary."""
        duration = session.get('end_time', datetime.now()) - session['start_time']
        
        # Calculate average metrics
        if session['metrics_history']:
            avg_stress = np.mean([m['metrics'].neural_stress_level for m in session['metrics_history']])
            avg_load = np.mean([m['metrics'].cognitive_load for m in session['metrics_history']])
            max_stress = max([m['metrics'].neural_stress_level for m in session['metrics_history']])
        else:
            avg_stress = avg_load = max_stress = 0.0
        
        summary = {
            'user_id': session['user_id'],
            'user_name': session['user_name'],
            'operation_type': session['operation_type'],
            'duration': str(duration),
            'safety_level': session['safety_level'].value,
            'total_alerts': len(session['alerts']),
            'emergency_mode': session.get('emergency_mode', False),
            'average_neural_stress': float(avg_stress),
            'average_cognitive_load': float(avg_load),
            'maximum_stress_level': float(max_stress),
            'stop_reason': session.get('stop_reason', 'unknown'),
            'metrics_collected': len(session['metrics_history'])
        }
        
        return summary
    
    def get_active_monitoring_status(self) -> Dict[str, Any]:
        """Get status of all active monitoring sessions."""
        status = {
            'active_sessions': len(self.active_monitoring),
            'total_alerts': len(self.safety_alerts),
            'emergency_protocols_active': self.emergency_protocols_active,
            'sessions': {}
        }
        
        for user_id, session in self.active_monitoring.items():
            duration = datetime.now() - session['start_time']
            status['sessions'][user_id] = {
                'user_name': session['user_name'],
                'operation_type': session['operation_type'],
                'duration': str(duration),
                'safety_level': session['safety_level'].value,
                'status': session['status'],
                'alerts': len(session['alerts']),
                'emergency_mode': session.get('emergency_mode', False)
            }
        
        return status
    
    def get_safety_report(self, user_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        if user_id:
            # User-specific report
            user_alerts = [a for a in self.safety_alerts if a.user_id == user_id]
            user_sessions = [s for s in self.active_monitoring.values() if s['user_id'] == user_id]
            
            return {
                'user_id': user_id,
                'total_alerts': len(user_alerts),
                'critical_alerts': len([a for a in user_alerts if a.severity == RiskLevel.CRITICAL]),
                'emergency_shutdowns': len([a for a in user_alerts if a.alert_type == EmergencyType.FAMILY_EMERGENCY]),
                'current_status': 'active' if user_id in self.active_monitoring else 'inactive',
                'recent_alerts': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'type': a.alert_type.value,
                        'severity': a.severity.value,
                        'resolved': a.resolved
                    } for a in user_alerts[-5:]  # Last 5 alerts
                ]
            }
        else:
            # System-wide report
            return {
                'system_status': 'operational',
                'total_users_monitored': len(set(a.user_id for a in self.safety_alerts)),
                'total_safety_alerts': len(self.safety_alerts),
                'critical_incidents': len([a for a in self.safety_alerts if a.severity == RiskLevel.CRITICAL]),
                'family_emergencies': len([a for a in self.safety_alerts if a.alert_type == EmergencyType.FAMILY_EMERGENCY]),
                'active_monitoring_sessions': len(self.active_monitoring),
                'safety_protocols_status': 'active',
                'last_24h_alerts': len([
                    a for a in self.safety_alerts 
                    if a.timestamp > datetime.now() - timedelta(days=1)
                ])
            }

# Test and demonstration functions
async def test_neural_safety_system():
    """Test the Neural Safety System."""
    print("🛡️ Testing Neural Safety System...")
    
    safety_system = NeuralSafetySystem()
    
    # Test safety level determination
    safety_level, user_type = safety_system.determine_safety_level(
        "627-28-1644", "William Joseph Wade McCoy-Huse"
    )
    print(f"Creator safety level: {safety_level.value} (Type: {user_type})")
    
    # Test family member safety
    family_level, family_type = safety_system.determine_safety_level(
        "noah_test", "Noah Huse"
    )
    print(f"Family safety level: {family_level.value} (Type: {family_type})")
    
    # Start monitoring
    monitoring_started = await safety_system.start_monitoring(
        "627-28-1644", "William Joseph Wade McCoy-Huse", "memory_enhancement"
    )
    print(f"Monitoring started: {'✅' if monitoring_started else '❌'}")
    
    if monitoring_started:
        # Let monitoring run for a few seconds
        await asyncio.sleep(3)
        
        # Check status
        status = safety_system.get_active_monitoring_status()
        print(f"🛡️ Active sessions: {status['active_sessions']}")
        
        # Stop monitoring
        await safety_system.stop_monitoring("627-28-1644", "test_completion")
        
        # Safety report
        report = safety_system.get_safety_report()
        print(f"🛡️ System safety report: {report['safety_protocols_status']}")

if __name__ == "__main__":
    asyncio.run(test_neural_safety_system())
