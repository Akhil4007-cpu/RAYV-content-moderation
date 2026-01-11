"""
Action Recognition Module
Detects specific actions/gestures that indicate violence
Future: Can be extended with specialized action recognition models
"""
import numpy as np
from typing import List, Dict, Tuple
import config

class ActionRecognizer:
    """
    Action recognition based on object tracking and motion patterns
    Future: Replace with SlowFast, ActionTubes, or Pose-based action recognition
    """
    
    def __init__(self):
        # Action patterns (heuristic-based, can be learned)
        self.violent_action_patterns = {
            "punching": {"required": ["person", "person"], "motion_threshold": 2.0, "proximity": True},
            "kicking": {"required": ["person", "person"], "motion_threshold": 2.5, "proximity": True},
            "throwing": {"required": ["person"], "motion_threshold": 2.0, "object_motion": True},
            "chasing": {"required": ["person", "person"], "motion_threshold": 3.0, "trajectory": True},
            "grabbing": {"required": ["person", "person"], "motion_threshold": 1.5, "proximity": True}
        }
    
    def detect_actions(self, window_objects: List[List[str]], motion_scores: List[float], 
                      captions: List[str], tags: str) -> Dict[str, float]:
        """
        Detect violent actions based on patterns
        Returns: Dict of action -> confidence
        """
        actions_detected = {}
        
        # Check for punching/kicking (multiple persons + high motion)
        person_counts = [len([obj for obj in objs if obj == "person"]) for objs in window_objects]
        avg_persons = np.mean(person_counts) if person_counts else 0
        
        if avg_persons >= 2 and np.mean(motion_scores) > 2.0:
            # Potential conflict - multiple people + high motion
            if any("fight" in cap.lower() or "attack" in cap.lower() for cap in captions):
                actions_detected["fighting"] = 0.7
            elif any("person" in tag.lower() and "person" in tag.lower() for tag in [tags]):
                actions_detected["conflict"] = 0.6
        
        # Check for throwing (high motion + object changes)
        if np.mean(motion_scores) > 2.5 and len(set([len(objs) for objs in window_objects])) > 2:
            if any("throw" in cap.lower() or "throwing" in cap.lower() for cap in captions):
                actions_detected["throwing"] = 0.65
        
        # Check for chasing (increasing motion)
        if len(motion_scores) >= 3:
            if motion_scores[-1] > motion_scores[0] * 1.5:  # Escalating motion
                if any("chase" in cap.lower() or "run" in cap.lower() for cap in captions):
                    actions_detected["chasing"] = 0.6
        
        # Check tags for action indicators
        tags_lower = tags.lower()
        if "fight" in tags_lower or "conflict" in tags_lower:
            actions_detected["fighting"] = actions_detected.get("fighting", 0.5) + 0.2
        
        if "attack" in tags_lower or "assault" in tags_lower:
            actions_detected["attacking"] = 0.7
        
        return actions_detected
    
    def calculate_action_risk(self, actions: Dict[str, float]) -> float:
        """Calculate risk contribution from detected actions"""
        if not actions:
            return 0.0
        
        # Weight different actions
        action_weights = {
            "fighting": 0.5,
            "attacking": 0.6,
            "throwing": 0.4,
            "chasing": 0.3,
            "conflict": 0.4
        }
        
        total_risk = 0.0
        for action, confidence in actions.items():
            weight = action_weights.get(action, 0.3)
            total_risk += weight * confidence
        
        return min(total_risk, 1.0)  # Cap at 1.0


# Global instance
_action_recognizer = None

def get_action_recognizer() -> ActionRecognizer:
    """Get global ActionRecognizer instance"""
    global _action_recognizer
    if _action_recognizer is None:
        _action_recognizer = ActionRecognizer()
    return _action_recognizer
