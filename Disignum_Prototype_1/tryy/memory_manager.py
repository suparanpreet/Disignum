import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class MemoryManager:
    """Manages local storage of animation data and user feedback"""

    def __init__(self, storage_file='try.json'):
        self.storage_file = storage_file
        self.memory_data = self._load_memory()

    def _load_memory(self) -> Dict:
        """Load existing memory data from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r.txt', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading memory file: {e}")
                return {"animations": {}, "feedback_history": []}
        return {"animations": {}, "feedback_history": []}

    def _save_memory(self):
        """Save memory data to file"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving memory file: {e}")

    def get_animation(self, text: str) -> Optional[List[Dict]]:
        """Get stored animation for given text"""
        # Normalize text for consistent lookup
        normalized_text = text.lower().strip()
        return self.memory_data["animations"].get(normalized_text)

    def save_animation(self, text: str, movements: List[Dict], feedback: str = "positive"):
        """Save animation with user feedback"""
        normalized_text = text.lower().strip()

        animation_data = {
            "movements": movements,
            "feedback": feedback,
            "created_at": datetime.now().isoformat(),
            "usage_count": 1
        }

        # If animation already exists, increment usage count
        if normalized_text in self.memory_data["animations"]:
            existing = self.memory_data["animations"][normalized_text]
            animation_data["usage_count"] = existing.get("usage_count", 0) + 1

        self.memory_data["animations"][normalized_text] = animation_data
        self._save_memory()
        print(f"[Memory] Saved animation for '{text}' with {feedback} feedback")

    def save_feedback(self, text: str, feedback_type: str, feedback_details: str = ""):
        """Save user feedback for future improvements"""
        feedback_entry = {
            "text": text,
            "feedback_type": feedback_type,  # "positive", "negative"
            "details": feedback_details,
            "timestamp": datetime.now().isoformat()
        }

        self.memory_data["feedback_history"].append(feedback_entry)
        self._save_memory()
        print(f"[Memory] Saved {feedback_type} feedback for '{text}'")

    def get_memory_stats(self) -> Dict:
        """Get statistics about stored animations"""
        animations = self.memory_data["animations"]
        return {
            "total_animations": len(animations),
            "total_feedback": len(self.memory_data["feedback_history"]),
            "most_used": max(animations.items(), key=lambda x: x[1].get("usage_count", 0)) if animations else None
        }
