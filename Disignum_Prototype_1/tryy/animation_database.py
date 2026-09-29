import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class AnimationDatabase:
    """Local database for storing animation data using JSON files"""

    def __init__(self, db_file="try.json"):
        self.db_file = db_file
        self.data = self._load_database()

    def _load_database(self) -> Dict:
        """Load the database from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r.txt', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[v0] Error loading database: {e}")
                return {"animations": {}, "metadata": {"created": datetime.now().isoformat()}}
        else:
            return {"animations": {}, "metadata": {"created": datetime.now().isoformat()}}

    def _save_database(self):
        """Save the database to JSON file"""
        try:
            self.data["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print(f"[v0] Database saved successfully")
        except IOError as e:
            print(f"[v0] Error saving database: {e}")

    def get_animation(self, text: str) -> Optional[Dict]:
        """Get animation data for a given text"""
        normalized_text = text.lower().strip()

        if normalized_text in self.data["animations"]:
            animation_data = self.data["animations"][normalized_text]

            # Handle references
            if "ref" in animation_data:
                ref_name = animation_data["ref"]
                print(f"[v0] '{text}' refers to '{ref_name}'")
                return self.get_animation(ref_name)

            print(f"[v0] Found cached animation for: '{text}'")
            return animation_data

        print(f"[v0] No cached animation found for: '{text}'")
        return None

    def save_animation(self, text: str, movements: List[Dict], feedback: str = "positive"):
        """Save animation data for a given text"""
        normalized_text = text.lower().strip()

        animation_entry = {
            "text": text,
            "movements": movements,
            "feedback": feedback,
            "created": datetime.now().isoformat(),
            "usage_count": 1
        }

        # If animation already exists, increment usage count
        if normalized_text in self.data["animations"]:
            existing = self.data["animations"][normalized_text]
            animation_entry["usage_count"] = existing.get("usage_count", 0) + 1
            animation_entry["last_used"] = datetime.now().isoformat()

        self.data["animations"][normalized_text] = animation_entry
        self._save_database()
        print(f"[v0] Saved animation for: '{text}' with {feedback} feedback")

    def delete_animation(self, text: str):
        """Delete animation data for a given text"""
        normalized_text = text.lower().strip()

        if normalized_text in self.data["animations"]:
            del self.data["animations"][normalized_text]
            self._save_database()
            print(f"[v0] Deleted animation for: '{text}'")
            return True

        print(f"[v0] No animation found to delete for: '{text}'")
        return False

    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_animations = len(self.data["animations"])
        total_usage = sum(anim.get("usage_count", 1) for anim in self.data["animations"].values())

        return {
            "total_animations": total_animations,
            "total_usage": total_usage,
            "database_size": f"{os.path.getsize(self.db_file) / 1024:.2f} KB" if os.path.exists(
                self.db_file) else "0 KB"
        }
