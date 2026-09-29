import asyncio
from typing import Dict, List, Optional


class FeedbackSystem:
    """System for collecting user feedback in the terminal"""

    def __init__(self):
        self.feedback_history = []

    async def collect_feedback(self, text: str, movements: List[Dict]) -> Dict:
        """Collect feedback from user in terminal"""
        print("\n" + "=" * 60)
        print(f"🎭 ANIMATION FEEDBACK REQUEST")
        print("=" * 60)
        print(f"Text: '{text}'")
        print(f"Animation frames: {len(movements)}")
        print("-" * 60)

        # Get user feedback
        while True:
            try:
                print("\nDid the animation look good? (y/n/q)")
                print("  y = Yes, save this animation")
                print("  n = No, needs improvement")
                print("  q = Quit without saving")

                # Use asyncio to get input without blocking
                response = await self._get_async_input("Your choice: ")
                response = response.lower().strip()

                if response in ['y', 'yes']:
                    feedback_result = {
                        "action": "save",
                        "rating": "positive",
                        "comments": "",
                        "text": text,
                        "movements": movements
                    }
                    print("✅ Animation will be saved to database!")
                    break

                elif response in ['n', 'no']:
                    print("\n💭 What was wrong with the animation?")
                    print("Please describe the issues (or press Enter to skip):")

                    comments = await self._get_async_input("Feedback: ")

                    feedback_result = {
                        "action": "regenerate",
                        "rating": "negative",
                        "comments": comments.strip(),
                        "text": text,
                        "original_movements": movements
                    }
                    print("🔄 Will generate improved animation...")
                    break

                elif response in ['q', 'quit']:
                    feedback_result = {
                        "action": "skip",
                        "rating": "neutral",
                        "comments": "User chose to skip",
                        "text": text
                    }
                    print("⏭️  Skipping without saving...")
                    break

                else:
                    print("❌ Please enter 'y', 'n', or 'q'")
                    continue

            except KeyboardInterrupt:
                print("\n\n⚠️  Feedback collection interrupted by user")
                feedback_result = {
                    "action": "skip",
                    "rating": "interrupted",
                    "comments": "Interrupted by user",
                    "text": text
                }
                break
            except Exception as e:
                print(f"\n❌ Error collecting feedback: {e}")
                feedback_result = {
                    "action": "skip",
                    "rating": "error",
                    "comments": f"Error: {e}",
                    "text": text
                }
                break

        print("=" * 60)

        # Store feedback in history
        self.feedback_history.append(feedback_result)

        return feedback_result

    async def _get_async_input(self, prompt: str) -> str:
        """Get user input asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)

    def get_feedback_history(self) -> List[Dict]:
        """Get all feedback history"""
        return self.feedback_history.copy()

    def clear_feedback_history(self):
        """Clear feedback history"""
        self.feedback_history.clear()
        print("[v0] Feedback history cleared")
