import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from animation_database import AnimationDatabase
from feedback_system import FeedbackSystem

# load_dotenv()


class MovementFrame(BaseModel):
    """A single frame of movement data for the 3D model"""
    duration: int = Field(default=1000, description="Duration of this frame in milliseconds")
    head_y: float = Field(default=0, description="Head Y rotation (-90 to 90)")
    left_arm_x: float = Field(default=0, description="Left arm X rotation (0 to -90)")
    left_arm_y: float = Field(default=0, description="Left arm Y rotation (0 to -90)")
    left_arm_z: float = Field(default=0, description="Left arm Z rotation (-90 to 90)")
    left_elbow_x: float = Field(default=0, description="Left elbow X rotation (0 to -145)")
    left_elbow_z: float = Field(default=0, description="Left elbow Z rotation (up/down, -45 to 45)")
    left_hand_x: float = Field(default=0, description="Left hand X rotation (0 to -70)")
    left_hand_y: float = Field(default=0, description="Left hand Y rotation (0 to -70)")
    left_hand_z: float = Field(default=0, description="Left hand Z rotation (90 to -90)")
    right_arm_x: float = Field(default=0, description="Right arm X rotation (0 to 90)")
    right_arm_y: float = Field(default=0, description="Right arm Y rotation (0 to 90)")
    right_arm_z: float = Field(default=0, description="Right arm Z rotation (-90 to 90)")
    right_elbow_x: float = Field(default=0, description="Right elbow X rotation (0 to 145)")
    right_elbow_z: float = Field(default=0, description="Right elbow Z rotation (up/down, -45 to 45)")
    right_hand_x: float = Field(default=0, description="Right hand X rotation (0 to 100)")
    right_hand_y: float = Field(default=0, description="Right hand Y rotation (0 to 90)")
    right_hand_z: float = Field(default=0, description="Right hand Z rotation (-90 to 90)")
    left_thumb_1_x: float = Field(default=0, description="Left thumb MCP X rotation (-60 to 60)")
    left_thumb_2_x: float = Field(default=0, description="Left thumb PIP X rotation (0 to 80)")
    left_thumb_3_x: float = Field(default=0, description="Left thumb DIP X rotation (0 to 90)")
    left_index_1_z: float = Field(default=0, description="Left index MCP Z rotation (-30 to 80)")
    left_index_2_z: float = Field(default=0, description="Left index PIP Z rotation (0 to 120)")
    left_index_3_z: float = Field(default=0, description="Left index DIP Z rotation (0 to 80)")
    left_middle_1_z: float = Field(default=0, description="Left middle MCP Z rotation (-30 to 80)")
    left_middle_2_z: float = Field(default=0, description="Left middle PIP Z rotation (0 to 120)")
    left_middle_3_z: float = Field(default=0, description="Left middle DIP Z rotation (0 to 80)")
    left_ring_1_z: float = Field(default=0, description="Left ring MCP Z rotation (-30 to 80)")
    left_ring_2_z: float = Field(default=0, description="Left ring PIP Z rotation (0 to 120)")
    left_ring_3_z: float = Field(default=0, description="Left ring DIP Z rotation (0 to 80)")
    left_pinky_1_z: float = Field(default=0, description="Left pinky MCP Z rotation (-30 to 80)")
    left_pinky_2_z: float = Field(default=0, description="Left pinky PIP Z rotation (0 to 120)")
    left_pinky_3_z: float = Field(default=0, description="Left pinky DIP Z rotation (0 to 80)")
    right_thumb_1_x: float = Field(default=0, description="Right thumb MCP X rotation (-60 to 60)")
    right_thumb_2_x: float = Field(default=0, description="Right thumb PIP X rotation (0 to 80)")
    right_thumb_3_x: float = Field(default=0, description="Right thumb DIP X rotation (0 to 90)")
    right_index_1_z: float = Field(default=0, description="Right index MCP Z rotation (-30 to 80)")
    right_index_2_z: float = Field(default=0, description="Right index PIP Z rotation (0 to 120)")
    right_index_3_z: float = Field(default=0, description="Right index DIP Z rotation (0 to 80)")
    right_middle_1_z: float = Field(default=0, description="Right middle MCP Z rotation (-30 to 80)")
    right_middle_2_z: float = Field(default=0, description="Right middle PIP Z rotation (0 to 120)")
    right_middle_3_z: float = Field(default=0, description="Right middle DIP Z rotation (0 to 80)")
    right_ring_1_z: float = Field(default=0, description="Right ring MCP Z rotation (-30 to 80)")
    right_ring_2_z: float = Field(default=0, description="Right ring PIP Z rotation (0 to 120)")
    right_ring_3_z: float = Field(default=0, description="Right ring DIP Z rotation (0 to 80)")
    right_pinky_1_z: float = Field(default=0, description="Right pinky MCP Z rotation (-30 to 80)")
    right_pinky_2_z: float = Field(default=0, description="Right pinky PIP Z rotation (0 to 120)")
    right_pinky_3_z: float = Field(default=0, description="Right pinky DIP Z rotation (0 to 80)")


class MovementsOutput(BaseModel):
    """Complete movements output containing all animation frames"""
    movements: List[MovementFrame] = Field(description="List of movement frames for the animation sequence")


class EnhancedSignLanguageAgent:
    """Enhanced AI agent with memory and feedback capabilities"""

    def __init__(self, api_key=None):
        # Initialize database and feedback system
        self.database = AnimationDatabase()
        self.feedback_system = FeedbackSystem()
        apis = ["add your openrouter api key"]  # 55c is jarvis and paid one

        # Initialize the LLM"
        # self.llm = LLM(model="gemini/gemini-2.5-flash-preview-05-20", temperature=0.1)
        self.llm = LLM(model="openrouter/openai/gpt-6-luna-pro", base_url="https://openrouter.ai/api/v1", temperature=0.4,api_key=apis[1])

        # Create the sign language expert agent
        self.sign_language_expert = Agent(
            role="Sign Language Expert",
            goal="Translate spoken language to precise sign language movements",
            backstory="""You are an expert in sign language with deep knowledge of how to
            translate spoken language into precise hand, arm, movements. You understand
            the nuances of sign language and can describe how to show any text in ASL.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

        # Create the 3D animation expert agent
        self.animation_expert = Agent(
            role="3D Animation Expert",
            goal="Convert sign language descriptions into precise 3D model control parameters",
            backstory="""You are an expert in 3D animation who specializes in translating
            sign language movements into precise rotation angles for 3D model joints. You know
            exactly how to position each joint to create realistic signing movements.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def generate_movements(self, text_input: str, feedback_context: Optional[Dict] = None):
        """Generate 3D model movements - feedback collection handled separately"""

        print(f"\n[v0] Processing text: '{text_input}'")

        # Check cache first
        cached_animation = self.database.get_animation(text_input)
        if cached_animation and not feedback_context:
            print(f"[v0] Using cached animation for: '{text_input}'")
            return cached_animation["movements"]

        # Generate new animation
        movements_data = await self._generate_new_animation(text_input, feedback_context)

        if movements_data and "movements" in movements_data:
            return movements_data["movements"]

        return None

    async def _generate_new_animation(self, text_input: str, feedback_context: Optional[Dict] = None):
        """Generate a new animation using the AI agents"""

        # Build the task description with feedback context if available
        sign_language_description = f"""Translate the following text into detailed sign language movements:
        '{text_input}'

        Describe the precise movements needed for each word or phrase, including:
        - Head position and rotation
        - Arm positions and rotations
        - Hand positions and rotations
        - Finger positions for each sign

        You should use arm for waving the hand not elbows.
        You should accurately decide what body part to move to show the {text_input} accurately.


        Be specific about the sequence of movements and transitions between signs."""

        if feedback_context:
            sign_language_description += f"""

        IMPORTANT: The previous animation had issues. User feedback: "{feedback_context.get('comments', '')}"
        Please address these concerns and create an improved version."""

        # Task for the sign language expert
        sign_language_task = Task(
            description=sign_language_description,
            agent=self.sign_language_expert,
            expected_output="Detailed description of how to show the input text in ASL. "
        )

        animation_description = f"""Convert the sign language movements into precise 3D model control parameters for: '{text_input}'

        Each frame should specify exact rotation angles (in degrees) for all joints.
        You have to use 10-12 Frames for best ASL Conversion.

        How to make rotational angle:
        - For making left arm come in front (forward)(Arm X rotation) always use negative angles from -0 to -90 degrees only.
        - For rotating the left shoulder so that we can see inside arm, you must use negative angles from 0 to -90.
        - For closing the left arm downside use positive angle from 0 to 90 and for making the left arm up in the direction of head you will use negative angles ranging from 0 to -90.
        - For left elbow rotation always use negative angles from 0 to -145.
        - For left hand(wrist) x,y movement always use negative angles from 0 to -90
        Note: Hand Z rotation ranges from -90 to +90.
        - For making right arm come in front (forward)(Arm X Rotation) always use positive angles from 0 to 90 degrees only.
        - For rotating the right shoulder so that we can see inside arm, you must use positive angles from 0 to 90.
        - For closing the right arm downside use positive angle from 0 to 90 and for making the left arm up in the direction of head you will use negative angles ranging from 0 to -90.
        - For right elbow rotation always use positive angles from 0 to 145.
        - For right hand(wrist) x,y movement always use positive angles from 0 to 90
        For left elbow up use -90 and for down 90. 
        For right elbow up use -90 and for down 90. 
        Note: Hand Z rotation ranges from -90 to +90.
        - For rotating the hand wrist in upward use -90 of Z and for downwards use +90 for both right and left hand.
        - For fingers bend (all 3 joints) towards the palm of hand always use positive angles from 0 to 120.
        - For right hand's thumb closing use negative values from 0 to -120.
        Duration should be in milliseconds and above 800ms.
        Use a little high rotations and in correct direction as specified above. 


        IMPORTANT: Stay within these anatomical limits for realistic human movement.
        Create smooth transitions between poses that accurately represent the sign language for the input text.
        Start with a neutral pose, move through the signing positions, and end in a neutral or final pose."""

        if feedback_context:
            animation_description += f"""

        CRITICAL: Address the user's feedback: "{feedback_context.get('comments', '')}"
        Make specific improvements to fix the mentioned issues."""

        animation_task = Task(
            description=animation_description,
            agent=self.animation_expert,
            expected_output="Structured movement data with anatomically correct joint angles for 3D animation",
            output_json=MovementsOutput,
            context=[sign_language_task]
        )

        # Create the crew with sequential process
        crew = Crew(
            agents=[self.sign_language_expert, self.animation_expert],
            tasks=[sign_language_task, animation_task],
            verbose=True,
            process=Process.sequential
        )

        # Execute the crew's tasks
        result = crew.kickoff()

        try:
            # Access the structured JSON output from the animation task
            if hasattr(result, 'json_dict') and result.json_dict:
                movements_data = result.json_dict
                print(
                    f"[v0] Successfully extracted structured movements: {len(movements_data.get('movements', []))} frames")
                return movements_data
            else:
                # Fallback to parsing raw output if structured output fails
                print("[v0] Structured output not available, trying to parse raw result")
                return self._parse_raw_result(str(result))

        except Exception as e:
            print(f"[v0] Error extracting structured output: {e}")
            print(f"[v0] Raw result type: {type(result)}")
            print(f"[v0] Raw result: {result}")
            return self._parse_raw_result(str(result))

    async def _generate_improved_animation(self, text_input: str, original_movements: List[Dict],
                                           feedback_comments: str):
        """Generate an improved animation based on user feedback"""

        feedback_context = {
            "comments": feedback_comments,
            "original_movements": original_movements
        }

        improved_data = await self._generate_new_animation(text_input, feedback_context)

        if improved_data and "movements" in improved_data:
            return improved_data["movements"]

        return None

    def _parse_raw_result(self, result_str):
        """Fallback method to parse raw string result"""
        try:
            # Find JSON in the result
            json_start = result_str.find('{')
            json_end = result_str.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = result_str[json_start:json_end + 1]
                movements_data = json.loads(json_str)
                return movements_data
            else:
                # If no JSON found, try to extract it from markdown code blocks
                import re
                json_match = re.search(re.txt('\`\`\`(?:json)?\s*({.*?})\s*\`\`\`'), result_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    movements_data = json.loads(json_str)
                    return movements_data
                else:
                    raise ValueError("Could not extract JSON from the result")
        except Exception as e:
            print(f"[v0] Error parsing raw JSON: {e}")
            # Return a default movement if parsing fails
            return {
                "movements": [
                    {
                        "duration": 1000,
                        "head_y": 0,
                        "left_arm_x": -30, "left_arm_y": 0, "left_arm_z": 45,
                        "left_elbow_x": 0, "left_elbow_z": 0,
                        "right_arm_x": -30, "right_arm_y": 0, "right_arm_z": -45,
                        "right_elbow_x": 0, "right_elbow_z": 0,
                        "left_hand_x": 20, "left_hand_y": 20, "left_hand_z": 20,
                        "right_hand_x": -20, "right_hand_y": -20, "right_hand_z": -20,
                        "left_thumb_1_x": 0, "left_thumb_2_x": 0, "left_thumb_3_x": 0,
                        "left_index_1_z": 30, "left_index_2_z": 45, "left_index_3_z": 45,
                        "left_middle_1_z": 30, "left_middle_2_z": 45, "left_middle_3_z": 45,
                        "left_ring_1_z": 30, "left_ring_2_z": 45, "left_ring_3_z": 45,
                        "left_pinky_1_z": 30, "left_pinky_2_z": 45, "left_pinky_3_z": 45,
                        "right_thumb_1_x": 0, "right_thumb_2_x": 0, "right_thumb_3_x": 0,
                        "right_index_1_z": 30, "right_index_2_z": 45, "right_index_3_z": 45,
                        "right_middle_1_z": 30, "right_middle_2_z": 45, "right_middle_3_z": 45,
                        "right_ring_1_z": 30, "right_ring_2_z": 45, "right_ring_3_z": 45,
                        "right_pinky_1_z": 30, "right_pinky_2_z": 45, "right_pinky_3_z": 45
                    }
                ]
            }

    def get_database_stats(self):
        """Get database statistics"""
        return self.database.get_stats()

    def get_feedback_history(self):
        """Get feedback history"""
        return self.feedback_system.get_feedback_history()
