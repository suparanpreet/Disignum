import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class MovementFrame(BaseModel):
    """A single frame of movement data for the 3D model"""
    duration: int = Field(default=1000, description="Duration of this frame in milliseconds")
    head_y: float = Field(default=0, description="Head Y rotation (-90 to 90)")
    left_arm_x: float = Field(default=0, description="Left arm X rotation (-60 to 180)")
    left_arm_y: float = Field(default=0, description="Left arm Y rotation (-90 to 90)")
    left_arm_z: float = Field(default=0, description="Left arm Z rotation (-180 to 180)")
    left_elbow_x: float = Field(default=0, description="Left elbow X rotation (0 to 145)")
    left_hand_x: float = Field(default=0, description="Left hand X rotation (-80 to 70)")
    left_hand_y: float = Field(default=0, description="Left hand Y rotation (-20 to 30)")
    left_hand_z: float = Field(default=0, description="Left hand Z rotation (-80 to 80)")
    right_arm_x: float = Field(default=0, description="Right arm X rotation (-60 to 180)")
    right_arm_y: float = Field(default=0, description="Right arm Y rotation (-90 to 90)")
    right_arm_z: float = Field(default=0, description="Right arm Z rotation (-180 to 180)")
    right_elbow_x: float = Field(default=0, description="Right elbow X rotation (0 to 145)")
    right_hand_x: float = Field(default=0, description="Right hand X rotation (-80 to 70)")
    right_hand_y: float = Field(default=0, description="Right hand Y rotation (-20 to 30)")
    right_hand_z: float = Field(default=0, description="Right hand Z rotation (-80 to 80)")
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


class SignLanguageAgent:
    def __init__(self, api_key=None):
        # Use provided API key or get from environment
        # self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # if not self.api_key:
        #     raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
        #
        # Initialize the LLM
        self.llm = LLM(model="gemini/gemini-2.5-flash-preview-05-20", temperature=0.7)
        # apis = ["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",
        #         "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",
        #         "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"]  # 55c is jarvis and paid one

        # self.llm = LLM(model="openrouter/openai/gpt-4o-mini", temperature=0.7, base_url="https://openrouter.ai/api/v1",api_key=apis[1])

        # Create the sign language expert agent
        self.sign_language_expert = Agent(
            role="Sign Language Expert",
            goal="Translate spoken language to precise sign language movements",
            backstory="""You are an expert in sign language with deep knowledge of how to 
            translate spoken language into precise hand, arm, and body movements. You understand 
            the nuances of sign language and can describe the exact angles and positions needed 
            for accurate signing.""",
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

    def generate_movements(self, text_input):
        """Generate 3D model movements based on the text input"""

        # Task for the sign language expert
        sign_language_task = Task(
            description=f"""Translate the following text into detailed sign language movements: 
            '{text_input}'

            Describe the precise movements needed for each word or phrase, including:
            - Head position and rotation
            - Arm positions and rotations
            - Hand positions and rotations
            - Finger positions for each sign

            Be specific about the sequence of movements and transitions between signs.
            """,
            agent=self.sign_language_expert,
            expected_output="Detailed description of sign language movements for the input text"
        )

        animation_task = Task(
            description=f"""Convert the sign language movements into precise 3D model control parameters for: '{text_input}'

            Each frame should specify exact rotation angles (in degrees) for all joints.
            You have to use 10-12 Frames for best ASL Conversion. 

            Guidelines for rotations (ANATOMICALLY CORRECT RANGES):
            - Head Y rotation: -90 to 90 degrees (natural neck movement)
            - Arm X rotation: -60 to 180 degrees (shoulder flexion/extension)
            - Arm Y rotation: -90 to 90 degrees (shoulder internal/external rotation)  
            - Arm Z rotation: -180 to 180 degrees (shoulder abduction/adduction)
            - Elbow X rotation: 0 to 145 degrees (elbow flexion only, no hyperextension)
            - Hand X rotation: -80 to 70 degrees (wrist flexion/extension)
            - Hand Y rotation: -20 to 30 degrees (wrist radial/ulnar deviation)
            - Hand Z rotation: -80 to 80 degrees (wrist pronation/supination)
            - Finger MCP joints (joint 1): -30 to 80 degrees (can extend backward slightly)
            - Finger PIP joints (joint 2): 0 to 120 degrees (middle knuckle flexion)
            - Finger DIP joints (joint 3): 0 to 80 degrees (fingertip flexion)
            - Thumb joints: Different ranges for natural thumb movement
            - Duration: 800-1500ms per frame for natural movement

            IMPORTANT: Stay within these anatomical limits for realistic human movement.
            Create smooth transitions between poses that accurately represent the sign language for the input text.
            Start with a neutral pose, move through the signing positions, and end in a neutral or final pose.
            """,
            agent=self.animation_expert,
            expected_output="Structured movement data with anatomically correct joint angles for 3D animation",
            output_json=MovementsOutput,  # Use Pydantic model for structured output
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
                json_match = re.search(r.txt'\`\`\`(?:json)?\s*({.*?})\s*\`\`\`', result_str, re.DOTALL)
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
                        "right_arm_x": -30, "right_arm_y": 0, "right_arm_z": -45,
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


# Example usage
if __name__ == "__main__":
    # For testing purposes
    agent = SignLanguageAgent()
    movements = agent.generate_movements("Hello, how are you?")
    print(json.dumps(movements, indent=2))

