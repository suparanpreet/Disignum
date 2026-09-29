import asyncio
import json
import websockets
import os
from enhanced_ai_agent import EnhancedSignLanguageAgent


class EnhancedSignLanguageWebSocketServer:
    """Enhanced WebSocket server with memory and feedback capabilities"""

    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.connected_clients = set()

        try:
            self.agent = EnhancedSignLanguageAgent()
            print("Enhanced AI Agent with memory and feedback initialized successfully")
        except Exception as e:
            print(f"Error initializing Enhanced AI Agent: {e}")
            self.agent = None

    async def handle_client(self, websocket):
        """Handle a client connection"""
        # Register the client
        self.connected_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.connected_clients)}")

        try:
            async for message in websocket:
                print(f"Received message: {message}")

                try:
                    # Parse the message as JSON
                    data = json.loads(message)

                    # Check if this is a text input for sign language generation
                    if 'text' in data:
                        text_input = data['text']
                        print(f"Processing text input: {text_input}")

                        if self.agent:
                            movements_data = await self.process_text_input_with_feedback(text_input, websocket)

                            if not movements_data:
                                print("[v0] No movements data received from AI agent")
                                error_msg = {"error": "No valid movements generated"}
                                await websocket.send(json.dumps(error_msg))
                        else:
                            error_msg = {"error": "Enhanced AI Agent not initialized"}
                            await websocket.send(json.dumps(error_msg))
                    else:
                        # Unknown message type
                        await websocket.send(json.dumps({"error": "Invalid message format"}))

                except json.JSONDecodeError:
                    # Not a JSON message
                    await websocket.send(json.dumps({"error": "Message must be valid JSON"}))
                except Exception as e:
                    # Other errors
                    print(f"Error processing message: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))

        except websockets.exceptions.ConnectionClosed:
            print("Client connection closed")
        finally:
            # Unregister the client
            self.connected_clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    async def process_text_input_with_feedback(self, text_input, websocket):
        """Process text input with memory and feedback system - animation first, then feedback"""

        cached_animation = self.agent.database.get_animation(text_input)
        if cached_animation:
            print(f"[v0] Using cached animation for: '{text_input}' - No feedback needed")
            movements = cached_animation["movements"]

            if isinstance(movements, list):
                response = json.dumps({"movements": movements})
                await websocket.send(response)
                print(f"[v0] Sent {len(movements)} cached animation frames to client")

            return movements

        print(f"[v0] Generating new animation for: '{text_input}'")
        movements_data = await self.agent._generate_new_animation(text_input)

        if movements_data and "movements" in movements_data:
            movements = movements_data["movements"]

            if isinstance(movements, list):
                response = json.dumps({"movements": movements})
                await websocket.send(response)
                print(f"[v0] Sent {len(movements)} new animation frames to client")
            elif isinstance(movements, dict) and 'movements' in movements:
                response = json.dumps({"movements": movements['movements']})
                await websocket.send(response)
                print(f"[v0] Sent {len(movements['movements'])} animation frames to client")

            await self._collect_feedback_after_animation(text_input, movements)
            return movements

        return None

    async def _collect_feedback_after_animation(self, text_input, movements):
        """Collect feedback after animation has been displayed"""

        print(f"[v0] Animation sent to frontend. Waiting for animation to complete...")
        await asyncio.sleep(3)  # Wait 3 seconds for animation to start playing

        feedback_result = await self.agent.feedback_system.collect_feedback(text_input, movements)

        if feedback_result["action"] == "save":
            # Save to database with positive feedback
            self.agent.database.save_animation(text_input, movements, "positive")
            print(f"[v0] Animation for '{text_input}' saved to database")

        elif feedback_result["action"] == "regenerate":
            # Generate improved animation based on feedback
            print(f"[v0] Regenerating animation based on feedback...")
            improved_movements = await self.agent._generate_improved_animation(
                text_input, movements, feedback_result["comments"]
            )

            if improved_movements:
                print(f"[v0] Improved animation generated. You can test it by entering the same text again.")
                # Note: We don't automatically send the improved animation to avoid confusion
                # User can re-enter the text to see the improved version
            else:
                print("[v0] Failed to generate improved animation")

        else:  # skip or other actions
            print("[v0] Animation not saved to database")

    async def start_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        print(f"Enhanced WebSocket server started at ws://{self.host}:{self.port}")
        print("Features enabled: Memory system, Feedback collection, Animation caching")

        if self.agent:
            stats = self.agent.get_database_stats()
            print(f"Database stats: {stats}")

        return server


# Run the server when the script is executed directly
if __name__ == "__main__":
    # Create and start the enhanced server
    server = EnhancedSignLanguageWebSocketServer()

    # Start the asyncio event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.start_server())

    try:
        print("Enhanced server running. Press Ctrl+C to stop.")
        loop.run_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        loop.close()
