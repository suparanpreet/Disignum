
import asyncio
import webbrowser
import os
import http.server
import socketserver
import threading
from enhanced_websocket_Server import EnhancedSignLanguageWebSocketServer

# Configuration
WEBSOCKET_HOST = 'localhost'
WEBSOCKET_PORT = 8765
HTTP_PORT = 8080
HTTP_PORT_ALT = 8081  # Alternative HTTP port if default is in use

# Path to the HTML file
HTML_FILE = 'try.html'


class HttpServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.httpd = None

    def start(self):
        """Start a simple HTTP server to serve the HTML file"""
        handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer((self.host, self.port), handler)
        print(f"HTTP server started at http://{self.host}:{self.port}")
        print(f"Open {HTML_FILE} in your browser at http://{self.host}:{self.port}/{HTML_FILE}")
        self.httpd.serve_forever()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()


async def main():
    # Check if HTTP port is available
    import socket
    http_port_to_use = HTTP_PORT

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((WEBSOCKET_HOST, HTTP_PORT))
        sock.close()

        if result == 0:
            print(f"Warning: Port {HTTP_PORT} is already in use. Trying to use alternative port {HTTP_PORT_ALT}.")
            http_port_to_use = HTTP_PORT_ALT

            # Check if alternative port is also in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((WEBSOCKET_HOST, HTTP_PORT_ALT))
            sock.close()

            if result == 0:
                print(f"Error: Alternative port {HTTP_PORT_ALT} is also in use. Please free up one of these ports.")
                return
    except Exception as e:
        print(f"Error checking HTTP port: {e}")

    # Create and start the HTTP server with the available port
    http_server = HttpServer(WEBSOCKET_HOST, http_port_to_use)
    http_thread = threading.Thread(target=http_server.start, daemon=True)
    http_thread.start()

    # Give the HTTP server time to start
    await asyncio.sleep(2)

    # Check if the server is running
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((WEBSOCKET_HOST, http_port_to_use))
        if result != 0:
            print(f"Warning: HTTP server may not be running properly on port {http_port_to_use}")
        else:
            print(f"HTTP server confirmed running on port {http_port_to_use}")
            # Open the HTML file in the default browser
            webbrowser.open(f"http://{WEBSOCKET_HOST}:{http_port_to_use}/{HTML_FILE}")
        sock.close()
    except Exception as e:
        print(f"Error checking HTTP server: {e}")

    try:
        # Check if WebSocket port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((WEBSOCKET_HOST, WEBSOCKET_PORT))
        sock.close()

        if result == 0:
            print(f"Warning: Port {WEBSOCKET_PORT} is already in use. Trying to use a different port.")
            # Try a different port
            WEBSOCKET_PORT_ALT = WEBSOCKET_PORT + 1
            server = EnhancedSignLanguageWebSocketServer(host=WEBSOCKET_HOST, port=WEBSOCKET_PORT_ALT)
            await server.start_server()
        else:
            server = EnhancedSignLanguageWebSocketServer(host=WEBSOCKET_HOST, port=WEBSOCKET_PORT)
            await server.start_server()
    except Exception as e:
        print(f"Error starting Enhanced WebSocket server: {e}")
        raise


if __name__ == "__main__":
    print("Starting Enhanced Sign Language Animation System...")
    print("🎭 Features: Memory System, Feedback Collection, Animation Caching")
    print("📝 Feedback will be collected in this terminal after each animation")
    try:
        # Get the event loop
        loop = asyncio.get_event_loop()

        # Run the main coroutine
        loop.run_until_complete(main())

        # Keep the event loop running indefinitely
        print("Enhanced server running. Press Ctrl+C to stop.")
        print("💡 After each animation, you'll be asked for feedback in this terminal!")
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down the enhanced server...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the event loop
        if 'loop' in locals() and loop.is_running():
            loop.close()
