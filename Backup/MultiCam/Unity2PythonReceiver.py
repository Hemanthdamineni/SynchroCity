import socket
import base64
import os
import threading
import time
from datetime import datetime
import yaml
import json

class TCPImageReceiver:
    def __init__(self, host='localhost', port=25002, save_folder='test_multi_camera'):
        self.host = host
        self.port = port
        self.save_folder = save_folder
        self.running = False
        self.server_socket = None
        self.image_counters = {}  # Counter for each camera
        
        # Create save folder if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            
        print(f"TCP Image Receiver initialized on {host}:{port}")
        print(f"Images will be saved to: {os.path.abspath(self.save_folder)}")

    def start_server(self):
        """Start the TCP server to receive images"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"Server listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection established with {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")
                        
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            self.stop_server()

    def handle_client(self, client_socket, client_address):
        """Handle incoming client connection"""
        try:
            while self.running:
                # First, receive the size of the incoming data
                size_data = self.receive_exact(client_socket, 4)
                if not size_data:
                    break
                    
                data_size = int.from_bytes(size_data, byteorder='big')
                print(f"Expecting {data_size} bytes of image data")
                
                # Receive the actual image data
                message_data = self.receive_exact(client_socket, data_size)
                if not message_data:
                    break
                
                # Process the received message
                self.process_message(message_data.decode('utf-8'), client_address)
                
                # Send acknowledgment
                client_socket.send(b"ACK")
                
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection with {client_address} closed")

    def receive_exact(self, sock, num_bytes):
        """Receive exactly num_bytes from socket"""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def process_message(self, message, client_address):
        """Process the received JSON message containing camera info and image data"""
        try:
            # Parse the JSON message
            data = json.loads(message)
            camera_id = data.get('camera_id', 'unknown')
            base64_data = data.get('image_data', '')
            
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Create camera-specific folder if it doesn't exist
            camera_folder = os.path.join(self.save_folder, camera_id)
            if not os.path.exists(camera_folder):
                os.makedirs(camera_folder)
            
            # Generate filename with timestamp and counter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize counter for this camera if not exists
            if camera_id not in self.image_counters:
                self.image_counters[camera_id] = 0
                
            self.image_counters[camera_id] += 1
            filename = f"{camera_id}_image_{timestamp}_{self.image_counters[camera_id]:04d}.png"
            filepath = os.path.join(camera_folder, filename)
            
            # Save image to file
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
                
            print(f"Image saved: {filename} ({len(image_bytes)} bytes) from camera '{camera_id}' at {client_address[0]}")
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON message: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")

    def stop_server(self):
        """Stop the TCP server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")

def load_config():
    """Load port configuration"""
    try:
        # Try different possible paths for the config file
        config_paths = [
            'configs/port_config.yaml',
            '../configs/port_config.yaml',
            '../../configs/port_config.yaml',
            'c:/Users/heman/Desktop/Traffic Simulator/configs/port_config.yaml'
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
                
        print("Could not find config file in any expected location")
        return {}
    except Exception as e:
        print(f"Could not load config: {e}")
        return {}

def main():
    # Load configuration
    config = load_config()
    
    # Use configured port or default
    port = config.get('unity_image_tcp', 25002)
    
    # Create and start the receiver
    receiver = TCPImageReceiver(port=port)
    
    try:
        receiver.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        receiver.stop_server()

if __name__ == "__main__":
    main()