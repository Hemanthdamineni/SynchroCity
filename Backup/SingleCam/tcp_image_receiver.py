import socket
import base64
import os
import threading
import time
from datetime import datetime
import yaml

class TCPImageReceiver:
    def __init__(self, host='localhost', port=25002, save_folder='received_images'):
        self.host = host
        self.port = port
        self.save_folder = save_folder
        self.running = False
        self.server_socket = None
        self.image_counter = 0
        
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
                image_data = self.receive_exact(client_socket, data_size)
                if not image_data:
                    break
                
                # Process the received image
                self.process_image_data(image_data.decode('utf-8'), client_address)
                
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

    def process_image_data(self, base64_data, client_address):
        """Process and save the received base64 image data"""
        try:
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Generate filename with timestamp and counter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.image_counter += 1
            filename = f"unity_image_{timestamp}_{self.image_counter:04d}.png"
            filepath = os.path.join(self.save_folder, filename)
            
            # Save image to file
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
                
            print(f"Image saved: {filename} ({len(image_bytes)} bytes) from {client_address[0]}")
            
        except Exception as e:
            print(f"Error processing image data: {e}")

    def stop_server(self):
        """Stop the TCP server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")

def load_config():
    """Load port configuration"""
    try:
        with open('configs/port_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
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