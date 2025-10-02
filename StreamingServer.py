import os
import json
import base64
import sys
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import socket
import struct
import logging

app = Flask(__name__)
CORS(app)

# Configuration
FLASK_PORT = 5000
TCP_PORT = 25002
WEBPAGE_FOLDER = os.path.join(os.getcwd(), 'Webpage')

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"WEBPAGE_FOLDER resolved to: {WEBPAGE_FOLDER}")
print(f"WEBPAGE_FOLDER exists: {os.path.exists(WEBPAGE_FOLDER)}")
if os.path.exists(WEBPAGE_FOLDER):
    print(f"Files in WEBPAGE_FOLDER: {os.listdir(WEBPAGE_FOLDER)}")

# Global state for storing latest images and ground truth data
latest_images = {}  # {camera_id: {image_data: bytes, timestamp: float, ground_truth: list}}
image_counters = {}  # {camera_id: counter}

class TCPImageReceiver:
    def __init__(self, host='localhost', port=TCP_PORT):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        
    def start_server(self):
        """Start the TCP server to receive images"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"TCP Image Receiver listening on {self.host}:{self.port}")
            print("Press Ctrl+C to stop the server")
            
            while self.running:
                try:
                    # Set timeout to allow for periodic checking of self.running
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection established with {client_address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    # This is expected, just continue checking self.running
                    continue
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
            # Set a timeout for the client socket
            client_socket.settimeout(30.0)
            
            while self.running:
                try:
                    print(f"\n--- Waiting for data from {client_address} ---")
                    # First, receive the size of the incoming data
                    size_data = self.receive_exact(client_socket, 4)
                    if not size_data:
                        print(f"No size data received from {client_address}")
                        break
                    
                    # Debug: Print the raw bytes received for size
                    print(f"Raw size bytes received: {size_data}")
                    print(f"Size bytes as hex: {[hex(b) for b in size_data]}")
                    
                    # Try both byte orders to see which one gives a reasonable result
                    data_size_big = int.from_bytes(size_data, byteorder='big')
                    data_size_little = int.from_bytes(size_data, byteorder='little')
                    print(f"Data size (big-endian): {data_size_big}")
                    print(f"Data size (little-endian): {data_size_little}")
                    
                    # Use the more reasonable value (the smaller one that's not zero)
                    data_size = data_size_big
                    if data_size_big > 10000000 and data_size_little <= 10000000:  # 10MB limit
                        data_size = data_size_little
                        print(f"Using little-endian interpretation: {data_size}")
                    else:
                        print(f"Using big-endian interpretation: {data_size}")
                    
                    print(f"Expecting {data_size} bytes of image data")
                    
                    # Check for unreasonable data sizes
                    if data_size <= 0 or data_size > 10000000:  # 10MB limit
                        print(f"Unreasonable data size received: {data_size}")
                        print("This might indicate a byte order issue or corrupted data")
                        break
                    
                    # Receive the actual image data
                    message_data = self.receive_exact(client_socket, data_size)
                    if not message_data:
                        print(f"No message data received from {client_address}")
                        break
                    
                    print(f"Received {len(message_data)} bytes of message data")
                    
                    # Process the received message
                    self.process_message(message_data.decode('utf-8'), client_address)
                    
                    # Send acknowledgment
                    client_socket.send(b"ACK")
                    print(f"Sent ACK to {client_address}")
                    
                except socket.timeout:
                    print(f"Client {client_address} timed out")
                    break
                except Exception as e:
                    print(f"Error handling client {client_address}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"Error in client handler for {client_address}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            client_socket.close()
            print(f"Connection with {client_address} closed")

    def receive_exact(self, sock, num_bytes):
        """Receive exactly num_bytes from socket"""
        data = b''
        attempts = 0
        max_attempts = 100  # Prevent infinite loop
        
        print(f"Attempting to receive exactly {num_bytes} bytes")
        
        while len(data) < num_bytes and attempts < max_attempts:
            try:
                # Calculate how many bytes we still need
                remaining_bytes = num_bytes - len(data)
                print(f"  Need {remaining_bytes} more bytes, total so far: {len(data)}")
                
                chunk = sock.recv(remaining_bytes)
                print(f"  Received chunk of {len(chunk)} bytes: {chunk}")
                
                if not chunk:
                    print("  Received empty chunk, connection may be closed")
                    return None
                    
                data += chunk
                attempts += 1
                
            except socket.error as e:
                print(f"Socket receive error: {e}")
                return None
                
        print(f"Finished receiving, got {len(data)} bytes: {data}")
        
        if len(data) != num_bytes:
            print(f"Warning: Expected {num_bytes} bytes but got {len(data)} bytes")
            return None
            
        return data

    def process_message(self, message, client_address):
        """Process the received JSON message containing camera info and image data"""
        try:
            print(f"Processing message of length: {len(message)}")
            print(f"First 200 chars of message: {message[:200]}")
            
            # Parse the JSON message
            data = json.loads(message)
            camera_id = data.get('camera_id', 'unknown')
            base64_data = data.get('image_data', '')
            ground_truth_objects = data.get('ground_truth_objects', [])
            
            print(f"Camera ID: {camera_id}, Base64 data length: {len(base64_data)}, Objects: {len(ground_truth_objects)}")
            
            # Skip empty data
            if not base64_data:
                print(f"Empty image data received from camera '{camera_id}'")
                return
            
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Store image and ground truth data in memory
            latest_images[camera_id] = {
                'image_data': image_bytes,
                'timestamp': time.time(),
                'ground_truth': ground_truth_objects
            }
            
            # Initialize counter for this camera if not exists
            if camera_id not in image_counters:
                image_counters[camera_id] = 0
                
            image_counters[camera_id] += 1
            
            print(f"Image received: {len(image_bytes)} bytes from camera '{camera_id}' at {client_address[0]} with {len(ground_truth_objects)} objects")
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON message: {e}")
            print(f"Message content: {message[:100]}...")
        except base64.binascii.Error as e:
            print(f"Base64 decode error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def get_latest_image(self, camera_id):
        """Get the latest image for a specific camera"""
        return latest_images.get(camera_id)

    def get_camera_list(self):
        """Get list of available cameras"""
        cameras = []
        for camera_id in latest_images.keys():
            cameras.append({
                'id': len(cameras) + 1,
                'name': camera_id,
                'status': 'online',
                'last_seen': latest_images[camera_id]['timestamp'],
                'object_count': len(latest_images[camera_id].get('ground_truth', []))
            })
        return cameras

    def stop_server(self):
        """Stop the TCP server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("TCP Server stopped")

# Global TCP receiver instance
tcp_receiver = TCPImageReceiver()

@app.route('/')
def index():
    """Serve the main dashboard"""
    print(f"Serving index.html from: {WEBPAGE_FOLDER}")
    return send_from_directory(WEBPAGE_FOLDER, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    print(f"Serving static file: {filename} from: {WEBPAGE_FOLDER}")
    return send_from_directory(WEBPAGE_FOLDER, filename)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    cameras = tcp_receiver.get_camera_list()
    online_cameras = len(cameras)
    
    total_objects = 0
    for camera_id in latest_images.keys():
        total_objects += len(latest_images[camera_id].get('ground_truth', []))
    
    return jsonify({
        'connection_active': True,
        'total_cameras': len(cameras),
        'cameras_online': online_cameras,
        'total_frames': sum(image_counters.get(cam['name'], 0) for cam in cameras),
        'total_objects': total_objects,
        'last_update': time.time()
    })

@app.route('/api/cameras')
def get_cameras():
    """Get list of cameras"""
    return jsonify(tcp_receiver.get_camera_list())

@app.route('/api/ground_truth/<camera_name>')
def get_ground_truth(camera_name):
    """Get ground truth data for a specific camera"""
    camera_data = tcp_receiver.get_latest_image(camera_name)
    
    if camera_data and 'ground_truth' in camera_data:
        return jsonify(camera_data['ground_truth'])
    else:
        return jsonify([])

@app.route('/frame/<camera_name>')
def get_frame(camera_name):
    """Get the latest frame for a specific camera"""
    camera_data = tcp_receiver.get_latest_image(camera_name)
    
    if camera_data and 'image_data' in camera_data:
        try:
            # Return image data as base64
            encoded_image = base64.b64encode(camera_data['image_data']).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"
            
            return jsonify({
                'image': image_url,
                'status': 'active',
                'frame_count': image_counters.get(camera_name, 0),
                'camera': camera_name,
                'object_count': len(camera_data.get('ground_truth', []))
            })
        except Exception as e:
            return jsonify({
                'error': f'Failed to process image: {str(e)}',
                'status': 'error'
            }), 500
    else:
        return jsonify({
            'error': 'No image available for this camera',
            'status': 'offline'
        }), 404

def start_tcp_receiver():
    """Start the TCP receiver in a separate thread"""
    tcp_thread = threading.Thread(target=tcp_receiver.start_server)
    tcp_thread.daemon = True
    tcp_thread.start()
    print("TCP Receiver started")

if __name__ == '__main__':
    # Start the TCP receiver
    start_tcp_receiver()
    
    # Disable Flask/Werkzeug logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    try:
        print(f"Starting web server on http://localhost:{FLASK_PORT}")
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
        tcp_receiver.stop_server()