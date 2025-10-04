import socket
import base64
import os
import threading
from datetime import datetime
import yaml
import json

class TCPImageReceiver:
    def __init__(self, host="localhost", port=25002, save_folder="MultiCamImages"):
        self.host = host
        self.port = port
        self.save_folder = save_folder
        self.running = False
        self.server_socket = None
        self.image_counters = {}  # Counter for each camera
        self.stats = {
            'total_received': 0,
            'empty_ground_truth': 0,
            'with_ground_truth': 0
        }

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
            print("Press Ctrl+C to stop the server\n")

            while self.running:
                try:
                    # Set timeout to allow for periodic checking of self.running
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection established with {client_address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket, client_address)
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
                    # First, receive the size of the incoming data
                    size_data = self.receive_exact(client_socket, 4)
                    if not size_data:
                        break

                    data_size = int.from_bytes(size_data, byteorder="big")

                    # Receive the actual image data
                    message_data = self.receive_exact(client_socket, data_size)
                    if not message_data:
                        break

                    # Process the received message
                    success = self.process_message(message_data.decode("utf-8"), client_address)

                    # Send acknowledgment
                    if success:
                        client_socket.send(b"ACK")
                    else:
                        client_socket.send(b"ERR")

                except socket.timeout:
                    print(f"Client {client_address} timed out")
                    break
                except Exception as e:
                    print(f"Error handling client {client_address}: {e}")
                    break

        except Exception as e:
            print(f"Error in client handler for {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection with {client_address} closed")
            self.print_stats()

    def receive_exact(self, sock, num_bytes):
        """Receive exactly num_bytes from socket"""
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.error as e:
                print(f"Socket receive error: {e}")
                return None
        return data

    def process_message(self, message, client_address):
        """Process the received JSON message containing camera info and image data"""
        try:
            # Parse the JSON message
            data = json.loads(message)
            camera_id = data.get("camera_id", "unknown")
            base64_data = data.get("image_data", "")
            ground_truth = data.get("ground_truth", [])

            self.stats['total_received'] += 1

            # Skip empty data
            if not base64_data:
                print(f"Empty image data received from camera '{camera_id}'")
                return

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
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Process ground truth data
            if not isinstance(ground_truth, list):
                print(f"Warning: Ground truth data not in list format from camera '{camera_id}'")
                ground_truth = []
            
            # Only print and save if ground truth is not empty
            if ground_truth:
                self.stats['with_ground_truth'] += 1
                
                # Save ground truth data to file
                gt_filename = f"{camera_id}_ground_truth_{timestamp}_{self.image_counters[camera_id]:04d}.txt"
                gt_filepath = os.path.join(camera_folder, gt_filename)
                
                with open(gt_filepath, "w") as f:
                    for item in ground_truth:
                        if len(item) >= 6:  # label, tag, centerX, centerY, width, height
                            label, tag, cx, cy, w, h = item[0], item[1], item[2], item[3], item[4], item[5]
                            f.write(f"{label} {tag} {cx} {cy} {w} {h}\n")
                        else:
                            # Fallback for old format
                            f.write(f"{' '.join(map(str, item))}\n")
                
                # Print detailed information
                print(f"\n{'='*60}")
                print(f"Camera: '{camera_id}'")
                print(f"Image: {filename} ({len(image_bytes)} bytes)")
                print(f"Objects detected: {len(ground_truth)}")
                print(f"Ground Truth saved to: {gt_filename}")
                print(f"Objects in view:")
                for i, item in enumerate(ground_truth, 1):
                    if len(item) >= 6:
                        label, tag, cx, cy, w, h = item[0], item[1], item[2], item[3], item[4], item[5]
                        print(f"  {i}. {label} (tag: {tag}) - Center: ({cx:.3f}, {cy:.3f}), Size: ({w:.3f}, {h:.3f})")
                    else:
                        print(f"  {i}. {item}")
                print(f"{'='*60}\n")
            else:
                self.stats['empty_ground_truth'] += 1
                # Optionally save empty ground truth file for consistency
                # Uncomment if you want to keep track of frames with no objects
                # gt_filename = f"{camera_id}_ground_truth_{timestamp}_{self.image_counters[camera_id]:04d}.txt"
                # with open(os.path.join(camera_folder, gt_filename), "w") as f:
                #     pass  # Empty file

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON message: {e}")
            print(f"Message content: {message[:100]}...")
        except base64.binascii.Error as e:
            print(f"Base64 decode error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def print_stats(self):
        """Print statistics about received data"""
        print(f"\n{'='*60}")
        print("SESSION STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames received: {self.stats['total_received']}")
        print(f"Frames with objects: {self.stats['with_ground_truth']}")
        print(f"Frames without objects: {self.stats['empty_ground_truth']}")
        if self.stats['total_received'] > 0:
            percentage = (self.stats['with_ground_truth'] / self.stats['total_received']) * 100
            print(f"Detection rate: {percentage:.1f}%")
        print(f"{'='*60}\n")

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
            "configs/port_config.yaml",
            "../configs/port_config.yaml",
            "../../configs/port_config.yaml",
            "c:/Users/heman/Desktop/Traffic Simulator/configs/port_config.yaml",
        ]

        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                print(f"Loaded config from: {path}")
                return config

        print("Could not find config file in any expected location, using defaults")
        return {}
    except Exception as e:
        print(f"Could not load config: {e}")
        return {}


def main():
    # Load configuration
    config = load_config()

    # Use configured port or default
    port = config.get("unity_image_tcp", 25002)

    # Create and start the receiver
    receiver = TCPImageReceiver(port=port)

    try:
        receiver.start_server()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        receiver.stop_server()


if __name__ == "__main__":
    main()