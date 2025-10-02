# Step 1: Install required modules
# pip install pyzmq

import zmq
import json
import time
import sys

# Step 2: Setup ZeroMQ context and sockets
context = zmq.Context()

# Subscriber socket to receive data from SUMO (matches C# PUB socket connection)
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:5556")
sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
sub_socket.setsockopt(zmq.RCVHWM, 1000)   # High water mark

# Dealer socket to send data to SUMO (matches C# ROUTER socket connection)  
dealer_socket = context.socket(zmq.DEALER)
dealer_socket.connect("tcp://localhost:5557")
dealer_socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark

print("Attempting to connect to SUMO via ZeroMQ...")
print("Connecting to ports 5556 (SUB) and 5557 (DEALER)")

# Step 3: Define Variables
total_speed = 0
step_count = 0

# Step 4: Define Functions
def process_sumo_data(data_json):
    """Process received SUMO data"""
    global total_speed
    try:
        data = json.loads(data_json)
        # Process based on your data structure
        # This depends on what format your C#/SUMO integration sends
        
        if 'vehicles' in data:
            total_speed = 0  # Reset for this step
            for vehicle in data['vehicles']:
                if 'speed' in vehicle and 'position' in vehicle:
                    vehicle_id = vehicle.get('id', 'unknown')
                    speed = vehicle['speed']
                    position = vehicle['position']
                    total_speed += speed
                    print(f"Vehicle {vehicle_id} - Speed: {speed:.2f} m/s, Position: {position}")
        
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

# Step 5: Main monitoring loop
print("Starting ZeroMQ SUMO monitoring...")
print("Press Ctrl+C to stop")

try:
    while True:
        try:
            # Try to receive data from SUMO
            try:
                sumo_data = sub_socket.recv_string(zmq.NOBLOCK)
                print(f"Received data from SUMO: {sumo_data[:100]}...")  # Show first 100 chars
                
                # Process the received data
                processed_data = process_sumo_data(sumo_data)
                step_count += 1
                
                # Print summary every 50 steps
                if step_count % 50 == 0:
                    print(f"Step {step_count}: Total Speed={total_speed:.2f}")
                    
            except zmq.Again:
                # No message available, continue
                pass
            
            # Small delay to prevent 100% CPU usage
            time.sleep(0.01)  # 10ms delay
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
            
except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
except Exception as e:
    print(f"Unexpected error: {e}")

# Step 6: Cleanup
print("Closing ZeroMQ connections...")
sub_socket.close()
dealer_socket.close()
context.term()
print("ZeroMQ connections closed.")