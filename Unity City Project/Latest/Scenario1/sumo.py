import traci

# Connect to SUMO started by Sumo2UnityTool
traci.connect(port=5556)

print("Connected to SUMO running via Sumo2UnityTool!")
