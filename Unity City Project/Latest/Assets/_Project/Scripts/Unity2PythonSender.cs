using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.IO;
using System;

public class ImageSender : MonoBehaviour
{
    // Serializable classes for ground truth data
    [System.Serializable]
    public class GroundTruthObject
    {
        public enum ObjectType
        {
            Unknown,
            Vehicle,
            Pedestrian,
            TrafficLight,
            Sign,
            Bicycle
        }
        
        public ObjectType Type;
        public BoundingBox BoundingBox;
        public float Confidence;
        public string ObjectId;
        public SerializableVector3 Position; // Add position information
        public string Name; // Add object name for better identification
    }

    [System.Serializable]
    public class BoundingBox
    {
        public float x;
        public float y;
        public float width;
        public float height;
    }

    [System.Serializable]
    public class SerializableVector3
    {
        public float x;
        public float y;
        public float z;

        public SerializableVector3(Vector3 vector)
        {
            x = vector.x;
            y = vector.y;
            z = vector.z;
        }

        public SerializableVector3() { } // Parameterless constructor for JSON serialization

        public Vector3 ToVector3()
        {
            return new Vector3(x, y, z);
        }
    }

    // New serializable class for frame data
    [System.Serializable]
    public class FrameData
    {
        public string camera_id;
        public string image_data;
        public GroundTruthObject[] ground_truth_objects;
    }

    [Header("Network Settings")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 25002;
    
    [Header("Camera Settings")]
    public Camera[] captureCameras; // Array of cameras to capture from
    public string[] cameraNames;    // Names for each camera (e.g., "front", "rear", "side")
    public int captureWidth = 640;
    public int captureHeight = 480;
    public int captureRate = 30; // FPS
    
    [Header("Image Settings")]
    public bool encodeToJPG = true;
    [Range(10, 100)]
    public int jpgQuality = 75;
    
    [Header("Ground Truth Detection")]
    public bool enableGroundTruthDetection = true;
    public LayerMask vehicleLayerMask = -1; // All layers by default
    public LayerMask pedestrianLayerMask = -1;
    public LayerMask trafficLightLayerMask = -1;
    
    private TcpClient client;
    private NetworkStream stream;
    private RenderTexture[] renderTextures;
    private Texture2D[] textures;
    private bool isConnected = false;
    private Thread connectionThread;
    private float lastCaptureTime = 0f;
    private int currentCameraIndex = 0;
    private bool isConnecting = false;
    private object connectionLock = new object();
    
    void Start()
    {
        // Initialize components if not assigned
        if (captureCameras == null || captureCameras.Length == 0)
        {
            // Auto-find all active cameras in the scene
            captureCameras = FindObjectsOfType<Camera>();
            Debug.Log($"Auto-assigned {captureCameras.Length} cameras");
        }
        
        // Initialize camera names if not provided
        if (cameraNames == null || cameraNames.Length != captureCameras.Length)
        {
            cameraNames = new string[captureCameras.Length];
            for (int i = 0; i < captureCameras.Length; i++)
            {
                // Use camera name or generate a default name
                if (captureCameras[i] != null && !string.IsNullOrEmpty(captureCameras[i].name))
                {
                    cameraNames[i] = captureCameras[i].name;
                }
                else
                {
                    cameraNames[i] = $"Camera_{i+1}";
                }
                Debug.Log($"Assigned camera {i}: {cameraNames[i]}");
            }
        }
        
        // Create render textures and textures for each camera
        renderTextures = new RenderTexture[captureCameras.Length];
        textures = new Texture2D[captureCameras.Length];
        
        for (int i = 0; i < captureCameras.Length; i++)
        {
            renderTextures[i] = new RenderTexture(captureWidth, captureHeight, 24);
            textures[i] = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        }
        
        // Start connection in separate thread
        ConnectToServer();
        
        Debug.Log($"Image Sender initialized with {captureCameras.Length} cameras. Capturing at {captureWidth}x{captureHeight} @{captureRate}fps");
    }
    
    void ConnectToServer()
    {
        lock (connectionLock)
        {
            if (isConnecting || isConnected) return;
            isConnecting = true;
        }
        
        try
        {
            // Close existing connection if any
            if (client != null)
            {
                try
                {
                    client.Close();
                }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"Error closing existing client: {e.Message}");
                }
                client = null;
            }
            
            Debug.Log($"Attempting to connect to server at {serverIP}:{serverPort}");
            client = new TcpClient();
            client.Connect(IPAddress.Parse(serverIP), serverPort);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log($"Connected to server at {serverIP}:{serverPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to server: {e.Message}");
            isConnected = false;
        }
        finally
        {
            isConnecting = false;
        }
    }
    
    void Update()
    {
        // Check if it's time to capture a frame
        if (Time.time - lastCaptureTime >= 1f / captureRate)
        {
            lastCaptureTime = Time.time;
            
            // Check connection status
            if (!isConnected || client == null || !client.Connected)
            {
                Debug.Log("Connection check: Not connected, attempting to reconnect...");
                // Try to reconnect
                ConnectToServer();
            }
            
            // Only proceed if we have a valid connection and cameras
            if (isConnected && client != null && client.Connected && captureCameras.Length > 0)
            {
                // Cycle through cameras
                CaptureAndSendFrame(currentCameraIndex);
                currentCameraIndex = (currentCameraIndex + 1) % captureCameras.Length;
            }
            else
            {
                if (captureCameras.Length == 0)
                {
                    Debug.LogWarning("No cameras configured for capture");
                }
                else if (!isConnected || client == null || !client.Connected)
                {
                    Debug.LogWarning("Not connected to server, skipping frame capture");
                }
            }
        }
    }
    
    void CaptureAndSendFrame(int cameraIndex)
    {
        try
        {
            Camera cam = captureCameras[cameraIndex];
            string camName = cameraNames[cameraIndex];
            
            // Ensure camera name is not null or empty
            if (string.IsNullOrEmpty(camName))
            {
                camName = $"Camera_{cameraIndex + 1}";
                Debug.LogWarning($"Camera name was null or empty, using default name: {camName}");
            }
            
            // Capture camera view
            cam.targetTexture = renderTextures[cameraIndex];
            cam.Render();
            RenderTexture.active = renderTextures[cameraIndex];
            textures[cameraIndex].ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            textures[cameraIndex].Apply();
            cam.targetTexture = null;
            RenderTexture.active = null;
            
            // Encode to JPG or PNG
            byte[] imageData;
            if (encodeToJPG)
            {
                imageData = textures[cameraIndex].EncodeToJPG(jpgQuality);
            }
            else
            {
                imageData = textures[cameraIndex].EncodeToPNG();
            }
            
            // Check if we have valid image data
            if (imageData == null || imageData.Length == 0)
            {
                Debug.LogWarning($"No image data captured for camera {camName}");
                return;
            }
            
            // Convert to base64
            string base64String = System.Convert.ToBase64String(imageData);
            
            // Detect ground truth objects if enabled
            List<GroundTruthObject> groundTruthObjects = new List<GroundTruthObject>();
            if (enableGroundTruthDetection)
            {
                groundTruthObjects = DetectGroundTruthObjects(cam);
            }
            
            // Create a serializable frame data object
            FrameData frameData = new FrameData
            {
                camera_id = camName,
                image_data = base64String,
                ground_truth_objects = groundTruthObjects.ToArray() // Convert to array for proper JSON serialization
            };
            
            // Serialize to JSON
            string message = JsonUtility.ToJson(frameData);
            
            // Debug: Log the actual frameData object before serialization
            Debug.Log($"FrameData object - Camera ID: {frameData.camera_id}, Image data length: {(frameData.image_data != null ? frameData.image_data.Length : 0)}, Ground truth objects: {(frameData.ground_truth_objects != null ? frameData.ground_truth_objects.Length : 0)}");
            
            // Validate JSON message
            if (string.IsNullOrEmpty(message))
            {
                Debug.LogError("Failed to serialize frame data to JSON - message is null or empty");
                return;
            }
            
            // Debug: Log the actual JSON message
            Debug.Log($"Serialized JSON message: {message}");
            
            // Check if the message is just an empty object
            if (message == "{}")
            {
                Debug.LogError("Serialized JSON is empty object - serialization failed, trying fallback");
                
                // Fallback: Create a simple test message
                string fallbackMessage = $"{{\"camera_id\":\"{camName}\",\"image_data\":\"{base64String}\",\"ground_truth_objects\":[]}}";
                message = fallbackMessage;
                Debug.Log($"Using fallback message: {fallbackMessage}");
            }
            
            // Convert message to bytes
            byte[] dataBytes = System.Text.Encoding.UTF8.GetBytes(message);
            
            // Send data size first (4 bytes) in big-endian format
            int dataSize = dataBytes.Length;
            byte[] dataSizeBytes = System.BitConverter.GetBytes(dataSize);
            
            // Ensure big-endian byte order for network transmission
            if (BitConverter.IsLittleEndian)
                Array.Reverse(dataSizeBytes);
                
            // Debug logs with more detail
            Debug.Log($"=== Frame Transmission Debug ===");
            Debug.Log($"Camera: {camName}");
            Debug.Log($"Image data size: {imageData.Length} bytes");
            Debug.Log($"Base64 string length: {base64String.Length}");
            Debug.Log($"JSON message length: {message.Length}");
            Debug.Log($"Message byte length: {dataBytes.Length}");
            Debug.Log($"Data size bytes: [{dataSizeBytes[0]}, {dataSizeBytes[1]}, {dataSizeBytes[2]}, {dataSizeBytes[3]}]");
            Debug.Log($"Data size as int (big-endian): {BitConverter.ToInt32(new byte[] { dataSizeBytes[3], dataSizeBytes[2], dataSizeBytes[1], dataSizeBytes[0] }, 0)}");
            Debug.Log($"First 100 chars of JSON: {message.Substring(0, Math.Min(100, message.Length))}");
            
            // Verify we have a valid connection before sending
            if (client == null || !client.Connected)
            {
                Debug.LogError("Client is not connected, attempting to reconnect...");
                ConnectToServer();
                if (client == null || !client.Connected)
                {
                    Debug.LogError("Failed to reconnect to server");
                    return;
                }
            }
            
            try
            {
                // Send data size first
                Debug.Log("Sending data size...");
                stream.Write(dataSizeBytes, 0, dataSizeBytes.Length);
                stream.Flush();
                Debug.Log("Data size sent and flushed");
                
                // Small delay to ensure data is processed
                System.Threading.Thread.Sleep(1);
                
                // Send message data
                Debug.Log("Sending message data...");
                stream.Write(dataBytes, 0, dataBytes.Length);
                stream.Flush();
                Debug.Log("Message data sent and flushed");
                
                // Small delay to ensure data is processed
                System.Threading.Thread.Sleep(1);
                
                // Wait for acknowledgment with timeout
                Debug.Log("Waiting for acknowledgment...");
                stream.ReadTimeout = 5000; // 5 second timeout
                byte[] ackBuffer = new byte[3];
                int bytesRead = stream.Read(ackBuffer, 0, ackBuffer.Length);
                string ack = System.Text.Encoding.UTF8.GetString(ackBuffer, 0, bytesRead);
                Debug.Log($"Received acknowledgment: '{ack}' ({bytesRead} bytes)");
                
                if (ack == "ACK")
                {
                    Debug.Log($"Frame from '{camName}' sent successfully. Size: {imageData.Length} bytes, Objects: {groundTruthObjects.Count}");
                }
                else
                {
                    Debug.LogWarning($"Unexpected acknowledgment from server: {ack}");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Network error during transmission: {e.Message}");
                isConnected = false;
                // Try to reconnect
                ConnectToServer();
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error capturing/sending frame: {e.Message}\n{e.StackTrace}");
            isConnected = false;
        }
    }
    
    List<GroundTruthObject> DetectGroundTruthObjects(Camera cam)
    {
        List<GroundTruthObject> objects = new List<GroundTruthObject>();
        
        // Get all GameObjects in the camera's view
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        
        foreach (GameObject obj in allObjects)
        {
            if (obj.activeInHierarchy && IsObjectInView(cam, obj))
            {
                GroundTruthObject.ObjectType objectType = GetObjectType(obj);
                if (objectType != GroundTruthObject.ObjectType.Unknown)
                {
                    Bounds bounds = GetObjectBounds(obj);
                    Vector2 screenMin, screenMax;
                    
                    if (WorldBoundsToScreenBounds(cam, bounds, out screenMin, out screenMax))
                    {
                        objects.Add(new GroundTruthObject
                        {
                            Type = objectType,
                            Name = obj.name, // Include the object name
                            Position = new SerializableVector3(obj.transform.position), // Include the world position
                            BoundingBox = new BoundingBox
                            {
                                x = screenMin.x,
                                y = screenMin.y,
                                width = screenMax.x - screenMin.x,
                                height = screenMax.y - screenMin.y
                            },
                            Confidence = 1.0f,
                            ObjectId = obj.GetInstanceID().ToString()
                        });
                    }
                }
            }
        }
        
        return objects;
    }
    
    bool IsObjectInView(Camera cam, GameObject obj)
    {
        Vector3 screenPoint = cam.WorldToViewportPoint(obj.transform.position);
        return screenPoint.z > 0 && screenPoint.x > 0 && screenPoint.x < 1 && screenPoint.y > 0 && screenPoint.y < 1;
    }
    
    GroundTruthObject.ObjectType GetObjectType(GameObject obj)
    {
        string objName = obj.name.ToLower();
        string tag = obj.tag.ToLower();
        
        if (objName.Contains("car") || objName.Contains("vehicle") || tag.Contains("vehicle"))
            return GroundTruthObject.ObjectType.Vehicle;
        else if (objName.Contains("person") || objName.Contains("pedestrian") || tag.Contains("pedestrian"))
            return GroundTruthObject.ObjectType.Pedestrian;
        else if (objName.Contains("traffic") && objName.Contains("light"))
            return GroundTruthObject.ObjectType.TrafficLight;
        else if (objName.Contains("sign"))
            return GroundTruthObject.ObjectType.Sign;
        else if (objName.Contains("bike") || objName.Contains("bicycle"))
            return GroundTruthObject.ObjectType.Bicycle;
        
        return GroundTruthObject.ObjectType.Unknown;
    }
    
    Bounds GetObjectBounds(GameObject obj)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null)
            return renderer.bounds;
            
        Collider collider = obj.GetComponent<Collider>();
        if (collider != null)
            return collider.bounds;
            
        return new Bounds(obj.transform.position, Vector3.one);
    }
    
    bool WorldBoundsToScreenBounds(Camera cam, Bounds bounds, out Vector2 min, out Vector2 max)
    {
        Vector3[] corners = new Vector3[8];
        corners[0] = bounds.min;
        corners[1] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
        corners[2] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
        corners[3] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
        corners[4] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
        corners[5] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
        corners[6] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);
        corners[7] = bounds.max;
        
        min = new Vector2(float.MaxValue, float.MaxValue);
        max = new Vector2(float.MinValue, float.MinValue);
        
        bool anyInFront = false;
        
        foreach (Vector3 corner in corners)
        {
            Vector3 screenPoint = cam.WorldToScreenPoint(corner);
            if (screenPoint.z > 0) // In front of camera
            {
                anyInFront = true;
                min.x = Mathf.Min(min.x, screenPoint.x);
                min.y = Mathf.Min(min.y, captureHeight - screenPoint.y); // Flip Y coordinate
                max.x = Mathf.Max(max.x, screenPoint.x);
                max.y = Mathf.Max(max.y, captureHeight - screenPoint.y);
            }
        }
        
        if (!anyInFront)
            return false;
            
        // Clamp to screen bounds
        min.x = Mathf.Clamp(min.x, 0, captureWidth);
        min.y = Mathf.Clamp(min.y, 0, captureHeight);
        max.x = Mathf.Clamp(max.x, 0, captureWidth);
        max.y = Mathf.Clamp(max.y, 0, captureHeight);
        
        return (max.x - min.x) > 1 && (max.y - min.y) > 1; // Must have some area
    }
    
    void OnDisable()
    {
        // Cleanup
        if (stream != null)
        {
            try
            {
                stream.Close();
            }
            catch (Exception e)
            {
                Debug.LogError($"Error closing stream: {e.Message}");
            }
            stream = null;
        }
        
        if (client != null)
        {
            try
            {
                client.Close();
            }
            catch (Exception e)
            {
                Debug.LogError($"Error closing client: {e.Message}");
            }
            client = null;
        }
        
        if (renderTextures != null)
        {
            for (int i = 0; i < renderTextures.Length; i++)
            {
                if (renderTextures[i] != null)
                    Destroy(renderTextures[i]);
            }
        }
        
        if (textures != null)
        {
            for (int i = 0; i < textures.Length; i++)
            {
                if (textures[i] != null)
                    Destroy(textures[i]);
            }
        }
    }
    
    void OnDestroy()
    {
        OnDisable();
    }
}