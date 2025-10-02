using UnityEngine;
using System;
using System.Collections;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using System.IO;
using System.Collections.Generic;
using System.Net.Sockets; // Add this for TCP communication

public class Unity2Python : MonoBehaviour
{
    [Header("Assign your cameras in order: Camera1, Camera2, Camera3")]
    public Camera[] cameras;           
    public RenderTexture[] renderTextures; // each camera should have a corresponding RT
    
    [Header("TCP Settings")]
    public string tcpServerAddress = "127.0.0.1";
    public int tcpServerPort = 25002; // Match the port in the Python server
    public int maxFrameRate = 30; // Maximum frames per second to send
    
    [Header("Ground Truth Detection")]
    public bool enableGroundTruthDetection = true;
    public LayerMask vehicleLayerMask = -1; // All layers by default
    public LayerMask pedestrianLayerMask = -1;
    public LayerMask trafficLightLayerMask = -1;
    
    [Header("Advanced Settings")]
    public bool forceGarbageCollection = true;
    public int gcEveryNFrames = 50;
    public bool useCoroutineMode = true;
    public int jpegQuality = 60; // JPEG compression quality (0-100)

    [Header("Debug Options")]
    public bool enableDebugLogs = true;
    public bool enableVerboseLogging = true;  // Temporarily enable for debugging
    public bool enableGroundTruthVisualization = false;

    private TcpClient tcpClient;
    private NetworkStream networkStream;
    private bool isConnected = false;
    private float lastFrameSent = 0f;
    private int totalFramesSent = 0;
    private bool isCoroutineRunning = false;
    private bool shouldStop = false;
    private float frameInterval;
    
    // Ground truth detection
    private Dictionary<int, List<GroundTruthObject>> groundTruthData = new Dictionary<int, List<GroundTruthObject>>();

    void Start()
    {
        shouldStop = false;
        Application.runInBackground = true;
        frameInterval = 1.0f / maxFrameRate;
        
        // Add debug info about camera setup
        if (enableDebugLogs)
        {
            Debug.Log("Unity2Python starting with " + (cameras != null ? cameras.Length : 0) + " cameras and " + (renderTextures != null ? renderTextures.Length : 0) + " render textures");
            if (cameras != null)
            {
                for (int i = 0; i < cameras.Length; i++)
                {
                    Debug.Log("Camera " + i + ": " + (cameras[i] != null ? cameras[i].name : "NULL"));
                }
            }
        }
        
        InitializeTCPConnection();
        
        if (useCoroutineMode)
        {
            StartCoroutine(SendFramesCoroutine());
        }
    }

    void InitializeTCPConnection()
    {
        try
        {
            // Cleanup any existing connections first
            CleanupConnection();
            
            // Create new TCP client
            tcpClient = new TcpClient();
            tcpClient.Connect(tcpServerAddress, tcpServerPort);
            networkStream = tcpClient.GetStream();
            
            isConnected = true;
            totalFramesSent = 0;
            lastFrameSent = Time.time;
            
            if (enableDebugLogs)
                Debug.Log("✅ TCP Client connected to " + tcpServerAddress + ":" + tcpServerPort);
        }
        catch (Exception e)
        {
            if (enableDebugLogs)
                Debug.LogError("❌ Failed to connect to TCP server: " + e.Message);
            isConnected = false;
        }
    }

    void Update()
    {
        if (!isConnected)
        {
            InitializeTCPConnection();
            return;
        }

        // Non-coroutine mode fallback
        if (!useCoroutineMode)
        {
            if (Time.time - lastFrameSent >= frameInterval)
            {
                SendAllFrames();
            }
        }
    }
    
    IEnumerator SendFramesCoroutine()
    {
        isCoroutineRunning = true;
        float coroutineStartTime = Time.time;
        int framesAtStart = totalFramesSent;
        
        if (enableVerboseLogging)
            Debug.Log("🎬 Frame coroutine started at frame " + totalFramesSent);
        
        while (!shouldStop)
        {
            if (isConnected)
            {
                // Ensure we sample after the frame is fully rendered
                yield return new WaitForEndOfFrame();

                // Rate limiting
                if (Time.time - lastFrameSent < frameInterval)
                {
                    yield return null;
                    continue;
                }

                try
                {
                    SendAllFrames();
                    
                    // Log progress every 25 frames
                    if (enableVerboseLogging && (totalFramesSent - framesAtStart) % 25 == 0 && totalFramesSent > framesAtStart)
                    {
                        float elapsed = Time.time - coroutineStartTime;
                        Debug.Log($"📈 Coroutine progress: {totalFramesSent - framesAtStart} frames sent in {elapsed:F1}s (FPS: {(totalFramesSent - framesAtStart)/elapsed:F1})");
                    }
                    
                    // Force garbage collection periodically
                    if (forceGarbageCollection && totalFramesSent % gcEveryNFrames == 0)
                    {
                        System.GC.Collect();
                        if (enableVerboseLogging)
                            Debug.Log($"🗑️ Forced garbage collection at frame {totalFramesSent}");
                    }
                }
                catch (Exception e)
                {
                    if (enableDebugLogs)
                        Debug.LogError("❌ Coroutine error at frame " + totalFramesSent + ": " + e.Message + "\nStack: " + e.StackTrace);
                    
                    isConnected = false;
                    CleanupConnection();
                    break;
                }
            }
            else
            {
                if (enableDebugLogs)
                    Debug.Log("⚠️ Coroutine detected disconnection at frame " + totalFramesSent);
                InitializeTCPConnection();
                yield return new WaitForSeconds(1.0f);
            }
            
            yield return null; // Wait for next frame
        }
        
        float totalTime = Time.time - coroutineStartTime;
        int framesSent = totalFramesSent - framesAtStart;
        
        if (enableDebugLogs)
            Debug.Log("🏁 Frame coroutine ended: " + framesSent + " frames sent in " + totalTime.ToString("F1") + "s (avg: " + (framesSent/totalTime).ToString("F1") + " fps)");
        
        isCoroutineRunning = false;
    }
    
    void SendAllFrames()
    {
        if (!isConnected || cameras == null || renderTextures == null || networkStream == null)
        {
            if (enableDebugLogs)
                Debug.LogWarning("Cannot send frames - Connected: " + isConnected + ", Cameras: " + (cameras != null) + ", RenderTextures: " + (renderTextures != null) + ", NetworkStream: " + (networkStream != null));
            return;
        }

        int count = Mathf.Min(cameras.Length, renderTextures.Length);
        
        if (count == 0)
        {
            if (enableDebugLogs)
                Debug.LogWarning("No cameras or render textures assigned!");
            return;
        }
        
        if (enableVerboseLogging)
            Debug.Log("Attempting to send frames for " + count + " cameras");
        
        for (int i = 0; i < count; i++)
        {
            if (cameras[i] != null && renderTextures[i] != null)
            {
                // Use 1-based camera IDs to match external systems
                int cameraIdOneBased = i + 1;
                SendCameraFrame(cameras[i], renderTextures[i], cameraIdOneBased);
            }
            else
            {
                if (enableDebugLogs)
                    Debug.LogWarning("Camera " + i + " or RenderTexture " + i + " is null");
            }
        }
        
        lastFrameSent = Time.time;
        
        if (enableVerboseLogging && totalFramesSent % 25 == 0)
            Debug.Log("✓ Total frames sent: " + totalFramesSent);
    }

    void SendCameraFrame(Camera cam, RenderTexture rt, int camId)
    {
        if (!isConnected || networkStream == null || cam == null || rt == null) 
            return;

        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = cam.targetTexture;
        
        try
        {
            if (!rt.IsCreated())
                rt.Create();

            cam.targetTexture = rt;
            cam.Render();
            
            RenderTexture.active = rt;
            Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();

            byte[] imageBytes = tex.EncodeToJPG(jpegQuality);

            if (imageBytes != null && imageBytes.Length > 0)
            {
                // Detect ground truth objects if enabled
                List<GroundTruthObject> groundTruthObjects = new List<GroundTruthObject>();
                if (enableGroundTruthDetection)
                {
                    groundTruthObjects = DetectGroundTruthObjects(cam, rt);
                    groundTruthData[camId] = groundTruthObjects;
                }

                // Create frame data with the specified format but include ground truth data
                var frameData = new
                {
                    camera_id = "Camera" + camId,
                    image_data = Convert.ToBase64String(imageBytes),
                    ground_truth_objects = groundTruthObjects
                };

                // Serialize to JSON
                string jsonData = JsonUtility.ToJson(frameData);
                
                // Send via TCP
                try
                {
                    byte[] messageBytes = System.Text.Encoding.UTF8.GetBytes(jsonData);
                    byte[] sizeBytes = BitConverter.GetBytes(messageBytes.Length);
                    
                    // Send size first, then the actual data
                    networkStream.Write(sizeBytes, 0, sizeBytes.Length);
                    networkStream.Write(messageBytes, 0, messageBytes.Length);
                    networkStream.Flush();
                    
                    totalFramesSent++;
                    
                    if (enableDebugLogs && totalFramesSent % 100 == 0)
                    {
                        Debug.Log("📷 Sent " + totalFramesSent + " frames - Latest: Camera " + camId + " (" + imageBytes.Length + " bytes, " + groundTruthObjects.Count + " objects)");
                    }
                    
                    // Wait for acknowledgment
                    byte[] ackBuffer = new byte[3];
                    int bytesRead = networkStream.Read(ackBuffer, 0, ackBuffer.Length);
                    string ack = System.Text.Encoding.UTF8.GetString(ackBuffer, 0, bytesRead);
                    if (ack != "ACK")
                    {
                        if (enableDebugLogs)
                            Debug.LogWarning("⚠️ Unexpected acknowledgment: " + ack);
                    }
                }
                catch (Exception sendEx)
                {
                    if (enableDebugLogs)
                        Debug.LogError("❌ Failed to send frame for camera " + camId + ": " + sendEx.Message);
                    isConnected = false;
                    throw;
                }
            }

            DestroyImmediate(tex);
        }
        catch (Exception ex)
        {
            if (enableDebugLogs)
                Debug.LogError("❌ SendCameraFrame error: " + ex.Message);
            throw; // Re-throw to be caught by caller
        }
        finally
        {
            cam.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
        }
    }

    List<GroundTruthObject> DetectGroundTruthObjects(Camera cam, RenderTexture rt)
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
                    
                    if (WorldBoundsToScreenBounds(cam, bounds, out screenMin, out screenMax, rt.width, rt.height))
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
    
    bool WorldBoundsToScreenBounds(Camera cam, Bounds bounds, out Vector2 min, out Vector2 max, int screenWidth, int screenHeight)
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
                min.y = Mathf.Min(min.y, screenHeight - screenPoint.y); // Flip Y coordinate
                max.x = Mathf.Max(max.x, screenPoint.x);
                max.y = Mathf.Max(max.y, screenHeight - screenPoint.y);
            }
        }
        
        if (!anyInFront)
            return false;
            
        // Clamp to screen bounds
        min.x = Mathf.Clamp(min.x, 0, screenWidth);
        min.y = Mathf.Clamp(min.y, 0, screenHeight);
        max.x = Mathf.Clamp(max.x, 0, screenWidth);
        max.y = Mathf.Clamp(max.y, 0, screenHeight);
        
        return (max.x - min.x) > 1 && (max.y - min.y) > 1; // Must have some area
    }

    void OnApplicationQuit()
    {
        shouldStop = true;
        CleanupConnection();
    }

    void OnDisable()
    {
        shouldStop = true;
        CleanupConnection();
    }

    void OnDestroy()
    {
        shouldStop = true;
        CleanupConnection();
    }
    
    private void CleanupConnection()
    {
        isConnected = false;
        
        try
        {
            // Stop coroutine
            shouldStop = true;
            
            // Close TCP connection
            if (networkStream != null)
            {
                try { networkStream.Close(); } catch { }
                networkStream = null;
            }
            
            if (tcpClient != null)
            {
                try { tcpClient.Close(); } catch { }
                tcpClient = null;
            }
        }
        catch (Exception e)
        {
            if (enableDebugLogs)
                Debug.LogWarning($"Error during cleanup: {e.Message}");
        }

        if (enableDebugLogs)
            Debug.Log("🔌 Unity2Python TCP connection closed");
    }
    
    public List<GroundTruthObject> GetGroundTruthData(int cameraId)
    {
        if (groundTruthData.ContainsKey(cameraId))
            return groundTruthData[cameraId];
        return new List<GroundTruthObject>();
    }
}

[System.Serializable]
public class FrameData
{
    public int CameraId;
    public float Timestamp;
    public int FrameNumber;
    public string ImageDataBase64;
    public List<GroundTruthObject> GroundTruthObjects;
}

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

    public Vector3 ToVector3()
    {
        return new Vector3(x, y, z);
    }
}
