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
    
    [Header("Ground Truth Settings")]
    public LayerMask objectLayers = ~0; // All layers by default
    public bool includeInactiveObjects = false;
    public float minObjectSize = 10f; // Minimum size in pixels to include
    
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
                cameraNames[i] = captureCameras[i].name;
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
        
        // TEST: Check scene objects
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        int renderableCount = 0;
        foreach (GameObject obj in allObjects)
        {
            if (obj.GetComponent<Renderer>() != null)
                renderableCount++;
        }
        Debug.Log($"=== STARTUP TEST: Found {allObjects.Length} total GameObjects, {renderableCount} with Renderers ===");
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
                client.Close();
                client = null;
            }
            
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
            
            if (!isConnected)
            {
                // Try to reconnect
                ConnectToServer();
            }
            
            if (isConnected && client != null && client.Connected && captureCameras.Length > 0)
            {
                // Cycle through cameras
                CaptureAndSendFrame(currentCameraIndex);
                currentCameraIndex = (currentCameraIndex + 1) % captureCameras.Length;
            }
        }
    }
    
    void CaptureAndSendFrame(int cameraIndex)
    {
        try
        {
            Camera cam = captureCameras[cameraIndex];
            string camName = cameraNames[cameraIndex];
            
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
            
            // Convert to base64
            string base64String = System.Convert.ToBase64String(imageData);
            
            // Collect ground truth for ALL objects with renderers in camera view
            List<string> gtList = new List<string>();
            GameObject[] allObjects = includeInactiveObjects ? 
                Resources.FindObjectsOfTypeAll<GameObject>() : 
                FindObjectsOfType<GameObject>();
            
            int totalChecked = 0;
            int inView = 0;
            int totalObjects = 0;
            int withRenderers = 0;
            int inCorrectLayer = 0;
            
            foreach (GameObject obj in allObjects)
            {
                // Skip if object is part of the editor scene or is inactive (unless includeInactiveObjects is true)
                if (obj.hideFlags != HideFlags.None || (!includeInactiveObjects && !obj.activeInHierarchy))
                    continue;
                
                // Check if object is in the specified layers
                if (((1 << obj.layer) & objectLayers) == 0)
                    continue;
                
                Renderer rend = obj.GetComponent<Renderer>();
                if (rend != null)
                {
                    totalChecked++;
                    
                    Bounds b = rend.bounds;
                    Vector3 center = b.center;
                    Vector3 screenPos = cam.WorldToViewportPoint(center);
                    
                    // Check if object is in front of camera and within viewport
                    if (screenPos.z > 0 && screenPos.x > 0 && screenPos.x < 1 && 
                        screenPos.y > 0 && screenPos.y < 1)
                    {
                        // Calculate screen space bounding box
                        Vector3[] boundingPoints = new Vector3[8];
                        boundingPoints[0] = b.min;
                        boundingPoints[1] = b.max;
                        boundingPoints[2] = new Vector3(b.min.x, b.min.y, b.max.z);
                        boundingPoints[3] = new Vector3(b.min.x, b.max.y, b.min.z);
                        boundingPoints[4] = new Vector3(b.max.x, b.min.y, b.min.z);
                        boundingPoints[5] = new Vector3(b.min.x, b.max.y, b.max.z);
                        boundingPoints[6] = new Vector3(b.max.x, b.min.y, b.max.z);
                        boundingPoints[7] = new Vector3(b.max.x, b.max.y, b.min.z);
                        
                        float minX = float.MaxValue;
                        float maxX = float.MinValue;
                        float minY = float.MaxValue;
                        float maxY = float.MinValue;
                        
                        bool anyPointInFront = false;
                        foreach (Vector3 point in boundingPoints)
                        {
                            Vector3 screenPoint = cam.WorldToScreenPoint(point);
                            if (screenPoint.z > 0)
                            {
                                anyPointInFront = true;
                                minX = Mathf.Min(minX, screenPoint.x);
                                maxX = Mathf.Max(maxX, screenPoint.x);
                                minY = Mathf.Min(minY, screenPoint.y);
                                maxY = Mathf.Max(maxY, screenPoint.y);
                            }
                        }
                        
                        if (anyPointInFront)
                        {
                            // Clamp to screen bounds
                            minX = Mathf.Clamp(minX, 0, captureWidth);
                            maxX = Mathf.Clamp(maxX, 0, captureWidth);
                            minY = Mathf.Clamp(minY, 0, captureHeight);
                            maxY = Mathf.Clamp(maxY, 0, captureHeight);
                            
                            float width = maxX - minX;
                            float height = maxY - minY;
                            
                            // Only include if object is large enough
                            if (width >= minObjectSize && height >= minObjectSize)
                            {
                                string label = obj.name;
                                string tag = "Untagged";
                                try
                                {
                                    tag = obj.tag;
                                }
                                catch
                                {
                                    // Tag not defined, use default
                                }
                                
                                // Normalize coordinates to 0-1 range (YOLO format)
                                float centerX = (minX + maxX) / 2f / captureWidth;
                                float centerY = (minY + maxY) / 2f / captureHeight;
                                float normWidth = width / captureWidth;
                                float normHeight = height / captureHeight;
                                
                                inView++;
                                gtList.Add($"[\"{label}\",\"{tag}\",{centerX:F6},{centerY:F6},{normWidth:F6},{normHeight:F6}]");
                            }
                        }
                    }
                }
            }
            
            // Merge ground truth into JSON
            string groundTruthJson = string.Join(",", gtList);
            string message = $"{{\"camera_id\":\"{camName}\",\"image_data\":\"{base64String}\",\"ground_truth\":[{groundTruthJson}]}}";
            
            // Send data size first (4 bytes)
            byte[] dataSizeBytes = System.BitConverter.GetBytes(message.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(dataSizeBytes);
                
            stream.Write(dataSizeBytes, 0, dataSizeBytes.Length);
            
            // Send message data
            byte[] dataBytes = System.Text.Encoding.UTF8.GetBytes(message);
            stream.Write(dataBytes, 0, dataBytes.Length);
            
            // Wait for acknowledgment with timeout
            stream.ReadTimeout = 5000; // 5 second timeout
            byte[] ackBuffer = new byte[3];
            int bytesRead = stream.Read(ackBuffer, 0, ackBuffer.Length);
            string ack = System.Text.Encoding.UTF8.GetString(ackBuffer, 0, bytesRead);
            
            if (ack == "ACK")
            {
                Debug.Log($"Camera '{camName}': Sent {imageData.Length} bytes. Checked {totalChecked} objects, found {inView} in view.");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error capturing/sending frame: {e.Message}");
            isConnected = false;
        }
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