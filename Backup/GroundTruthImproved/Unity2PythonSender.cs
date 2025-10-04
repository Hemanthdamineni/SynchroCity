using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.IO;
using System;
using System.Linq;

public class ImageSender : MonoBehaviour
{
    [Header("Network Settings")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 25002;
    
    [Header("Camera Settings")]
    public Camera[] captureCameras;
    public string[] cameraNames;
    public int captureWidth = 640;
    public int captureHeight = 480;
    public int captureRate = 30;
    
    [Header("Image Settings")]
    public bool encodeToJPG = true;
    [Range(10, 100)]
    public int jpgQuality = 75;
    
    [Header("Vehicle Detection Settings")]
    public string[] vehiclePartKeywords = new string[] { "Body", "Wheel", "Door", "Hood", "Trunk" };
    public string[] excludeKeywords = new string[] { "Segment", "Road", "Lane", "Ground", "Terrain", "Sky" };
    public float minVehicleSize = 30f; // Minimum bounding box size in pixels
    public int minPartsForVehicle = 2; // Minimum parts to consider it a vehicle
    
    private TcpClient client;
    private NetworkStream stream;
    private RenderTexture[] renderTextures;
    private Texture2D[] textures;
    private bool isConnected = false;
    private float lastCaptureTime = 0f;
    private int currentCameraIndex = 0;
    private bool isConnecting = false;
    private object connectionLock = new object();
    
    void Start()
    {
        if (captureCameras == null || captureCameras.Length == 0)
        {
            captureCameras = FindObjectsOfType<Camera>();
            Debug.Log($"Auto-assigned {captureCameras.Length} cameras");
        }
        
        if (cameraNames == null || cameraNames.Length != captureCameras.Length)
        {
            cameraNames = new string[captureCameras.Length];
            for (int i = 0; i < captureCameras.Length; i++)
            {
                cameraNames[i] = captureCameras[i].name;
            }
        }
        
        renderTextures = new RenderTexture[captureCameras.Length];
        textures = new Texture2D[captureCameras.Length];
        
        for (int i = 0; i < captureCameras.Length; i++)
        {
            renderTextures[i] = new RenderTexture(captureWidth, captureHeight, 24);
            textures[i] = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        }
        
        ConnectToServer();
        Debug.Log($"Image Sender initialized. Auto-detecting vehicles by parts: {string.Join(", ", vehiclePartKeywords)}");
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
        if (Time.time - lastCaptureTime >= 1f / captureRate)
        {
            lastCaptureTime = Time.time;
            
            if (!isConnected)
            {
                ConnectToServer();
            }
            
            if (isConnected && client != null && client.Connected && captureCameras.Length > 0)
            {
                CaptureAndSendFrame(currentCameraIndex);
                currentCameraIndex = (currentCameraIndex + 1) % captureCameras.Length;
            }
        }
    }
    
    bool IsVehiclePart(GameObject obj)
    {
        // Check if excluded
        foreach (string keyword in excludeKeywords)
        {
            if (obj.name.Contains(keyword))
                return false;
        }
        
        // Check if it's a vehicle part
        foreach (string keyword in vehiclePartKeywords)
        {
            if (obj.name.Contains(keyword))
                return true;
        }
        
        return false;
    }
    
    Transform FindVehicleRoot(GameObject obj)
    {
        // Find the common parent of all vehicle parts
        Transform current = obj.transform;
        Transform vehicleRoot = current;
        
        // Go up until we find the root vehicle object
        while (current.parent != null)
        {
            // If parent has vehicle-related parts, it's likely the vehicle root
            bool hasVehicleParts = false;
            foreach (Transform child in current.parent)
            {
                if (IsVehiclePart(child.gameObject))
                {
                    hasVehicleParts = true;
                    break;
                }
            }
            
            if (hasVehicleParts)
            {
                vehicleRoot = current.parent;
                current = current.parent;
            }
            else
            {
                break;
            }
        }
        
        return vehicleRoot;
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
            
            // Encode image
            byte[] imageData;
            if (encodeToJPG)
            {
                imageData = textures[cameraIndex].EncodeToJPG(jpgQuality);
            }
            else
            {
                imageData = textures[cameraIndex].EncodeToPNG();
            }
            
            string base64String = System.Convert.ToBase64String(imageData);
            
            // Find all vehicle parts and group them by parent
            Dictionary<Transform, List<GameObject>> vehicleGroups = new Dictionary<Transform, List<GameObject>>();
            GameObject[] allObjects = FindObjectsOfType<GameObject>();
            
            foreach (GameObject obj in allObjects)
            {
                if (!obj.activeInHierarchy || obj.hideFlags != HideFlags.None)
                    continue;
                
                if (IsVehiclePart(obj))
                {
                    Transform root = FindVehicleRoot(obj);
                    
                    if (!vehicleGroups.ContainsKey(root))
                    {
                        vehicleGroups[root] = new List<GameObject>();
                    }
                    vehicleGroups[root].Add(obj);
                }
            }
            
            // Create bounding boxes for each vehicle
            List<string> gtList = new List<string>();
            int vehiclesChecked = 0;
            int vehiclesInView = 0;
            
            foreach (var kvp in vehicleGroups)
            {
                Transform vehicleRoot = kvp.Key;
                List<GameObject> parts = kvp.Value;
                
                // Need minimum parts to be considered a vehicle
                if (parts.Count < minPartsForVehicle)
                    continue;
                
                vehiclesChecked++;
                
                // Calculate combined bounding box for all parts
                Bounds? combinedBounds = null;
                bool anyPartInView = false;
                
                foreach (GameObject part in parts)
                {
                    Renderer rend = part.GetComponent<Renderer>();
                    if (rend != null)
                    {
                        Bounds b = rend.bounds;
                        Vector3 center = b.center;
                        Vector3 screenPos = cam.WorldToViewportPoint(center);
                        
                        if (screenPos.z > 0)
                        {
                            if (!combinedBounds.HasValue)
                            {
                                combinedBounds = b;
                            }
                            else
                            {
                                combinedBounds.Value.Encapsulate(b);
                            }
                            
                            if (screenPos.x > 0 && screenPos.x < 1 && screenPos.y > 0 && screenPos.y < 1)
                            {
                                anyPartInView = true;
                            }
                        }
                    }
                }
                
                if (!combinedBounds.HasValue || !anyPartInView)
                    continue;
                
                // Calculate screen space bounding box for the entire vehicle
                Bounds vBounds = combinedBounds.Value;
                Vector3[] boundingPoints = new Vector3[8];
                boundingPoints[0] = vBounds.min;
                boundingPoints[1] = vBounds.max;
                boundingPoints[2] = new Vector3(vBounds.min.x, vBounds.min.y, vBounds.max.z);
                boundingPoints[3] = new Vector3(vBounds.min.x, vBounds.max.y, vBounds.min.z);
                boundingPoints[4] = new Vector3(vBounds.max.x, vBounds.min.y, vBounds.min.z);
                boundingPoints[5] = new Vector3(vBounds.min.x, vBounds.max.y, vBounds.max.z);
                boundingPoints[6] = new Vector3(vBounds.max.x, vBounds.min.y, vBounds.max.z);
                boundingPoints[7] = new Vector3(vBounds.max.x, vBounds.max.y, vBounds.min.z);
                
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
                
                if (!anyPointInFront)
                    continue;
                
                // Clamp to screen bounds
                minX = Mathf.Clamp(minX, 0, captureWidth);
                maxX = Mathf.Clamp(maxX, 0, captureWidth);
                minY = Mathf.Clamp(minY, 0, captureHeight);
                maxY = Mathf.Clamp(maxY, 0, captureHeight);
                
                float width = maxX - minX;
                float height = maxY - minY;
                
                // Check minimum size
                if (width < minVehicleSize || height < minVehicleSize)
                    continue;
                
                // Normalize coordinates (YOLO format)
                float centerX = (minX + maxX) / 2f / captureWidth;
                float centerY = (minY + maxY) / 2f / captureHeight;
                float normWidth = width / captureWidth;
                float normHeight = height / captureHeight;
                
                string vehicleName = vehicleRoot.name;
                string classLabel = "vehicle"; // You can customize this based on vehicle type
                
                vehiclesInView++;
                gtList.Add($"[\"{classLabel}\",\"{vehicleName}\",{centerX:F6},{centerY:F6},{normWidth:F6},{normHeight:F6}]");
            }
            
            // Create JSON message
            string groundTruthJson = string.Join(",", gtList);
            string message = $"{{\"camera_id\":\"{camName}\",\"image_data\":\"{base64String}\",\"ground_truth\":[{groundTruthJson}]}}";
            
            // Send data
            byte[] dataSizeBytes = System.BitConverter.GetBytes(message.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(dataSizeBytes);
                
            stream.Write(dataSizeBytes, 0, dataSizeBytes.Length);
            
            byte[] dataBytes = System.Text.Encoding.UTF8.GetBytes(message);
            stream.Write(dataBytes, 0, dataBytes.Length);
            
            // Wait for acknowledgment
            stream.ReadTimeout = 5000;
            byte[] ackBuffer = new byte[3];
            int bytesRead = stream.Read(ackBuffer, 0, ackBuffer.Length);
            string ack = System.Text.Encoding.UTF8.GetString(ackBuffer, 0, bytesRead);
            
            if (ack == "ACK")
            {
                if (vehiclesInView > 0)
                {
                    Debug.Log($"<color=green>Camera '{camName}': {vehiclesInView} vehicles detected ({vehiclesChecked} total checked)</color>");
                }
                else
                {
                    Debug.Log($"Camera '{camName}': {vehiclesChecked} vehicles checked, 0 in view");
                }
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
        if (stream != null)
        {
            try { stream.Close(); }
            catch (Exception e) { Debug.LogError($"Error closing stream: {e.Message}"); }
            stream = null;
        }
        
        if (client != null)
        {
            try { client.Close(); }
            catch (Exception e) { Debug.LogError($"Error closing client: {e.Message}"); }
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