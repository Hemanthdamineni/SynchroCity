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
            
            // Create a JSON-like message with camera info
            string message = $"{{\"camera_id\":\"{camName}\",\"image_data\":\"{base64String}\"}}";
            
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
                Debug.Log($"Frame from '{camName}' sent successfully. Size: {imageData.Length} bytes");
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