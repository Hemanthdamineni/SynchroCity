using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections;
using System;

/// <summary>
/// Automatically sets up Unity2Python with cameras and provides TCP fallback for frame transmission
/// This script combines auto-camera assignment with dual communication protocols:
/// - Primary: ZeroMQ (PUB-SUB) via existing Unity2Python script (user preference)
/// - Fallback: TCP sockets for reliable string-based base64 image transfer
/// </summary>
public class AutoSetupUnity2Python : MonoBehaviour
{
    [Header("Auto Setup Settings")]
    public bool runOnStart = true;
    public bool createRenderTextures = true;
    public int renderTextureWidth = 640;
    public int renderTextureHeight = 640;
    
    [Header("TCP Image Transfer Settings (Fallback)")]
    public bool useTcpFallback = false; // Default to ZeroMQ as per user preference
    public int tcpPort = 25002;
    public float frameRate = 15f; // Frames per second
    public int jpegQuality = 60;
    
    [Header("Debug")]
    public bool enableDebugLogs = true;
    
    // TCP Server
    private TcpListener tcpServer;
    private TcpClient tcpClient;
    private NetworkStream networkStream;
    private Thread tcpThread;
    private bool tcpRunning = false;
    
    // Frame capture
    private Camera[] assignedCameras;
    private RenderTexture[] assignedRenderTextures;
    private bool isCapturing = false;
    private float lastFrameTime = 0f;
    
    void Start()
    {
        if (runOnStart)
        {
            SetupUnity2PythonAutomatically();
        }
        
        if (useTcpFallback)
        {
            StartTcpImageServer();
        }
    }
    
    void OnDestroy()
    {
        StopTcpServer();
    }
    
    void OnApplicationQuit()
    {
        StopTcpServer();
    }
    
    void SetupUnity2PythonAutomatically()
    {
        Debug.Log("🚀 AutoSetupUnity2Python: Starting automatic setup...");
        
        // Find Unity2Python component
        Unity2Python unity2Python = FindObjectOfType<Unity2Python>();
        if (unity2Python == null)
        {
            Debug.LogError("❌ Unity2Python component not found! Please add Unity2Python script to a GameObject in the scene.");
            return;
        }
        
        // Check if cameras are already assigned
        if (unity2Python.cameras != null && unity2Python.cameras.Length > 0)
        {
            Debug.Log("✅ Unity2Python already has cameras assigned. Skipping auto-setup.");
            return;
        }
        
        // Find all active cameras
        Camera[] allCameras = FindObjectsOfType<Camera>();
        System.Collections.Generic.List<Camera> validCameras = new System.Collections.Generic.List<Camera>();
        System.Collections.Generic.List<RenderTexture> renderTextures = new System.Collections.Generic.List<RenderTexture>();
        
        Debug.Log("🔍 Found " + allCameras.Length + " total cameras in scene");
        
        foreach (Camera cam in allCameras)
        {
            // Only use active cameras that are likely to be main scene cameras
            if (!cam.gameObject.activeInHierarchy)
                continue;
                
            // Skip UI cameras and other special cameras
            if (cam.name.ToLower().Contains("ui") || 
                cam.name.ToLower().Contains("menu") ||
                cam.cullingMask == 0)
                continue;
            
            validCameras.Add(cam);
            
            // Create render texture
            if (createRenderTextures)
            {
                RenderTexture rt = new RenderTexture(renderTextureWidth, renderTextureHeight, 24, RenderTextureFormat.ARGB32);
                rt.name = "AutoRT_" + cam.name;
                rt.Create();
                renderTextures.Add(rt);
                
                Debug.Log("📹 Created RenderTexture for camera: " + cam.name);
            }
            else
            {
                renderTextures.Add(null);
            }
            
            Debug.Log("✅ Added camera: " + cam.name + " at position " + cam.transform.position);
        }
        
        // Assign to Unity2Python
        if (validCameras.Count > 0)
        {
            unity2Python.cameras = validCameras.ToArray();
            unity2Python.renderTextures = renderTextures.ToArray();
            
            // Enable verbose logging temporarily to see what's happening
            unity2Python.enableDebugLogs = true;
            unity2Python.enableVerboseLogging = true;
            
            Debug.Log("🎯 Successfully auto-assigned " + validCameras.Count + " cameras to Unity2Python!");
            
            // Store reference for TCP transmission
            assignedCameras = validCameras.ToArray();
            assignedRenderTextures = renderTextures.ToArray();
            
            Debug.Log("📡 Cameras ready for both NetMQ and TCP transmission");
            
            // List assigned cameras
            for (int i = 0; i < validCameras.Count; i++)
            {
                Debug.Log("   Camera " + (i + 1) + ": " + validCameras[i].name);
            }
        }
        else
        {
            Debug.LogWarning("⚠️ No valid cameras found for Unity2Python!");
            Debug.Log("💡 Make sure you have active cameras in your scene that are not UI cameras.");
        }
    }
    
    // Provide a manual trigger via context menu
    [ContextMenu("Setup Unity2Python")]
    void ManualSetup()
    {
        SetupUnity2PythonAutomatically();
    }
    
    void Update()
    {
        // Handle TCP frame transmission
        if (useTcpFallback && tcpRunning && isCapturing && 
            assignedCameras != null && assignedCameras.Length > 0)
        {
            if (Time.time - lastFrameTime >= (1f / frameRate))
            {
                StartCoroutine(CaptureAndSendFrame());
                lastFrameTime = Time.time;
            }
        }
    }
    
    void StartTcpImageServer()
    {
        try
        {
            tcpServer = new TcpListener(IPAddress.Any, tcpPort);
            tcpServer.Start();
            tcpRunning = true;
            
            if (enableDebugLogs)
                Debug.Log("🌐 TCP Image Server started on port " + tcpPort);
            
            // Start listening for connections in a separate thread
            tcpThread = new Thread(ListenForTcpClients);
            tcpThread.IsBackground = true;
            tcpThread.Start();
        }
        catch (Exception e)
        {
            if (enableDebugLogs)
                Debug.LogError("❌ Failed to start TCP server: " + e.Message);
        }
    }
    
    void ListenForTcpClients()
    {
        while (tcpRunning)
        {
            try
            {
                if (enableDebugLogs)
                    Debug.Log("📡 Waiting for Python client connection...");
                
                tcpClient = tcpServer.AcceptTcpClient();
                networkStream = tcpClient.GetStream();
                
                if (enableDebugLogs)
                    Debug.Log("✅ Python client connected!");
                
                isCapturing = true;
                
                // Keep connection alive and handle disconnection
                while (tcpRunning && tcpClient.Connected)
                {
                    Thread.Sleep(100);
                }
                
                if (enableDebugLogs)
                    Debug.Log("🔌 Python client disconnected");
                
                isCapturing = false;
                
            }
            catch (Exception e)
            {
                if (tcpRunning && enableDebugLogs)
                    Debug.LogWarning("⚠️ TCP connection error: " + e.Message);
            }
        }
    }
    
    IEnumerator CaptureAndSendFrame()
    {
        if (assignedCameras == null || assignedRenderTextures == null || 
            !isCapturing || networkStream == null || !tcpClient.Connected)
            yield break;
        
        for (int i = 0; i < assignedCameras.Length; i++)
        {
            Camera cam = assignedCameras[i];
            RenderTexture rt = assignedRenderTextures[i];
            
            if (cam == null || rt == null) continue;
            
            yield return StartCoroutine(CaptureAndSendCameraFrame(cam, rt, i + 1));
        }
    }
    
    IEnumerator CaptureAndSendCameraFrame(Camera cam, RenderTexture rt, int cameraId)
    {
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture previousTarget = cam.targetTexture;
        
        try
        {
            // Render camera to texture
            if (!rt.IsCreated())
                rt.Create();
                
            cam.targetTexture = rt;
            cam.Render();
            
            RenderTexture.active = rt;
            Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            
            // Convert to base64
            byte[] imageBytes = tex.EncodeToJPG(jpegQuality);
            string base64Image = Convert.ToBase64String(imageBytes);
            
            // Create frame data similar to the temp string format
            string frameData = "CAMERA_" + cameraId + "|" + Time.time + "|" + base64Image;
            
            // Send via TCP
            if (networkStream != null && tcpClient.Connected)
            {
                byte[] data = Encoding.UTF8.GetBytes(frameData + "\n"); // Add newline delimiter
                networkStream.Write(data, 0, data.Length);
                
                if (enableDebugLogs && cameraId == 1) // Only log for camera 1 to avoid spam
                {
                    Debug.Log("📸 Sent frame via TCP: Camera " + cameraId + ", Size: " + imageBytes.Length + " bytes");
                }
            }
            
            DestroyImmediate(tex);
        }
        catch (Exception e)
        {
            if (enableDebugLogs)
                Debug.LogError("❌ Error capturing/sending frame for camera " + cameraId + ": " + e.Message);
        }
        finally
        {
            cam.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
        }
        
        yield return null; // Required for IEnumerator
    }
    
    void StopTcpServer()
    {
        tcpRunning = false;
        isCapturing = false;
        
        try
        {
            if (networkStream != null)
            {
                networkStream.Close();
                networkStream = null;
            }
            
            if (tcpClient != null)
            {
                tcpClient.Close();
                tcpClient = null;
            }
            
            if (tcpServer != null)
            {
                tcpServer.Stop();
                tcpServer = null;
            }
            
            if (tcpThread != null)
            {
                tcpThread.Join(1000); // Wait max 1 second
                tcpThread = null;
            }
            
            if (enableDebugLogs)
                Debug.Log("🔌 TCP Image Server stopped");
        }
        catch (Exception e)
        {
            if (enableDebugLogs)
                Debug.LogWarning("⚠️ Error stopping TCP server: " + e.Message);
        }
    }
}   