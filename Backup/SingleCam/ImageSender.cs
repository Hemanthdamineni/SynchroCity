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
    public Camera captureCamera;
    public int captureWidth = 640;
    public int captureHeight = 480;
    public int captureRate = 30; // FPS
    
    [Header("Image Settings")]
    public bool encodeToJPG = true;
    [Range(10, 100)]
    public int jpgQuality = 75;
    
    private TcpClient client;
    private NetworkStream stream;
    private RenderTexture renderTexture;
    private Texture2D texture;
    private bool isConnected = false;
    private Thread connectionThread;
    private float lastCaptureTime = 0f;
    
    void Start()
    {
        // Initialize components if not assigned
        if (captureCamera == null)
            captureCamera = GetComponent<Camera>();
        
        // Create render texture for capturing
        renderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        texture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        
        // Start connection in separate thread
        connectionThread = new Thread(ConnectToServer);
        connectionThread.IsBackground = true;
        connectionThread.Start();
        
        Debug.Log($"Image Sender initialized. Capturing at {captureWidth}x{captureHeight} @{captureRate}fps");
    }
    
    void ConnectToServer()
    {
        try
        {
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
    }
    
    void Update()
    {
        // Check if it's time to capture a frame
        if (Time.time - lastCaptureTime >= 1f / captureRate)
        {
            lastCaptureTime = Time.time;
            
            if (isConnected && client != null && client.Connected)
            {
                CaptureAndSendFrame();
            }
            else if (!isConnected)
            {
                // Try to reconnect periodically
                if (connectionThread == null || !connectionThread.IsAlive)
                {
                    connectionThread = new Thread(ConnectToServer);
                    connectionThread.IsBackground = true;
                    connectionThread.Start();
                }
            }
        }
    }
    
    void CaptureAndSendFrame()
    {
        try
        {
            // Capture camera view
            captureCamera.targetTexture = renderTexture;
            captureCamera.Render();
            RenderTexture.active = renderTexture;
            texture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            texture.Apply();
            captureCamera.targetTexture = null;
            RenderTexture.active = null;
            
            // Encode to JPG or PNG
            byte[] imageData;
            if (encodeToJPG)
            {
                imageData = texture.EncodeToJPG(jpgQuality);
            }
            else
            {
                imageData = texture.EncodeToPNG();
            }
            
            // Convert to base64
            string base64String = System.Convert.ToBase64String(imageData);
            
            // Send data size first (4 bytes)
            byte[] dataSizeBytes = System.BitConverter.GetBytes(base64String.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(dataSizeBytes);
                
            stream.Write(dataSizeBytes, 0, dataSizeBytes.Length);
            
            // Send base64 data
            byte[] dataBytes = System.Text.Encoding.UTF8.GetBytes(base64String);
            stream.Write(dataBytes, 0, dataBytes.Length);
            
            // Wait for acknowledgment
            byte[] ackBuffer = new byte[3];
            int bytesRead = stream.Read(ackBuffer, 0, ackBuffer.Length);
            string ack = System.Text.Encoding.UTF8.GetString(ackBuffer, 0, bytesRead);
            
            if (ack == "ACK")
            {
                Debug.Log($"Frame sent successfully. Size: {imageData.Length} bytes");
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
            stream.Close();
            stream = null;
        }
        
        if (client != null)
        {
            client.Close();
            client = null;
        }
        
        if (renderTexture != null)
        {
            Destroy(renderTexture);
        }
        
        if (texture != null)
        {
            Destroy(texture);
        }
    }
    
    void OnDestroy()
    {
        OnDisable();
    }
}