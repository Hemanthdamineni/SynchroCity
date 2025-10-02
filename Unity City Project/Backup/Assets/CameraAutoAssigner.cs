using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class CameraAutoAssigner : MonoBehaviour
{
    [Header("Auto-Assignment Settings")]
    public bool autoAssignOnStart = true;
    public bool createRenderTextures = true;
    public int renderTextureWidth = 1920;
    public int renderTextureHeight = 1080;
    public RenderTextureFormat renderTextureFormat = RenderTextureFormat.ARGB32;
    
    [Header("Debug")]
    public bool enableDebugLogs = true;
    
    private Unity2Python unity2Python;
    
    void Start()
    {
        if (autoAssignOnStart)
        {
            AssignCamerasToUnity2Python();
        }
    }
    
    [ContextMenu("Auto-Assign Cameras")]
    public void AssignCamerasToUnity2Python()
    {
        // Find Unity2Python component
        unity2Python = FindObjectOfType<Unity2Python>();
        if (unity2Python == null)
        {
            if (enableDebugLogs)
                Debug.LogError("❌ Unity2Python component not found in scene!");
            return;
        }
        
        // Find all cameras in the scene
        Camera[] allCameras = FindObjectsOfType<Camera>();
        List<Camera> validCameras = new List<Camera>();
        List<RenderTexture> renderTextures = new List<RenderTexture>();
        
        if (enableDebugLogs)
            Debug.Log($"🔍 Found {allCameras.Length} cameras in scene");
        
        foreach (Camera cam in allCameras)
        {
            // Skip cameras that are disabled or have specific exclusion criteria
            if (!cam.gameObject.activeInHierarchy)
            {
                if (enableDebugLogs)
                    Debug.Log($"⏭️ Skipping inactive camera: {cam.name}");
                continue;
            }
            
            // Skip UI cameras or other special cameras
            if (cam.name.ToLower().Contains("ui") || 
                cam.name.ToLower().Contains("menu") ||
                cam.cullingMask == 0)
            {
                if (enableDebugLogs)
                    Debug.Log($"⏭️ Skipping UI/special camera: {cam.name}");
                continue;
            }
            
            validCameras.Add(cam);
            
            // Create render texture for this camera
            RenderTexture rt = null;
            if (createRenderTextures)
            {
                rt = new RenderTexture(renderTextureWidth, renderTextureHeight, 24, renderTextureFormat);
                rt.name = $"RT_{cam.name}";
                rt.Create();
                
                if (enableDebugLogs)
                    Debug.Log($"📹 Created RenderTexture for camera: {cam.name} ({renderTextureWidth}x{renderTextureHeight})");
            }
            
            renderTextures.Add(rt);
            
            if (enableDebugLogs)
                Debug.Log($"✅ Added camera: {cam.name} (Position: {cam.transform.position})");
        }
        
        // Assign to Unity2Python
        if (validCameras.Count > 0)
        {
            unity2Python.cameras = validCameras.ToArray();
            unity2Python.renderTextures = renderTextures.ToArray();
            
            if (enableDebugLogs)
            {
                Debug.Log($"🎯 Successfully assigned {validCameras.Count} cameras to Unity2Python:");
                for (int i = 0; i < validCameras.Count; i++)
                {
                    Debug.Log($"   Camera {i + 1}: {validCameras[i].name}");
                }
            }
        }
        else
        {
            if (enableDebugLogs)
                Debug.LogWarning("⚠️ No valid cameras found to assign!");
        }
    }
    
    [ContextMenu("List All Cameras")]
    public void ListAllCameras()
    {
        Camera[] allCameras = FindObjectsOfType<Camera>();
        Debug.Log($"📋 All cameras in scene ({allCameras.Length} total):");
        
        for (int i = 0; i < allCameras.Length; i++)
        {
            Camera cam = allCameras[i];
            string status = cam.gameObject.activeInHierarchy ? "Active" : "Inactive";
            string parent = cam.transform.parent != null ? cam.transform.parent.name : "Root";
            
            Debug.Log($"   {i + 1}. {cam.name} ({status}) - Parent: {parent} - Position: {cam.transform.position}");
        }
    }
    
    [ContextMenu("Test Unity2Python Status")]
    public void TestUnity2PythonStatus()
    {
        unity2Python = FindObjectOfType<Unity2Python>();
        if (unity2Python == null)
        {
            Debug.LogError("❌ Unity2Python not found!");
            return;
        }
        
        Debug.Log($"📊 Unity2Python Status:");
        Debug.Log($"   Cameras assigned: {(unity2Python.cameras != null ? unity2Python.cameras.Length : 0)}");
        Debug.Log($"   RenderTextures assigned: {(unity2Python.renderTextures != null ? unity2Python.renderTextures.Length : 0)}");
        Debug.Log($"   Debug logs enabled: {unity2Python.enableDebugLogs}");
        Debug.Log($"   Verbose logging enabled: {unity2Python.enableVerboseLogging}");
        Debug.Log($"   Max frame rate: {unity2Python.maxFrameRate}");
        Debug.Log($"   Publisher address: {unity2Python.publisherAddress}");
        
        if (unity2Python.cameras != null)
        {
            for (int i = 0; i < unity2Python.cameras.Length; i++)
            {
                Camera cam = unity2Python.cameras[i];
                RenderTexture rt = unity2Python.renderTextures != null && i < unity2Python.renderTextures.Length ? 
                                  unity2Python.renderTextures[i] : null;
                
                Debug.Log($"   Camera {i + 1}: {(cam != null ? cam.name : "NULL")} | RT: {(rt != null ? rt.name : "NULL")}");
            }
        }
    }
}