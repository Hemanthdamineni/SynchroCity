using UnityEngine;

public class DiagnosticScript : MonoBehaviour
{
    public Camera testCamera;
    
    void Start()
    {
        if (testCamera == null)
            testCamera = Camera.main;
            
        Debug.Log("=== DIAGNOSTIC START ===");
        
        // Find all objects
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        Debug.Log($"Total GameObjects in scene: {allObjects.Length}");
        
        int withRenderer = 0;
        int withMeshRenderer = 0;
        int withSkinnedMeshRenderer = 0;
        int active = 0;
        
        foreach (GameObject obj in allObjects)
        {
            if (obj.activeInHierarchy)
                active++;
                
            Renderer rend = obj.GetComponent<Renderer>();
            if (rend != null)
            {
                withRenderer++;
                
                if (rend is MeshRenderer)
                {
                    withMeshRenderer++;
                    Debug.Log($"Found MeshRenderer on: {obj.name} (Layer: {LayerMask.LayerToName(obj.layer)}, Active: {obj.activeInHierarchy})");
                }
                
                if (rend is SkinnedMeshRenderer)
                {
                    withSkinnedMeshRenderer++;
                    Debug.Log($"Found SkinnedMeshRenderer on: {obj.name} (Layer: {LayerMask.LayerToName(obj.layer)}, Active: {obj.activeInHierarchy})");
                }
                
                // Check if in camera view
                if (testCamera != null)
                {
                    Bounds b = rend.bounds;
                    Vector3 screenPos = testCamera.WorldToViewportPoint(b.center);
                    bool inView = screenPos.z > 0 && screenPos.x > 0 && screenPos.x < 1 && screenPos.y > 0 && screenPos.y < 1;
                    
                    if (inView)
                    {
                        Debug.Log($"  -> IN CAMERA VIEW: {obj.name} at viewport ({screenPos.x:F2}, {screenPos.y:F2})");
                    }
                }
            }
        }
        
        Debug.Log($"Active GameObjects: {active}");
        Debug.Log($"Objects with Renderer: {withRenderer}");
        Debug.Log($"Objects with MeshRenderer: {withMeshRenderer}");
        Debug.Log($"Objects with SkinnedMeshRenderer: {withSkinnedMeshRenderer}");
        Debug.Log("=== DIAGNOSTIC END ===");
    }
}