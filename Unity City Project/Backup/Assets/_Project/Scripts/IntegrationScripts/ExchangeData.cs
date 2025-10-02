﻿﻿﻿using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;
using UnityEngine;
using System;

[System.Serializable]
public class CommonMessage
{
    public string type;    // "command" or "vehicles"
    public string command; // Used if type == "command"
}

public static class RecordingManager
{
    public static bool startRecordingFromZero = false;
    public static float recordingStartTime = 0f;
}

public class ExchangeData : MonoBehaviour
{
    private SimulationController _SimulationController;

    // Thread for background communication
    private Thread _communicationThread;
    private bool _isRunning = false;


    public void Start()
    {
        _SimulationController = GetComponent<SimulationController>();

        // Start the communication thread
        _isRunning = true;
        _communicationThread = new Thread(Run);
        _communicationThread.Start();
    }

    void OnDestroy()
    {
        // Stop the communication thread
        _isRunning = false;
        if (_communicationThread != null && _communicationThread.IsAlive)
        {
            try
            {
                if (!_communicationThread.Join(1000)) // Wait max 1 second
                {
                    _communicationThread.Abort();
                }
            }
            catch { }
        }

        try
        {
            NetMQConfig.Cleanup();
        }
        catch { }
        
        Debug.Log("ExchangeData component destroyed and cleaned up.");
    }

    private void Run()
    {
        ForceDotNet.Force();

        try
        {
            using (var subSocket = new SubscriberSocket())
            using (var dealerSocket = new DealerSocket())
            {
                // Configure socket options
                subSocket.Options.ReceiveHighWatermark = 1000;
                subSocket.Options.Linger = TimeSpan.FromMilliseconds(100);
                dealerSocket.Options.SendHighWatermark = 1000;
                dealerSocket.Options.Linger = TimeSpan.FromMilliseconds(100);
                
                try
                {
                    // Connect to SUMO's PUB socket
                    subSocket.Connect("tcp://localhost:5556");
                    subSocket.Subscribe("");

                    // Connect to SUMO's ROUTER socket
                    dealerSocket.Connect("tcp://localhost:5557");
                    
                    Debug.Log("Connected to SUMO sockets successfully");
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Failed to connect to SUMO: {ex.Message}");
                    return;
                }

                while (_isRunning)
                {
                    try
                    {
                        // Check if SUMO connection is still alive
                        if (!_isRunning) break;
                        
                        // --- Send Data to SUMO ---
                        if (_SimulationController != null)
                        {
                            string vehicleDataJson = _SimulationController.GetVehicleDataJson();

                            if (!string.IsNullOrEmpty(vehicleDataJson))
                            {
                                try
                                {
                                    dealerSocket.SendFrame(vehicleDataJson);
                                }
                                catch (Exception sendEx)
                                {
                                    Debug.LogWarning($"Failed to send data to SUMO: {sendEx.Message}");
                                }
                            }
                        }

                        // --- Receive Data from SUMO ---
                        string sumoDataJson = null;
                        try
                        {
                            bool gotMessage = subSocket.TryReceiveFrameString(TimeSpan.FromMilliseconds(50), out sumoDataJson);
                            
                            if (gotMessage && !string.IsNullOrEmpty(sumoDataJson))
                            {
                                // Enqueue the message to be handled on the main thread
                                if (_SimulationController != null)
                                {
                                    _SimulationController.EnqueueOnMainThread(sumoDataJson);
                                }
                            }
                        }
                        catch (Exception receiveEx)
                        {
                            // Timeout or other receive issues are normal, just continue
                        }
                    }
                    catch (NetMQ.TerminatingException)
                    {
                        Debug.Log("NetMQ context terminating, stopping SUMO communication");
                        break;
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"Exception in SUMO communication loop: {ex.Message}");
                        // Don't break the loop for minor exceptions
                    }

                    // Sleep briefly to prevent 100% CPU usage
                    Thread.Sleep(10);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Exception in SUMO background thread: {ex.Message}\n{ex.StackTrace}");
        }
        finally
        {
            try
            {
                NetMQConfig.Cleanup();
            }
            catch { }
            Debug.Log("ExchangeData thread terminated gracefully.");
        }
    }
}
