// Global state
let currentGridSize = '2x2';
let selectedCameras = [];
let isSimulationRunning = false;
let leftPanelCollapsed = false;
let updateInterval;
let unityStreamingEnabled = true; // Always use streaming mode now
let streamingPort = 5000;
let availableUnityCameras = [];
let frameUpdateInterval;
let cameraNames = []; // Store actual camera names
let groundTruthData = {}; // Store ground truth data for each camera

// Grid size configurations
const gridConfigurations = {
    '2x2': { cameras: 4, columns: 2, rows: 2 },
    '3x3': { cameras: 9, columns: 3, rows: 3 },
    '4x4': { cameras: 16, columns: 4, rows: 4 },
    '5x5': { cameras: 25, columns: 5, rows: 5 }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeCameraGrid();
    startDataUpdates();
    setupEventListeners();
    updateCameraRange();
    checkUnityStreaming();

    // Start frame updates
    setTimeout(() => {
        startFrameUpdates();
    }, 2000);
});

// Setup event listeners
function setupEventListeners() {
    // Camera feed click events
    document.addEventListener('click', function (e) {
        if (e.target.closest('.camera-feed')) {
            toggleCameraFullscreen(e.target.closest('.camera-feed'));
        }
    });
}

// Change grid size
function changeGridSize(size) {
    currentGridSize = size;
    const config = gridConfigurations[size];

    // Update selected cameras to fit new grid using available camera names
    selectedCameras = cameraNames.slice(0, config.cameras);

    updateCameraGrid();
    updateCameraRange();
}

// Update camera selection based on input
function updateCameraSelection(value) {
    if (!value.trim()) {
        selectedCameras = [];
        updateCameraGrid();
        return;
    }

    const newSelection = [];

    // Handle range format (e.g., "1-5") - convert to camera names
    if (value.includes('-')) {
        const ranges = value.split(',');
        ranges.forEach(range => {
            if (range.includes('-')) {
                const [start, end] = range.trim().split('-').map(Number);
                if (!isNaN(start) && !isNaN(end)) {
                    for (let i = start; i <= Math.min(end, 50); i++) {
                        const cameraName = `Camera${i}`;
                        if (cameraNames.includes(cameraName) && !newSelection.includes(cameraName)) {
                            newSelection.push(cameraName);
                        }
                    }
                }
            } else {
                const num = parseInt(range.trim());
                if (!isNaN(num) && num >= 1 && num <= 50) {
                    const cameraName = `Camera${num}`;
                    if (cameraNames.includes(cameraName) && !newSelection.includes(cameraName)) {
                        newSelection.push(cameraName);
                    }
                }
            }
        });
    } else {
        // Handle comma-separated format (e.g., "1,3,5,7") - convert to camera names
        const cameras = value.split(',');
        cameras.forEach(cam => {
            const num = parseInt(cam.trim());
            if (!isNaN(num) && num >= 1 && num <= 50) {
                const cameraName = `Camera${num}`;
                if (cameraNames.includes(cameraName) && !newSelection.includes(cameraName)) {
                    newSelection.push(cameraName);
                }
            }
        });
    }

    // Limit to current grid capacity
    const maxCameras = gridConfigurations[currentGridSize].cameras;
    selectedCameras = newSelection.slice(0, maxCameras);

    updateCameraGrid();
}

// Update camera range input field
function updateCameraRange() {
    const input = document.getElementById('cameraRange');
    if (selectedCameras.length === 0) {
        input.value = '';
    } else if (selectedCameras.length <= 5) {
        // Convert camera names back to numbers for display
        const numbers = selectedCameras.map(name => name.replace('Camera', '')).join(',');
        input.value = numbers;
    } else {
        // Convert camera names back to numbers for range display
        const firstNum = selectedCameras[0].replace('Camera', '');
        const lastNum = selectedCameras[selectedCameras.length - 1].replace('Camera', '');
        input.value = `${firstNum}-${lastNum}`;
    }
}

// Initialize camera grid
function initializeCameraGrid() {
    updateCameraGrid();
}

// Update camera grid display
function updateCameraGrid() {
    const grid = document.getElementById('cameraGrid');
    const config = gridConfigurations[currentGridSize];

    // Update grid CSS class
    grid.className = `camera-grid grid-${currentGridSize}`;

    // Clear existing cameras
    grid.innerHTML = '';

    // Add camera feeds
    for (let i = 0; i < config.cameras; i++) {
        const cameraId = selectedCameras[i] || null;
        const cameraFeed = createCameraFeed(cameraId, i + 1);
        grid.appendChild(cameraFeed);
    }

    // If we have actual camera names, update the feeds
    if (cameraNames.length > 0) {
        updateCameraFeedsWithNames();
    }
}

// Create individual camera feed element
function createCameraFeed(cameraId, position) {
    const feedDiv = document.createElement('div');
    feedDiv.className = `camera-feed ${cameraId ? 'active' : ''}`;
    feedDiv.dataset.position = position;

    if (cameraId) {
        feedDiv.dataset.cameraId = cameraId;

        // Use camera name directly (cameraId is now the camera name like "Camera1")
        const cameraName = cameraId;
        const unityCamera = availableUnityCameras.find(cam => cam.name === cameraId);
        const status = unityCamera ? unityCamera.status : 'offline';

        feedDiv.innerHTML = `
            <div class="camera-header-info">
                <span class="camera-id">${cameraName}</span>
                <span class="camera-status ${status}" id="status-${cameraId}">
                    ${status === 'online' ? 'LIVE' : 'OFFLINE'}
                </span>
            </div>
            <div class="camera-content">
                <div class="detection-overlay" id="detection-overlay-${cameraId}"></div>
                <img class="unity-stream" id="stream-${cameraId}" 
                     src="" 
                     alt="${cameraName}" 
                     style="width: 100%; height: 100%; object-fit: cover; display: none;" />
                <div class="stream-overlay" id="overlay-${cameraId}">
                    <div class="loading-spinner">
                        <i class="fas fa-circle-notch fa-spin"></i>
                        <span>Connecting...</span>
                    </div>
                </div>
            </div>
            <div class="vehicle-count" id="count-${cameraId}">
                <i class="fas fa-eye"></i> <span id="detections-${cameraId}">0</span>
            </div>
        `;
    } else {
        feedDiv.innerHTML = `
            <div class="camera-content">
                <div class="camera-placeholder">
                    <i class="fas fa-video-slash"></i>
                    <span>No Camera<br>Selected</span>
                </div>
            </div>
        `;
    }

    return feedDiv;
}

// Update camera feeds with actual camera names
function updateCameraFeedsWithNames() {
    const cameraFeeds = document.querySelectorAll('.camera-feed.active');
    cameraFeeds.forEach((feed, index) => {
        const cameraId = feed.dataset.cameraId;
        if (cameraId) {
            // Find the actual camera name from the available cameras
            const unityCamera = availableUnityCameras.find(cam => cam.id == cameraId);
            const cameraName = unityCamera ? unityCamera.name : `CAM ${cameraId.toString().padStart(2, '0')}`;
            const idElement = feed.querySelector('.camera-id');
            if (idElement) {
                idElement.textContent = cameraName;
            }
        }
    });
}

// Draw detection boxes on the camera feed
function drawDetectionBoxes(cameraId, groundTruthObjects) {
    const overlay = document.getElementById(`detection-overlay-${cameraId}`);
    if (!overlay) return;

    // Clear existing boxes
    overlay.innerHTML = '';

    // Draw new boxes
    groundTruthObjects.forEach(obj => {
        if (obj.BoundingBox) {
            const box = document.createElement('div');
            box.className = 'detection-box';

            // Set position and size based on bounding box
            box.style.left = `${obj.BoundingBox.x}px`;
            box.style.top = `${obj.BoundingBox.y}px`;
            box.style.width = `${obj.BoundingBox.width}px`;
            box.style.height = `${obj.BoundingBox.height}px`;

            // Set color based on object type
            let color = '#00d4aa'; // Default color
            switch (obj.Type) {
                case 1: // Vehicle
                    color = '#ff6b6b';
                    break;
                case 2: // Pedestrian
                    color = '#4ecdc4';
                    break;
                case 3: // TrafficLight
                    color = '#ffe66d';
                    break;
                case 4: // Sign
                    color = '#a05195';
                    break;
                case 5: // Bicycle
                    color = '#f95d6a';
                    break;
            }

            box.style.borderColor = color;
            box.style.borderWidth = '2px';
            box.style.borderStyle = 'solid';
            box.style.position = 'absolute';
            box.style.pointerEvents = 'none';

            // Add label
            const label = document.createElement('div');
            label.className = 'detection-label';
            label.style.position = 'absolute';
            label.style.top = '-20px';
            label.style.left = '0';
            label.style.color = color;
            label.style.fontSize = '12px';
            label.style.fontWeight = 'bold';
            label.style.whiteSpace = 'nowrap';

            let labelName = 'Unknown';
            switch (obj.Type) {
                case 1:
                    labelName = 'Vehicle';
                    break;
                case 2:
                    labelName = 'Pedestrian';
                    break;
                case 3:
                    labelName = 'Traffic Light';
                    break;
                case 4:
                    labelName = 'Sign';
                    break;
                case 5:
                    labelName = 'Bicycle';
                    break;
            }

            label.textContent = labelName;
            box.appendChild(label);

            overlay.appendChild(box);
        }
    });
}

// Generate random vehicle detection boxes
function generateRandomDetections() {
    const detections = [];
    const numDetections = Math.floor(Math.random() * 4) + 1;

    for (let i = 0; i < numDetections; i++) {
        const left = Math.random() * 70 + 10; // 10-80%
        const top = Math.random() * 60 + 20;  // 20-80%
        const width = Math.random() * 15 + 8; // 8-23%
        const height = Math.random() * 10 + 6; // 6-16%

        detections.push(`
            <div class="detection-box" style="
                left: ${left}%; 
                top: ${top}%; 
                width: ${width}%; 
                height: ${height}%
            "></div>
        `);
    }

    return detections.join('');
}

// Toggle simulation
function toggleSimulation() {
    isSimulationRunning = !isSimulationRunning;
    const playIcon = document.getElementById('playIcon');
    const simStatus = document.getElementById('simStatus');

    if (isSimulationRunning) {
        playIcon.className = 'fas fa-pause';
        simStatus.textContent = 'Stop Simulation';
        startTrafficAnimation();
    } else {
        playIcon.className = 'fas fa-play';
        simStatus.textContent = 'Start Simulation';
        stopTrafficAnimation();
    }
}

// Start traffic animation
function startTrafficAnimation() {
    // Animate traffic signals
    const signals = document.querySelectorAll('.traffic-signal');
    signals.forEach((signal, index) => {
        setInterval(() => {
            const statuses = ['green', 'yellow', 'red'];
            const currentStatus = signal.dataset.status;
            const currentIndex = statuses.indexOf(currentStatus);
            const nextIndex = (currentIndex + 1) % statuses.length;
            signal.dataset.status = statuses[nextIndex];
        }, 3000 + index * 1000); // Stagger signal changes
    });

    // Update camera feeds more frequently
    setInterval(() => {
        if (isSimulationRunning) {
            updateCameraDetections();
        }
    }, 2000);
}

// Stop traffic animation
function stopTrafficAnimation() {
    // Reset all signals to green
    const signals = document.querySelectorAll('.traffic-signal');
    signals.forEach(signal => {
        signal.dataset.status = 'green';
    });
}

// Update camera detections
function updateCameraDetections() {
    const activeFeeds = document.querySelectorAll('.camera-feed.active');
    activeFeeds.forEach(feed => {
        const overlay = feed.querySelector('.detection-overlay');
        const vehicleCount = feed.querySelector('.vehicle-count');

        if (overlay) {
            overlay.innerHTML = generateRandomDetections();
        }

        if (vehicleCount) {
            const count = Math.floor(Math.random() * 15) + 1;
            vehicleCount.innerHTML = `<i class="fas fa-car"></i> ${count}`;
        }
    });
}

// Toggle left panel
function toggleLeftPanel() {
    leftPanelCollapsed = !leftPanelCollapsed;
    const panel = document.querySelector('.left-panel');
    const icon = document.getElementById('collapseIcon');

    if (leftPanelCollapsed) {
        panel.classList.add('collapsed');
        icon.className = 'fas fa-chevron-right';
    } else {
        panel.classList.remove('collapsed');
        icon.className = 'fas fa-chevron-left';
    }
}

// Toggle camera fullscreen
function toggleCameraFullscreen(cameraElement) {
    // Remove any existing fullscreen
    const existingFullscreen = document.querySelector('.camera-feed.fullscreen');
    if (existingFullscreen) {
        existingFullscreen.classList.remove('fullscreen');
        existingFullscreen.style.cssText = '';
    }

    // Add fullscreen to clicked camera
    if (existingFullscreen !== cameraElement && cameraElement.classList.contains('active')) {
        cameraElement.classList.add('fullscreen');
        cameraElement.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80vw;
            height: 80vh;
            z-index: 1000;
            background: #0f1419;
            border: 3px solid #00d4aa;
            box-shadow: 0 0 50px rgba(0, 0, 0, 0.8);
        `;
    }
}

// Camera selection functions
function selectAllCameras() {
    const maxCameras = gridConfigurations[currentGridSize].cameras;
    selectedCameras = cameraNames.slice(0, maxCameras);
    updateCameraGrid();
    updateCameraRange();
}

function clearCameraSelection() {
    selectedCameras = [];
    updateCameraGrid();
    updateCameraRange();
}

function randomizeCameras() {
    const maxCameras = gridConfigurations[currentGridSize].cameras;
    const availableCameras = [...cameraNames];
    selectedCameras = [];

    for (let i = 0; i < Math.min(maxCameras, availableCameras.length); i++) {
        const randomIndex = Math.floor(Math.random() * availableCameras.length);
        selectedCameras.push(availableCameras.splice(randomIndex, 1)[0]);
    }

    // Sort by camera number
    selectedCameras.sort((a, b) => {
        const numA = parseInt(a.replace('Camera', ''));
        const numB = parseInt(b.replace('Camera', ''));
        return numA - numB;
    });
    updateCameraGrid();
    updateCameraRange();
}

// Start real-time data updates
function startDataUpdates() {
    updateInterval = setInterval(() => {
        fetchUnityStats();
        fetchUnityCameras();
        updateGPSData();
        updateSystemStatus();
    }, 2000);
}

// Fetch Unity cameras
function fetchUnityCameras() {
    fetch(`http://localhost:${streamingPort}/api/cameras`)
        .then(response => response.json())
        .then(cameras => {
            availableUnityCameras = cameras;
            cameraNames = cameras.map(cam => cam.name);

            // Update camera selection to use actual camera names
            if (cameras.length > 0) {
                const maxCameras = gridConfigurations[currentGridSize].cameras;
                // Use actual camera names directly
                selectedCameras = cameras.slice(0, maxCameras).map(cam => cam.name);
                updateCameraGrid();
                updateCameraRange();
            }
        })
        .catch(error => {
            console.log('Camera data not available:', error);
        });
}

// Update GPS data with realistic values
function updateGPSData() {
    const activeVehicles = document.getElementById('activeVehicles');
    const avgSpeed = document.getElementById('avgSpeed');
    const trafficDensity = document.getElementById('trafficDensity');
    const congestionPoints = document.getElementById('congestionPoints');
    const signalEfficiency = document.getElementById('signalEfficiency');
    const lastUpdate = document.getElementById('lastUpdate');

    if (activeVehicles) {
        const baseVehicles = 247;
        const variation = Math.floor(Math.random() * 40) - 20;
        activeVehicles.textContent = Math.max(200, baseVehicles + variation);
    }

    if (avgSpeed) {
        const baseSpeed = 32.5;
        const variation = (Math.random() * 8) - 4;
        avgSpeed.textContent = (baseSpeed + variation).toFixed(1) + ' km/h';
    }

    if (trafficDensity) {
        const densities = ['Low', 'Medium', 'High'];
        const classes = ['status-success', 'status-warning', 'status-error'];
        const randomIndex = Math.floor(Math.random() * 3);
        trafficDensity.textContent = densities[randomIndex];
        trafficDensity.className = 'value ' + classes[randomIndex];
    }

    if (congestionPoints) {
        const points = Math.floor(Math.random() * 6);
        congestionPoints.textContent = points;
        congestionPoints.className = points > 3 ? 'value status-error' :
            points > 1 ? 'value status-warning' :
                'value status-success';
    }

    if (signalEfficiency) {
        const baseEfficiency = 87.3;
        const variation = (Math.random() * 10) - 5;
        const efficiency = Math.max(70, Math.min(98, baseEfficiency + variation));
        signalEfficiency.textContent = efficiency.toFixed(1) + '%';
        signalEfficiency.className = efficiency > 85 ? 'value status-success' :
            efficiency > 75 ? 'value status-warning' :
                'value status-error';
    }

    if (lastUpdate) {
        lastUpdate.textContent = Math.floor(Math.random() * 5) + 1 + 's ago';
    }
}

// Update system status indicators
function updateSystemStatus() {
    const statusItems = document.querySelectorAll('.status-item');
    statusItems.forEach((item, index) => {
        const circle = item.querySelector('.fas.fa-circle');
        if (circle) {
            // Simulate occasional disconnections
            if (Math.random() > 0.95) {
                circle.className = 'fas fa-circle status-error';
                if (index === 0) {
                    item.innerHTML = '<i class="fas fa-circle status-error"></i> SUMO Reconnecting...';
                } else {
                    const activeCams = Math.floor(Math.random() * 5) + 75;
                    item.innerHTML = `<i class="fas fa-circle status-warning"></i> ${activeCams} Cameras Active`;
                }
            } else {
                circle.className = 'fas fa-circle status-online';
                if (index === 0) {
                    item.innerHTML = '<i class="fas fa-circle status-online"></i> SUMO Connected';
                } else {
                    // Show actual camera count
                    fetch(`http://localhost:${streamingPort}/api/stats`)
                        .then(response => response.json())
                        .then(stats => {
                            item.innerHTML = `<i class="fas fa-circle status-online"></i> ${stats.cameras_online}/${stats.total_cameras} Cameras Active`;
                        })
                        .catch(() => {
                            item.innerHTML = '<i class="fas fa-circle status-online"></i> 80 Cameras Active';
                        });
                }
            }
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    if (frameUpdateInterval) {
        clearInterval(frameUpdateInterval);
    }
});

// Handle escape key to exit fullscreen
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const fullscreenCamera = document.querySelector('.camera-feed.fullscreen');
        if (fullscreenCamera) {
            fullscreenCamera.classList.remove('fullscreen');
            fullscreenCamera.style.cssText = '';
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    // Adjust grid if needed
    setTimeout(() => {
        updateCameraGrid();
    }, 100);
});

// Unity Streaming Integration Functions
function checkUnityStreaming() {
    // Always use streaming mode
    enableUnityStreamingMode();
    console.log('Direct streaming mode enabled');
}

function enableUnityStreamingMode() {
    // Fetch real cameras
    fetchUnityCameras();

    // Update UI to indicate streaming mode
    const logo = document.querySelector('.logo h1');
    if (logo) {
        logo.textContent = 'SynchroCity - Direct Streaming';
    }

    // Add streaming indicator
    const statusIndicators = document.querySelector('.status-indicators');
    if (statusIndicators) {
        // Remove existing status if present
        const existingStatus = statusIndicators.querySelector('.streaming-status');
        if (existingStatus) {
            existingStatus.remove();
        }

        const streamingStatus = document.createElement('span');
        streamingStatus.className = 'status-item streaming-status';
        streamingStatus.innerHTML = '<i class="fas fa-circle status-online"></i> Direct Streaming Active';
        statusIndicators.appendChild(streamingStatus);
    }

    console.log('Direct streaming mode enabled');
}

function fetchUnityStats() {
    fetch(`http://localhost:${streamingPort}/api/stats`)
        .then(response => response.json())
        .then(stats => {
            updateUnityStats(stats);

            // Update connection status
            const streamingStatusElement = document.querySelector('.streaming-status');
            if (streamingStatusElement) {
                const isConnected = stats.connection_active;
                const statusClass = isConnected ? 'status-online' : 'status-error';
                const statusText = isConnected ? 'Direct Streaming Active' : 'Streaming Disconnected';

                streamingStatusElement.innerHTML = `<i class="fas fa-circle ${statusClass}"></i> ${statusText}`;
            }
        })
        .catch(error => {
            console.error('Error fetching stats:', error);

            // Update status to indicate error
            const streamingStatusElement = document.querySelector('.streaming-status');
            if (streamingStatusElement) {
                streamingStatusElement.innerHTML = '<i class="fas fa-circle status-error"></i> Streaming Connection Error';
            }
        });
}

function updateUnityStats(stats) {
    // Update GPS data with stats
    const activeVehicles = document.getElementById('activeVehicles');
    const activeCameras = document.getElementById('activeCameras');

    if (activeVehicles && stats.total_frames) {
        // Use frame count as approximation for vehicle activity
        const vehicleEstimate = Math.max(Math.floor(stats.total_frames / 10), 50);
        activeVehicles.textContent = Math.min(vehicleEstimate, 999);
    }

    // Update camera count in status
    const statusItems = document.querySelectorAll('.status-item');
    statusItems.forEach(item => {
        if (item.textContent.includes('Cameras Active')) {
            const onlineCameras = stats.cameras_online || 0;
            const totalCameras = stats.total_cameras || 0;
            item.innerHTML = `<i class="fas fa-circle ${onlineCameras > 0 ? 'status-online' : 'status-error'}"></i> ${onlineCameras}/${totalCameras} Cameras Active`;
        }
    });
}

// Start frame updates for streaming
function startFrameUpdates() {
    // Update camera statistics every 2 seconds
    if (frameUpdateInterval) {
        clearInterval(frameUpdateInterval);
    }

    frameUpdateInterval = setInterval(() => {
        fetchUnityCameras();
    }, 2000);

    // Update individual camera frames more frequently
    const imageUpdateInterval = setInterval(() => {
        // Get all active cameras from the API
        fetch(`http://localhost:${streamingPort}/api/cameras`)
            .then(response => response.json())
            .then(cameras => {
                cameras.forEach(camera => {
                    updateCameraFrame(camera.name);
                });
            })
            .catch(error => {
                console.warn('Error fetching camera list:', error);
            });
    }, 200); // Update every 200ms for smoother video

    console.log('Started frame updates for direct streaming');
}

// Update individual camera frame
function updateCameraFrame(cameraName) {
    // Find the camera feed that matches this camera name
    const targetFeed = document.querySelector(`.camera-feed[data-camera-id="${cameraName}"]`);

    if (!targetFeed) return;

    const imgElementInFeed = targetFeed.querySelector('.unity-stream');
    const overlayElementInFeed = targetFeed.querySelector('.stream-overlay');
    const statusElementInFeed = targetFeed.querySelector('.camera-status');
    const detectionOverlay = targetFeed.querySelector('.detection-overlay');
    const detectionsElement = targetFeed.querySelector('.vehicle-count span');

    if (!imgElementInFeed) return;

    // Fetch the frame data
    fetch(`http://localhost:${streamingPort}/frame/${cameraName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.image) {
                imgElementInFeed.src = data.image;
                imgElementInFeed.style.display = 'block';

                // Hide loading overlay
                if (overlayElementInFeed) {
                    overlayElementInFeed.style.display = 'none';
                }

                // Update status
                if (statusElementInFeed) {
                    statusElementInFeed.textContent = 'LIVE';
                    statusElementInFeed.className = 'camera-status online';
                }

                // Update detection count
                if (detectionsElement && data.object_count !== undefined) {
                    detectionsElement.textContent = data.object_count;
                }

                // Mark feed as streaming
                targetFeed.classList.add('streaming');

                // Fetch and draw ground truth data
                fetchGroundTruthData(cameraName);
            }
        })
        .catch(error => {
            console.warn(`Error updating camera ${cameraName}:`, error.message);

            // Show error state
            imgElementInFeed.style.display = 'none';

            if (overlayElementInFeed) {
                overlayElementInFeed.style.display = 'flex';
                overlayElementInFeed.innerHTML = `
                    <div class="loading-spinner">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Waiting for images...</span>
                    </div>
                `;
            }

            // Update status
            if (statusElementInFeed) {
                statusElementInFeed.textContent = 'OFFLINE';
                statusElementInFeed.className = 'camera-status offline';
            }
        });
}

// Fetch and draw ground truth data
function fetchGroundTruthData(cameraName) {
    fetch(`http://localhost:${streamingPort}/api/ground_truth/${cameraName}`)
        .then(response => response.json())
        .then(groundTruthObjects => {
            // Store the ground truth data
            groundTruthData[cameraName] = groundTruthObjects;

            // Draw detection boxes
            drawDetectionBoxes(cameraName, groundTruthObjects);
        })
        .catch(error => {
            console.warn(`Error fetching ground truth for ${cameraName}:`, error);
        });
}

// Handle Unity stream error
function handleUnityStreamError(img, cameraId) {
    console.warn(`Stream error for camera ${cameraId}`);

    // Hide the overlay
    const overlay = document.getElementById(`overlay-${cameraId}`);
    if (overlay) {
        overlay.style.display = 'none';
    }

    // Show error placeholder
    const placeholder = document.createElement('div');
    placeholder.className = 'camera-placeholder unity-error';
    placeholder.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>Stream<br>Disconnected</span>
    `;

    // Replace the image with placeholder
    if (img.parentNode) {
        img.parentNode.replaceChild(placeholder, img);
    }

    // Update status
    const statusElement = document.getElementById(`status-${cameraId}`);
    if (statusElement) {
        statusElement.textContent = 'OFFLINE';
        statusElement.className = 'camera-status offline';
    }
}

function handleUnityStreamLoad(img) {
    // Stream loaded successfully
    img.style.opacity = '1';

    const feed = img.closest('.camera-feed');
    if (feed) {
        feed.classList.add('streaming');

        // Hide the loading overlay
        const cameraId = feed.dataset.cameraId;
        const overlay = document.getElementById(`overlay-${cameraId}`);
        if (overlay) {
            overlay.style.display = 'none';
        }

        // Update status
        const statusElement = document.getElementById(`status-${cameraId}`);
        if (statusElement) {
            statusElement.textContent = 'LIVE';
            statusElement.className = 'camera-status online';
        }
    }
}