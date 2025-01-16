// 傳遞用的variable
var g_landmarks;
var currentPose="Null";
var IsTraining = false;
var doAlert = false;
// 以下為原始程式碼
const controls = window;
const LandmarkGrid = window.LandmarkGrid;
const drawingUtils = window;
const mpPose = window;
const options = {
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}/${file}`;
    }
};
// Our input frames will come from here.
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const controlsElement = document.getElementsByClassName('control-panel')[0];
const canvasCtx = canvasElement.getContext('2d');
// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new controls.FPS();
// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
    spinner.style.display = 'none';
};
const landmarkContainer = document.getElementsByClassName('landmark-grid-container')[0];
const grid = new LandmarkGrid(landmarkContainer, {
    connectionColor: 0xCCCCCC,
    definedColors: [{ name: 'LEFT', value: 0xffa500 }, { name: 'RIGHT', value: 0x00ffff }],
    range: 2,
    fitToGrid: true,
    labelSuffix: 'm',
    landmarkSize: 2,
    numCellsPerAxis: 4,
    showHidden: false,
    centered: true,
});
let activeEffect = 'mask';
function onResults(results) {
    g_landmarks = results.poseLandmarks;
    // Hide the spinner.
    document.body.classList.add('loaded');
    // Update the frame rate.
    fpsControl.tick();
    // Draw the overlays.
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.segmentationMask) {
        canvasCtx.drawImage(results.segmentationMask, 0, 0, canvasElement.width, canvasElement.height);
        // Only overwrite existing pixels.
        if (activeEffect === 'mask' || activeEffect === 'both') {
            canvasCtx.globalCompositeOperation = 'source-in';
            // This can be a color or a texture or whatever...
            canvasCtx.fillStyle = '#00FF007F';
            canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
        }
        else {
            canvasCtx.globalCompositeOperation = 'source-out';
            canvasCtx.fillStyle = '#0000FF7F';
            canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
        }
        // Only overwrite missing pixels.
        canvasCtx.globalCompositeOperation = 'destination-atop';
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.globalCompositeOperation = 'source-over';
    }
    else {
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    }
    if (results.poseLandmarks) {
        drawingUtils.drawConnectors(canvasCtx, results.poseLandmarks, mpPose.POSE_CONNECTIONS, { visibilityMin: 0.65, color: 'white' });
        drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_LEFT)
            .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' });
        drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_RIGHT)
            .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' });
        drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_NEUTRAL)
            .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'white' });
    }
    canvasCtx.restore();
    if (results.poseWorldLandmarks) {
        grid.updateLandmarks(results.poseWorldLandmarks, mpPose.POSE_CONNECTIONS, [
            { list: Object.values(mpPose.POSE_LANDMARKS_LEFT), color: 'LEFT' },
            { list: Object.values(mpPose.POSE_LANDMARKS_RIGHT), color: 'RIGHT' },
        ]);
    }
    else {
        grid.updateLandmarks([]);
    }
}
const pose = new mpPose.Pose(options);
pose.onResults(onResults);
// Present a control panel through which the user can manipulate the solution
// options.
new controls
    .ControlPanel(controlsElement, {
    selfieMode: true,
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    effect: 'background',
})
    .add([
    new controls.StaticText({ title: 'MediaPipe Pose' }),
    fpsControl,
    new controls.Toggle({ title: 'Selfie Mode', field: 'selfieMode' }),
    new controls.Toggle({ title: 'Enable Alert', field: 'enableAlert' }),
    new controls.SourcePicker({
        onSourceChanged: () => {
            // Resets because this model gives better results when reset between
            // source changes.
            pose.reset();
        },
        onFrame: async (input, size) => {
            const aspect = size.height / size.width;
            let width, height;
            if (window.innerWidth > window.innerHeight) {
                height = window.innerHeight;
                width = height / aspect;
            }
            else {
                width = window.innerWidth;
                height = width * aspect;
            }
            canvasElement.width = width;
            canvasElement.height = height;
            await pose.send({ image: input });
        },
    }),
    
])
    .on(x => {
    const options = x;
    videoElement.classList.toggle('selfie', options.selfieMode);
    activeEffect = x['effect'];
    pose.setOptions(options);
});

// 以上為原始程式碼，以下為新增程式碼
// 我在控制面板中添加了一些新的按鈕，以便在按下時發送POST請求 包含了增加訓練資料、開始訓練、重置環境等功能
// 在controlsElement->control-panel-shell->control-panel後面加入新的<div>...</div>

// onload function
window.onload = function() {
    fetch('/init', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Server initialized successfully.");
            updateFileCounts();
        } else {
            alert("Error initializing the server.");
        }
    })
};

// 向後端請求獲取文件數量
function updateFileCounts() {
    fetch('/get_file_counts', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('normalCount').innerHTML = `Normal Files: ${data.normal}`;
            document.getElementById('abnormalCount').innerHTML = `Abnormal Files: ${data.abnormal}`;
        } else {
            alert("Error getting file counts.");
        }
    });
}

// 把前端算的landmarks傳到後端->server評估完回傳結果->前端顯示(無限循環)
// hint: 如果要蒐集資料，將var score = data.prediction中的score存到一個array中
// SCORE>0.5 -> Good, SCORE<0.5 -> Bad
async function sendLandmarksToServer() {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ landmarks: g_landmarks }),
        });
        const data = await response.json();
        
        if (data.error) {
            console.error('Error:', data.error);
            document.querySelector('.poseResult').innerHTML = 'Null';
            return null;
        } else {
            var score = data.prediction;
            if (score > 0.5) {
                // score -> 標準化輸出0.2f
                document.querySelector('.poseResult').innerHTML = `Pose: Good(${score.toFixed(2)})`;
                document.querySelector('.poseResult').style.color = 'green';
                currentPose = data.prediction;
            } else {
                document.querySelector('.poseResult').innerHTML = `Pose: Bad(${score.toFixed(2)})`;
                document.querySelector('.poseResult').style.color = 'red';
                currentPose = data.prediction;
            }
            return data.prediction;
        }
    } catch (error) {
        console.error('Error:', error);
        document.querySelector('.poseResult').innerHTML = 'Null';
    }
}

// 持續執行 sendLandmarksToServer 函數
async function continuouslySendLandmarks() {
    var record_buffer = [];
    const audio = new Audio('/static/audio/alert.mp3'); // 引用 MP3 文件

    while (true) {
        if (IsTraining) {
            document.querySelector('.poseResult').innerHTML = 'Training...';
            document.querySelector('.poseResult').style.color = 'blue';
        } else {
            res = await sendLandmarksToServer();
            if (res != null && doAlert) {
                record_buffer.push(res);
                if (record_buffer.length == 7) {
                    var count = 0;
                    for (var i = 0; i < record_buffer.length; i++) {
                        if (record_buffer[i] < 0.4) {
                            count++;
                        }
                    }
                    if (count > 5) {
                        audio.play(); // 播放音樂
                        record_buffer = [];
                    }
                    // pop the first element
                    record_buffer.shift();
                }
            }
        }
        await new Promise(resolve => setTimeout(resolve, 1));
    }
}
// 開始執行
continuouslySendLandmarks();
// end



// 在控制面板中添加新的按鈕
const controlPanel = document.querySelector('.control-panel .control-panel-shell  .control-panel');
if (controlPanel) {
    const newHtml = `
        <div class="control-panel-entry control-panel-text">
            <p id="normalCount">Normal Files: 0</p>
            <button id="startNormal">Start Normal Recording</button>
            <button id="saveNormal" disabled>Save Normal Recording</button>
        </div>
        <div class="control-panel-entry control-panel-text">
            <p id="abnormalCount">Abnormal Files: 0</p>
            <button id="startAbnormal">Start Abnormal Recording</button>
            <button id="saveAbnormal" disabled>Save Abnormal Recording</button>
        </div>
        <div class="control-panel-entry control-panel-text" style="display: flex; flex-direction: column;">
            <button id="startTraining">Start Training</button>
            <button id="resetEnv">Reset Environment</button>
        </div>
        <div class="control-panel-entry control-panel-text">
            <button onclick="window.location.href='/visualize';">Visualize Data</button>
        </div>
        
    `;
    controlPanel.insertAdjacentHTML('beforeend', newHtml);
}
// 為新的按鈕添加點擊事件
// document.addEventListener('DOMContentLoaded', (event) => {
//     const alertToggle = document.querySelector('.control-panel-entry.control-panel-toggle.yes');
//     alertToggle.addEventListener('click', () => {
//         const label = alertToggle.querySelector('.label').textContent;
//         if (label === 'Enable Alert') {
//             doAlert = false;
//         }
//     });
// });

document.addEventListener('DOMContentLoaded', (event) => {
    const alertToggle = document.querySelector('.control-panel-entry.control-panel-toggle.no');
    alertToggle.addEventListener('click', () => {
        const label = alertToggle.querySelector('.label').textContent;
        if (label === 'Enable Alert') {
            doAlert = !doAlert;
            console.log(doAlert);
        }
    });
});


document.getElementById('startNormal').onclick = function() {
    // 詢問是否要開始錄製
    if (!confirm("Do you want to start recording normal data?")) {
        return;
    }
    fetch('/start_recording_normal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type: 'normal' })
    });
    this.disabled = true;
    document.getElementById('saveNormal').disabled = false;
    document.getElementById('startAbnormal').disabled = true;
};

document.getElementById('saveNormal').onclick = function() {
    alert("done");
    fetch('/stop_recording_normal', {
        method: 'POST'
    });
    this.disabled = true;
    document.getElementById('startNormal').disabled = false;
    document.getElementById('startAbnormal').disabled = false;
    updateFileCounts();
};

document.getElementById('startAbnormal').onclick = function() {
    // 詢問是否要開始錄製
    if (!confirm("Do you want to start recording abnormal data?")) {
        return;
    }
    fetch('/start_recording_abnormal', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type: 'abnormal' })
    });
    this.disabled = true;
    document.getElementById('saveAbnormal').disabled = false;
    document.getElementById('startNormal').disabled = true;
};

document.getElementById('saveAbnormal').onclick = function() {
    alert("done");
    fetch('/stop_recording_abnormal', {
        method: 'POST'
    });
    this.disabled = true;
    document.getElementById('startAbnormal').disabled = false;
    document.getElementById('startNormal').disabled = false;
    updateFileCounts();
};

document.getElementById('startTraining').onclick = function() {
    // Show the popup
    document.getElementById('popup').style.display = 'flex';
    IsTraining = true;

    fetch('/start_training', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        // Hide the popup
        document.getElementById('popup').style.display = 'none';
        if (data.success) {
            alert("Training completed with accuracy: " + data.accuracy);
        } else {
            alert("Error starting training.");
        }
        IsTraining = false;
    })
    .catch(error => {
        // In case of an error, hide the popup and show the error
        document.getElementById('popup').style.display = 'none';
        alert("An error occurred: " + error);
        IsTraining = false;
    });
};
document.getElementById('resetEnv').onclick = function() {
    fetch('/reset_env', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Environment reset successfully.");
            updateFileCounts();
        } else {
            alert("Error resetting the environment.");
        }
    });
}
