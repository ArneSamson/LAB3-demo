let webcamMode = true;

const poseModelURL = "../src/poseModel/";
const imageModelURL = "../src/imageModel/";

let model, webcam, ctx, labelContainer, labelName, labelConfidence, maxPredictions;

labelContainer = document.getElementById("label");
labelName = document.getElementById("label-name");
labelPrediction = document.getElementById("label-confidence");

function toggleMode() {
    webcamMode = !webcamMode;
    updateUI();
}

function updateUI() {
    const poseContainer = document.getElementById('pose-container');
    const imageContainer = document.getElementById('image-container');

    if (webcamMode) {
        poseContainer.style.display = 'block';
        imageContainer.style.display = 'none';
    } else {
        poseContainer.style.display = 'none';
        imageContainer.style.display = 'block';
    }
}

async function initPose() {
    const modelURL = poseModelURL + "model.json";
    const metadataURL = poseModelURL + "metadata.json";

    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    
    const size = 200;
    const flip = true;
    webcam = new tmPose.Webcam(200, 200, flip);
    await webcam.setup();
    await webcam.play();
    window.requestAnimationFrame(loop);
    
    const canvas = document.getElementById("canvas");
    canvas.width = size; canvas.height = size;
    ctx = webcam.canvas.getContext('2d');

    labelContainer.appendChild(webcam.canvas);

}

function initImage() {
}

function handleFileUpload(event) {
}

async function loop() {
    webcam.update();
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
        // Prediction 2: run input through teachable machine classification model
        const prediction = await model.predict(posenetOutput);

        let highestPrediction = { className: "", probability: 0 };
        for (let i = 0; i < maxPredictions; i++) {
            if (prediction[i].probability > highestPrediction.probability) {
                highestPrediction = prediction[i];
            }
        }

        labelName.innerHTML = highestPrediction.className + " ";
        labelPrediction.innerHTML = (highestPrediction.probability * 100).toFixed(0) + "% ";

        drawPose(pose);
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}