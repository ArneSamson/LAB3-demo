let webcamMode = true;

const poseModelURL = "../src/poseModel/";
const imageModelURL = "../src/imageModel/";

let model, webcam, ctx, poseLabelContainer, poseLabelName, labelConfidence, maxPredictions;

poseLabelContainer = document.getElementById("pose-label-container");
poseLabelName = document.getElementById("pose-label-name");
poseLabelPrediction = document.getElementById("pose-label-confidence");

imageLabelContainer = document.getElementById("image-label-container");
imageLabelName = document.getElementById("image-label-name");
imageLabelPrediction = document.getElementById("image-label-confidence");

function toggleMode() {
    webcamMode = !webcamMode;
    updateUI();
}

function updateUI() {
    const poseContainer = document.getElementById('pose-container');
    const imageContainer = document.getElementById('image-container');
    const poseLabels = document.getElementById('pose-label-container');
    const imageLabels = document.getElementById('image-label-container');

    if (webcamMode) {
        poseContainer.style.display = 'flex';
        poseLabels.style.display = 'flex';
        imageContainer.style.display = 'none';
        imageLabels.style.display = 'none';

    } else {
        poseContainer.style.display = 'none';
        poseLabels.style.display = 'none';
        imageContainer.style.display = 'flex';
        imageLabels.style.display = 'flex';
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

    poseLabelContainer.appendChild(webcam.canvas);

}

function initImage() {
    predictImage();
}

function handleFileUpload(event) {
    const inputElement = event.target;
    const file = inputElement.files[0];
    
    if (file) {
        const image = document.getElementById("myImg");
        image.src = URL.createObjectURL(file);
    }
}

async function predictImage() {
    const image = document.getElementById("myImg");
    const modelURL = imageModelURL + "model.json";
    const metadataURL = imageModelURL + "metadata.json";

    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    const prediction = await model.predict(image);
    let highestPrediction = { className: "", probability: 0 };

    for (let i = 0; i < maxPredictions; i++) {
        if (prediction[i].probability > highestPrediction.probability) {
            highestPrediction = prediction[i];
        }
    }

    imageLabelName.innerHTML = highestPrediction.className;
    imageLabelPrediction.innerHTML = (highestPrediction.probability * 100).toFixed(0) + "%";
}

async function loop() {
    webcam.update();
    await predictPose();
    window.requestAnimationFrame(loop);
}

async function predictPose() {
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
        // Prediction 2: run input through teachable machine classification model
        const prediction = await model.predict(posenetOutput);

        let highestPrediction = { className: "", probability: 0 };
        for (let i = 0; i < maxPredictions; i++) {
            if (prediction[i].probability > highestPrediction.probability) {
                highestPrediction = prediction[i];
            }
        }

        poseLabelName.innerHTML = highestPrediction.className + " ";
        poseLabelPrediction.innerHTML = (highestPrediction.probability * 100).toFixed(0) + "% ";

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