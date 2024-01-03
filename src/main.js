let isWebcamMode = true;
let model, webcam, ctx, labelContainer, labelName, labelPrediction, maxPredictions;

labelContainer = document.getElementById("label");
labelName = document.getElementById("label-name");
labelPrediction = document.getElementById("label-confidence");

async function initPose() {
    isWebcamMode = true;
    await init();
}

async function initImage() {
    isWebcamMode = false;
    await init();
}

async function init() {
    const poseModelURL = "src/poseModel/";
    const imageModelURL = "src/imageModel/";

    const modelURL = isWebcamMode ? poseModelURL + "model.json" : imageModelURL + "model.json";
    const metadataURL = isWebcamMode ? poseModelURL + "metadata.json" : imageModelURL + "metadata.json";

    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    if (isWebcamMode) {
        const size = 200;
        const flip = true;
        webcam = new tmPose.Webcam(size, size, flip);
        await webcam.setup();
        await webcam.play();
        window.requestAnimationFrame(loop);
    } else {
        const img = document.getElementById("myImg");
        img.onload = () => {
            predictImage();
        };
        const fileInput = document.getElementById("upload-file");
        const file = fileInput.files[0];
        img.src = URL.createObjectURL(file);
    }
}

function toggleMode() {
    isWebcamMode = !isWebcamMode;
    if (isWebcamMode) {
        document.getElementById("pose-container").style.display = "block";
        document.getElementById("image-container").style.display = "none";
    } else {
        document.getElementById("pose-container").style.display = "none";
        document.getElementById("image-container").style.display = "block";
    }
}

async function loop(timestamp) {
    if (isWebcamMode) {
        webcam.update();
    }

    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    const { pose, posenetOutput } = await model.estimatePose(isWebcamMode ? webcam.canvas : document.getElementById("myImg"));
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

function predictImage() {
    predict();
}

function drawPose(pose) {
    if (isWebcamMode && webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}
