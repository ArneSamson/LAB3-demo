let isWebcamMode = false;

const poseModelURL = "../src/poseModel/";
const imageModelURL = "../src/imageModel/";
let model, webcam, ctx, labelContainer, labelName, labelConfidence, maxPredictions;

labelContainer = document.getElementById("label");
labelName = document.getElementById("label-name");
labelPrediction = document.getElementById("label-confidence");



async function init() {
    if (isWebcamMode) {
        const modelURL = poseModelURL + "model.json";
        const metadataURL = poseModelURL + "metadata.json";
        model = await tmPose.load(modelURL, metadataURL);
    } else {
        const modelURL = imageModelURL + "model.json";
        const metadataURL = imageModelURL + "metadata.json";
        model = await tmPose.load(modelURL, metadataURL);
    }

    maxPredictions = model.getTotalClasses();

    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Convenience function to setup a webcam
    const size = 200;
    const flip = true; // whether to flip the webcam
    webcam = new tmPose.Webcam(size, size, flip); // width, height, flip
    await webcam.setup(); // request access to the webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // append/get elements to the DOM
    const canvas = document.getElementById("canvas");
    canvas.width = size; canvas.height = size;
    ctx = canvas.getContext("2d");
}

function handleFileUpload(event) {
    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        isWebcamMode = false;
        // Reset webcam if it's already initialized
        if (webcam) {
            webcam.stop();
            webcam = null;
        }

        // Display the uploaded image
        const img = document.getElementById("myImg");
        img.src = URL.createObjectURL(file);

        // Start processing the uploaded file
        init();
    }
}

async function loop(timestamp) {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Prediction #1: run input through posenet
    // estimatePose can take in an image, video or canvas html element
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    let highestPrediction = { className: "", probability: 0 };
    for (let i = 0; i < maxPredictions; i++) {
        if (prediction[i].probability > highestPrediction.probability) {
            highestPrediction = prediction[i];
        }
    }

    // const classPrediction = highestPrediction.className + ": " + highestPrediction.probability.toFixed(2);
    labelName.innerHTML = highestPrediction.className + " ";
    //percentage of prediction
    labelPrediction.innerHTML = (highestPrediction.probability * 100).toFixed(0) + "% ";
    // finally draw the poses
    drawPose(pose);
    // console.log(highestPrediction.className);
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}