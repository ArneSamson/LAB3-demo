let webcamMode = true;

const poseModelURL = "../src/poseModel/";
const imageModelURL = "../src/imageModel/";

let model, webcam, ctx, labelContainer, labelName, labelConfidence, maxPredictions;

labelContainer = document.getElementById("label-container");
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
        labelContainer.innerHTML = `
        <h3>You are performing this pose</h3>
            <div class="labels__label" id="label">
                <div class="label__name" id="label-name"></div>
                <p> with a </p>
                <div class="label__confidence" id="label-confidence"></div>
                <p> accuracy</p>
            </div>
        `;


    } else {
        poseContainer.style.display = 'none';
        imageContainer.style.display = 'block';

        labelContainer.innerHTML = `
        <h3>This pose is</h3>
        <div class="labels__label" id="label">
            <div class="label__name" id="label-name"></div>
            <p> with a </p>
            <div class="label__confidence" id="label-confidence"></div>
            <p> accuracy</p>
        </div>
        `;
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

    labelName.innerHTML = highestPrediction.className + " ";
    labelPrediction.innerHTML = (highestPrediction.probability * 100).toFixed(0) + "% ";
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