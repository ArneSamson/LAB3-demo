let webcamMode = true;

const poseModelURL = "../src/poseModel/";
const imageModelURL = "../src/imageModel/";

let model, webcam, labelContainer, maxPredictions;

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

function initPose() {
    //tensorflow pose detection code
    const poseContainer = document.getElementById('pose-container');

}

function initImage() {
}

function handleFileUpload(event) {
}
