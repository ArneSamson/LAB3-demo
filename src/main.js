
const URL = "../src/model/";

let model, labelContainer, maxPredictions;

async function init() {
    const modelURL = URL + "model.json";

    // load the model
    model = await tf.loadLayersModel(modelURL);

    // setup the file picker
    const uploadFile = document.getElementById("upload-file");
    uploadFile.addEventListener("change", function (event) {
        const files = event.target.files;
        if (files.length > 0) {
            const file = files[0];
            if (URL.createObjectURL(file)) {
                document.getElementById("myImg").src = URL.createObjectURL(file);
            }
        }
    });
}

async function predict() {
    // get the image element
    const imgElement = document.getElementById("myImg");

    // convert the image to a tensor
    let img = tf.browser.fromPixels(imgElement);
    img = img.expandDims(0);

    // predict can take in a tensor
    const prediction = await model.predict(img);

    // get the class with the highest probability
    const classId = prediction.argMax(-1).dataSync()[0];

    // display the class prediction
    labelContainer.innerHTML = "Class: " + classId;
}