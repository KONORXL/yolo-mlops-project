const fileInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const confidenceInput = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidence-value");
const detectButton = document.querySelector("#detect-button");
const statusMessage = document.querySelector("#status-message");
const resultsSection = document.querySelector("#results-section");
const originalImage = document.querySelector("#original-image");
const resultImage = document.querySelector("#result-image");
const uploadPreview = document.querySelector("#upload-preview");
const detectionsList = document.querySelector("#detections-list");
const detectionCount = document.querySelector("#detection-count");

const acceptedTypes = new Set(["image/jpeg", "image/png"]);
let selectedFile = null;
let originalImageUrl = null;
let resultImageUrl = null;

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = "status-message";

  if (type) {
    statusMessage.classList.add(`is-${type}`);
  }
}

function setLoading(isLoading) {
  detectButton.disabled = isLoading || !selectedFile;
  detectButton.textContent = isLoading ? "Running detection..." : "Detect PPE";
}

function revokeUrl(url) {
  if (url) {
    URL.revokeObjectURL(url);
  }
}

function isValidImage(file) {
  return acceptedTypes.has(file.type);
}

function selectImage(file) {
  if (!file) {
    return;
  }

  if (!isValidImage(file)) {
    selectedFile = null;
    fileInput.value = "";
    fileName.textContent = "No image selected";
    uploadPreview.removeAttribute("src");
    dropZone.classList.remove("has-preview");
    setLoading(false);
    setStatus("Please select a JPG, JPEG, or PNG image.", "error");
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  revokeUrl(originalImageUrl);
  originalImageUrl = URL.createObjectURL(file);
  originalImage.src = originalImageUrl;
  uploadPreview.src = originalImageUrl;
  dropZone.classList.add("has-preview");
  resultsSection.hidden = true;
  setStatus("");
  setLoading(false);
}

function createFormData(file) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  return formData;
}

async function parseError(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return payload.detail || response.statusText;
  }

  return response.statusText || "Request failed";
}

async function requestPredictions(file, confidence) {
  const query = new URLSearchParams({ conf: confidence });

  const predictionRequest = fetch(`/predict?${query}`, {
    method: "POST",
    body: createFormData(file),
  });

  const imageRequest = fetch(`/predict-image?${query}`, {
    method: "POST",
    body: createFormData(file),
  });

  const [predictionResponse, imageResponse] = await Promise.all([
    predictionRequest,
    imageRequest,
  ]);

  if (!predictionResponse.ok) {
    throw new Error(await parseError(predictionResponse));
  }

  if (!imageResponse.ok) {
    throw new Error(await parseError(imageResponse));
  }

  return {
    prediction: await predictionResponse.json(),
    imageBlob: await imageResponse.blob(),
  };
}

function renderDetections(detections = []) {
  const sortedDetections = [...detections].sort(
    (first, second) => second.confidence - first.confidence,
  );

  detectionsList.replaceChildren();
  detectionCount.textContent = sortedDetections.length
    ? `${sortedDetections.length} found`
    : "";

  if (!sortedDetections.length) {
    const emptyState = document.createElement("p");
    emptyState.className = "empty-state";
    emptyState.textContent = "No objects detected at this confidence threshold.";
    detectionsList.append(emptyState);
    return;
  }

  const fragment = document.createDocumentFragment();

  for (const detection of sortedDetections) {
    const row = document.createElement("div");
    row.className = "detection-row";

    const className = document.createElement("span");
    className.className = "detection-class";
    className.textContent = detection.class_name || `Class ${detection.class_id}`;

    const confidence = document.createElement("span");
    confidence.className = "detection-confidence";
    confidence.textContent = `${(Number(detection.confidence) * 100).toFixed(1)}%`;

    row.append(className, confidence);
    fragment.append(row);
  }

  detectionsList.append(fragment);
}

async function runDetection() {
  if (!selectedFile) {
    setStatus("Select an image before running detection.", "error");
    return;
  }

  setLoading(true);
  setStatus("Running inference...");

  try {
    const confidence = Number(confidenceInput.value).toFixed(2);
    const { prediction, imageBlob } = await requestPredictions(
      selectedFile,
      confidence,
    );

    revokeUrl(resultImageUrl);
    resultImageUrl = URL.createObjectURL(imageBlob);
    resultImage.src = resultImageUrl;

    renderDetections(prediction.detections);
    resultsSection.hidden = false;
    setStatus("Detection complete.", "success");
  } catch (error) {
    setStatus(error.message || "Unable to run detection.", "error");
  } finally {
    setLoading(false);
  }
}

fileInput.addEventListener("change", () => {
  selectImage(fileInput.files[0]);
});

confidenceInput.addEventListener("input", () => {
  confidenceValue.textContent = Number(confidenceInput.value).toFixed(2);
});

detectButton.addEventListener("click", runDetection);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];

  if (file) {
    fileInput.files = event.dataTransfer.files;
    selectImage(file);
  }
});
