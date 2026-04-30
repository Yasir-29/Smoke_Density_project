const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const cameraStatus = document.getElementById("cameraStatus");

const resultText = document.getElementById("resultText");
const tEstPreview = document.getElementById("tEstPreview");
const highlightPreview = document.getElementById("highlightPreview");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let cameraStream = null;
let cameraTimer = null;
let isProcessing = false;

function setResult(labelName, smokePct) {
  resultText.textContent = `${labelName} — Smoke density: ${smokePct}%`;
}

function setError(message) {
  resultText.textContent = "Prediction failed.";
  resultText.style.color = "#b91c1c";
  if (message) console.error(message);
}

async function predictImageBlob(blob, statusEl) {
  isProcessing = true;
  try {
    statusEl.textContent = "Predicting...";

    const formData = new FormData();
    formData.append("image", blob, "upload.jpg");

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Server error");
    }

    setResult(data.label_name, data.smoke_pct);
    if (data.t_est_preview_b64) {
      tEstPreview.src = `data:image/png;base64,${data.t_est_preview_b64}`;
    }
    if (data.highlight_preview_b64) {
      highlightPreview.src = `data:image/png;base64,${data.highlight_preview_b64}`;
    }
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = String(err?.message || err);
    setError(err?.message);
  } finally {
    isProcessing = false;
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    uploadStatus.textContent = "Choose an image first.";
    return;
  }
  resultText.style.color = "#111827";
  await predictImageBlob(file, uploadStatus);
});

startCameraBtn.addEventListener("click", async () => {
  resultText.style.color = "#111827";
  cameraStatus.textContent = "Requesting camera permission...";

  if (cameraStream) return;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    video.srcObject = cameraStream;
    stopCameraBtn.disabled = false;
    startCameraBtn.disabled = true;
    cameraStatus.textContent = "Running... (updates every ~1.2s)";

    cameraTimer = setInterval(() => {
      if (isProcessing) return;
      const w = video.videoWidth || 480;
      const h = video.videoHeight || 360;
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);

      canvas.toBlob(
        async (blob) => {
          if (!blob) return;
          await predictImageBlob(blob, cameraStatus);
        },
        "image/jpeg",
        0.75
      );
    }, 1200);
  } catch (err) {
    cameraStatus.textContent = String(err?.message || err);
    setError(err?.message);
  }
});

stopCameraBtn.addEventListener("click", () => {
  if (cameraTimer) clearInterval(cameraTimer);
  cameraTimer = null;

  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
  }
  cameraStream = null;
  video.srcObject = null;

  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  cameraStatus.textContent = "Camera stopped.";
});

