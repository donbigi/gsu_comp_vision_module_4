"use strict";

const $ = (id) => document.getElementById(id);

const mode = $("mode");
const source = $("source");
const fileInput = $("file");
const runBtn = $("run");

let uploadedDataUrl = null;

// ---- control visibility --------------------------------------------------

function syncControls() {
  $("upload-control").hidden = source.value !== "upload";
}

source.addEventListener("change", syncControls);

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => { uploadedDataUrl = reader.result; };
  reader.readAsDataURL(f);
});

// ---- run -----------------------------------------------------------------

async function run() {
  runBtn.disabled = true;
  runBtn.textContent = "Detecting…";

  const payload = { mode: mode.value };
  if (source.value === "upload") {
    if (!uploadedDataUrl) {
      alert("Choose an image file first (Image source = Upload).");
      runBtn.disabled = false;
      runBtn.textContent = "Detect boundary";
      return;
    }
    payload.image = uploadedDataUrl;
  }

  try {
    const res = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    render(data);
  } catch (err) {
    console.error(err);
    alert("Detection failed: " + err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Detect boundary";
  }
}

function render(d) {
  $("img-original").src = d.original;
  $("img-boundary").src = d.boundary;
  $("img-overlay").src = d.overlay;
  $("img-gradient").src = d.gradient;
  $("img-edges").src = d.edges;
  $("img-spectrum").src = d.spectrum;

  // Spectral-residual saliency is only produced for RGB.
  const saliencyPanel = $("saliency-panel");
  if (d.saliency) {
    $("img-saliency").src = d.saliency;
    saliencyPanel.hidden = false;
  } else {
    saliencyPanel.hidden = true;
  }

  $("m-coverage").textContent = (100 * d.coverage).toFixed(1) + "%";
  $("m-vertices").textContent = d.boundary_vertices;
  $("m-length").textContent = d.boundary_length.toFixed(0) + " px";

  const iou = $("m-iou");
  if (d.sam2 && d.sam2.iou !== undefined) {
    iou.textContent = d.sam2.iou.toFixed(3);
    iou.title = "Dice " + d.sam2.dice.toFixed(3) +
      " · pixel acc " + d.sam2.pixel_accuracy.toFixed(3) +
      " · boundary F1 " + d.sam2.boundary_f1.toFixed(3);
  } else {
    iou.textContent = "—";
    iou.title = "SAM2 comparison is only available for the default test images.";
  }
}

runBtn.addEventListener("click", run);
syncControls();
run(); // load the demonstration immediately
