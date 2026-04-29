from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from inference import load_classifier, predict_smoke_from_bytes


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.joblib"

app = Flask(__name__)


@app.get("/health")
def health() -> tuple[object, int]:
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists()}), 200


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    if not MODEL_PATH.exists():
        return jsonify({"error": f"Model not found at {MODEL_PATH}"}), 500

    if "image" not in request.files:
        return jsonify({"error": "Missing form field: image"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image upload"}), 400

    try:
        result = predict_smoke_from_bytes(MODEL_PATH, image_bytes)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200


if __name__ == "__main__":
    # For production, run with a real WSGI server; this is for local demo only.
    load_classifier(str(MODEL_PATH))  # warm-start / fail fast
    app.run(host="0.0.0.0", port=5000, debug=True)

