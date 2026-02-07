"""Flask routes for the tree identification web app."""

import os
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from webapp.model_loader import get_classifier

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WebP."}), 400

    # Save uploaded file
    upload_dir = Path(current_app.config.get("UPLOAD_FOLDER", "webapp/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = upload_dir / filename

    try:
        file.save(filepath)

        # Validate it's a real image
        img = Image.open(filepath)
        img.verify()

        # Run prediction
        top_k = request.form.get("top_k", 5, type=int)
        top_k = min(max(top_k, 1), 10)

        classifier = get_classifier()
        predictions = classifier.predict(str(filepath), top_k=top_k)

        # Format species names for display
        for pred in predictions:
            slug = pred["species"]
            pred["display_name"] = slug.replace("_", " ").title()
            pred["confidence_pct"] = round(pred["confidence"] * 100, 1)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()
