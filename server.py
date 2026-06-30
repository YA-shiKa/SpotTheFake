import os, tempfile
from flask import Flask, request, jsonify, send_from_directory
import predict

_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=_DIR)


@app.route("/")
def index():
    return send_from_directory(_DIR, "demo.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "no image field in form-data"}), 400

    file = request.files["image"]
    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        score = predict.predict(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

    return jsonify({
        "score": score,
        "label": "FAKE" if score >= 0.5 else "REAL",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)