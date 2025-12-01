from flask import Flask, render_template, request
import numpy as np
import joblib

from stylometry import stylometry_10

app = Flask(__name__)

# === Load model (dan scaler kalau ada) ===
svm_model = joblib.load("models/svm_stylo_model.pkl")

# Kalau kamu pakai scaler waktu training, load juga:
stylo_scaler = joblib.load("models/stylo_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    prob_ai = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "")

        # 1. Fitur stylometry
        feat = np.array(stylometry_10(input_text)).reshape(1, -1)

        # 2. Scaling
        feat_scaled = stylo_scaler.transform(feat)

        # 3. Probabilitas AI
        proba = svm_model.predict_proba(feat_scaled)[0]
        prob_ai = float(proba[1])   # kelas 1 = AI

        # 4. THRESHOLD
        THRESHOLD = 0.45  # misal kita buat sedikit lebih sensitif ke AI
        y_pred = 1 if prob_ai >= THRESHOLD else 0

        # 5. Mapping label
        if y_pred == 1:
            prediction_text = "Teks terdeteksi sebagai HASIL AI."
        else:
            prediction_text = "Teks terdeteksi sebagai TULISAN MANUSIA."

    return render_template(
        "index.html",
        prediction=prediction_text,
        prob_ai=prob_ai,
        input_text=input_text,
    )

if __name__ == "__main__":
    app.run(debug=True)
