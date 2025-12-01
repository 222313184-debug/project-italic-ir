from flask import Flask, render_template, request
import numpy as np
import joblib

from stylometry import stylometry_10

app = Flask(__name__)

# === Load model (dan scaler kalau ada) ===
svm_model = joblib.load("models/svm_stylo_model.pkl")

# Kalau kamu pakai scaler waktu training, load juga:
# stylo_scaler = joblib.load("models/stylo_scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_ai = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "")

        # 1. Hitung fitur stylometry
        stylo_feat = np.array(stylometry_10(input_text)).reshape(1, -1)

        # 2. Kalau pakai scaler:
        stylo_feat = stylo_scaler.transform(stylo_feat)

        # 3. Prediksi
        y_pred = svm_model.predict(stylo_feat)[0]

        # kalau SVM kamu di-set probability=True
        try:
            proba = svm_model.predict_proba(stylo_feat)[0]
            prob_ai = float(proba[1])  # probabilitas kelas 1 (AI)
        except Exception:
            prob_ai = None

        # mapping label ke kalimat
        if y_pred == 1:
            prediction = "Teks terdeteksi sebagai HASIL AI."
        else:
            prediction = "Teks terdeteksi sebagai TULISAN MANUSIA."

    return render_template(
        "index.html",
        prediction=prediction,
        prob_ai=prob_ai,
        input_text=input_text,
    )


if __name__ == "__main__":
    app.run(debug=True)
