from flask import Flask, render_template, request, jsonify
from text_processor import process_text_html

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    text = data.get("text", "")

    # 1. Jalankan CRF untuk memiringkan istilah asing
    html_raw = process_text_html(text)

    # 2. Normalisasi newline
    normalized = (
        html_raw.replace("\r\n", "\n")
                .replace("\r", "\n")
    )

    # 3. Setiap baris non-kosong dianggap 1 paragraf
    lines = normalized.split("\n")
    paragraphs = [line.strip() for line in lines if line.strip()]

    # 4. Bungkus tiap paragraf dengan <p>...</p>
    wrapped_paragraphs = []
    for p in paragraphs:
        # kalau di dalam satu baris masih ada \n (jarang terjadi), bisa diubah ke <br>
        p = p.replace("\n", "<br>")
        wrapped_paragraphs.append(f"<p>{p}</p>")

    final_html = "\n".join(wrapped_paragraphs)

    # 5. Kirim ke frontend
    return jsonify({"html": final_html})


if __name__ == "__main__":
    app.run(debug=True)
