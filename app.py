"""
app.py — AI Government Scheme Recommender
Flask + Sentence-BERT Transformer + Scraped schemes.csv

RUN:
    pip install flask sentence-transformers scikit-learn numpy waitress
    python app.py
    Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np, csv, os, warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer, util
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

app.config["PROPAGATE_EXCEPTIONS"] = True

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Server busy, please try again."}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Page not found."}), 404


def load_schemes():
    csv_path = os.path.join(BASE, "schemes.csv")
    if not os.path.exists(csv_path):
        print("[WARN] schemes.csv not found — run scraper.py first!")
        return []
    schemes = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["age_min"] = int(row.get("age_min", 0) or 0)
                row["age_max"] = int(row.get("age_max", 120) or 120)
                if not row.get("eligibility_text"):
                    row["eligibility_text"] = (row.get("name","") + " " + row.get("benefit","")).lower()
                schemes.append(row)
            except Exception:
                continue
    print(f"[CSV] Loaded {len(schemes)} schemes from schemes.csv")
    return schemes

SCHEMES = load_schemes()

print("[AI] Loading Sentence-BERT transformer model...")
if TORCH_AVAILABLE and SCHEMES:
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SCHEME_EMBEDDINGS = MODEL.encode(
        [s["eligibility_text"] for s in SCHEMES],
        convert_to_tensor=True, show_progress_bar=False
    )
    print(f"[AI] Model ready! {len(SCHEMES)} schemes embedded.")
else:
    MODEL = None
    np.random.seed(42)
    SCHEME_EMBEDDINGS = np.random.rand(max(len(SCHEMES),1), 384)
    if not TORCH_AVAILABLE:
        print("[WARN] sentence-transformers not installed.")


def profile_to_text(p):
    parts = []
    age    = int(p.get("age", 25))
    income = int(p.get("income", 100000))
    occ    = p.get("occupation", "").lower()
    cat    = p.get("category", "General")
    gender = p.get("gender", "Male").lower()

    if age <= 2:    parts.append("infant newborn below 2 years")
    elif age <= 6:  parts.append(f"toddler child {age} years anganwadi")
    elif age <= 14: parts.append(f"child student {age} years school elementary")
    elif age <= 18: parts.append(f"teenager student {age} years secondary education")
    elif age <= 25: parts.append(f"young adult {age} years college youth")
    elif age <= 40: parts.append(f"adult {age} years working age")
    elif age <= 59: parts.append(f"middle aged adult {age} years")
    elif age <= 79: parts.append(f"senior citizen elderly {age} years above 60")
    else:           parts.append(f"very elderly aged {age} years above 80")

    if income < 50000:    parts.append("very low income poor BPL below poverty destitute")
    elif income < 100000: parts.append("low income poor BPL below poverty line")
    elif income < 250000: parts.append(f"low middle income {income}")
    elif income < 600000: parts.append(f"middle income APL above poverty line {income}")
    else:                 parts.append(f"higher income APL above poverty taxpayer {income}")

    parts.append(f"occupation {occ}")
    occ_keywords = {
        "farmer":          "farmer agricultural kisan land crop cultivation",
        "daily wage":      "daily wage laborer unskilled rural poor worker",
        "self-employed":   "self employed entrepreneur small business shop",
        "student":         "student education school college young",
        "street vendor":   "street vendor hawker roadside cart urban",
        "artisan":         "artisan craftsperson traditional skill handcraft",
        "fisherman":       "fisherman fishing boat coastal marine",
        "construction":    "construction worker mason builder labor",
        "homemaker":       "homemaker housewife domestic woman",
        "unemployed":      "unemployed no job seeking work",
        "salaried private":"salaried private sector employee professional",
        "salaried govt":   "salaried government employee central state",
        "retired":         "retired pension senior ex-employee",
    }
    for k, v in occ_keywords.items():
        if k in occ: parts.append(v)

    parts.append(f"category {cat}")
    if "SC" in cat:      parts.append("SC scheduled caste disadvantaged Dalit")
    if "ST" in cat:      parts.append("ST scheduled tribe tribal disadvantaged")
    if "OBC" in cat:     parts.append("OBC other backward class")
    if "EWS" in cat:     parts.append("EWS economically weaker section")
    if "Minority" in cat:parts.append("minority Muslim Christian Sikh Buddhist")

    if "female" in gender or "woman" in gender:
        parts.append("woman female girl mahila")

    if p.get("disability"): parts.append("disability disabled divyang handicapped")
    if p.get("bpl"):        parts.append("BPL card below poverty line ration poor")
    if p.get("apl"):        parts.append("APL card above poverty line middle class ration")
    if p.get("land"):       parts.append("agricultural land owner farmer kisan cultivator")
    if p.get("widow"):      parts.append("widow destitute woman death husband")
    if p.get("pregnant"):   parts.append("pregnant lactating mother maternity delivery")
    parts.append("bank account savings" if p.get("bank", True) else "no bank account unbanked")
    return " ".join(parts)


def recommend(profile, threshold=0.15):
    age    = int(profile.get("age", 25))
    gender = profile.get("gender", "Male").lower()

    # Strict age filter
    age_pool = [s for s in SCHEMES
                if int(s.get("age_min",0)) <= age <= int(s.get("age_max",120))]

    # Gender filter
    filtered = []
    for s in age_pool:
        sg = s.get("gender","any").lower()
        if sg == "female" and "female" not in gender and "woman" not in gender:
            continue
        filtered.append(s)

    if not filtered:
        return [], profile_to_text(profile)

    profile_text    = profile_to_text(profile)
    eligible_indices = [SCHEMES.index(s) for s in filtered]

    if TORCH_AVAILABLE and MODEL:
        profile_emb = MODEL.encode(profile_text, convert_to_tensor=True)
        all_scores  = util.cos_sim(profile_emb, SCHEME_EMBEDDINGS)[0].cpu().numpy()
        scores = [(filtered[i], float(all_scores[eligible_indices[i]])) for i in range(len(filtered))]
    else:
        np.random.seed(hash(str(sorted(profile.items()))) % 2**31)
        pv = np.random.rand(1, 384)
        ev = np.array(SCHEME_EMBEDDINGS)
        all_sc = cosine_similarity(pv, ev)[0]
        scores = [(filtered[i], float(all_sc[eligible_indices[i]])) for i in range(len(filtered))]

    scores.sort(key=lambda x: -x[1])
    results = []
    for scheme, score in scores:
        if score < threshold: continue
        s = dict(scheme)
        s["score"]    = round(score, 4)
        s["match"]    = round(min(score * 100, 99), 1)
        s["priority"] = "HIGH" if score > 0.42 else "MEDIUM" if score > 0.27 else "LOW"
        results.append(s)

    return results, profile_text


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    try:
        results, profile_text = recommend(data)
        return jsonify({
            "success":      True,
            "total":        len(results),
            "high":         sum(1 for r in results if r["priority"] == "HIGH"),
            "medium":       sum(1 for r in results if r["priority"] == "MEDIUM"),
            "low":          sum(1 for r in results if r["priority"] == "LOW"),
            "profile_text": profile_text,
            "schemes":      results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI Government Scheme Recommender")
    print(f"  Schemes loaded : {len(SCHEMES)}")
    print("  Transformer    : Sentence-BERT (all-MiniLM-L6-v2)")
    print("  Multi-user     : 16 threads (Waitress)")
    print("  Open browser   : http://localhost:5000")
    print("="*55 + "\n")
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000, threads=16)
