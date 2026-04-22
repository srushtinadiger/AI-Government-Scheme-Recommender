"""
AI Recommender Engine — loads schemes from CSV, uses Sentence-BERT for matching
"""
import csv, os, numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer, util
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))

def _read(filename):
    path = os.path.join(BASE, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_states_districts():
    sd = defaultdict(list)
    for row in _read("states_districts.csv"):
        sd[row["state"]].append(row["district"])
    return dict(sd)

def load_district_hubs():
    hubs = defaultdict(list)
    for row in _read("district_hubs.csv"):
        hubs[row["district"]].append({
            k: row[k] for k in ("name", "address", "phone", "type")
        })
    return dict(hubs)

def load_schemes():
    schemes = []
    for row in _read("schemes.csv"):
        schemes.append({
            **row,
            "age_min":   int(row["age_min"]),
            "age_max":   int(row["age_max"]),
            "documents": row["documents"].split("|"),
            "url":       row.get("url", "https://www.myscheme.gov.in"),
            "gender":    row.get("gender", "any").strip().lower(),
        })
    return schemes

def load_occupations():
    rows    = _read("occupations.csv")
    labels  = [r["display_label"] for r in rows]
    mapping = {r["display_label"]: r["internal_key"] for r in rows}
    return labels, mapping

def load_categories():
    return [r["category"] for r in _read("categories.csv")]

STATES_DISTRICTS     = load_states_districts()
ALL_STATES           = sorted(STATES_DISTRICTS.keys())
DISTRICT_HUBS        = load_district_hubs()
DEFAULT_HUB          = DISTRICT_HUBS.get("DEFAULT", [
    {"name":"District Collectorate","address":"Contact local DC office",
     "phone":"1916","type":"Revenue / Collectorate"},
    {"name":"Common Service Centre","address":"Nearest Gram Panchayat / CSC",
     "phone":"1800-121-3468","type":"Citizen Service Centre"},
])
SCHEMES              = load_schemes()
OCC_LABELS, OCC_MAP  = load_occupations()
CATEGORIES           = load_categories()

OCC_KEYWORDS = {
    "farmer":              "farmer agricultural kisan land crop cultivation",
    "daily wage":          "daily wage laborer unskilled poor worker",
    "self-employed":       "self employed entrepreneur business shop",
    "student":             "student education school college",
    "street vendor":       "street vendor hawker roadside cart",
    "artisan":             "artisan craftsperson traditional skill handcraft",
    "fisherman":           "fisherman fishing boat coastal marine",
    "construction":        "construction worker mason builder labor",
    "homemaker":           "homemaker housewife domestic woman",
    "unemployed":          "unemployed no job seeking work",
    "salaried private":    "salaried employee private sector job",
    "salaried government": "government employee salaried service",
    "retired":             "retired pension senior",
}

def profile_to_text(p):
    parts  = []
    age    = int(p.get("age", 25))
    income = int(p.get("income", 100000))
    occ    = p.get("occupation", "").lower()
    cat    = p.get("category", "General")
    gender = p.get("gender", "Male").lower()

    # Age repeated 3x for stronger matching weight
    if   age <=  2: parts.extend(["infant newborn baby below 2 years"] * 3)
    elif age <=  5: parts.extend([f"toddler child {age} years below 5"] * 3)
    elif age <= 10: parts.extend([f"child {age} years school going"] * 3)
    elif age <= 14: parts.extend([f"child student {age} years elementary education"] * 3)
    elif age <= 18: parts.extend([f"teenager youth {age} years secondary education"] * 3)
    elif age <= 25: parts.extend([f"young adult {age} years college graduate"] * 3)
    elif age <= 40: parts.extend([f"adult {age} years working age"] * 3)
    elif age <= 59: parts.extend([f"middle aged adult {age} years"] * 3)
    elif age <= 79: parts.extend([f"senior citizen elderly {age} years above 60"] * 3)
    else:           parts.extend([f"very elderly aged {age} years above 80"] * 3)

    if   income <  50000: parts.append("very low income poor BPL below poverty line destitute")
    elif income < 100000: parts.append("low income poor BPL below poverty line")
    elif income < 200000: parts.append(f"low income annual {income}")
    elif income < 500000: parts.append(f"middle income annual {income}")
    else:                 parts.append(f"higher income annual {income}")

    # Occupation repeated 2x
    parts.append(f"occupation {occ}")
    parts.append(f"occupation {occ}")
    for k, v in OCC_KEYWORDS.items():
        if k in occ:
            parts.append(v)
            parts.append(v)

    parts.append(f"social category {cat}")
    if "SC"       in cat:         parts.append("SC scheduled caste disadvantaged")
    if "ST"       in cat:         parts.append("ST scheduled tribe tribal disadvantaged")
    if "OBC"      in cat:         parts.append("OBC other backward class")
    if "EWS"      in cat:         parts.append("EWS economically weaker section")
    if "minority" in cat.lower(): parts.append("minority Muslim Christian Sikh Buddhist")

    # Gender repeated 4x for strong signal
    if "female" in gender:
        parts.extend(["woman female girl"] * 4)
    else:
        parts.extend(["man male boy"] * 2)

    if p.get("disability"):     parts.extend(["disability disabled divyang handicapped"] * 3)
    if p.get("bpl"):            parts.extend(["BPL card below poverty line ration card"] * 2)
    if p.get("land"):           parts.extend(["agricultural land farmer kisan"] * 2)
    if p.get("widow"):          parts.extend(["widow destitute woman"] * 3)
    if p.get("pregnant"):       parts.extend(["pregnant lactating mother maternity"] * 3)
    if not p.get("bank", True): parts.append("no bank account unbanked")
    else:                       parts.append("bank account savings")

    return " ".join(parts)

class SchemeRecommender:
    _instance = None

    def __init__(self):
        print("[AI] Loading Sentence-BERT model...")
        if TORCH_AVAILABLE:
            self.model = SentenceTransformer("models/all-MiniLM-L6-v2")
            texts = [s["eligibility_text"] for s in SCHEMES]
            self.embeddings = self.model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False)
            print(f"[AI] Ready! {len(SCHEMES)} schemes encoded.")
        else:
            print("[WARN] sentence-transformers not found — using fallback mode")
            self.model = None
            np.random.seed(42)
            self.embeddings = np.random.rand(len(SCHEMES), 384)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def recommend(self, profile, threshold=0.15):
        age    = int(profile.get("age", 25))
        gender = profile.get("gender", "Male").lower()

        # Hard filter 1: age eligibility
        age_eligible = [s for s in SCHEMES
                        if s["age_min"] <= age <= s["age_max"]]
        if not age_eligible:
            return []

        # Hard filter 2: gender
        # Females see all schemes (gender-neutral + female-specific)
        # Males only see gender-neutral schemes
        if "female" in gender:
            eligible = age_eligible
        else:
            eligible = [s for s in age_eligible if s["gender"] != "female"]

        if not eligible:
            return []

        ptext = profile_to_text(profile)
        idx   = [SCHEMES.index(s) for s in eligible]

        if TORCH_AVAILABLE and self.model:
            pe     = self.model.encode(ptext, convert_to_tensor=True)
            all_sc = util.cos_sim(pe, self.embeddings)[0].cpu().numpy()
            pairs  = [(eligible[i], float(all_sc[idx[i]]))
                      for i in range(len(eligible))]
        else:
            np.random.seed(hash(str(sorted(profile.items()))) % 2**31)
            pv    = np.random.rand(1, 384)
            ev    = np.array(self.embeddings)
            sc    = cosine_similarity(pv, ev)[0]
            pairs = [(eligible[i], float(sc[idx[i]]))
                     for i in range(len(eligible))]

        pairs.sort(key=lambda x: -x[1])
        results = []
        for scheme, score in pairs:
            if score < threshold:
                continue
            s             = scheme.copy()
            s["score"]    = round(score, 4)
            s["match"]    = round(score * 100, 1)
            s["priority"] = ("HIGH"   if score > 0.42 else
                             "MEDIUM" if score > 0.28 else "LOW")
            results.append(s)
        return results
