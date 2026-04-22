"""
scraper.py — Fixed scraper for myscheme.gov.in
Uses Selenium with full JS rendering for each scheme page

INSTALL:
    pip install selenium webdriver-manager beautifulsoup4 requests

RUN:
    python scraper.py

OUTPUT:
    schemes.csv  — ready to use with app.py
"""

import csv, time, os, re, json, logging
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemes.csv")
BASE_URL    = "https://www.myscheme.gov.in"

# ══════════════════════════════════════════════════════════════
# ALL SCHEME SLUGS — verified from myscheme.gov.in
# ══════════════════════════════════════════════════════════════
SCHEME_SLUGS = [
    # Health & Nutrition
    "ab-pmjay",
    "jsy",
    "jssk",
    "mission-indradhanush",
    "poshan-abhiyaan",
    "rbsk",
    "pm-poshan",
    "nphce",
    "icds",
    "pmuy",
    # Education
    "rte",
    "samagra-shiksha",
    "nmms",
    "pre-matric-scholarship-sc",
    "pre-matric-scholarship-st",
    "post-matric-scholarship-sc",
    "post-matric-scholarship-st",
    "nsp-minority",
    "kgbv",
    "emrs",
    # Employment & Skill
    "pmkvy",
    "ddu-gky",
    "pmmy",
    "pm-svanidhi",
    "pm-vishwakarma",
    "sui",
    "mgnregs",
    "eshram",
    "pmegp",
    "naps",
    # Women & Child
    "pmmvy",
    "ssy",
    "pmuy",
    "ignwps",
    "bbbp",
    "swadhar-greh",
    # Agriculture
    "pm-kisan",
    "kcc",
    "pmfby",
    "pmmsy",
    "pkvy",
    "pmksy",
    "shc",
    # Financial & Insurance
    "pmjdy",
    "pmjjby",
    "pmsby",
    "pmgkay",
    "pmayg",
    "pmay-u",
    "apy",
    # Senior Citizens
    "ignoaps",
    "scss",
    "rvy",
    "pmvvy",
    "annapurna",
    # Disability
    "adip",
    "divyangjan-scholarship",
    "nhfdc",
    "ddrs",
    # Social Security & Labour
    "nsap",
    "bocw",
    "aam-aadmi-bima-yojana",
    # SC/ST
    "post-matric-scholarship-obc",
    "national-sc-st-hub",
    "vcf-sc",
    # APL Schemes
    "nps",
    "cgtmse",
    "msme-credit-guarantee",
]

# ══════════════════════════════════════════════════════════════
# CATEGORY DETECTION
# ══════════════════════════════════════════════════════════════
CATEGORY_MAP = {
    "agriculture": "Agriculture",
    "kisan":       "Agriculture",
    "fasal":       "Agriculture",
    "fisherm":     "Agriculture",
    "education":   "Education",
    "scholarship": "Education",
    "school":      "Education",
    "health":      "Health & Nutrition",
    "nutrition":   "Health & Nutrition",
    "ayushman":    "Health & Nutrition",
    "employment":  "Employment & Skill",
    "skill":       "Employment & Skill",
    "mudra":       "Employment & Skill",
    "loan":        "Employment & Skill",
    "housing":     "Housing",
    "awas":        "Housing",
    "women":       "Women & Child",
    "child":       "Women & Child",
    "maternity":   "Women & Child",
    "ujjwala":     "Women & Child",
    "pension":     "Social Security",
    "insurance":   "Insurance",
    "shram":       "Social Security",
    "disability":  "Disability",
    "divyang":     "Disability",
    "financial":   "Financial",
    "food":        "Food Security",
    "anna":        "Food Security",
}

FEMALE_KEYWORDS = [
    "woman", "women", "female", "girl", "widow",
    "mother", "pregnant", "lactating", "mahila",
    "maternity", "wife", "daughter", "sukanya"
]

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def clean_text(t):
    if not t: return ""
    t = re.sub(r'\s+', ' ', str(t)).strip()
    # Remove common noise
    for noise in ["Get in touch", "Connect on Social", "MeitY", "©", "®"]:
        t = t.replace(noise, "")
    return t.strip()

def extract_age(text):
    t = text.lower()
    # Pattern: "18 to 40 years" or "18-40 years"
    m = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*year', t)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Pattern: "above 60" or "minimum 18 years"
    m = re.search(r'above\s+(\d+)|minimum.*?(\d+)\s*year', t)
    if m:
        val = next(x for x in m.groups() if x)
        return int(val), 120
    # Pattern: "below 10" or "under 14"
    m = re.search(r'below\s+(\d+)|under\s+(\d+)', t)
    if m:
        val = next(x for x in m.groups() if x)
        return 0, int(val)
    # Keywords
    if any(w in t for w in ["infant", "newborn", "child birth"]):
        return 0, 2
    if any(w in t for w in ["anganwadi", "toddler"]):
        return 0, 6
    if any(w in t for w in ["senior citizen", "elderly", "old age"]):
        return 60, 120
    if any(w in t for w in ["pregnant", "maternity", "lactating"]):
        return 18, 45
    if any(w in t for w in ["student", "school", "college", "scholarship"]):
        return 6, 35
    if any(w in t for w in ["child", "minor"]):
        return 0, 18
    return 0, 120

def detect_gender(text):
    t = text.lower()
    female_only = [
        "only for women", "women only", "girl child only",
        "widow", "sukanya", "mahila yojana", "maternity benefit",
        "beti bachao", "pregnant women"
    ]
    if any(k in t for k in female_only):
        return "female"
    if any(k in t for k in FEMALE_KEYWORDS):
        if any(k in t for k in ["male", "men", "all citizen", "family", "farmer"]):
            return "any"
        return "female"
    return "any"

def detect_category(name, elig):
    combined = (name + " " + elig).lower()
    for kw, cat in CATEGORY_MAP.items():
        if kw in combined:
            return cat
    return "Social Security"

def clean_name(name):
    """Remove noise from scheme names"""
    for noise in [
        "| myScheme", "- myScheme", "myScheme",
        "Government of India", "Apply Online",
        "Get in touch", "Home"
    ]:
        name = name.replace(noise, "")
    return clean_text(name)

# ══════════════════════════════════════════════════════════════
# MAIN SCRAPER — Selenium with full JS rendering
# ══════════════════════════════════════════════════════════════
def setup_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    return driver


def scrape_scheme(driver, slug):
    """Scrape one scheme page using Selenium (full JS rendered)"""
    url = f"{BASE_URL}/schemes/{slug}"
    try:
        driver.get(url)
        # Wait for JS to render — key fix!
        time.sleep(4)

        # Try __NEXT_DATA__ first (fastest and most accurate)
        try:
            next_data = driver.execute_script(
                "return document.getElementById('__NEXT_DATA__') ? "
                "document.getElementById('__NEXT_DATA__').textContent : null"
            )
            if next_data:
                data = json.loads(next_data)
                scheme = extract_from_nextdata(data, slug, url)
                if scheme and len(scheme.get("name","")) > 5:
                    return scheme
        except Exception:
            pass

        # Fallback: parse rendered HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return parse_rendered_html(soup, slug, url)

    except Exception as e:
        log.warning(f"  ❌ {slug}: {e}")
        return None


def extract_from_nextdata(data, slug, url):
    """Extract scheme data from Next.js __NEXT_DATA__ JSON"""
    try:
        props = data.get("props", {}).get("pageProps", {})

        # Try all possible data keys
        scheme_data = (
            props.get("schemeData") or
            props.get("scheme") or
            props.get("data") or
            props.get("schemeDetail") or
            props.get("schemeInfo") or {}
        )
        if not scheme_data:
            return None

        # Extract name
        name = (
            scheme_data.get("schemeName") or
            scheme_data.get("name") or
            scheme_data.get("title") or ""
        )
        if isinstance(name, dict):
            name = name.get("en", "") or list(name.values())[0]
        name = clean_name(str(name))
        if not name or len(name) < 5:
            return None

        # Extract ministry
        ministry = (
            scheme_data.get("ministry") or
            scheme_data.get("department") or
            scheme_data.get("implementingMinistry") or
            scheme_data.get("nodal_ministry") or
            "Government of India"
        )
        if isinstance(ministry, dict):
            ministry = ministry.get("en", "") or list(ministry.values())[0]
        ministry = clean_text(str(ministry))[:100]

        # Extract benefit / description
        benefit = (
            scheme_data.get("shortDescription") or
            scheme_data.get("benefit") or
            scheme_data.get("description") or
            scheme_data.get("briefDescription") or
            name
        )
        if isinstance(benefit, dict):
            benefit = benefit.get("en", "") or list(benefit.values())[0]
        benefit = clean_text(BeautifulSoup(str(benefit), "html.parser").get_text())[:300]

        # Extract eligibility
        elig = (
            scheme_data.get("eligibility") or
            scheme_data.get("eligibilityCriteria") or
            scheme_data.get("eligibilityText") or
            scheme_data.get("eligibility_criteria") or ""
        )
        if isinstance(elig, dict):
            elig = elig.get("en", "") or list(elig.values())[0]
        if isinstance(elig, list):
            elig = " ".join(str(e) for e in elig)
        elig_text = clean_text(
            BeautifulSoup(str(elig), "html.parser").get_text()
        )[:500]

        # Extract documents
        docs_raw = (
            scheme_data.get("documents") or
            scheme_data.get("requiredDocuments") or
            scheme_data.get("documentsRequired") or []
        )
        docs = []
        if isinstance(docs_raw, list):
            for d in docs_raw[:6]:
                if isinstance(d, dict):
                    t = d.get("en", "") or d.get("name", "") or str(d)
                else:
                    t = str(d)
                t = clean_text(
                    BeautifulSoup(str(t), "html.parser").get_text()
                )
                if t and len(t) > 3:
                    docs.append(t)
        if not docs:
            docs = infer_documents(name, elig_text)

        # Extract last date
        last_date = (
            scheme_data.get("lastDate") or
            scheme_data.get("applicationDeadline") or
            scheme_data.get("deadline") or
            "Open year-round — visit myscheme.gov.in"
        )
        if isinstance(last_date, dict):
            last_date = last_date.get("en", "") or "Open year-round"
        last_date = clean_text(str(last_date)) or "Open year-round — visit myscheme.gov.in"

        age_min, age_max = extract_age(elig_text + " " + benefit)
        gender   = detect_gender(elig_text + " " + benefit + " " + name)
        category = detect_category(name, elig_text)

        log.info(f"  ✅[JSON] {name[:50]} | {age_min}-{age_max} | {gender} | {category}")

        return {
            "id":               slug.upper().replace("-","")[:8],
            "name":             name,
            "ministry":         ministry,
            "category":         category,
            "age_min":          age_min,
            "age_max":          age_max,
            "benefit":          benefit,
            "eligibility_text": elig_text,
            "documents":        " | ".join(docs),
            "last_date":        last_date,
            "url":              url,
            "gender":           gender,
        }

    except Exception as e:
        log.debug(f"  NextData parse failed for {slug}: {e}")
        return None


def parse_rendered_html(soup, slug, url):
    """Parse fully rendered HTML page from Selenium"""
    page_text = soup.get_text(" ", strip=True)

    # Skip pages with no real content
    if len(page_text) < 300:
        return None
    if "page not found" in page_text.lower():
        return None

    # Skip noise words
    skip = ["quick", "menu", "sign", "myscheme", "useful links",
            "navigation", "footer", "get in touch", "connect on social"]

    # Find scheme name from h1/h2
    name = ""
    for tag in ["h1", "h2", "h3"]:
        for el in soup.find_all(tag):
            t = clean_text(el.get_text())
            if len(t) > 8 and not any(w in t.lower() for w in skip):
                name = t
                break
        if name:
            break

    # Fallback to page title
    if not name:
        title = soup.find("title")
        if title:
            name = clean_name(title.get_text())

    if not name or len(name) < 5:
        return None

    # Find ministry
    ministry = "Government of India"
    for kw in ["Ministry of", "Department of", "Ministry for"]:
        idx = page_text.find(kw)
        if idx != -1:
            ministry = clean_text(page_text[idx:idx+100]).split("  ")[0]
            break

    # Find benefit from meta description
    benefit = name
    meta = (
        soup.find("meta", {"name": "description"}) or
        soup.find("meta", {"property": "og:description"})
    )
    if meta and meta.get("content", "").strip():
        benefit = clean_text(meta["content"])[:300]

    # Find eligibility section
    elig_text = ""
    for kw in ["eligib", "criteria", "who can apply", "who is eligible"]:
        sec = (
            soup.find(id=re.compile(kw, re.I)) or
            soup.find(class_=re.compile(kw, re.I)) or
            soup.find(string=re.compile(kw, re.I))
        )
        if sec:
            parent = sec if hasattr(sec, 'get_text') else sec.parent
            elig_text = clean_text(parent.get_text())[:500]
            break
    if not elig_text:
        elig_text = (name + " " + benefit).lower()

    # Find documents section
    docs = []
    for kw in ["document", "required doc", "papers needed"]:
        sec = (
            soup.find(id=re.compile(kw, re.I)) or
            soup.find(class_=re.compile(kw, re.I))
        )
        if sec:
            for li in sec.find_all("li")[:6]:
                t = clean_text(li.get_text())
                if t and len(t) > 3:
                    docs.append(t)
            if docs:
                break
    if not docs:
        docs = infer_documents(name, elig_text)

    age_min, age_max = extract_age(elig_text + " " + benefit)
    gender   = detect_gender(elig_text + " " + benefit + " " + name)
    category = detect_category(name, elig_text)

    log.info(f"  ✅[HTML] {name[:50]} | {age_min}-{age_max} | {gender} | {category}")

    return {
        "id":               slug.upper().replace("-", "")[:8],
        "name":             name,
        "ministry":         ministry[:100],
        "category":         category,
        "age_min":          age_min,
        "age_max":          age_max,
        "benefit":          benefit,
        "eligibility_text": elig_text,
        "documents":        " | ".join(docs),
        "last_date":        "Open year-round — visit myscheme.gov.in",
        "url":              url,
        "gender":           gender,
    }


def infer_documents(name, elig):
    """Infer required documents from scheme name/eligibility"""
    docs = ["Aadhaar card"]
    combined = (name + " " + elig).lower()
    if any(w in combined for w in ["income", "bpl", "poverty"]):
        docs.append("Income certificate")
    if any(w in combined for w in ["sc", "scheduled caste", "st", "tribe", "obc"]):
        docs.append("Caste certificate")
    if any(w in combined for w in ["student", "school", "college", "scholarship"]):
        docs.append("Mark sheet / Bonafide certificate")
    if any(w in combined for w in ["bank", "loan", "finance"]):
        docs.append("Bank passbook")
    if any(w in combined for w in ["farmer", "kisan", "land", "agriculture"]):
        docs.append("Land ownership documents")
    if any(w in combined for w in ["disability", "divyang", "handicap"]):
        docs.append("Disability certificate")
    if any(w in combined for w in ["widow", "death"]):
        docs.append("Death certificate of spouse")
    if any(w in combined for w in ["pregnant", "maternity", "delivery"]):
        docs.append("ANC registration card")
    docs.append("Passport photo")
    return docs[:6]


def deduplicate(schemes):
    seen, unique = set(), []
    for s in schemes:
        key = s["name"].lower().strip()[:40]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def assign_ids(schemes):
    counter = {}
    for s in schemes:
        prefix = re.sub(r'[^A-Z]', '', s["category"].upper())[:2] or "GN"
        counter[prefix] = counter.get(prefix, 0) + 1
        s["id"] = f"{prefix}{counter[prefix]:02d}"
    return schemes


# ══════════════════════════════════════════════════════════════
# CSV SAVE
# ══════════════════════════════════════════════════════════════
FIELDNAMES = [
    "id", "name", "ministry", "category",
    "age_min", "age_max", "benefit", "eligibility_text",
    "documents", "last_date", "url", "gender"
]

def save_csv(schemes):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for s in schemes:
            writer.writerow({k: s.get(k, "") for k in FIELDNAMES})
    log.info(f"✅ Saved {len(schemes)} schemes → {OUTPUT_FILE}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("🔍 myscheme.gov.in Scraper — Selenium Full Render Mode")
    log.info("=" * 60)

    # Check selenium is installed
    try:
        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        log.error("❌ Run: pip install selenium webdriver-manager beautifulsoup4")
        return

    log.info(f"\n📋 Schemes to scrape: {len(SCHEME_SLUGS)}")
    log.info("⏳ Starting Chrome in headless mode...\n")

    driver = setup_driver()

    schemes = []
    failed  = []

    try:
        for i, slug in enumerate(SCHEME_SLUGS, 1):
            log.info(f"[{i}/{len(SCHEME_SLUGS)}] Scraping: {slug}")
            scheme = scrape_scheme(driver, slug)
            if scheme:
                schemes.append(scheme)
            else:
                failed.append(slug)
                log.warning(f"  ⚠️  Skipped: {slug}")
            # Polite delay to avoid being blocked
            time.sleep(2)

    finally:
        driver.quit()
        log.info("\n🔒 Browser closed.")

    if not schemes:
        log.error("❌ No schemes scraped! Check your internet connection.")
        return

    # Clean up
    schemes = deduplicate(schemes)
    schemes = assign_ids(schemes)
    schemes.sort(key=lambda x: x["category"])

    # Summary
    from collections import Counter
    log.info(f"\n{'='*60}")
    log.info(f"📊 RESULTS")
    log.info(f"{'='*60}")
    log.info(f"✅ Successfully scraped : {len(schemes)} schemes")
    log.info(f"❌ Failed / Skipped     : {len(failed)} slugs")
    if failed:
        log.info(f"   Failed slugs: {', '.join(failed)}")
    log.info(f"\n📂 Categories:")
    for cat, n in sorted(Counter(s["category"] for s in schemes).items()):
        log.info(f"   {cat:<30} : {n}")
    log.info(f"\n👤 Gender breakdown:")
    for g, n in Counter(s["gender"] for s in schemes).items():
        log.info(f"   {g} : {n}")

    save_csv(schemes)

    log.info(f"\n{'='*60}")
    log.info(f"✅ schemes.csv saved!")
    log.info(f"   Next step: python app.py")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
