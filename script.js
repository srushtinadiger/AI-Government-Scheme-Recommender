/* ══════════════════════════════════════════════════════════════
   AI Government Scheme Recommender — Frontend Script
   ══════════════════════════════════════════════════════════════ */

// ── Load districts when state is selected ────────────────────────
const stateSelect    = document.getElementById("stateSelect");
const districtSelect = document.getElementById("districtSelect");

if (stateSelect) {
  stateSelect.addEventListener("change", async () => {
    const state = stateSelect.value;
    if (!state) return;

    districtSelect.innerHTML = '<option value="" disabled selected>Loading...</option>';

    try {
      const res  = await fetch(`/get_districts?state=${encodeURIComponent(state)}`);
      const data = await res.json();

      districtSelect.innerHTML = '<option value="" disabled selected>Select district</option>';
      data.forEach(dist => {
        const opt   = document.createElement("option");
        opt.value   = dist;
        opt.textContent = dist;
        districtSelect.appendChild(opt);
      });

      // Add "Other" option
      const other   = document.createElement("option");
      other.value   = "Other";
      other.textContent = "Other";
      districtSelect.appendChild(other);

    } catch (err) {
      districtSelect.innerHTML = '<option value="" disabled selected>Error loading districts</option>';
      console.error("Failed to load districts:", err);
    }
  });
}

// ── Show loading overlay on form submit ──────────────────────────
const form       = document.getElementById("schemeForm");
const submitBtn  = document.getElementById("submitBtn");

// Create loading overlay dynamically
const overlay = document.createElement("div");
overlay.className = "loading-overlay";
overlay.innerHTML = `
  <div class="spinner"></div>
  <div class="loading-text">🤖 AI is finding your schemes...</div>
  <div class="loading-text" style="font-size:13px; color: var(--muted);">This may take a few seconds</div>
`;
document.body.appendChild(overlay);

if (form) {
  form.addEventListener("submit", (e) => {
    // Basic validation
    const age    = form.querySelector('[name="age"]').value;
    const income = form.querySelector('[name="income"]').value;
    const gender = form.querySelector('[name="gender"]').value;
    const occ    = form.querySelector('[name="occupation"]').value;
    const state  = form.querySelector('[name="state"]').value;
    const dist   = form.querySelector('[name="district"]').value;
    const cat    = form.querySelector('[name="category"]').value;

    const errors = [];
    if (!age || age < 0 || age > 120) errors.push("• Please enter a valid Age (0–120)");
    if (!income || income < 0)        errors.push("• Please enter a valid Annual Income");
    if (!gender)                      errors.push("• Please select Gender");
    if (!occ)                         errors.push("• Please select Occupation");
    if (!state)                       errors.push("• Please select State");
    if (!dist)                        errors.push("• Please select District");
    if (!cat)                         errors.push("• Please select Social Category");

    if (errors.length > 0) {
      e.preventDefault();
      alert("Please fix the following:\n\n" + errors.join("\n"));
      return;
    }

    // Show loading
    overlay.classList.add("show");
    submitBtn.disabled    = true;
    submitBtn.textContent = "Analysing…";
  });
}

// ── Toggle scheme details (documents + deadline) ─────────────────
function toggleDetails(schemeId, btn) {
  const details = document.getElementById(`details-${schemeId}`);
  if (!details) return;

  const isHidden = details.style.display === "none";
  details.style.display = isHidden ? "block" : "none";
  btn.textContent = isHidden
    ? "▲  Hide Details"
    : "▼  Show Documents & Deadline";
}

// ── Animate scheme cards on load ─────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const cards = document.querySelectorAll(".scheme-card");
  cards.forEach((card, i) => {
    card.style.opacity   = "0";
    card.style.transform = "translateY(16px)";
    card.style.transition = `opacity 0.3s ease ${i * 0.05}s, transform 0.3s ease ${i * 0.05}s`;
    setTimeout(() => {
      card.style.opacity   = "1";
      card.style.transform = "translateY(0)";
    }, 50);
  });
});
