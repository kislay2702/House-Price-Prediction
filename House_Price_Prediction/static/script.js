/**
 * Estate IQ — client-side logic
 * Handles form submission, live range labels, prediction requests,
 * animated price reveal, session-only history, and reset/validation.
 */
(function () {
  "use strict";

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnLabel = submitBtn.querySelector(".btn-label");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const errorBox = document.getElementById("form-error");
  const resetBtn = document.getElementById("reset-btn");
  const predictAgainBtn = document.getElementById("predict-again-btn");
  const resultSection = document.getElementById("result-section");
  const resultPriceEl = document.getElementById("result-price");
  const resultUsdEl = document.getElementById("result-usd");
  const historySection = document.getElementById("history-section");
  const historyList = document.getElementById("history-list");
  const toast = document.getElementById("toast");

  const USD_TO_INR = 83; // approximate, for display purposes only
  const sessionHistory = [];

  // ---- live range value labels ------------------------------------------
  ["OverallQual", "OverallCond"].forEach((id) => {
    const input = document.getElementById(id);
    const label = document.getElementById(id + "-val");
    input.addEventListener("input", () => { label.textContent = input.value; });
  });

  // ---- currency formatting -----------------------------------------------
  function formatINR(amount) {
    // Indian numbering (lakh/crore) grouping via locale
    return "₹ " + Math.round(amount).toLocaleString("en-IN");
  }
  function formatUSD(amount) {
    return "$" + Math.round(amount).toLocaleString("en-US") + " USD (model native currency)";
  }

  // ---- toast ---------------------------------------------------------------
  let toastTimer = null;
  function showToast(message, kind) {
    clearTimeout(toastTimer);
    toast.textContent = message;
    toast.className = "toast toast--visible" + (kind ? " toast--" + kind : "");
    toast.hidden = false;
    toastTimer = setTimeout(() => {
      toast.classList.remove("toast--visible");
      setTimeout(() => { toast.hidden = true; }, 300);
    }, 3200);
  }

  // ---- animated count-up ----------------------------------------------------
  function animateCount(el, from, to, formatter, duration) {
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      const value = from + (to - from) * eased;
      el.textContent = formatter(value);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ---- client-side validation ------------------------------------------------
  function validateForm(data) {
    const errors = [];
    const numericChecks = {
      GrLivArea: [200, 10000], TotalBsmtSF: [0, 6000], LotArea: [500, 250000],
      FullBath: [0, 5], HalfBath: [0, 3], OverallQual: [1, 10],
      OverallCond: [1, 10], GarageCars: [0, 5], GarageArea: [0, 1500],
      YearBuilt: [1870, 2026], YearRemodAdd: [1870, 2026],
    };
    for (const [field, [min, max]] of Object.entries(numericChecks)) {
      const val = Number(data[field]);
      const fieldEl = document.getElementById(field);
      if (Number.isNaN(val) || val < min || val > max) {
        errors.push(`${field} must be between ${min} and ${max}.`);
        fieldEl.closest(".field").classList.add("field--invalid");
      } else {
        fieldEl.closest(".field").classList.remove("field--invalid");
      }
    }
    if (Number(data.YearRemodAdd) < Number(data.YearBuilt)) {
      errors.push("Year Remodeled cannot be earlier than Year Built.");
      document.getElementById("YearRemodAdd").closest(".field").classList.add("field--invalid");
    }
    return errors;
  }

  // ---- history rendering -----------------------------------------------------
  function renderHistory() {
    if (sessionHistory.length === 0) {
      historySection.hidden = true;
      return;
    }
    historySection.hidden = false;
    historyList.innerHTML = "";
    sessionHistory.slice().reverse().forEach((entry) => {
      const card = document.createElement("div");
      card.className = "history__item";
      card.innerHTML = `
        <p class="h-price">${formatINR(entry.priceInr)}</p>
        <div class="h-meta">
          <span>${entry.neighborhood} · ${entry.style}</span>
          <span>${entry.time}</span>
        </div>`;
      historyList.appendChild(card);
    });
  }

  // ---- submit handler ----------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    const errors = validateForm(data);
    if (errors.length > 0) {
      errorBox.textContent = errors[0];
      errorBox.hidden = false;
      showToast("Please fix the highlighted fields.", "error");
      return;
    }

    submitBtn.disabled = true;
    btnLabel.textContent = "Estimating…";
    btnSpinner.hidden = false;

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const payload = await res.json();

      if (!res.ok || !payload.success) {
        throw new Error(payload.error || "Prediction failed.");
      }

      const priceUsd = payload.predicted_price;
      const priceInr = priceUsd * USD_TO_INR;

      resultSection.hidden = false;
      animateCount(resultPriceEl, 0, priceInr, formatINR, 1100);
      resultUsdEl.textContent = formatUSD(priceUsd);

      sessionHistory.push({
        priceInr,
        neighborhood: data.Neighborhood,
        style: data.HouseStyle,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
      renderHistory();

      resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      showToast("Estimate generated successfully.", "success");
    } catch (err) {
      errorBox.textContent = err.message || "Something went wrong. Please try again.";
      errorBox.hidden = false;
      showToast(err.message || "Prediction failed.", "error");
    } finally {
      submitBtn.disabled = false;
      btnLabel.textContent = "Estimate price";
      btnSpinner.hidden = true;
    }
  });

  // ---- reset ----------------------------------------------------------------------
  resetBtn.addEventListener("click", () => {
    form.reset();
    document.getElementById("OverallQual-val").textContent = document.getElementById("OverallQual").value;
    document.getElementById("OverallCond-val").textContent = document.getElementById("OverallCond").value;
    document.querySelectorAll(".field--invalid").forEach((el) => el.classList.remove("field--invalid"));
    errorBox.hidden = true;
    resultSection.hidden = true;
    showToast("Form reset.", null);
  });

  predictAgainBtn.addEventListener("click", () => {
    document.getElementById("estimate").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
