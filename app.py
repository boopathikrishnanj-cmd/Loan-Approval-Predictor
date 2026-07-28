import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown("""
<style>
:root {
    --accent: #4f5bd5;
    --accent-soft: #eef0fd;
    --approve: #1a7f37;
    --approve-bg: #e9f7ee;
    --reject: #b3261e;
    --reject-bg: #fbe9e8;
    --ink: #1b1f2a;
    --muted: #6b7280;
}

html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

.block-container { padding-top: 2rem; max-width: 1150px; }

.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.6rem;
}
.app-header .icon { font-size: 2rem; }
.app-header h1 { font-size: 1.6rem; margin: 0; color: var(--ink); }
.app-header p { margin: 0; color: var(--muted); font-size: 0.95rem; }

.panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
}
.panel h3 { margin-top: 0; font-size: 1.05rem; color: var(--ink); }

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
    margin: 0.9rem 0 0.3rem 0;
}

.verdict-card {
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.verdict-approve { background: var(--approve-bg); border: 1px solid #bfe3c8; }
.verdict-reject  { background: var(--reject-bg); border: 1px solid #f0c2be; }
.verdict-title { font-size: 1.25rem; font-weight: 700; margin: 0 0 2px 0; }
.verdict-approve .verdict-title { color: var(--approve); }
.verdict-reject .verdict-title { color: var(--reject); }
.verdict-sub { color: var(--muted); font-size: 0.88rem; margin: 0; }

.prob-track {
    background: #eef0f3;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 10px;
}
.prob-fill { height: 100%; border-radius: 999px; }

.factor-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.9rem;
    color: var(--ink);
    border-bottom: 1px solid #f1f2f4;
}
.factor-row:last-child { border-bottom: none; }
.factor-pos { color: var(--approve); font-weight: 700; }
.factor-neg { color: var(--reject); font-weight: 700; }

.placeholder-box {
    text-align: center;
    color: var(--muted);
    padding: 3rem 1rem;
    font-size: 0.92rem;
}

.stButton > button {
    width: 100%;
    background: var(--accent);
    color: white;
    border-radius: 10px;
    border: none;
    height: 2.9em;
    font-weight: 600;
    font-size: 0.95rem;
}
.stButton > button:hover { background: #3f4ac0; }

.disclaimer {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 1.6rem;
    padding-top: 0.8rem;
    border-top: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    model = joblib.load("loan_status_model.pkl")
    features = joblib.load("loan_status_features.pkl")
    return model, features

try:
    model, FEATURES = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    load_error = str(e)

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class="app-header">
    <div class="icon">🏦</div>
    <div>
        <h1>Loan Approval Predictor</h1>
        <p>Estimate loan approval odds from applicant &amp; loan details</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error(
        "Couldn't load the model files (loan_status_model.pkl / "
        f"loan_status_features.pkl). Details: {load_error}"
    )
    st.stop()

# ==========================================================
# LAYOUT
# ==========================================================
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Application details")

    with st.form("loan_form"):
        st.markdown('<div class="section-label">Applicant</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
            married = st.selectbox("Married", ["Yes", "No", "Unknown"])
        with c2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+", "Unknown"])
        self_employed = st.selectbox("Self employed", ["No", "Yes", "Unknown"])

        st.markdown('<div class="section-label">Income</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            applicant_income = st.number_input(
                "Applicant income (monthly)", min_value=0.0, value=5000.0, step=100.0
            )
        with c4:
            coapplicant_income = st.number_input(
                "Coapplicant income (monthly)", min_value=0.0, value=0.0, step=100.0
            )

        st.markdown('<div class="section-label">Loan</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            loan_amount = st.number_input(
                "Loan amount (in thousands)", min_value=0.0, value=120.0, step=5.0,
                help="Matches the training data convention, e.g. 120 = 120,000 in local currency."
            )
        with c6:
            loan_term = st.selectbox(
                "Loan term (months)", [360, 180, 120, 84, 60, 36, 12, 6], index=0
            )
        c7, c8 = st.columns(2)
        with c7:
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        with c8:
            credit_history = st.selectbox(
                "Credit history on file", ["Yes (good standing)", "No / negative", "Unknown"]
            )

        submitted = st.form_submit_button("Predict loan approval")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# FEATURE ENCODING (matches loan_status_features.pkl exactly)
# ==========================================================
def prepare_features():
    row = {f: 0 for f in FEATURES}
    row["ApplicantIncome"] = applicant_income
    row["CoapplicantIncome"] = coapplicant_income
    row["LoanAmount"] = loan_amount
    row["Loan_Amount_Term"] = loan_term

    if gender == "Male":
        row["Gender_Male"] = 1
    elif gender == "Unknown":
        row["Gender_unknown"] = 1

    if married == "Yes":
        row["Married_Yes"] = 1
    elif married == "Unknown":
        row["Married_unknown"] = 1

    if dependents in ("1", "2", "3+"):
        row[f"Dependents_{dependents}"] = 1
    elif dependents == "Unknown":
        row["Dependents_unknown"] = 1

    if education == "Not Graduate":
        row["Education_Not Graduate"] = 1

    if self_employed == "Yes":
        row["Self_Employed_Yes"] = 1
    elif self_employed == "Unknown":
        row["Self_Employed_unknown"] = 1

    if property_area == "Semiurban":
        row["Property_Area_Semiurban"] = 1
    elif property_area == "Urban":
        row["Property_Area_Urban"] = 1

    if credit_history.startswith("Yes"):
        row["Credit_History_1.0"] = 1
    elif credit_history == "Unknown":
        row["Credit_History_unknown"] = 1
    # "No / negative" -> both flags stay 0, which the model reads as
    # a known-bad credit history (the reference category)

    return pd.DataFrame([row])[FEATURES]


def explain(row_df, proba):
    """Small rule-based explainer: rank global feature importances,
    then describe how the applicant's own values line up with them."""
    if not hasattr(model, "feature_importances_"):
        return []
    importances = dict(zip(FEATURES, model.feature_importances_))
    row = row_df.iloc[0]
    notes = []

    def add(cond_true_text, cond_false_text, active, weight):
        text = cond_true_text if active else cond_false_text
        notes.append((weight, active, text))

    if "Credit_History_1.0" in importances:
        w = importances["Credit_History_1.0"]
        if row.get("Credit_History_unknown", 0) == 1:
            notes.append((w, None, "Credit history wasn't provided — this is the single biggest factor the model relies on, so the estimate here is less certain."))
        else:
            add("Good credit history on file — the strongest positive signal in this model.",
                "No positive credit history on file — the strongest negative signal in this model.",
                row.get("Credit_History_1.0", 0) == 1, w)

    if "ApplicantIncome" in importances:
        add("Applicant income is comfortably above the typical range in the training data.",
            "Applicant income is on the lower side relative to the training data.",
            applicant_income >= 4000, importances["ApplicantIncome"])

    if "LoanAmount" in importances:
        add("Requested loan amount is modest relative to income.",
            "Requested loan amount is large relative to typical applicants.",
            loan_amount <= 150, importances["LoanAmount"])

    if "Property_Area_Semiurban" in importances:
        add("Semiurban properties historically see slightly higher approval rates in this data.",
            "Property area isn't the semiurban category that tends to score best.",
            property_area == "Semiurban", importances["Property_Area_Semiurban"])

    notes.sort(key=lambda t: -t[0])
    return notes[:4]


# ==========================================================
# RESULT PANEL
# ==========================================================
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Result")

    if submitted:
        try:
            X = prepare_features()
            pred = model.predict(X)[0]
            proba_approve = model.predict_proba(X)[0][1]

            if pred == 1:
                st.markdown(f"""
                <div class="verdict-card verdict-approve">
                    <p class="verdict-title">✅ Likely Approved</p>
                    <p class="verdict-sub">Estimated approval probability: {proba_approve*100:.1f}%</p>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{proba_approve*100:.1f}%; background:var(--approve);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-card verdict-reject">
                    <p class="verdict-title">❌ Likely Rejected</p>
                    <p class="verdict-sub">Estimated approval probability: {proba_approve*100:.1f}%</p>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{proba_approve*100:.1f}%; background:var(--reject);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            notes = explain(X, proba_approve)
            if notes:
                st.markdown('<div class="section-label">Key factors</div>', unsafe_allow_html=True)
                for _, active, text in notes:
                    if active is True:
                        mark = '<span class="factor-pos">▲</span>'
                    elif active is False:
                        mark = '<span class="factor-neg">▼</span>'
                    else:
                        mark = '<span>•</span>'
                    st.markdown(f'<div class="factor-row">{mark} {text}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown(
            '<div class="placeholder-box">Fill in the application details and click '
            '<b>Predict loan approval</b> to see the result here.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# FOOTER / TRANSPARENCY NOTE
# ==========================================================
st.markdown("""
<div class="disclaimer">
Model: Random Forest trained on a public loan-approval dataset (~490 labeled applications,
~81% test accuracy). This is an educational demo, not a real credit decision — actual lenders
use far more data and regulated underwriting processes.
</div>
""", unsafe_allow_html=True)
