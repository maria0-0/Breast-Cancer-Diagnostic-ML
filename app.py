# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import streamlit.components.v1 as components
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data.load_data import get_data

# Set page title and layout
st.set_page_config(
    page_title="Breast Cancer Predictor",
    layout="wide"
)
# ── 1. Sleek Dark Glass Style (Lets the vibrant cells shine through clearly) ──
st.markdown("""
<style>
    /* Make Streamlit background completely transparent */
    .stApp {
        background: transparent !important;
    }

    /* Dark semi-transparent card so the cells are crisp, but text is readable */
    .block-container {
        background: rgba(10, 15, 30, 0.65) !important;
        backdrop-filter: blur(4px) !important;
        -webkit-backdrop-filter: blur(4px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6) !important;
    }

    /* Crisp white text for headings, paragraphs, and labels */
    h1, h2, h3, h4, h5, p, span, label {
        color: #f8fafc !important;
    }

    /* Style the tabs to match the dark aesthetic */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── 2. Inject Vanta.js CELLS Background ──
components.html("""
<script>
    const parentDoc = window.parent.document;
    const parentWin = window.parent;

    // Only inject once so Streamlit re-runs don't stack multiple 3D canvases
    if (!parentDoc.getElementById('vanta-bg')) {
        const bgDiv = parentDoc.createElement('div');
        bgDiv.id = 'vanta-bg';
        bgDiv.style.position = 'fixed';
        bgDiv.style.top = '0';
        bgDiv.style.left = '0';
        bgDiv.style.width = '100vw';
        bgDiv.style.height = '100vh';
        bgDiv.style.zIndex = '-1';
        bgDiv.style.pointerEvents = 'none';
        parentDoc.body.prepend(bgDiv);

        function loadScript(src, callback) {
            const s = parentDoc.createElement('script');
            s.src = src;
            s.onload = callback;
            parentDoc.head.appendChild(s);
        }

        // Load Three.js then Vanta Cells CDN
        loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js', function() {
            loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.cells.min.js', function() {
                parentWin.VANTA.CELLS({
                    el: bgDiv,
                    mouseControls: true,
                    touchControls: true,
                    gyroControls: false,
                    minHeight: 200.00,
                    minWidth: 200.00,
                    scale: 1.00,
                    color1: 0x2828aa,
                    color2: 0xca1414,
                    size: 2.20,
                    speed: 2.00
                });
            });
        });
    }
</script>
""", height=0)


# Cache the model training so it only runs ONCE
@st.cache_resource
def load_and_train_model():
    X_train, X_test, y_train, y_test, feature_names, target_names = get_data()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(X_train_s, y_train)
    
    return model, scaler, X_train, feature_names, target_names

model, scaler, X_train, FEATURE_NAMES, TARGET_NAMES = load_and_train_model()

# ── UI Header ──
st.title("Breast Cancer Diagnostic Predictor")
st.markdown("""
This tool uses a **Logistic Regression** model trained on the Wisconsin Breast Cancer Dataset.
Adjust the tumor measurements below (defaults are set to the dataset averages), then click **Predict Diagnosis**.
""")

# ── Feature Sliders organized in Tabs ──
st.header(" Tumor Biopsy Measurements")

tab_mean, tab_se, tab_worst = st.tabs([
    " Mean Features (1-10)", 
    " Standard Error (11-20)", 
    " Worst Values (21-30)"
])

# We will collect the 30 slider values into this list
user_inputs = []

def create_sliders_for_range(start_idx, end_idx):
    col1, col2 = st.columns(2)
    inputs = []
    for i in range(start_idx, end_idx):
        name = FEATURE_NAMES[i]
        min_v = float(X_train[:, i].min())
        max_v = float(X_train[:, i].max())
        mean_v = float(X_train[:, i].mean())
        
        # Put half the sliders in column 1 and half in column 2
        col = col1 if (i - start_idx) < (end_idx - start_idx) // 2 else col2
        val = col.slider(
            label=name.replace("_", " ").title(),
            min_value=round(min_v, 3),
            max_value=round(max_v, 3),
            value=round(mean_v, 3),
            key=f"feat_{i}"
        )
        inputs.append(val)
    return inputs

with tab_mean:
    user_inputs.extend(create_sliders_for_range(0, 10))

with tab_se:
    user_inputs.extend(create_sliders_for_range(10, 20))

with tab_worst:
    user_inputs.extend(create_sliders_for_range(20, 30))

# ── Prediction Action & Results ──
st.markdown("---")

col_btn, _ = st.columns([1, 3])
with col_btn:
    run_prediction = st.button("Run Diagnostic Prediction", type="primary", use_container_width=True)

if run_prediction:
    # 1. Prepare and scale the input data
    input_data = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # 2. Get probabilities from the model
    # Note: 0 = Malignant, 1 = Benign
    probabilities = model.predict_proba(input_scaled)[0]
    prob_malignant = float(probabilities[0])
    prob_benign = float(probabilities[1])
    
    predicted_class = int(np.argmax(probabilities))
    diagnosis = TARGET_NAMES[predicted_class].capitalize()
    
    # 3. Display diagnostic outcome
    st.subheader("Diagnostic Assessment")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if predicted_class == 0:  # Malignant
            st.error(f"Classification: **{diagnosis}**")
            st.write("The model identified characteristics consistent with a malignant tumor.")
        else:  # Benign
            st.success(f"Classification: **{diagnosis}**")
            st.write("The model identified characteristics consistent with a benign tumor.")
            
    with res_col2:
        st.metric(label="Model Confidence", value=f"{probabilities[predicted_class]:.1%}")

    # Probability breakdown
    st.markdown("##### Probability Breakdown")
    p_col1, p_col2 = st.columns(2)
    p_col1.write(f"**Benign Probability:** {prob_benign:.2%}")
    p_col1.progress(prob_benign)
    
    p_col2.write(f"**Malignant Probability:** {prob_malignant:.2%}")
    p_col2.progress(prob_malignant)

