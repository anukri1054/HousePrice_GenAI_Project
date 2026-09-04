import streamlit as st
import pandas as pd
import joblib
import os

from config import MODEL_PATH, OUTPUT_DIR

from genai_service import (
    initialize_gemini,
    generate_house_analysis
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS FOR FRONTEND DESIGN
# =========================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #f4f7fb;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #0f172a !important;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #475569 !important;
    margin-bottom: 30px;
}

/* Normal Cards */
.card {
    background-color: #ffffff !important;
    color: #111827 !important;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

/* Prediction Card */
.prediction-card {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white !important;
}

/* Prediction Card Text */
.prediction-card div,
.prediction-card p {
    color: white !important;
}

/* Metric Title */
.metric-title {
    font-size: 18px;
    font-weight: bold;
}

/* Metric Value */
.metric-value {
    font-size: 38px;
    font-weight: bold;
}

/* Section Title */
.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #1e3a8a !important;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Information Boxes */
.info-box {
    background-color: #ffffff !important;
    color: #111827 !important;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.10);
}

/* Force dark text inside white cards */
.info-box h1,
.info-box h2,
.info-box h3,
.info-box h4,
.info-box p,
.info-box div {
    color: #111827 !important;
}

/* Sidebar Background */
section[data-testid="stSidebar"] {
    background-color: #1e3a8a;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569 !important;
    padding: 20px;
    margin-top: 30px;
}

/* General Text */
p, span, label {
    color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏠 AI-Powered House Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning + Generative AI for Intelligent Real Estate Analysis</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/619/619153.png",
    width=100
)

st.sidebar.title("🏠 House Details")

st.sidebar.markdown(
    "Enter the property and location details below."
)

st.sidebar.divider()


# =========================================================
# INPUT FEATURES
# =========================================================

MedInc = st.sidebar.number_input(
    "Median Income",
    min_value=0.0,
    value=5.0,
    step=0.1
)

HouseAge = st.sidebar.number_input(
    "House Age",
    min_value=1.0,
    value=20.0,
    step=1.0
)

AveRooms = st.sidebar.number_input(
    "Average Rooms",
    min_value=0.0,
    value=6.0,
    step=0.1
)

AveBedrms = st.sidebar.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0,
    step=0.1
)

Population = st.sidebar.number_input(
    "Population",
    min_value=0.0,
    value=1500.0,
    step=100.0
)

AveOccup = st.sidebar.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0,
    step=0.1
)

Latitude = st.sidebar.number_input(
    "Latitude",
    value=34.0,
    step=0.01
)

Longitude = st.sidebar.number_input(
    "Longitude",
    value=-118.0,
    step=0.01
)


st.sidebar.divider()


predict_button = st.sidebar.button(
    "🔮 Predict House Price",
    use_container_width=True
)


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = pd.DataFrame([{
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}])


# =========================================================
# HOME DASHBOARD
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🏠 Prediction Dashboard",
    "📊 Model Analysis",
    "ℹ️ About Project"
])


# =========================================================
# TAB 1 - PREDICTION DASHBOARD
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">House Price Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown(
            """
            <div class="info-box">
            <h3>📋 Property Input Summary</h3>
            Enter the property characteristics using the sidebar.
            The trained Gradient Boosting model will analyze the
            information and estimate the house price.
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="info-box">
            <h3>🤖 AI Technology Used</h3>
            • Linear Regression<br>
            • Gradient Boosting Regressor<br>
            • Hyperparameter Tuning<br>
            • Cross Validation<br>
            • Google Gemini Generative AI
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict_button:

        prediction = model.predict(input_data)[0]

        price_dollars = prediction * 100000


        st.markdown("### 🎯 Prediction Result")


        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="metric-title">
                    Estimated House Price
                </div>

                <div class="metric-value">
                    ${price_dollars:,.2f}
                </div>

                <p>
                    Prediction generated using Gradient Boosting Regressor
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        # -------------------------------------------------
        # FEATURE INPUT DISPLAY
        # -------------------------------------------------

        st.markdown("### 📊 Input Features Used")


        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)


        feature_col1.metric(
            "Median Income",
            f"{MedInc:.2f}"
        )

        feature_col2.metric(
            "House Age",
            f"{HouseAge:.0f} Years"
        )

        feature_col3.metric(
            "Average Rooms",
            f"{AveRooms:.2f}"
        )

        feature_col4.metric(
            "Population",
            f"{Population:.0f}"
        )


        st.divider()


        # =================================================
        # GENERATIVE AI ANALYSIS
        # =================================================

        st.markdown(
            '<div class="section-title">🤖 Generative AI Property Analysis</div>',
            unsafe_allow_html=True
        )


        try:

            api_key = st.secrets["GEMINI_API_KEY"]

            initialize_gemini(api_key)


            feature_importance = """
1. Median Income - Most Important Feature
2. Average Occupancy
3. Longitude
4. Latitude
5. House Age
6. Average Number of Rooms
"""


            with st.spinner(
                "🤖 Gemini AI is analyzing the property..."
            ):

                ai_report = generate_house_analysis(
                    input_data.iloc[0].to_dict(),
                    price_dollars,
                    feature_importance
                )


            st.markdown(
                f"""
                <div class="card">
                {ai_report}
                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.warning(
                "Generative AI analysis is currently unavailable."
            )

            st.info(
                "Add a valid Gemini API key in .streamlit/secrets.toml to enable AI analysis."
            )


    else:

        st.info(
            "👈 Enter house details in the sidebar and click 'Predict House Price'."
        )


# =========================================================
# TAB 2 - MODEL ANALYSIS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Machine Learning Model Analysis</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    st.subheader("📈 Model Performance Comparison")


    performance_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Gradient Boosting"
        ],

        "RMSE": [
            0.7456,
            0.5114
        ],

        "MAE": [
            0.5332,
            0.3484
        ],

        "R² Score": [
            0.5758,
            0.8004
        ]
    })


    st.dataframe(
        performance_data,
        use_container_width=True
    )


    st.success(
        "🏆 Gradient Boosting Regressor was selected as the final model because it achieved lower errors and a higher R² score."
    )


    st.divider()


    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader("⭐ Feature Importance")


    feature_path = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    )


    if os.path.exists(feature_path):

        st.image(
            feature_path,
            caption="Feature Importance Generated by Gradient Boosting Model",
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # RESIDUAL PLOT
    # -----------------------------------------------------

    st.subheader("📉 Residual Analysis")


    residual_path = os.path.join(
        OUTPUT_DIR,
        "residual_plot.png"
    )


    if os.path.exists(residual_path):

        st.image(
            residual_path,
            caption="Residual Plot",
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader("🎯 Actual vs Predicted Values")


    prediction_plot_path = os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    )


    if os.path.exists(prediction_plot_path):

        st.image(
            prediction_plot_path,
            caption="Actual vs Predicted House Prices",
            use_container_width=True
        )


# =========================================================
# TAB 3 - ABOUT PROJECT
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">About This Project</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        ### 🎓 Project Title

        **AI-Powered House Price Prediction and Generative AI Insight System**

        ### 🔬 Objective

        The objective of this project is to predict house prices using
        Machine Learning algorithms and provide intelligent explanations
        using Generative AI.

        ### 🛠 Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit
        - Gradient Boosting Regressor
        - Linear Regression
        - Google Gemini API
        - Matplotlib
        - Seaborn

        ### 📊 Dataset

        California Housing Dataset from Scikit-learn.

        ### 🤖 Machine Learning Workflow

        Data Collection
        → Data Exploration
        → Correlation Analysis
        → Train-Test Split
        → Model Training
        → Hyperparameter Tuning
        → Model Evaluation
        → Feature Importance
        → Generative AI Analysis

        ### 📏 Evaluation Metrics

        - RMSE
        - MAE
        - R² Score

        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>AI-Powered House Price Prediction System</b><br>
        B.Tech Final Year Project | Machine Learning + Generative AI
    </div>
    """,
    unsafe_allow_html=True
)