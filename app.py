"""
=====================================================================
Titanic Survival Predictor
Version 5.1

Professional Machine Learning Dashboard
Algorithm: Support Vector Machine (SVM)

Developed with:
• Gradio 6.20
• Scikit-Learn
• Pandas
=====================================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# THEME COLORS
# ==========================================================

PRIMARY_COLOR = "#2563eb"
SUCCESS_COLOR = "#16a34a"
DANGER_COLOR = "#dc2626"

BACKGROUND_COLOR = "#f4f7fb"
CARD_COLOR = "#ffffff"

TEXT_COLOR = "#0f172a"
SUBTEXT_COLOR = "#64748b"

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "titanic_svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")
SEX_ENCODER_PATH = os.path.join(MODEL_DIR, "sex_encoder.pkl")
EMBARKED_ENCODER_PATH = os.path.join(MODEL_DIR, "embarked_encoder.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
sex_encoder = joblib.load(SEX_ENCODER_PATH)
embarked_encoder = joblib.load(EMBARKED_ENCODER_PATH)

# ==========================================================
# HTML HELPERS
# ==========================================================

def create_progress_bar(value: float, color: str) -> str:
    """
    Returns an animated HTML progress bar.
    """

    value = max(0, min(100, float(value)))

    return f"""
    <div style="margin-bottom:18px;">

        <div style="
            display:flex;
            justify-content:space-between;
            font-weight:600;
            margin-bottom:6px;
            color:{TEXT_COLOR};
        ">
            <span>{value:.2f}%</span>
        </div>

        <div style="
            width:100%;
            height:18px;
            background:#e5e7eb;
            border-radius:999px;
            overflow:hidden;
        ">

            <div style="
                width:{value:.2f}%;
                height:18px;
                background:{color};
                transition:width .8s ease;
            ">
            </div>

        </div>

    </div>
    """


def create_prediction_card(prediction: int) -> str:
    """
    Creates prediction banner.
    """

    if prediction == 1:
        colour = SUCCESS_COLOR
        icon = "🟢"
        title = "SURVIVED"
        subtitle = "Passenger is predicted to survive."

    else:
        colour = DANGER_COLOR
        icon = "🔴"
        title = "DID NOT SURVIVE"
        subtitle = "Passenger is predicted not to survive."

    return f"""
    <div style="
        background:{colour};
        color:white;
        padding:28px;
        border-radius:18px;
        text-align:center;
        box-shadow:0 10px 22px rgba(0,0,0,.18);
        margin-bottom:18px;
    ">

        <div style="font-size:42px;">
            {icon}
        </div>

        <div style="
            font-size:30px;
            font-weight:700;
            margin-top:10px;
        ">
            {title}
        </div>

        <div style="
            margin-top:8px;
            font-size:16px;
            opacity:.95;
        ">
            {subtitle}
        </div>

    </div>
    """


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_survival(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
):
    """
    Predict Titanic passenger survival.
    """

    family_size = sibsp + parch + 1

    sex_encoded = sex_encoder.transform([sex])[0]
    embarked_encoded = embarked_encoder.transform([embarked])[0]

    passenger = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Sex": [sex_encoded],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked_encoded],
            "FamilySize": [family_size],
        }
    )

    passenger_scaled = scaler.transform(passenger)

    prediction = model.predict(passenger_scaled)[0]
    probability = model.predict_proba(passenger_scaled)[0]

    survive_probability = probability[1] * 100
    nonsurvive_probability = probability[0] * 100

    prediction_card = create_prediction_card(prediction)

    survive_bar = create_progress_bar(
        survive_probability,
        SUCCESS_COLOR,
    )

    nonsurvive_bar = create_progress_bar(
        nonsurvive_probability,
        DANGER_COLOR,
    )

    return (
        prediction_card,
        survive_bar,
        nonsurvive_bar,
        f"{survive_probability:.2f}%",
        f"{nonsurvive_probability:.2f}%",
    )

    # ==========================================================
# CUSTOM CSS
# ==========================================================

CSS = """

.gradio-container{
    max-width:1400px !important;
    margin:auto;
    background:#f4f7fb;
}

footer{
    display:none !important;
}

.card{
    background:white;
    border-radius:18px;
    border:1px solid #dbe4f0;
    padding:22px;
    box-shadow:0 8px 22px rgba(0,0,0,.08);
}

.section-title{
    font-size:24px;
    font-weight:700;
    color:#0f172a;
    margin-bottom:18px;
}

.model-table{
    width:100%;
    border-collapse:collapse;
}

.model-table td{
    padding:10px;
    border-bottom:1px solid #edf2f7;
}

button{
    border-radius:12px !important;
    font-weight:700 !important;
}

"""

# ==========================================================
# USER INTERFACE
# ==========================================================

with gr.Blocks(
    title="Titanic Survival Predictor",
    fill_width=True,
) as demo:

    # ======================================================
    # HEADER
    # ======================================================

    gr.HTML(
        """
        <div style="
            background:linear-gradient(90deg,#1d4ed8,#2563eb);
            border-radius:20px;
            padding:32px;
            color:white;
            text-align:center;
            margin-bottom:22px;
            box-shadow:0 10px 26px rgba(0,0,0,.20);
        ">

            <h1 style="
                margin:0;
                font-size:42px;
                font-weight:800;
            ">
                🚢 Titanic Survival Predictor
            </h1>

            <p style="
                margin-top:12px;
                font-size:18px;
            ">
                Professional Machine Learning Dashboard
            </p>

            <p style="
                margin-top:6px;
                font-size:15px;
                opacity:.9;
            ">
                Support Vector Machine (SVM)
            </p>

        </div>
        """
    )

    # ======================================================
    # MAIN CONTENT
    # ======================================================

    with gr.Row(equal_height=True):

        # ==================================================
        # LEFT COLUMN
        # ==================================================

        with gr.Column(scale=1):

            gr.HTML(
                """
                <div class="card">
                    <div class="section-title">
                        👤 Passenger Information
                    </div>
                </div>
                """
            )

            pclass = gr.Dropdown(
                label="Passenger Class",
                choices=[1, 2, 3],
                value=3,
            )

            sex = gr.Dropdown(
                label="Gender",
                choices=["male", "female"],
                value="male",
            )

            with gr.Row():

                age = gr.Number(
                    label="Age",
                    value=25,
                    minimum=0,
                )

                fare = gr.Number(
                    label="Fare",
                    value=50,
                    minimum=0,
                )

            with gr.Row():

                sibsp = gr.Number(
                    label="Siblings / Spouse",
                    value=0,
                    precision=0,
                    minimum=0,
                )

                parch = gr.Number(
                    label="Parents / Children",
                    value=0,
                    precision=0,
                    minimum=0,
                )

            embarked = gr.Dropdown(
                label="Port of Embarkation",
                choices=["C", "Q", "S"],
                value="S",
            )

            with gr.Row():

                predict_button = gr.Button(
                    "🚀 Predict Survival",
                    variant="primary",
                    scale=2,
                )

                clear_button = gr.Button(
                    "🧹 Clear",
                    variant="secondary",
                    scale=1,
                )

        # ==================================================
        # RIGHT COLUMN
        # ==================================================

        with gr.Column(scale=1):

            gr.HTML(
                """
                <div class="card">
                    <div class="section-title">
                        🤖 AI Prediction
                    </div>
                </div>
                """
            )

            prediction_output = gr.HTML(
                value=""
            )

            gr.Markdown("### ✅ Survival Probability")

            survive_bar_output = gr.HTML()

            survive_text = gr.Textbox(
                label="Confidence",
                interactive=False,
            )

            gr.Markdown("### ❌ Non-Survival Probability")

            nonsurvive_bar_output = gr.HTML()

            nonsurvive_text = gr.Textbox(
                label="Confidence",
                interactive=False,
            )

                    # ==================================================
        # MODEL INFORMATION
        # ==================================================

        gr.HTML(
            """
            <div class="card">

                <div class="section-title">
                    ⚙️ Model Information
                </div>

                <table class="model-table">

                    <tr>
                        <td><b>Algorithm</b></td>
                        <td>Support Vector Machine (SVM)</td>
                    </tr>

                    <tr>
                        <td><b>Dataset</b></td>
                        <td>Kaggle Titanic Dataset</td>
                    </tr>

                    <tr>
                        <td><b>Feature Engineering</b></td>
                        <td>Family Size</td>
                    </tr>

                    <tr>
                        <td><b>Feature Scaling</b></td>
                        <td>StandardScaler</td>
                    </tr>

                    <tr>
                        <td><b>Encoding</b></td>
                        <td>LabelEncoder</td>
                    </tr>

                    <tr>
                        <td><b>Programming Language</b></td>
                        <td>Python</td>
                    </tr>

                    <tr>
                        <td><b>Framework</b></td>
                        <td>Gradio 6.20</td>
                    </tr>

                </table>

            </div>
            """
        )

    # ======================================================
    # BUTTON EVENTS
    # ======================================================

    predict_button.click(
        fn=predict_survival,
        inputs=[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked,
        ],
        outputs=[
            prediction_output,
            survive_bar_output,
            nonsurvive_bar_output,
            survive_text,
            nonsurvive_text,
        ],
    )

    # ======================================================
    # CLEAR BUTTON
    # ======================================================

    def clear_outputs():

        return (
            3,                  # Passenger Class
            "male",             # Gender
            25,                 # Age
            0,                  # SibSp
            0,                  # Parch
            50,                 # Fare
            "S",                # Embarked
            "",                 # Prediction HTML
            "",                 # Survival Bar
            "",                 # Non Survival Bar
            "",                 # Survival %
            "",                 # Non Survival %
        )


    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked,
            prediction_output,
            survive_bar_output,
            nonsurvive_bar_output,
            survive_text,
            nonsurvive_text,
        ],
    )

    # ======================================================
    # EXAMPLE PASSENGERS
    # ======================================================

    gr.Examples(
        examples=[
            [1, "female", 29, 0, 0, 211.3375, "S"],
            [3, "male", 22, 1, 0, 7.2500, "S"],
            [2, "female", 30, 1, 1, 26.0000, "C"],
            [1, "male", 54, 0, 0, 51.8625, "S"],
            [3, "female", 18, 0, 0, 7.2292, "C"],
            [3, "male", 35, 0, 0, 8.0500, "S"],
            [1, "female", 42, 1, 0, 71.2833, "C"],
            [2, "male", 28, 0, 0, 13.0000, "S"],
        ],
        inputs=[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked,
        ],
        outputs=[
            prediction_output,
            survive_bar_output,
            nonsurvive_bar_output,
            survive_text,
            nonsurvive_text,
        ],
        fn=predict_survival,
        cache_examples=False,
        label="🧪 Try Example Passengers",
    )

    # ======================================================
    # USER INSTRUCTIONS
    # ======================================================

    gr.Markdown(
        """
        ### 💡 How to Use

        1. Select the passenger class.
        2. Enter the passenger details.
        3. Click **🚀 Predict Survival**.
        4. Review the prediction and confidence scores.
        5. Click **🧹 Clear** to reset the form.
        """
    )

    # ======================================================
    # FOOTER
    # ======================================================

    gr.HTML(
        """
        <hr>

        <div style="
            text-align:center;
            padding:18px;
            color:#64748b;
            font-size:15px;
        ">

            <h3 style="margin-bottom:8px;">
                🚢 Titanic Survival Predictor
            </h3>

            <p>
                Professional Machine Learning Dashboard
            </p>

            <p>
                Support Vector Machine (SVM)
            </p>

            <p>
                Built with <b>Python</b> •
                <b>Scikit-Learn</b> •
                <b>Pandas</b> •
                <b>Gradio 6.20</b>
            </p>

            <small>
                Version 5.1
            </small>

        </div>
        """
    )
    # ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    demo.queue()

    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True,
    inbrowser=True,
    css=CSS,
)