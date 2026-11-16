# Medical No-Show Predictor & AI Assistant

An end-to-end Machine Learning and Generative AI application designed to predict patient appointment no-shows at medical clinics, providing automated risk scoring, model comparative metrics, and local LLM-powered clinical explanations with actionable front-desk recommendations.

---

## Key Features

- **Predictive Machine Learning Pipeline:** Evaluates and compares multiple algorithms (**LightGBM, XGBoost, Random Forest, and Decision Tree**) to estimate no-show probabilities based on demographics, medical history, and scheduling wait times.
- **Optimized Threshold Tuning:** Threshold analysis and validation pipeline to balance precision, recall, and F1-score according to clinical management needs.
- **Generative AI Clinical Assistant:** Integrates locally with **Ollama (Llama 3 / Llama 3.2:1b)** to translate statistical risks into objective, professional clinical summaries and actionable receptionist recommendations (e.g., automated SMS reminders).
- **REST API:** Built with **FastAPI** and **Pydantic** validation to expose robust endpoints (`/predict`, `/explain`, `/health`).
- **Interactive Web Interface:** A user-friendly dashboard built with **Streamlit** for real-time patient risk analysis and AI-driven clinical assistant insights.

---

## 📊 Model Performance & Comparison

Based on our validation and test benchmarks using the Brazilian Medical Appointment No-Shows dataset (2016), the models achieve a general accuracy ranging between **72% and 77%**, striking a solid balance for clinic operational planning and empty-slot reduction:

| Algorithm | Overall Accuracy | Precision (Faults - 1) | Recall (Faults - 1) | F1-Score (Faults - 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Optimized Random Forest** | **76.92%** | **0.40** | 0.27 | 0.32 |
| **Optimized LightGBM** | 74.68% | 0.38 | **0.40** | **0.39** |
| **Optimized XGBoost** | 74.58% | 0.38 | **0.40** | **0.39** |
| **Decision Tree** | 72.27% | 0.36 | **0.46** | **0.40** |

---

## Project Architecture

```text
app/
├── routes/
│   └── routes.py         # FastAPI router endpoints (/predict, /explain, /health)
├── schemas/
│   └── schema.py         # Pydantic data validation schemas & models
├── services/
│   ├── data_formating.py # Data cleaning and feature engineering
│   ├── date_processing.py# Date and waiting days utility functions
│   ├── frontend.py       # Streamlit user interface dashboard
│   ├── ml_training.py    # Comparative evaluations for ML models
│   └── model_test.py     # Validation, threshold tuning, and artifact generation
└── main.py               # FastAPI application entry point
data/                     # Datasets, processed CSVs, and evaluation metrics
models/                   # Serialized model artifacts (model.pkl)
tests/
├── multi_test.py         # Batch testing script over multiple records
└── single_test.py        # CLI script for single patient prediction testing

README.md                 # Project documentation
requirements.txt          # Python dependencies
``` 

---

## Tech Stack

- Backend / API: FastAPI, Uvicorn, Pydantic
- Machine Learning & Data: LightGBM, Scikit-Learn, Pandas, NumPy, Joblib, CatBoost, XGBoost
- Generative AI: Ollama (Llama 3.2:1b)
- Frontend Dashboard: Streamlit, Requests
  
---

## Installation & Setup
1. Clone the repository and navigate to the root directory
2. Create and activate a virtual environment (Recommended)

`python -m venv venv`

### On Windows:
`venv\Scripts\activate`

### On macOS/Linux:
`source venv/bin/activate`


3. Install dependencies
`pip install -r requirements.txt`

---

## Running the Application

To run the full application stack, you will need two separate terminal windows (with your virtual environment activated).

1. Run Ollama: `Ollama run llama3.2:1b`

2. Start the FastAPI Backend. From the root directory, run: `python app/main.py`
(The API will be available at http://localhost:8000. Interactive Swagger documentation can be accessed at http://localhost:8000/docs).

3. Start the Streamlit Frontend Dashboard. In a separate terminal window, run: `streamlit run app/services/frontend.py`
(The Streamlit dashboard will automatically open in your browser at http://localhost:8501).

---

## ML Pipeline & Design Decisions

- Preventing Data Leakage: Features are engineered strictly around the time of appointment scheduling (e.g., computing Waiting_Days as AppointmentDay - ScheduledDay), avoiding features like SMS_received if they occur post-scheduling.
- Preprocessing (data_formating.py): Cleans raw clinic records, handles datetimes, maps categorical attributes, and builds robust feature sets.
- Training & Validation (model_test.py): Splits data into training, validation, and test sets with stratification, evaluating thresholds to optimize operational metrics and serializing the artifact into models/model.pkl.

# License

This project is open-source and intended for healthcare management optimization, leadership mentoring, and educational purposes.

The dataset is available in: https://www.kaggle.com/datasets/joniarroba/noshowappointments 
