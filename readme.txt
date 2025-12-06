Advanced AI-Powered IPL Match Outcome Prediction System

A hybrid AI engine that predicts IPL match outcomes using machine learning, live weather intelligence, player form analytics, and a locally fine-tuned LLaMA 3.2 cricket LLM.
Built with a futuristic Streamlit UI, this system delivers real-time win probabilities along with expert-level narrative analysis — completely offline.

🚀 Features
->Machine Learning Predictor
  Logistic Regression model trained on engineered IPL match-state features, achieving ~81% accuracy.
->Weather-Aware Predictions
  Integrates real-time weather from wttr.in (temperature, humidity, wind, conditions).
->Player Form Engine
 Last 10-match performance insights with fuzzy name matching.
->Local LLM Cricket Analyst
 LLaMA 3.2 (3B) running via Ollama generates expert match commentary.
->Streamlit Sci-Fi UI
 A neon-styled dashboard for match inputs, probability results, and AI-driven insights.

🧠 How It Works:

User inputs current match details.
    ↓
ML pipeline generates win/lose probability.
    ↓
Weather module fetches live stadium conditions.
    ↓
Player form metrics are computed (optional).
    ↓
LLaMA 3.2 fuses all signals → produces expert analysis.
    ↓
UI displays probabilities, match summary & AI commentary.


🧩 Tech Stack

Python - 3.11, NumPy 1.x, Pandas
scikit-learn 1.2.2
Streamlit
Ollama + LLaMA 3.2
FuzzyWuzzy
wttr.in Weather API

📁 Repository Structure
├── application.py
├── ipl_pred.pkl
├── data_processing.ipynb
├── background_image.jpg
├── requirements.txt
└── README.md

⚙️ Installation & Setup

Install Python 3.11 or 3.12
Install dependencies:
pip install -r requirements.txt


Install Ollama:https://ollama.com

Pull the LLaMA model:
ollama pull llama3.2

Run the application:
streamlit run application.py

🔭 Future Enhancements
Over-by-over win probability charts
XGBoost / Deep Neural Networks
SHAP explainability
Live match scraping via APIs
Enhanced visualization dashboard

⭐ Contributions
Pull requests and improvements are welcome!

_______________________________________________________________________

🚀 Setup Instructions

1️⃣ Install Python (Required Version)

This project must use:
Python 3.11 (recommended)
Python 3.12 also works
❌ Do not use Python 3.13 or 3.14 — incompatible with NumPy 1.x + scikit-learn 1.2.2.

Download Python 3.11 here:
https://www.python.org/downloads/windows/

During installation:
✔ Add Python to PATH
✔ Install for all users


2️⃣ Install Ollama (Required for LLM Reasoning)

Download Ollama for Windows:
https://ollama.com/download

Then verify installation:
ollama list

Pull the required model (example):
ollama pull llama3

3️⃣ Install Required Python Dependencies
First, uninstall incompatible NumPy if installed:
py -3.11 -m pip uninstall -y numpy

Install the exact required versions:
py -3.11 -m pip install "numpy<2.0"
py -3.11 -m pip install scikit-learn==1.2.2

Install all remaining libraries:
py -3.11 -m pip install -r requirements_enhanced.txt


If you don’t have a requirements file, install manually:
py -3.11 -m pip install streamlit pandas fuzzywuzzy python-Levenshtein requests joblib ollama

4️⃣ Run the Application

Navigate to the project folder:
cd Desktop\ipl_prediction


Then launch the Streamlit app:
py -3.11 -m streamlit run application_enhanced.py


The app will open automatically in your browser at:
http://localhost:8501


⚠️ Troubleshooting

🔸 ModuleNotFoundError
If any module is missing:
py -3.11 -m pip install <module_name>

🔸 Ollama not running
Start Ollama manually:

ollama serve
🔸 NumPy/Sklearn errors

Ensure:
numpy < 2.0
scikit-learn == 1.2.2
python == 3.11