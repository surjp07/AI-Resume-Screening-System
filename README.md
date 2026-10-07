# 🤖 AI Resume Screening & Job Recommendation System

College-level AIML project using NLP, TF-IDF and Random Forest.

## Features
- Upload resume PDF or paste text
- PDF text extraction
- TF-IDF + Random Forest classification
- Top 5 job-role predictions
- Skill extraction and grouping
- Match visualization
- Accuracy, classification report and confusion matrix
- Streamlit dashboard
- GitHub / Streamlit Cloud ready

## Run locally
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## GitHub + Streamlit Cloud
1. Create a GitHub repository.
2. Upload this project.
3. Open Streamlit Community Cloud and connect the repo.
4. Set main file to `app.py` and deploy.
5. `requirements.txt` is included. If the model file is absent, the app trains it automatically.

## Dataset
The included dataset contains 180 synthetic educational resume examples across 12 job categories. It is for demonstration, not real hiring.

## Suggested title
**AI-Based Resume Screening and Job Role Recommendation System Using Machine Learning**
