# Academic Advisor AI

A student academic-advice demonstration built with Python, scikit-learn, Pickle, and Flask.

## Features
- Collects CGPA, attendance, weekly study hours, and number of failed courses.
- Predicts one of three categories: Good standing, Needs improvement, or At risk.
- Displays practical study recommendations in a web interface.
- Includes a model-training script suitable for running in Google Colab.

## Important limitation
The included training examples are synthetic demonstration data, not real student records. Predictions are for demonstration only and should not be presented as validated academic assessments.

## Run locally
1. Install Python 3.10 or newer and Git.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:
   - Windows PowerShell: `python -m venv .venv` then `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `python -m pip install -r requirements.txt`
5. Train the model: `python train_model.py`
6. Start the web app: `python app.py`
7. Open the local address printed in the terminal (usually `http://127.0.0.1:5000`).

## Google Colab model training
Upload `train_model.py` to Google Colab or paste its contents into a cell. Run it after installing scikit-learn if needed. Download `advisor_model.pkl` from the Colab file panel and place it in the same folder as `app.py`.

## Git and GitHub
From this folder, run:
```bash
git init
git add .
git commit -m "Build AI academic advisor"
```
Create a new empty repository on GitHub, then follow the commands GitHub shows to add the remote and push the project. Do not upload passwords, access tokens, or private student records.

## Suggested software engineering practices
- Requirements: accept academic inputs and return guidance.
- Design: separate model training (`train_model.py`) from web serving (`app.py`).
- Testing: validate input ranges and test each prediction category.
- Version control: commit source code and document changes with Git.
- Maintenance: retrain only with approved, representative, anonymized data and evaluate model performance before real use.
