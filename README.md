Smart Tumor Classification – MLOps

1. Project Overview

This project uses Machine Learning to classify breast cancer data into Benign and Malignant categories.

It demonstrates a basic MLOps workflow, including data preprocessing, model training, model evaluation, and API development.

Note: This project is for educational purposes only and is not a medical diagnostic tool.

2. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Uvicorn
- Git and GitHub
- MLflow

3. Project Structure

smart-mlops-project/
├── artifacts/
├── data/
├── src/
│   ├── api.py
│   ├── data.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── requirements.txt
└── README.md

4. Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Confusion Matrix Generation
6. FastAPI Development
7. GitHub Version Control

5. How to Run

Install dependencies:

pip install -r requirements.txt

Run preprocessing:

python src/preprocess.py

Train model:

python src/train.py

Evaluate model:

python src/evaluate.py

Start API:

python -m uvicorn src.api:app --host 127.0.0.1 --port 5000

Open API documentation:

http://127.0.0.1:5000/docs

6. Model Artifacts

- best_model.joblib
- scaler.joblib
- processed_data.joblib
- confusion_matrix.png

7. Future Scope

- MLflow experiment tracking
- Model versioning
- Automated testing
- Model monitoring
- Cloud deployment

8. Author

Sanika Rasal