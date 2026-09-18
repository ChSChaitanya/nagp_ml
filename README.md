# Customer Churn Prediction

## Overview
This project builds a Decision Tree model to predict customer churn using the Telco Customer Churn dataset.

## Project Structure
```
nagp_ml/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   └── churn_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Project Files
- `data/` — contains the Telco Customer Churn dataset and data dictionary.
- `notebook/churn_analysis.ipynb` — end-to-end workflow from exploration to model evaluation.
- `model/churn_model.pkl` — saved preprocessing and prediction pipeline.
- `app.py` — FastAPI application exposing a `/predict` endpoint.
- `requirements.txt` — Python package dependencies.
- `sample_request.json` — example payload for prediction requests.

## Modeling Summary
- Target variable: `Churn` (`Yes` or `No`)
- Train/test split: 70:30 with `random_state=42`
- Final model: Decision Tree Classifier selected using F1 score, recall, and precision comparison.

## How to Run the API
1. Create a python virtual environment: `python -m venv churn_env`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook `notebook/churn_analysis.ipynb` to create a new model
4. Start the service: `uvicorn app:app --reload`
5. Send a POST request to `/predict` with the JSON structure in `sample_request.json` using 
   `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json`

## Sample Response
```json
{
  "prediction": "Yes",
  "churn_probability": 0.5247
}
```
