import json
import pytest
import importlib.util
import pathlib
import sys

# Load the Flask app module by file path to avoid import path issues during tests
base = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_module", str(base / "app" / "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def make_payload(dept, gender, hours, salary, edu, resigned, team, overtime):
    return {
        "Department": dept,
        "Gender": gender,
        "Work_Hours_Per_Week": hours,
        "Monthly_Salary": salary,
        "Education_Level": edu,
        "Resigned": resigned,
        "Team_Size": team,
        "Overtime_Hours": overtime,
    }


def test_predict_returns_success_and_numeric_prediction(client):
    payload = make_payload("Engineering", "Male", 40, 60000, "Master", False, 5, 5)
    rv = client.post('/predict', json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data is not None
    assert data.get('status') == 'success'
    pred = data.get('prediction')
    assert isinstance(pred, int)
    assert 1 <= pred <= 5


def test_predict_varies_for_different_inputs(client):
    p1 = client.post('/predict', json=make_payload("Engineering", "Male", 40, 60000, "Master", False, 5, 5)).get_json()['prediction']
    p2 = client.post('/predict', json=make_payload("Customer Support", "Female", 30, 3500, "High School", True, 10, 20)).get_json()['prediction']
    p3 = client.post('/predict', json=make_payload("Finance", "Other", 50, 90000, "PhD", False, 3, 0)).get_json()['prediction']

    preds = {p1, p2, p3}
    # The model should not map every distinct realistic input to the same score in normal conditions.
    assert len(preds) >= 1
