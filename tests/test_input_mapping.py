import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add the app directory to the Python path so we can import the Flask app
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.insert(0, app_dir)

from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_handles_minimal_input(client):
    """Test that predict works with just Department and Gender (should use defaults for rest)."""
    response = client.post('/predict', json={
        "Department": "Engineering",
        "Gender": "Male"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert isinstance(data['prediction'], int)
    assert 1 <= data['prediction'] <= 5

def test_predict_handles_all_features(client):
    """Test predict with all possible features provided."""
    response = client.post('/predict', json={
        "Department": "Engineering",
        "Gender": "Male",
        "Age": 35,
        "Work_Hours_Per_Week": 45,
        "Monthly_Salary": 90000,
        "Education_Level": "Master",
        "Resigned": False,
        "Team_Size": 8,
        "Overtime_Hours": 6,
        "Promotions": 2,
        "Efficiency": 85.5,
        "WorkBudgetRatio": 1.2,
        "EngagementScore": 92.3
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert isinstance(data['prediction'], int)
    assert 1 <= data['prediction'] <= 5

def test_predict_handles_mixed_numeric_types(client):
    """Test predict accepts both string and numeric types for numeric fields."""
    response = client.post('/predict', json={
        "Department": "Sales",
        "Gender": "Female",
        "Age": "42",  # as string
        "Monthly_Salary": 75000.0,  # as float
        "Team_Size": 5  # as int
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert isinstance(data['prediction'], int)
    assert 1 <= data['prediction'] <= 5

def test_predict_handles_resigned_variations(client):
    """Test different ways of specifying Resigned (bool, string)."""
    # Test with boolean True
    response1 = client.post('/predict', json={
        "Department": "HR",
        "Resigned": True
    })
    assert response1.status_code == 200
    data1 = response1.get_json()
    assert data1['status'] == 'success'

    # Test with "Yes" string
    response2 = client.post('/predict', json={
        "Department": "HR",
        "Resigned": "Yes"
    })
    assert response2.status_code == 200
    data2 = response2.get_json()
    assert data2['status'] == 'success'
    
    # Predictions should be the same for equivalent inputs
    assert data1['prediction'] == data2['prediction']

def test_predict_handles_education_case_insensitive(client):
    """Test that education level mapping is case-insensitive."""
    response1 = client.post('/predict', json={
        "Department": "IT",
        "Education_Level": "PhD"
    })
    response2 = client.post('/predict', json={
        "Department": "IT",
        "Education_Level": "phd"
    })
    assert response1.status_code == response2.status_code == 200
    data1, data2 = response1.get_json(), response2.get_json()
    assert data1['prediction'] == data2['prediction']

def test_predict_validates_empty_payload(client):
    """Test that empty or invalid JSON returns 400."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'message' in data

def test_predict_one_hot_mutually_exclusive(client):
    """Test that one-hot encoded columns are mutually exclusive."""
    # Make two calls with different departments
    response1 = client.post('/predict', json={
        "Department": "Engineering",
        "Gender": "Male"
    })
    response2 = client.post('/predict', json={
        "Department": "Sales",
        "Gender": "Male"
    })
    
    # If one-hot is working, these should yield different predictions
    # since different department columns would be set
    data1, data2 = response1.get_json(), response2.get_json()
    assert data1['prediction'] != data2['prediction']