import pytest
import random

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def randomized_inference_data():
    """Fixture to provide randomized data for inference."""
    country = random.choice(["United-States", "China", "Cuba", "?", "Italy", "Vietnam", "Laos", "Mexico", "Germany", "South", "Philippines", "Columbia", "Canada", "Dominican-Republic", "France", "Guatemala", "Scotland", "Peru", "Hong", "Puerto-Rico", "Greece",
                            "Iran", "India", "Portugal", "Taiwan", "Japan", "Outlying-US(Guam-USVI-etc)", "El-Salvador", "Poland", "Cambodia", "Hungary", "Haiti", "Jamaica", "Nicaragua", "Ecuador", "England", "Ireland", "Trinadad&Tobago", "Thailand", "Honduras", "Yugoslavia"])
    age = random.randint(18, 80)
    workclass = random.choice([
        "Private",
        "Self-emp-not-inc",
        "State-gov",
        "?",
        "Self-emp-inc",
        "Local-gov",
        "Federal-gov",
        "Without-pay",
        "Never-worked"
    ])
    fnlgt = random.randint(10000, 300000)
    education = random.choice([
        "Bachelors",
        "HS-grad",
        "Some-college",
        "Doctorate",
        "5th-6th",
        "11th",
        "7th-8th",
        "Prof-school",
        "Masters",
        "12th",
        "9th",
        "Assoc-acdm",
        "Assoc-voc",
        "10th",
        "1st-4th",
        "Preschool"
    ])
    education_num = random.randint(1, 16)
    marital_status = random.choice(["Divorced", "Married-civ-spouse", "Married-spouse-absent",
                                   "Never-married", "Widowed", "Separated", "Married-AF-spouse"])
    occupation = random.choice(["Sales", "Transport-moving", "Farming-fishing", "Exec-managerial", "Other-service", "Craft-repair", "Adm-clerical",
                               "Prof-specialty", "?", "Protective-serv", "Machine-op-inspct", "Handlers-cleaners", "Tech-support", "Priv-house-serv", "Armed-Forces"])

    relationship = random.choice(["Not-in-family", "Husband", "Unmarried", "Own-child", "Wife", "Other-relative"])
    race = random.choice(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Black", "Other"])
    sex = random.choice(["Male", "Female"])
    capital_gain = random.randint(0, 99999)
    capital_loss = random.randint(0, 4356)
    hours_per_week = random.randint(1, 100)
    native_country = country

    return {
        "age": age,
        "workclass": workclass,
        "fnlgt": fnlgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }


@pytest.mark.parametrize("_", range(30))
def test_randomized_inference_prediction(_, randomized_inference_data):
    response = client.post("/inference", json=randomized_inference_data)
    assert response.status_code == 200
    assert "inference_result" in response.json()
