"""API tests with FastAPI's TestClient (no server, no CSVs needed).
They use the committed models/model.joblib."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD_ROOM = {
    "distance_to_campus_km": 4.5,
    "room_area_sqft": 350,
    "sharing": 2,
    "meals_per_day": 2,
    "building_age_yr": 15,
    "wifi_mbps": 100,
    "furnishing": "semi",
    "locality": "Sector West",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_good_room():
    response = client.post("/predict", json=GOOD_ROOM)
    assert response.status_code == 200
    rent = response.json()["prediction"]
    assert 1000 < rent < 100000          # a sensible monthly rent in rupees


def test_room_area_above_650_is_rejected():
    response = client.post("/predict", json={**GOOD_ROOM, "room_area_sqft": 651})
    assert response.status_code == 422


def test_room_area_650_is_allowed():
    response = client.post("/predict", json={**GOOD_ROOM, "room_area_sqft": 650})
    assert response.status_code == 200


def test_negative_wifi_is_rejected():
    response = client.post("/predict", json={**GOOD_ROOM, "wifi_mbps": -1})
    assert response.status_code == 422


def test_missing_field_is_rejected():
    room = dict(GOOD_ROOM)
    del room["locality"]
    assert client.post("/predict", json=room).status_code == 422

