"""Contract tests for the Volition API."""


def test_health_reports_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["max_neurons"] == 512
    assert "VISp" in body["known_regions"]
    assert body["metrics"]["headline_auc"]["mean"] == 0.79


def test_decode_returns_probability_and_contributions(client):
    neurons = [{"rate": 0.5, "region": "VISp"}, {"rate": -1.2, "region": "CP"},
               {"rate": 0.1, "region": "MOs"}, {"rate": 2.0, "region": "VISp"}]
    r = client.post("/decode", json={"neurons": neurons})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["choice"] in ("Right (choice = +1)", "Left (choice = -1)")
    assert body["n_neurons"] == 4
    assert sum(n["attention"] for n in body["top_neurons"]) <= 1.0 + 1e-6
    assert body["region_contributions"][0]["attention"] >= body["region_contributions"][-1]["attention"]


def test_decode_rejects_empty(client):
    r = client.post("/decode", json={"neurons": []})
    assert r.status_code == 422


def test_unknown_region_is_accepted_as_padding(client):
    r = client.post("/decode", json={"neurons": [{"rate": 0.3, "region": "NOT_A_REGION"}]})
    assert r.status_code == 200


def test_assistant_grounded_answer(client):
    r = client.post("/assistant", json={"message": "How does the model work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert len(body["answer"]) > 0
    assert len(body["followups"]) == 3


def test_assistant_declines_off_topic(client):
    r = client.post("/assistant", json={"message": "What is the best pizza recipe in Rome?"})
    assert r.status_code == 200
    assert "Volition" in r.json()["answer"]


def test_assistant_handles_greeting(client):
    r = client.post("/assistant", json={"message": "hello"})
    assert r.status_code == 200
    assert len(r.json()["answer"]) > 0
