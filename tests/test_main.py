"""Тесты ai-service: эндпоинт /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

import main


def test_health(client: TestClient):
    """GET /health возвращает 200 и статус ok с информацией о моделях."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "standard_horizons" in body
    assert "buffet_horizons" in body
    assert "schools" in body
    assert "model_version" in body
