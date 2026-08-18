from __future__ import annotations

from app.config.settings import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.embedding_dimension == 512
    assert settings.api_key_enabled is False


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("VERIFICATION_SIMILARITY_THRESHOLD", "0.5")
    monkeypatch.setenv("API_KEY_ENABLED", "true")

    settings = Settings(_env_file=None)
    assert settings.verification_similarity_threshold == 0.5
    assert settings.api_key_enabled is True
