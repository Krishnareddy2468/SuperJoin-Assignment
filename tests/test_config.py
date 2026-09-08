from pathlib import Path

from factlayer.config import DEFAULT_LLM_MODEL, load_config


def clear_factlayer_environment(monkeypatch) -> None:
    for name in (
        "FACTLAYER_DATA_DIR",
        "FACTLAYER_LLM_MODEL",
        "FACTLAYER_NO_LLM",
        "GEMINI_API_KEY",
        "FACTLAYER_LLM_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_config_uses_explicit_data_directory(tmp_path, monkeypatch) -> None:
    clear_factlayer_environment(monkeypatch)

    config = load_config(data_dir=tmp_path, env_file=None)

    assert config.db_path == tmp_path / "factlayer.db"
    assert config.upload_dir == tmp_path / "uploads"
    assert config.cache_dir == tmp_path / "cache"
    assert config.llm_model == DEFAULT_LLM_MODEL
    assert config.llm_enabled is False
    assert config.why_llm_disabled() == "no GEMINI_API_KEY found in environment or .env"


def test_explicit_offline_mode_wins_over_api_key(tmp_path, monkeypatch) -> None:
    clear_factlayer_environment(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")

    config = load_config(data_dir=tmp_path, no_llm=True, env_file=None)

    assert config.api_key == "test-key-not-a-secret"
    assert config.llm_enabled is False
    assert config.why_llm_disabled() == "FACTLAYER_NO_LLM is set"


def test_environment_overrides_defaults(tmp_path, monkeypatch) -> None:
    clear_factlayer_environment(monkeypatch)
    data_dir = tmp_path / "runtime"
    monkeypatch.setenv("FACTLAYER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("FACTLAYER_LLM_MODEL", "example-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")

    config = load_config(env_file=None)

    assert config.db_path == data_dir / "factlayer.db"
    assert config.llm_model == "example-model"
    assert config.llm_enabled is True
    assert config.why_llm_disabled() is None


def test_ensure_dirs_creates_runtime_directories(tmp_path, monkeypatch) -> None:
    clear_factlayer_environment(monkeypatch)
    config = load_config(data_dir=tmp_path / "new-data", no_llm=True, env_file=None)

    config.ensure_dirs()

    assert config.db_path.parent.is_dir()
    assert config.upload_dir.is_dir()
    assert config.cache_dir.is_dir()
    assert all(isinstance(path, Path) for path in (config.db_path, config.upload_dir, config.cache_dir))
