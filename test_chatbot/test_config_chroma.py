import importlib.util
import sys
import types
import uuid
from pathlib import Path
import pytest

CONFIG_CHROMA_PATH = Path("/Users/senghok/Documents/Chat_bot_Climate/chatbot/config_chroma.py")

# Client: save the collections and embeddings in RAM only: for quick experiment
# Persistent: save to the disk: for stocking histories 

def _build_fake_chromadb(* , has_persistent: bool, has_settings_config: bool, top_level_settings: bool): 
    
    mod = types.ModuleType("chromadb")
    
    class FakeSettings:
        def __init__(self, chroma_db_impl=None, persist_directory=None, anonymized_telemetry=None):
            self.chroma_db_impl = chroma_db_impl
            self.persist_directory = persist_directory
            self.anonymized_telemetry = anonymized_telemetry

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakePersistentClient:
        def __init__(self, *, path=None, settings=None):
            self.path = path
            self.settings = settings


    mod.Client = FakeClient
    if has_persistent:
        mod.PersistenClient = FakePersistentClient
    # about setting
    if top_level_settings:
        mod.Settings = FakeSettings
    
   
    cfg_mod = None
    if has_settings_config:
        cfg_mod = types.ModuleType("chromadb.config")
        cfg_mod.Settings = FakeSettings

    return mod, cfg_mod, FakeClient, FakePersistentClient, FakeSettings


def _load_config_chroma(monkeypatch, chromadb_mod, chromadb_config_mod=None):
    # Inject/override chromadb modules before import
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_mod)
    if chromadb_config_mod is not None:
        monkeypatch.setitem(sys.modules, "chromadb.config", chromadb_config_mod)
    else:
        # Ensure no stray module interferes when we want the import to fail
        sys.modules.pop("chromadb.config", None)

    # Load the target module from file with a unique name to avoid caching
    mod_name = f"config_chroma_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, str(CONFIG_CHROMA_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_new_api_uses_persistent_client_and_top_level_settings(monkeypatch):
    chromadb_mod, chromadb_cfg_mod, _, FakePersistentClient, FakeSettings = _build_fake_chromadb(
        has_persistent=True, has_settings_config=False, top_level_settings=True
    )
    cfg = _load_config_chroma(monkeypatch, chromadb_mod, chromadb_cfg_mod)

    client = cfg.get_chroma_client("/tmp/mydb", anonymized_telemetry=True)
    assert isinstance(client, FakePersistentClient)
    assert client.path == "/tmp/mydb"
    assert isinstance(client.settings, FakeSettings)
    assert client.settings.anonymized_telemetry is True


def test_legacy_without_settings_calls_plain_client(monkeypatch):
    chromadb_mod, chromadb_cfg_mod, FakeClient, _, _ = _build_fake_chromadb(
        has_persistent=False, has_settings_config=False, top_level_settings=False
    )
    cfg = _load_config_chroma(monkeypatch, chromadb_mod, chromadb_cfg_mod)

    client = cfg.get_chroma_client("/any", anonymized_telemetry=False)
    assert isinstance(client, FakeClient)
    # Legacy path without Settings should pass no args to Client()
    assert client.args == ()
    assert client.kwargs == {}


def test_legacy_with_settings_passes_expected_arguments(monkeypatch):
    chromadb_mod, chromadb_cfg_mod, FakeClient, _, FakeSettings = _build_fake_chromadb(
        has_persistent=False, has_settings_config=True, top_level_settings=False
    )
    cfg = _load_config_chroma(monkeypatch, chromadb_mod, chromadb_cfg_mod)

    client = cfg.get_chroma_client("db_dir_123", anonymized_telemetry=True)
    assert isinstance(client, FakeClient)
    # Should be called as Client(Settings(...))
    assert len(client.args) == 1
    settings_arg = client.args[0]
    assert isinstance(settings_arg, FakeSettings)
    assert settings_arg.chroma_db_impl == "duckdb+parquet"
    assert settings_arg.persist_directory == "db_dir_123"
    assert settings_arg.anonymized_telemetry is True
    assert client.kwargs == {}


def test_exports_chroma_settings_shim(monkeypatch):
    chromadb_mod, chromadb_cfg_mod, *_ = _build_fake_chromadb(
        has_persistent=False, has_settings_config=False, top_level_settings=False
    )
    cfg = _load_config_chroma(monkeypatch, chromadb_mod, chromadb_cfg_mod)
    assert hasattr(cfg, "CHROMA_SETTINGS")
    assert getattr(cfg.CHROMA_SETTINGS, "anonymized_telemetry", None) is False
