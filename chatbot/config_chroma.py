
from __future__ import annotations
from typing import Any
import chromadb


try:  
        from chromadb import PersistentClient  
        _HAS_PERSISTENT = True
except Exception:  
        PersistentClient = None  
        _HAS_PERSISTENT = False

try:
        
        from chromadb.config import Settings 
except Exception:  
        Settings = None  

DEFAULT_DB_DIR = "db"


def get_chroma_client(persist_directory: str = DEFAULT_DB_DIR, *, anonymized_telemetry: bool = False):
        """Return an appropriate Chroma client for current library version.

        1. Prefer new PersistentClient(path=...).
        2. Fallback to legacy chromadb.Client(Settings(...)).
        """
        if _HAS_PERSISTENT and PersistentClient is not None:
                return PersistentClient(path=persist_directory, settings=chromadb.Settings(anonymized_telemetry=anonymized_telemetry))
        # Legacy path
        if Settings is None:
               
                return chromadb.Client()
        return chromadb.Client(
                Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persist_directory,
                        anonymized_telemetry=anonymized_telemetry,
                )
        )


# For legacy imports elsewhere expecting CHROMA_SETTINGS we expose a minimal
# object; LangChain will ignore unknown attrs when passing a concrete client.
class _LegacySettingsShim:
        anonymized_telemetry = False


CHROMA_SETTINGS = _LegacySettingsShim()  # backward compatibility export

__all__ = ["get_chroma_client", "CHROMA_SETTINGS", "DEFAULT_DB_DIR"]
