import hashlib
import json
import os
from pathlib import Path
from schemas import Artifact

# Directory path inside the sandbox
STORE_DIR = Path(__file__).parent / "sandbox" / "artifacts"

# 40 KB threshold as specified in requirements
ARTIFACT_THRESHOLD_BYTES = 40 * 1024


def _parse_id(artifact_id: str) -> str:
    """Validate format and extract the hash prefix from the artifact handle."""
    if not artifact_id or not artifact_id.startswith("art:"):
        raise ValueError(f"Invalid artifact ID format: {artifact_id}. Must start with 'art:'")
    return artifact_id[4:]


def exists(artifact_id: str) -> bool:
    """Check if the artifact files (both binary and metadata) exist on disk."""
    if not artifact_id or not artifact_id.startswith("art:"):
        return False
    hash_prefix = artifact_id[4:]
    bin_path = STORE_DIR / f"{hash_prefix}.bin"
    json_path = STORE_DIR / f"{hash_prefix}.json"
    return bin_path.exists() and json_path.exists()


def put(blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
    """Store raw bytes and metadata, returning a content-addressable ID (art:<sha256-prefix>)."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Content-addressable hashing (first 16 hex chars)
    sha256 = hashlib.sha256(blob).hexdigest()
    hash_prefix = sha256[:16]
    artifact_id = f"art:{hash_prefix}"
    
    bin_path = STORE_DIR / f"{hash_prefix}.bin"
    json_path = STORE_DIR / f"{hash_prefix}.json"
    
    # Save binary data if it doesn't already exist (deduplication)
    if not bin_path.exists():
        bin_path.write_bytes(blob)
        
    # Always save/refresh metadata
    art = Artifact(
        id=artifact_id,
        content_type=content_type,
        size_bytes=len(blob),
        source=source,
        descriptor=descriptor
    )
    json_path.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    
    return artifact_id


def get_bytes(artifact_id: str) -> bytes:
    """Fetch raw binary blob from the artifact store."""
    hash_prefix = _parse_id(artifact_id)
    bin_path = STORE_DIR / f"{hash_prefix}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Artifact binary file not found for ID: {artifact_id}")
    return bin_path.read_bytes()


def get_meta(artifact_id: str) -> Artifact:
    """Fetch metadata of the artifact."""
    hash_prefix = _parse_id(artifact_id)
    json_path = STORE_DIR / f"{hash_prefix}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Artifact metadata file not found for ID: {artifact_id}")
    return Artifact.model_validate_json(json_path.read_text(encoding="utf-8"))


def clear():
    """Wipe all stored artifacts from the disk directory."""
    if not STORE_DIR.exists():
        return
    for item in STORE_DIR.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except OSError:
                pass
