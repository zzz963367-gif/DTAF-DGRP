"""Assemble the GDSC input archive and extract it beside this script."""
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.digest()


def unpack(folder):
    parts = json.loads((folder / "parts.json").read_text())
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        staging = Path(temp)
        archive = staging / "inputs.zip"
        with archive.open("wb") as combined:
            for name, expected in sorted(parts.items()):
                path = folder / name
                if digest(path).hex() != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
                with path.open("rb") as part:
                    shutil.copyfileobj(part, combined)
        extracted = staging / "inputs"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        files = [p for p in extracted.rglob("*") if p.is_file()]
        for source in files:
            target = folder / source.relative_to(extracted)
            if target.exists() and digest(target) != digest(source):
                raise FileExistsError(f"Existing input differs: {target}")
        for source in files:
            target = folder / source.relative_to(extracted)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source.replace(target)
    print(f"Ready: {len(files)} input files in {folder}")


if __name__ == "__main__":
    unpack(Path(__file__).resolve().parent)
