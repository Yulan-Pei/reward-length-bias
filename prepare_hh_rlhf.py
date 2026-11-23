"""Download and verify the HH-RLHF helpful-base files used by the experiment."""
import hashlib
import urllib.request
from pathlib import Path


FILES = {
    "train.jsonl.gz": (
        "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz",
        "518a5bf288456fc9f3b7c980c54116fba0c52f274d3f4d344675d83e4058f6f4",
    ),
    "test.jsonl.gz": (
        "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/test.jsonl.gz",
        "8be3fc1a13b27901631696f2be6f184c799f1baeaf145f53ac5db24960adc37b",
    ),
}
OUTPUT = Path(__file__).resolve().parent / "data" / "helpful-base"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_hash) in FILES.items():
        target = OUTPUT / name
        if target.exists() and sha256(target) == expected_hash:
            print(f"verified: {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        print(f"downloading: {url}")
        urllib.request.urlretrieve(url, temporary)
        actual_hash = sha256(temporary)
        if actual_hash != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual_hash}")
        temporary.replace(target)
        print(f"verified: {target}")


if __name__ == "__main__":
    main()

