"""Static local publication-content checks; ownership clearance is a separate gate."""

from pathlib import Path
import re
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", ".audit", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
RESTRICTED = {".pdf", ".docx", ".ipynb", ".pkl", ".joblib", ".pt", ".pth", ".zip"}
PRIVATE = re.compile(r"/(?:home|Users)/[A-Za-z0-9]|/mnt/[a-z]/Users/")
SECRET = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
)


def main():
    problems = []
    checked = 0
    # Audit Git's prospective publication set, including untracked nonignored files.
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    )
    names = sorted(set(n for n in result.stdout.decode().split("\0") if n))
    for name in names:
        path = ROOT / name
        rel = path.relative_to(ROOT)
        if not path.is_file():
            problems.append(f"Missing candidate file: {rel}")
            continue
        checked += 1
        if path.suffix.lower() in RESTRICTED or path.stat().st_size > 2_000_000:
            problems.append(f"Restricted type or oversized artifact: {rel}")
        if "data" in rel.parts or "runs" in rel.parts:
            problems.append(f"Generated data/run artifact included: {rel}")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            problems.append(f"Unexpected binary file: {rel}")
            continue
        if PRIVATE.search(text) or SECRET.search(text):
            problems.append(f"Private path or credential pattern: {rel}")
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#")[0]
                if not (path.parent / target).exists():
                    problems.append(f"Broken Markdown target: {rel} -> {target}")
    if problems:
        print("\n".join(problems))
        return 1
    print(
        f"PASS: {checked} files checked; no restricted artifacts, private paths, credential patterns or broken local Markdown links"
    )
    print("Publication remains blocked by the separate ownership and dataset-use review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
