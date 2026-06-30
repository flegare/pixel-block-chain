#!/usr/bin/env python3
"""
Audit the repository for material that should not be published.

Run from the repository root:
    python tools/sanitize_for_publication_git.py

The script is intentionally conservative. It reports:
  - CRITICAL findings that should block publication.
  - WARNING findings that require human review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CRITICAL_EXTENSIONS = {
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".env",
    ".db",
    ".sqlite",
}

CRITICAL_FILENAMES = {
    "server.key",
    "id_rsa",
    "id_ed25519",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".html",
    ".js",
    ".json",
    ".latex",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CRITICAL_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)\b(password|secret)\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"OneDrive\\", re.IGNORECASE),
    re.compile(r"\b[A-Z]:\\dev\\", re.IGNORECASE),
]

WARNING_PATTERNS = [
    re.compile(r"(?i)\bconfidential\b"),
    re.compile(r"(?i)\bprovisional patent\b"),
    re.compile(r"(?i)\bdo not (share|publish|commit)\b"),
]

LARGE_FILE_WARNING_MB = 25


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        with path.open("rb") as fh:
            sample = fh.read(2048)
        return b"\x00" not in sample
    except OSError:
        return False


def scan_text(path: Path, root: Path):
    rel = path.relative_to(root)
    findings = []
    if rel.as_posix() == "tools/sanitize_for_publication_git.py":
        return findings
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        findings.append(("WARNING", rel, f"could not read text: {exc}"))
        return findings

    for idx, line in enumerate(text.splitlines(), start=1):
        for pattern in CRITICAL_PATTERNS:
            if pattern.search(line):
                findings.append(("CRITICAL", rel, f"line {idx}: {pattern.pattern}"))
        for pattern in WARNING_PATTERNS:
            if pattern.search(line):
                findings.append(("WARNING", rel, f"line {idx}: {pattern.pattern}"))
    return findings


def audit(root: Path):
    findings = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        name = path.name.lower()
        suffix = path.suffix.lower()

        if name in CRITICAL_FILENAMES or suffix in CRITICAL_EXTENSIONS:
            findings.append(("CRITICAL", rel, "credential-like filename or extension"))

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb >= LARGE_FILE_WARNING_MB:
            findings.append(("WARNING", rel, f"large file: {size_mb:.1f} MB"))

        if is_text_file(path):
            findings.extend(scan_text(path, root))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to audit. Defaults to the current directory.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    findings = audit(root)
    critical = [f for f in findings if f[0] == "CRITICAL"]
    warnings = [f for f in findings if f[0] == "WARNING"]

    print(f"Publication audit root: {root}")
    print(f"Critical findings: {len(critical)}")
    print(f"Warnings: {len(warnings)}")

    for level, rel, detail in findings:
        print(f"{level}: {rel}: {detail}")

    if critical:
        print("\nPublication audit failed: resolve CRITICAL findings before pushing.")
        return 1

    print("\nPublication audit passed: no critical findings found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
