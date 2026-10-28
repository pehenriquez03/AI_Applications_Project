"""API key intake for the app: where the key lives, how it is tested, saved and removed.

Rules: the key is stored only in <repo>/.env (gitignored, mode 600) or in the running process;
it is never logged, never echoed back beyond its last four characters, and never written
anywhere else. The label says whose key it is: "class" (the key shared in class, the default)
or "personal".
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Optional

ENV_NAME = "ANTHROPIC_API_KEY"
LABEL_NAME = "ANTHROPIC_KEY_LABEL"
LABELS = ("class", "personal")
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def _parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m and not line.lstrip().startswith("#"):
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


def read_env_file(root: Path) -> dict:
    f = Path(root) / ".env"
    try:
        return _parse(f.read_text())
    except OSError:
        return {}


def last4(key: str) -> str:
    return key[-4:] if key else ""


def valid_format(key: str) -> bool:
    """Anthropic keys start with sk-ant- and are long; nothing else is checked here."""
    key = (key or "").strip()
    return key.startswith("sk-ant-") and len(key) >= 40 and " " not in key


def status(root: Path, environ: Optional[dict] = None) -> dict:
    """{found, source: 'process'|'file'|None, label, last4, env_path}. Never returns the key."""
    environ = os.environ if environ is None else environ
    env_path = Path(root) / ".env"
    k = environ.get(ENV_NAME, "").strip()
    if k:
        label = environ.get(LABEL_NAME) or read_env_file(root).get(LABEL_NAME) or "unlabeled"
        return dict(found=True, source="process", label=label, last4=last4(k), env_path=str(env_path))
    d = read_env_file(root)
    k = d.get(ENV_NAME, "")
    if k:
        return dict(found=True, source="file", label=d.get(LABEL_NAME, "unlabeled"), last4=last4(k), env_path=str(env_path))
    return dict(found=False, source=None, label=None, last4="", env_path=str(env_path))


def test_key(key: str, model: str, timeout: float = 30.0) -> tuple[bool, str]:
    """One minimal call (max_tokens=5) with the given key. Returns (ok, message); the message never has the key."""
    if not valid_format(key):
        return False, "That does not look like an Anthropic key (they start with sk-ant-)."
    try:
        import anthropic
    except ImportError:
        return False, "The anthropic package is not installed in this environment."
    try:
        client = anthropic.Anthropic(api_key=key.strip(), max_retries=1, timeout=timeout)
        r = client.messages.create(model=model, max_tokens=5,
                                   messages=[{"role": "user", "content": "Reply with the word ready."}])
        used = r.usage.input_tokens + r.usage.output_tokens
        return True, f"Connected to {model} · key ending …{last4(key.strip())} · test call used {used} tokens."
    except anthropic.AuthenticationError:
        return False, "Anthropic rejected this key (401). Check that it was copied whole and is still active."
    except anthropic.PermissionDeniedError:
        return False, "This key is not allowed to use that model (403). Ask for access or pick another model."
    except anthropic.NotFoundError:
        return False, f"The model {model} is not available to this key (404)."
    except anthropic.RateLimitError:
        return True, f"Key accepted but rate-limited right now (429) · key ending …{last4(key.strip())}."
    except anthropic.APIConnectionError:
        return False, "Could not reach api.anthropic.com. Check the network and try again."
    except Exception as e:  # noqa: BLE001
        return False, f"Test call failed: {type(e).__name__}."


def save_key(root: Path, key: str, label: str, environ: Optional[dict] = None) -> Path:
    """Write ANTHROPIC_API_KEY and its label to <root>/.env (mode 600), keeping other lines; set the process env."""
    environ = os.environ if environ is None else environ
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    key = key.strip()
    if not valid_format(key):
        raise ValueError("invalid key format")
    f = Path(root) / ".env"
    kept = []
    if f.exists():
        for line in f.read_text().splitlines():
            m = _LINE.match(line)
            if m and m.group(1) in (ENV_NAME, LABEL_NAME):
                continue
            kept.append(line)
    lines = [ln for ln in kept if ln.strip()] + [f"{ENV_NAME}={key}", f"{LABEL_NAME}={label}"]
    f.write_text("\n".join(lines) + "\n")
    os.chmod(f, stat.S_IRUSR | stat.S_IWUSR)
    environ[ENV_NAME] = key
    environ[LABEL_NAME] = label
    return f


def use_for_session(key: str, label: str, environ: Optional[dict] = None) -> None:
    """Process environment only: nothing touches the disk."""
    environ = os.environ if environ is None else environ
    if not valid_format(key):
        raise ValueError("invalid key format")
    environ[ENV_NAME] = key.strip()
    environ[LABEL_NAME] = label


def remove_key(root: Path, environ: Optional[dict] = None) -> None:
    """Delete the key and label from .env and from the process environment."""
    environ = os.environ if environ is None else environ
    f = Path(root) / ".env"
    if f.exists():
        kept = [ln for ln in f.read_text().splitlines()
                if not (_LINE.match(ln) and _LINE.match(ln).group(1) in (ENV_NAME, LABEL_NAME))]
        f.write_text("\n".join(ln for ln in kept if ln.strip()) + ("\n" if any(ln.strip() for ln in kept) else ""))
    environ.pop(ENV_NAME, None)
    environ.pop(LABEL_NAME, None)
