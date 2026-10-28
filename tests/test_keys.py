"""Key intake: format check, .env read/write/remove, status; no network."""
import stat

from research_engine.app import keys as K


def test_valid_format():
    assert K.valid_format("sk-ant-" + "a" * 40)
    assert not K.valid_format("sk-proj-" + "a" * 40)
    assert not K.valid_format("sk-ant-short")
    assert not K.valid_format("")


def test_status_prefers_process_then_file(tmp_path):
    assert K.status(tmp_path, {})["found"] is False
    (tmp_path / ".env").write_text("OTHER=1\nANTHROPIC_API_KEY='sk-ant-" + "b" * 40 + "'\nANTHROPIC_KEY_LABEL=class\n")
    s = K.status(tmp_path, {})
    assert s["found"] and s["source"] == "file" and s["label"] == "class" and s["last4"] == "bbbb"
    s = K.status(tmp_path, {"ANTHROPIC_API_KEY": "sk-ant-" + "c" * 40, "ANTHROPIC_KEY_LABEL": "personal"})
    assert s["source"] == "process" and s["label"] == "personal" and s["last4"] == "cccc"


def test_save_keeps_other_lines_sets_mode_and_env(tmp_path):
    (tmp_path / ".env").write_text("OTHER=1\nANTHROPIC_API_KEY=sk-ant-old\n")
    env = {}
    f = K.save_key(tmp_path, "sk-ant-" + "d" * 40, "personal", env)
    text = f.read_text()
    assert "OTHER=1" in text and "sk-ant-old" not in text and "ANTHROPIC_KEY_LABEL=personal" in text
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert env["ANTHROPIC_API_KEY"].endswith("dddd") and env["ANTHROPIC_KEY_LABEL"] == "personal"
    K.remove_key(tmp_path, env)
    assert "ANTHROPIC_API_KEY" not in f.read_text() and "OTHER=1" in f.read_text() and env == {}


def test_save_rejects_bad_input(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        K.save_key(tmp_path, "nope", "class", {})
    with pytest.raises(ValueError):
        K.save_key(tmp_path, "sk-ant-" + "e" * 40, "team", {})


def test_test_key_rejects_format_without_network():
    ok, msg = K.test_key("bad", "claude-sonnet-5-5")
    assert not ok and "sk-ant-" in msg
