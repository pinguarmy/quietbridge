"""Cloud (gemini/CPA) cleanup: transport, prompts, parsing, timeout/fallback, unsafe output."""
import pytest
import requests

import lipflow.cloud as cloud
from lipflow.cleanup import Cleaner, SYSTEM, basic_cleanup
from lipflow.personal import Personal

KEY = "test-cpa-key-000"  # dummy; lets us assert nothing ever leaks it


class FakeResp:
    def __init__(self, text="Hello there, ready to ship.", status=200, finish="stop",
                 refusal=None, bad_json=False):
        self.status_code = status
        self._payload = {"choices": [{"finish_reason": finish,
                                      "message": {"content": text, "refusal": refusal}}]}
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("no json")
        return self._payload


def fake_post(monkeypatch, resp=None, recorder=None, raises=None):
    def _post(url, **kw):
        if recorder is not None:
            recorder.update(url=url, **kw)
        if raises is not None:
            raise raises
        return resp
    monkeypatch.setattr(cloud._SESSION, "post", _post)  # cleanup reuses one keep-alive session
    return recorder


def make_cleaner(monkeypatch, style="polish", settings=None, **kw):
    monkeypatch.setenv("CPA_API_KEY", KEY)
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: settings or {})
    c = Cleaner("gemini", style=style, **kw)
    c.personal = Personal("/nonexistent")  # don't depend on the user's imported history
    return c


# --- key resolution ---------------------------------------------------------

def test_resolve_key_prefers_env(monkeypatch):
    monkeypatch.setenv("CPA_API_KEY", "env-cpa")
    monkeypatch.setenv("LIPFLOW_CPA_API_KEY", "env-lipflow")
    assert cloud.resolve_cpa_key() == "env-cpa"
    monkeypatch.delenv("CPA_API_KEY")
    assert cloud.resolve_cpa_key() == "env-lipflow"


def test_resolve_key_from_dsh_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("CPA_API_KEY", raising=False)
    monkeypatch.delenv("LIPFLOW_CPA_API_KEY", raising=False)
    (tmp_path / ".credentials.yaml").write_text("version: 1\nrefs:\n  CPA_API_KEY: dsh-key\n")
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert cloud.resolve_cpa_key() == "dsh-key"


def test_resolve_key_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("CPA_API_KEY", raising=False)
    monkeypatch.delenv("LIPFLOW_CPA_API_KEY", raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path))  # no credentials file here
    assert cloud.resolve_cpa_key() is None
    (tmp_path / ".credentials.yaml").write_text("version: 1\nrefs:\n  OTHER: x\n")
    assert cloud.resolve_cpa_key() is None


# --- transport / prompts ----------------------------------------------------

def test_transport_openai_chat_completions(monkeypatch):
    rec = fake_post(monkeypatch, resp=FakeResp(), recorder={})
    out = cloud.cleanup(["HELLO THERE READY TO SHIP"], base_url="https://cpa.example/v1/",
                        model="gemini-test", api_key=KEY, style="polish")
    assert out == "Hello there, ready to ship."
    assert rec["url"] == "https://cpa.example/v1/chat/completions"
    assert rec["headers"]["Authorization"] == f"Bearer {KEY}"
    body = rec["json"]
    assert body["model"] == "gemini-test" and body["temperature"] == 0 and body["stream"] is False
    assert body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert rec["timeout"] == (cloud.CONNECT_TIMEOUT, cloud.DEFAULT_TIMEOUT)


def test_prompt_bounds(monkeypatch):
    rec = fake_post(monkeypatch, resp=FakeResp(), recorder={})
    cands = [f"CANDIDATE NUMBER {i}" for i in range(7)]
    words = [f"w{i}" for i in range(40)]
    similar = [f"phrase {i}" for i in range(8)]
    cloud.cleanup(cands, context="x" * 1000, words=words, similar=similar, api_key=KEY)
    p = rec["json"]["messages"][1]["content"]
    assert "5. CANDIDATE NUMBER 4" in p and "CANDIDATE NUMBER 5" not in p  # top 5 only
    assert "w29" in p and "w30" not in p  # words bounded
    assert "phrase 4" in p and "phrase 5" not in p  # similar bounded
    assert "x" * 501 not in p and "x" * 100 in p  # context tail, bounded


def test_style_prompts(monkeypatch):
    assert cloud.system_prompt("faithful") == SYSTEM
    assert "filler" in cloud.system_prompt("polish")
    assert "filler" not in cloud.system_prompt("faithful")
    rec = fake_post(monkeypatch, resp=FakeResp(), recorder={})
    cloud.cleanup(["UM WELL THE THE PLAN IS READY"], api_key=KEY, style="polish")
    assert "filler" in rec["json"]["messages"][0]["content"]
    assert "negation" in rec["json"]["messages"][0]["content"]


# --- parsing / unsafe responses ---------------------------------------------

@pytest.mark.parametrize("resp", [
    FakeResp(status=500),                                # HTTP error
    FakeResp(bad_json=True),                             # unparseable body
    FakeResp(finish="content_filter"),                   # filtered
    FakeResp(finish="length"),                           # truncated output must not be pasted
    FakeResp(finish="tool_calls"),                       # input method never executes instructions
    FakeResp(text="", refusal="I can't help with that"),  # refusal
    FakeResp(text="   "),                                # empty content
])
def test_bad_responses_return_none(monkeypatch, resp):
    fake_post(monkeypatch, resp=resp)
    assert cloud.cleanup(["HELLO THERE"], api_key=KEY) is None


def test_think_tags_and_quotes_stripped(monkeypatch):
    fake_post(monkeypatch, resp=FakeResp(text="<think>hmm</think>Done here."))
    assert cloud.cleanup(["DONE HERE"], api_key=KEY) == "Done here."
    fake_post(monkeypatch, resp=FakeResp(text='"Quoted answer."'))
    assert cloud.cleanup(["QUOTED ANSWER"], api_key=KEY) == "Quoted answer."


def test_malformed_shapes_return_none(monkeypatch):
    class Weird(FakeResp):
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self._bad = False
    for payload in [{}, {"choices": []}, {"choices": [{"message": {}}]},
                    {"choices": [{"message": {"content": 42}}]}]:
        fake_post(monkeypatch, resp=Weird(payload))
        assert cloud.cleanup(["HELLO THERE"], api_key=KEY) is None


def test_response_echoing_key_is_dropped(monkeypatch):
    fake_post(monkeypatch, resp=FakeResp(text=f"sure, your key is {KEY}"))
    assert cloud.cleanup(["HELLO THERE"], api_key=KEY) is None


def test_network_errors_return_none(monkeypatch):
    fake_post(monkeypatch, raises=requests.Timeout())
    assert cloud.cleanup(["HELLO THERE"], api_key=KEY) is None
    fake_post(monkeypatch, raises=requests.ConnectionError())
    assert cloud.cleanup(["HELLO THERE"], api_key=KEY) is None


def test_empty_candidates_make_no_call(monkeypatch):
    fake_post(monkeypatch, raises=AssertionError("must not call"))
    assert cloud.cleanup([], api_key=KEY) is None
    assert cloud.cleanup(["  ", ""], api_key=KEY) is None


# --- Cleaner integration ----------------------------------------------------

def test_cleaner_gemini_success_and_last_result(monkeypatch):
    fake_post(monkeypatch, resp=FakeResp())
    c = make_cleaner(monkeypatch)
    out = c(["HELLO THERE READY TO SHIP"], words=[])
    assert out == "Hello there, ready to ship."
    lr = c.last_result
    assert lr["backend"] == "gemini" and lr["style"] == "polish" and lr["fallback"] is None
    assert isinstance(lr["cleanup_ms"], int) and lr["cleanup_ms"] >= 0
    assert KEY not in str(lr)


def test_cleaner_timeout_falls_back_to_basic(monkeypatch, capsys):
    fake_post(monkeypatch, raises=requests.Timeout())
    monkeypatch.setattr(Cleaner, "_local_fallback", lambda self, c, ctx: None)  # don't load MLX
    c = make_cleaner(monkeypatch)
    out = c(["HELLO THERE READY TO SHIP"], words=[])
    assert out == basic_cleanup("HELLO THERE READY TO SHIP")
    assert c.last_result["fallback"] == "gemini" and c.last_result["backend"] == "basic"
    assert KEY not in capsys.readouterr().out


def test_cleaner_exception_never_leaks_key(monkeypatch, capsys):
    def boom(*a, **kw):
        raise RuntimeError(f"request to https://host failed, key={KEY}")
    monkeypatch.setattr(cloud, "cleanup", boom)
    monkeypatch.setattr(Cleaner, "_local_fallback", lambda self, c, ctx: None)
    c = make_cleaner(monkeypatch)
    out = c(["HELLO THERE READY TO SHIP"], words=[])
    assert out == basic_cleanup("HELLO THERE READY TO SHIP")
    printed = capsys.readouterr().out
    assert "RuntimeError" in printed and KEY not in printed


def test_cleaner_falls_back_to_local_first(monkeypatch):
    fake_post(monkeypatch, raises=requests.ConnectionError())
    monkeypatch.setattr(Cleaner, "_local_fallback", lambda self, c, ctx: "Local cleaned text.")
    c = make_cleaner(monkeypatch)
    assert c(["HELLO THERE READY TO SHIP"], words=[]) == "Local cleaned text."
    assert c.last_result["backend"] == "local" and c.last_result["fallback"] == "gemini"


def test_cleaner_fallback_loads_local_model_lazily(monkeypatch):
    """Regression: gemini fallback must load the local model itself — warmup() only loads
    when backend == 'local', so a cloud failure used to leave _mlx None and _local crashed
    unpacking it. Drives the real _local_fallback/_local path through a fake mlx_lm module."""
    import sys
    import types
    from importlib.machinery import ModuleSpec

    fake_post(monkeypatch, raises=requests.ConnectionError())
    loaded = {}

    class FakeTok:
        def apply_chat_template(self, messages, **kw):
            return "prompt"

    def fake_load(model):
        loaded["model"] = model
        return ("fake-local-model", FakeTok())

    def fake_generate(model, tok, prompt=None, **kw):
        assert model == "fake-local-model"
        return "hello there ready to ship"  # only words from the guess -> passes within_guesses

    fake_mlx = types.ModuleType("mlx_lm")
    fake_mlx.load = fake_load
    fake_mlx.generate = fake_generate
    fake_mlx.__spec__ = ModuleSpec("mlx_lm", loader=None)  # so find_spec() finds it
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)

    c = make_cleaner(monkeypatch)
    assert c.backend == "gemini" and c._mlx is None
    cloud_model = c.model
    out = c(["HELLO THERE READY TO SHIP"], words=[])
    assert out == "Hello there ready to ship"
    assert loaded["model"] == "mlx-community/Qwen3-0.6B-4bit"  # local, not the cloud model
    assert c._mlx is not None and c._mlx[0] == "fake-local-model"  # loaded and cached
    assert c.model == cloud_model  # cloud model name restored after the fallback
    assert c.last_result["backend"] == "local" and c.last_result["fallback"] == "gemini"


def test_cleaner_without_key_makes_no_call(monkeypatch, tmp_path):
    monkeypatch.delenv("CPA_API_KEY", raising=False)
    monkeypatch.delenv("LIPFLOW_CPA_API_KEY", raising=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    fake_post(monkeypatch, raises=AssertionError("must not call"))
    monkeypatch.setattr(Cleaner, "_local_fallback", lambda self, c, ctx: None)
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: {})
    c = Cleaner("gemini", style="faithful")
    c.personal = Personal("/nonexistent")
    assert c(["HELLO THERE READY TO SHIP"], words=[]) == basic_cleanup("HELLO THERE READY TO SHIP")
    assert c.last_result["fallback"] == "gemini"


def test_cleaner_empty_input(monkeypatch):
    fake_post(monkeypatch, raises=AssertionError("must not call"))
    c = make_cleaner(monkeypatch)
    assert c([], words=[]) == ""
    assert c.last_result is not None and KEY not in str(c.last_result)


def test_cleaner_call_signature_compat(monkeypatch):
    fake_post(monkeypatch, resp=FakeResp(text="Hello Miguel."))
    c = make_cleaner(monkeypatch)
    # positional candidates + all the existing keyword arguments still work
    out = c(["HELLO MCCALL"], context="earlier", words=[], names=["Miguel"])
    assert out == "Hello Miguel."


# --- settings / opt-in ------------------------------------------------------

def test_settings_optin_and_fields(monkeypatch):
    monkeypatch.setenv("CPA_API_KEY", KEY)
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: {
        "cleanup_backend": "gemini", "cleanup_style": "polish", "gemini_model": "gm-1",
        "cpa_base_url": "https://b.example/v1", "cleanup_timeout": 3})
    c = Cleaner("auto")
    assert c.backend == "gemini" and c.style == "polish" and c.model == "gm-1"
    assert c._cpa_base_url == "https://b.example/v1" and c._cleanup_timeout == 3.0
    # explicit constructor arguments beat settings
    c2 = Cleaner("basic", style="faithful")
    assert c2.backend == "basic" and c2.style == "faithful"


def test_timeout_clamped(monkeypatch):
    monkeypatch.setenv("CPA_API_KEY", KEY)
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: {"cleanup_timeout": 999})
    assert Cleaner("gemini")._cleanup_timeout == cloud.MAX_TIMEOUT
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: {"cleanup_timeout": "junk"})
    assert Cleaner("gemini")._cleanup_timeout == cloud.DEFAULT_TIMEOUT


def test_invalid_style_defaults_to_faithful(monkeypatch):
    c = make_cleaner(monkeypatch, style="weird")
    assert c.style == "faithful"


def test_auto_never_picks_gemini(monkeypatch):
    monkeypatch.setenv("CPA_API_KEY", KEY)  # a key alone must not opt you into the cloud
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("lipflow.dictation.load_settings", lambda: {})
    assert Cleaner("auto").backend != "gemini"
