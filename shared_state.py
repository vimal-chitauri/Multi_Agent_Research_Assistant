import threading
from datetime import datetime

_ps_lock      = threading.Lock()
_proceed_event = threading.Event()

_ps: dict = {
    "running":          False,
    "stop_requested":   False,
    "step":             0,
    "sub_text":         "",
    "sub_pct":          0,
    "logs":             [],
    "agent_logs":       {},
    "trend_result":     None,
    "content_result":   None,
    "image_result":     None,
    "video_result":     None,
    "verify_result":    None,
    "publish_result":   None,
    "approved_posts":   {},
    "error":            None,
    "waiting_proceed":  False,
    "waiting_at_step":  0,
    "selected_trends":  None,
    "retry_r1":         None,
    "retry_r2":         None,
    "retry_r3":         None,
    "retry_r4":         None,
}


def _log(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"{ts}  {msg}"
    with _ps_lock:
        _ps["logs"].append(line)
        if len(_ps["logs"]) > 400:
            _ps["logs"] = _ps["logs"][-400:]
    print(line)


def _alog(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"{ts}  {msg}"
    with _ps_lock:
        _ps["logs"].append(line)
        if len(_ps["logs"]) > 400:
            _ps["logs"] = _ps["logs"][-400:]
        aid = _ps.get("step", 0)
        if aid > 0:
            if aid not in _ps["agent_logs"]:
                _ps["agent_logs"][aid] = []
            _ps["agent_logs"][aid].append(line)
            if len(_ps["agent_logs"][aid]) > 150:
                _ps["agent_logs"][aid] = _ps["agent_logs"][aid][-150:]
    print(line)


def _sub(text: str, pct: int) -> None:
    with _ps_lock:
        _ps["sub_text"] = text[:100]
        _ps["sub_pct"]  = max(0, min(100, pct))


def _stopped() -> bool:
    with _ps_lock:
        return _ps.get("stop_requested", False)


def _wait_proceed(step: int) -> bool:
    _proceed_event.clear()
    with _ps_lock:
        _ps["waiting_proceed"] = True
        _ps["waiting_at_step"] = step
    _log(f"⏸  Manual review — step {step} done. Waiting for your approval to continue…")
    _proceed_event.wait(timeout=600)
    with _ps_lock:
        _ps["waiting_proceed"] = False
        _ps["waiting_at_step"] = 0
    return _proceed_event.is_set()


def _abort(msg: str = "Stopped by user") -> None:
    _log(f"⛔ {msg}")
    with _ps_lock:
        _ps["running"]         = False
        _ps["stop_requested"]  = False
        _ps["step"]            = 0
        _ps["sub_text"]        = "Pipeline stopped"
        _ps["sub_pct"]         = 0
        _ps["waiting_proceed"] = False
        _ps["waiting_at_step"] = 0
    _proceed_event.set()
