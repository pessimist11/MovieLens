"""Start the packaged API twice and verify it over a private loopback socket."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
HTTP = build_opener(ProxyHandler({}))


def read(base, path):
    with HTTP.open(base + path, timeout=3) as response:
        body = response.read().decode()
        return json.loads(body) if response.headers.get_content_type() == "application/json" else body


def cache_state():
    return {
        path.relative_to(ROOT): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (ROOT / "artifacts" / "models").rglob("*")
        if path.is_file()
    }


def check(base):
    health = read(base, "/health")
    assert health["ready"], health.get("load_error")
    assert "MovieLens 推荐系统" in read(base, "/")
    user = read(base, "/users?limit=1")["user_ids"][0]
    queries = [read(base, f"/recommend/{user}?k=10"), read(base, "/recommend/-1?k=10")]
    personal, cold = queries
    assert not personal["cold_start"] and cold["cold_start"]
    assert all(len({item["movie_id"] for item in row["items"]}) == 10 for row in queries)
    assert any(item["source"] != "popular" for item in personal["items"])
    assert all(item["source"] == "popular" for item in cold["items"])
    assert all(item["rank_score"] is None for item in cold["items"])
    movie = personal["items"][0]["movie_id"]
    assert read(base, f"/movies/{movie}")["movie_id"] == movie
    metrics = read(base, "/metrics")
    assert metrics["dataset"] == health["dataset"]
    assert metrics["ablation"] and 0 <= metrics["test"]["recall@k"] <= 1
    return health, queries, len(metrics["ablation"])


def run_once():
    with socket.socket() as listener, tempfile.TemporaryFile() as log:
        listener.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{listener.getsockname()[1]}"
        fd = listener.fileno()
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        start = monotonic()
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "movielens.api.main:app", "--fd", str(fd)],
            cwd=ROOT, env=env, pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            while server.poll() is None and monotonic() - start < 30:
                try:
                    health = read(base, "/health")
                except (URLError, TimeoutError):
                    sleep(0.1)
                    continue
                assert health["ready"], health.get("load_error")
                ready_seconds = monotonic() - start
                result = check(base)
                print(f"ready in {ready_seconds:.2f}s; users={health['n_users']}; "
                      f"movies={health['n_items']}; ablations={result[2]}", flush=True)
                return result
            raise RuntimeError("API did not start within 30 seconds")
        except Exception:
            log.seek(0)
            print(log.read().decode(errors="replace"), file=sys.stderr)
            raise
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    before = cache_state()
    assert before, "Packaged model cache is missing"
    first = run_once()
    second = run_once()
    assert first[0] == second[0]
    assert [row["items"] for row in first[1]] == [row["items"] for row in second[1]]
    assert cache_state() == before, "Model cache changed during service startup"
    print("PASS: homepage, readiness, users, recommendations, cold start, movie, metrics and restart")


if __name__ == "__main__":
    main()
