from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_reject_unsupported_extension(monkeypatch):
    monkeypatch.setattr(main.worker, "enqueue", lambda job_id: None)
    files = {"file": ("bad.txt", b"hello", "text/plain")}
    data = {"engine": "faster_whisper"}
    response = client.post("/transcriptions", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.text


def test_reject_file_too_large(monkeypatch):
    monkeypatch.setattr(main.worker, "enqueue", lambda job_id: None)
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 5)

    files = {"file": ("audio.mp3", b"123456", "audio/mpeg")}
    data = {"engine": "faster_whisper"}
    response = client.post("/transcriptions", files=files, data=data)
    assert response.status_code == 413


def test_reject_invalid_json_payload():
    response = client.post("/transcriptions", json={"engine": "faster_whisper"})
    assert response.status_code == 400


def test_create_url_job(monkeypatch):
    monkeypatch.setattr(main.worker, "enqueue", lambda job_id: None)

    class _Resp:
        def raise_for_status(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(main.requests, "get", lambda *args, **kwargs: _Resp())
    response = client.post(
        "/transcriptions",
        json={"podcast_url": "https://example.com/audio.mp3", "engine": "faster_whisper"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"

    status = client.get(f"/transcriptions/{body['job_id']}")
    assert status.status_code == 200
    assert status.json()["engine"] == "faster_whisper"


def test_reject_unreachable_url(monkeypatch):
    monkeypatch.setattr(main.worker, "enqueue", lambda job_id: None)

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(main.requests, "get", _boom)
    response = client.post(
        "/transcriptions",
        json={"podcast_url": "https://example.com/audio.mp3", "engine": "faster_whisper"},
    )
    assert response.status_code == 400
    assert "Unreachable URL" in response.text
