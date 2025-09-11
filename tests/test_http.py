import responses

from utils.http import request


@responses.activate
def test_request_mocked() -> None:
    responses.add(responses.GET, "https://example.com", json={"ok": True}, status=200)
    resp = request("GET", "https://example.com")
    assert resp.json() == {"ok": True}
