import uuid

import requests


def test_set_preview_success(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    """PUT /api/assets/{id}/preview sets the preview and returns the full Asset."""
    main_data = make_asset_bytes("main_asset.png", 2048)
    preview_data = make_asset_bytes("preview_asset.png", 1024)

    main = asset_factory("main_asset.png", ["input", "unit-tests"], {}, main_data)
    preview = asset_factory("preview_asset.png", ["input", "unit-tests"], {}, preview_data)

    r = http.put(
        f"{api_base}/api/assets/{main['id']}/preview",
        json={"preview_id": preview["id"]},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["id"] == main["id"]
    assert body["preview_id"] == preview["id"]


def test_clear_preview_success(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    """DELETE /api/assets/{id}/preview clears the preview and returns 204."""
    main_data = make_asset_bytes("main_clear.png", 2048)
    preview_data = make_asset_bytes("prev_clear.png", 1024)

    main = asset_factory("main_clear.png", ["input", "unit-tests"], {}, main_data)
    preview = asset_factory("prev_clear.png", ["input", "unit-tests"], {}, preview_data)

    # First set a preview
    r_set = http.put(
        f"{api_base}/api/assets/{main['id']}/preview",
        json={"preview_id": preview["id"]},
        timeout=120,
    )
    assert r_set.status_code == 200

    # Now clear it
    r_del = http.delete(f"{api_base}/api/assets/{main['id']}/preview", timeout=120)
    assert r_del.status_code == 204

    # Verify preview_id is cleared
    r_get = http.get(f"{api_base}/api/assets/{main['id']}", timeout=120)
    detail = r_get.json()
    assert r_get.status_code == 200, detail
    assert detail.get("preview_id") is None


def test_set_preview_asset_not_found(http: requests.Session, api_base: str):
    """PUT /api/assets/{id}/preview returns 404 for non-existent asset ID."""
    fake_id = str(uuid.uuid4())
    r = http.put(
        f"{api_base}/api/assets/{fake_id}/preview",
        json={"preview_id": str(uuid.uuid4())},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"


def test_clear_preview_asset_not_found(http: requests.Session, api_base: str):
    """DELETE /api/assets/{id}/preview returns 404 for non-existent asset ID."""
    fake_id = str(uuid.uuid4())
    r = http.delete(f"{api_base}/api/assets/{fake_id}/preview", timeout=120)
    body = r.json()
    assert r.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"


def test_set_preview_missing_body(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    """PUT /api/assets/{id}/preview returns 400 when preview_id is not provided."""
    main_data = make_asset_bytes("main_nobody.png", 2048)
    main = asset_factory("main_nobody.png", ["input", "unit-tests"], {}, main_data)

    r = http.put(
        f"{api_base}/api/assets/{main['id']}/preview",
        json={},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


def test_set_preview_invalid_preview_id(
    http: requests.Session, api_base: str, asset_factory, make_asset_bytes
):
    """PUT /api/assets/{id}/preview returns 404 when preview_id references non-existent asset."""
    main_data = make_asset_bytes("main_badprev.png", 2048)
    main = asset_factory("main_badprev.png", ["input", "unit-tests"], {}, main_data)

    fake_preview_id = str(uuid.uuid4())
    r = http.put(
        f"{api_base}/api/assets/{main['id']}/preview",
        json={"preview_id": fake_preview_id},
        timeout=120,
    )
    body = r.json()
    assert r.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"
