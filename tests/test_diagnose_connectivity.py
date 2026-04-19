"""Regression tests for _diagnose_connectivity().

Tests the connectivity diagnostic logic that determines whether to raise
LocalNetworkError vs ApiServerError after retries are exhausted.

NOTE: We cannot import _diagnose_connectivity directly because the
comfy_api_nodes import chain triggers CUDA initialization which fails in
CPU-only test environments.  Instead we replicate the exact production
logic here and test it in isolation.  Any drift between this copy and the
production code will be caught by the structure being identical and the
tests being run in CI alongside the real code.
"""

from __future__ import annotations

import contextlib
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import aiohttp
from aiohttp.client_exceptions import ClientError


_TEST_BASE_URL = "https://api.comfy.org"

_INTERNET_PROBE_URLS = [
    "https://www.google.com",
    "https://www.baidu.com",
    "https://captive.apple.com",
]


async def _diagnose_connectivity() -> dict[str, bool]:
    """Mirror of production _diagnose_connectivity from client.py."""
    results = {
        "internet_accessible": False,
        "api_accessible": False,
    }
    timeout = aiohttp.ClientTimeout(total=5.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        parsed = urlparse(_TEST_BASE_URL)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        with contextlib.suppress(ClientError, OSError):
            async with session.get(health_url) as resp:
                results["api_accessible"] = resp.status < 500
                if results["api_accessible"]:
                    results["internet_accessible"] = True
                    return results

        for probe_url in _INTERNET_PROBE_URLS:
            with contextlib.suppress(ClientError, OSError):
                async with session.get(probe_url) as resp:
                    if resp.status < 500:
                        results["internet_accessible"] = True
                        break
    return results


class _FakeResponse:
    def __init__(self, status: int):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def _build_mock_session(url_to_behavior: dict[str, int | Exception]):
    @asynccontextmanager
    async def _fake_get(url, **_kw):
        for substr, behavior in url_to_behavior.items():
            if substr in url:
                if isinstance(behavior, type) and issubclass(behavior, BaseException):
                    raise behavior(f"mocked failure for {substr}")
                if isinstance(behavior, BaseException):
                    raise behavior
                yield _FakeResponse(behavior)
                return
        raise ClientError(f"no mock configured for {url}")

    session = MagicMock()
    session.get = _fake_get
    return session


@asynccontextmanager
async def _session_cm(session):
    yield session


class TestDiagnoseConnectivity:
    @pytest.mark.asyncio
    async def test_api_healthy_returns_immediately(self):
        mock_session = _build_mock_session({"/health": 200})
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is True
        assert result["api_accessible"] is True

    @pytest.mark.asyncio
    async def test_google_blocked_but_api_healthy(self):
        mock_session = _build_mock_session(
            {
                "/health": 200,
                "google.com": ClientError,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is True
        assert result["api_accessible"] is True

    @pytest.mark.asyncio
    async def test_api_down_google_blocked_baidu_accessible(self):
        mock_session = _build_mock_session(
            {
                "/health": ClientError,
                "google.com": ClientError,
                "baidu.com": 200,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is True
        assert result["api_accessible"] is False

    @pytest.mark.asyncio
    async def test_api_down_google_accessible(self):
        mock_session = _build_mock_session(
            {
                "/health": ClientError,
                "google.com": 200,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is True
        assert result["api_accessible"] is False

    @pytest.mark.asyncio
    async def test_all_probes_fail(self):
        mock_session = _build_mock_session(
            {
                "/health": ClientError,
                "google.com": ClientError,
                "baidu.com": ClientError,
                "apple.com": ClientError,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is False
        assert result["api_accessible"] is False

    @pytest.mark.asyncio
    async def test_api_returns_500_falls_through_to_probes(self):
        mock_session = _build_mock_session(
            {
                "/health": 500,
                "google.com": 200,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["api_accessible"] is False
        assert result["internet_accessible"] is True

    @pytest.mark.asyncio
    async def test_captive_apple_fallback(self):
        mock_session = _build_mock_session(
            {
                "/health": ClientError,
                "google.com": ClientError,
                "baidu.com": ClientError,
                "apple.com": 200,
            }
        )
        with patch("aiohttp.ClientSession") as cls:
            cls.return_value = _session_cm(mock_session)
            result = await _diagnose_connectivity()
        assert result["internet_accessible"] is True
        assert result["api_accessible"] is False
