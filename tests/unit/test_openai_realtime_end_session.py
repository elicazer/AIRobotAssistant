"""
Unit tests for OpenAIRealtimeClient.end_session method.

Tests:
- is_active set to False
- WebSocket closed gracefully
- Errors suppressed if WebSocket already closed
- Event task cancelled
- Audio queue drained
- Works when called without an active session (no-op safe)
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.openai_realtime_client import OpenAIRealtimeClient


@pytest.fixture
def client():
    """Create an OpenAIRealtimeClient with a test API key."""
    return OpenAIRealtimeClient(
        model_id="gpt-4o-realtime-preview",
        voice_id="alloy",
        system_prompt="Test prompt",
        api_key="test-api-key-123",
    )


@pytest_asyncio.fixture
async def active_client(client):
    """Create a client with a mocked active session."""
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()

    async def fake_connect(url, **kwargs):
        return mock_ws

    with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect):
        await client.start_session()

    return client


class TestEndSessionState:
    """Tests for state transitions during end_session."""

    @pytest.mark.asyncio
    async def test_end_session_sets_is_active_false(self, active_client):
        """end_session should set is_active to False."""
        assert active_client.is_active is True
        await active_client.end_session()
        assert active_client.is_active is False

    @pytest.mark.asyncio
    async def test_end_session_clears_ws_reference(self, active_client):
        """end_session should set _ws to None."""
        assert active_client._ws is not None
        await active_client.end_session()
        assert active_client._ws is None

    @pytest.mark.asyncio
    async def test_end_session_clears_event_task(self, active_client):
        """end_session should set _event_task to None."""
        assert active_client._event_task is not None
        await active_client.end_session()
        assert active_client._event_task is None


class TestEndSessionWebSocket:
    """Tests for WebSocket handling during end_session."""

    @pytest.mark.asyncio
    async def test_end_session_closes_websocket(self, active_client):
        """end_session should call close on the WebSocket."""
        ws = active_client._ws
        await active_client.end_session()
        ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_session_suppresses_ws_close_error(self, client):
        """end_session should suppress errors if WebSocket close fails."""
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("Already closed"))
        client._ws = mock_ws
        client.is_active = True

        # Should not raise
        await client.end_session()
        assert client._ws is None
        assert client.is_active is False

    @pytest.mark.asyncio
    async def test_end_session_no_ws_does_not_raise(self, client):
        """end_session should not raise when _ws is None."""
        client._ws = None
        client.is_active = True

        await client.end_session()
        assert client.is_active is False


class TestEndSessionEventTask:
    """Tests for event task cancellation during end_session."""

    @pytest.mark.asyncio
    async def test_end_session_cancels_event_task(self, active_client):
        """end_session should cancel the background event task."""
        task = active_client._event_task
        assert not task.cancelled()

        await active_client.end_session()
        # Task should be cancelled or done
        assert task.done()

    @pytest.mark.asyncio
    async def test_end_session_no_event_task_does_not_raise(self, client):
        """end_session should not raise when _event_task is None."""
        client._event_task = None
        client.is_active = True

        await client.end_session()
        assert client._event_task is None


class TestEndSessionAudioQueue:
    """Tests for audio queue draining during end_session."""

    @pytest.mark.asyncio
    async def test_end_session_drains_audio_queue(self, client):
        """end_session should drain all items from the audio queue."""
        client.is_active = True
        # Add items to the queue
        client._audio_queue.put_nowait(b"audio_chunk_1")
        client._audio_queue.put_nowait(b"audio_chunk_2")
        client._audio_queue.put_nowait(b"audio_chunk_3")

        assert client._audio_queue.qsize() == 3

        await client.end_session()
        assert client._audio_queue.empty()

    @pytest.mark.asyncio
    async def test_end_session_empty_queue_does_not_raise(self, client):
        """end_session should not raise when audio queue is already empty."""
        client.is_active = True
        assert client._audio_queue.empty()

        await client.end_session()
        assert client._audio_queue.empty()


class TestEndSessionIdempotent:
    """Tests for calling end_session multiple times or from inactive state."""

    @pytest.mark.asyncio
    async def test_end_session_when_not_active(self, client):
        """end_session should work safely when is_active is already False."""
        assert client.is_active is False
        await client.end_session()
        assert client.is_active is False

    @pytest.mark.asyncio
    async def test_end_session_called_twice(self, active_client):
        """end_session should be safe to call multiple times."""
        await active_client.end_session()
        assert active_client.is_active is False

        # Second call should not raise
        await active_client.end_session()
        assert active_client.is_active is False
