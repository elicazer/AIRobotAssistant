"""
Unit tests for OpenAIRealtimeClient.start_session method.

Tests:
- API key resolution (parameter, env var, settings file)
- WebSocket connection establishment with correct URL and headers
- session.update event sent with correct structure
- is_active set to True after successful connection
- Background task started for _process_server_events
- ConnectionError raised on connection failure/timeout
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

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


@pytest.fixture
def client_no_key():
    """Create an OpenAIRealtimeClient without an API key."""
    return OpenAIRealtimeClient(
        model_id="gpt-4o-realtime-preview",
        voice_id="alloy",
        system_prompt="Test prompt",
        api_key=None,
    )


class TestApiKeyResolution:
    """Tests for API key resolution priority."""

    def test_api_key_from_constructor_parameter(self, client):
        """API key passed to constructor should be used first."""
        key = client._resolve_api_key()
        assert key == "test-api-key-123"

    def test_api_key_from_env_var(self, client_no_key):
        """OPENAI_API_KEY env var should be used when no constructor key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            key = client_no_key._resolve_api_key()
            assert key == "env-key-456"

    def test_api_key_from_settings_file(self, client_no_key):
        """Settings file key should be used when no constructor key or env var."""
        settings_data = json.dumps({"openai_api_key": "settings-key-789"})
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)
            with patch("builtins.open", mock_open(read_data=settings_data)):
                key = client_no_key._resolve_api_key()
                assert key == "settings-key-789"

    def test_api_key_missing_raises_connection_error(self, client_no_key):
        """ConnectionError should be raised when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with patch("builtins.open", side_effect=OSError("File not found")):
                with pytest.raises(ConnectionError, match="No OpenAI API key found"):
                    client_no_key._resolve_api_key()

    def test_constructor_key_takes_priority_over_env(self, client):
        """Constructor key should take priority over env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-should-not-use"}):
            key = client._resolve_api_key()
            assert key == "test-api-key-123"

    def test_env_key_takes_priority_over_settings(self, client_no_key):
        """Env var should take priority over settings file."""
        settings_data = json.dumps({"openai_api_key": "settings-key-should-not-use"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-priority"}):
            with patch("builtins.open", mock_open(read_data=settings_data)):
                key = client_no_key._resolve_api_key()
                assert key == "env-key-priority"


class TestStartSessionConnection:
    """Tests for WebSocket connection in start_session."""

    @pytest.mark.asyncio
    async def test_start_session_connects_with_correct_url(self, client):
        """start_session should connect to the correct WebSocket URL."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        with patch("src.openai_realtime_client.websockets.connect", new_callable=lambda: lambda *a, **kw: mock_ws) as mock_connect:
            # Make websockets.connect return a coroutine that resolves to mock_ws
            async def fake_connect(url, **kwargs):
                return mock_ws

            with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect) as mock_conn:
                await client.start_session()

                mock_conn.assert_called_once()
                call_args = mock_conn.call_args
                assert call_args[0][0] == "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
                assert call_args[1]["additional_headers"]["Authorization"] == "Bearer test-api-key-123"
                assert call_args[1]["additional_headers"]["OpenAI-Beta"] == "realtime=v1"

        # Cleanup
        if client._event_task:
            client._event_task.cancel()
            try:
                await client._event_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_session_sends_session_update(self, client):
        """start_session should send a session.update event with correct structure."""
        mock_ws = AsyncMock()
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        mock_ws.send = capture_send

        async def fake_connect(url, **kwargs):
            return mock_ws

        with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect):
            await client.start_session()

        assert len(sent_messages) == 1
        event = json.loads(sent_messages[0])

        assert event["type"] == "session.update"
        session = event["session"]
        assert session["instructions"] == "Test prompt"
        assert session["voice"] == "alloy"
        assert session["input_audio_format"] == "pcm16"
        assert session["output_audio_format"] == "pcm16"
        assert session["input_audio_transcription"] == {"model": "whisper-1"}
        assert session["turn_detection"] == {"type": "server_vad"}
        assert session["tools"] == []

        # Cleanup
        if client._event_task:
            client._event_task.cancel()
            try:
                await client._event_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_session_with_tool_config(self, client):
        """start_session with tool_config should include converted tools."""
        mock_ws = AsyncMock()
        sent_messages = []

        async def capture_send(msg):
            sent_messages.append(msg)

        mock_ws.send = capture_send

        async def fake_connect(url, **kwargs):
            return mock_ws

        tool_config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "recall_memory",
                            "description": "Recall stored memories",
                            "inputSchema": {
                                "json": {"type": "object", "properties": {"query": {"type": "string"}}}
                            }
                        }
                    }
                ]
            }
        })

        with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect):
            await client.start_session(tool_config=tool_config)

        event = json.loads(sent_messages[0])
        tools = event["session"]["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "recall_memory"
        assert tools[0]["description"] == "Recall stored memories"
        assert tools[0]["parameters"] == {"type": "object", "properties": {"query": {"type": "string"}}}

        # Cleanup
        if client._event_task:
            client._event_task.cancel()
            try:
                await client._event_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_session_sets_is_active(self, client):
        """start_session should set is_active to True."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        async def fake_connect(url, **kwargs):
            return mock_ws

        assert client.is_active is False

        with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect):
            await client.start_session()

        assert client.is_active is True

        # Cleanup
        if client._event_task:
            client._event_task.cancel()
            try:
                await client._event_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_session_starts_event_task(self, client):
        """start_session should start a background task for _process_server_events."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        async def fake_connect(url, **kwargs):
            return mock_ws

        with patch("src.openai_realtime_client.websockets.connect", side_effect=fake_connect):
            await client.start_session()

        assert client._event_task is not None
        assert isinstance(client._event_task, asyncio.Task)

        # Cleanup
        client._event_task.cancel()
        try:
            await client._event_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_start_session_timeout_raises_connection_error(self, client):
        """start_session should raise ConnectionError on timeout."""

        async def slow_connect(url, **kwargs):
            await asyncio.sleep(20)  # Longer than 10s timeout
            return AsyncMock()

        with patch("src.openai_realtime_client.websockets.connect", side_effect=slow_connect):
            with pytest.raises(ConnectionError, match="timed out"):
                await client.start_session()

        assert client.is_active is False

    @pytest.mark.asyncio
    async def test_start_session_connection_refused_raises_connection_error(self, client):
        """start_session should raise ConnectionError when connection is refused."""

        async def failing_connect(url, **kwargs):
            raise OSError("Connection refused")

        with patch("src.openai_realtime_client.websockets.connect", side_effect=failing_connect):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await client.start_session()

        assert client.is_active is False

    @pytest.mark.asyncio
    async def test_start_session_no_api_key_raises_connection_error(self):
        """start_session should raise ConnectionError when no API key is available."""
        client = OpenAIRealtimeClient(api_key=None)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with patch("builtins.open", side_effect=OSError("File not found")):
                with pytest.raises(ConnectionError, match="No OpenAI API key found"):
                    await client.start_session()
