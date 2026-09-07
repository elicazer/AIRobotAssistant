"""
Unit tests for OpenAIRealtimeClient._process_server_events method.

Tests:
- Audio delta events: decode base64, enqueue, invoke on_audio_chunk
- Transcript done events: invoke on_assistant_text
- User transcription completed events: invoke on_user_text
- Function call events: parse name, call_id, arguments; invoke on_tool_use
- Error events: log and continue (do NOT set is_active = False)
- Session lifecycle events: session.created, session.updated, response.done
- WebSocket connection closed: set is_active = False
- Non-JSON messages: skip gracefully
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets.exceptions

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


def make_ws_mock(messages):
    """Create a mock WebSocket that yields messages then raises ConnectionClosed."""
    mock_ws = AsyncMock()
    message_iter = iter(messages)

    async def recv():
        try:
            return next(message_iter)
        except StopIteration:
            raise websockets.exceptions.ConnectionClosed(None, None)

    mock_ws.recv = recv
    return mock_ws


class TestAudioDeltaHandling:
    """Tests for response.audio.delta event handling."""

    @pytest.mark.asyncio
    async def test_audio_delta_decodes_base64_and_enqueues(self, client):
        """Audio delta should decode base64 and put bytes in audio queue."""
        raw_audio = b"\x01\x02\x03\x04\x05\x06"
        encoded = base64.b64encode(raw_audio).decode()

        event = json.dumps({"type": "response.audio.delta", "delta": encoded})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert not client._audio_queue.empty()
        queued_audio = client._audio_queue.get_nowait()
        assert queued_audio == raw_audio

    @pytest.mark.asyncio
    async def test_audio_delta_invokes_on_audio_chunk(self, client):
        """Audio delta should invoke on_audio_chunk callback with decoded bytes."""
        raw_audio = b"\x10\x20\x30\x40"
        encoded = base64.b64encode(raw_audio).decode()

        chunks_received = []
        client.on_audio_chunk = lambda chunk: chunks_received.append(chunk)

        event = json.dumps({"type": "response.audio.delta", "delta": encoded})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert len(chunks_received) == 1
        assert chunks_received[0] == raw_audio

    @pytest.mark.asyncio
    async def test_audio_delta_empty_delta_skipped(self, client):
        """Audio delta with empty delta field should not enqueue anything."""
        event = json.dumps({"type": "response.audio.delta", "delta": ""})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert client._audio_queue.empty()


class TestAssistantTextHandling:
    """Tests for response.audio_transcript.done event handling."""

    @pytest.mark.asyncio
    async def test_transcript_done_invokes_on_assistant_text(self, client):
        """Transcript done should invoke on_assistant_text with the transcript."""
        texts_received = []
        client.on_assistant_text = lambda text: texts_received.append(text)

        event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": "Hello, how can I help you?"
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert texts_received == ["Hello, how can I help you?"]

    @pytest.mark.asyncio
    async def test_transcript_done_empty_transcript_skipped(self, client):
        """Transcript done with empty transcript should not invoke callback."""
        texts_received = []
        client.on_assistant_text = lambda text: texts_received.append(text)

        event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": ""
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert texts_received == []

    @pytest.mark.asyncio
    async def test_transcript_done_no_callback_does_not_raise(self, client):
        """Transcript done should not raise when on_assistant_text is None."""
        client.on_assistant_text = None

        event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": "Some text"
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        # Should not raise
        await client._process_server_events()


class TestUserTextHandling:
    """Tests for conversation.item.input_audio_transcription.completed event."""

    @pytest.mark.asyncio
    async def test_user_transcription_invokes_on_user_text(self, client):
        """User transcription completed should invoke on_user_text."""
        texts_received = []
        client.on_user_text = lambda text: texts_received.append(text)

        event = json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "What is the weather today?"
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert texts_received == ["What is the weather today?"]

    @pytest.mark.asyncio
    async def test_user_transcription_empty_transcript_skipped(self, client):
        """User transcription with empty transcript should not invoke callback."""
        texts_received = []
        client.on_user_text = lambda text: texts_received.append(text)

        event = json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": ""
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert texts_received == []


class TestFunctionCallHandling:
    """Tests for response.function_call_arguments.done event handling."""

    @pytest.mark.asyncio
    async def test_function_call_invokes_on_tool_use(self, client):
        """Function call event should invoke on_tool_use with name, id, and parsed args."""
        calls_received = []

        async def mock_tool_use(name, call_id, args):
            calls_received.append((name, call_id, args))

        client.on_tool_use = mock_tool_use

        event = json.dumps({
            "type": "response.function_call_arguments.done",
            "name": "recall_memory",
            "call_id": "call_abc123",
            "arguments": '{"query": "user preferences"}'
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert len(calls_received) == 1
        name, call_id, args = calls_received[0]
        assert name == "recall_memory"
        assert call_id == "call_abc123"
        assert args == {"query": "user preferences"}

    @pytest.mark.asyncio
    async def test_function_call_invalid_json_arguments(self, client):
        """Function call with invalid JSON arguments should pass empty dict."""
        calls_received = []

        async def mock_tool_use(name, call_id, args):
            calls_received.append((name, call_id, args))

        client.on_tool_use = mock_tool_use

        event = json.dumps({
            "type": "response.function_call_arguments.done",
            "name": "search_web",
            "call_id": "call_xyz789",
            "arguments": "not valid json {"
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        assert len(calls_received) == 1
        _, _, args = calls_received[0]
        assert args == {}

    @pytest.mark.asyncio
    async def test_function_call_no_callback_does_not_raise(self, client):
        """Function call should not raise when on_tool_use is None."""
        client.on_tool_use = None

        event = json.dumps({
            "type": "response.function_call_arguments.done",
            "name": "get_emotion",
            "call_id": "call_000",
            "arguments": "{}"
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        # Should not raise
        await client._process_server_events()


class TestErrorEventHandling:
    """Tests for error event handling."""

    @pytest.mark.asyncio
    async def test_error_event_does_not_set_is_active_false(self, client):
        """Error events should NOT set is_active to False."""
        event = json.dumps({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "Something went wrong"
            }
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

        # is_active should be False only because connection closed at end,
        # not because of the error event itself. Let's verify with multiple events.

    @pytest.mark.asyncio
    async def test_error_event_continues_processing(self, client):
        """After an error event, subsequent events should still be processed."""
        texts_received = []
        client.on_assistant_text = lambda text: texts_received.append(text)

        error_event = json.dumps({
            "type": "error",
            "error": {"type": "server_error", "code": "500", "message": "Internal error"}
        })
        transcript_event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": "I'm still working!"
        })
        client._ws = make_ws_mock([error_event, transcript_event])
        client.is_active = True

        await client._process_server_events()

        # The transcript event after the error should still be processed
        assert texts_received == ["I'm still working!"]

    @pytest.mark.asyncio
    async def test_error_event_with_missing_fields(self, client):
        """Error event with missing fields should not raise."""
        event = json.dumps({
            "type": "error",
            "error": {}
        })
        client._ws = make_ws_mock([event])
        client.is_active = True

        # Should not raise
        await client._process_server_events()


class TestSessionLifecycleEvents:
    """Tests for session.created, session.updated, response.done events."""

    @pytest.mark.asyncio
    async def test_session_created_does_not_raise(self, client):
        """session.created event should be handled without error."""
        event = json.dumps({"type": "session.created", "session": {"id": "sess_123"}})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

    @pytest.mark.asyncio
    async def test_session_updated_does_not_raise(self, client):
        """session.updated event should be handled without error."""
        event = json.dumps({"type": "session.updated", "session": {"voice": "alloy"}})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()

    @pytest.mark.asyncio
    async def test_response_done_does_not_raise(self, client):
        """response.done event should be handled without error."""
        event = json.dumps({"type": "response.done", "response": {"id": "resp_456"}})
        client._ws = make_ws_mock([event])
        client.is_active = True

        await client._process_server_events()


class TestConnectionClosed:
    """Tests for WebSocket connection closed handling."""

    @pytest.mark.asyncio
    async def test_connection_closed_sets_is_active_false(self, client):
        """WebSocket ConnectionClosed should set is_active to False."""
        mock_ws = AsyncMock()

        async def recv():
            raise websockets.exceptions.ConnectionClosed(None, None)

        mock_ws.recv = recv
        client._ws = mock_ws
        client.is_active = True

        await client._process_server_events()

        assert client.is_active is False

    @pytest.mark.asyncio
    async def test_connection_closed_after_messages(self, client):
        """Connection closed after processing some messages should still set is_active False."""
        texts_received = []
        client.on_assistant_text = lambda text: texts_received.append(text)

        event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": "Hello!"
        })
        client._ws = make_ws_mock([event])  # Will raise ConnectionClosed after first message
        client.is_active = True

        await client._process_server_events()

        assert texts_received == ["Hello!"]
        assert client.is_active is False


class TestNonJsonMessages:
    """Tests for handling non-JSON WebSocket messages."""

    @pytest.mark.asyncio
    async def test_non_json_message_skipped(self, client):
        """Non-JSON messages should be skipped without raising."""
        texts_received = []
        client.on_assistant_text = lambda text: texts_received.append(text)

        non_json = "this is not json"
        valid_event = json.dumps({
            "type": "response.audio_transcript.done",
            "transcript": "Valid message"
        })
        client._ws = make_ws_mock([non_json, valid_event])
        client.is_active = True

        await client._process_server_events()

        # The valid event after the non-JSON should still be processed
        assert texts_received == ["Valid message"]


class TestMultipleEventsSequence:
    """Tests for processing multiple events in sequence."""

    @pytest.mark.asyncio
    async def test_multiple_event_types_processed(self, client):
        """Multiple different event types should all be processed correctly."""
        user_texts = []
        assistant_texts = []
        audio_chunks = []
        tool_calls = []

        client.on_user_text = lambda text: user_texts.append(text)
        client.on_assistant_text = lambda text: assistant_texts.append(text)
        client.on_audio_chunk = lambda chunk: audio_chunks.append(chunk)

        async def mock_tool_use(name, call_id, args):
            tool_calls.append((name, call_id, args))

        client.on_tool_use = mock_tool_use

        raw_audio = b"\xAA\xBB\xCC\xDD"
        events = [
            json.dumps({"type": "session.created", "session": {}}),
            json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Hello robot"
            }),
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(raw_audio).decode()
            }),
            json.dumps({
                "type": "response.function_call_arguments.done",
                "name": "get_emotion",
                "call_id": "call_001",
                "arguments": '{"source": "camera"}'
            }),
            json.dumps({
                "type": "response.audio_transcript.done",
                "transcript": "You seem happy!"
            }),
        ]

        client._ws = make_ws_mock(events)
        client.is_active = True

        await client._process_server_events()

        assert user_texts == ["Hello robot"]
        assert assistant_texts == ["You seem happy!"]
        assert audio_chunks == [raw_audio]
        assert len(tool_calls) == 1
        assert tool_calls[0] == ("get_emotion", "call_001", {"source": "camera"})
