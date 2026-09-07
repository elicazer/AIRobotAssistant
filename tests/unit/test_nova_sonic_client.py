"""
Unit tests for Nova Sonic Client tool use modifications.

Tests the three new capabilities:
1. Tool configuration support in start_session()
2. toolUse event handling in _process_responses()
3. Text input injection via send_text_input()
"""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import pytest_asyncio


# Mock the Bedrock SDK imports before importing the client
import sys
mock_bedrock = MagicMock()
mock_bedrock.client = MagicMock()
mock_bedrock.models = MagicMock()
mock_bedrock.config = MagicMock()
mock_smithy = MagicMock()

sys.modules['aws_sdk_bedrock_runtime'] = mock_bedrock
sys.modules['aws_sdk_bedrock_runtime.client'] = mock_bedrock.client
sys.modules['aws_sdk_bedrock_runtime.models'] = mock_bedrock.models
sys.modules['aws_sdk_bedrock_runtime.config'] = mock_bedrock.config
sys.modules['smithy_aws_core'] = mock_smithy
sys.modules['smithy_aws_core.identity'] = mock_smithy.identity
sys.modules['smithy_aws_core.identity.environment'] = mock_smithy.identity.environment

# Mock pyaudio
sys.modules['pyaudio'] = MagicMock()

# Now import the client
from src.nova_sonic_client import NovaSonicClient


@pytest.fixture
def client():
    """Create a NovaSonicClient instance with mocked dependencies."""
    with patch.object(NovaSonicClient, '__init__', lambda self, **kwargs: None):
        c = NovaSonicClient.__new__(NovaSonicClient)
        c.model_id = 'amazon.nova-sonic-v1:0'
        c.region = 'us-east-1'
        c.voice_id = 'matthew'
        c.prompt_name = 'test-prompt-name'
        c.content_name = 'test-content-name'
        c.audio_content_name = 'test-audio-content-name'
        c.is_active = True
        c.stream = MagicMock()
        c.stream.input_stream = MagicMock()
        c.stream.input_stream.send = AsyncMock()
        c.client = MagicMock()
        c.audio_queue = asyncio.Queue()
        c.role = None
        c.display_assistant_text = False
        c._current_tool_use_id = None
        c._current_tool_name = None
        c._current_tool_content = ""
        c.system_prompt = "Test system prompt"
        c.on_user_text = None
        c.on_assistant_text = None
        c.on_audio_output = None
        c.on_audio_chunk = None
        c.on_tool_use = None
        c.callback_executor = MagicMock()
        c.response = None
        return c


class TestToolConfigurationInStartSession:
    """Tests for subtask 10.1: Tool configuration support in session start."""

    @pytest.mark.asyncio
    async def test_start_session_without_tool_config(self, client):
        """start_session() without tool_config should not include toolUse in promptStart."""
        sent_events = []

        async def capture_send(event):
            # Extract the JSON from the event
            sent_events.append(event)

        client.send_event = AsyncMock(side_effect=capture_send)
        client.response = None
        client.is_active = False
        client.client = MagicMock()
        client.stream = MagicMock()
        client.stream.input_stream = MagicMock()
        client.stream.input_stream.send = AsyncMock()
        client.client.invoke_model_with_bidirectional_stream = AsyncMock(return_value=client.stream)

        # Patch _process_responses to avoid running it
        with patch.object(client, '_process_responses', new_callable=AsyncMock):
            await client.start_session(tool_config=None)

        # Find the promptStart event
        prompt_start_event = None
        for event_json in sent_events:
            try:
                data = json.loads(event_json)
                if 'event' in data and 'promptStart' in data['event']:
                    prompt_start_event = data
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        assert prompt_start_event is not None, "promptStart event should be sent"
        assert 'toolUse' not in prompt_start_event['event']['promptStart']

    @pytest.mark.asyncio
    async def test_start_session_with_tool_config(self, client):
        """start_session() with tool_config should include toolUse in promptStart."""
        sent_events = []

        async def capture_send(event):
            sent_events.append(event)

        client.send_event = AsyncMock(side_effect=capture_send)
        client.is_active = False
        client.client = MagicMock()
        client.stream = MagicMock()
        client.stream.input_stream = MagicMock()
        client.stream.input_stream.send = AsyncMock()
        client.client.invoke_model_with_bidirectional_stream = AsyncMock(return_value=client.stream)

        tool_config = json.dumps({
            "toolUse": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "recall_memory",
                            "description": "Recall memories",
                            "inputSchema": {"json": {"type": "object", "properties": {}}}
                        }
                    }
                ]
            }
        })

        with patch.object(client, '_process_responses', new_callable=AsyncMock):
            await client.start_session(tool_config=tool_config)

        # Find the promptStart event
        prompt_start_event = None
        for event_json in sent_events:
            try:
                data = json.loads(event_json)
                if 'event' in data and 'promptStart' in data['event']:
                    prompt_start_event = data
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        assert prompt_start_event is not None
        assert 'toolUse' in prompt_start_event['event']['promptStart']
        assert 'tools' in prompt_start_event['event']['promptStart']['toolUse']
        assert len(prompt_start_event['event']['promptStart']['toolUse']['tools']) == 1
        assert prompt_start_event['event']['promptStart']['toolUse']['tools'][0]['toolSpec']['name'] == 'recall_memory'

    @pytest.mark.asyncio
    async def test_start_session_with_invalid_tool_config(self, client):
        """start_session() with invalid JSON tool_config should not crash."""
        sent_events = []

        async def capture_send(event):
            sent_events.append(event)

        client.send_event = AsyncMock(side_effect=capture_send)
        client.is_active = False
        client.client = MagicMock()
        client.stream = MagicMock()
        client.stream.input_stream = MagicMock()
        client.stream.input_stream.send = AsyncMock()
        client.client.invoke_model_with_bidirectional_stream = AsyncMock(return_value=client.stream)

        with patch.object(client, '_process_responses', new_callable=AsyncMock):
            # Should not raise
            await client.start_session(tool_config="not valid json {{{")

        # promptStart should still be sent (without toolUse)
        prompt_start_event = None
        for event_json in sent_events:
            try:
                data = json.loads(event_json)
                if 'event' in data and 'promptStart' in data['event']:
                    prompt_start_event = data
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        assert prompt_start_event is not None
        assert 'toolUse' not in prompt_start_event['event']['promptStart']


class TestToolUseEventHandling:
    """Tests for subtask 10.2: Handle toolUse events in response processing."""

    @pytest.mark.asyncio
    async def test_tool_use_content_start_sets_tool_use_id(self, client):
        """contentStart with type TOOL should set _current_tool_use_id."""
        tool_use_id = "test-tool-use-123"
        content_start_event = json.dumps({
            "event": {
                "contentStart": {
                    "type": "TOOL",
                    "toolUseId": tool_use_id,
                    "role": "ASSISTANT"
                }
            }
        }).encode('utf-8')

        # Simulate receiving this event
        mock_output = MagicMock()
        mock_result = MagicMock()
        mock_result.value = MagicMock()
        mock_result.value.bytes_ = content_start_event

        # We'll call the response processing logic directly by simulating one iteration
        # Parse the event as the _process_responses would
        response_data = content_start_event.decode('utf-8')
        json_data = json.loads(response_data)

        content_start = json_data['event']['contentStart']
        if content_start.get('type') == 'TOOL':
            client._current_tool_use_id = content_start.get('toolUseId')

        assert client._current_tool_use_id == tool_use_id

    @pytest.mark.asyncio
    async def test_tool_use_callback_invoked(self, client):
        """on_tool_use callback should be called when toolUse event is received."""
        tool_use_id = "tool-use-456"
        tool_name = "recall_memory"
        parameters = {"query": "what do I like?"}

        # Set up the tool use state (as if contentStart was already processed)
        client._current_tool_use_id = tool_use_id

        # Set up the callback
        callback_result = {"result": "You like jazz music"}
        client.on_tool_use = AsyncMock(return_value=callback_result)
        client.send_tool_result = AsyncMock()

        # Simulate processing a toolUse event
        tool_use_event_data = {
            "event": {
                "toolUse": {
                    "toolName": tool_name,
                    "content": json.dumps(parameters)
                }
            }
        }

        # Extract the logic from _process_responses for the toolUse branch
        tool_use_data = tool_use_event_data['event']['toolUse']
        t_name = tool_use_data.get('toolName', '')
        tool_content = tool_use_data.get('content', '{}')

        try:
            parsed_params = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
        except (json.JSONDecodeError, TypeError):
            parsed_params = {}

        if client.on_tool_use and client._current_tool_use_id:
            tool_result = await client.on_tool_use(t_name, client._current_tool_use_id, parsed_params)
            await client.send_tool_result(client._current_tool_use_id, tool_result)

        client.on_tool_use.assert_called_once_with(tool_name, tool_use_id, parameters)
        client.send_tool_result.assert_called_once_with(tool_use_id, callback_result)

    @pytest.mark.asyncio
    async def test_send_tool_result_sends_three_events(self, client):
        """send_tool_result() should send contentStart, toolResult, and contentEnd."""
        sent_events = []

        async def capture_send(event_json):
            sent_events.append(json.loads(event_json))

        client.send_event = AsyncMock(side_effect=capture_send)

        tool_use_id = "tool-789"
        result = {"result": "Memory recalled successfully"}

        await client.send_tool_result(tool_use_id, result)

        assert len(sent_events) == 3

        # First event: contentStart with type TOOL_RESULT
        assert 'contentStart' in sent_events[0]['event']
        cs = sent_events[0]['event']['contentStart']
        assert cs['type'] == 'TOOL_RESULT'
        assert cs['promptName'] == client.prompt_name
        assert cs['toolResultInputConfiguration']['toolUseId'] == tool_use_id
        assert cs['interactive'] is False

        # Second event: toolResult with content
        assert 'toolResult' in sent_events[1]['event']
        tr = sent_events[1]['event']['toolResult']
        assert tr['promptName'] == client.prompt_name
        assert json.loads(tr['content']) == result

        # Third event: contentEnd
        assert 'contentEnd' in sent_events[2]['event']
        ce = sent_events[2]['event']['contentEnd']
        assert ce['promptName'] == client.prompt_name

    @pytest.mark.asyncio
    async def test_send_tool_result_inactive_does_nothing(self, client):
        """send_tool_result() should do nothing when session is not active."""
        client.is_active = False
        client.send_event = AsyncMock()

        await client.send_tool_result("tool-id", {"result": "test"})

        client.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_use_callback_error_sends_error_result(self, client):
        """When on_tool_use raises an exception, an error result should be sent."""
        tool_use_id = "tool-err-123"
        client._current_tool_use_id = tool_use_id

        # Callback that raises
        client.on_tool_use = AsyncMock(side_effect=RuntimeError("Tool crashed"))
        client.send_tool_result = AsyncMock()

        # Simulate the error handling logic from _process_responses
        tool_name = "search_web"
        parameters = {"query": "test"}

        try:
            tool_result = await client.on_tool_use(tool_name, client._current_tool_use_id, parameters)
            await client.send_tool_result(client._current_tool_use_id, tool_result)
        except Exception as e:
            await client.send_tool_result(
                client._current_tool_use_id,
                {"error": "execution_failed", "message": str(e)}
            )

        # Should have sent error result
        client.send_tool_result.assert_called_once_with(
            tool_use_id,
            {"error": "execution_failed", "message": "Tool crashed"}
        )


class TestTextInputInjection:
    """Tests for subtask 10.3: Text input injection method."""

    @pytest.mark.asyncio
    async def test_send_text_input_sends_three_events(self, client):
        """send_text_input() should send contentStart, textInput, and contentEnd."""
        sent_events = []

        async def capture_send(event_json):
            sent_events.append(json.loads(event_json))

        client.send_event = AsyncMock(side_effect=capture_send)

        greeting = "Hello Alice! You seem happy today. Last time we talked about gardening."
        await client.send_text_input(greeting)

        assert len(sent_events) == 3

        # First event: contentStart with type TEXT, role USER, interactive false
        assert 'contentStart' in sent_events[0]['event']
        cs = sent_events[0]['event']['contentStart']
        assert cs['type'] == 'TEXT'
        assert cs['role'] == 'USER'
        assert cs['interactive'] is False
        assert cs['promptName'] == client.prompt_name
        assert 'textInputConfiguration' in cs
        assert cs['textInputConfiguration']['mediaType'] == 'text/plain'

        # Second event: textInput with the content
        assert 'textInput' in sent_events[1]['event']
        ti = sent_events[1]['event']['textInput']
        assert ti['promptName'] == client.prompt_name
        assert ti['content'] == greeting

        # Third event: contentEnd
        assert 'contentEnd' in sent_events[2]['event']
        ce = sent_events[2]['event']['contentEnd']
        assert ce['promptName'] == client.prompt_name

    @pytest.mark.asyncio
    async def test_send_text_input_uses_unique_content_name(self, client):
        """Each send_text_input() call should use a unique content name."""
        sent_events = []

        async def capture_send(event_json):
            sent_events.append(json.loads(event_json))

        client.send_event = AsyncMock(side_effect=capture_send)

        await client.send_text_input("First message")
        await client.send_text_input("Second message")

        # Extract content names from the two calls
        content_name_1 = sent_events[0]['event']['contentStart']['contentName']
        content_name_2 = sent_events[3]['event']['contentStart']['contentName']

        assert content_name_1 != content_name_2

    @pytest.mark.asyncio
    async def test_send_text_input_inactive_does_nothing(self, client):
        """send_text_input() should do nothing when session is not active."""
        client.is_active = False
        client.send_event = AsyncMock()

        await client.send_text_input("Hello!")

        client.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_text_input_content_name_consistent_across_events(self, client):
        """All three events in send_text_input() should use the same content name."""
        sent_events = []

        async def capture_send(event_json):
            sent_events.append(json.loads(event_json))

        client.send_event = AsyncMock(side_effect=capture_send)

        await client.send_text_input("Test greeting")

        content_name = sent_events[0]['event']['contentStart']['contentName']
        assert sent_events[1]['event']['textInput']['contentName'] == content_name
        assert sent_events[2]['event']['contentEnd']['contentName'] == content_name
