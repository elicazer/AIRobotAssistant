"""
OpenAI Realtime API client for voice interaction via WebSocket.

This module provides an alternative voice backend alongside NovaSonicClient,
connecting to the OpenAI Realtime API for bidirectional audio streaming and
function calling (tool use).
"""

import asyncio
import json
import logging
import os
from typing import Callable, Optional

import websockets

logger = logging.getLogger(__name__)


def convert_tools_nova_to_openai(tool_config_json: str) -> list[dict]:
    """Convert Nova Sonic toolUse config to OpenAI function definitions.

    Input (Nova Sonic format):
    {
        "toolUse": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "recall_memory",
                        "description": "...",
                        "inputSchema": {
                            "json": { "type": "object", "properties": {...} }
                        }
                    }
                }
            ]
        }
    }

    Output (OpenAI format):
    [
        {
            "type": "function",
            "name": "recall_memory",
            "description": "...",
            "parameters": { "type": "object", "properties": {...} }
        }
    ]

    Args:
        tool_config_json: JSON string containing Nova Sonic tool configuration.

    Returns:
        List of OpenAI function definition dicts. Returns empty list if
        input is invalid JSON, missing expected structure, or has no tools.
    """
    try:
        config = json.loads(tool_config_json)
    except (json.JSONDecodeError, TypeError):
        return []

    try:
        tools = config["toolUse"]["tools"]
    except (KeyError, TypeError):
        return []

    if not isinstance(tools, list):
        return []

    result = []
    for tool in tools:
        try:
            tool_spec = tool["toolSpec"]
            name = tool_spec["name"]
        except (KeyError, TypeError):
            # Skip tools without required toolSpec or name
            continue

        description = tool_spec.get("description", "")
        input_schema = tool_spec.get("inputSchema", {})
        if isinstance(input_schema, dict):
            parameters = input_schema.get("json", {})
        else:
            parameters = {}

        result.append({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        })

    return result


def resample_16k_to_24k(pcm16_bytes: bytes) -> bytes:
    """Resample PCM16 mono audio from 16000 Hz to 24000 Hz.

    Uses linear interpolation (ratio 3:2). Each pair of input samples
    produces 3 output samples.

    Args:
        pcm16_bytes: Raw PCM16 mono audio bytes at 16000 Hz.
            Must be even length (2 bytes per sample).

    Returns:
        Resampled PCM16 mono audio bytes at 24000 Hz.
    """
    import array

    # Edge case: empty input
    if not pcm16_bytes:
        return b""

    samples = array.array("h")
    samples.frombytes(pcm16_bytes)

    # Edge case: single sample
    if len(samples) <= 1:
        return pcm16_bytes

    # 16kHz → 24kHz is a 3:2 ratio
    output = array.array("h")
    for i in range(len(samples) - 1):
        s0 = samples[i]
        s1 = samples[i + 1]
        output.append(s0)
        output.append(int(s0 + (s1 - s0) / 3))
        output.append(int(s0 + 2 * (s1 - s0) / 3))
    # Last sample
    output.append(samples[-1])

    return output.tobytes()


class OpenAIRealtimeClient:
    """Client for OpenAI Realtime API voice interaction via WebSocket.

    Exposes the same callback interface as NovaSonicClient so that
    voice_assistant_server.py can use either client interchangeably.

    Callbacks:
        on_user_text: Called when user speech is transcribed
        on_assistant_text: Called when assistant response text is available
        on_audio_output: Called when complete audio response is ready
        on_audio_chunk: Called with real-time audio chunks for playback/visualization
        on_tool_use: Called when the model invokes a tool (async callable)
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-realtime-preview",
        voice_id: str = "alloy",
        system_prompt: Optional[str] = None,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        # Configuration
        self.model_id = model_id
        self.voice_id = voice_id
        self.system_prompt = system_prompt or (
            "You are a friendly robot assistant. Keep your responses short and natural, "
            "generally two or three sentences. You are speaking out loud, so be conversational."
        )
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self._api_key = api_key

        # Public state
        self.is_active: bool = False

        # Callbacks (same signature as NovaSonicClient)
        self.on_user_text: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_audio_output: Optional[Callable[[bytes], None]] = None
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_tool_use: Optional[Callable] = None  # async (name, id, params) -> dict

        # Internal state
        self._ws = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._event_task: Optional[asyncio.Task] = None

        # Measured speaker output-buffer latency (ms). Visemes are computed at
        # audio-write time, but PortAudio holds the audio in its output buffer
        # before it is audible; delaying the mouth by this much keeps the avatar
        # in sync with what is actually heard. Populated in play_audio().
        self.output_latency_ms = 0

    def _safe_audio_callback(self, audio_data: bytes) -> None:
        """Safely execute on_audio_chunk callback without raising."""
        try:
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_data)
        except Exception:
            # Silently ignore callback errors to prevent audio disruption
            pass

    def _resolve_api_key(self) -> str:
        """Resolve API key from parameter, env var, or settings file (in that order).

        Returns:
            The resolved API key string.

        Raises:
            ConnectionError: If no API key can be found from any source.
        """
        # 1. Check constructor parameter
        if self._api_key:
            return self._api_key

        # 2. Check environment variable
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        # 3. Check settings file
        try:
            settings_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "voice_assistant_settings.json",
            )
            with open(settings_path, "r") as f:
                settings = json.load(f)
            settings_key = settings.get("openai_api_key", "")
            if settings_key:
                return settings_key
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read settings file for API key: {e}")

        raise ConnectionError(
            "No OpenAI API key found. Set OPENAI_API_KEY environment variable, "
            "pass api_key to constructor, or add openai_api_key to settings file."
        )

    async def start_session(self, tool_config: Optional[str] = None) -> None:
        """Connect WebSocket, send session.update with tools/voice/instructions.

        Args:
            tool_config: Optional JSON string with tool definitions in Nova Sonic format.

        Raises:
            ConnectionError: If WebSocket connection cannot be established within 10s.
        """
        # Resolve API key
        api_key = self._resolve_api_key()

        # Build WebSocket URL and headers
        url = f"wss://api.openai.com/v1/realtime?model={self.model_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        # Beta header only needed for preview models, not GA models like gpt-realtime-2
        if "preview" in self.model_id:
            headers["OpenAI-Beta"] = "realtime=v1"

        # Establish WebSocket connection with 10s timeout
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, additional_headers=headers),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"WebSocket connection to {url} timed out after 10 seconds."
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to OpenAI Realtime API at {url}: {e}"
            )

        # Convert tool definitions if provided
        tools = []
        if tool_config:
            tools = convert_tools_nova_to_openai(tool_config)

        # Send session.update event
        # GA models (non-preview) use a different schema than beta
        if "preview" not in self.model_id:
            session_config = {
                "type": "realtime",
                "instructions": self.system_prompt,
                "audio": {
                    "output": {"voice": self.voice_id},
                },
                "tools": tools,
            }
        else:
            session_config = {
                "instructions": self.system_prompt,
                "voice": self.voice_id,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 700,
                },
                "tools": tools,
            }

        session_update = {
            "type": "session.update",
            "session": session_config,
        }
        await self._ws.send(json.dumps(session_update))

        # Mark session as active
        self.is_active = True

        # Start background event processing task
        self._event_task = asyncio.create_task(self._process_server_events())
        logger.info(f"OpenAI Realtime session started (model={self.model_id}, voice={self.voice_id})")

    async def end_session(self) -> None:
        """Close WebSocket connection and clean up resources.

        Steps:
        1. Set is_active = False (stops capture/play loops)
        2. Cancel background event processing task
        3. Close WebSocket connection (suppress errors if already closed)
        4. Drain audio queue
        5. Clear internal references
        """
        # 1. Stop loops immediately
        self.is_active = False

        # 2. Cancel the event processing task if it exists
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None

        # 3. Close WebSocket connection gracefully
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # 4. Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("OpenAI Realtime session ended")

    async def start_audio_input(self) -> None:
        """No-op for OpenAI (audio buffer is always accepting input).

        Unlike NovaSonicClient which may need explicit signaling to begin
        accepting audio, the OpenAI Realtime API audio buffer is always
        ready to receive input once the session is established.
        """
        logger.debug("start_audio_input called (no-op for OpenAI Realtime)")

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Resample 16kHz→24kHz, base64 encode, send input_audio_buffer.append.

        Args:
            audio_bytes: Raw PCM16 mono audio at 16000 Hz.
        """
        import base64

        if not self.is_active:
            return

        # Resample from 16kHz to 24kHz
        resampled = resample_16k_to_24k(audio_bytes)

        # Base64 encode the resampled audio
        encoded = base64.b64encode(resampled).decode("ascii")

        # Send input_audio_buffer.append event
        event = {
            "type": "input_audio_buffer.append",
            "audio": encoded,
        }
        await self._ws.send(json.dumps(event))

    async def _send_audio_chunk_raw(self, audio_bytes: bytes) -> None:
        """Base64 encode and send already-24kHz audio. No resampling.

        Args:
            audio_bytes: Raw PCM16 mono audio already at 24000 Hz.
        """
        import base64

        if not self.is_active or not self._ws:
            return

        encoded = base64.b64encode(audio_bytes).decode("ascii")
        event = {
            "type": "input_audio_buffer.append",
            "audio": encoded,
        }
        await self._ws.send(json.dumps(event))

    async def end_audio_input(self) -> None:
        """Signal end of audio input turn.

        With server-side VAD (Voice Activity Detection) enabled, this is a
        no-op because the server automatically detects when the user stops
        speaking and commits the audio buffer.

        If manual turn detection were used instead, this method would send
        an `input_audio_buffer.commit` event to signal the end of the user's
        turn, like so:
            event = {"type": "input_audio_buffer.commit"}
            await self._ws.send(json.dumps(event))
        """
        # No-op with server VAD — turn detection is handled automatically
        logger.debug("end_audio_input called (no-op with server VAD)")

    async def send_text_input(self, text: str) -> None:
        """Create a user message item and trigger a response.

        Sends a conversation.item.create event with a user message containing
        the text, followed by a response.create event to trigger the model's reply.

        Args:
            text: The text message to send as user input.
        """
        if not self._ws:
            logger.warning("Cannot send text input: WebSocket not connected")
            return

        # Build conversation.item.create event with user message
        item_create_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        }
        await self._ws.send(json.dumps(item_create_event))

        # Send response.create to trigger model response
        response_create_event = {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"],
            },
        }
        await self._ws.send(json.dumps(response_create_event))

    async def send_tool_result(self, tool_use_id: str, result: dict) -> None:
        """Send conversation.item.create (function_call_output) + response.create.

        Args:
            tool_use_id: The call ID from the function call event.
            result: The tool execution result dictionary.
        """
        if not self._ws:
            logger.warning("Cannot send tool result: WebSocket not connected")
            return

        # Build conversation.item.create event with function_call_output
        item_create_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": json.dumps(result),
            },
        }
        await self._ws.send(json.dumps(item_create_event))

        # Send response.create to trigger model follow-up response
        response_create_event = {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio"],
            },
        }
        await self._ws.send(json.dumps(response_create_event))

    async def capture_audio(self) -> None:
        """Capture audio from microphone and send chunks to OpenAI.

        Opens a PyAudio input stream at 24000 Hz and streams raw audio
        directly to the API. OpenAI's server-side VAD handles turn detection.
        """
        import pyaudio
        from concurrent.futures import ThreadPoolExecutor

        SAMPLE_RATE = 24000
        CHANNELS = 1
        FORMAT = pyaudio.paInt16
        CHUNK_SIZE = 2400  # 100ms chunks — good balance of latency vs overhead

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=self.input_device_index,
        )

        logger.info(f"🎤 Microphone capture started (24kHz, mono, PCM16, device={self.input_device_index})")

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            while self.is_active:
                audio_data = await loop.run_in_executor(
                    executor, stream.read, CHUNK_SIZE, False
                )
                await self._send_audio_chunk_raw(audio_data)

        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            executor.shutdown(wait=False)
            logger.info("Audio capture stopped.")

    async def play_audio(self) -> None:
        """Dequeue audio bytes and write to speaker output.

        Opens a PyAudio output stream at 24000 Hz (PCM16 mono) and plays
        audio from the internal queue while is_active is True.
        Uses a dedicated thread for blocking PyAudio writes to avoid
        blocking the asyncio event loop.
        """
        import pyaudio
        from concurrent.futures import ThreadPoolExecutor

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            output_device_index=self.output_device_index,
        )

        # Estimate the output latency so the mouth can be delayed to match
        # audible playback (see output_latency_ms docstring). NOTE: PortAudio's
        # get_output_latency() is wildly inflated for some devices — a Bluetooth
        # headset here reports >1000ms — which would delay the mouth by ~1s and
        # make speech onsets look badly late. The device's advertised
        # defaultLowOutputLatency is far more realistic, so prefer it and cap to
        # a sane range. Users can still fine-tune by ear via viseme_sync_delay_ms.
        try:
            if self.output_device_index is not None:
                info = p.get_device_info_by_index(self.output_device_index)
            else:
                info = p.get_default_output_device_info()
            est_ms = float(info.get("defaultLowOutputLatency", 0.0)) * 1000
            self.output_latency_ms = int(round(max(20.0, min(est_ms, 400.0))))
        except Exception:
            self.output_latency_ms = 0

        # Separate executors: one for audio write (must be serial), one for callbacks
        write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oai_audio_wr")
        cb_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oai_audio_cb")
        loop = asyncio.get_running_loop()

        logger.info(
            f"🔊 Audio playback started (24kHz, mono, PCM16, device={self.output_device_index}, "
            f"output_latency={self.output_latency_ms}ms)"
        )

        # Paced-writer state. We deliberately bound how far ahead of real-time
        # we buffer audio. Without this, we hand PortAudio the whole queue as
        # fast as it accepts it; some devices (e.g. Bluetooth) buffer >1s, so
        # the mouth (computed at write time) would need a ~1s delay to match
        # steady-state playback — yet at speech onset the buffer is empty and
        # audio is audible almost immediately, so no single constant delay can
        # fit both. By capping the buffered-ahead amount to ~the device's
        # latency, onset and steady-state latency converge and one sync delay
        # works everywhere. `buffered_until` tracks the wall-clock time up to
        # which audio has been queued.
        BYTES_PER_SEC = 24000 * 2  # PCM16 mono @ 24 kHz
        target_lead = min(0.5, max(0.12, self.output_latency_ms / 1000.0))
        buffered_until = loop.time()

        try:
            while self.is_active:
                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Write in ~40ms slices so the mouth-animation callback is
                # time-aligned to playback (one viseme per whole chunk would be
                # end-of-chunk biased). 1920 bytes = 40ms of 24kHz PCM16 mono.
                SLICE_BYTES = 1920
                for i in range(0, len(audio_data), SLICE_BYTES):
                    if not self.is_active:
                        break
                    slice_bytes = audio_data[i:i + SLICE_BYTES]

                    # Pace: never queue more than `target_lead` seconds ahead of
                    # real playback. Reset the cursor if the buffer has drained
                    # (gap between utterances) so we don't stall on fresh audio.
                    now = loop.time()
                    if buffered_until < now:
                        buffered_until = now
                    ahead = buffered_until - now
                    if ahead > target_lead:
                        await asyncio.sleep(ahead - target_lead)

                    # Write audio to speaker in dedicated thread (non-blocking to event loop)
                    await loop.run_in_executor(write_executor, stream.write, slice_bytes)
                    buffered_until += len(slice_bytes) / BYTES_PER_SEC
                    # Mouth animation callback in separate thread pool (non-blocking)
                    loop.run_in_executor(cb_executor, self._safe_audio_callback, slice_bytes)

        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            write_executor.shutdown(wait=False)
            cb_executor.shutdown(wait=False)
            logger.info("Audio playback stopped")

    async def _process_server_events(self) -> None:
        """Listen for and process incoming WebSocket messages.

        Routes events to appropriate handlers based on event type.
        Runs as a background task while is_active is True.
        """
        import base64

        try:
            while self.is_active:
                try:
                    message = await self._ws.recv()
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {e}")
                    self.is_active = False
                    break

                try:
                    event = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Received non-JSON WebSocket message, skipping")
                    continue

                event_type = event.get("type", "")

                if event_type in ("response.audio.delta", "response.output_audio.delta"):
                    # Decode base64 audio to PCM16 bytes
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        await self._audio_queue.put(audio_bytes)

                elif event_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    transcript = event.get("transcript", "")
                    if transcript and self.on_assistant_text:
                        self.on_assistant_text(transcript)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript and self.on_user_text:
                        self.on_user_text(transcript)

                elif event_type == "response.function_call_arguments.done":
                    name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    arguments_str = event.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        parsed_args = {}
                        logger.warning(f"Failed to parse function call arguments: {arguments_str}")

                    if self.on_tool_use:
                        try:
                            await self.on_tool_use(name, call_id, parsed_args)
                        except asyncio.TimeoutError:
                            logger.error(f"Tool '{name}' timed out (call_id={call_id})")
                            await self.send_tool_result(call_id, {
                                "error": "execution_failed",
                                "message": f"Tool '{name}' timed out",
                            })
                        except Exception as e:
                            logger.error(f"Tool '{name}' raised an exception (call_id={call_id}): {e}")
                            await self.send_tool_result(call_id, {
                                "error": "execution_failed",
                                "message": str(e),
                            })

                elif event_type == "error":
                    error_info = event.get("error", {})
                    error_type = error_info.get("type", "unknown")
                    error_code = error_info.get("code", "")
                    error_message = error_info.get("message", "")
                    logger.error(
                        f"OpenAI Realtime API error: type={error_type}, "
                        f"code={error_code}, message={error_message}"
                    )
                    # Continue processing - do NOT set is_active = False

                elif event_type == "session.created":
                    logger.info("OpenAI Realtime session created")

                elif event_type == "session.updated":
                    logger.info("OpenAI Realtime session configuration updated")

                elif event_type == "response.done":
                    logger.debug("OpenAI Realtime response completed")

                else:
                    logger.debug(f"Unhandled event type: {event_type}")

        except asyncio.CancelledError:
            # Task was cancelled (e.g., during end_session), exit gracefully
            raise
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {e}")
            self.is_active = False
        except Exception as e:
            logger.error(f"Unexpected error in event processing loop: {e}")
            self.is_active = False
