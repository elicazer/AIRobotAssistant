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
        model_id: str = "gpt-realtime-2.1",
        voice_id: str = "alloy",
        system_prompt: Optional[str] = None,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        api_key: Optional[str] = None,
        acoustic_tail_ms: int = 700,
        barge_in_enabled: bool = True,
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
        self.acoustic_tail_ms = max(0, int(acoustic_tail_ms))
        self.barge_in_enabled = bool(barge_in_enabled)

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
        self._input_audio_lock = asyncio.Lock()
        self._assistant_playback_active = False
        self._assistant_audio_done = True
        self._mic_suppressed_until = 0.0
        self._playback_interrupted = False
        self._discard_assistant_audio = False
        self._response_active = False
        self._pending_response_create = False
        # Track whether a model response is currently in progress so tool results
        # don't fire a second response.create (which errors with
        # conversation_already_has_active_response and wedges the session).
        self._response_active = False
        self._pending_response_create = False

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
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.7,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 700,
                            "create_response": True,
                            "interrupt_response": self.barge_in_enabled,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.voice_id,
                    },
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
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                },
                "tools": tools,
            }

        session_update = {
            "type": "session.update",
            "session": session_config,
        }
        await self._ws.send(json.dumps(session_update))

        # Reset audio-turn state before making this session active.
        self._assistant_playback_active = False
        self._assistant_audio_done = True
        self._mic_suppressed_until = 0.0
        self._playback_interrupted = False
        self._discard_assistant_audio = False
        self._response_active = False
        self._pending_response_create = False

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
        self._assistant_playback_active = False
        self._assistant_audio_done = True
        self._mic_suppressed_until = 0.0
        self._playback_interrupted = False
        self._discard_assistant_audio = False
        self._response_active = False
        self._pending_response_create = False

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

    def _input_audio_is_suppressed(self) -> bool:
        """Return whether half-duplex mode should drop microphone audio."""
        if self.barge_in_enabled:
            return False
        return (
            self._assistant_playback_active
            or asyncio.get_running_loop().time() < self._mic_suppressed_until
        )

    async def _send_input_audio(self, audio_bytes: bytes) -> bool:
        """Append 24 kHz PCM16 input unless half-duplex suppression is active."""
        import base64

        if not self.is_active or not self._ws or self._input_audio_is_suppressed():
            return False

        encoded = base64.b64encode(audio_bytes).decode("ascii")
        event = {
            "type": "input_audio_buffer.append",
            "audio": encoded,
        }

        # Re-check under the lock so half-duplex playback cannot clear the
        # server buffer and then lose a race to an append waiting to send.
        async with self._input_audio_lock:
            if not self.is_active or not self._ws or self._input_audio_is_suppressed():
                return False
            await self._ws.send(json.dumps(event))
        return True

    async def _begin_assistant_playback(self) -> None:
        """Track playback and engage input suppression when barge-in is off."""
        if self._assistant_playback_active:
            return

        self._assistant_playback_active = True
        if self.barge_in_enabled:
            logger.debug("🎙️ Barge-in active during assistant playback")
            return

        async with self._input_audio_lock:
            if self.is_active and self._ws:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        logger.info("🔇 Microphone forwarding paused during assistant playback")

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Resample 16kHz→24kHz and forward it when the mic is enabled.

        Args:
            audio_bytes: Raw PCM16 mono audio at 16000 Hz.
        """
        if not self.is_active or self._input_audio_is_suppressed():
            return

        await self._send_input_audio(resample_16k_to_24k(audio_bytes))

    async def _send_audio_chunk_raw(self, audio_bytes: bytes) -> None:
        """Forward already-24kHz PCM16 audio when the mic is enabled."""
        await self._send_input_audio(audio_bytes)

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

        # Trigger the model response (gated so we never double-create).
        if self._response_active:
            self._pending_response_create = True
        else:
            await self._send_response_create()

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

        # Only request a follow-up response if one isn't already active. If a
        # response is in progress (common when the model fires multiple tool
        # calls, e.g. set_expression), defer a single response.create until the
        # active response finishes. This avoids the
        # conversation_already_has_active_response error that otherwise wedges
        # the session and makes the assistant "stop responding".
        if self._response_active:
            self._pending_response_create = True
            logger.debug("Tool result stored; deferring response.create (response active)")
        else:
            await self._send_response_create()

    async def _send_response_create(self):
        """Send a response.create and mark a response as active."""
        self._pending_response_create = False
        self._response_active = True
        await self._ws.send(json.dumps({
            "type": "response.create",
            "response": {"output_modalities": ["audio"]},
        }))

    @staticmethod
    def _resolve_audio_device(sd, configured_index: Optional[int], direction: str):
        """Return a usable sounddevice index, falling back to the OS default."""
        if configured_index is None:
            return None

        channel_key = "max_input_channels" if direction == "input" else "max_output_channels"
        try:
            device_index = int(configured_index)
            device = sd.query_devices(device_index)
            if int(device.get(channel_key, 0)) < 1:
                raise ValueError(f"device has no {direction} channels")
            return device_index
        except (TypeError, ValueError, sd.PortAudioError) as exc:
            logger.warning(
                "Configured %s device %r is unavailable (%s); using the system default",
                direction,
                configured_index,
                exc,
            )
            return None

    @staticmethod
    def _device_name(sd, index, direction):
        """Return the device name for an index (used to re-find it after the
        device sleeps/wakes and its index shifts)."""
        if index is None:
            return None
        try:
            return sd.query_devices(int(index)).get("name")
        except Exception:
            return None

    @staticmethod
    def _resolve_input_by_name(sd, name):
        """Find an input-capable device index by its name, or None."""
        if not name:
            return None
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("name") == name and int(d.get("max_input_channels", 0)) >= 1:
                    return i
        except Exception:
            pass
        return None

    @classmethod
    def _open_sounddevice_stream(
        cls,
        sd,
        stream_type,
        configured_index: Optional[int],
        direction: str,
        **stream_options,
    ):
        """Open and start a raw stream, retrying once with the OS default."""
        device_index = cls._resolve_audio_device(sd, configured_index, direction)
        candidates = [device_index]
        if device_index is not None:
            candidates.append(None)

        errors = []
        for candidate in candidates:
            stream = None
            try:
                check_settings = (
                    sd.check_input_settings
                    if direction == "input"
                    else sd.check_output_settings
                )
                check_settings(
                    device=candidate,
                    channels=stream_options["channels"],
                    dtype=stream_options["dtype"],
                    samplerate=stream_options["samplerate"],
                )
                logger.info("Opening %s stream (device=%s)", direction, candidate)
                stream = stream_type(device=candidate, **stream_options)
                stream.start()
                return stream, candidate
            except Exception as exc:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                errors.append(f"device {candidate}: {exc}")
                if candidate is not None:
                    logger.warning(
                        "Could not open configured %s device %s (%s); retrying the system default",
                        direction,
                        candidate,
                        exc,
                    )

        raise RuntimeError(
            f"Unable to open a 24 kHz mono {direction} stream: {'; '.join(errors)}"
        )

    async def capture_audio(self) -> None:
        """Capture 24 kHz PCM16 microphone audio with sounddevice.

        The OpenAI path intentionally uses sounddevice instead of PyAudio. On
        macOS, the PyAudio C extension can segfault inside PyAudio_OpenStream
        when the camera and ML runtimes are active; sounddevice avoids that
        wrapper while retaining the same PortAudio device support.
        """
        import sounddevice as sd
        from concurrent.futures import ThreadPoolExecutor

        sample_rate = 24000
        chunk_size = 2400  # 100 ms
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oai_audio_in")
        loop = asyncio.get_running_loop()

        # Remember the configured mic BY NAME. When a USB mic (e.g. DJI) sleeps
        # or briefly disconnects, PortAudio can reassign indices; re-resolving by
        # name lets us reconnect to the same physical mic without a restart.
        target_name = self._device_name(sd, self.input_device_index, "input")
        backoff = 0.5

        try:
            # Reconnect loop: a device error (sleep/unplug) no longer ends the
            # session — we close the stream and retry until the mic returns.
            while self.is_active:
                stream = None
                try:
                    idx = self.input_device_index
                    if target_name is not None:
                        named = self._resolve_input_by_name(sd, target_name)
                        if named is None:
                            # The specific mic isn't present (asleep/unplugged).
                            # Wait for it rather than grabbing the laptop mic.
                            raise RuntimeError(f"configured mic {target_name!r} not present")
                        idx = named
                    stream, device_index = self._open_sounddevice_stream(
                        sd,
                        sd.RawInputStream,
                        idx,
                        "input",
                        samplerate=sample_rate,
                        blocksize=chunk_size,
                        channels=1,
                        dtype="int16",
                    )
                    # If we have a specific target and the open fell back to the
                    # system default, don't use it — wait for the named mic.
                    if target_name is not None and device_index is None:
                        raise RuntimeError(f"configured mic {target_name!r} unavailable; waiting")
                    logger.info(
                        "🎤 Microphone capture started (24kHz, mono, PCM16, device=%s, name=%r)",
                        device_index, target_name,
                    )
                    backoff = 0.5  # reset after a healthy open

                    while self.is_active:
                        audio_data, overflowed = await loop.run_in_executor(
                            executor, stream.read, chunk_size
                        )
                        if overflowed:
                            logger.warning("Microphone input overflowed")

                        # Always drain the physical input stream, including while
                        # Sunny speaks, so stale DJI/PortAudio audio cannot
                        # accumulate. Forwarding is paused during playback + tail.
                        if self._input_audio_is_suppressed():
                            continue
                        await self._send_audio_chunk_raw(bytes(audio_data))

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Mic likely slept/disconnected. Keep the SESSION alive and
                    # retry — the assistant simply can't hear until it returns.
                    logger.warning(
                        "🎤 Microphone stream lost (%s); reconnecting in %.1fs…",
                        exc, backoff,
                    )
                    if stream is not None:
                        try:
                            stream.abort()
                        except Exception:
                            pass
                        try:
                            stream.close()
                        except Exception:
                            pass
                        stream = None
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 3.0)
                    continue
                finally:
                    if stream is not None:
                        try:
                            stream.abort()
                        except Exception:
                            pass
                        try:
                            stream.close()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Audio capture stopped")

    async def play_audio(self) -> None:
        """Play streamed 24 kHz PCM16 audio with sounddevice."""
        import sounddevice as sd
        from concurrent.futures import ThreadPoolExecutor

        sample_rate = 24000
        write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oai_audio_wr")
        cb_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oai_audio_cb")
        callback_future = None
        stream = None
        loop = asyncio.get_running_loop()

        try:
            stream, device_index = self._open_sounddevice_stream(
                sd,
                sd.RawOutputStream,
                self.output_device_index,
                "output",
                samplerate=sample_rate,
                blocksize=0,
                channels=1,
                dtype="int16",
                latency="low",
            )

            try:
                est_ms = float(stream.latency) * 1000
                self.output_latency_ms = int(round(max(20.0, min(est_ms, 400.0))))
            except (TypeError, ValueError):
                self.output_latency_ms = 0

            logger.info(
                "🔊 Audio playback started (24kHz, mono, PCM16, device=%s, output_latency=%sms)",
                device_index,
                self.output_latency_ms,
            )

            bytes_per_second = sample_rate * 2  # PCM16 mono
            target_lead = min(0.5, max(0.12, self.output_latency_ms / 1000.0))
            buffered_until = loop.time()

            while self.is_active:
                if self._playback_interrupted:
                    # Drop audio already buffered locally so the user does not
                    # have to talk over the remainder of Sunny's response.
                    await loop.run_in_executor(write_executor, stream.abort)
                    await loop.run_in_executor(write_executor, stream.start)
                    buffered_until = loop.time()
                    self._playback_interrupted = False
                    self._assistant_playback_active = False
                    self._assistant_audio_done = True
                    logger.info("⏹️ Assistant playback interrupted by user speech")

                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Reopen the microphone only after OpenAI has finished the
                    # audio item, all queued PCM has been written, and the
                    # device's output buffer should no longer be audible.
                    audible_until = buffered_until + self.output_latency_ms / 1000.0
                    if (
                        self._assistant_playback_active
                        and self._assistant_audio_done
                        and self._audio_queue.empty()
                        and loop.time() >= audible_until
                    ):
                        self._assistant_playback_active = False
                        if self.barge_in_enabled:
                            self._mic_suppressed_until = 0.0
                            logger.debug("Assistant playback ended; barge-in remained active")
                        else:
                            self._mic_suppressed_until = (
                                loop.time() + self.acoustic_tail_ms / 1000.0
                            )
                            logger.info(
                                "🎤 Assistant playback ended; microphone forwarding resumes in %sms",
                                self.acoustic_tail_ms,
                            )
                    continue

                await self._begin_assistant_playback()

                # Write ~40 ms slices so animation callbacks stay aligned with
                # audible playback instead of firing once per large API chunk.
                slice_bytes_count = 1920
                for offset in range(0, len(audio_data), slice_bytes_count):
                    if not self.is_active or self._playback_interrupted:
                        break
                    slice_bytes = audio_data[offset:offset + slice_bytes_count]

                    now = loop.time()
                    if buffered_until < now:
                        buffered_until = now
                    ahead = buffered_until - now
                    if ahead > target_lead:
                        await asyncio.sleep(ahead - target_lead)

                    await loop.run_in_executor(write_executor, stream.write, slice_bytes)
                    buffered_until += len(slice_bytes) / bytes_per_second

                    # Animation is disposable: retain at most one callback so
                    # slow analysis cannot build an unbounded shutdown backlog.
                    if callback_future is None or callback_future.done():
                        callback_future = loop.run_in_executor(
                            cb_executor, self._safe_audio_callback, slice_bytes
                        )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.is_active = False
            logger.error("Error playing audio: %s", exc)
        finally:
            self._assistant_playback_active = False
            self._assistant_audio_done = True
            self._mic_suppressed_until = 0.0
            self._playback_interrupted = False
            self._discard_assistant_audio = False
            if stream is not None:
                try:
                    stream.abort()
                except Exception:
                    pass
            write_executor.shutdown(wait=True, cancel_futures=True)
            if callback_future is not None and not callback_future.done():
                callback_future.cancel()
            cb_executor.shutdown(wait=False, cancel_futures=True)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
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
                    # After a barge-in, ignore late deltas from the interrupted
                    # response until its response.done event arrives.
                    if self._discard_assistant_audio:
                        continue

                    # Decode base64 audio to PCM16 bytes. Mark generation active
                    # when the delta arrives so an early done event cannot be
                    # overwritten when local playback begins later.
                    self._assistant_audio_done = False
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        await self._audio_queue.put(audio_bytes)

                elif event_type in ("response.audio.done", "response.output_audio.done"):
                    self._assistant_audio_done = True
                    logger.debug("OpenAI Realtime assistant audio generation completed")

                elif event_type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    transcript = event.get("transcript", "")
                    if transcript and self.on_assistant_text:
                        self.on_assistant_text(transcript)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript and self.on_user_text:
                        self.on_user_text(transcript)

                elif event_type == "input_audio_buffer.speech_started":
                    logger.info(
                        "🎙️ OpenAI VAD speech started (item=%s, audio_start_ms=%s)",
                        event.get("item_id", "unknown"),
                        event.get("audio_start_ms", "unknown"),
                    )
                    if self.barge_in_enabled and self._assistant_playback_active:
                        self._playback_interrupted = True
                        self._discard_assistant_audio = True
                        while not self._audio_queue.empty():
                            try:
                                self._audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        logger.info("🗣️ Barge-in detected; stopping assistant audio")

                elif event_type == "input_audio_buffer.speech_stopped":
                    logger.info(
                        "🎙️ OpenAI VAD speech stopped (item=%s, audio_end_ms=%s)",
                        event.get("item_id", "unknown"),
                        event.get("audio_end_ms", "unknown"),
                    )

                elif event_type == "input_audio_buffer.cleared":
                    logger.debug("OpenAI pending microphone buffer cleared")

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

                elif event_type == "response.created":
                    # A model response is now in progress.
                    self._response_active = True

                elif event_type == "session.created":
                    logger.info("OpenAI Realtime session created")

                elif event_type == "session.updated":
                    logger.info("OpenAI Realtime session configuration updated")

                elif event_type == "response.done":
                    # Current GA/preview servers emit a dedicated audio-done
                    # event first. This is a safe fallback for older variants.
                    if self._assistant_playback_active:
                        self._assistant_audio_done = True
                    self._discard_assistant_audio = False
                    # Response finished: clear active flag and flush any tool
                    # result that was waiting for a follow-up response.create.
                    self._response_active = False
                    if self._pending_response_create:
                        await self._send_response_create()
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
