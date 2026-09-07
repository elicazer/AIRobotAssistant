# AI Robot Assistant

AI Robot Assistant is a real-time, speech-to-speech social robot named **Sunny**. It combines the **OpenAI Realtime API** (ChatGPT-style live voice conversations) or **Amazon Nova 2 Sonic** with a browser-based VRM avatar, audio-driven lip sync, face tracking, face recognition, emotion-aware behavior, and optional animatronic servos.

Sunny is designed to help people practice social conversations in a friendly, low-pressure setting. The project can run as a virtual avatar with no servo hardware, or drive a physical robot head through an FT232H and PCA9685 controller.

## Demo

[![AI Robot Assistant Demo](https://img.youtube.com/vi/v6-OhUJxlb0/0.jpg)](https://www.youtube.com/shorts/v6-OhUJxlb0)

_Click the image to watch the robot in action._

## What the project supports now

- **OpenAI Realtime API** as the configured default speech-to-speech provider
- **Amazon Nova 2 Sonic** as an alternate provider and automatic fallback when no OpenAI API key is available
- **Low-latency streaming audio** with live user and assistant transcripts
- **VRM avatar** with gaze, blinking, lip sync, and model-controlled expressions
- **Physical jaw animation** driven by speech amplitude
- **Phoneme-aware avatar visemes** with HeadAudio-derived analysis and amplitude fallback
- **Face tracking** with OpenCV and virtual or physical eye movement
- **Face recognition** with DeepFace/Facenet and a local enrollment database
- **Emotion analysis** with HSEmotion ONNX or DeepFace FER
- **Shared voice tools** for memory recall, emotion checks, expressions, web search, face enrollment, and local identity removal
- **Optional AWS Strands and AgentCore Memory** for remembered conversations and personalized greetings
- **Browser control panel** for providers, audio devices, camera tracking, avatar behavior, and servo calibration
- **Graceful hardware fallback** when the servo controller or camera is unavailable

## Architecture

```text
Microphone
    │
    ├── OpenAI Realtime API (24 kHz PCM)
    │                  or
    └── Amazon Nova 2 Sonic (16 kHz input / 24 kHz output)
                         │
                         ├── live transcripts
                         ├── function/tool calls
                         └── streamed assistant audio
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
             Speaker playback                  Lip-sync pipeline
                                                ├── VRM visemes
                                                └── jaw servo

Webcam ── OpenCV tracking ── gaze/blink ── VRM avatar and optional eye servos
   │
   └── DeepFace identity + HSEmotion/DeepFace emotion
             │
             └── optional Strands tools and AgentCore Memory
```

`run.py` loads `src/voice_assistant_server.py`, which coordinates the voice provider, audio, web server, avatar, camera analysis, tools, and optional servo hardware. Voice sessions start only after **Start Listening** is selected in the web UI.

## Voice providers

| Provider | Configuration value | Credentials | Current model setting | Notes |
|---|---|---|---|---|
| OpenAI Realtime | `openai_realtime` | `OPENAI_API_KEY` | `gpt-realtime-2.1` | Configured default; direct WebSocket speech-to-speech |
| Amazon Nova 2 Sonic | `nova_sonic` | AWS credential chain | `amazon.nova-2-sonic-v1:0` | Uses Amazon Bedrock bidirectional streaming |

When `voice_model` is `openai_realtime` but no OpenAI key is available, the server logs the issue and falls back to Nova Sonic. AWS credentials must then be available or the voice session cannot start.

Both providers use the same assistant prompt, callbacks, animation pipeline, and tool definitions. Provider changes take effect the next time a voice session starts.

## Requirements

### Software-only avatar

- macOS or Linux
- Python 3.12 (the repository pins Python 3.12.9)
- Microphone and speaker supported by PortAudio/PyAudio
- Modern browser with WebGL
- Internet access for the selected voice provider
- Internet access on first use for DeepFace/HSEmotion model downloads
- Webcam when face tracking or recognition is enabled

The launch scripts are Unix shell scripts. Windows is not currently documented or tested as a supported runtime. Raspberry Pi and Jetson-specific configuration exists, but the active camera path currently uses standard OpenCV capture and should be considered experimental on those platforms.

The avatar and voice assistant work without servo hardware. To run without a webcam, set `face_tracking_enabled` and `face_recognition_enabled` to `false` in `config/voice_assistant_settings.json`.

### Optional physical robot hardware

- FT232H USB-to-I²C adapter
- PCA9685 16-channel servo driver at address `0x40`
- Standard positional servos
- 5–6 V servo power supply sized for the connected servos
- Common ground between the servo supply and controller

Do not power a bank of servos directly from the computer or FT232H.

### Default servo channel layout

| Channel | Function |
|---:|---|
| 0 | Left eye horizontal |
| 1 | Left eye vertical |
| 2 | Left upper eyelid |
| 3 | Left lower eyelid |
| 4 | Right eye horizontal |
| 5 | Right eye vertical |
| 6 | Right upper eyelid |
| 7 | Right lower eyelid |
| 8 | Jaw |

The `inmoov`, `original`, and `simple` layouts are defined in `src/servo_config.py`. The eight-channel InMoov eye layout is the primary configuration used by the current tracking implementation.

## Installation

From the repository root:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyAudio requires PortAudio. If PyAudio cannot build on macOS, install PortAudio first:

```bash
brew install portaudio
python -m pip install -r requirements.txt
```

The full requirements include both voice providers, Flask/Socket.IO, audio processing, robot hardware libraries, OpenCV, DeepFace, TensorFlow compatibility, HSEmotion ONNX, Strands, and test dependencies.

### First-use model downloads

- DeepFace downloads Facenet weights to `~/.deepface/weights/`.
- HSEmotion downloads an ONNX model to `~/.hsemotion/`.

These downloads happen once and require internet access. Face tracking itself uses OpenCV and does not require DeepFace; identity recognition does.

## Credentials

### OpenAI Realtime

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

`.env` is ignored by Git. Prefer this environment variable over storing a key in `config/voice_assistant_settings.json`. The local and full launch scripts load `.env`; for a direct `python run.py` launch, export the variable in the current shell first.

Credential priority in the main application is:

1. `OPENAI_API_KEY`
2. `openai_api_key` in the settings file

Your OpenAI account must have access to the model configured by `openai_model_id`.

### Amazon Nova 2 Sonic and AWS features

Nova Sonic, Strands, Bedrock, Secrets Manager, and AgentCore Memory use the normal AWS credential chain. Depending on your environment, configure an AWS profile or export temporary credentials:

```bash
export AWS_PROFILE=robot
aws sso login --profile "$AWS_PROFILE"
```

Or use `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, for temporary credentials, `AWS_SESSION_TOKEN`.

The full `start.sh` launcher defaults to the `robot` profile. It checks AWS SSO credentials and attempts to retrieve the OpenAI key from the `robot/openai-api-key` secret in `us-east-1` when a key is not already loaded.

### Optional AgentCore Memory

Create or update the memory resource and write its ID to the settings file with:

```bash
source venv/bin/activate
python scripts/setup_agentcore_memory.py --region us-east-1
```

Use `--no-update-settings` to create the resource without modifying `config/voice_assistant_settings.json`.

AgentCore memory is optional. Without an `agentcore_memory_resource_id`, realtime voice, the avatar, tracking, recognition, and local enrollment can still operate, but long-term conversation recall is unavailable.

## Configuration

Settings are loaded from `config/voice_assistant_settings.json` and merged over built-in defaults in `src/voice_assistant_server.py`.

A representative configuration is:

```json
{
  "voice_model": "openai_realtime",
  "openai_model_id": "gpt-realtime-2.1",
  "openai_voice_id": "alloy",
  "openai_barge_in_enabled": true,
  "openai_acoustic_tail_ms": 700,
  "voice_id": "matthew",
  "microphone_index": null,
  "speaker_index": null,
  "face_tracking_enabled": true,
  "camera_index": 0,
  "face_recognition_enabled": true,
  "recognition_confidence_threshold": 0.75,
  "emotion_backend": "hsemotion",
  "emotion_min_confidence": 0.4,
  "emotion_neutral_bias": 0.05,
  "emotion_smoothing_window": 3,
  "servo_config": "inmoov",
  "jaw_open_angle": 100,
  "jaw_close_angle": 0,
  "jaw_servo_min_change": 2,
  "avatar_vrm_file": "Avatar 1.vrm",
  "viseme_analyzer": "headaudio",
  "viseme_sync_delay_ms": null,
  "agentcore_memory_resource_id": "",
  "aws_region": "us-east-1",
  "tool_timeout_seconds": 10
}
```

Do not copy API keys into documentation or commit them to the settings file.

### Important settings

| Setting | Purpose |
|---|---|
| `voice_model` | `openai_realtime` or `nova_sonic` |
| `openai_model_id` | OpenAI Realtime model sent in the WebSocket URL |
| `openai_voice_id` | OpenAI output voice |
| `openai_barge_in_enabled` | `true` keeps the microphone live and lets detected user speech interrupt Sunny; `false` uses half-duplex suppression |
| `openai_acoustic_tail_ms` | Post-playback microphone discard interval used only when barge-in is disabled; defaults to `700` |
| `voice_id` | Nova Sonic output voice |
| `microphone_index` / `speaker_index` | PyAudio device indexes; `null` uses system defaults |
| `face_tracking_enabled` | Opens the configured camera and drives gaze/blinking |
| `camera_index` | OpenCV camera device index |
| `face_recognition_enabled` | Enables identity, emotion, and the current tool stack initialization |
| `face_database_path` | Local JSON database for enrolled face embeddings |
| `recognition_confidence_threshold` | Identity match threshold |
| `emotion_backend` | `hsemotion` or `deepface` |
| `deepface_analysis_interval` | Seconds between background identity/emotion analyses |
| `emotion_min_confidence` | Suppresses low-confidence non-neutral results |
| `emotion_neutral_bias` | Requires an emotion to beat neutral by this margin |
| `emotion_smoothing_window` | Number of recent results used to reduce flicker |
| `servo_config` | `inmoov`, `original`, or `simple` |
| `jaw_open_angle` / `jaw_close_angle` | Physical jaw endpoints in degrees |
| `avatar_vrm_file` | Basename of a `.vrm` file in `templates/avatars/` |
| `viseme_analyzer` | `headaudio` for phoneme visemes or `amplitude` for fallback shapes |
| `viseme_sync_delay_ms` | Manual avatar/audio sync delay; `null` uses measured output latency |
| `agentcore_memory_resource_id` | Optional AgentCore Memory resource ID |
| `aws_region` | Region for Strands and AgentCore operations |
| `tool_timeout_seconds` | Maximum tool execution time |

### Voice choices

The web UI keeps separate selections for each provider and changes the voice dropdown when the provider changes. Voice changes apply when the next voice session starts.

**OpenAI Realtime:** Alloy, Ash, Ballad, Cedar, Coral, Echo, Marin, Sage, Shimmer, and Verse. OpenAI recommends **Marin** and **Cedar** for the best quality. The dropdown intentionally excludes voices that exist in the broader OpenAI text-to-speech catalog but are not included in the Realtime session voice enum.

**Nova Sonic:** Matthew, Tiffany, Amy, Ambre, Florian, Beatrice, Lorenzo, Greta, Lennart, Lupe, and Carlos. The Nova labels in the UI include their configured language and voice presentation.

See the [OpenAI Realtime client-events reference](https://developers.openai.com/api/reference/resources/realtime/client-events) for the current Realtime voice enum. Content was rephrased for compliance with licensing restrictions.

The checked-in audio device indexes may not exist on another computer. Select valid devices in the UI or set both indexes to `null` before the first run.

### Avatar models

Place VRM files in `templates/avatars/` and set `avatar_vrm_file` to the file’s basename. The server rejects paths outside this directory and only serves `.vrm` files.

## Running the assistant

### Local launcher

Use this for the simplest OpenAI/local workflow after creating `venv` and `.env`:

```bash
./scripts/run_local.sh
```

This script loads `.env`, applies a macOS certificate-path workaround when available, and runs the application without performing AWS SSO or Secrets Manager setup.

### Full AWS-aware launcher

```bash
./start.sh
```

This launcher:

- Requires the existing `venv`
- Loads `.env`
- Checks the configured AWS profile and may start SSO login
- Attempts to obtain `OPENAI_API_KEY` from AWS Secrets Manager when needed
- Stops an existing process on port 8080
- Starts `python run.py` and records its PID in `.robot.pid`

Override the default AWS profile when needed:

```bash
AWS_PROFILE=my-profile ./start.sh
```

### Direct launch

```bash
source venv/bin/activate
export OPENAI_API_KEY=your_openai_api_key  # omit when using Nova only
python run.py
```

Then open:

```text
http://127.0.0.1:8080
```

The web server intentionally binds to loopback, so it is not exposed to other devices on the network.

### Starting and stopping sessions

1. Open the web UI.
2. Select the provider and audio devices.
3. Select **Start Listening** to open a realtime voice session.
4. Select **Stop** to end the active voice session while leaving the server and camera process running.
5. Press `Ctrl+C` in the launching terminal to shut down the application.

For a process started by `start.sh`, use:

```bash
./stop.sh
```

`stop.sh` first sends an interrupt to the recorded process. If no PID file exists, it checks for the process using port 8080.

## Web interface and local API

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8080/` | Main VRM avatar and control panel |
| `http://127.0.0.1:8080/avatar` | Standalone VRM viewer |
| `GET /avatar/model` | Serves the configured local VRM file |
| `GET /api/status` | Current viseme, text, and mouth intensity |
| `GET /api/devices` | Available PyAudio input/output devices |
| `GET /api/settings` | Current UI settings with the OpenAI key masked |
| `POST /api/settings` | Updates the supported provider, audio, and tracking settings |

The main UI includes provider and audio selection, Start/Stop controls, face-tracking controls, live identity/emotion state, jaw tests, eye-servo calibration, gaze, blinking, expressions, and lip-synced VRM animation.

Advanced avatar tuning:

- Add `?visemeDelay=NN` to manually delay visemes by `NN` milliseconds.
- Add `?teach=0` or `?teach=1` to disable or enable the emotion-teaching label.
- Add `?debug` to `/avatar` to show avatar diagnostics.

The browser loads Socket.IO, Three.js, and `@pixiv/three-vrm` from public CDNs, so the UI is not fully offline even though the VRM model is local.

## Face tracking, recognition, and emotion

### Face tracking

OpenCV uses the included Haar cascade to locate the nearest face. Coordinates are mapped to virtual gaze and, when available, physical eye servos. Blinking is generated for both virtual and physical eyes.

### Face recognition

DeepFace/Facenet generates face embeddings. Enrolled identities are stored locally in:

```text
config/face_database.json
```

Enrollment and identity removal are available through the tool layer and Socket.IO UI controls. Avoid committing real biometric enrollment data.

### Emotion detection

The default backend is `hsemotion`, which uses an AffectNet-based ONNX model. Set `emotion_backend` to `deepface` to use DeepFace’s legacy facial-expression model instead. Identity recognition continues to use DeepFace regardless of the emotion backend.

Sunny can use emotion state to adapt conversational tone and can display `happy`, `sad`, `angry`, `relaxed`, and `neutral` expressions on its own avatar.

## Tools and memory

When face recognition and the tool stack initialize successfully, either voice provider can use:

- `recall_memory` — retrieve remembered context for the recognized person
- `get_emotion` — read the current smoothed emotion state
- `set_expression` — animate Sunny’s VRM expression
- `search_web` — search through the Strands agent
- `enroll_face` — capture and remember a named face locally
- `forget_person` — remove a person from the local face database

Memory and web search require a successfully initialized Strands agent and relevant AWS configuration. In the current implementation, tool initialization is tied to the face-recognition subsystem; disabling recognition also disables these model tools.

`forget_person` removes local face data only. It does not delete events or extracted memories already stored in AgentCore Memory.

## Lip sync

The output audio path runs at 24 kHz:

```text
Assistant PCM audio
    ├── RMS amplitude → smoothing → physical jaw opening
    └── HeadAudio-derived phoneme analysis → VRM aa/ih/ou/ee/oh visemes
                                      └── amplitude fallback when unavailable
```

Audio callbacks are aligned with speaker writes. The OpenAI client also estimates output-buffer latency, and `viseme_sync_delay_ms` can override the automatic compensation when manual tuning is needed.

## Testing and diagnostics

Run the unit tests:

```bash
source venv/bin/activate
python -m pytest tests/unit
```

Run all discovered tests:

```bash
python -m pytest
```

Current automated coverage focuses on OpenAI Realtime session/event handling, Nova Sonic behavior, provider tool conversion, audio resampling, and viseme analysis. The `tests/integration/` and `tests/properties/` directories are placeholders.

The root-level diagnostics interact with local audio or hardware:

```bash
python test_audio_mouth.py       # microphone/amplitude display
python test_i2c_connection.py    # scans I²C and may move the jaw servo
python test_jaw_servo.py         # repeatedly moves the channel-8 jaw servo
```

Disconnect mechanical loads or verify safe servo limits before running hardware diagnostics.

## Troubleshooting

### macOS launcher permission or native-extension errors

If the local launcher reports `permission denied`, run it explicitly through Bash:

```bash
bash scripts/run_local.sh
```

Alternatively, restore its executable permission once and use the normal command:

```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

A copied, moved, interrupted, or stale virtual environment can leave compiled macOS packages unusable. Typical messages include:

- `library load disallowed by system policy`
- `code signature ... not valid for use in process`
- `No module named 'pyaudio._portaudio'`
- `No module named 'numpy._core._multiarray_umath'`
- A failure while loading `cv2.abi3.so`

Stop the assistant, then rebuild the generated `venv` from the repository requirements:

```bash
python3.12 -m venv --clear venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This clears only installed packages inside `venv`; it does not remove project files, settings, face data, or `.env`. Do not copy a virtual environment between machines or Python installations—recreate it from `requirements.txt` instead.

Verify the native audio and vision dependencies before restarting:

```bash
venv/bin/python -c "import pyaudio, sounddevice, numpy, cv2; print('Native dependencies OK')"
bash scripts/run_local.sh
```

### OpenAI session does not start

- Confirm `OPENAI_API_KEY` is exported or present in `.env` when using a launcher.
- Confirm the configured `openai_model_id` is available to your account.
- Check the console for WebSocket authentication or model errors.
- If no key is found, expect the application to fall back to Nova Sonic and require AWS credentials.

### Segmentation fault when an OpenAI session starts

On macOS, older revisions opened OpenAI microphone and speaker streams through the PyAudio C extension. Under the full camera and ML workload, macOS could terminate Python with a native stack ending in `PyAudio_OpenStream`; the later `resource_tracker` semaphore warning was a consequence of that abrupt shutdown.

The OpenAI Realtime path now uses the pinned `sounddevice` binding for capture and playback, while Nova Sonic and `/api/devices` continue to use PyAudio. Install the current requirements before retrying:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

If the crash persists after updating, rebuild `venv` using the macOS native-extension procedure above. A normal Python traceback indicates a different problem; look for the first error above it rather than the final cleanup warning.

### Nova Sonic or memory does not start

- Run `aws sts get-caller-identity` to verify the active credential chain.
- For SSO profiles, run `aws sso login --profile <profile>`.
- Confirm the configured region supports the AWS service and model being used.
- AgentCore Memory is optional; remove or correct an invalid resource ID if only voice is needed.

### No microphone or speaker audio

- Grant microphone permission to Terminal, the IDE, or Python on macOS.
- Open `/api/devices` or use the UI to find valid device indexes.
- Set `microphone_index` and `speaker_index` to `null` to try system defaults.
- If PyAudio fails to import or open a stream, verify PortAudio is installed.
- OpenAI validates the selected device and its 24 kHz mono format through `sounddevice`; if a saved device is missing, has the wrong channel direction, is busy, or rejects the format, it retries the operating-system default.

Bluetooth and USB devices can renumber every PortAudio index when connected or disconnected. Connect the device before starting the app, reload the web UI, and reselect both devices. Many wireless microphones—including DJI microphone receivers—are input-only, so select a separate output such as **MacBook Pro Speakers**. Do not assume that a previously saved numeric index still names the same hardware.

List the indexes currently visible to the OpenAI audio path with:

```bash
venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Barge-in and speaker feedback

The checked-in OpenAI configuration enables barge-in with `openai_barge_in_enabled: true`. The DJI microphone remains live while Sunny speaks, and server VAD can interrupt the active response when you begin talking. On a detected barge-in, the client discards queued assistant audio and resets the local output stream so Sunny stops promptly.

The GA Realtime session uses server VAD with a `0.7` threshold, 300 ms prefix padding, 700 ms silence detection, automatic response creation, and response interruption matching the barge-in setting. Console messages named `OpenAI VAD speech started`, `Barge-in detected`, and `OpenAI VAD speech stopped` show the turn lifecycle.

High microphone gain or close speaker placement can make speaker output look like a barge-in. Reduce the DJI gain first and keep the microphone separated from the speakers. If acoustic feedback still interrupts Sunny or creates extra turns, set `openai_barge_in_enabled` to `false` and restart. That fallback uses deterministic half-duplex mode: the microphone remains open and drained, but input is discarded during playback and for `openai_acoustic_tail_ms` afterward; pending uncommitted server input is also cleared when playback begins.

Face recognition and emotion analysis remain active in both modes, and their UI events do not create voice turns. See the [OpenAI Realtime VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad) for the protocol settings. Content was rephrased for compliance with licensing restrictions.

### Camera does not open

- Grant camera permission to the launching application.
- Close other applications using the webcam.
- Try `camera_index` values such as `0`, `1`, or `2`.
- Disable `face_tracking_enabled` and `face_recognition_enabled` for voice/avatar-only use.

If `cv2` has no `CascadeClassifier`, remove an incompatible pre-release and reinstall the pinned 4.x dependency from `requirements.txt`.

### Face recognition or emotion initialization fails

- Confirm all requirements were installed in the active virtual environment.
- Allow first-use model downloads to complete.
- Verify `tf-keras` is installed for TensorFlow 2.16+ compatibility.
- Corrupt or incompatible local model caches may need to be downloaded again.

The server degrades to voice plus avatar when optional face-analysis initialization fails.

### Servo controller is unavailable

- Confirm the FT232H is connected and visible to the operating system.
- Verify PCA9685 address `0x40`, I²C wiring, common ground, and external servo power.
- Use conservative angles before running test or sweep controls.
- The application continues in virtual-avatar mode when ServoKit initialization fails.

### Port 8080 is already in use

Stop the existing launcher-managed process with `./stop.sh`, or identify the application already bound to `127.0.0.1:8080`. Note that `start.sh` attempts to stop an existing process on that port automatically.

## Project structure

```text
AIRobotAssistant/
├── run.py                              # Main entry point
├── start.sh / stop.sh                  # AWS-aware process launcher and stop script
├── requirements.txt                    # Runtime and test dependencies
├── config/
│   ├── voice_assistant_settings.json   # Runtime settings
│   └── face_database.json              # Local face embeddings
├── scripts/
│   ├── run_local.sh                    # Minimal local launcher
│   ├── activate_venv.sh                # Virtual-environment helper
│   ├── setup_agentcore_memory.py       # AgentCore Memory setup
│   └── refresh_aws_creds.sh            # Environment-specific legacy credential helper
├── src/
│   ├── voice_assistant_server.py       # Runtime coordinator and Sunny prompt
│   ├── openai_realtime_client.py       # OpenAI Realtime WebSocket client
│   ├── nova_sonic_client.py            # Amazon Nova 2 Sonic client
│   ├── tool_handler.py                 # Shared model tool definitions and routing
│   ├── strands_agent.py                # Bedrock/Strands/AgentCore integration
│   ├── audio_mouth_controller.py       # Amplitude-based jaw/lip analysis
│   ├── viseme_driver.py                # Phoneme-aware VRM visemes
│   ├── face_tracker.py                 # OpenCV face tracking
│   ├── face_recognition_system.py      # Enrollment and identity matching
│   ├── deepface_analyzer.py            # Background identity/emotion analysis
│   ├── emotion_detector.py             # HSEmotion/DeepFace emotion state
│   ├── eye_controller.py               # Eye and eyelid servo control
│   ├── servo_config.py                 # Servo layout presets
│   └── mouth_visualizer.py             # Flask/Socket.IO web backend
├── templates/
│   ├── mouth.html                      # Main controls and embedded avatar
│   ├── avatar_test.html                # Standalone VRM viewer
│   └── avatars/                        # Local VRM files
└── tests/unit/                          # Automated unit tests
```

## Privacy and security

- The web server binds to `127.0.0.1`, not all network interfaces.
- OpenAI and AWS voice providers receive streamed microphone audio and conversation content.
- Face embeddings are stored locally in `config/face_database.json`.
- Optional AgentCore Memory stores conversation events in AWS under the configured resource.
- The avatar page loads JavaScript dependencies from public CDNs.
- Keep `.env`, cloud credentials, API keys, and real biometric data out of source control.
- Local identity removal does not automatically delete AgentCore cloud memory.

Review the data-handling requirements for your users before enabling recognition or long-term memory, especially in deployments involving children.

## Contributing

Contributions and bug reports are welcome. Before submitting a change, run the unit test suite and describe any required camera, audio, AWS, or servo hardware used for validation.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Amazon Nova Sonic](https://aws.amazon.com/nova/speech/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Kiro](https://kiro.dev/)
- [InMoov](https://inmoov.fr/)
- [Adafruit](https://www.adafruit.com/)
- [OpenCV](https://opencv.org/)
- [three-vrm](https://github.com/pixiv/three-vrm)
