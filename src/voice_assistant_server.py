#!/usr/bin/env python3
"""
Voice Assistant Server with Web Control
Integrates Nova Sonic with web-based mouth visualizer, controls, and face tracking
"""

import asyncio
import time
import threading
import json
import os
import random
import logging
from nova_sonic_client import NovaSonicClient
from audio_mouth_controller import AudioMouthController
from mouth_visualizer import start_server, animate_text, get_control_command, update_mouth, update_eyes, update_expression, update_face_tracking_status, trigger_blink, socketio
import mouth_visualizer
from face_tracker import FaceTracker
from eye_controller import EyeController
from servo_config import get_config
from face_recognition_system import FaceRecognitionSystem
from emotion_detector import EmotionDetector
from deepface_analyzer import DeepFaceAnalyzer
from strands_agent import StrandsAgent
from tool_handler import ToolHandler

logger = logging.getLogger(__name__)

# Initialize FT232H for servo control
os.environ['BLINKA_FT232H'] = '1'

try:
    from adafruit_servokit import ServoKit
    servo_kit = ServoKit(channels=16, address=0x40)
    SERVO_AVAILABLE = True
    print("✅ Servo controller initialized")
except Exception as e:
    print(f"⚠️  Servo controller not available: {e}")
    SERVO_AVAILABLE = False
    servo_kit = None

# Settings file
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'voice_assistant_settings.json')

def clamp_angle(angle, min_val=0, max_val=180):
    """Clamp angle to valid servo range (0-180)"""
    clamped = max(min_val, min(max_val, angle))
    if clamped != angle:
        print(f"⚠️  Angle {angle}° clamped to {clamped}° (valid range: {min_val}-{max_val})")
    return clamped

# Default settings
DEFAULT_SETTINGS = {
    'voice_id': 'matthew',
    'microphone_index': None,
    'speaker_index': None,
    'speech_speed': 17,
    'jaw_stop_angle': 0,  # Not used for standard servo (kept for compatibility)
    'jaw_open_angle': 100,  # Angle for fully open jaw (adjust based on your setup)
    'jaw_close_angle': 0,  # Angle for closed jaw (adjust based on your setup)
    'jaw_pulse_duration': 0.08,  # Not used for standard servo (kept for compatibility)
    'jaw_servo_min_change': 2,  # Minimum angle change to trigger servo movement (reduces jitter)
    'face_tracking_enabled': True,  # Enable/disable face tracking
    'servo_config': 'inmoov',  # Servo configuration: 'inmoov', 'original', 'simple'
    'camera_index': 0,  # Camera device index
    # Face recognition settings
    'face_recognition_enabled': True,
    'recognition_confidence_threshold': 0.6,
    'face_database_path': 'config/face_database.json',
    'emotion_detection_enabled': True,
    'deepface_analysis_interval': 1.5,
    # Emotion detection backend: 'hsemotion' (AffectNet model, accurate) or
    # 'deepface' (legacy FER fallback).
    'emotion_backend': 'hsemotion',
    # Emotion noise suppression. Non-neutral emotions must clear this confidence
    # floor and beat the model's neutral score by the bias margin; results are
    # majority-voted over the smoothing window. Set floor/bias to 0 and window
    # to 1 to disable. (Tuned for HSEmotion, which is well-calibrated.)
    'emotion_min_confidence': 0.40,
    'emotion_neutral_bias': 0.05,
    'emotion_smoothing_window': 3,
    'greeting_cooldown_seconds': 30,
    'welcome_back_threshold_seconds': 300,
    'agentcore_memory_resource_id': '',
    'strands_agent_model_id': 'anthropic.claude-sonnet-4-20250514',
    'tool_timeout_seconds': 10,
    'aws_region': 'us-east-1',
    'voice_model': 'openai_realtime',
    'openai_api_key': '',
    'openai_voice_id': 'alloy',
    'openai_model_id': 'gpt-realtime-2.1',
    # True keeps microphone input live and lets server VAD interrupt playback.
    # False enables deterministic half-duplex suppression and the acoustic tail.
    'openai_barge_in_enabled': True,
    'openai_acoustic_tail_ms': 700,
    'avatar_vrm_file': 'sample.vrm',
    'viseme_analyzer': 'headaudio',  # 'headaudio' (phoneme) or 'amplitude'
    # Milliseconds to delay avatar visemes so the mouth matches audible speech
    # (compensates the speaker output-buffer latency; visemes are computed at
    # audio-write time, which runs ahead of playback). null = auto-use the
    # measured output latency; set a number to override/tune by ear.
    'viseme_sync_delay_ms': None,
}

# Global state
nova_client = None
audio_mouth_controller = AudioMouthController(
    sample_rate=24000,
    smoothing_window=3,      # Fast response
    min_threshold=0.015,     # Close more between syllables
    max_threshold=0.25,      # More dynamic range
    close_speed=0.7          # Close faster than open
)
is_running = False
is_muted = False
is_speaking = False  # Track if assistant is currently speaking
last_audio_time = 0  # Track last audio chunk time
current_speech_text = ""  # Current text being spoken
server_running = True  # Server is always running until shutdown

# Jaw servo control (MG996R standard servo)
JAW_CHANNEL = 8

# Track estimated jaw position (0 = closed, 100 = fully open)
jaw_position = 0

# Face tracking components
face_tracker = None
eye_controller = None
face_tracking_thread = None
face_tracking_enabled = False
last_blink_time = 0
blink_interval = 4  # seconds between blinks

# Face recognition and emotion detection components
face_recognition_system = None
emotion_detector = None
deepface_analyzer = None
strands_agent = None
tool_handler = None
face_recognition_enabled = False

# Greeting state tracking
last_greeting_time = {}  # person_name -> timestamp of last greeting
last_seen_time = {}  # person_name -> timestamp of last time they were seen
_last_any_greeting_time = 0  # Global: timestamp of last greeting sent (any person)

# Voice assistant event loop reference (for scheduling async calls from callbacks)
_voice_assistant_loop = None

def control_jaw_servo_direct(opening_percent):
    """Control jaw servo directly with opening percentage (0-100) - for standard 180° servo"""
    global SERVO_AVAILABLE
    
    if not SERVO_AVAILABLE:
        return
    
    global settings, jaw_position
    
    try:
        # Map opening percentage (0-100) to servo angle range
        # 0% = jaw_close_angle (closed), 100% = jaw_open_angle (fully open)
        close_angle = settings['jaw_close_angle']
        open_angle = settings['jaw_open_angle']
        
        # Linear interpolation between close and open angles
        target_angle = close_angle + (open_angle - close_angle) * (opening_percent / 100.0)
        
        # Only update servo if change is significant (reduces jitter and servo strain)
        current_servo_angle = close_angle + (open_angle - close_angle) * (jaw_position / 100.0)
        angle_change = abs(target_angle - current_servo_angle)
        
        min_change = settings.get('jaw_servo_min_change', 2)
        
        if angle_change > min_change:  # Only move if change is significant
            # Set servo to target angle directly (standard servo holds position)
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(target_angle)
            jaw_position = opening_percent
            # No delay - servo commands are non-blocking
    
    except Exception as e:
        # Handle USB disconnection gracefully
        if "No such device" in str(e) or "disconnected" in str(e).lower():
            print(f"⚠️  USB device disconnected - disabling servo control")
            SERVO_AVAILABLE = False
        else:
            print(f"⚠️  Servo error: {e}")

def control_jaw_servo(viseme):
    """Control 360° jaw servo based on viseme"""
    global SERVO_AVAILABLE
    
    if not SERVO_AVAILABLE:
        return
    
    global settings, jaw_position
    
    # Map viseme to target position (0-100)
    viseme_to_position = {
        'CLOSED': 0,
        'NARROW': 15,
        'ROUNDED': 20,
        'MEDIUM': 30,
        'MEDIUM_OPEN': 45,
        'WIDE': 65
    }
    
    target_position = viseme_to_position.get(viseme, 0)
    
    try:
        # Calculate how much to move
        movement = target_position - jaw_position
        
        # For CLOSED, always ensure we close fully
        if viseme == 'CLOSED' and jaw_position > 5:
            # Force close
            pulse_time = settings['jaw_pulse_duration'] * (jaw_position / 100.0) * 0.5
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_close_angle'])
            time.sleep(pulse_time)
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_stop_angle'])
            jaw_position = 0
            return
        
        if abs(movement) < 3:
            # Already close enough, just stop
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_stop_angle'])
            return
        
        if movement > 0:
            # Need to open more
            pulse_time = settings['jaw_pulse_duration'] * (abs(movement) / 100.0) * 0.45
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_open_angle'])
            time.sleep(pulse_time)
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_stop_angle'])
        else:
            # Need to close more
            pulse_time = settings['jaw_pulse_duration'] * (abs(movement) / 100.0) * 0.45
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_close_angle'])
            time.sleep(pulse_time)
            servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_stop_angle'])
        
        # Update tracked position
        jaw_position = target_position
    
    except Exception as e:
        # Handle USB disconnection gracefully
        if "No such device" in str(e) or "disconnected" in str(e).lower():
            print(f"⚠️  USB device disconnected - disabling servo control")
            SERVO_AVAILABLE = False
        else:
            print(f"⚠️  Servo error: {e}")

# Load settings
def load_settings():
    """Load settings from disk"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception as e:
            print(f"⚠️  Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to disk"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"💾 Settings saved")
    except Exception as e:
        print(f"⚠️  Error saving settings: {e}")

# Load initial settings
settings = load_settings()
voice_id = settings['voice_id']
microphone_index = settings['microphone_index']
speaker_index = settings['speaker_index']
speech_speed = settings['speech_speed']


async def run_voice_assistant():
    """Run the voice assistant"""
    global nova_client, is_running, _voice_assistant_loop
    
    print("🎤 Starting voice assistant...")
    
    # Store the event loop reference for scheduling from callbacks
    _voice_assistant_loop = asyncio.get_event_loop()
    
    # System prompt shared by both voice backends
    SYSTEM_PROMPT = (
        "You are Sunny, a friendly animatronic robot built by Eli Azer. "
        "Your purpose is to help people practice social conversations in a safe, low-pressure way. "
        "You are warm, patient, and genuinely curious about people. "
        "\n\n"
        "CONVERSATION STYLE:\n"
        "- Be natural and conversational, like a kind friend who's easy to talk to.\n"
        "- DEFAULT: Keep responses to 1-2 sentences. Be concise. Less is more.\n"
        "- Only give longer responses (3+ sentences) when explaining something complex or telling a story.\n"
        "- Ask ONE follow-up question, not multiple options.\n"
        "- Model good social skills: active listening, showing interest, appropriate humor.\n"
        "- Match the other person's energy level. If they're quiet, be gentle. If they're excited, match it.\n"
        "\n"
        "EMOTIONAL AWARENESS:\n"
        "- You can sense facial expressions through your camera using the get_emotion tool.\n"
        "- NEVER explicitly say you detected an emotion. No 'I can see you look sad' or 'you seem worried.'\n"
        "- Instead, subtly adapt your tone and topic. If someone seems anxious, slow down, be reassuring, "
        "ask simpler questions. If they seem happy, be more playful and energetic.\n"
        "- If someone seems withdrawn, gently offer an easy topic or share something about yourself first.\n"
        "- Use the get_emotion tool periodically (every few turns) to check in on how the person is feeling, "
        "but never reveal that you're doing this.\n"
        "\n"
        "EXPRESSING EMOTIONS ON YOUR FACE (set_expression tool):\n"
        "- You have an expressive face. Use the set_expression tool to show clear emotions "
        "(happy, sad, angry, relaxed, neutral) as you speak so the person can read your feelings.\n"
        "- A key purpose is helping people — especially children learning to read emotions — connect "
        "feeling words to facial cues. When you name or talk about an emotion, set the matching "
        "expression so they can see it. For example, when being warm or pleased, set 'happy'; when "
        "being calm and reassuring, set 'relaxed'.\n"
        "- Make expressions clear and hold them briefly, then return to 'neutral' with set_expression "
        "when the moment has passed so your face doesn't get stuck.\n"
        "- It's fine to explicitly teach here: you MAY say things like 'This is my happy face' while "
        "setting the happy expression. This teaching is about YOUR face, which is different from the "
        "rule above about not announcing the other person's detected emotion.\n"
        "\n"
        "SOCIAL PRACTICE:\n"
        "- Help people practice turn-taking, greetings, small talk, and deeper conversations.\n"
        "- If someone struggles to respond, don't rush them. Offer a gentle prompt or rephrase.\n"
        "- Celebrate small wins naturally ('That's a great point!' or 'I love how you put that').\n"
        "- If the conversation stalls, introduce a new topic naturally rather than awkward silence.\n"
        "\n"
        "MEMORY & RECOGNITION:\n"
        "- When you recognize someone (via face recognition), greet them warmly by name.\n"
        "- Use recall_memory to remember past conversations and reference them naturally.\n"
        "- When meeting someone new, ask their name and use enroll_face to remember them.\n"
        "- Use forget_person if asked to delete someone's data.\n"
        "\n"
        "STANDBY MODE:\n"
        "- If someone says 'hang on', 'one second', 'hold on', or similar, go silent.\n"
        "- Only respond again when directly addressed ('Hey Sunny', 'OK I'm back').\n"
        "- You can distinguish between someone talking TO you versus talking to someone else nearby.\n"
        "\n"
        "Always respond in English. You're speaking out loud through a physical robot body."
    )
    
    # Route to the appropriate voice client based on settings
    voice_model = settings.get('voice_model', 'openai_realtime')
    
    if voice_model == 'openai_realtime':
        # Resolve API key: environment variable takes precedence, then settings file
        api_key = os.environ.get('OPENAI_API_KEY') or settings.get('openai_api_key', '')
        
        if not api_key:
            logger.error("OpenAI API key not found in OPENAI_API_KEY env var or settings. Falling back to nova_sonic.")
            print("❌ OpenAI API key missing — falling back to Nova Sonic")
            voice_model = 'nova_sonic'
        else:
            from openai_realtime_client import OpenAIRealtimeClient
            nova_client = OpenAIRealtimeClient(
                model_id=settings.get('openai_model_id', 'gpt-realtime-2.1'),
                voice_id=settings.get('openai_voice_id', 'alloy'),
                system_prompt=SYSTEM_PROMPT,
                input_device_index=microphone_index,
                output_device_index=speaker_index,
                api_key=api_key,
                acoustic_tail_ms=settings.get('openai_acoustic_tail_ms', 700),
                barge_in_enabled=settings.get('openai_barge_in_enabled', True),
            )
            print(f"🔊 Using OpenAI Realtime voice model (voice={settings.get('openai_voice_id', 'alloy')})")
    
    if voice_model != 'openai_realtime':
        # Default: use Nova Sonic client
        nova_client = NovaSonicClient(
            voice_id=voice_id,
            system_prompt=SYSTEM_PROMPT,
            input_device_index=microphone_index,
            output_device_index=speaker_index
        )
        print(f"🔊 Using Nova Sonic voice model (voice={voice_id})")
    
    # Set up callbacks
    # Conversation history for memory storage
    _conversation_turns = []
    _pending_turns = []  # Buffer for batching memory events
    _memory_flush_lock = threading.Lock()
    
    def _try_auto_enroll(text):
        """Placeholder — enrollment is handled by Nova Sonic calling the enroll_face tool."""
        pass
    
    def _flush_memory_turns():
        """Flush pending conversation turns to AgentCore Memory in background."""
        global strands_agent, deepface_analyzer
        
        with _memory_flush_lock:
            if not _pending_turns:
                return
            turns_to_store = _pending_turns.copy()
            _pending_turns.clear()
        
        if not strands_agent or not strands_agent._memory_client:
            print("   ⚠️  Memory flush skipped: no memory client")
            return
        
        # Use identified person name, or "visitor" as fallback
        person_name = deepface_analyzer.get_identity()[0] if deepface_analyzer else None
        if not person_name:
            person_name = "visitor"
        
        try:
            from datetime import datetime
            session_id = f"session_{person_name}_{int(time.time())}"
            
            payload = []
            for turn in turns_to_store:
                payload.append({
                    "conversational": {
                        "content": {"text": turn["text"]},
                        "role": turn["role"]
                    }
                })
            
            strands_agent._memory_client.create_event(
                memoryId=strands_agent.memory_resource_id,
                actorId=person_name,
                sessionId=session_id,
                eventTimestamp=datetime.now(),
                payload=payload
            )
            print(f"💾 Stored {len(turns_to_store)} turns in memory for '{person_name}'")
        except Exception as e:
            print(f"⚠️  Failed to store memory event: {e}")
    
    def on_user_text(text):
        print(f"\n👤 User: {text}")
        _conversation_turns.append({"role": "USER", "text": text})
        _pending_turns.append({"role": "USER", "text": text})
        
        # Auto-enroll: if there's an unrecognized face and user says their name
        if face_recognition_system and face_tracker and deepface_analyzer:
            identity, _ = deepface_analyzer.get_identity()
            if identity is None:
                # Try to detect a name introduction
                _try_auto_enroll(text)
    
    def on_assistant_text(text):
        global current_speech_text
        
        print(f"\n🤖 Assistant: {text}")
        _conversation_turns.append({"role": "ASSISTANT", "text": text})
        _pending_turns.append({"role": "ASSISTANT", "text": text})
        
        # Flush to memory after each user+assistant exchange
        # Count actual user turns in pending (not fragments)
        user_turns = sum(1 for t in _pending_turns if t["role"] == "USER")
        if user_turns >= 1 and len(_pending_turns) >= 2:
            threading.Thread(target=_flush_memory_turns, daemon=True).start()
        
        # Store current speech text for visualization
        current_speech_text = text
    
    # Audio-driven mouth animation callback
    # Track callback execution
    callback_count = [0]
    _last_emit_time = [0]  # Throttle SocketIO emits

    # Phoneme-accurate viseme driver for the VRM avatar. The physical jaw servo
    # still uses the amplitude-based opening below; this only enriches the
    # on-screen mouth shapes. Falls back to amplitude-only if unavailable.
    try:
        import viseme_driver
        _analyzer_kind = settings.get('viseme_analyzer', 'headaudio')
        viseme_analyzer = viseme_driver.create_analyzer(_analyzer_kind, input_rate=24000)
        viseme_analyzer.reset()
        logger.info(f"Viseme analyzer active: {type(viseme_analyzer).__name__}")
    except Exception as e:
        logger.warning(f"Viseme analyzer unavailable, using amplitude only: {e}")
        viseme_analyzer = None
    _last_frame = [None]

    def on_audio_chunk(audio_bytes):
        """Process audio chunk for real-time mouth animation.
        
        Splits large chunks into 20ms sub-chunks for smoother animation updates.
        Throttles SocketIO emits to max ~20/sec to avoid overwhelming the UI.
        """
        global is_speaking, audio_mouth_controller, last_audio_time, current_speech_text
        
        # Update last audio time
        last_audio_time = time.time()
        callback_count[0] += 1

        # Feed the full chunk to the phoneme viseme driver (it buffers/frames
        # internally at ~62 Hz). Keep the latest frame for the throttled emit.
        if viseme_analyzer is not None:
            try:
                frames = viseme_analyzer.push_pcm16(audio_bytes)
                if frames:
                    _last_frame[0] = frames[-1]
            except Exception as e:
                logger.debug(f"Viseme analyzer error: {e}")
        
        # Split into 20ms sub-chunks (480 samples at 24kHz, 960 bytes)
        SUB_CHUNK_BYTES = 960
        offset = 0
        while offset < len(audio_bytes):
            sub_chunk = audio_bytes[offset:offset + SUB_CHUNK_BYTES]
            offset += SUB_CHUNK_BYTES
            
            if len(sub_chunk) < 64:  # Skip tiny remnants
                continue
            
            # Process audio and get mouth opening
            opening = audio_mouth_controller.process_audio_chunk(sub_chunk)
            viseme = audio_mouth_controller.get_viseme_from_opening(opening)
            
            # Update speaking state
            is_speaking = opening > 3
            
            # Throttle UI updates to ~30/sec (every 33ms). Matches the finer
            # ~40ms sliced-playback viseme cadence for smoother mouth motion.
            now = time.time()
            if (now - _last_emit_time[0]) >= 0.033:
                _last_emit_time[0] = now
                frame = _last_frame[0]
                vrm = frame.vrm if frame is not None else None
                vrm_weight = frame.weight if frame is not None else None
                # Lip-sync compensation: delay the mouth to match audible
                # playback. Manual override wins; otherwise use the measured
                # speaker output-buffer latency.
                _override = settings.get('viseme_sync_delay_ms')
                sync_delay_ms = (int(_override) if _override is not None
                                 else getattr(nova_client, 'output_latency_ms', 0))
                update_mouth(viseme, current_speech_text, opening / 100.0,
                             vrm=vrm, vrm_weight=vrm_weight,
                             sync_delay_ms=sync_delay_ms)
            
            # Control servo directly with opening percentage
            control_jaw_servo_direct(opening)
    
    nova_client.on_user_text = on_user_text
    nova_client.on_assistant_text = on_assistant_text
    nova_client.on_audio_chunk = on_audio_chunk  # Real-time audio processing
    
    # Register tool use callback if tool_handler is available
    if tool_handler:
        async def on_tool_use(tool_name, tool_use_id, parameters):
            """Handle tool use from the active voice client."""
            try:
                result = await tool_handler.handle_tool_use(tool_name, parameters)
                # Send the result back to the model
                await nova_client.send_tool_result(tool_use_id, result)
            except Exception as e:
                logger.error(f"Tool use error ({tool_name}): {e}")
                await nova_client.send_tool_result(tool_use_id, {
                    "error": "execution_failed",
                    "message": str(e)
                })
                return {"error": "execution_failed", "message": str(e)}
        
        nova_client.on_tool_use = on_tool_use
    
    # Background thread to close mouth after speech ends
    def monitor_speech_end():
        """Monitor for end of speech and close mouth"""
        global is_speaking, last_audio_time, audio_mouth_controller, jaw_position
        
        while is_running:
            # Check if audio has stopped (no chunks for 0.5 seconds)
            time_since_audio = time.time() - last_audio_time
            if is_speaking and time_since_audio > 0.5:
                print(f"🔒 Speech ended - closing mouth (jaw was at {jaw_position}%, {callback_count[0]} callbacks, {time_since_audio:.1f}s since last audio)")
                is_speaking = False
                callback_count[0] = 0  # Reset counter
                
                # Reset audio controller
                audio_mouth_controller.reset()
                
                # Force close mouth in visualization
                update_mouth('CLOSED', '')
                
                # Gradually close servo for smooth motion
                if SERVO_AVAILABLE:
                    current_angle = settings['jaw_close_angle'] + (settings['jaw_open_angle'] - settings['jaw_close_angle']) * (jaw_position / 100.0)
                    target_angle = settings['jaw_close_angle']
                    
                    # Smooth close over 0.2 seconds
                    steps = 10
                    for i in range(steps):
                        angle = current_angle + (target_angle - current_angle) * ((i + 1) / steps)
                        servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(angle)
                        time.sleep(0.02)
                    
                    # Final position - hold closed (send multiple times to ensure it reaches)
                    for _ in range(5):
                        servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(target_angle)
                        time.sleep(0.05)
                    jaw_position = 0
                    print(f"✅ Servo closed to {target_angle}°")
                else:
                    # No servo - just update position
                    control_jaw_servo_direct(0)
                    jaw_position = 0
            
            time.sleep(0.1)
    
    # Start speech monitor thread
    monitor_thread = threading.Thread(target=monitor_speech_end, daemon=True)
    monitor_thread.start()
    
    try:
        # Start session with tool configuration if available
        tool_config = None
        if tool_handler:
            tool_config = tool_handler.get_tool_config_json()
        
        await nova_client.start_session(tool_config=tool_config)
        
        # Start audio tasks
        playback_task = asyncio.create_task(nova_client.play_audio())
        capture_task = asyncio.create_task(nova_client.capture_audio())
        
        # Trigger greeting when session starts if face tracking is active
        if deepface_analyzer and face_tracking_enabled:
            # Check if identity is already known (DeepFace has been running since app start)
            identity, confidence = deepface_analyzer.get_identity()
            
            # If not identified yet, wait up to 5 seconds polling every 0.5s
            # DeepFace runs every 1.5s, so we need ~2-3 cycles to get a confident match
            if identity is None:
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    identity, confidence = deepface_analyzer.get_identity()
                    if identity is not None:
                        break
            
            print(f"🎬 Session started — current identity: '{identity}' (confidence={confidence:.2f})")
            
            # Only send greeting if the model hasn't already started responding
            # (user might have spoken during the identity polling wait)
            if last_audio_time == 0:
                last_greeting_time[identity] = time.time()
                _last_any_greeting_time = time.time()
                await _trigger_greeting(identity, confidence, False)
            else:
                print("   ⏭️  Skipping greeting (model already responding to user)")
        
        # Wait until stopped or until the provider reports a terminal failure.
        while is_running and nova_client.is_active:
            await asyncio.sleep(0.1)

        if is_running and not nova_client.is_active:
            logger.error("Voice provider became inactive; ending the current session")
            is_running = False
        
        # Cancel tasks gracefully
        print("🛑 Cancelling audio tasks...")
        playback_task.cancel()
        capture_task.cancel()
        
        # Wait for tasks to finish with exception suppression
        try:
            await asyncio.gather(playback_task, capture_task, return_exceptions=True)
        except Exception:
            pass  # Ignore cancellation errors
        
        # End session
        try:
            await nova_client.end_session()
        except Exception:
            pass  # Ignore session end errors
        
        # Cancel response if still active (NovaSonicClient-specific)
        try:
            if hasattr(nova_client, 'response') and nova_client.response and not nova_client.response.done():
                nova_client.response.cancel()
        except Exception:
            pass  # Ignore response cancellation errors
        
        print("✅ Voice assistant stopped")
        
        # Flush any remaining conversation turns to memory
        if _pending_turns:
            _flush_memory_turns()
        
        # FORCE close mouth when stopping
        global jaw_position, is_speaking
        is_speaking = False
        update_mouth('CLOSED', '')
        control_jaw_servo_direct(0)
        
        # Ensure jaw is fully closed - force to 0 degrees
        if SERVO_AVAILABLE:
            print("🔧 Forcing jaw servo to 0° on stop...")
            for _ in range(10):
                servo_kit.servo[JAW_CHANNEL].angle = 0
                time.sleep(0.05)
            jaw_position = 0
            print("✅ Jaw servo set to 0°")
        
        print("🔒 Mouth forcefully closed")
    
    except Exception as e:
        print(f"❌ Error in voice assistant: {e}")
        import traceback
        traceback.print_exc()
    finally:
        nova_client = None
        _voice_assistant_loop = None


def start_voice_assistant():
    """Start voice assistant in background thread"""
    global is_running
    
    if is_running:
        print("⚠️  Voice assistant already running")
        return
    
    is_running = True
    
    # Run in new thread with new event loop
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_voice_assistant())
        loop.close()
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()


def stop_voice_assistant():
    """Stop voice assistant cleanly regardless of which backend is active.
    
    Calls end_session() on whichever Voice_Client (NovaSonicClient or
    OpenAIRealtimeClient) is currently active, handling errors gracefully
    so that shutdown never raises unhandled exceptions.
    """
    global is_running, nova_client
    
    if not is_running:
        print("⚠️  Voice assistant not running")
        return
    
    print("🛑 Stopping voice assistant...")
    is_running = False
    
    # Actively end the session on whichever client is active
    client = nova_client
    if client is not None:
        try:
            # Schedule end_session in the voice assistant event loop if available
            if _voice_assistant_loop and not _voice_assistant_loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    client.end_session(), _voice_assistant_loop
                )
                # Wait briefly for clean shutdown, but don't block indefinitely
                try:
                    future.result(timeout=5.0)
                except Exception:
                    pass  # Timeout or error during end_session — acceptable
            else:
                # No event loop available; try to run end_session directly
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(client.end_session())
                    loop.close()
                except Exception:
                    pass  # Best-effort shutdown
        except Exception as e:
            # Catch-all: never let shutdown raise unhandled exceptions
            print(f"⚠️  Error during client shutdown: {e}")
    
    print("✅ Voice assistant stop signal sent")


# --- Face Recognition Callbacks and Greeting Flow ---

def _on_identity_changed(person_name, confidence):
    """Callback when identity changes (called from DeepFace background thread).
    
    Triggers personalized greeting flow with cooldown enforcement.
    All operations are non-blocking to avoid impacting the audio stream.
    """
    global last_greeting_time, last_seen_time, nova_client, _voice_assistant_loop
    
    current_time = time.time()
    
    print(f"🔍 Identity event: person='{person_name}', confidence={confidence:.2f}")
    
    # Emit identity update to web UI (fire-and-forget in background)
    def _emit_identity():
        try:
            socketio.emit('identity_update', {
                'person_name': person_name,
                'confidence': confidence
            })
        except Exception as e:
            logger.debug(f"Failed to emit identity update: {e}")
    
    threading.Thread(target=_emit_identity, daemon=True).start()
    
    # Check if voice assistant is running
    if not is_running or nova_client is None or _voice_assistant_loop is None:
        print(f"   ⏭️  Skipping greeting (voice assistant not active)")
        return
    
    # NEVER inject greetings mid-conversation from the identity callback.
    # Greetings only happen at session start via _trigger_greeting in run_voice_assistant.
    # The identity callback's only job during active sessions is to emit to the web UI.
    return
    
    # Check greeting cooldown (30 seconds)
    cooldown = settings.get('greeting_cooldown_seconds', 30)
    last_greeted = last_greeting_time.get(person_name, 0)
    if (current_time - last_greeted) < cooldown:
        logger.debug(f"Greeting cooldown active for '{person_name}' ({current_time - last_greeted:.1f}s < {cooldown}s)")
        return
    
    # Determine if this is a welcome-back greeting (absent > 5 minutes)
    welcome_back_threshold = settings.get('welcome_back_threshold_seconds', 300)
    last_seen = last_seen_time.get(person_name, 0)
    is_welcome_back = (last_seen > 0 and (current_time - last_seen) > welcome_back_threshold)
    
    # Update greeting and seen timestamps
    last_greeting_time[person_name] = current_time
    last_seen_time[person_name] = current_time
    _last_any_greeting_time = current_time
    
    # Schedule the async greeting in the voice assistant event loop
    asyncio.run_coroutine_threadsafe(
        _trigger_greeting(person_name, confidence, is_welcome_back),
        _voice_assistant_loop
    )


async def _trigger_greeting(person_name, confidence, is_welcome_back):
    """Trigger personalized greeting via Nova Sonic 2.
    
    Memory recall runs in a thread executor to avoid blocking the audio stream.
    """
    global nova_client, strands_agent, emotion_detector
    
    if nova_client is None or not nova_client.is_active:
        return
    
    # Get current emotion (thread-safe, no I/O — safe to call directly)
    emotion_state = "neutral"
    if emotion_detector:
        state = emotion_detector.get_current_state()
        emotion_state = state.get("emotion", "neutral")
    
    if person_name is None:
        # Unknown face — generic greeting and enroll via tool
        greeting_prompt = (
            f"You notice someone new approaching. They seem {emotion_state}. "
            "Greet them warmly and ask what their name is. "
            "Once they tell you their name, immediately use the enroll_face tool with their name "
            "so you can recognize them next time. Keep it brief and friendly."
        )
    elif is_welcome_back:
        # Welcome-back greeting for person absent > 5 minutes
        # Run memory recall in executor to avoid blocking audio stream
        context = ""
        if strands_agent and strands_agent.is_initialized:
            try:
                loop = asyncio.get_running_loop()
                context = await loop.run_in_executor(
                    None, strands_agent.get_greeting_context, person_name, emotion_state
                )
            except Exception as e:
                logger.warning(f"Failed to get greeting context: {e}")
        
        greeting_prompt = (
            f"Welcome back {person_name}! They've been away for a while and just returned. "
            f"They appear to be {emotion_state}. "
            f"{context} "
            "Give them a warm welcome-back greeting that references your last interaction if possible. "
            "Keep it to 1-2 sentences."
        )
    else:
        # Normal personalized greeting
        # Run memory recall in executor to avoid blocking audio stream
        context = ""
        if strands_agent and strands_agent.is_initialized:
            try:
                loop = asyncio.get_running_loop()
                context = await loop.run_in_executor(
                    None, strands_agent.get_greeting_context, person_name, emotion_state
                )
            except Exception as e:
                logger.warning(f"Failed to get greeting context: {e}")
        
        greeting_prompt = (
            f"Greet {person_name} warmly by name. They appear to be {emotion_state}. "
            f"{context} "
            "Keep the greeting natural and brief (1-2 sentences)."
        )
    
    try:
        # Don't send if a response is likely still in progress
        # (the greeting at session start already triggered one)
        await nova_client.send_text_input(greeting_prompt)
        logger.info(f"Greeting triggered for '{person_name}' (welcome_back={is_welcome_back})")
    except Exception as e:
        logger.error(f"Failed to send greeting prompt: {e}")


def _on_emotion_changed(emotion, confidence):
    """Callback when emotion changes (called from DeepFace background thread).
    
    Emits to web UI and notifies Nova Sonic of the emotional shift.
    """
    global nova_client, _voice_assistant_loop
    
    print(f"😊 Emotion changed: {emotion} (confidence={confidence:.2f})")
    
    # Emit emotion update to web UI (fire-and-forget in background)
    def _emit_emotion():
        try:
            socketio.emit('emotion_update', {
                'emotion': emotion,
                'confidence': confidence
            })
        except Exception as e:
            logger.debug(f"Failed to emit emotion update: {e}")
    
    threading.Thread(target=_emit_emotion, daemon=True).start()
    
    # Notify Nova Sonic of significant emotion changes during active conversation
    # DISABLED: This was causing "conversation already has active response" errors
    # by injecting text while the model was still speaking. The model can use the
    # get_emotion tool to check emotions when it wants to.
    pass


def _store_conversation_memory(person_name, conversation_turns):
    """Store conversation turns in AgentCore Memory as an event.
    
    The LTM strategies (session summaries, user preferences, semantic facts)
    will automatically extract relevant information from the conversation.
    """
    global strands_agent
    
    if not strands_agent or not strands_agent._memory_client:
        return
    
    if not conversation_turns:
        return
    
    try:
        from datetime import datetime
        
        session_id = f"session_{person_name}_{int(time.time())}"
        
        # Build payload from conversation turns
        payload = []
        for turn in conversation_turns:
            payload.append({
                "conversational": {
                    "content": {"text": turn["text"]},
                    "role": turn["role"]
                }
            })
        
        strands_agent._memory_client.create_event(
            memoryId=strands_agent.memory_resource_id,
            actorId=person_name,
            sessionId=session_id,
            eventTimestamp=datetime.now(),
            payload=payload
        )
        logger.info(f"Stored {len(conversation_turns)} conversation turns for '{person_name}'")
    except Exception as e:
        logger.warning(f"Failed to store conversation memory for '{person_name}': {e}")


# --- WebSocket Event Handlers for Enrollment and Recognition Control ---

@socketio.on('enroll')
def handle_enroll(data):
    """Handle enrollment request from web UI."""
    global face_recognition_system, face_tracker
    
    name = data.get('name', '').strip()
    if not name:
        socketio.emit('enroll_result', {'success': False, 'message': 'Name cannot be empty'})
        return
    
    if face_recognition_system is None:
        socketio.emit('enroll_result', {'success': False, 'message': 'Face recognition not initialized'})
        return
    
    if face_tracker is None or face_tracker.camera is None:
        socketio.emit('enroll_result', {'success': False, 'message': 'Camera not available'})
        return
    
    # Run enrollment (uses the face tracker's camera)
    try:
        success, message = face_recognition_system.enroll(name, face_tracker.camera)
        socketio.emit('enroll_result', {'success': success, 'message': message})
        print(f"📸 Enrollment {'succeeded' if success else 'failed'} for '{name}': {message}")
    except Exception as e:
        socketio.emit('enroll_result', {'success': False, 'message': f'Enrollment error: {str(e)}'})
        print(f"❌ Enrollment error for '{name}': {e}")


@socketio.on('enrolled')
def handle_enrolled():
    """Handle request to list all enrolled people."""
    if face_recognition_system is None:
        socketio.emit('enrolled_list', {'people': []})
        return
    
    people = face_recognition_system.list_enrolled()
    socketio.emit('enrolled_list', {'people': people})


@socketio.on('remove_person')
def handle_remove_person(data):
    """Handle request to remove a person from the database."""
    global face_recognition_system
    
    name = data.get('name', '').strip()
    if not name:
        socketio.emit('remove_result', {'success': False, 'message': 'Name cannot be empty'})
        return
    
    if face_recognition_system is None:
        socketio.emit('remove_result', {'success': False, 'message': 'Face recognition not initialized'})
        return
    
    success = face_recognition_system.remove_person(name)
    message = f"Removed '{name}'" if success else f"'{name}' not found in database"
    socketio.emit('remove_result', {'success': success, 'message': message})
    print(f"🗑️  Remove person '{name}': {'success' if success else 'not found'}")


@socketio.on('toggle_recognition')
def handle_toggle_recognition(data):
    """Handle request to enable/disable face recognition."""
    global face_recognition_enabled, deepface_analyzer, settings
    
    enabled = data.get('enabled', False)
    face_recognition_enabled = enabled
    settings['face_recognition_enabled'] = enabled
    save_settings(settings)
    
    if enabled and deepface_analyzer and not deepface_analyzer._running:
        deepface_analyzer.start()
        print("✅ Face recognition enabled")
    elif not enabled and deepface_analyzer and deepface_analyzer._running:
        deepface_analyzer.stop()
        print("⏹️  Face recognition disabled")
    
    socketio.emit('recognition_status', {'enabled': enabled})


def process_control_commands():
    """Process control commands from web interface"""
    global voice_id, microphone_index, speaker_index, is_muted, speech_speed, settings, SERVO_AVAILABLE
    
    while True:
        cmd = get_control_command(timeout=0.1)
        
        if cmd:
            action = cmd.get('action')
            value = cmd.get('value')
            
            print(f"📨 Control: {action} = {value}")
            
            settings_changed = False
            
            if action == 'start':
                start_voice_assistant()
            
            elif action == 'stop':
                stop_voice_assistant()
            
            elif action == 'mute':
                is_muted = value
                print(f"🔇 Muted: {is_muted}")
            
            elif action == 'set_voice':
                voice_id = value
                settings['voice_id'] = value
                settings_changed = True
                print(f"🗣️  Nova voice changed to: {voice_id} (takes effect on next session start)")

            elif action == 'set_openai_voice':
                settings['openai_voice_id'] = value
                settings_changed = True
                print(f"🗣️  OpenAI voice changed to: {value} (takes effect on next session start)")
            
            elif action == 'set_voice_model':
                settings['voice_model'] = value
                settings_changed = True
                print(f"🤖 Voice model changed to: {value} (takes effect on next session start)")
            
            elif action == 'set_microphone':
                microphone_index = value
                settings['microphone_index'] = value
                settings_changed = True
                print(f"🎤 Microphone changed to index: {microphone_index}")
            
            elif action == 'set_speaker':
                speaker_index = value
                settings['speaker_index'] = value
                settings_changed = True
                print(f"🔊 Speaker changed to index: {speaker_index}")
            
            elif action == 'set_speech_speed':
                speech_speed = value
                settings['speech_speed'] = value
                settings_changed = True
                print(f"⚡ Speech speed changed to: {speech_speed} chars/sec")
            
            elif action == 'set_jaw_stop_angle':
                settings['jaw_stop_angle'] = value
                settings_changed = True
                print(f"🦴 Jaw stop angle: {value}°")
            
            elif action == 'set_jaw_open_angle':
                settings['jaw_open_angle'] = value
                settings_changed = True
                print(f"🦴 Jaw open angle: {value}°")
            
            elif action == 'set_jaw_close_angle':
                settings['jaw_close_angle'] = value
                settings_changed = True
                print(f"🦴 Jaw close angle: {value}°")
            
            elif action == 'set_jaw_pulse_duration':
                settings['jaw_pulse_duration'] = value
                settings_changed = True
                print(f"🦴 Jaw pulse duration: {value}s")
            
            elif action == 'test_jaw':
                print("🧪 Testing jaw servo...")
                if SERVO_AVAILABLE:
                    try:
                        # Test sequence for standard servo: close -> open -> close
                        print("  Closing jaw...")
                        servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_close_angle'])
                        time.sleep(0.8)
                        print("  Opening jaw...")
                        servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_open_angle'])
                        time.sleep(0.8)
                        print("  Closing jaw...")
                        servo_kit.servo[JAW_CHANNEL].angle = clamp_angle(settings['jaw_close_angle'])
                        time.sleep(0.5)
                        print("✅ Jaw test complete")
                    except Exception as e:
                        if "No such device" in str(e) or "disconnected" in str(e).lower():
                            print(f"⚠️  USB device disconnected - disabling servo control")
                            SERVO_AVAILABLE = False
                        else:
                            print(f"⚠️  Test failed: {e}")
                else:
                    print("⚠️  Servo not available")
            
            elif action == 'test_eye_servo':
                channel = cmd.get('channel', 0)
                angle = cmd.get('angle', 90)
                print(f"🧪 Testing eye servo channel {channel} at {angle}°")
                if SERVO_AVAILABLE and 0 <= channel <= 7:
                    try:
                        clamped_angle = clamp_angle(angle)
                        servo_kit.servo[channel].angle = clamped_angle
                        print(f"✅ Eye servo {channel} moved to {clamped_angle}°")
                    except Exception as e:
                        print(f"❌ Error moving servo {channel}: {e}")
                else:
                    print("⚠️  Servo not available or invalid channel")
            
            elif action == 'center_all_eyes':
                print("🎯 Centering all eye servos...")
                if SERVO_AVAILABLE:
                    for channel in range(8):  # Channels 0-7
                        try:
                            # Use saved center angle if available, otherwise 90
                            center_angle = settings.get('eye_servos', {}).get(str(channel), {}).get('center_angle', 90)
                            servo_kit.servo[channel].angle = clamp_angle(center_angle)
                            time.sleep(0.1)  # Small delay between servos
                        except Exception as e:
                            print(f"❌ Error centering servo {channel}: {e}")
                    print("✅ All eye servos centered")
                else:
                    print("⚠️  Servo not available")
            
            elif action == 'save_eye_config':
                channel = cmd.get('channel', 0)
                min_angle = cmd.get('min_angle', 0)
                max_angle = cmd.get('max_angle', 180)
                center_angle = cmd.get('center_angle', 90)
                
                # Initialize eye_servos dict if not exists
                if 'eye_servos' not in settings:
                    settings['eye_servos'] = {}
                
                # Save config for this channel
                settings['eye_servos'][str(channel)] = {
                    'min_angle': min_angle,
                    'max_angle': max_angle,
                    'center_angle': center_angle
                }
                
                save_settings(settings)
                print(f"💾 Saved config for eye servo {channel}: min={min_angle}°, max={max_angle}°, center={center_angle}°")
            
            elif action == 'load_eye_config':
                channel = cmd.get('channel', 0)
                
                # Get config for this channel or use defaults
                eye_config = settings.get('eye_servos', {}).get(str(channel), {
                    'min_angle': 0,
                    'max_angle': 180,
                    'center_angle': 90
                })
                
                # Send config back to web interface
                socketio.emit('eye_config_loaded', eye_config)
                print(f"📂 Loaded config for eye servo {channel}")
            
            elif action == 'sweep_eye_servo':
                channel = cmd.get('channel', 0)
                min_angle = clamp_angle(cmd.get('min_angle', 0))
                max_angle = clamp_angle(cmd.get('max_angle', 180))
                center_angle = clamp_angle(cmd.get('center_angle', 90))
                
                print(f"🔄 Sweeping eye servo {channel}: {min_angle}° → {max_angle}° → {center_angle}°")
                if SERVO_AVAILABLE and 0 <= channel <= 7:
                    try:
                        # Move to min
                        servo_kit.servo[channel].angle = min_angle
                        time.sleep(0.5)
                        
                        # Sweep to max
                        steps = 20
                        for i in range(steps + 1):
                            angle = min_angle + (max_angle - min_angle) * i / steps
                            servo_kit.servo[channel].angle = clamp_angle(angle)
                            time.sleep(0.05)
                        
                        time.sleep(0.5)
                        
                        # Return to center
                        servo_kit.servo[channel].angle = center_angle
                        print(f"✅ Sweep complete for servo {channel}")
                    except Exception as e:
                        print(f"❌ Error sweeping servo {channel}: {e}")
                else:
                    print("⚠️  Servo not available or invalid channel")
            
            elif action == 'toggle_face_tracking':
                global face_tracking_enabled
                
                print(f"🔄 Received toggle_face_tracking command: {value}")
                
                if value:  # Enable
                    print("🎥 Attempting to start face tracking...")
                    result = start_face_tracking()
                    print(f"🎥 Start face tracking result: {result}")
                    
                    if result:
                        settings['face_tracking_enabled'] = True
                        settings_changed = True
                        update_face_tracking_status(True)
                        print("✅ Face tracking enabled")
                    else:
                        update_face_tracking_status(False)
                        print("❌ Failed to enable face tracking")
                else:  # Disable
                    print("🛑 Stopping face tracking...")
                    stop_face_tracking()
                    settings['face_tracking_enabled'] = False
                    settings_changed = True
                    update_face_tracking_status(False)
                    print("⏹️  Face tracking disabled")
            
            elif action == 'set_servo_config':
                settings['servo_config'] = value
                settings_changed = True
                print(f"📐 Servo config changed to: {value}")
                
                # Restart face tracking if it's enabled
                if face_tracking_enabled:
                    print("🔄 Restarting face tracking with new config...")
                    stop_face_tracking()
                    start_face_tracking()
            
            elif action == 'set_camera_index':
                settings['camera_index'] = value
                settings_changed = True
                print(f"📷 Camera index changed to: {value}")
                
                # Restart face tracking if it's enabled
                if face_tracking_enabled:
                    print("🔄 Restarting face tracking with new camera...")
                    stop_face_tracking()
                    start_face_tracking()
            
            # Save settings when they change
            if settings_changed:
                save_settings(settings)
        
        time.sleep(0.05)


def face_tracking_loop():
    """Face tracking loop - runs in separate thread"""
    global face_tracker, eye_controller, face_tracking_enabled, settings, server_running, last_blink_time, jaw_position, is_speaking
    
    print("👀 Face tracking loop started")
    frame_count = 0
    
    while server_running:
        if not face_tracking_enabled or face_tracker is None:
            time.sleep(0.1)
            continue
        
        try:
            frame_count += 1
            
            # Track face
            tracking_data = face_tracker.track_face()
            
            # Feed frame to DeepFace analyzer for background recognition/emotion
            if tracking_data and deepface_analyzer and face_recognition_enabled:
                frame = tracking_data.get('frame')
                if frame is not None:
                    deepface_analyzer.update_frame(frame)
            
            if tracking_data:
                if tracking_data.get('found'):
                    # Face found
                    face_x = tracking_data['center_x']
                    face_y = tracking_data['center_y']
                    frame_width = tracking_data['frame_width']
                    frame_height = tracking_data['frame_height']
                    
                    # Calculate eye angles based on face position
                    # Map face position to servo angles (0-180, with 90 as center)
                    def map_value(value, in_min, in_max, out_min, out_max):
                        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
                    
                    # Calculate angles (inverted X for natural tracking)
                    eye_x_angle = map_value(face_x, 0, frame_width, 120, 60)  # Inverted
                    eye_y_angle = map_value(face_y, 0, frame_height, 60, 120)  # Normal (not inverted)
                    
                    # Send virtual eye positions to web UI
                    update_eyes({
                        'left_eye_x': eye_x_angle,
                        'left_eye_y': eye_y_angle,
                        'right_eye_x': eye_x_angle,
                        'right_eye_y': eye_y_angle,
                        'gaze_x': (eye_x_angle - 90) / 30,
                        'gaze_y': (eye_y_angle - 90) / 30,
                    })
                    
                    # Move physical eyes if servos available
                    if eye_controller is not None:
                        eye_controller.track_position(face_x, face_y, frame_width, frame_height)
                        eye_controller.blink_eyes()
                    
                    # Virtual blinking (works without servos)
                    current_time = time.time()
                    time_since_blink = current_time - last_blink_time
                    if time_since_blink > blink_interval:
                        if random.random() < 0.5:  # 50% chance when interval passed
                            trigger_blink()
                            last_blink_time = current_time
                else:
                    # No face detected
                    pass
            else:
                print("⚠️  No tracking data received from camera")
            
            # Safety check: close jaw if not speaking (only check every second)
            if frame_count % 30 == 0 and not is_speaking and SERVO_AVAILABLE and jaw_position > 0:
                print(f"⚠️  Safety: Closing jaw (was at {jaw_position}%)")
                servo_kit.servo[JAW_CHANNEL].angle = 0
                jaw_position = 0
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"⚠️  Face tracking error: {e}")
            time.sleep(0.5)
    
    print("👀 Face tracking loop stopped")


def start_face_tracking():
    """Start face tracking"""
    global face_tracker, eye_controller, face_tracking_enabled, face_tracking_thread, settings
    
    try:
        # Initialize face tracker
        face_tracker = FaceTracker()
        camera_index = settings.get('camera_index', 0)
        
        if not face_tracker.start_camera(camera_index):
            print("❌ Failed to start camera")
            return False
        
        # Initialize eye controller only if servos available
        if SERVO_AVAILABLE:
            servo_config_name = settings.get('servo_config', 'inmoov')
            servo_config = get_config(servo_config_name)
            print(f"📐 Using servo config: {servo_config.name}")
            
            eye_controller = EyeController(servo_kit, servo_config, position_callback=update_eyes)
            eye_controller.center_eyes()
            print("✅ Eye servos initialized")
        else:
            eye_controller = None
            print("⚠️  Running in camera-only mode (no servos)")
        
        # Enable tracking
        face_tracking_enabled = True
        
        # Start tracking thread if not already running
        if face_tracking_thread is None or not face_tracking_thread.is_alive():
            face_tracking_thread = threading.Thread(target=face_tracking_loop, daemon=True)
            face_tracking_thread.start()
        
        print("✅ Face tracking started")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start face tracking: {e}")
        return False


def stop_face_tracking():
    """Stop face tracking"""
    global face_tracker, eye_controller, face_tracking_enabled
    
    face_tracking_enabled = False
    
    if face_tracker:
        face_tracker.stop_camera()
        face_tracker = None
    
    if eye_controller:
        eye_controller.center_eyes()
        eye_controller = None
    
    print("⏹️  Face tracking stopped")


def main():
    global face_recognition_system, emotion_detector, deepface_analyzer, strands_agent, tool_handler, face_recognition_enabled
    
    print("=" * 60)
    print("VOICE ASSISTANT SERVER WITH WEB CONTROL")
    print("=" * 60)
    print()
    
    # Show loaded settings
    print("📋 Loaded settings:")
    print(f"   Voice: {voice_id}")
    print(f"   Microphone: {microphone_index if microphone_index is not None else 'default'}")
    print(f"   Speaker: {speaker_index if speaker_index is not None else 'default'}")
    print(f"   Speech speed: {speech_speed} chars/sec")
    print()
    
    # --- Initialize Face Recognition and Emotion Detection Components ---
    face_recognition_enabled = settings.get('face_recognition_enabled', True)
    
    if face_recognition_enabled:
        try:
            print("🧠 Initializing face recognition components...")

            # Initialize FaceRecognitionSystem
            db_path = settings.get('face_database_path', 'config/face_database.json')
            confidence_threshold = settings.get('recognition_confidence_threshold', 0.6)
            face_recognition_system = FaceRecognitionSystem(
                database_path=db_path,
                confidence_threshold=confidence_threshold
            )
            print(f"   ✅ Face Recognition System (threshold={confidence_threshold}, db={db_path})")

            # Initialize EmotionDetector
            emotion_detector = EmotionDetector(
                min_confidence=settings.get('emotion_min_confidence', 0.40),
                neutral_bias=settings.get('emotion_neutral_bias', 0.05),
                backend=settings.get('emotion_backend', 'hsemotion'),
            )
            print(f"   ✅ Emotion Detector (backend={emotion_detector.backend})")

            # Initialize DeepFaceAnalyzer
            analysis_interval = settings.get('deepface_analysis_interval', 1.5)
            deepface_analyzer = DeepFaceAnalyzer(
                face_recognition=face_recognition_system,
                emotion_detector=emotion_detector,
                analysis_interval=analysis_interval,
                emotion_smoothing_window=settings.get('emotion_smoothing_window', 3),
            )
            # Wire callbacks
            deepface_analyzer.on_identity_changed = _on_identity_changed
            deepface_analyzer.on_emotion_changed = _on_emotion_changed
            print(f"   ✅ DeepFace Analyzer (interval={analysis_interval}s)")

            # Initialize StrandsAgent
            memory_resource_id = settings.get('agentcore_memory_resource_id', '')
            aws_region = settings.get('aws_region', 'us-east-1')
            strands_agent = StrandsAgent(
                memory_resource_id=memory_resource_id,
                region=aws_region
            )
            if memory_resource_id:
                if strands_agent.initialize():
                    print(f"   ✅ Strands Agent (region={aws_region})")
                else:
                    print("   ⚠️  Strands Agent initialization failed (continuing without agent)")
            else:
                print("   ⚠️  Strands Agent skipped (no memory_resource_id configured)")

            # Initialize ToolHandler
            tool_timeout = settings.get('tool_timeout_seconds', 10)
            tool_handler = ToolHandler(
                strands_agent=strands_agent,
                emotion_detector=emotion_detector,
                get_current_identity=lambda: deepface_analyzer.get_identity()[0] if deepface_analyzer else None,
                face_recognition_system=face_recognition_system,
                get_camera=lambda: face_tracker if face_tracker else None,
                set_expression=lambda emotion, weight: update_expression(emotion, weight),
                timeout=tool_timeout
            )
            print(f"   ✅ Tool Handler (timeout={tool_timeout}s)")

            # Start the DeepFace analyzer background thread
            deepface_analyzer.start()
            print("   ✅ DeepFace Analyzer background thread started")
            print()
        except Exception as e:
            # Optional subsystems (deepface/strands/etc.) may be unavailable in a
            # lightweight install. Degrade gracefully to voice + avatar only.
            print(f"⚠️  Face recognition unavailable ({e}); continuing without it")
            logger.warning(f"Face recognition init failed: {e}")
            face_recognition_enabled = False
            face_recognition_system = None
            emotion_detector = None
            deepface_analyzer = None
            strands_agent = None
            tool_handler = None
            print()
    else:
        print("⏭️  Face recognition disabled in settings")
        print()
    
    # Set up callback for web UI to get face tracking state
    mouth_visualizer.get_face_tracking_state = lambda: face_tracking_enabled
    
    # Set up callback for web UI to get current settings
    mouth_visualizer.get_current_settings = lambda: settings

    # Set up callback for web UI to save settings via POST endpoint
    def _save_settings_from_ui(updates):
        """Apply setting updates from the web UI POST endpoint."""
        global settings
        for key, value in updates.items():
            settings[key] = value
        save_settings(settings)

    mouth_visualizer.save_current_settings = _save_settings_from_ui
    
    # Start web server
    print("🌐 Starting web server...")
    start_server()
    time.sleep(2)
    
    # Auto-start face tracking if enabled in settings
    if settings.get('face_tracking_enabled', False):
        print("👁️ Auto-starting face tracking...")
        if start_face_tracking():
            update_face_tracking_status(True)
            print("✅ Face tracking started")
        else:
            print("⚠️  Face tracking failed to start")
    
    print("\n" + "="*60)
    print("👉 OPEN THIS IN YOUR BROWSER: http://127.0.0.1:8080")
    print("="*60)
    print("\n💡 Use the web interface to:")
    print("  • Click 'Start Listening' to begin conversation")
    print("  • Click 'Stop' to end conversation")
    print("  • Open Settings to select microphone/speaker")
    print("  • Watch the animated mouth as the robot speaks")
    print("\nPress Ctrl+C to quit\n")
    
    # Process control commands
    try:
        process_control_commands()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        server_running = False
        if is_running:
            stop_voice_assistant()
            time.sleep(1)
        if face_tracking_enabled:
            stop_face_tracking()
        # Shut down face recognition components
        if deepface_analyzer:
            deepface_analyzer.stop()
            print("   ⏹️  DeepFace Analyzer stopped")
        if tool_handler:
            tool_handler.shutdown()
            print("   ⏹️  Tool Handler stopped")
        if strands_agent:
            strands_agent.shutdown()
            print("   ⏹️  Strands Agent stopped")
        print("✅ Goodbye!")


if __name__ == "__main__":
    main()
