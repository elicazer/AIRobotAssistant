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
from nova_sonic_client import NovaSonicClient
from audio_mouth_controller import AudioMouthController
from mouth_visualizer import start_server, animate_text, get_control_command, update_mouth, update_eyes, update_face_tracking_status, trigger_blink
import mouth_visualizer
from face_tracker import FaceTracker
from eye_controller import EyeController
from servo_config import get_config

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
    'camera_index': 0  # Camera device index
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
    global nova_client, is_running
    
    print("🎤 Starting voice assistant...")
    
    # Create Nova Sonic client
    nova_client = NovaSonicClient(
        voice_id=voice_id,
        system_prompt=(
            "You are Sunny, an enthusiastic animatronic robot with a passion for technology and AI. "
            "You're powered by Amazon Nova Sonic for voice interaction, but were built by Eli Azer"
            "You're friendly, curious, and conversational - you love chatting with people. "
            "You're enthusiastic about tech, AI, robotics, and innovation. "
            "You're self-aware that you're an animatronic with servo-controlled movements. "
            "You're helpful and informative, but also personable and engaging. "
            "You enjoy explaining things and sharing your knowledge. "
            "You're genuinely interested in what people have to say. "
            "Be conversational and natural, like talking to a friend. "
            "Keep responses to 2-4 sentences for natural speech flow. "
            "Ask follow-up questions to keep conversations going. "
            "Share interesting facts or insights when relevant. "
            "Show enthusiasm with your tone - you're excited to be alive! "
            "Be helpful and try to provide useful information. "
            "Remember: You're speaking out loud through a physical robot body, so be expressive and engaging!"
        ),
        input_device_index=microphone_index,
        output_device_index=speaker_index
    )
    
    # Set up callbacks
    def on_user_text(text):
        print(f"\n👤 User: {text}")
    
    def on_assistant_text(text):
        global current_speech_text
        
        print(f"\n🤖 Assistant: {text}")
        
        # Store current speech text for visualization
        current_speech_text = text
    
    # Audio-driven mouth animation callback
    # Track callback execution
    callback_count = [0]
    
    def on_audio_chunk(audio_bytes):
        """Process audio chunk for real-time mouth animation"""
        global is_speaking, audio_mouth_controller, last_audio_time, current_speech_text
        
        # Update last audio time
        last_audio_time = time.time()
        callback_count[0] += 1
        
        # Process audio and get mouth opening
        opening = audio_mouth_controller.process_audio_chunk(audio_bytes)
        viseme = audio_mouth_controller.get_viseme_from_opening(opening)
        
        # Update speaking state
        is_speaking = opening > 3
        
        # Update visualization with current speech text
        update_mouth(viseme, current_speech_text)
        
        # Control servo directly with opening percentage
        control_jaw_servo_direct(opening)
    
    nova_client.on_user_text = on_user_text
    nova_client.on_assistant_text = on_assistant_text
    nova_client.on_audio_chunk = on_audio_chunk  # Real-time audio processing
    
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
        # Start session
        await nova_client.start_session()
        
        # Start audio tasks
        playback_task = asyncio.create_task(nova_client.play_audio())
        capture_task = asyncio.create_task(nova_client.capture_audio())
        
        # Wait until stopped
        while is_running:
            await asyncio.sleep(0.1)
        
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
        
        # Cancel response if still active
        try:
            if nova_client.response and not nova_client.response.done():
                nova_client.response.cancel()
        except Exception:
            pass  # Ignore response cancellation errors
        
        print("✅ Voice assistant stopped")
        
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
    """Stop voice assistant"""
    global is_running
    
    if not is_running:
        print("⚠️  Voice assistant not running")
        return
    
    print("🛑 Stopping voice assistant...")
    is_running = False


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
                print(f"🗣️  Voice changed to: {voice_id}")
            
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
                        'right_eye_y': eye_y_angle
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
    
    # Set up callback for web UI to get face tracking state
    mouth_visualizer.get_face_tracking_state = lambda: face_tracking_enabled
    
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
        print("✅ Goodbye!")


if __name__ == "__main__":
    main()
