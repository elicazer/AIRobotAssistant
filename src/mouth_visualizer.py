"""
Visual Mouth Animation for Lip Sync Testing
Flask web app that shows animated mouth movements
"""

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from flask_socketio import SocketIO, emit
import threading
import queue
import time
import os

# Get the project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
AVATAR_DIR = os.path.realpath(os.path.join(TEMPLATE_DIR, 'avatars'))

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = 'robot-mouth-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Queue for mouth updates
mouth_queue = queue.Queue()
current_viseme = 'CLOSED'
current_text = ''
current_intensity = 0.0

# Callback to get current face tracking state
get_face_tracking_state = None

# Callback to get current settings
get_current_settings = None

# Callback to save settings (receives dict of fields to update)
save_current_settings = None


@app.route('/')
def index():
    """Main page with mouth animation"""
    return render_template('mouth.html')


@app.route('/avatar')
def avatar():
    """Render the VRM avatar driven by the existing Socket.IO events."""
    return render_template('avatar_test.html', model_url=url_for('avatar_model'))


@app.route('/avatar/model')
def avatar_model():
    """Serve the configured local VRM without exposing arbitrary files."""
    current_settings = get_current_settings() if get_current_settings else {}
    filename = current_settings.get('avatar_vrm_file', 'sample.vrm')

    if (
        not isinstance(filename, str)
        or os.path.basename(filename) != filename
        or not filename.lower().endswith('.vrm')
    ):
        abort(404)

    model_path = os.path.realpath(os.path.join(AVATAR_DIR, filename))
    try:
        is_contained = os.path.commonpath([AVATAR_DIR, model_path]) == AVATAR_DIR
    except ValueError:
        is_contained = False

    if not is_contained or not os.path.isfile(model_path):
        abort(404)

    return send_file(model_path, mimetype='model/gltf-binary', conditional=True)


@app.route('/api/status')
def status():
    """Get current mouth status"""
    return jsonify({
        'viseme': current_viseme,
        'text': current_text,
        'intensity': current_intensity,
    })


@app.route('/api/devices')
def get_devices():
    """Get available audio devices"""
    import pyaudio
    
    devices = {
        'microphones': [],
        'speakers': []
    }
    
    # Get audio devices
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            device = {
                'index': i,
                'name': info['name'],
                'channels': info['maxInputChannels'] if info['maxInputChannels'] > 0 else info['maxOutputChannels']
            }
            
            if info['maxInputChannels'] > 0:
                devices['microphones'].append(device)
            if info['maxOutputChannels'] > 0:
                devices['speakers'].append(device)
        p.terminate()
    except Exception as e:
        print(f"Error enumerating audio devices: {e}")
    
    return jsonify(devices)


@app.route('/api/settings')
def get_settings():
    """Get current saved settings for the web UI"""
    if get_current_settings:
        settings = get_current_settings()

        # Mask the OpenAI API key: show last 4 chars if set, otherwise empty
        raw_key = settings.get('openai_api_key', '')
        if raw_key and len(raw_key) > 4:
            masked_key = '****' + raw_key[-4:]
        else:
            masked_key = ''

        return jsonify({
            'microphone_index': settings.get('microphone_index'),
            'speaker_index': settings.get('speaker_index'),
            'voice_id': settings.get('voice_id', 'matthew'),
            'face_tracking_enabled': settings.get('face_tracking_enabled', False),
            'face_recognition_enabled': settings.get('face_recognition_enabled', False),
            'voice_model': settings.get('voice_model', 'openai_realtime'),
            'openai_voice_id': settings.get('openai_voice_id', 'alloy'),
            'openai_model_id': settings.get('openai_model_id', 'gpt-realtime-2.1'),
            'openai_api_key': masked_key,
        })
    return jsonify({})


@app.route('/api/settings', methods=['POST'])
def post_settings():
    """Update settings from the web UI"""
    if not save_current_settings:
        return jsonify({'error': 'Settings save not available'}), 500

    data = request.get_json(silent=True) or {}

    # Fields that the POST endpoint accepts and persists
    allowed_fields = [
        'voice_model', 'openai_voice_id', 'openai_model_id', 'openai_api_key',
        'microphone_index', 'speaker_index', 'voice_id',
        'face_tracking_enabled', 'face_recognition_enabled',
    ]

    updates = {}
    for field in allowed_fields:
        if field in data:
            value = data[field]
            # Don't overwrite the real API key with the masked placeholder
            if field == 'openai_api_key' and isinstance(value, str) and value.startswith('****'):
                continue
            updates[field] = value

    if updates:
        save_current_settings(updates)

    return jsonify({'status': 'ok'})


@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('Client connected')
    emit('viseme_update', {
        'viseme': current_viseme,
        'text': current_text,
        'intensity': current_intensity,
    })
    
    # Send current face tracking state if callback is set
    if get_face_tracking_state:
        enabled = get_face_tracking_state()
        emit('face_tracking_status', {'enabled': enabled})
        print(f'Sent initial face tracking state: {enabled}')


@socketio.on('control')
def handle_control(data):
    """Handle control commands from web interface"""
    action = data.get('action')
    value = data.get('value')
    
    print(f'Control command: {action} = {value}')
    
    # Store control commands in queue for main app to process
    mouth_queue.put({'action': action, 'value': value})
    
    emit('control_ack', {'action': action, 'status': 'received'})


def update_mouth(viseme, text='', intensity=None, vrm=None, vrm_weight=None,
                 sync_delay_ms=None):
    """Update the mouth shape and optional continuous opening intensity.

    Args:
        viseme: Amplitude-based category (CLOSED/NARROW/.../WIDE) for the
            legacy CSS head and servo mapping.
        text: Current spoken text.
        intensity: Continuous mouth opening in [0, 1] (amplitude-derived).
        vrm: Optional phoneme-accurate VRM expression name
            (aa/ih/ou/ee/oh/neutral) from the viseme driver. When present, the
            VRM avatar uses this instead of mapping the amplitude category.
        vrm_weight: Weight in [0, 1] for the VRM expression.
    """
    global current_viseme, current_text, current_intensity

    fallback_intensities = {
        'CLOSED': 0.0,
        'NARROW': 0.15,
        'ROUNDED': 0.20,
        'MEDIUM': 0.30,
        'MEDIUM_OPEN': 0.45,
        'WIDE': 0.65,
    }
    if intensity is None:
        intensity = fallback_intensities.get(viseme, 1.0 if viseme.lower() in {
            'aa', 'ih', 'ou', 'ee', 'oh'
        } else 0.0)

    current_viseme = viseme
    current_text = text
    current_intensity = max(0.0, min(1.0, float(intensity)))

    payload = {
        'viseme': viseme,
        'text': text,
        'intensity': current_intensity,
    }
    if vrm is not None:
        payload['vrm'] = vrm
        payload['vrm_weight'] = max(0.0, min(1.0, float(vrm_weight if vrm_weight is not None else 0.0)))
    if sync_delay_ms is not None:
        payload['sync_delay_ms'] = int(sync_delay_ms)

    socketio.emit('viseme_update', payload)


def update_expression(emotion: str, weight: float = 1.0, hold_ms: int = 0):
    """Set the avatar's facial emotion expression in the web UI.

    Args:
        emotion: Expression name (happy/angry/sad/relaxed/neutral).
        weight: Intensity in [0, 1].
        hold_ms: If > 0, the client returns to neutral after this many
            milliseconds; 0 holds until changed.
    """
    socketio.emit('expression_update', {
        'emotion': str(emotion),
        'weight': max(0.0, min(1.0, float(weight))),
        'hold_ms': int(hold_ms),
    })


def update_eyes(eye_angles):
    """
    Update eye positions in web UI
    
    Args:
        eye_angles: Dict of servo_name -> angle
    """
    # Extract relevant eye angles
    data = {}
    
    # Left eye
    if 'left_eye_x' in eye_angles:
        data['left_x'] = eye_angles['left_eye_x']
    if 'left_eye_y' in eye_angles:
        data['left_y'] = eye_angles['left_eye_y']
    
    # Right eye
    if 'right_eye_x' in eye_angles:
        data['right_x'] = eye_angles['right_eye_x']
    if 'right_eye_y' in eye_angles:
        data['right_y'] = eye_angles['right_eye_y']
    
    # Shared axes (for simpler configs)
    if 'eye_x' in eye_angles:
        data['left_x'] = eye_angles['eye_x']
        data['right_x'] = eye_angles['eye_x']
    if 'eye_y' in eye_angles:
        data['left_y'] = eye_angles['eye_y']
        data['right_y'] = eye_angles['eye_y']
    
    # Explicit normalized gaze takes precedence for browser avatars.
    if 'gaze_x' in eye_angles:
        data['gaze_x'] = eye_angles['gaze_x']
    if 'gaze_y' in eye_angles:
        data['gaze_y'] = eye_angles['gaze_y']

    if data:
        socketio.emit('eye_position_update', data)


def update_face_tracking_status(enabled: bool):
    """
    Update face tracking status in web UI
    
    Args:
        enabled: Whether face tracking is enabled
    """
    socketio.emit('face_tracking_status', {'enabled': enabled})


def trigger_blink():
    """Trigger eye blink animation in web UI"""
    socketio.emit('blink_eyes', {})


def animate_text(text, phoneme_viseme_pairs, duration=None):
    """
    Animate mouth for text with phonemes
    
    Args:
        text: The text being spoken
        phoneme_viseme_pairs: List of (phoneme, viseme) tuples
        duration: Total duration in seconds (auto-calculated if None)
    """
    if not phoneme_viseme_pairs:
        return
    
    # Calculate timing
    if duration is None:
        # Rough estimate: 12 phonemes per second (faster)
        duration = len(phoneme_viseme_pairs) / 12.0
    
    time_per_phoneme = duration / len(phoneme_viseme_pairs)
    
    # Animate through phonemes
    for phoneme, viseme in phoneme_viseme_pairs:
        update_mouth(viseme, f"{text} [{phoneme}]")
        time.sleep(time_per_phoneme)
    
    # Return to rest immediately
    update_mouth('CLOSED', '')
    time.sleep(0.1)  # Brief pause before next animation


def run_flask():
    """Run Flask server in background"""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    socketio.run(app, host='127.0.0.1', port=8080, debug=False, use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)


# Global server thread
server_thread = None


def get_control_command(timeout=0.1):
    """
    Get control command from web interface
    
    Args:
        timeout: How long to wait for command
        
    Returns:
        Dict with action and value, or None if no command
    """
    try:
        return mouth_queue.get(timeout=timeout)
    except:
        return None


def start_server():
    """Start the Flask server"""
    global server_thread
    if server_thread is None or not server_thread.is_alive():
        server_thread = threading.Thread(target=run_flask, daemon=True)
        server_thread.start()
        time.sleep(2)  # Give server time to start
        print("\n🌐 Mouth visualizer starting at: http://127.0.0.1:8080")
        print("   Open this URL in your browser to see the animated mouth!")
        print("   (Wait a few seconds for the server to fully start)\n")


if __name__ == '__main__':
    start_server()
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
