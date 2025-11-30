"""
Visual Mouth Animation for Lip Sync Testing
Flask web app that shows animated mouth movements
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import queue
import time
import os

# Get the project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = 'robot-mouth-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Queue for mouth updates
mouth_queue = queue.Queue()
current_viseme = 'CLOSED'
current_text = ''

# Callback to get current face tracking state
get_face_tracking_state = None


@app.route('/')
def index():
    """Main page with mouth animation"""
    return render_template('mouth.html')


@app.route('/api/status')
def status():
    """Get current mouth status"""
    return jsonify({
        'viseme': current_viseme,
        'text': current_text
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


@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('Client connected')
    emit('viseme_update', {'viseme': current_viseme, 'text': current_text})
    
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


def update_mouth(viseme, text=''):
    """Update the mouth shape"""
    global current_viseme, current_text
    current_viseme = viseme
    current_text = text
    socketio.emit('viseme_update', {'viseme': viseme, 'text': text})


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
    socketio.run(app, host='127.0.0.1', port=8080, debug=False, use_reloader=False, log_output=False)


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
