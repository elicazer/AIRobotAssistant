# AI Robot Assistant

An AI-powered voice assistant and face tracking system for animatronic robot heads with real-time lip sync, eye tracking, and natural blinking behavior.

Built with **Amazon Nova Sonic**, **Amazon Bedrock**, and **Kiro AI**.

## 🎬 Demo

[![AI Robot Assistant Demo](https://img.youtube.com/vi/v6-OhUJxlb0/0.jpg)](https://www.youtube.com/shorts/v6-OhUJxlb0)

*Click the image above to watch the AI Robot Assistant in action!*

## 🎯 Features

- **Natural Voice Conversations**: Amazon Nova Sonic AI for intelligent, context-aware responses
- **Real-time Lip Sync**: Audio-driven jaw animation synchronized with speech (like VR avatars)
- **Virtual Robot Avatar**: Works without physical hardware - test and demo using web visualization
- **Face Tracking**: OpenCV-based face detection with automatic eye following
- **Natural Blinking**: Automatic blinking behavior with realistic timing
- **Web Control Interface**: Browser-based control panel with live visualization
- **Multi-Servo Support**: Configurable servo layouts (InMoov, Original, Simple)
- **Servo Calibration**: Test and configure all servos through web interface

## 🏗️ Project Structure

```
AIRobotAssistant/
├── run.py                        # Main entry point
├── start.sh                      # Quick start script
├── requirements.txt              # Python dependencies
├── config/                       # Configuration
│   └── voice_assistant_settings.json
├── src/                          # Source code
│   ├── voice_assistant_server.py # Main server
│   ├── nova_sonic_client.py      # Voice AI client
│   ├── audio_mouth_controller.py # Audio-driven lip sync
│   ├── face_tracker.py           # Face detection
│   ├── eye_controller.py         # Eye servo control
│   ├── mouth_visualizer.py       # Web UI backend
│   ├── servo_config.py           # Servo configurations
│   └── platform_config.py        # Platform detection
├── scripts/                      # Utilities
│   └── activate_venv.sh          # Virtual environment helper
└── templates/                    # Web UI
    └── mouth.html                # Web interface
```

## 🔧 Hardware Requirements

### Software Only (Virtual Robot)
- **Computer**: Mac/PC/Raspberry Pi
- **Microphone**: For voice input
- **Speaker**: For voice output
- **Webcam** (optional): For face tracking

The system works perfectly without physical hardware using the web-based virtual robot avatar!

### Physical Robot (Optional)
- **FT232H**: USB-to-I2C adapter
- **PCA9685**: 16-channel servo driver board
- **Servos**:
  - 1x MG996R standard servo (jaw - channel 8)
  - 8x Standard 180° servos (eyes - channels 0-7)
- **Power Supply**: 5-6V with sufficient amperage for servos

### Servo Channel Layout

| Channel | Servo | Function |
|---------|-------|----------|
| 0 | Left Eye X | Horizontal movement |
| 1 | Left Eye Y | Vertical movement |
| 2 | Left Upper Eyelid | Blinking |
| 3 | Left Lower Eyelid | Blinking |
| 4 | Right Eye X | Horizontal movement |
| 5 | Right Eye Y | Vertical movement |
| 6 | Right Upper Eyelid | Blinking |
| 7 | Right Lower Eyelid | Blinking |
| 8 | Jaw | Mouth open/close |

## 📦 Installation

### Dependencies
```bash
# Clone repository
git clone <repository-url>
cd AIRobotAssistant

# Create virtual environment (Python 3.12+ required)
python3.12 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start
```bash
# Option 1: Use the start script
./start.sh

# Option 2: Manual start
python run.py
```

Open browser to: `http://localhost:8080`

### Web Interface

1. **Start/Stop** - Control voice assistant
2. **Settings Tab**: 
   - Configure jaw servo angles
   - Enable/disable face tracking
   - Select servo configuration
   - Test individual servos

### Voice Conversation

Just speak naturally! The robot will:
- Respond with synthesized speech
- Animate jaw synchronized with speech (visible in web interface)
- Track your face with its eyes (if face tracking enabled)
- Blink naturally

**Note**: The system works fully without physical hardware - all movements are visualized in the web interface!

## ⚙️ Configuration

Edit `config/voice_assistant_settings.json`:

```json
{
  "jaw_open_angle": 100,
  "jaw_close_angle": 0,
  "jaw_servo_min_change": 2,
  "face_tracking_enabled": true,
  "servo_config": "inmoov",
  "camera_index": 0
}
```

### Servo Configurations

- **InMoov** (8 servos): 4 per eye (X, Y, Upper Lid, Lower Lid)
- **Original** (6 servos): Shared X/Y, separate eyelids
- **Simple** (2 servos): Just X and Y movement

### Jaw Servo Settings

- **jaw_open_angle**: Angle for fully open jaw (0-180°)
- **jaw_close_angle**: Angle for closed jaw (0-180°)
- **jaw_servo_min_change**: Minimum angle change to trigger movement (reduces jitter)

## 🎮 Controls

- **Start Listening**: Begin voice conversation
- **Stop**: End conversation
- **Settings**: Configure servos, camera, and face tracking
- **Servo Test Sliders**: Test individual servo movements

## 🔬 Technical Details

### Audio-Driven Lip Sync

The system analyzes incoming audio in real-time:
```
Audio Stream → RMS Amplitude → Smoothing → Mouth Opening %
```

- **Sample Rate**: 24kHz
- **Smoothing Window**: 3 samples
- **Threshold Range**: 0.015 - 0.25 amplitude
- **Asymmetric Speed**: Closes 70% faster than it opens

### Face Tracking

Face coordinates are mapped to servo angles:
```python
servo_angle = map_value(face_position, 0, video_dimension, servo_max, servo_min)
```

- **Detection**: OpenCV Haar Cascade
- **Update Rate**: Real-time
- **Auto-start**: Configurable on launch

## 🐛 Troubleshooting

### Servo Issues
- Check 5-6V power supply (sufficient amperage)
- Verify I2C connections to FT232H
- Use Settings tab to test individual servos
- Adjust `jaw_servo_min_change` if servo is jittery

### Camera Issues
- Check camera permissions
- Try different `camera_index` values (0, 1, 2)
- Verify camera works with other applications

### Audio Issues
- Check microphone/speaker in system settings
- Verify Amazon Nova Sonic credentials
- Ensure Python 3.12+ is installed

### Mouth Doesn't Close
- Check console for "🔒 Speech ended" message
- Verify servo power supply
- Increase `jaw_open_angle` if range is too small

## 📁 Module Descriptions

- **voice_assistant_server.py**: Main server coordinating all components
- **nova_sonic_client.py**: Amazon Nova Sonic voice AI integration
- **audio_mouth_controller.py**: Real-time audio analysis for lip sync
- **face_tracker.py**: OpenCV face detection and tracking
- **eye_controller.py**: Servo control for eye movements and blinking
- **mouth_visualizer.py**: Flask/SocketIO web interface backend
- **servo_config.py**: Servo configuration presets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon Web Services** - Nova Sonic voice AI
- **Kiro AI** - AI development assistant (https://kiro.dev/)
- **InMoov Project** - Robot design inspiration (https://inmoov.fr/)
- **Adafruit** - Servo libraries and hardware
- **OpenCV** - Computer vision tools

## 🔗 Links

- Amazon Nova Sonic: https://aws.amazon.com/nova/speech/
- Kiro AI: https://kiro.dev/
- InMoov Robot: https://inmoov.fr/

## 💡 Technology Stack

- **Python 3.12+** - Core language
- **Amazon Nova Sonic** - AI voice assistant
- **Amazon Bedrock** - AI intelligence
- **OpenCV** - Face detection and tracking
- **Adafruit ServoKit** - Servo control
- **Flask & SocketIO** - Web interface
- **NumPy** - Audio processing
