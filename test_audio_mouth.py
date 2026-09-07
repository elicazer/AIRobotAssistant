#!/usr/bin/env python3
"""
Test Audio-Driven Mouth Animation
Run this to test the audio mouth controller with your microphone
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_mouth_controller import AudioMouthController
import pyaudio
import time

def main():
    print("=" * 60)
    print("AUDIO-DRIVEN MOUTH ANIMATION TEST")
    print("=" * 60)
    print()
    print("This will analyze audio from your microphone in real-time")
    print("and show how the mouth opening would be controlled.")
    print()
    print("Speak into your microphone to see the mouth movement!")
    print("Press Ctrl+C to stop")
    print()
    
    # Create controller
    controller = AudioMouthController(
        sample_rate=24000,
        smoothing_window=5,
        min_threshold=0.01,
        max_threshold=0.3
    )
    
    # Set up audio stream
    p = pyaudio.PyAudio()
    
    # List available devices
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    print()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Listening... Speak now!")
        print()
        
        while True:
            # Read audio chunk
            audio_data = stream.read(1024, exception_on_overflow=False)
            
            # Process and get mouth opening
            opening = controller.process_audio_chunk(audio_data)
            viseme = controller.get_viseme_from_opening(opening)
            
            # Display with visual bar
            if opening > 5:
                bar = '█' * int(opening / 5)
                status = "🗣️ SPEAKING"
            else:
                bar = ''
                status = "🤐 SILENT "
            
            print(f"\r{status} | Opening: {opening:5.1f}% [{bar:<20}] {viseme:12}    ", end='', flush=True)
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n✅ Test stopped")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
