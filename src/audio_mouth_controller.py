"""
Audio-Driven Mouth Animation
Real-time mouth movement based on audio amplitude analysis
Similar to VR/game character lip sync
"""

import numpy as np
import time
from collections import deque


class AudioMouthController:
    """
    Controls mouth movement based on real-time audio amplitude
    Uses sliding window for smooth, natural movement
    """
    
    def __init__(self, 
                 sample_rate=24000,
                 smoothing_window=3,  # Reduced for faster response
                 min_threshold=0.015,  # Slightly higher to close more
                 max_threshold=0.25,   # Lower for more dynamic range
                 close_speed=0.7):     # How fast mouth closes (0-1, higher = faster)
        """
        Initialize audio mouth controller
        
        Args:
            sample_rate: Audio sample rate (Hz)
            smoothing_window: Number of samples to average for smoothing
            min_threshold: Minimum amplitude to trigger mouth movement
            max_threshold: Amplitude for maximum mouth opening
            close_speed: Speed multiplier for closing (0-1, higher = faster close)
        """
        self.sample_rate = sample_rate
        self.smoothing_window = smoothing_window
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.close_speed = close_speed
        
        # Sliding window for smoothing
        self.amplitude_history = deque(maxlen=smoothing_window)
        
        # Current state
        self.current_opening = 0
        self.target_opening = 0
        self.is_speaking = False
        self.silence_counter = 0
        
    def process_audio_chunk(self, audio_bytes):
        """
        Process audio chunk and return mouth opening percentage
        
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
            
        Returns:
            float: Mouth opening percentage (0-100)
        """
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Calculate RMS (Root Mean Square) amplitude
        rms = np.sqrt(np.mean(audio_data.astype(float)**2))
        
        # Normalize to 0-1 range
        normalized_amplitude = rms / 32768.0  # 16-bit audio max value
        
        # Add to history for smoothing
        self.amplitude_history.append(normalized_amplitude)
        
        # Calculate smoothed amplitude
        smoothed_amplitude = np.mean(self.amplitude_history)
        
        # Map amplitude to mouth opening (0-100%)
        if smoothed_amplitude < self.min_threshold:
            self.target_opening = 0
            self.silence_counter += 1
            self.is_speaking = False
        else:
            # Non-linear mapping for more dynamic movement
            # Use power curve to emphasize differences
            normalized = (smoothed_amplitude - self.min_threshold) / (self.max_threshold - self.min_threshold)
            normalized = max(0, min(1, normalized))
            
            # Apply power curve (exponent < 1 = more sensitive to quiet sounds)
            # This makes the mouth more responsive to variations
            self.target_opening = (normalized ** 0.8) * 100
            
            self.silence_counter = 0
            self.is_speaking = self.target_opening > 3
        
        # Smooth transition with asymmetric speed (close faster than open)
        if self.target_opening < self.current_opening:
            # Closing - faster response
            step = (self.current_opening - self.target_opening) * self.close_speed
            self.current_opening = max(self.target_opening, self.current_opening - step)
        else:
            # Opening - normal response
            step = (self.target_opening - self.current_opening) * 0.4
            self.current_opening = min(self.target_opening, self.current_opening + step)
        
        # Force close if silent for multiple frames
        if self.silence_counter > 2:
            self.current_opening = 0
        
        return self.current_opening
    
    def get_viseme_from_opening(self, opening):
        """
        Convert opening percentage to viseme category
        
        Args:
            opening: Mouth opening percentage (0-100)
            
        Returns:
            str: Viseme name
        """
        if opening < 5:
            return 'CLOSED'
        elif opening < 20:
            return 'NARROW'
        elif opening < 35:
            return 'ROUNDED'
        elif opening < 50:
            return 'MEDIUM'
        elif opening < 70:
            return 'MEDIUM_OPEN'
        else:
            return 'WIDE'
    
    def reset(self):
        """Reset controller state"""
        self.amplitude_history.clear()
        self.current_opening = 0
        self.target_opening = 0
        self.is_speaking = False
        self.silence_counter = 0


class EnhancedAudioMouthController(AudioMouthController):
    """
    Enhanced version with frequency analysis for more natural movement
    Analyzes both volume and frequency content
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frequency_bands = {
            'low': (80, 250),      # Bass/rumble
            'mid': (250, 2000),    # Vowels
            'high': (2000, 8000)   # Consonants
        }
    
    def process_audio_chunk_enhanced(self, audio_bytes):
        """
        Enhanced processing with frequency analysis
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            tuple: (opening_percentage, viseme_hint)
        """
        # Basic amplitude processing
        opening = self.process_audio_chunk(audio_bytes)
        
        # Convert to numpy for FFT
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(float)
        
        # Perform FFT for frequency analysis
        if len(audio_data) > 0:
            fft = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            magnitudes = np.abs(fft)
            
            # Analyze frequency bands
            mid_energy = self._get_band_energy(freqs, magnitudes, 
                                               self.frequency_bands['mid'])
            high_energy = self._get_band_energy(freqs, magnitudes, 
                                                self.frequency_bands['high'])
            
            # Adjust opening based on frequency content
            # High frequencies (consonants) = less opening
            # Mid frequencies (vowels) = more opening
            if high_energy > mid_energy * 1.5:
                opening *= 0.7  # Reduce for consonants
            elif mid_energy > high_energy * 1.5:
                opening *= 1.2  # Increase for vowels
                opening = min(100, opening)
        
        viseme = self.get_viseme_from_opening(opening)
        return opening, viseme
    
    def _get_band_energy(self, freqs, magnitudes, band):
        """Calculate energy in frequency band"""
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(mask):
            return np.mean(magnitudes[mask])
        return 0


# Example usage
if __name__ == "__main__":
    import pyaudio
    
    # Create controller
    controller = AudioMouthController()
    
    # Set up audio stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        input=True,
        frames_per_buffer=1024
    )
    
    print("Listening to audio... Speak to see mouth opening values")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Read audio chunk
            audio_data = stream.read(1024, exception_on_overflow=False)
            
            # Process and get mouth opening
            opening = controller.process_audio_chunk(audio_data)
            viseme = controller.get_viseme_from_opening(opening)
            
            # Display
            if opening > 5:
                bar = '█' * int(opening / 5)
                print(f"\rOpening: {opening:5.1f}% [{bar:<20}] {viseme}    ", end='')
            else:
                print(f"\rOpening: {opening:5.1f}% [{'':20}] {viseme}    ", end='')
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopped")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
