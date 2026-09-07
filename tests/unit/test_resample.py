"""Unit tests for resample_16k_to_24k function."""

import array
import struct

import pytest

from src.openai_realtime_client import resample_16k_to_24k


class TestResample16kTo24k:
    """Tests for the resample_16k_to_24k function."""

    def test_empty_input_returns_empty_bytes(self):
        """Empty input should return empty bytes."""
        result = resample_16k_to_24k(b"")
        assert result == b""

    def test_single_sample_returns_same(self):
        """A single sample (2 bytes) should be returned as-is."""
        sample = struct.pack("<h", 1000)
        result = resample_16k_to_24k(sample)
        assert result == sample

    def test_two_samples_produce_four_output_samples(self):
        """Two input samples should produce 4 output samples (3 interpolated + last)."""
        # Two samples: 0 and 300
        samples = array.array("h", [0, 300])
        result = resample_16k_to_24k(samples.tobytes())

        output = array.array("h")
        output.frombytes(result)

        # Expected: s0=0, interp1=100, interp2=200, last=300
        assert len(output) == 4
        assert output[0] == 0
        assert output[1] == 100  # 0 + (300-0)/3
        assert output[2] == 200  # 0 + 2*(300-0)/3
        assert output[3] == 300  # last sample

    def test_three_samples_produce_seven_output_samples(self):
        """Three input samples should produce 7 output samples (3+3+1)."""
        samples = array.array("h", [0, 300, 600])
        result = resample_16k_to_24k(samples.tobytes())

        output = array.array("h")
        output.frombytes(result)

        # For pair (0, 300): 0, 100, 200
        # For pair (300, 600): 300, 400, 500
        # Last: 600
        assert len(output) == 7
        assert output[0] == 0
        assert output[1] == 100
        assert output[2] == 200
        assert output[3] == 300
        assert output[4] == 400
        assert output[5] == 500
        assert output[6] == 600

    def test_output_sample_count_ratio(self):
        """Output sample count should be approximately 1.5x input count."""
        # 100 input samples
        samples = array.array("h", range(100))
        result = resample_16k_to_24k(samples.tobytes())

        output = array.array("h")
        output.frombytes(result)

        # Formula: 3 * (n-1) + 1 = 3n - 2 for n input samples
        # For 100 input: 298 output samples
        expected = 3 * (100 - 1) + 1  # 298
        assert len(output) == expected

    def test_negative_samples(self):
        """Should handle negative sample values correctly."""
        samples = array.array("h", [-300, 300])
        result = resample_16k_to_24k(samples.tobytes())

        output = array.array("h")
        output.frombytes(result)

        assert output[0] == -300
        assert output[1] == -100  # -300 + (600)/3 = -300 + 200 = -100
        assert output[2] == 100   # -300 + 2*(600)/3 = -300 + 400 = 100
        assert output[3] == 300

    def test_constant_signal(self):
        """A constant signal should remain constant after resampling."""
        samples = array.array("h", [500, 500, 500, 500])
        result = resample_16k_to_24k(samples.tobytes())

        output = array.array("h")
        output.frombytes(result)

        # All output samples should be 500
        assert all(s == 500 for s in output)

    def test_returns_bytes_type(self):
        """Result should always be bytes."""
        samples = array.array("h", [100, 200])
        result = resample_16k_to_24k(samples.tobytes())
        assert isinstance(result, bytes)
