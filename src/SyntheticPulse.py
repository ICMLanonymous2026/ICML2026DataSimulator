import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from .generate_ofdm import generate_ofdm

SAMPLING_FREQUENCY = 4e9

class SyntheticPulse:
    def __init__(
        self,
        carrier_frequency,
        pw,
        modulation_type=None,
        modulation_value=None,
        amplitude_mask=None,
        power=1,
        fe=None,
        iq=False
    ):
        """
        Public-safe synthetic radar pulse generator.
        NLFM modulation is intentionally NOT supported.
        """
        self.fe = SAMPLING_FREQUENCY if fe is None else fe
        self.carrier_frequency = carrier_frequency
        self.pw = pw
        self.modulation_type = modulation_type
        self.modulation_value = modulation_value
        self.amplitude_mask = amplitude_mask
        self.power = power
        self.iq = iq

        # Check that NLFM is not used
        if self.modulation_type == "NLFM":
            raise NotImplementedError(
                "NLFM modulation is not available in the public release."
            )

        # Pre-compute and store bandwidth
        self.bandwidth = self.calculate_bandwidth()

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------
    def calculate_bandwidth(self):
        """
        Compute the signal bandwidth given the modulation type.
        NLFM is not supported in public version.
        """

        if self.modulation_type == "LFM":
            return float(self.modulation_value)

        if self.modulation_type == "FSK":
            mv = np.asarray(self.modulation_value)
            if mv.size == 0:
                return 0.0
            return float(mv.max() - mv.min())

        if self.modulation_type in ["PSK", "QAM"]:
            if self.modulation_value is None or self.pw == 0:
                return 0.0
            return len(self.modulation_value) / self.pw

        return 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def describe(self):
        """
        Prints a description of the pulse, including bandwidth.
        """
        print("----- Pulse Description -----")
        print("carrier frequency =", self.carrier_frequency)
        print("pulse width       =", self.pw)
        print("modulation type   =", self.modulation_type)
        print("modulation value  =", self.modulation_value)
        print("sampling frequency =", self.fe)
        print("power             =", self.power)
        print("bandwidth         =", self.bandwidth, "Hz")

    # ------------------------------------------------------------------
    # Phase for PSK only
    # ------------------------------------------------------------------
    def generate_phase(self, time):
        """
        Generates phase modulation pattern for PSK only.
        NLFM has been removed in the public version.
        """
        number_of_points = len(time)
        phase = np.zeros(number_of_points)

        if self.modulation_type == "PSK":
            if self.modulation_value is None or len(self.modulation_value) == 0:
                return phase

            segment_length = number_of_points // len(self.modulation_value) + 1

            for i, phase_value in enumerate(self.modulation_value):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length
                phase[start_idx:end_idx] = phase_value

        return phase

    # ------------------------------------------------------------------
    # FSK / LFM frequency generation
    # ------------------------------------------------------------------
    def generate_frequency(self, time):
        """
        Generates frequency modulation pattern.
        NLFM is not supported in public version.
        """
        if self.modulation_type == "FSK":
            costas_code = np.asarray(self.modulation_value)
            segment_duration = self.pw / len(costas_code)
            frequency = np.zeros_like(time)

            for i, code_value in enumerate(costas_code):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                segment_indices = np.logical_and(
                    time >= start_time, time < end_time
                )
                frequency[segment_indices] = self.carrier_frequency + code_value

            return frequency

        elif self.modulation_type == "LFM":
            return (
                self.carrier_frequency - self.modulation_value / 2
                + time * self.modulation_value / (2 * self.pw)
            )

        else:
            # Constant carrier
            return np.ones_like(time) * self.carrier_frequency

    # ------------------------------------------------------------------
    # QAM-specific helper
    # ------------------------------------------------------------------
    def _qam_symbols_to_time(self, num_points):
        """
        Expand QAM symbols into a sample-level baseband sequence.
        """
        if self.modulation_value is None:
            raise ValueError(
                "For QAM, 'modulation_value' must be a sequence of complex symbols."
            )

        symbols = np.asarray(self.modulation_value, dtype=np.complex128)
        if symbols.size == 0:
            return np.zeros(num_points, dtype=np.complex128)

        K = symbols.size
        samples_per_symbol = max(1, num_points // K)

        seq = np.repeat(symbols, samples_per_symbol)
        if seq.size < num_points:
            pad = np.full(num_points - seq.size, symbols[-1], dtype=np.complex128)
            seq = np.concatenate([seq, pad])

        return seq[:num_points]

    # ------------------------------------------------------------------
    # Pulse generation
    # ------------------------------------------------------------------
    def generate_pulse(self):
        """
        Generate the time-domain radar pulse.
        NLFM is not supported.
        """

        # --------------------------------------------------------
        # 0. OFDM special case
        # --------------------------------------------------------
        if self.modulation_type == "OFDM":
            T = self.pw
            B = float(self.modulation_value) if self.modulation_value is not None else self.bandwidth
            Fc = self.carrier_frequency
            Fs = self.fe

            t, x_passband, x_baseband = generate_ofdm(
                fc=Fc,
                B=B,
                T=T,
                Fs=Fs
            )

            return x_baseband if self.iq else x_passband

        # --------------------------------------------------------
        # 1. Determine the exact number of samples
        # --------------------------------------------------------
        if self.amplitude_mask is not None:
            N = len(self.amplitude_mask)
        else:
            N = int(round(self.pw * self.fe))

        # Exact and consistent time steps
        time_steps = np.arange(N) / self.fe

        # --------------------------------------------------------
        # 2. Amplitude envelope
        # --------------------------------------------------------
        amplitude = np.sqrt(2 * self.power / self.fe)

        if self.amplitude_mask is None:
            tukey_win = sig.windows.tukey(N, 0.02)
            env = amplitude * tukey_win
        else:
            # self.amplitude_mask is already length N
            env = amplitude * self.amplitude_mask

        # --------------------------------------------------------
        # 3. QAM special case
        # --------------------------------------------------------
        if self.modulation_type == "QAM":
            baseband_symbols = self._qam_symbols_to_time(N)
            baseband = env * baseband_symbols

            carrier_phase = 2 * np.pi * self.carrier_frequency * time_steps
            analytic = baseband * np.exp(1j * carrier_phase)

            return analytic if self.iq else np.real(analytic)

        # --------------------------------------------------------
        # 4. All other modulations (FSK, PSK, etc.)
        # --------------------------------------------------------
        # Make sure phase and frequency are also length N
        phase = self.generate_phase(time_steps)
        frequency = self.generate_frequency(time_steps)
        carrier = 2 * np.pi * frequency * time_steps + phase

        return (
            env * np.exp(1j * carrier) if self.iq
            else env * np.sin(carrier)
        )
