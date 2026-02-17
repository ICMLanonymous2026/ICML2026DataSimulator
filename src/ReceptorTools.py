import math
import random
import itertools
import threading
import time

import scipy.signal

import numpy as np
from .config_settings import get_settings
from .SignalAnalysis import SignalAnalysis

global_params = get_settings("GLOBAL")
SAMPLING_FREQUENCY = float(global_params['SAMPLING_FREQUENCY'])

MIN_FREQUENCY_BANDS = float(global_params['MIN_FREQUENCY_BANDS'])
MAX_FREQUENCY_BANDS = float(global_params['MAX_FREQUENCY_BANDS'])
FREQUENCY_BANDS_WIDTH = float(global_params['FREQUENCY_BANDS_WIDTH'])
OVERLAP = float(global_params['OVERLAP'])

MIN_FREQUENCY_ACQUISITION = float(global_params['MIN_FREQUENCY_ACQUISITION'])
MAX_FREQUENCY_ACQUISITION = float(global_params['MAX_FREQUENCY_ACQUISITION'])

# Duration of ou reception, in seconds.
LISTENING_TIME = float(global_params['LISTENING_TIME'])

receptor_params = get_settings("RECEPTOR")

# Impedance of the ADC
IMPEDANCE = float(receptor_params['IMPEDANCE'])

# Reception channel noise density in dBm/kHz
RECEPTION_CHANNEL_NOISE_DENSITY = float(receptor_params['RECEPTION_CHANNEL_NOISE_DENSITY'])

# Quantification noise power in dBfs
QUANTIFICATION_NOISE_POWER = float(receptor_params['QUANTIFICATION_NOISE_POWER'])

# Frequency modulation settings data
MIN_FREQUENCY_FMOP = float(receptor_params['MIN_FREQUENCY_FMOP'])
MAX_FREQUENCY_FMOP = float(receptor_params['MAX_FREQUENCY_FMOP'])

# Minimal and maximal distance between our receptor and the transmitters within our environment, in meters.
MINIMAL_DISTANCE = float(global_params['MINIMAL_DISTANCE'])
MAXIMAL_DISTANCE = float(global_params['MAXIMAL_DISTANCE'])

MAXIMAL_ERP = float(global_params['MAXIMAL_ERP'])


class ReceptorTools(SignalAnalysis):
    def __init__(self, fe=SAMPLING_FREQUENCY, min_frequency_bands=MIN_FREQUENCY_BANDS,
                 max_frequency_bands=MAX_FREQUENCY_BANDS, band_width=FREQUENCY_BANDS_WIDTH, overlap=OVERLAP,
                 min_freq_acq=MIN_FREQUENCY_ACQUISITION, max_freq_acq=MAX_FREQUENCY_ACQUISITION,
                 listening_time=LISTENING_TIME, impedance=IMPEDANCE,
                 reception_channel_noise_density=RECEPTION_CHANNEL_NOISE_DENSITY,
                 quantification_noise_power=QUANTIFICATION_NOISE_POWER, min_distance=MINIMAL_DISTANCE,
                 max_distance=MAXIMAL_DISTANCE, max_erp=MAXIMAL_ERP):

        self.acquisition_start_time = 0
        self.is_running = False

        if fe is None:
            self.fe = SAMPLING_FREQUENCY
        else:
            self.fe = fe

        self.min_frequency_bands = min_frequency_bands
        self.max_frequency_bands = max_frequency_bands
        self.band_width = band_width
        self.overlap = overlap

        self.min_freq_acq = min_freq_acq
        self.max_freq_acq = max_freq_acq

        self.bands = self.get_frequency_bands()

        self.listening_time = listening_time
        self.impedance = impedance
        self.reception_channel_noise_density = reception_channel_noise_density
        self.quantification_noise_power = quantification_noise_power

        self.minimal_distance = min_distance
        self.maximal_distance = max_distance

        self.max_erp = max_erp

    def describe_acquisition_params(self, print=True):
        json_to_return = {}
        json_to_return['fe'] = self.fe
        json_to_return['min_frequency_bands'] = self.min_frequency_bands
        json_to_return['max_frequency_bands'] = self.max_frequency_bands
        json_to_return['band_width'] = self.band_width
        json_to_return['overlap'] = self.overlap
        json_to_return['min_freq_acq'] = self.min_freq_acq
        json_to_return['max_freq_acq'] = self.max_freq_acq
        json_to_return['listening_time'] = self.listening_time
        json_to_return['impedance'] = self.impedance
        json_to_return['reception_channel_noise_density'] = self.reception_channel_noise_density
        json_to_return['quantification_noise_power'] = self.quantification_noise_power
        json_to_return['minimal_distance'] = self.minimal_distance
        json_to_return['maximal_distance'] = self.maximal_distance
        if print:
            for key in json_to_return.keys():
                print(key, ' : ', json_to_return[key])
        return json_to_return

    def get_frequency_bands(self):
        # Generate frequency bands with specified width and overlap
        bands = []
        current_freq = self.min_frequency_bands
        current_index = 0

        while current_freq <= self.max_frequency_bands:
            band = {
                "start_frequency": current_freq,
                "end_frequency": min(current_freq + self.band_width, self.max_frequency_bands),
                "band_id": current_index,
                "pulses": []
            }
            bands.append(band)
            current_index += 1
            current_freq += self.band_width - self.overlap

        return bands

    def get_band_for_freq(self, f):
        into = []
        for band in self.bands:
            if band['start_frequency'] < f < band['end_frequency']:
                into.append(band['band_id'])
        return into

    def get_active_bands(self):
        active_bands = []
        for band in self.bands:
            if len(band['pulses']):
                active_bands.append(band['band_id'])
        return active_bands

    def shift_frequency_for_band(self, f, band):
        dif_frequency = f - band['start_frequency']
        return dif_frequency + self.min_freq_acq

    def apply_bandpass_filter(self, signal_data):
        # Fréquence de Nyquist
        nyquist = 0.5 * self.fe

        # Calcul des fréquences normalisées des coupures
        high = (self.fe - self.min_freq_acq) / nyquist
        low = (self.fe - self.max_freq_acq) / nyquist

        # Ordonnée et absisse du filtre
        b, a = scipy.signal.butter(4, [low, high], btype='band')

        # Application du filtre passe-bande au signal
        filtered_signal = scipy.signal.lfilter(b, a, signal_data)

        return filtered_signal

    def compute_signal_from_pulse(self, band_id, consider_listening_time=False, listening_time=None):
        """
        Computes the signal from pulses in the specified band.

        Args:
            band_id (int): Band identifier.
            consider_listening_time (bool): Indicates whether real listening time of the receptor should be considered.
            listening_time (list or float or None): Specified listening period, [time1, time2] or time.

        Returns:
            numpy.ndarray: Computed signal from pulses.

        """
        # distinguish the case you considere a listening time or not 
        if listening_time:
            # distinguish the case of a listening time as a list or juste a time 
            try:
                time1=listening_time[0]
                time2=listening_time[1]
            except:
                time1=0
                time2=listening_time

            # Filtering pulses based on the specified listening period
            pulses_list = [pulse for pulse in self.bands[band_id]['pulses'] if time1 < pulse['emission_time'] < time2]

            # Filtering and retrieving emission times
            pulses = [self.apply_bandpass_filter(pulse_json['pulse'].generate_pulse()) for pulse_json in pulses_list]
            emission_times = [pulse_json["emission_time"] for pulse_json in pulses_list]

            # Calculating the total length of the signal
            max_emission_time = time2-time1
            total_length = int(np.ceil(max_emission_time* self.fe))
        else:
            # Using all pulses in the band if no listening period is specified
            pulses_list = self.bands[band_id]['pulses']

            # Filtering and retrieving emission times
            pulses = [self.apply_bandpass_filter(pulse_json['pulse'].generate_pulse()) for pulse_json in pulses_list]
            emission_times = [pulse_json["emission_time"] for pulse_json in pulses_list]

            time1=0

            max_time_index = np.argmax(emission_times) 
            max_emission_time = emission_times[max_time_index]
            total_length = int(np.ceil(max_emission_time* self.fe)) + len(pulses[max_time_index])
            
            # If no pulses return empty list
            if len(pulses_list) == 0:
                        return []

        # Initialize the signal as a zero array of length equals to total_length
        final_signal = np.zeros(total_length)

        # Full the array where signal is present
        for i, signal in enumerate(pulses):
            start_idx = int(np.ceil(emission_times[i] * self.fe)) - int(np.ceil(time1*self.fe))
            if start_idx + len(signal)>len(final_signal):
                len_signal = len(final_signal[start_idx:])
                final_signal[start_idx:] += signal[:len_signal]
            else:
                final_signal[start_idx:start_idx + len(signal)] += signal

        # Make empty all periods when the receptor antenna is not listening
        if consider_listening_time:
            for i in range(len(final_signal)):
                t = i / self.fe
                if t < (band_id * self.listening_time) % len(self.bands) or t > (
                        (band_id + 1) * self.listening_time) % len(
                    self.bands):
                    final_signal[i] = 0

        return final_signal, pulses_list

    def compute_signal_with_noise(self, band_id, emitter_mode=False, listening_time=None):
        """
        This function calculates a signal with added noise for a given frequency band.

        Args:
            band_id (int): The ID of the frequency band.

        Returns:
            tuple: A tuple containing the noisy signal and the maximum signal-to-noise ratio (SNR).
        """

        # Compute the received signal based on the specified frequency band.
        received_signal, pulses = self.compute_signal_from_pulse(band_id, listening_time = listening_time)

        max_emitted_sig_power = np.max([pulse['pulse'].power for pulse in self.bands[band_id]['pulses']])
        if emitter_mode:
            max_emitted_sig_power = max_emitted_sig_power / (2 * np.pi * self.distance) ** 2
      

        # Calculate the maximum signal power based on parameters.
        max_sig_power = 1.64 * self.max_erp / 2 * (
                2.99e8 / (self.bands[band_id]['start_frequency'] * 4 * np.pi * self.minimal_distance)) ** 2

        # Calculate thermal noise based on constants and temperature.
        k = 1.37e-23  # Boltzmann's constant
        T = 293  # Temperature in Kelvin
        bruit_thermique = k * T * FREQUENCY_BANDS_WIDTH

        # Generate thermal noise using a normal distribution.
        noise_t = np.random.normal(0, np.sqrt(bruit_thermique), len(received_signal))

        # Calculate quantization noise parameters.
        p_max = 100 * max_sig_power  # Maximum signal power
        bits = 12  # Number of bits for quantization
        delta = 2 * p_max / (2 ** bits)
        sigma = delta / 12

        # Generate quantization noise using a normal distribution.
        noise_q = np.random.normal(0, sigma, len(received_signal))

        # Combine the received signal with thermal noise and quantization noise.
        signal_with_noise = received_signal + noise_t + noise_q

        # Return a tuple containing the noisy signal and the SNR.
        return signal_with_noise, pulses

    def get_snr_for_band(self, band_id, snr_type=0):
        # Calculate the maximum signal power based on parameters.
        max_sig_power = 1.64 * self.max_erp * (
                2.99e8 / (self.bands[band_id]['start_frequency'] * 4 * np.pi * self.minimal_distance)) ** 2

        # Calculate thermal noise based on constants and temperature.
        k = 1.37e-23  # Boltzmann's constant
        T = 293  # Temperature in Kelvin
        bruit_thermique = k * T * FREQUENCY_BANDS_WIDTH

        # Calculate quantization noise parameters.
        p_max = 100 * max_sig_power  # Maximum signal power
        bits = 12  # Number of bits for quantization
        delta = 2 * p_max / (2 ** bits)
        sigma = delta / 12

        if snr_type:
            list_snr = [10 * np.log10(pulse['pulse'].power / pulse['pulse'].pw / (sigma ** 2 + bruit_thermique)) for pulse
                        in
                        self.bands[band_id]['pulses']]
        else:
            list_snr = [10 * np.log10(pulse['pulse'].power / (sigma ** 2 + bruit_thermique)) for
                        pulse
                        in
                        self.bands[band_id]['pulses']]

        return list_snr

    def get_distance_for_max_snr(self, snr, band_id, emitter_mode=False, snr_type=0):
        """
        Voir doc
        """

        max_sig_power = 1.64 * self.max_erp * (
                2.99e8 / (self.bands[band_id]['start_frequency'] * 4 * np.pi * self.minimal_distance)) ** 2

        # Calculate thermal noise based on constants and temperature.
        k = 1.37e-23  # Boltzmann's constant
        T = 293  # Temperature in Kelvin
        bruit_thermique = k * T * FREQUENCY_BANDS_WIDTH

        # Calculate quantization noise parameters.
        p_max = 100 * max_sig_power  # Maximum signal power
        bits = 12  # Number of bits for quantization
        delta = 2 * p_max / (2 ** bits)
        sigma = delta / 12
        bruit_quantification = sigma ** 2

        max_snr_pulse_id = np.argmax(self.get_snr_for_band(band_id))
        pulse = self.bands[band_id]['pulses'][max_snr_pulse_id]['pulse']
        
        K_max = pulse.power * pulse.pw * self.distance ** 2 

        puissance_bruit = bruit_thermique + bruit_quantification
        energie_bruit = puissance_bruit*self.acquisition_time

        if emitter_mode:
            K_max = K_max / (4 * np.pi) ** 2
            distance = (K_max / energie_bruit * 10 ** (- snr / 10)) ** (1 / 4)
        else:
            distance = np.sqrt(K_max / energie_bruit * 10 ** (- snr / 10) )

        return distance
    
    def snr_from_L2S_to_avantix(self,snr, delta_f):
        return snr - 10*np.log10(FREQUENCY_BANDS_WIDTH/delta_f)


