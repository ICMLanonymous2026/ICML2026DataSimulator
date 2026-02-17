import asyncio
import random
import itertools
import threading
import time
import uuid

import scipy.signal

import numpy as np
from src.simulator.SyntheticPulse import SyntheticPulse
from src.simulator.ReceptorTools import ReceptorTools


class Transmitter(ReceptorTools):
    def __init__(self, frequency_values, pri_values, pw_values, modulation_values, erp_values,
                 frequency_pattern='stagger',
                 pri_pattern='stagger', pw_pattern='stagger', modulation_pattern='stagger', erp_pattern='stagger',
                 distance=1, iq = False,
                 id=0):
        
        self.iq = iq

        if not id:
            self.id = str(uuid.uuid4())
        else:
            self.id = id

        super().__init__()

        self.frequency_values = frequency_values
        self.pri_values = pri_values
        self.pw_values = pw_values
        self.modulation_values = modulation_values
        self.erp_values = erp_values

        self.frequency_pattern = frequency_pattern
        self.pri_pattern = pri_pattern
        self.modulation_pattern = modulation_pattern
        self.pw_pattern = pw_pattern
        self.erp_pattern = erp_pattern

        self.current_frequency_index = 0
        self.current_pri_index = 0
        self.current_pw_index = 0
        self.current_modulation_index = 0
        self.current_erp_index = 0

        self.pulses = []
        self.signal = []
        self.time = []

        self.is_running = False
        self.pulse_time = 0
        self.thread = None
        self.distance = distance

    def describe(self):
        """
        Prints a description of the transmitter's attributes.
        """
        print('----- Transmitter with ID:', self.id, ' description -----')
        print('... values ...')
        print('Frequency:', self.frequency_values)
        print('PRI:', self.pri_values)
        print('PW:', self.pw_values)
        print('Modulation:', self.modulation_values)
        print('Effective radiated power:', self.erp_values)
        print('... pattern ...')
        print('Frequency :', self.frequency_pattern)
        print('PRI :', self.pri_pattern)
        print('Modulation :', self.modulation_pattern)
        print('PW :', self.pw_pattern)
        print('Effective radiated power :', self.modulation_pattern)

    def update_index(self, index, pattern_type, values):
        """
        Updates the index based on the specified pattern type.
        Parameters:
            index (int): Current index value.
            pattern_type (str): Type of pattern ("linear" or "jitter").
            values (list): List of values for the pattern.
        Returns:
            int: Updated index value.
        """
        if pattern_type == 'linear':
            return (index + 1) % len(values)
        if pattern_type == 'constant':
            return index
        # stagger case
        else:
            return random.choice(range(len(values)))

    def update_caracteristic_value(self, pattern_type, values, index=None):
        if pattern_type == 'jitter':
            min_value = values['min']
            max_value = values['max']
            return random.uniform(min_value, max_value)
        # stagger, linear, user, sinusoidal
        else:
            return values[index]

    def get_all_possible_pulses(self, consider_transmitter_distance=True):
        """
        Generates all possible pulse combinations based on provided frequency, PRI, PW, and modulation values.
        Parameters:
            consider_transmitter_distance (bool, optional): If True, considers the distance between transmitter and
                receptor for time of arrival calculations.
        """
        # Generate all combinations of frequency, PRI, PW, and modulation values
        all_possible_combinations = list(
            itertools.product(self.frequency_values, self.pri_values, self.pw_values, self.modulation_values,
                              self.erp_values))

        # Iterate through each combination of parameters
        for params in all_possible_combinations:
            freq, pri, pw, mod, erp = params

            # Get the frequency bands for the current frequency
            bands = self.get_band_for_freq(freq)

            # Iterate through each band for the current frequency
            for band_id in bands:
                band = self.bands[band_id]

                # Shift the frequency based on the band
                new_freq = self.shift_frequency_for_band(freq, band)

                # Create a SyntheticPulse instance with the shifted frequency and other parameters
                pulse = SyntheticPulse(new_freq, pw, mod['type'], mod['value'], erp=erp, distance=self.distance,
                                       fe=self.fe, iq = self.iq)

                # Calculate the time of arrival for the pulse
                if consider_transmitter_distance:
                    time_of_arrival = self.pulse_time + self.distance / 3e9
                else:
                    time_of_arrival = self.pulse_time

                # Create a JSON representation of the pulse and its emission time
                pulse_json = {'pulse': pulse, 'emission_time': time_of_arrival, 'transmitter_id': self.id}

                # Append the pulse to the list of pulses in the corresponding band
                self.bands[band['band_id']]['pulses'].append(pulse_json)

            # Update the pulse time based on the PRI
            self.pulse_time += pri

        # Reset the pulse time after generating all pulses
        self.pulse_time = 0

    def describe_pulses(self, band_id):
        """
        Prints descriptions of pulses in the specified frequency band.
        Parameters:
            band_id (int): ID of the frequency band.
        """
        pulses = self.bands[band_id]['pulses']
        if len(pulses) == 0:
            return
        index = 0
        print('---- Pulses description ----')
        for pulse in pulses:
            print('=> Pulse number :', index, ' emitted at :', pulse['emission_time'], ' by :', pulse['transmitter_id'])
            pulse['pulse'].describe()
            index += 1

    async def start_signal_acquisition(self, duration):
        """
        Starts the simulation of real-time pulse signal acquisition.
        Parameters:
            duration (float): Duration of the simulation in seconds.
        """
        self.acquisition_time = duration
        if not self.is_running:
            self.is_running = True
            self.pulse_time = 0
            await asyncio.create_task(self.simulate_real_time_acquisition(duration))
        return True

    async def stop_signal_acquisition(self):
        """
       Stops the simulation of real-time pulse signal acquisition.
       """
        self.is_running = False

    async def initialize(self):
        self.pulses = []
        self.signal = []
        self.time = []
        for band in self.bands:
            band['pulses'] = []

    async def simulate_real_time_acquisition(self, duration=None, max_pulse_number=None,
                                             consider_transmitter_distance=True):
        """
        Simulates real-time pulse signal acquisition.

        Parameters:
            duration (float, optional):
                Duration of the simulation in seconds. If None, the simulation continues until manually stopped.
            consider_transmitter_distance (bool, optional):
                If True, considers the distance between transmitter and receptor for time of arrival calculations.
        """

        # Loop as long as the is_running flag is True
        while self.is_running:
            # Get current parameter values based on pattern and indices
            frequency = self.update_caracteristic_value(self.frequency_pattern,
                                                        self.frequency_values, self.current_frequency_index)
            pw = self.update_caracteristic_value(self.pw_pattern, self.pw_values, self.current_pw_index)
            pri = self.update_caracteristic_value(self.pri_pattern, self.pri_values, self.current_pri_index)
            try:
                modulation = self.update_caracteristic_value(self.modulation_pattern, self.modulation_values,
                                                         self.current_modulation_index)
            except:
                modulation = None
            erp = self.update_caracteristic_value(self.erp_pattern, self.erp_values, self.current_erp_index)

            wave_length = 2.99e8/frequency
            power = 1.64 * erp * (wave_length ** 2) / (4 * np.pi * self.distance) ** 2

            # Update indices based on respective patterns
            self.current_frequency_index = self.update_index(self.current_frequency_index, self.frequency_pattern,
                                                             self.frequency_values)
            self.current_pri_index = self.update_index(self.current_pri_index, self.pri_pattern, self.pri_values)
            self.current_pw_index = self.update_index(self.current_pw_index, self.pw_pattern, self.pw_values)
            try:
                self.current_modulation_index = self.update_index(self.current_modulation_index, self.modulation_pattern,
                                                              self.modulation_values)
            except:
                self.current_modulation_index = 0
            self.current_erp_index = self.update_index(self.current_erp_index, self.erp_pattern,
                                                       self.erp_values)

            # Get the bands associated with the current frequency
            bands = self.get_band_for_freq(frequency)

            pulse_number = 0
            time_of_arrival = 0

            # Iterate through each band and generate pulses
            for band_id in bands:
                band = self.bands[band_id]
                new_freq = self.shift_frequency_for_band(frequency, band)

                # Calculate the time of arrival for the pulse
                if consider_transmitter_distance:
                    time_of_arrival = self.pulse_time + self.distance / 3e9
                else:
                    time_of_arrival = self.pulse_time

                # Create a SyntheticPulse instance
                try:
                    pulse = SyntheticPulse(new_freq, pw, modulation['type'],
                                       modulation['value'], power=power, fe=self.fe, iq = self.iq)
                except:
                    pulse = SyntheticPulse(new_freq, pw, power=power, fe=self.fe, iq = self.iq)

                pulse_number += 1

                # Create a JSON representation of the pulse and its emission time
                pulse_json = {'pulse': pulse, 'emission_time': time_of_arrival, 'transmitter_id': self.id}

                # Append the pulse to the list of pulses in the corresponding band
                self.bands[band['band_id']]['pulses'].append(pulse_json)

            # Check if the duration limit has been reached
            if duration is not None and time_of_arrival > duration:
                break

            # Update the pulse time based on the PRI
            self.pulse_time = self.pulse_time + pri

            if max_pulse_number is not None and pulse_number >= max_pulse_number:
                break

        # Reset the pulse time after the simulation
        self.pulse_time = 0
        return True

    def update_distance(self, distance):
        for band in self.bands:
            for pulse in band['pulses']:
                pulse['pulse'].power = pulse['pulse'].power*(self.distance/distance)**2
        self.distance = distance
