import asyncio
import time

import numpy as np
from .ReceptorTools import ReceptorTools


class Environment(ReceptorTools):
    def __init__(self, transmitters, start_times=None, iq = False):
        """
        Class representing the simulation environment containing multiple transmitters.

        Parameters:
            transmitters (list): List of Transmitter objects representing the transmitters in the environment.
            start_times (list, optional): List of start times for each transmitter's signal acquisition.
        """
        super().__init__()
        self.transmitters = transmitters
        self.is_running = False
        self.thread = None
        self.iq = iq

        # Set start times for each transmitter's signal acquisition
        if start_times is None:
            self.start_times = [0 for i in self.transmitters]
        else:
            self.start_times = start_times

    def describe(self):
        """
        Describe the environment by printing details of each transmitter.
        """
        for transmitter in self.transmitters:
            transmitter.describe()
            print('')

    def describe_pulses(self, band_id):
        """
        Describe the pulses emitted by each transmitter in the environment.
        """
        for transmitter in self.transmitters:
            transmitter.describe_pulses(band_id)

    async def initialize(self):
        """
        Stop signal acquisition for all transmitters and adjust pulse emission times based on start times.
        """
        tasks = [transmitter.initialize() for transmitter in self.transmitters]
        await asyncio.gather(*tasks)

    async def stop_signal_acquisition(self):
        """
        Stop signal acquisition for all transmitters and adjust pulse emission times based on start times.
        """
        tasks = [transmitter.stop_signal_acquisition() for transmitter in self.transmitters]
        await asyncio.gather(*tasks)

        for i, transmitter in enumerate(self.transmitters):
            for band in transmitter.bands:
                band_id = band['band_id']
                new_pulses = []
                for pulse_json in band['pulses']:
                    pulse_json['emission_time'] += self.start_times[i]
                    new_pulses.append(pulse_json)
                self.bands[band_id]['pulses'] = np.concatenate((self.bands[band_id]['pulses'], new_pulses))

    async def start_signal_acquisition(self, duration=None, max_pulse_number=None):
        """
        Start signal acquisition for all transmitters.
        """
        if max_pulse_number:
            max_pris = [np.max(transmitter.pri_values) for transmitter in self.transmitters]
            max_pri = np.max(max_pris)
            duration = max_pri * max_pulse_number
        self.acquisition_time = duration
        tasks = [transmitter.start_signal_acquisition(duration=duration) for transmitter in self.transmitters]
        await asyncio.gather(*tasks)


