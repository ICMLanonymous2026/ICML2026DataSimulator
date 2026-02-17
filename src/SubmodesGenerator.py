import random
import pandas as pd
import uuid

from src.simulator.Phase_generator import PhaseGenerator


class SubmodesGenerator:
    def __init__(self, lpi=False):
        self.dataset = []
        self.lpi = lpi

    def create_simple_pulse_submodes(self, number, concat=True):
        data = []

        for _ in range(number):
            frequency = round(random.uniform(8.9e9, 10.3e9), 0)
            pulse_duration = round(random.uniform(20e-9, 200e-9), 10)
            pulse_repitition_interval = round(random.uniform(50, 150) * pulse_duration, 10)
            erp = round(random.uniform(1000, 10000), 0)

            if self.lpi:
                pulse_duration = pulse_duration * 1000
                pulse_repitition_interval = 3 * pulse_duration
                erp = erp / 1000

            radar = {
                'index': _,
                'frequency_values': [frequency],
                'pri_values': [pulse_repitition_interval],
                'pw_values': [pulse_duration],
                'frequency_pattern': 'Constant',
                'pri_pattern': 'Constant',
                'pw_pattern': 'Constant',
                'em_mode_id': str(uuid.uuid4()),
                'erp_values': [erp],
                'modulation_values': None,
                'modulation_pattern': 'Constant',
                'erp_pattern': 'Constant'
            }
            data.append(radar)

        df = pd.DataFrame(data)

        if concat:
            self.dataset.append(df)

        return df

    def create_complex_pri_pulse_submodes(self, number, concat=True, jitter=True):
        df = self.create_simple_pulse_submodes(number, concat=False)

        if jitter:
            for i in range(number):
                pulse_duration = df.at[i, 'pw_values'][0]
                if self.lpi:
                    min_pri = 2.5*pulse_duration
                    max_pri = 5*pulse_duration
                else:
                    min_pri = 50 * pulse_duration
                    max_pri = 150 * pulse_duration
                df.at[i, 'pri_values'] = {'min': min_pri, 'max': max_pri}
                df.at[i, 'pri_pattern'] = 'jitter'
        else:
            for i in range(number):
                pulse_repitition_interval = []
                pulse_duration = df.at[i, 'pw_values'][0]
                for j in range(i + 2):
                    if self.lpi:
                        pulse_repitition_interval.append(round(random.uniform(2.5, 4) * pulse_duration, 10))
                    else:
                        pulse_repitition_interval.append(round(random.uniform(50, 150) * pulse_duration, 10))
                df.at[i, 'pri_values'] = pulse_repitition_interval
                df.at[i, 'pri_pattern'] = random.choice(('stagger', 'linear'))

        if concat:
            self.dataset.append(df)

        return df

    def create_complex_frequency_pulse_submodes(self, number, complex_pri=True, concat=True, jitter=True):
        if complex_pri:
            df = self.create_complex_pri_pulse_submodes(number, concat=False)
        else:
            df = self.create_simple_pulse_submodes(number, concat=False)

        if jitter: 
            for i in range(number):
                 df.at[i, 'frequency_values'] = {'min': 8.9e9, 'max': 10.3e9}
                 df.at[i, 'frequency_pattern'] = 'jitter'
        else:
            for i in range(number):
                frequencies = []
                for j in range(i + 2):
                    frequencies.append(round(random.uniform(8.9e9, 10.3e9), 0))
                df.at[i, 'frequency_values'] = frequencies
                df.at[i, 'frequency_pattern'] = random.choice(('stagger', 'linear'))

        if concat:
            self.dataset.append(df)

        return df
    
    def create_fmcw_submodes(self, number, simple=True, concat=True, modulation_sign=0, jitter=True):
        if simple:
            df = self.create_simple_pulse_submodes(number, concat=False)
        else:
            df = self.create_complex_pri_pulse_submodes(number, concat=False)

        for i in range(number):
            if not jitter:
                frequencies = []
                mod_values = []
                for j in range(random.randint(15, 30)):
                    freq = round(random.uniform(8.9e9, 10.3e9), 0)
                    frequencies.append(freq)
                    mod_value = round(random.uniform(0.05e9, 0.5e9), 0)
                    if freq + mod_value >= 10.4e9:
                        mod_value = -mod_value
                    mod_values.append(mod_value)
                df.at[i, 'frequency_values'] = frequencies
                df.at[i, 'frequency_pattern'] = 'linear'
                df.at[i, 'modulation_pattern'] = 'linear'
                df.at[i, 'modulation_values'] = [{'type': 'linear_frequency', 'value': mod_value} for mod_value in mod_values]
            else:
                freq = df.loc[i, 'frequency_values'][0]
                mod_value = round(random.uniform(0.05e9, 1e9), 0)
                if modulation_sign =='negative':
                    mod_value = -mod_value
                elif modulation_sign == 0:
                    mod_value = random.choice([mod_value, -mod_value])
                df.at[i, 'modulation_values'] = [{'type': 'linear_frequency', 'value': mod_value}]

        if concat:
            self.dataset.append(df)

        return df

    def create_complex_pulse_submodes(self, number, concat=True):
        df = self.create_complex_frequency_pulse_submodes(number, complex_pri=True, concat=False)
        if concat:
            self.dataset.append(df)
        return df

    def create_psk_submodes(self, number, simple=True, phase_type='barker_biphasique', moments_number=None, concat=False):
        """
            Creates a DataFrame of Phase Modulated radars with varying frequencies and modulation patterns.

            Args:
            number (int): Number of radars to generate.
            simple (bool): Determines whether to use simple or complex radar characteristics.

            Returns:
            pandas.DataFrame: DataFrame containing FMCW radar characteristics.
            """
        # Using the appropriate function to generate base characteristics
        if simple:
            df = self.create_simple_pulse_submodes(number, concat=False)
        else:
            df = self.create_complex_pulse_submodes(number, concat=False)

        for i in range(number):
            if phase_type == 'barker_biphasique':
                if not moments_number:
                    moments_number = random.choice([2, 3, 11, 13])
            else:
                if not moments_number:
                    moments_number = random.randint(3, 15) ** 2

            phases_gen = PhaseGenerator(phase_type=phase_type, moments_number=moments_number)
            phase = phases_gen.generate_phase_list()
            complete_type = phase_type + '_' + str(moments_number)
            df.at[i, 'modulation_values'] = [{'type': 'phase_shift', 'value': phase, 'complete_type':complete_type}]

        if concat:
            self.dataset.append(df)

        return df

    def get_generated_datasets(self):
        return pd.concat(self.dataset, ignore_index=True)