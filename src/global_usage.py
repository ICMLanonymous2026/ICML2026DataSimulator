import asyncio
import copy
import datetime
import os
import random
import json
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import stft
from scipy.signal import find_peaks
import math
from scipy.spatial.distance import cdist
import matplotlib.patches as patches
from PIL import Image
import io
import base64
import torch

from tqdm import tqdm

from .SyntheticPulse import SyntheticPulse
from .Transmitter import Transmitter
from .Environment import Environment
from .ReceptorTools import ReceptorTools

######################################################
# Submodes table and data storage
######################################################
def updatdes_submodes(submodes_tab):
    new_submodes_tab = submodes_tab.rename(
        columns={'freq_pattern': 'frequency_values', 'pri_pattern': 'pri_values', 'pw_pattern': 'pw_values',
                 'frequencyLawType': 'frequency_pattern', 'priLawType': 'pri_pattern', 'pwLawType': 'pw_pattern'})

    def format_wave_type(row):
        if 'Chirp' in row['Type of wave']:
            return 'linear_frequency'
        if 'No Mod' in row['Type of wave']:
            return 0
        else:
            return 'phase_shift'

    new_submodes_tab['modulation_values'] = submodes_tab.apply(
        lambda row: [{'type': format_wave_type(row), 'value': row['modulation']}], axis=1)
    new_submodes_tab['erp_values'] = submodes_tab.apply(
        lambda row: [row['erp']], axis=1)
    new_submodes_tab['frequency_values'] = new_submodes_tab['frequency_values'].apply(
        lambda freq_list: [freq * 10 ** 6 for freq in freq_list])
    new_submodes_tab['pri_values'] = new_submodes_tab['pri_values'].apply(
        lambda pri_list: [pri * 10 ** (-6) for pri in pri_list])
    new_submodes_tab['pw_values'] = new_submodes_tab['pw_values'].apply(
        lambda pw_list: [pw * 10 ** (-6) for pw in pw_list])
    new_submodes_tab.drop(columns=['Type of wave', 'modulation'], inplace=True)
    new_submodes_tab = new_submodes_tab.assign(modulation_pattern='Constant')
    new_submodes_tab = new_submodes_tab.assign(erp_pattern='Constant')
    return new_submodes_tab


def from_submodes_to_transmitters(submodes_tab):
    """
        Create instances of the Transmitter class from submodes_tab data.

        Parameters:
            submodes_tab (pd.DataFrame): The DataFrame containing data for each Transmitter.
        Returns:
            list: List of Transmitter instances.
        """
    transmitters = []  # List to store Transmitter instances

    for index, row in submodes_tab.iterrows():
        # Extract values from DataFrame columns
        frequency_values = row['frequency_values']
        pri_values = row['pri_values']
        pw_values = row['pw_values']
        modulation_values = row['modulation_values']
        erp_values = row['erp_values']

        # Extract patterns if necessary (from DataFrame columns)
        frequency_pattern = row['frequency_pattern']
        pri_pattern = row['pri_pattern']
        pw_pattern = row['pw_pattern']
        modulation_pattern = row['modulation_pattern']
        erp_pattern = row['erp_pattern']

        # if math.isnan(row['em_mode_id']):
        #     transmetter_id = None 
        # else: 
        transmetter_id = row['em_mode_id']

        # Create an instance of Transmitter with the extracted values
        transmitter = Transmitter(
            frequency_values, pri_values, pw_values, modulation_values, erp_values,
            frequency_pattern=frequency_pattern,
            pri_pattern=pri_pattern, pw_pattern=pw_pattern,
            modulation_pattern=modulation_pattern, erp_pattern=erp_pattern,
            distance=1, id=transmetter_id
        )

        transmitters.append(transmitter)  # Add the instance to the list

    return transmitters


def get_random_transmitters(transmitter_list, n):
    """
    Get a random selection of transmitters from the list.

    Parameters:
        transmitter_list (list): List of Transmitter instances.
        n (int): Number of transmitters to select.
    Returns:
        list: List of randomly selected Transmitter instances.
    """
    if n > len(transmitter_list):
        raise ValueError("Number of transmitters to select is greater than the available transmitters.")

    random_transmitters = random.sample(transmitter_list, n)
    return random_transmitters


def get_multiple_random_transmitters(transmitter_list, number_of_acq, max_number_transmitter=10):
    """
    Get a list of m sets of random transmitters, each with a number of transmitters
    less than or equal to max_number_transmitter.

    Parameters:
        transmitter_list (list): List of Transmitter instances.
        m (int): Number of sets of random transmitters to generate.
        max_number_transmitter (int): Maximum number of transmitters in each set.
    Returns:
        list: List of lists containing sets of random Transmitter instances.
    """
    random_transmitter_sets = []
    for _ in range(number_of_acq):
        n = random.randint(1, max_number_transmitter)
        random_transmitters = get_random_transmitters(transmitter_list, n)
        random_transmitter_sets.append(random_transmitters)
    return random_transmitter_sets


# def get_class_for_pulse(pulse):
#     if pulse.modulation_value is None:
#         pulse_class = 'pulse'
#     elif pulse.modulation_type == 'linear_frequency':
#         if pulse.modulation_value > 0:
#             pulse_class = 'ascendant_chirp'
#         else:
#             pulse_class = 'descendant_chirp'
#     elif pulse.modulation_type == 'phase_shift':
#         pulse_class = pulse.modulation_type + str(len(pulse.modulation_value))
#     elif pulse.modulation_type == 's-law':
#         if pulse.modulation_value > 0:
#             pulse_class = 'ascendant_s-law'
#         else:
#             pulse_class = 'descendant_s-law'
#     else:
#         pulse_class = pulse.modulation_type
#     return pulse_class



def from_stft_get_labels(frequencies, time_steps, pulses, add_freq=0, add_time=0):
    """
    Generate labels directly in normalized time and frequency ranges without pixel conversion.

    Args:
        frequencies (numpy.ndarray): Array of frequency values.
        time_steps (numpy.ndarray): Array of time values.
        pulses (list): List of pulses with their properties.
        add_freq (float): Additional margin to add to the frequency range (in Hz).
        add_time (float): Additional margin to add to the time range (in seconds).

    Returns:
        list: A list of dictionaries containing normalized labels for each pulse.
    """
    stft_labels = []
    total_time = time_steps[-1] - time_steps[0]
    total_freq = frequencies[-1] - frequencies[0]

    for pulse in pulses:
        # Define pulse properties
        width_time = pulse['pulse'].pw
        start_time = pulse['emission_time']
        end_time = start_time + width_time
        center_freq = pulse['pulse'].carrier_frequency
        bandwidth = pulse['pulse'].bandwidth

        pulse_class = pulse['pulse'].modulation_name

        # Calculate normalized time range
        left_time = max(start_time - add_time, time_steps[0])
        right_time = min(end_time + add_time, time_steps[-1])
        x_center = ((left_time + right_time) / 2 - time_steps[0]) / total_time
        x_dim = (right_time - left_time) / total_time

        # Calculate normalized frequency range
        low_freq = max(center_freq - bandwidth / 2 - add_freq, frequencies[0])
        high_freq = min(center_freq + bandwidth / 2 + add_freq, frequencies[-1])
        y_center = ((low_freq + high_freq) / 2 - frequencies[0]) / total_freq
        y_dim = (high_freq - low_freq) / total_freq

        # Add normalized label
        label = {
            'x_center': x_center,
            'y_center': y_center,
            'x_dim': x_dim,
            'y_dim': y_dim,
            'pulse_class': pulse_class,
            # 'pulse': pulse
        }

        stft_labels.append(label)
    
    return stft_labels


def get_relative_snr(x_dim, pw , snr, acq_duration=0.0005):
    """ Get the relative SNR that you got for the signal in your acquisition window"""
    pw_relative =  x_dim*acq_duration
    snr_relative = 10*np.log10(pw/pw_relative)+snr
    return snr_relative


def compute_spectrum_psnr(spectrum: torch.Tensor, fe: float, nfft: int, nperseg: int, T: float = 290.0):
    """
    Calcule le PSNR global d'un spectrogramme (obtenu sans bruit ajouté),
    en se basant sur une modélisation thermique du bruit blanc additif.

    Args:
        spectrum (torch.Tensor): Spectrogramme (puissance) sans bruit, de forme (freq, time)
        fe (float): Fréquence d'échantillonnage (Hz)
        nfft (int): Nombre de points FFT
        nperseg (int): Taille des fenêtres STFT
        T (float): Température du système en Kelvin (défaut : 290 K)

    Returns:
        float: PSNR global en dB
    """
    k = 1.38e-23  # Constante de Boltzmann (J/K)
    B = fe / 2    # Bande passante (Hz)
    
    # Puissance du bruit thermique (Watts)
    noise_power_total = k * T * B
    
    # Puissance moyenne du bruit attendue par cellule STFT
    noise_power_per_cell = noise_power_total / fe * (nperseg / nfft)

    max_signal_power = spectrum.max().item()

    if noise_power_per_cell <= 0 or max_signal_power <= 0:
        return float('inf')

    psnr = 10 * np.log10(max_signal_power / noise_power_per_cell)
    return psnr


def save_pulses_acquisition(file_path, band, transmitters=None, max_pulse_number=None, acquisition_params=None):
    """
    Save acquired pulses to a JSON file.

    Parameters:
        file_path (str): Path to the JSON file for saving acquired pulses.
        band (json): json containing the band information.
        transmitters (list, optional): List of transmitters used in the acquisition. Default is None.
        number_of_acq (int, optional): Number of acquisitions. Default is None.
        acquisition_params (json): Parameters of the acquisition. Default is None.
    """
    pulses = []
    for pulse in band['pulses']:
        pulse['pulse'] = pulse['pulse'].__dict__
        pulses.append(pulse)

    transmitters = [transmitter.__dict__ for transmitter in transmitters]

    # Prepare the JSON data to be saved, including acquired pulses, transmitters, and number of acquisitions
    json_to_save = {'acquisition': pulses, 'band_start_freq': band['start_frequency'],
                    'band_end_freq': band['end_frequency'], 'transmitters': transmitters,
                    'max_pulse_number': max_pulse_number, 'band_id': band['band_id'],
                    'acquisition_params': acquisition_params}

    # Write the JSON data to the specified file
    with open(file_path, 'w') as f:
        json.dump(json_to_save, f)

    for pulse in band['pulses']:
        pulse_object = object.__new__(SyntheticPulse)
        pulse_object.__dict__ = pulse['pulse']
        pulse['pulse'] = pulse_object
        pulses.append(pulse)


def simulate_and_store_acquisitions(storage_path, submodes_tab, number_of_acq=10, max_number_transmitter=10,
                                    max_pulse_number=100):
    """
    Simulate acquisitions based on submodes and store the acquired pulses.

    Parameters:
        storage_path (str): Path to store the acquired pulses.
        submodes_tab (pd.DataFrame): DataFrame containing submodes information.
        number_of_acq (int): Number of acquisitions to simulate.
        max_number_transmitter (int): Maximum number of transmitters per acquisition.
        max_pulse_number (int): Maximum number of pulses in an acquisition.
    """
    # Check if storage path is a directory that exists else return an exception
    if not os.path.isdir(storage_path):
        raise NotADirectoryError(f"Path {storage_path} does not exist or is not a directory.")

    # Create a directory in this path with the name 'data_stored_'+date of the day
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    storage_path = os.path.join(storage_path, f"data_stored_{today_date}")

    # Convert submodes DataFrame to transmitters
    print('start of the simulation')
    transmitters = from_submodes_to_transmitters(submodes_tab)

    # Generate random sets of transmitters for each acquisition
    random_transmitter_sets = get_multiple_random_transmitters(transmitters, number_of_acq, max_number_transmitter)

    # Loop through each set of random transmitters
    for i, transmitters_set in enumerate(tqdm(random_transmitter_sets, desc="Simulating acquisitions")):

        if not os.path.isdir(f"{storage_path}/acquisition_{i}"):
            os.makedirs(f"{storage_path}/acquisition_{i}")

        # Create an Environment instance with the selected transmitters
        env = Environment(transmitters_set)

        # Simulate acquisition and get acquired pulses
        asyncio.run(env.initialize())
        asyncio.run(env.start_signal_acquisition(max_pulse_number=max_pulse_number))
        asyncio.run(env.stop_signal_acquisition())

        acquisition_params = env.describe_acquisition_params(print=False)

        for band in env.bands:
            acquired_pulses = band['pulses']
            if len(acquired_pulses) > 0:
                acquisition_storage_path = f"{storage_path}/acquisition_{i}/{band['band_id']}.json"
                transmitter_ids = [transmitter.id for transmitter in transmitters_set]
                save_pulses_acquisition(acquisition_storage_path, band=band, transmitters=transmitter_ids,
                                        max_pulse_number=max_pulse_number, acquisition_params=acquisition_params,
                                        signal=env.compute_signal_with_noise(band['band_id']))


###################################################
# Getting data from storage and data preprocessing
###################################################

def compute_data_from_path(file_path):
    """
        Load a JSON from a file.

        Parameters:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Dictionary containing the JSON data.
        """
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    aqcuired_pulses = json_data['acquisition']
    acquisition_params = json_data['acquisition_params']
    receptor = ReceptorTools(**acquisition_params)
    signal = receptor.compute_signal_from_pulse(pulses=aqcuired_pulses)
    return signal


# def preprocess_images(image_path, target_size):
#     """
#     Preprocess an image for further analysis.

#     Parameters:
#         image_path (str): Path to the image.
#         target_size (tuple): Target size for resizing the image.

#     Returns:
#         numpy.ndarray: Preprocessed image.
#     """
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)

#     # Resize the image to the target size
#     image = cv2.resize(image, target_size)

#     # Convert the image to RGB color space
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Normalize pixel values to the range [0, 1]
#     image = image / 255.0

#     # Expand the dimensions of the image to match the model's input shape
#     image = np.expand_dims(image, axis=0)

#     return image


def preprocess_stft_from_path(file_path, target_size, fe=4e9, window='hann', nperseg=64, noverlap=None, ):
    """
    Preprocess STFT data from a signal.

    Parameters:
        file_path (str): Path to the signal data file.

    Returns:
        numpy.ndarray: Preprocessed STFT image.
    """
    # Compute signal data from the file path
    signal = np.load(file_path + '/signal.npy')

    frequencies, time_steps, stft_data = stft(signal, fe, window=window, nperseg=nperseg,
                                              noverlap=noverlap)

    return stft_data