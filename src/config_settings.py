# config_settings.py

def get_settings(section):
    """Retrieve settings for a specific section."""
    config = {
        "GLOBAL": {
            "SAMPLING_FREQUENCY": "4000000000",  # 1 MHz
            "MIN_FREQUENCY_BANDS": "2000000000",
            "MAX_FREQUENCY_BANDS": "4000000000",
            "FREQUENCY_BANDS_WIDTH": "2000000000",
            "OVERLAP": "0",
            "MIN_FREQUENCY_ACQUISITION": "2000000000",
            "MAX_FREQUENCY_ACQUISITION": "4000000000",
            "LISTENING_TIME": "10",  # 10 seconds
            "MINIMAL_DISTANCE": "10",  # in meters
            "MAXIMAL_DISTANCE": "10000",  # in meters
            "MAXIMAL_ERP": "50000",  # in watts
        },
        "RECEPTOR": {
            "IMPEDANCE": "50",  # in ohms
            "RECEPTION_CHANNEL_NOISE_DENSITY": "-174",  # in dBm/Hz
            "QUANTIFICATION_NOISE_POWER": "-90",  # in dBfs
            "MIN_FREQUENCY_FMOP": "1000",  # in Hz
            "MAX_FREQUENCY_FMOP": "200000",  # in Hz
        },
    }

    # Return the settings for the requested section
    return config.get(section, {})
