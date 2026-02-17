import sys 
sys.path.append('/data/RAWSIM/RMA/Thesis_work')
from iclr_simulator.simulator.multi_res_generation_with_seg import generate_emitters_scenarios, generate_and_store_spectrum_multi

F_E = 4e9
acquisition_time = 2048 * 64 / F_E

min_time = 128 / F_E 
B = F_E/2

signal_defs = {
    "waveforms": {

        "no_mod":              { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.0,       "bandwidth_max": 0.0,       "p": 1 },
        "LFM_short":           { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.005*B,   "bandwidth_max": 0.75*B,    "p": 1 },
        "random_biphasique":   { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },

        "QPSK":                { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "QAM16":               { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "QAM64":               { "duration_min": min_time, "duration_max": 1e-6,             "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },

        "FSK":                 { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.005*B,   "bandwidth_max": 0.75*B,    "p": 1 },
        "LFM_long":            { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.005*B,   "bandwidth_max": 0.9*B,     "p": 1 },

        # -------- Codes désactivés --------
        "NLFM_short":          { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "NLFM_long":           { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },

        "P1":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "P2":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "P3":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "P4":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },

        "frank":               { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },

        "T1":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "T2":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "T3":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
        "T4":                  { "duration_min": min_time, "duration_max": acquisition_time, "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },

        "none":                { "duration_min": 0.0,      "duration_max": 0.0,             "bandwidth_min": 0.0,        "bandwidth_max": 0.0,       "p": 0 },
    },

    "interferences": {
        "OFDM":                { "duration_min": acquisition_time, "duration_max": acquisition_time, "bandwidth_min": 0.005*B, "bandwidth_max": B/2,  "p": 1 },
        "FHSS":                { "duration_min": acquisition_time, "duration_max": acquisition_time, "bandwidth_min": 0.005*B, "bandwidth_max": B/2,  "p": 1 },
        "DSSS":                { "duration_min": acquisition_time, "duration_max": acquisition_time, "bandwidth_min": 0.005*B, "bandwidth_max": B/2,  "p": 1 },
        "none":                { "duration_min": 0.0, "duration_max": 0.0, "bandwidth_min": 0.0, "bandwidth_max": 0.0, "p": 0 },
    }
}

stft_cfgs = [
            {"nperseg": 512, 'nfft':512, "noverlap": 0, "fs": F_E},
            {"nperseg": 256, 'nfft':256, "noverlap": 0, "fs": F_E},
            {"nperseg": 128, 'nfft':128, "noverlap": 0, "fs": F_E},
            {"nperseg": 1024, 'nfft':1024, "noverlap": 0, "fs": F_E},
            {"nperseg": 2048, 'nfft':2048, "noverlap": 0, "fs": F_E},
        ]

print('start of generatting scenarios')

scenarios = generate_emitters_scenarios(nb_scenarios=100000, snr_range_db=(-20,20), signal_defs=signal_defs, stft_cfgs= stft_cfgs, max_nb_emitters=7, max_nb_interferences=3, inr_range_db=(-20,20), seed=444)

acquisition_time = 2048 * 64 / F_E
generate_and_store_spectrum_multi(scenarios=scenarios, base_path='/data/RAWSIM/RMA/Thesis_work/iclr_simulator/rf_dataset_v2', split_train_test=True, acquisition_time=acquisition_time, stft_cfgs=stft_cfgs, seed=444)
