import sys 
sys.path.append('/data/RAWSIM/RMA/Thesis_work')
from iclr_simulator.simulator.multi_res_generation_with_seg import generate_emitters_scenarios, generate_and_store_spectrum_multi

F_E = 4e9
acquisition_time = 2048 * 64 / F_E

min_time = 128 / F_E 
B = F_E/2

signal_defs = {
    "waveforms": {
        "FSK_CODE1":  { "duration_min": min_time*2, "duration_max": 2048 / F_E * 2 ,
                        "bandwidth_min": 0.01*B,  "bandwidth_max": 0.01*B, "p": 1 },

        "FSK_CODE2":  { "duration_min": min_time*2, "duration_max": 2048 / F_E * 2,
                        "bandwidth_min": 0.01*B,  "bandwidth_max": 0.01*B, "p": 1 },

        "FSK_CODE3":  { "duration_min": min_time*2, "duration_max": 2048 / F_E * 2,
                        "bandwidth_min": 0.03*B,  "bandwidth_max": 0.03*B, "p": 1 },

        "FSK_CODE4":  { "duration_min": min_time*2, "duration_max": 2048 / F_E * 2,
                        "bandwidth_min": 0.03*B,  "bandwidth_max": 0.03*B, "p": 1 },
    },
}


stft_cfgs = [
            {"nperseg": 128, 'nfft':128, "noverlap": 0, "fs": F_E},
            {"nperseg": 2048, 'nfft':2048, "noverlap": 0, "fs": F_E},
        ]

CLASS_INDEX_TO_NAME = {
    0: "FSK_CODE1",
    1: "FSK_CODE2",
    2: "FSK_CODE3",
    3: "FSK_CODE4",
}

print('start of generatting scenarios')

scenarios = generate_emitters_scenarios(nb_scenarios=10000, snr_range_db=(-10,10), signal_defs=signal_defs, stft_cfgs= stft_cfgs, max_nb_emitters=3, max_nb_interferences=0, inr_range_db=(-20,20))

acquisition_time = 2048 * 64 / F_E
generate_and_store_spectrum_multi(scenarios=scenarios, base_path='/data/RAWSIM/RMA/Thesis_work/iclr_simulator/fsk_codes_dataset', split_train_test=True, acquisition_time=acquisition_time, stft_cfgs=stft_cfgs, class_index_to_name = CLASS_INDEX_TO_NAME)
