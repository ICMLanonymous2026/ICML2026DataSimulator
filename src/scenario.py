import torch
import numpy as np
from .tools import generate_radar_pulse_train, compute_distance


def interceptor_signal_acquisition(interceptor, emitters, fe, acquisition_time):
    """
    Génère le signal acquis par l'intercepteur en combinant les signaux des émetteurs environnants.

    Parameters:
    - interceptor: Un dictionnaire contenant les informations sur l'intercepteur.
    - emitters: Une liste de dictionnaires contenant les informations sur les émetteurs (y compris les paramètres des impulsions).
    - fe: Fréquence d'échantillonnage (Hz).
    - acquisition_time: Temps d'acquisition (s).

    Returns:
    - acquired_signal: Le signal acquis combiné des différents émetteurs.
    """
    c=2.99e8
    
    # Initialiser le signal acquis par l'intercepteur
    acquired_signal = torch.zeros(int(acquisition_time * fe))
    pulses = []

    # Boucle sur tous les émetteurs
    for emitter in emitters:
        # Calculer la distance entre l'émetteur et l'intercepteur
        distance = compute_distance(interceptor['x'], interceptor['y'], emitter['x'], emitter['y'])
        
        # Déterminer le retard (delay) en fonction de la distance (on considère que le signal se propage à la vitesse de la lumière)
        # delay_time = distance / c  # Temps de propagation en secondes
        # delay_samples = int(delay_time * fe)  # Retard en nombre d'échantillons
        delay_time = 0
        delay_samples = 0
        
        # Générer le train d'impulsions de l'émetteur
        for waveform in emitter['waveforms']:
            wave_length = c/waveform['fp']
            waveform['power'] = waveform['erp'] * (wave_length ** 2) / (4 * np.pi * distance) ** 2

        pulse_train, generated_pulses = generate_radar_pulse_train(emitter['waveforms'], fe, acquisition_time)
        for pulse in generated_pulses: 
            # pulse['emission_time']+=delay_time
            pulse['emitter_label'] = emitter['label']
        pulses = np.concatenate((pulses,generated_pulses))
        
        # Insérer le signal de l'émetteur avec le retard calculé dans le signal acquis
        if delay_samples < len(acquired_signal):
            # Ajouter le signal émis avec le retard dans le signal global
            end_idx = min(len(acquired_signal), delay_samples + len(pulse_train))
            acquired_signal[delay_samples:end_idx] += pulse_train[:end_idx - delay_samples]

    k = 1.38e-23  # Constante de Boltzmann en J/K
    T = 290  # Température du système en Kelvin
    B = fe/2  # Bande passante (habituellement la moitié de la fréquence d'échantillonnage)
    noise_power = torch.tensor(k * T * B)  # Puissance du bruit en watts
    noise = torch.randn_like(torch.tensor(acquired_signal)) * torch.sqrt(noise_power/fe)
    noisy_signal = acquired_signal + noise
    # noisy_signal = acquired_signal 
    

    return noisy_signal, pulses


# def emitter_signal_acquisition(selected_entity, interceptor, emitters, acquisition_time):
#     """
#     Génère le signal acquis par l'émetteur sélectionné.

#     Args:
#         selected_entity (str): L'émetteur sélectionné.
#         interceptor (dict): Les informations sur l'intercepteur.
#         emitters (list): Liste d'émetteurs avec leurs paramètres.
#         acquisition_time (float): Temps d'acquisition en secondes.

#     Returns:
#         torch.Tensor: Le signal acquis par l'émetteur sélectionné.
#         list: Liste des impulsions générées.
#     """
#     # Filtrer pour trouver l'émetteur sélectionné
#     emitter = next((e for e in emitters if e['label'] == selected_entity), None)
#     if emitter is None:
#         raise ValueError(f"Émetteur {selected_entity} non trouvé dans la liste des émetteurs.")

#     # Calculer la bande passante maximale parmi les formes d'ondes de l'émetteur
#     max_bandwidth = max(waveform['bandwidth'] for waveform in emitter['waveforms'])

#     # Utiliser la bande passante maximale pour calculer la fréquence d'échantillonnage
#     fe = 2 * max_bandwidth  # Fréquence d'échantillonnage minimale selon la bande passante maximale

#     # Initialiser le signal acquis par l'émetteur
#     acquired_signal = torch.zeros(int(acquisition_time * fe), dtype=torch.float32)
    
#     # Calculer la distance entre l'émetteur et l'intercepteur
#     distance = compute_distance(interceptor['x'], interceptor['y'], emitter['x'], emitter['y'])
    
#     # Déterminer le retard en fonction de la distance (vitesse de la lumière)
#     speed_of_light = 3e8  # Vitesse de la lumière en m/s
#     delay_time = distance / speed_of_light  # Temps de propagation en secondes
#     delay_samples = int(delay_time * fe)  # Retard en nombre d'échantillons

#     # Générer le train d'impulsions de l'émetteur
#     for waveform in emitter['waveforms']:
#         # Calculer la puissance du signal reçu après l'aller-retour
#         wave_length = 3e8 / waveform['fp']
#         # Atténuation selon la distance au carré pour l'aller et le retour => distance^4
#         waveform['power'] = waveform['erp'] * (wave_length ** 2) / (4 * np.pi * (2*distance)) ** 4

#     pulse_train, generated_pulses = generate_radar_pulse_train(emitter['waveforms'], fe, acquisition_time)

#     # Insérer le signal de l'émetteur avec le retard calculé dans le signal acquis
#     if delay_samples < len(acquired_signal):
#         # Ajouter le signal émis avec le retard dans le signal global
#         end_idx = min(len(acquired_signal), delay_samples + len(pulse_train))
#         acquired_signal[delay_samples:end_idx] += pulse_train[:end_idx - delay_samples]

#     # Ajouter du bruit thermique
#     k = 1.38e-23  # Constante de Boltzmann en J/K
#     T = 290  # Température du système en Kelvin
#     B = fe  # Bande passante
#     noise_power = k * T * B  # Puissance du bruit en watts
#     noise = torch.randn_like(acquired_signal)  # Bruit gaussien
#     T_signal = len(acquired_signal) / fe
#     scaling_factor = torch.sqrt(noise_power * T_signal / torch.mean(noise ** 2))
#     noisy_signal = acquired_signal + noise * scaling_factor
    
#     return noisy_signal, generated_pulses, fe


# def aggregate_stats(stats_list, relative=False):
#     """
#     Agrège les statistiques de plusieurs segments pour obtenir une statistique finale par émetteur.
    
#     :param stats_list: Liste des dictionnaires contenant les statistiques (relatives ou absolues) de chaque segment.
#     :return: Un dictionnaire contenant les statistiques moyennes agrégées par émetteur.
#     """
#     # Initialisation des variables d'agrégation
#     total_error_height = []
#     total_error_width = []
#     total_error_location_x = []
#     total_error_location_y = []

#     total_absolute_height = []
#     total_absolute_width = []
    
#     # Dictionnaire pour agréger la classification par classe réelle
#     classification_matrix = defaultdict(lambda: defaultdict(int))
    
#     # Parcourir toutes les statistiques de chaque segment
#     for stats in stats_list:
#         if stats.get("mean_error_height") is not None:
#             total_error_height.append(stats["mean_error_height"])
#         if stats.get("mean_error_width") is not None:
#             total_error_width.append(stats["mean_error_width"])
#         if stats.get("mean_error_location_x") is not None:
#             total_error_location_x.append(stats["mean_error_location_x"])
#         if stats.get("mean_error_location_y") is not None:
#             total_error_location_y.append(stats["mean_error_location_y"])
#         if stats.get("image_width") is not None:
#             total_absolute_width.append(stats["image_width"])
#         if stats.get("image_height") is not None:
#             total_absolute_height.append(stats["image_height"])
        
#         # Agrégation des matrices de classification
#         for true_class, pred_counts in stats["classification_accuracy_by_class"].items():
#             for pred_class, count in pred_counts.items():
#                 classification_matrix[true_class][pred_class] += count

#     # Calcul des moyennes d'erreurs si des valeurs existent
#     mean_error_height = np.mean(total_error_height) if total_error_height else 0.0
#     mean_error_width = np.mean(total_error_width) if total_error_width else 0.0
#     mean_error_location_x = np.mean(total_error_location_x) if total_error_location_x else 0.0
#     mean_error_location_y = np.mean(total_error_location_y) if total_error_location_y else 0.0

#     referential_width = np.mean(total_absolute_width) if len(total_absolute_width) else 1.0
#     referential_height = np.mean(total_absolute_height) if len(total_absolute_height) else 1.0

#     if relative:
#         # Construction du résultat final avec la matrice de classification
#         aggregated_classification_accuracy = {}
#         for true_class, pred_counts in classification_matrix.items():
#             total_true_class = sum(pred_counts.values())
#             aggregated_classification_accuracy[true_class] = {}
#             for pred_class, count in pred_counts.items():
#                 aggregated_classification_accuracy[true_class][pred_class] = count / total_true_class if total_true_class > 0 else 0.0
#         classification_matrix = aggregated_classification_accuracy

#     return {
#         "mean_error_height": mean_error_height,
#         "mean_error_width": mean_error_width,
#         "mean_error_location_x": mean_error_location_x,
#         "mean_error_location_y": mean_error_location_y,
#         "classification_accuracy_by_class": classification_matrix, 
#         "referential_width":referential_width, 
#         "referential_height":referential_height
#     }


# def from_relative_to_absolute_stats(characterization, time_steps, frequencies):
#     """
#     Convertit les erreurs relatives de caractérisation en erreurs absolues en utilisant les time_steps et les fréquences.
    
#     :param characterization: Un dictionnaire contenant les erreurs relatives calculées via `evaluate_characterization`.
#     :param time_steps: Un tableau numpy contenant les valeurs temporelles des échantillons.
#     :param frequencies: Un tableau numpy contenant les valeurs de fréquence des échantillons.
#     :return: Un dictionnaire avec les erreurs absolues pour la hauteur, la largeur, la localisation en X et Y, et les statistiques de classification.
#     """
#     if not characterization:
#         return {
#             "mean_error_height": 0.0,
#             "mean_error_width": 0.0,
#             "mean_error_location_x": 0.0,
#             "mean_error_location_y": 0.0,
#             "classification_accuracy_by_class": characterization.get('classification_accuracy_by_class', {})
#         }

#     # Calcul des erreurs absolues
#     image_height = frequencies[-1] - frequencies[0]  # La hauteur de l'image correspond à la bande de fréquence
#     image_width = time_steps[-1] - time_steps[0]      # La largeur de l'image correspond à la durée temporelle totale

#     # Conversion des erreurs relatives en erreurs absolues
#     mean_absolute_error_height = characterization["mean_error_height"] * image_height
#     mean_absolute_error_width = characterization["mean_error_width"] * image_width
#     mean_absolute_error_location_x = characterization["mean_error_location_x"] * image_width
#     mean_absolute_error_location_y = characterization["mean_error_location_y"] * image_height

#     # Retourner les erreurs absolues avec les statistiques de classification
#     return {
#         "mean_error_height": mean_absolute_error_height,
#         "mean_error_width": mean_absolute_error_width,
#         "mean_error_location_x": mean_absolute_error_location_x,
#         "mean_error_location_y": mean_absolute_error_location_y,
#         "classification_accuracy_by_class": characterization.get("classification_accuracy_by_class", {}), 
#         "image_width":image_width, 
#         'image_height':image_height
#     }

# def convert_to_serializable(data):
#     """
#     Convert all float32 types in the data to Python's built-in float type.
#     This will ensure the data is JSON serializable.
#     """
#     if isinstance(data, dict):
#         return {k: convert_to_serializable(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [convert_to_serializable(v) for v in data]
#     elif isinstance(data, np.float32):
#         return float(data)
#     elif isinstance(data, torch.Tensor):  # In case there are tensors
#         return data.tolist()  # Convert to list
#     else:
#         return data

# async def process_model_signal_full(interceptor, emitters, fe, acquisition_time, selected_entity, segment_duration=None):
#     """
#     Traitement du signal brut de l'intercepteur en plusieurs segments temporaires.
#     Chaque segment est traité par le modèle de détection et de caractérisation
#     spécifié. Les résultats sont concaténés et retournés sous forme de liste
#     d'images contenant les boîtes englobantes des impulsions détectées.

#     Args:
#         interceptor (dict): Informations sur l'intercepteur, y compris son signal brut.
#         emitters (list): Informations sur les émetteurs, y compris leurs signaux bruts.
#         fe (float): Fréquence d'échantillonnage.
#         acquisition_time (float): Temps d'acquisition du signal brut.
#         selected_entity (str): Émetteur sélectionné pour lequel le modèle doit être exécuté.
#         segment_duration (float, optional): Durée de chaque segment temporel. Si None, le modèle est exécuté en un seul segment.

#     Returns:
#         tuple: Un tuple contenant les images avec les boîtes englobantes des impulsions détectées, les statistiques de détection et de caractérisation pour chaque émetteur, et les temps de traitement pour chaque étape du modèle.
#     """
#     model_params = {"fe": fe}
#     if interceptor['label']==selected_entity:
#         model_name = interceptor['model']
#     else: 
#         emitter = next((e for e in emitters if e['label'] == selected_entity), None)
#         model_name = emitter['model']

#     try:
#         matplotlib.use('Agg')  # Utiliser le backend sans interface graphique

#         model = choose_model(model_name=model_name, params=model_params)

#         # Initialiser les conteneurs de résultats
#         images_base64 = []
#         emitter_stats = {emitter['label']: [] for emitter in emitters}
#         detection_stats = {emitter['label']: {'true_positives': 0, 'false_positives': 0, 'total_pulses': 0} for emitter in emitters}

#         total_preprocess_time = 0
#         total_process_time = 0
#         total_postprocess_time = 0

#         total_false_positives = 0

#         if model_name != 'MATCHED_FILTER':
#             noisy_signal, pulses = interceptor_signal_acquisition(interceptor, emitters, fe, acquisition_time)

#         if model_name == 'GRID_GLRT':
#             # Code existant pour GRID_GLRT
#             final_labels, true_labels, preprocessed_data, time_metrics, frequencies, time_steps = model.run(noisy_signal, pulses, grid=(10,10))
#             total_preprocess_time += time_metrics['preprocess_time']
#             total_process_time += time_metrics['process_time']
#             total_postprocess_time += time_metrics['postprocess_time']

#             magnitude_spectrum = torch.abs(preprocessed_data).squeeze(0)

#             # Traiter les labels et générer l'image avec les boîtes englobantes
#             img_with_rectangles = draw_spectrum_with_boxes(
#                 magnitude_spectrum.numpy(),
#                 true_labels=true_labels,
#                 predicted_labels=final_labels,
#                 frequencies=frequencies,
#                 time_steps=time_steps
#             )
#             images_base64.append(img_with_rectangles)

#             # Mettre à jour les statistiques de détection et de caractérisation
#             true_positives_per_emitter, false_positives = evaluate_detections(final_labels, true_labels)
#             total_false_positives += false_positives

#             for emitter_label, true_positives in true_positives_per_emitter.items():
#                 detection_stats[emitter_label]['true_positives'] += true_positives
#                 detection_stats[emitter_label]['total_pulses'] += sum(1 for label in true_labels if label['pulse']['emitter_label'] == emitter_label)

#                 # Calculer les statistiques relatives et absolues
#                 relative_stats = evaluate_characterization([label for label in true_labels if label['pulse']['emitter_label'] == emitter_label], final_labels)
#                 absolute_stats = from_relative_to_absolute_stats(relative_stats, time_steps, frequencies)
#                 emitter_stats[emitter_label].append({'relative': relative_stats, 'absolute': absolute_stats})

#         elif model_name == 'MATCHED_FILTER':
#             num_segments=1
#             # Pour le modèle de filtrage adapté
#             noisy_signal, pulses, fe = emitter_signal_acquisition(selected_entity, interceptor, emitters, acquisition_time)
#             model.fe = fe

#             # Appliquer le modèle de filtrage adapté
#             results, threshold = model.run(noisy_signal, pulses, pfa=emitter['pfa'])

#             detections = results['detections']
#             time_metrics = results['time_metrics']

#             total_preprocess_time += time_metrics['preprocess_time']
#             total_process_time += time_metrics['process_time']
#             total_postprocess_time += time_metrics['postprocess_time']

#             images_base64 = visualize_autocorr_detections(
#                 pulses,
#                 fe=fe, 
#                 results=detections,
#                 threshold = threshold
#             )

#         else:
#             # Code pour d'autres modèles (par exemple YOLO)
#             total_samples = len(noisy_signal)
#             segment_samples = int(segment_duration * fe)
#             num_segments = total_samples // segment_samples

#             for segment_idx in range(num_segments):
#                 try:
#                     start_sample = segment_idx * segment_samples
#                     end_sample = start_sample + segment_samples
#                     signal_segment = noisy_signal[start_sample:end_sample]

#                     segment_start_time = start_sample / fe
#                     segment_end_time = end_sample / fe

#                     # Sélectionner les impulsions pertinentes pour ce segment
#                     segmented_pulses = [
#                         pulse for pulse in pulses
#                         if (pulse['emission_time'] < segment_end_time and pulse['emission_time'] + pulse['pulse'].pw > segment_start_time)
#                     ]

#                     # Passer les segmented_pulses à la méthode run du modèle
#                     predicted_labels, true_labels, preprocessed_data, time_metrics, frequencies, time_steps = model.run(signal_segment, segmented_pulses, from_time=segment_start_time)
                    
#                     total_preprocess_time += time_metrics['preprocess_time']
#                     total_process_time += time_metrics['process_time']
#                     total_postprocess_time += time_metrics['postprocess_time']

#                     # Traiter les labels et générer l'image avec les boîtes englobantes
#                     img_with_rectangles = draw_spectrum_with_boxes(
#                         preprocessed_data.numpy(),
#                         true_labels=true_labels,
#                         predicted_labels=predicted_labels,
#                         frequencies=frequencies,
#                         time_steps=time_steps
#                     )
#                     images_base64.append(img_with_rectangles)

#                     # Mettre à jour les statistiques de détection et de caractérisation basées sur les true_labels
#                     true_positives_per_emitter, false_positives = evaluate_detections(predicted_labels, true_labels)
#                     total_false_positives += false_positives

#                     for label in true_labels:
#                         emitter_label = label['pulse']['emitter_label']

#                         detection_stats[emitter_label]['true_positives'] += true_positives_per_emitter[emitter_label]
#                         detection_stats[emitter_label]['total_pulses'] += sum(1 for label in true_labels if label['pulse']['emitter_label'] == emitter_label)

#                         # Calculer les statistiques relatives et absolues
#                         relative_stats = evaluate_characterization([label], predicted_labels)
#                         absolute_stats = from_relative_to_absolute_stats(relative_stats, time_steps, frequencies)
#                         emitter_stats[emitter_label].append({'relative': relative_stats, 'absolute': absolute_stats})

#                 except Exception as e:
#                     print(f"Error processing segment {segment_idx}: {e}")
#                     continue

#         # Collecter les temps de traitement et les statistiques
#         try:
#             total_time = total_preprocess_time + total_process_time + total_postprocess_time
#             process_times = {
#                 "total_time": total_time,
#                 "average_preprocessing": total_preprocess_time / (1 if model_name == "GRID_GLRT" else num_segments),
#                 "average_processing": total_process_time / (1 if model_name == "GRID_GLRT" else num_segments),
#                 "average_postprocessing": total_postprocess_time / (1 if model_name == "GRID_GLRT" else num_segments),
#                 "total_preprocessing": total_preprocess_time,
#                 "total_processing": total_process_time,
#                 "total_postprocessing": total_postprocess_time,
#             }
#         except Exception as e:
#             print(f"Error calculating process times: {e}")
#             return images_base64, None, None

#         # Agréger les statistiques pour chaque émetteur
#         try:
#             final_stats = {}
#             for emitter_label, stats in emitter_stats.items():
#                 if stats:
#                     final_stats[emitter_label] = {
#                         "relative": aggregate_stats([stat['relative'] for stat in stats], relative=True),
#                         "absolute": aggregate_stats([stat['absolute'] for stat in stats], relative=False),
#                         "detection_stats": detection_stats[emitter_label]
#                     }
#             final_stats_serializable = convert_to_serializable(final_stats)

#         except Exception as e:
#             print(f"Error aggregating statistics: {e}")
#             return images_base64, None, process_times

#         return images_base64, final_stats_serializable, process_times

#     except Exception as e:
#         print(f"Unexpected error in process_interceptor_signal_full: {e}")
#         return None, None, None

