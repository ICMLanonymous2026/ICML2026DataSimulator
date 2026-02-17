import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import random
import torch.nn.functional as F
import base64
import matplotlib.patches as patches
import io
from collections import defaultdict

# Importer les fonctions de calcul de l'autocorrélation et du spectre temps-fréquence
from .SyntheticPulse import SyntheticPulse
from .Phase_generator import PhaseGenerator


def GetModulationParams(waveform_type, bandwidth, momentsNumber=None, pw=None, fp=None, fe=None):
    """
    Return modulation_type, modulation_value, amplitude_mask
    Extended to support multi-resolution FSK coded waveforms.
    """

    # =============== STANDARD RADAR WAVEFORMS ===============
    if waveform_type in ['LFM_short', 'LFM_long']:
        return 'LFM', bandwidth, None

    if waveform_type in ['NLFM_short', 'NLFM_long']:
        return 'NLFM', bandwidth, None

    if waveform_type in ['P1','P2','P3','P4','frank','barker_biphasique','random_biphasique']:
        modulation_type = 'PSK'
        if not momentsNumber:
            momentsNumber = int(bandwidth * pw)
            if momentsNumber == 0:
                momentsNumber = 1
        modulation_value = PhaseGenerator(
            phase_type=waveform_type,
            momentsNumber=momentsNumber
        ).generate_phase_list()
        return modulation_type, modulation_value, None

    if waveform_type in ['T1','T2','T3','T4']:
        modulation_type = 'PSK'
        number_of_phase_states = 2
        if waveform_type == 'T1':
            momentsNumber = int(np.sqrt(bandwidth*pw/number_of_phase_states))
        elif waveform_type == 'T2':
            momentsNumber = int(np.sqrt(bandwidth*pw/number_of_phase_states/2))
        momentsNumber = max(momentsNumber,1)

        modulation_value = PhaseGenerator(
            phase_type=waveform_type,
            momentsNumber=momentsNumber,
            signal_period=pw,
            number_of_phase_states=number_of_phase_states,
            delta_F=bandwidth,
            fe=fe
        ).generate_phase_list()
        return modulation_type, modulation_value, None

    # ===========================================================
    #           CODED FSK WAVEFORMS
    # ===========================================================

    if waveform_type in {"FSK_CODE1", "FSK_CODE2", "FSK_CODE3", "FSK_CODE4"}:
        code_idx = int(waveform_type[-1])     # 1, 2, 3, 4

        # --------------------------------------------------------------
        # 1. Compute K based on minimal symbol time
        # --------------------------------------------------------------
        T_sym = 128.0 / fe                    # minimal STFT window time
        K     = max(4, int(pw / T_sym))       # at least 4 symbols

        # --------------------------------------------------------------
        # 2. Frequency offsets from STFT resolution
        # --------------------------------------------------------------
        Df = fe / 2048 * 2
        f_c = fp

        # small and large spacing tones (in offsets)
        f1 = f_c - 0.5 * Df
        f2 = f1 + Df

        f3 = f_c - 1.5 * Df
        f4 = f3 + 3 * Df

        tones_12 = np.array([f1 - f_c, f2 - f_c]) 
        tones_34 = np.array([f3 - f_c, f4 - f_c])

        # --------------------------------------------------------------
        # 3. Generate symbol-level frequency and amplitude sequences
        # --------------------------------------------------------------
        symbols = np.arange(K)

        if code_idx == 1:
            idx      = symbols % 2
            freq_seq = tones_12[idx]
            amp_seq  = np.ones(K)

        elif code_idx == 2:
            idx      = (symbols // 2) % 2
            freq_seq = tones_12[idx]
            amp_seq  = (symbols % 2 == 0).astype(float)

        elif code_idx == 3:
            idx      = symbols % 2
            freq_seq = tones_34[idx]
            amp_seq  = np.ones(K)

        elif code_idx == 4:
            idx      = (symbols // 2) % 2
            freq_seq = tones_34[idx]
            amp_seq  = (symbols % 2 == 0).astype(float)

        # --------------------------------------------------------------
        # 4. Sample-level amplitude mask, fully consistent
        # --------------------------------------------------------------
        N_samples = int(round(pw * fe))

        # avoid rounding drift by distributing excess samples properly
        samples_per_symbol = [N_samples // K] * K
        remainder          = N_samples - sum(samples_per_symbol)
        for i in range(remainder):
            samples_per_symbol[i] += 1    # distribute remainders

        # Build mask
        amp_mask = np.zeros(N_samples, dtype=float)
        pos = 0
        for a, ns in zip(amp_seq, samples_per_symbol):
            end = pos + ns
            amp_mask[pos:end] = a
            pos = end

        return "FSK", freq_seq, amp_mask

    if waveform_type == 'FHSS' or waveform_type == 'FSK':
        if waveform_type == 'FHSS':
            num_hops = 200
        else: 
            num_hops = 6
        freq_min = - bandwidth / 2
        freq_max = bandwidth / 2
        modulation_type = 'FSK'
        modulation_value = np.random.uniform(freq_min, freq_max, size=num_hops)
        return modulation_type, modulation_value, None

    if waveform_type == 'DSSS':
        momentsNumber = int(bandwidth * pw)
        momentsNumber = max(momentsNumber, 1)
        modulation_value = PhaseGenerator(
            phase_type='random_biphasique',
            momentsNumber=momentsNumber
        ).generate_phase_list()
        return 'PSK', modulation_value, None

    return waveform_type, bandwidth, None




def generate_radar_pulse(waveform_type, power, pw, fe, fp, bandwidth=0, momentsNumber=0):
    """
    Génère l'impulsion radar propre sans bruit et répète le signal en fonction du nombre de cycles.

    Parameters:
    - waveform_type: Type de forme d'onde (LFM, NLFM, PSK, etc.).
    - erp: Puissance rayonnée équivalente de l'émetteur (W).
    - pw: Largeur d'impulsion (Pulse Width) (s).
    - fe: Fréquence d'échantillonnage (Hz).
    - fp: Fréquence porteuse (Hz).
    - f_mod: Largeur de bande de modulation (Hz).
    - momentsNumber: Nombre de moments pour les modulations de phase.
    - numberOfCycles: Nombre de cycles du signal à générer.

    Returns:
    - emitted_signal: Signal de l'impulsion radar émise répété sur le nombre de cycles.
    """
    # Récupération des paramètres de modulation en fonction du type d'onde
    modulation_type, modulation_value, amplitude_mask = GetModulationParams(
        waveform_type, bandwidth, momentsNumber, pw, fp, fe
    )

    pulse = SyntheticPulse(
        pw=pw,
        modulation_type=modulation_type,
        carrier_frequency=fp,
        modulation_value=modulation_value,
        amplitude_mask=amplitude_mask, 
        fe=fe,
        power=power
    )
    single_pulse = torch.tensor(pulse.generate_pulse(), dtype=torch.float32)  # Convertir en tensor
    if modulation_type == 'LFM' or modulation_type=='NLFM':
        pulse.modulation_name = modulation_type
    else:
        pulse.modulation_name = waveform_type
    pulse.bandwidth = bandwidth
        
    return single_pulse, pulse

def generate_radar_pulse_train(pulse_params_list, fe, acquisition_time, fixed_delay = None):
    """
    Génère un train d'impulsions radar en fonction d'une liste d'impulsions avec leurs paramètres et PRI respectives,
    tout en respectant un temps d'acquisition donné. Si le temps d'acquisition n'est pas terminé, le cycle redémarre avec le
    PRI de la dernière impulsion appliqué à la première impulsion. Le premier signal démarre après un délai aléatoire
    compris entre 0 et le temps de la dernière PRI.

    Parameters:
    - pulse_params_list: Liste des paramètres pour chaque impulsion, incluant le PRI et les autres paramètres nécessaires pour 
      générer l'impulsion.
    - fe: Fréquence d'échantillonnage (Hz).
    - acquisition_time: Temps d'acquisition total (s).

    Exemple d'un élément dans pulse_params_list :
    {
      'waveform_type': 'LFM',
      'power': 100,
      'pw': 0.001,
      'fe': 1e6,
      'fp': 5e9,
      'bandwidth': 2e6,
      'momentsNumber': 4,
      'numberOfCycles': 1,
      'pri': 0.01
    }

    Returns:
    - train_signal: Signal complet du train d'impulsions avec les intervalles PRI dans la limite du temps d'acquisition.
    """
    
    # Initialiser la liste pour stocker le train d'impulsions
    train_signal = []
    pulses = []
    
    # Variable pour suivre le temps accumulé
    current_time = 0

    # Dernier PRI pour calculer le délai de départ aléatoire
    last_pri = pulse_params_list[-1]['pri']
    first_pw = pulse_params_list[0]['pw']
    max_delay = 0.9*acquisition_time - first_pw
    
    # Délai aléatoire avant la première impulsion (entre 0 et le dernier PRI)
    random_delay = random.uniform(0, max_delay)
    if fixed_delay:
        random_delay = fixed_delay
    delay_samples = int(random_delay * fe)
    
    # Ajouter le délai aléatoire avant la première impulsion
    if delay_samples > 0:
        silence = torch.zeros(delay_samples, dtype=torch.float32)
        train_signal.append(silence)
        current_time += random_delay

    # print('proccessing pulses :',pulse_params_list)

    while current_time < acquisition_time:
        for pulse_index, pulse_params in enumerate(pulse_params_list):
            # Générer l'impulsion radar avec les paramètres donnés
            pulse_signal, pulse = generate_radar_pulse(
                waveform_type=pulse_params['waveform_type'],
                power=pulse_params['power'],
                pw=pulse_params['pw'],
                fe=fe,
                fp=pulse_params['fp'],
                bandwidth=pulse_params.get('bandwidth', 0),
                momentsNumber=pulse_params.get('momentsNumber', 0),
            )
            pulses.append({'pulse':pulse,'emission_time':current_time, 'waveform_id':pulse_params['waveform_id']})
            
            pri = pulse_params['pri']  # PRI en secondes

            # if current_time + pulse_params['pw']< acquisition_time:
            #     # Ajouter l'impulsion générée au train
            train_signal.append(pulse_signal)
            current_time += pulse_params['pw']
            
            # Vérifier si on peut ajouter cette impulsion dans le temps d'acquisition restant
            if current_time + pri > acquisition_time or current_time + pulse_params['pw'] > acquisition_time:
                # Calculer le temps restant et stopper la génération si on dépasse le temps d'acquisition
                remaining_time = acquisition_time - current_time
                silence_samples = int(remaining_time * fe)
                if silence_samples>0:
                    silence = torch.zeros(silence_samples, dtype=torch.float32)
                    train_signal.append(silence)
                    current_time += remaining_time
                break
            
            # Calculer la durée de silence en fonction du PRI et de la longueur de l'impulsion
            else:
                # Ajouter le silence (signal de zéros) entre les impulsions
                silence_samples = int(pri * fe)  # Convertir la durée en nombre d'échantillons
                silence = torch.zeros(silence_samples, dtype=torch.float32)
                train_signal.append(silence)
                current_time += pri

        # # Redémarrer le cycle si le temps d'acquisition n'est pas encore terminé
        # if current_time < acquisition_time:
        #     # Utiliser le PRI de la dernière impulsion pour démarrer la première impulsion du cycle suivant
        #     silence_duration = last_pri
        #     if current_time + silence_duration < acquisition_time:
        #         silence_samples = int(silence_duration * fe)
        #         silence = torch.zeros(silence_samples, dtype=torch.float32)
        #         train_signal.append(silence)
        #         current_time += silence_duration

    # Concaténer toutes les impulsions et silences pour former le train complet
    train_signal = torch.cat(train_signal)
    
    return train_signal, pulses

# def normalize_spectrum(spectrum, to_image = True):
#     norm_squared = torch.abs(spectrum)**2
#     sigma = torch.std(norm_squared)
#     scaled_spectrum = norm_squared/(sigma)
#     log_spectrum = 10*torch.log10(scaled_spectrum+1e-16)
#     clipped_spectrum = torch.clamp(log_spectrum,min=-25, max=25)
#     normalized_spectrum = (clipped_spectrum+25)/(50)
#     if to_image:
#         image_spectrum = (normalized_spectrum*255).to(torch.uint16).cpu()
#         return image_spectrum
#     else:
#         return normalized_spectrum
    

def normalize_spectrum(
    spectrum,
    min_clamp: float = -30.,
    max_clamp: float = 10.,
    noise_est_percentile: float = .5,
):
    # Puissance
    power = spectrum.abs().pow_(2)

    # Quantile exact sur l’ensemble des coefficients
    q = torch.quantile(power, noise_est_percentile)
    sigma2_hat = q / (-math.log1p(-noise_est_percentile))   # log1p pour la stabilité

    # Normalisation + dB + clamp
    log_spec = 10.0 * torch.log10(power / sigma2_hat + 1e-10)
    log_spec.clamp_(min=min_clamp, max=max_clamp)
    return (log_spec + min_clamp) / (max_clamp - min_clamp)
    

def resize_spectrum_and_axes(spectrum, f, t, target_size):
    """
    Resize the spectrogram along with the frequency and time axes.

    Args:
        spectrum (torch.Tensor): The input spectrogram.
        f (torch.Tensor): The original frequency axis.
        t (torch.Tensor): The original time axis.
        target_size (tuple): Target size (height, width) for the resized spectrogram.

    Returns:
        torch.Tensor: Resized spectrogram.
        torch.Tensor: Rescaled frequency axis.
        torch.Tensor: Rescaled time axis.
    """
    # Ensure target_size is a tuple
    if isinstance(target_size, int):
        target_size = (target_size, target_size)  # Convert single int to a tuple (height, width)

    # Get original sizes
    original_height, original_width = spectrum.shape[-2], spectrum.shape[-1]
    
    # Calculate scaling factors
    freq_scale = target_size[0] / original_height
    time_scale = target_size[1] / original_width
    
    # Resize spectrogram
    spectrum = spectrum.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    spectrum_resized = F.interpolate(spectrum, size=target_size, mode='bilinear', align_corners=False)
    spectrum_resized = spectrum_resized.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    # Resize frequency and time axes
    f_resized = torch.linspace(f.min().item(), f.max().item(), target_size[0])
    t_resized = torch.linspace(t.min().item(), t.max().item(), target_size[1])
    
    return spectrum_resized, f_resized, t_resized

def check_intersection(true_box, pred_box, json=False):
    """
    Vérifie s'il y a une intersection entre une boîte réelle et une boîte prédite.
    """
    if json:
        c_true, x_center_true, y_center_true, w_true, h_true = true_box['pulse_class'], true_box['x_center'], true_box['y_center'], true_box['x_dim'], true_box['y_dim']
        c_pred, x_center_pred, y_center_pred, w_pred, h_pred = pred_box['pulse_class'], pred_box['x_center'], pred_box['y_center'], pred_box['x_dim'], pred_box['y_dim']
    else:
        x_center_true, y_center_true, w_true, h_true = true_box
        x_center_pred, y_center_pred, w_pred, h_pred = pred_box

    # Calcul des coins haut-gauche et bas-droite pour la boîte réelle (true_box)
    x1_true = x_center_true - w_true / 2
    y1_true = y_center_true - h_true / 2
    x2_true = x_center_true + w_true / 2
    y2_true = y_center_true + h_true / 2

    # Calcul des coins haut-gauche et bas-droite pour la boîte prédite (pred_box)
    x1_pred = x_center_pred - w_pred / 2
    y1_pred = y_center_pred - h_pred / 2
    x2_pred = x_center_pred + w_pred / 2
    y2_pred = y_center_pred + h_pred / 2

    # Vérification de l'intersection
    if x1_true < x2_pred and x2_true > x1_pred and y1_true < y2_pred and y2_true > y1_pred:
        return True
    return False


def evaluate_detections(predicted_labels, true_labels):
    """
    Évalue les bonnes détections (true positives) par émetteur et les fausses alarmes totales
    en vérifiant l'intersection entre les boîtes réelles et les boîtes prédites.

    :param predicted_labels: Liste des boîtes prédites par le modèle (format [x_center, y_center, width, height])
    :param true_labels: Liste des boîtes réelles associées aux pulses (format [x_center, y_center, width, height, pulse.emitter_label])
    :return: dict contenant les true_positives par émetteur et le total des false_positives
    """
    true_positives_per_emitter = {}
    total_false_positives = 0

    # On garde la trace des boîtes réelles déjà associées à une bonne détection
    detected_true_boxes = set()

    for pred in predicted_labels:
        is_true_positive = False
        associated_emitter = None


        # Comparer chaque boîte prédite à toutes les boîtes réelles
        for idx, true_box in enumerate(true_labels):
            emitter_label = true_box['pulse']['emitter_label']

            # Si l'emitter n'existe pas encore dans le dictionnaire, l'ajouter
            if emitter_label not in true_positives_per_emitter:
                true_positives_per_emitter[emitter_label] = 0

            if check_intersection(true_box, pred, json=True):
                is_true_positive = True
                associated_emitter = emitter_label
                detected_true_boxes.add(idx)  # Marquer cette boîte réelle comme détectée
                break  # Sortir de la boucle dès qu'une intersection est trouvée

        # Si une boîte prédite correspond à une boîte réelle, c'est une bonne détection
        if is_true_positive:
            true_positives_per_emitter[associated_emitter] += 1
        else:
            # Sinon, c'est une fausse alarme
            total_false_positives += 1

    return true_positives_per_emitter, total_false_positives


def compute_distance(x1, y1, x2, y2):
    """Calcule la distance euclidienne entre deux points (x1, y1) et (x2, y2)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def draw_spectrum_with_boxes(spectrum, frequencies=None, time_steps=None, true_labels=None, predicted_labels=None, cmap='grey'):
    """
    Dessine les boîtes englobantes sur un spectre donné en fonction des labels réels et prédits.
    Retourne une image en base64.
    
    - spectrum: spectre à afficher (2D ou 3D, si 3D => RGB)
    - frequencies: axe des fréquences (optionnel)
    - time_steps: axe du temps (optionnel)
    - true_labels: boîtes réelles à afficher en bleu
    - predicted_labels: boîtes prédites à afficher en vert ou rouge
    """

    # Vérification des dimensions du spectre
    if spectrum.ndim == 2:  # Spectre en noir et blanc
        img = spectrum
    elif spectrum.ndim == 3 and spectrum.shape[0] == 3:  # RGB
        img = np.flipud(np.transpose(spectrum, (1, 2, 0))) 
    else:
        raise ValueError("Le spectre doit être 2D ou 3D avec 3 canaux pour RGB.")
    
    time_steps = np.array(time_steps)
    frequencies = np.array(frequencies)

    # Initialisation de la figure avec des dimensions ajustées
    fig, ax = plt.subplots(figsize=(15, 10))


    # Affichage du spectre
    if time_steps is not None and frequencies is not None:
        if spectrum.ndim == 2:
            # Ajustement des dimensions de frequencies et time_steps
            time_steps_adj = np.linspace(min(time_steps), max(time_steps), img.shape[1] + 1)
            frequencies_adj = np.linspace(min(frequencies), max(frequencies), img.shape[0] + 1)

            # Utilisation de pcolormesh avec les axes
            mesh = ax.pcolormesh(time_steps_adj, frequencies_adj, img, shading='auto', cmap=cmap)
            fig.colorbar(mesh, ax=ax, label='Amplitude')  # Ajouter une barre de couleur pour plus de clarté
        else:
            # Pour les spectres RGB, nous utilisons imshow
            ax.imshow(img, extent=[min(time_steps), max(time_steps), min(frequencies), max(frequencies)], aspect='auto')

    else:
        # Si les axes ne sont pas fournis, afficher l'image sans échelle
        ax.imshow(img, aspect='auto', cmap='gray')

    # Redimensionner les labels normalisés en fonction des axes
    def scale_label(label, time_steps, frequencies):
        time_range = time_steps[-1] - time_steps[0]
        freq_range = frequencies[-1] - frequencies[0]

        # Inverser les coordonnées de y_center pour aligner l'axe des fréquences avec le sens des labels
        x_center = label['x_center'] * time_range + time_steps[0]
        y_center = label['y_center'] * freq_range + frequencies[0] 

        x_dim = label['x_dim'] * time_range
        y_dim = label['y_dim'] * freq_range
        return x_center, y_center, x_dim, y_dim

    # Ajout des boîtes réelles en bleu
    if true_labels is not None:
        for label in true_labels:
            x_center, y_center, x_dim, y_dim = scale_label(label, time_steps, frequencies)
            top_left = (x_center - x_dim / 2, y_center - y_dim / 2)
            rect = patches.Rectangle(top_left, x_dim, y_dim, linewidth=2, edgecolor='blue', facecolor='none', label='true')
            ax.add_patch(rect)

    # Par défaut, la boîte est en rouge
    color = '#FF00FF00' 

    # Ajout des boîtes prédites en fonction des intersections
    if predicted_labels is not None:
        for pred_label in predicted_labels:
            # Vérification d'intersection avec les boîtes réelles
            if true_labels is not None:
                for true_label in true_labels: 
                    if check_intersection(true_label, pred_label, json=True):
                        color = '#00FF00'  # Vert fluo
                        break

            
            x_center, y_center, x_dim, y_dim = scale_label(pred_label, time_steps, frequencies)
            top_left_pred = (x_center - x_dim / 2, y_center - y_dim / 2)

            # Dessin de la boîte prédite avec la couleur adéquate
            rect_pred = patches.Rectangle(top_left_pred, x_dim, y_dim, linewidth=2, edgecolor=color, facecolor='none', label='predicted')
            ax.add_patch(rect_pred)


    # Axes et titres
    ax.set_title("Spectre avec boîtes englobantes")
    ax.set_xlabel("Temps" if time_steps is not None else "X-axis")
    ax.set_ylabel("Fréquence" if frequencies is not None else "Y-axis")
    
    ax.grid(False)

    # Ajustement des limites des axes pour correspondre aux échelles des données
    if time_steps is not None and frequencies is not None:
        ax.set_xlim([time_steps.min(), time_steps.max()])
        ax.set_ylim([frequencies.min(), frequencies.max()])

    # Conversion de la figure en base64
    img_byte_arr = io.BytesIO()
    plt.savefig(img_byte_arr, format='PNG', bbox_inches='tight')
    plt.close(fig)  # Fermeture de la figure pour libérer la mémoire
    img_byte_arr.seek(0)

    # Encodage en base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return img_base64


def evaluate_characterization(true_labels, predicted_labels):
    """
    Évalue la caractérisation en calculant les erreurs relatives pour la hauteur, la largeur et la localisation
    des boîtes, ainsi que des statistiques détaillées sur la classification des impulsions.
    
    :param true_labels: Liste des labels réels avec les informations sur les boîtes et la classe des impulsions.
    :param predicted_labels: Liste des labels prédits avec les mêmes informations.
    :return: Un dictionnaire contenant les erreurs relatives moyennes pour 'height', 'width', 'location_x', 'location_y' 
             et les statistiques de classification par classe en valeur absolue.
    """
    # Initialisation des erreurs relatives et matrices de classification
    errors_height = []
    errors_width = []
    errors_location_x = []
    errors_location_y = []
    
    classification_matrix = defaultdict(lambda: defaultdict(int))

    # Si true_labels ou predicted_labels sont vides ou nuls, retourner une erreur nulle par défaut
    if not true_labels or not predicted_labels:
        return {
            "mean_error_height": 0.0,
            "mean_error_width": 0.0,
            "mean_error_location_x": 0.0,
            "mean_error_location_y": 0.0,
            "classification_accuracy_by_class": {}
        }

    # Calcul des erreurs relatives et matrice de confusion pour la classification
    for true_label, pred_label in zip(true_labels, predicted_labels):
        # Calcul de l'erreur relative pour la hauteur (y_dim)
        if true_label['y_dim'] != 0:
            error_height = abs(true_label['y_dim'] - pred_label['y_dim'])
            errors_height.append(error_height)

        # Calcul de l'erreur relative pour la largeur (x_dim)
        if true_label['x_dim'] != 0:
            error_width = abs(true_label['x_dim'] - pred_label['x_dim']) 
            errors_width.append(error_width)

        # Calcul de l'erreur de localisation pour x et y séparément
        if true_label['x_dim'] != 0:
            error_location_x = abs(true_label['x_center'] - pred_label['x_center'])
            errors_location_x.append(error_location_x)
        if true_label['y_dim'] != 0:
            error_location_y = abs(true_label['y_center'] - pred_label['y_center']) 
            errors_location_y.append(error_location_y)

        # Comptage pour la classification : on stocke les transitions entre classes réelles et prédites
        true_class = true_label['pulse_class']
        pred_class = pred_label['pulse_class']
        classification_matrix[true_class][pred_class] += 1

    # Calcul des erreurs relatives moyennes (0.0 si les listes sont vides)
    mean_error_height = sum(errors_height) / len(errors_height) if errors_height else 0.0
    mean_error_width = sum(errors_width) / len(errors_width) if errors_width else 0.0
    mean_error_location_x = sum(errors_location_x) / len(errors_location_x) if errors_location_x else 0.0
    mean_error_location_y = sum(errors_location_y) / len(errors_location_y) if errors_location_y else 0.0

    # Retour des erreurs moyennes et de la matrice de classification en valeurs absolues
    return {
        "mean_error_height": mean_error_height,
        "mean_error_width": mean_error_width,
        "mean_error_location_x": mean_error_location_x,
        "mean_error_location_y": mean_error_location_y,
        "classification_accuracy_by_class": classification_matrix  
    }
