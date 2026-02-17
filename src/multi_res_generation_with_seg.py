"""generate_emitters_scenarios.py
---------------------------------------------------------------------
Génère des scénarios radar/télécom respectant une contrainte **PSNR**
sur **au moins une** configuration STFT parmi `stft_cfgs`.

© 2025 – Licence libre pour le projet « Génération de données »
"""

from __future__ import annotations

import random
import uuid
from typing import List, Dict, Any, Sequence, Optional, Tuple

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd

import torch.nn.functional as F

import torch
import os 
from .SignalAnalysis import SignalAnalysis
from .scenario import interceptor_signal_acquisition

from pathlib import Path
import json
import math
import shutil

standard_signal_defs = {
    "waveforms": {
        "no_mod":             { "duration_min": 0.0,     "duration_max": 0.0,     "bandwidth_min": 0.0,      "bandwidth_max": 0.0,       "p": 1/17 },
        "LFM_short":          { "duration_min": 1e-7,    "duration_max": 5e-5,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,       "p": 1/17 },
        "NLFM_short":         { "duration_min": 1e-7,    "duration_max": 5e-5,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,       "p": 1/17 },
        "random_biphasique":  { "duration_min": 1e-6,    "duration_max": 1e-5,    "bandwidth_min": 1e6,      "bandwidth_max": 50e6,       "p": 1/17 },
        "LFM_long":           { "duration_min": 5e-5,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 15e8,      "p": 1/17 },
        "NLFM_long":          { "duration_min": 5e-5,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 15e8,      "p": 1/17 },

        "P1":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "P2":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "P3":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "P4":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "frank":              { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },

        "T1":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "T2":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "T3":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },
        "T4":                 { "duration_min": 1e-7,    "duration_max": 1e-2,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },

        "FSK":                { "duration_min": 1e-6,    "duration_max": 1e-4,    "bandwidth_min": 1e6,      "bandwidth_max": 1e9,      "p": 1/17 },

        "none":               { "duration_min": 0.0,     "duration_max": 0.0,     "bandwidth_min": 0.0,      "bandwidth_max": 0.0,       "p": 1/17 },
    },
    "interferences": {
        "OFDM":               { "duration_min": 1,    "duration_max": 1,    "bandwidth_min": 50e6,     "bandwidth_max": 400e6,     "p": 0.25 },
        "FHSS":               { "duration_min": 1,    "duration_max": 1,    "bandwidth_min": 50e6,      "bandwidth_max": 400e6,       "p": 0.25 },
        "DSSS":               { "duration_min": 1,    "duration_max": 1,    "bandwidth_min": 50e6,      "bandwidth_max": 400e6,     "p": 0.25 },
        "none":               { "duration_min": 0.0,     "duration_max": 0.0,     "bandwidth_min": 0.0,      "bandwidth_max": 0.0,       "p": 0.25 },
    }
}


__all__ = [
    "generate_emitters_scenarios",
]

# ---------------------------------------------------------------------------
# CONSTANTES PHYSIQUES
# ---------------------------------------------------------------------------

F_E = 4.0e9               # fréquence d’échantillonnage (Hz)
LIGHT_SPEED = 2.99e8       # c (m/s)
K_BOLTZMANN = 1.38e-23     # J/K
STANDARD_TEMP = 290        # K
RX_BW = F_E / 2            # bande du récepteur (Hz)
NOISE_POWER = K_BOLTZMANN * STANDARD_TEMP * RX_BW
FIXED_DISTANCE = 1         # m (distance normalisée)

T_S = 2048 * 64 / F_E
MIN_W_R = 0.03       # >= 3 % de T
MIN_H_R = 0.03       # >= 3 % de B

PSNR_MIN = -3

# CLASS_INDEX_TO_NAME = {
#     0: 'no_mod', 1: 'LFM', 2: 'NLFM', 3: 'frank', 4: 'P1', 5: 'P2',
#     6: 'P3', 7: 'P4', 8: 'random_biphasique', 9: 'FSK', 10: 'DSSS',
#     11: 'T1', 12: 'T2', 13: 'T3', 14: 'T4'
# }

CLASS_INDEX_TO_NAME = {
    0: 'no_mod', 1: 'LFM', 2: 'random_biphasique', 3: 'FSK', 4: 'DSSS', 5: 'QPSK', 6:'QAM16', 7:'QAM64', 8:'FHSS', 9:'OFDM'
}

# CLASS_INDEX_TO_NAME = {
#     0: "FSK_CODE1",
#     1: "FSK_CODE2",
#     2: "FSK_CODE3",
#     3: "FSK_CODE4",
# }

# table inverse  nom → id  (pour un accès O(1))
NAME_TO_CLASS_ID = {v: k for k, v in CLASS_INDEX_TO_NAME.items()}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _draw_fp_with_bw(bw: float, fp_min: float, fp_max: float) -> float:
    """Retourne un fp aléatoire tel que la bande reste dans [fp_min, fp_max]."""
    low = fp_min + bw / 2
    high = fp_max - bw / 2
    if low >= high:
        return 0.5 * (fp_min + fp_max)
    return np.random.uniform(low, high)


def generate_emitters_scenarios(
    *,
    nb_scenarios: int = 1_000,
    snr_range_db: Tuple[float, float] = (-25.0, -15.0),
    inr_range_db: Tuple[float, float] = (-10.0, 0.0),
    max_nb_emitters: int = 5,
    random_nb_emitters: bool = True,
    max_nb_interferences: int = 1,
    random_nb_interferences: bool = True,
    signal_defs: dict | None = None,
    seed: int | None = None,
    stft_cfgs: List[dict] | None = None,
) -> list:
    """
    Retourne une liste de scénarios. Chaque scénario est une liste d’émetteurs.

    signal_defs doit contenir deux sous-dicts "waveforms" et "interferences",
    chacun mappant un label vers un dict avec :
      - duration_min, duration_max (en secondes)
      - bandwidth_min, bandwidth_max (en Hz)
      - p (probabilité de tirage, somme des p de chaque groupe = 1)

    Ex. pour les waveforms :
      "waveforms": {
        "LFM_short": {
          "duration_min": 1e-6, "duration_max": 1e-5,
          "bandwidth_min": 1e6, "bandwidth_max": 5e6,
          "p": 0.25
        }, …
      }
    """

    # seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # STFT par défaut
    if stft_cfgs is None:
        stft_cfgs = [
            {"nperseg": 128, 'nfft':128, "noverlap": 0, "fs": F_E},
            {"nperseg": 256, 'nfft':256, "noverlap": 0, "fs": F_E},
            {"nperseg": 512, 'nfft':512, "noverlap": 0, "fs": F_E},
            {"nperseg": 1024, 'nfft':1024, "noverlap": 0, "fs": F_E},
        ]

    # Définitions par défaut
    if signal_defs is None:
        signal_defs = standard_signal_defs

    # Préparer les clés et probabilités
    wf_items = list(signal_defs["waveforms"].items())
    wf_labels, wf_defs = zip(*wf_items)
    wf_ps = np.array([d["p"] for d in wf_defs])
    wf_ps = wf_ps / wf_ps.sum()

    int_items = list(signal_defs["interferences"].items())
    int_labels, int_defs = zip(*int_items)
    int_ps = np.array([d["p"] for d in int_defs])
    int_ps = int_ps / int_ps.sum()

    scenarios = []

    for _ in tqdm(range(nb_scenarios), desc="Generating scenarios"):
        emitters_list: list = []

        # --- radars ---
        n_emitters = (random.randint(1, max_nb_emitters) if random_nb_emitters else max_nb_emitters)
        for em_idx in range(n_emitters):
            stft_cfg = np.random.choice(stft_cfgs)
            w_type = np.random.choice(wf_labels, p=wf_ps)
            if w_type == 'none':
                continue
            def_w = signal_defs["waveforms"][w_type]
            
            pw = np.random.uniform(def_w["duration_min"], def_w["duration_max"])
            bw = np.random.uniform(def_w["bandwidth_min"], def_w["bandwidth_max"])
            fp = _draw_fp_with_bw(bw, 0.1 * F_E / 2, 0.9 * F_E / 2)

            snr_db = np.random.uniform(*snr_range_db)
            # snr_db = float(_compute_required_snr_db(pw, bw, targ_psnr_db, stft_cfg))
            wavelength = LIGHT_SPEED / fp
            erp = (10**(snr_db/10) * NOISE_POWER * (4*np.pi*FIXED_DISTANCE)**2) / wavelength**2
            pri = np.random.uniform(min(2*pw,1e3/LIGHT_SPEED), 10*pw)

            waveform_dict = {
                "waveform_id": str(uuid.uuid4()),
                "waveform_type": w_type,
                "erp": erp,
                "pw": pw,
                "fe": F_E,
                "fp": fp,
                "bandwidth": bw,
                "numberOfCycles": 1,
                "pri": pri,
                "target_snr_db": snr_db,
                # "target_psnr_db": targ_psnr_db,
            }
            emitters_list.append({
                "label":        f"Emitter_{em_idx+1}",
                "emitter_type": "radar",
                "x":            FIXED_DISTANCE,
                "y":            0.0,
                "waveforms":    [waveform_dict],
                "target_snr_db": snr_db,
            })

        # --- interférences ---
        n_interf = (random.randint(0, max_nb_interferences) if random_nb_interferences else max_nb_interferences)
        for int_idx in range(n_interf):
            stft_cfg = np.random.choice(stft_cfgs)
            i_type = np.random.choice(int_labels, p=int_ps)
            if i_type == 'none':
                continue
            def_i = signal_defs["interferences"][i_type]

            pw = np.random.uniform(def_i["duration_min"], def_i["duration_max"])
            bw = np.random.uniform(def_i["bandwidth_min"], def_i["bandwidth_max"])
            fp = _draw_fp_with_bw(bw, 0.1 * F_E / 2, 0.9 * F_E / 2)

            inr_db = np.random.uniform(*inr_range_db)
            # inr_db = float(_compute_required_snr_db(pw, bw, targ_pinr_db, stft_cfg))
            wavelength = LIGHT_SPEED / fp
            erp = (10**(inr_db/10) * NOISE_POWER * (4*np.pi*FIXED_DISTANCE)**2) / wavelength**2

            waveform_dict = {
                "waveform_id": str(uuid.uuid4()),
                "waveform_type": i_type,
                "erp": erp,
                "pw": pw,
                "fe": F_E,
                "fp": fp,
                "bandwidth": bw,
                "numberOfCycles": 1,
                "pri": 0.0,
                "target_inr_db": inr_db,
                # "target_psnr_db":targ_pinr_db,
            }
            emitters_list.append({
                "label":          f"Interference_{int_idx+1}",
                "emitter_type":   "telecom",
                "x":               FIXED_DISTANCE,
                "y":               0.0,
                "waveforms":      [waveform_dict],
                "target_inr_db":   inr_db,
            })

        scenarios.append(emitters_list)

    return scenarios


def _flatten_records(scenarios: Sequence[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Met en DataFrame toutes les *waveforms* d'une liste de scénarios."""

    rows: list[dict] = []
    for sid, scenario in enumerate(scenarios):
        for emitter in scenario:
            etype = emitter.get("emitter_type", "?")
            label = emitter.get("label", f"Emitter_{sid}")
            for wf in emitter.get("waveforms", []):
                rows.append(
                    {
                        "scenario_id": sid,
                        "emitter_label": label,
                        "emitter_type": etype,
                        "waveform_type": wf.get("waveform_type"),
                        "pw": wf.get("pw"),
                        "bandwidth": wf.get("bandwidth"),
                        "fp": wf.get("fp"),
                        "snr_db": wf.get("target_snr_db"),
                        "inr_db": wf.get("target_inr_db"),
                        "psnr_db": wf.get("target_psnr_db"),
                    }
                )
    return pd.DataFrame(rows)


def _describe_numeric(df: pd.DataFrame, col: str) -> Dict[str, float]:
    vals = df[col].dropna()
    if vals.empty:
        return {}
    return {
        "count": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "25%": float(vals.quantile(0.25)),
        "50%": float(vals.median()),
        "75%": float(vals.quantile(0.75)),
        "max": float(vals.max()),
    }

# ---------------------------------------------------------------------------
# ANALYSE PRINCIPALE
# ---------------------------------------------------------------------------

def analyze_scenarios(
    scenarios: Sequence[List[Dict[str, Any]]],
    *,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Analyse les *scenarios* et génère des statistiques + graphiques.

    Parameters
    ----------
    scenarios : list
        Liste de scénarios issus de `generate_emitters_scenarios`.
    show : bool, default True
        Affiche les figures via `plt.show()` si True.
    save_dir : str | None
        Si fourni, `fig.savefig()` sera appelé pour chaque figure.

    Returns
    -------
    dict
        Statistiques descriptives par colonne numérique.
    """

    if not scenarios:
        print("[analyse] Aucune donnée à analyser.")
        return {}

    df = _flatten_records(scenarios)

    # ----------------------------- STATS NUMÉRIQUES ----------------------
    numeric_cols = [c for c in ["snr_db", "inr_db", "psnr_db", "pw", "bandwidth", "fp"] if c in df.columns]
    stats = {col: _describe_numeric(df, col) for col in numeric_cols}

    figs: list[tuple[plt.Figure, str]] = []

    # 1) Histogrammes SNR & INR -----------------------------------------
    for col, title in [("snr_db", "Histogramme SNR (radar)"), ("inr_db", "Histogramme INR (interférences)"),  ("psnr_db", "Histogramme PSNR (radar)")]:
        if col in df and df[col].notna().any():
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=40)
            ax.set_xlabel("dB")
            ax.set_ylabel("Comptage")
            ax.set_title(title)
            figs.append((fig, f"hist_{col}.png"))

    # 2) Répartition des waveforms --------------------------------------
    if "waveform_type" in df:
        fig, ax = plt.subplots()
        df["waveform_type"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Occurrences")
        ax.set_title("Répartition des waveform_type")
        figs.append((fig, "bar_waveforms.png"))

    # 3) Nuage PW vs BW --------------------------------------------------
    if {"pw", "bandwidth"}.issubset(df.columns):
        fig, ax = plt.subplots()
        ax.scatter(df["pw"], df["bandwidth"], s=10)
        ax.set_xlabel("Pulse width [s]")
        ax.set_ylabel("Bandwidth [Hz]")
        ax.set_title("PW vs BW")
        figs.append((fig, "scatter_pw_bw.png"))

    # ----------------------------- ENREGISTREMENT / AFFICHAGE ----------
    for fig, fname in figs:
        if save_dir:
            fig.savefig(f"{save_dir}/{fname}", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    return stats


def _mkdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _init_split_dirs(base: str, split: str) -> dict:
    """
    Crée une arborescence standardisée :
      base/split/
        ├── data/
        ├── labels_detect/
        ├── labels_segment/
        ├── pulses/
        └── emitters/
    """
    root_split = _mkdir(os.path.join(base, split))
    return {
        "data": _mkdir(os.path.join(root_split, "data")),
        "labels_detect": _mkdir(os.path.join(root_split, "labels_detect")),
        "labels_segment": _mkdir(os.path.join(root_split, "labels_segment")),
        "pulses": _mkdir(os.path.join(root_split, "pulses")),
        "emitters": _mkdir(os.path.join(root_split, "emitters")),
    }


def resize_to_pow2(spec: torch.Tensor,
                           f: torch.Tensor,
                           t: torch.Tensor):
    """
    Redimensionne un spectrogramme complexe `(F,T)` vers la taille la plus proche
    en puissance de deux :
        F2 = 2^round(log2(F))
        T2 = 2^round(log2(T))

    Contrairement à la version 'downsample', on peut agrandir ou réduire selon la taille initiale.

    Paramètres
    ----------
    spec : (F,T) complex
        Spectrogramme complexe sur CPU ou GPU.
    f : (F,) réel
        Axe fréquentiel associé.
    t : (T,) réel
        Axe temporel associé.

    Retourne
    --------
    spec_rs : (F2,T2) complex
        Spectrogramme redimensionné.
    f_rs    : (F2,) réel
        Axe fréquentiel redimensionné.
    t_rs    : (T2,) réel
        Axe temporel redimensionné.
    """
    device = spec.device
    real_dtype = torch.float32 if spec.dtype in (torch.complex64, torch.float32) else torch.float64

    F_src, T_src = spec.shape
    F_tgt = 1 << int(round(torch.log2(torch.tensor(F_src, dtype=torch.float32)).item()))
    T_tgt = 1 << int(round(torch.log2(torch.tensor(T_src, dtype=torch.float32)).item()))

    # Rien à faire si déjà à la bonne taille
    if (F_src, T_src) == (F_tgt, T_tgt):
        return spec, f, t

    # Conversion complexe -> réel (2 canaux)
    ri = torch.view_as_real(spec).permute(2, 0, 1).unsqueeze(0).float()  # (1,2,F,T)
    
    # Interpolation bilinéaire (plus douce que 'area' pour upsampling)
    ri_rs = F.interpolate(ri, size=(F_tgt, T_tgt), mode='nearest')

    re_rs, im_rs = ri_rs[0, 0], ri_rs[0, 1]
    spec_rs = torch.complex(re_rs, im_rs)

    # Recalcul des grilles f/t linéaires
    f0, f1 = f[0].item(), f[-1].item()
    t0, t1 = t[0].item(), t[-1].item()
    f_rs = torch.linspace(f0, f1, F_tgt, dtype=real_dtype, device=device)
    t_rs = torch.linspace(t0, t1, T_tgt, dtype=real_dtype, device=device)

    return spec_rs, f_rs, t_rs


def rebuild_labels_from_pulsesfile_json_like(
    pulses: list,
    output_json: Path, 
    name_to_class_id = NAME_TO_CLASS_ID
) -> None:
    """
    Rebuilds YOLO-like labels from an in-memory list of pulses and writes
    a pure JSON list (not wrapped inside a 'labels' object).

    Parameters
    ----------
    pulses : list
        List of pulse descriptions, each containing fields such as
        'pulse', 'emission_time', 'snr_db', etc.
    output_json : Path
        Output file path where the JSON list will be written.
    """

    labels = []

    for p in pulses:
        pulse_obj = p["pulse"]

        # Modulation name (fallback to modulation_type if name unavailable)
        mod = getattr(
            pulse_obj,
            "modulation_name",
            getattr(pulse_obj, "modulation_type", None)
        )
        class_id = NAME_TO_CLASS_ID[mod]

        t0 = p["emission_time"]
        pw = getattr(pulse_obj, "pw", 0.0)
        bw = getattr(pulse_obj, "bandwidth", 0.0)
        fc = getattr(pulse_obj, "carrier_frequency", RX_BW / 2)

        xc = float(np.clip((t0 + pw / 2) / T_S, 0.0, 1.0))
        yc = float(np.clip(fc / RX_BW,         0.0, 1.0))
        w_r = float(max(pw / T_S,   MIN_W_R))
        h_r = float(max(bw / RX_BW, MIN_H_R))

        snr  = float(p.get("snr_db", np.nan))
        psnr = {k: float(v) for k, v in p.get("psnr_db", {}).items()}

        labels.append({
            "class": class_id,
            "xc": xc, "yc": yc,
            "w": w_r, "h": h_r,
            "snr": snr,
            "psnr": psnr
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as jf:
        json.dump(labels, jf, indent=2)


def normalize_spectrum(spec: torch.Tensor,
                       psnr_min: float,
                       psnr_max: float,
                       noise_power_per_cell: float
                      ) -> torch.Tensor:
    """
    Calcule le PSNR = 10·log10(power / σ²_bruit_théorique),
    puis remappe linéairement dans [0,1] :
        psnr_clamped = clamp(psnr, psnr_min, psnr_max)
        output = (psnr_clamped - psnr_min) / (psnr_max - psnr_min)

    Args:
        spec: tenseur complexe ou réel de spectre (valeurs STFT)
        psnr_min, psnr_max: bornes de normalisation du PSNR
        noise_power_per_cell: puissance de bruit théorique par cellule (Hz×s)
                              = NOISE_POWER / fe * (nperseg / nfft**2)
    """
    # 1) Puissance spectrale
    power = spec.abs().pow_(2)

    # 2) PSNR en dB (basé sur le bruit théorique)
    psnr = 10.0 * torch.log10(power / noise_power_per_cell)

    # 3) Clamp et normalisation linéaire
    psnr_clamped = psnr.clamp(psnr_min, psnr_max)
    psnr_norm = (psnr_clamped - psnr_min) / (psnr_max - psnr_min)

    return psnr_norm



def _place_pulse_in_frame(pulse_signal: torch.Tensor, start_idx: int, total_len: int) -> torch.Tensor:
    """
    Place le signal d'une impulsion dans un frame de longueur totale total_len (échantillons),
    en respectant l'alignement temporel (délai).
    """
    frame = torch.zeros(total_len, dtype=pulse_signal.dtype, device=pulse_signal.device)
    if start_idx >= total_len:
        return frame  # hors fenêtre -> ignoré
    end_idx = min(total_len, start_idx + pulse_signal.numel())
    if end_idx > start_idx:
        frame[start_idx:end_idx] = pulse_signal[:end_idx - start_idx]
    return frame


def compute_psnr_and_segmentation_per_pulse(
    pulses: list,
    fe: float,
    stft_cfgs: list[dict],
    acquisition_time: float,
    distinct_emitters: list[str],
    pow2_resize: bool = True,
    segmentation_db: int = 10, 
):
    """
    Combine le calcul du PSNR/SNR et de la segmentation pour chaque impulsion.

    Pour chaque impulsion :
      - génère le signal (pulse.generate_pulse)
      - replace l'impulsion dans un frame de longueur acquisition_time via son délai exact
      - calcule la STFT
      - calcule PSNR et SNR (bruit thermique selon NOISE_POWER)
      - binarise à -10 dB du max pour créer le masque de segmentation
      - stocke PSNR/SNR dans pulse_data["psnr_db"] et pulse_data["snr_db"]

    Fusionne ensuite les segmentations multi-résolutions.
    """
    # --- Bruit thermique global (défini dans les constantes) ---
    # NOISE_POWER = K_BOLTZMANN * STANDARD_TEMP * RX_BW
    # RX_BW = F_E / 2

    segs_multi_res = []
    total_len = int(round(acquisition_time * fe))

    for cfg in stft_cfgs:
        nfft = cfg["nfft"]
        nperseg = cfg["nperseg"]
        noverlap = cfg.get("noverlap", nperseg // 2)
        cfg_name = f"cfg{nperseg}"

        # Bruit par cellule temps-fréquence (même formule que ton ancienne fonction)
        noise_power_per_cell = NOISE_POWER / fe * (nperseg / nfft**2)

        segmentations = []

        for p_idx, p in enumerate(pulses):
            po = p.get("pulse", None)
            if po is None or not hasattr(po, "generate_pulse"):
                raise ValueError("Pulse manquante ou sans méthode generate_pulse().")

            # --- 1) Génération du signal ---
            pulse_signal_np = po.generate_pulse()
            pulse_signal = torch.tensor(pulse_signal_np, dtype=torch.float32)

            # --- 2) Alignement temporel ---
            start_idx = int(round(float(p["emission_time"]) * fe))
            frame = _place_pulse_in_frame(pulse_signal, start_idx, total_len)

            # --- 3) STFT ---
            SA = SignalAnalysis(frame, fe)
            f, t, spectrum = SA.stft_pytorch(
                nperseg=nperseg,
                nfft=nfft,
                noverlap=noverlap,
                one_side=True,
                apply_padding=False,
            )

            if pow2_resize:
                spectrum, f, t = resize_to_pow2(spectrum, f, t)

            # --- 4) Calcul PSNR et SNR ---
            power = torch.abs(spectrum) ** 2
            max_val = power.max().item()
            signal_power = pulse_signal.pow(2).mean().item()

            psnr = (
                10 * np.log10(max_val / noise_power_per_cell)
                if noise_power_per_cell > 0
                else float("inf")
            )
            snr = (
                10 * np.log10((signal_power * fe) / NOISE_POWER)
                if NOISE_POWER > 0
                else float("inf")
            )

            # --- Sauvegarde PSNR/SNR ---
            if "psnr_db" not in p:
                p["psnr_db"] = {}
            p["psnr_db"][cfg_name] = psnr
            p["snr_db"] = snr

            # --- 5) Segmentation (-10 dB du max) ---
            power_db = 10 * torch.log10(power + 1e-30)
            threshold_db = power_db.max() - segmentation_db
            segmentation = (power_db >= threshold_db).float()

            # --- 6) Métadonnées ---
            emitter_label = p["emitter_label"]
            if emitter_label not in distinct_emitters:
                raise ValueError(f"Emitter label '{emitter_label}' non trouvé dans distinct_emitters : {distinct_emitters}")
            
            emitter_id = distinct_emitters.index(emitter_label)
            mod = po.modulation_name
            class_id = NAME_TO_CLASS_ID[mod]

            segmentations.append(
                {"class_id": class_id, "emitter_id": emitter_id, "mask": segmentation}
            )

        # --- Fusion par résolution ---
        if len(segmentations) > 0:
            seg_map = build_segmentation_per_resolution(
                segmentations,
                num_classes=len(NAME_TO_CLASS_ID),
                shape=segmentations[0]["mask"].shape,
            )
            segs_multi_res.append(seg_map)


    if len(segs_multi_res) == 0:
        raise ValueError("Aucune segmentation valide trouvée (segs_multi_res vide).")

    merged_seg = merge_segmentations_multi_res(segs_multi_res)
    return {"seg_per_res": segs_multi_res, "seg_merged": merged_seg}



def build_segmentation_per_resolution(
    segmentations: List[Dict],
    num_classes: int,
    shape: torch.Size,
    binary_segmentation_threshold: float = 0.1
):
    """
    Construit un masque d'indices :
      - 0 = fond (aucun signal)
      - chaque pixel signal contient un code unique :
        code = 1 + class_id + num_classes * emitter_id
    """
    mask = torch.zeros(shape, dtype=torch.int32)

    for seg in segmentations:
        class_id = seg["class_id"]
        emitter_id = seg["emitter_id"]
        code = 1 + class_id + num_classes * emitter_id  # Décalage de +1 pour réserver 0 au fond
        mask[seg["mask"] > binary_segmentation_threshold] = code

    return {"mask": mask}

def merge_segmentations_multi_res(segs_multi_res: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Fusionne plusieurs segmentations codées par pixel.
    Utilise un vote majoritaire entre résolutions.
    """
    if len(segs_multi_res) == 0:
        raise ValueError("Aucune segmentation valide à fusionner.")

    # Détermine la taille max
    max_h = max(s["mask"].shape[-2] for s in segs_multi_res)
    max_w = max(s["mask"].shape[-1] for s in segs_multi_res)

    resized_masks = []
    for s in segs_multi_res:
        mask = s["mask"].unsqueeze(0).unsqueeze(0).float()
        mask_resized = F.interpolate(mask, size=(max_h, max_w), mode="nearest")
        resized_masks.append(mask_resized.squeeze())

    stack = torch.stack(resized_masks, dim=0)  # (N_res, H, W)
    # Vote majoritaire
    mask_merged, _ = torch.mode(stack, dim=0)

    return {"mask": mask_merged}


def generate_and_store_spectrum_multi(
    scenarios: Sequence[List[Dict]],
    *,
    stft_cfgs: Optional[List[Dict[str, int]]] = None,
    base_path: str = "./dataset",
    seed: Optional[int] = None,
    split_train_test: bool = True,
    train_ratio: float = 0.8,
    acquisition_time=1e-3,
    pow2_resize=True,
    snr_max_base=20, 
    class_index_to_name = CLASS_INDEX_TO_NAME):
    """
    Génère un dataset multi-résolution avec PSNR/SNR,
    spectres normalisés et labels de détection/segmentation.
    Les 3 premiers scénarios sont sauvegardés en détail dans ./examples/
    """

    if stft_cfgs is None:
        stft_cfgs = [
            {"nperseg": 128, 'nfft': 128, "noverlap": 0, "fs": F_E},
            {"nperseg": 256, 'nfft': 256, "noverlap": 0, "fs": F_E},
            {"nperseg": 512, 'nfft': 512, "noverlap": 0, "fs": F_E},
            {"nperseg": 1024, 'nfft': 1024, "noverlap": 0, "fs": F_E},
        ]

    fe = stft_cfgs[0]["fs"]

    name_to_class_id = {v: k for k, v in class_index_to_name.items()}

    # --------------------
    # Seed
    # --------------------
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # --------------------
    # Split train/val
    # --------------------
    indices = np.arange(len(scenarios))
    if split_train_test:
        np.random.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        split_map = {"train": indices[:n_train], "val": indices[n_train:]}
    else:
        split_map = {"train": indices}

    split_dirs = {split: _init_split_dirs(base_path, split) for split in split_map}

    # --------------------
    # Génération des scénarios
    # --------------------
    for split_name, idxs in split_map.items():
        dirs = split_dirs[split_name]

        for idx_counter, idx_global in enumerate(tqdm(idxs, desc=f"{split_name} scenarios")):
            emitters = scenarios[idx_global]
            idx_scenario = uuid.uuid4().hex
            base_name = f"sc_{idx_scenario}"

            # --- Acquisition du signal ---
            noisy_signal, pulses = interceptor_signal_acquisition(
                {"x": 0.0, "y": 0.0}, emitters, fe, acquisition_time
            )
            if pulses is None or len(pulses) == 0:
                continue

            # --- Initialisation PSNR ---
            for pulse_data in pulses:
                pulse_data["psnr_db"] = {}

            # --- STFT bruité ---
            SA = SignalAnalysis(noisy_signal, fe)
            spectra_norm_list = []
            raw_spectra_list = []

            for cfg in stft_cfgs:
                cfg_name = f"cfg{cfg['nperseg']}"
                nfft = cfg["nfft"]
                nperseg = cfg["nperseg"]
                noverlap = cfg.get("noverlap", nperseg // 2)

                f, t, spectrum = SA.stft_pytorch(
                    nperseg=nperseg,
                    nfft=nfft,
                    noverlap=noverlap,
                    one_side=True,
                    apply_padding=False,
                )

                if pow2_resize:
                    spectrum, f, t = resize_to_pow2(spectrum, f, t)

                raw_spectra_list.append(torch.abs(spectrum))

                psnr_max = snr_max_base + 10 * math.log10(nfft / 2)
                noise_power_per_cell = NOISE_POWER / fe * (nperseg / nfft**2)
                spec_norm = normalize_spectrum(torch.abs(spectrum), PSNR_MIN, psnr_max, noise_power_per_cell=noise_power_per_cell)
                spectra_norm_list.append(spec_norm)

            # --- Segmentation propre alignée ---

            results = compute_psnr_and_segmentation_per_pulse(
                pulses=pulses,
                fe=fe,
                stft_cfgs=stft_cfgs,
                acquisition_time=acquisition_time,
                pow2_resize=pow2_resize,
                distinct_emitters = [em['label'] for em in emitters]
            )

            seg_final = results["seg_merged"]
            seg_mask = seg_final["mask"].to(torch.int16)
            seg_path = Path(dirs["labels_segment"]) / f"{base_name}.pt"
            torch.save(seg_mask, seg_path)

            # --- Sauvegardes principales ---
            np.save(os.path.join(dirs["emitters"], base_name + ".npy"), emitters)
            np.save(os.path.join(dirs["pulses"], base_name + ".npy"), pulses)
            torch.save(spectra_norm_list, os.path.join(dirs["data"], base_name + ".pt"))

            label_json = Path(dirs["labels_detect"]) / f"{base_name}.json"
            rebuild_labels_from_pulsesfile_json_like(pulses, label_json, name_to_class_id = name_to_class_id)

            # --- EXAMPLES pour les 3 premiers scénarios ---
            if idx_counter < 3:
                example_dir = Path(base_path) / "examples" / base_name
                example_dir.mkdir(parents=True, exist_ok=True)

                # Sauvegarde des spectres bruts et normalisés
                for cfg, raw_spec, norm_spec, seg_map in zip(
                    stft_cfgs,
                    raw_spectra_list,
                    spectra_norm_list,
                    results["seg_per_res"],
                ):
                    cfg_name = f"cfg{cfg['nperseg']}"
                    torch.save(raw_spec, example_dir / f"raw_spectrum_{cfg_name}.pt")
                    torch.save(norm_spec, example_dir / f"norm_spectrum_{cfg_name}.pt")
                    torch.save(seg_map, example_dir / f"seg_{cfg_name}.pt")

                # Segmentation post-fusion
                torch.save(seg_final, example_dir / "seg_merged.pt")

                # Labels des boîtes
                shutil.copy(label_json, example_dir / "boxes.json")

    print("✅ Multi-résolution dataset generation complete.")

