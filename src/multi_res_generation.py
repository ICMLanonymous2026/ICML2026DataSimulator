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
from src.screen_services.tools_4_data_generation import compute_spectrum_psnr_per_pulse

from .SignalAnalysis import SignalAnalysis
from .scenario import interceptor_signal_acquisition


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

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _compute_required_snr_db(
    pw: float,
    bw: float,
    target_psnr_db: float,
    stft_cfg: dict,
) -> float:
    """Calcule le SNR (en dB) nécessaire pour obtenir *au minimum*
    `target_psnr_db` sur la STFT décrite par ``stft_cfg``.
    
    Paramètres
    ----------
    pw : float
        Pulse width (s).
    bw : float
        Bandwidth (Hz).
    target_psnr_db : float
        PSNR désiré dans le domaine temps–fréquence (dB).
    stft_cfg : dict
        Dictionnaire « nperseg / noverlap / fs ». Si ``noverlap`` est absent,
        on prend ``nperseg // 2``.
    """
    nperseg = stft_cfg["nperseg"]
    noverlap = stft_cfg["noverlap"]
    nfft = stft_cfg["nfft"]

    # Heuristique PSNR ≈ SNR - 10log(Nt) - 10log(Nf)
    return float(target_psnr_db - 10 * np.log10(nperseg - noverlap) - 10 * np.log10(nfft))


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


def _init_split_dirs(base: str, split: str, cfgs: list[dict]) -> dict:
    """Crée l'arborescence et retourne un dict de chemins."""
    root_split = _mkdir(os.path.join(base, split))
    paths = {
        "emitters": _mkdir(os.path.join(root_split, "emitters")),
        "pulses": _mkdir(os.path.join(root_split, "pulses")),
        "cfg": {},  # cfg_name -> {images, labels}
    }
    for cfg in cfgs:
        cfg_name = f"cfg{cfg['nperseg']}"
        cfg_root = _mkdir(os.path.join(root_split, cfg_name))
        paths["cfg"][cfg_name] = {
            "images": _mkdir(os.path.join(cfg_root, "images")),
            "labels": _mkdir(os.path.join(cfg_root, "labels")),
            "data": _mkdir(os.path.join(cfg_root, "data")),
        }
    return paths

def downsample_complex_to_pow2(spec: torch.Tensor,
                                     f: torch.Tensor,
                                     t: torch.Tensor):
    """
    Down-sample un spectrogramme complexe `(F,T)` → `(F2,T2)`
    avec F2 = 2^⌊log2 F⌋ et T2 = 2^⌊log2 T⌋.

    Paramètres
    ----------
    spec : (F,T) complex • GPU ou CPU
    f    : (F,) real     • mêmes device / dtype (réel) que `spec`
    t    : (T,) real     • idem

    Retourne
    --------
    spec_ds : (F2,T2) complex
    f_ds    : (F2,)    real
    t_ds    : (T2,)    real
    """
    device = spec.device
    real_dtype = torch.float32 if spec.dtype in (torch.complex64, torch.float32) else torch.float64

    F_src, T_src = spec.shape
    F_tgt = 1 << (F_src.bit_length() - 1)
    T_tgt = 1 << (T_src.bit_length() - 1)

    # déjà 2^k × 2^k
    if (F_src, T_src) == (F_tgt, T_tgt):
        return spec, f, t

    # (Re, Im) → (1,2,F,T) puis interpolation « area »
    ri = torch.view_as_real(spec).permute(2, 0, 1).unsqueeze(0).float()  # (1,2,F,T)
    ri_ds = F.interpolate(ri, size=(F_tgt, T_tgt), mode="area")          # (1,2,F2,T2)
    re_ds, im_ds = ri_ds[0, 0], ri_ds[0, 1]
    spec_ds = torch.complex(re_ds, im_ds)                                # (F2,T2)

    # grilles f / t linéaires sous PyTorch
    f0, f1 = f[0].item(), f[-1].item()
    t0, t1 = t[0].item(), t[-1].item()

    f_ds = torch.linspace(f0, f1, F_tgt, dtype=real_dtype, device=device)
    t_ds = torch.linspace(t0, t1, T_tgt, dtype=real_dtype, device=device)

    return spec_ds, f_ds, t_ds
    

def generate_and_store_spectrum_multi(
    scenarios: Sequence[List[Dict]],
    *,
    stft_cfgs: Optional[List[Dict[str, int]]] = None,
    base_path: str = "./dataset",
    seed: Optional[int] = None,
    split_train_test: bool = True,
    train_ratio: float = 0.8,
    acquisition_time=1e-3,
    pow2_downsample=False,
):
    """Génère un dataset multi‑résolution avec PSNR par configuration et SNR injecté dans chaque pulse"""

    if stft_cfgs is None:
        stft_cfgs = [
            {"nperseg": 128, 'nfft': 128, "noverlap": 0, "fs": F_E},
            {"nperseg": 256, 'nfft': 256, "noverlap": 0, "fs": F_E},
            {"nperseg": 512, 'nfft': 512, "noverlap": 0, "fs": F_E},
            {"nperseg": 1024, 'nfft': 1024, "noverlap": 0, "fs": F_E},
        ]

    fe = stft_cfgs[0]["fs"]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    indices = np.arange(len(scenarios))
    if split_train_test:
        np.random.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        split_map = {"train": indices[:n_train], "val": indices[n_train:]}
    else:
        split_map = {"full": indices}

    split_dirs = {s: _init_split_dirs(base_path, s, stft_cfgs) for s in split_map}

    for split_name, idxs in split_map.items():
        dirs = split_dirs[split_name]

        for idx_global in tqdm(idxs, desc=f"{split_name}  scenarios"):
            emitters = scenarios[idx_global]
            idx_scenario = uuid.uuid4().hex
            base_name = f"sc_{idx_scenario}"

            noisy_signal, pulses = interceptor_signal_acquisition(
                {"x": 0.0, "y": 0.0}, emitters, fe, acquisition_time
            )

            if pulses is None or len(pulses) == 0:
                continue

            for pulse_data in pulses:
                pulse_data['psnr_db'] = {}

            SA = SignalAnalysis(noisy_signal, fe)

            for cfg in stft_cfgs:
                cfg_name = f"cfg{cfg['nperseg']}"
                nperseg = cfg["nperseg"]
                nfft = cfg["nfft"]
                noverlap = cfg.get("noverlap", nperseg // 2)

                f, t, spectrum = SA.stft_pytorch(
                    nperseg=nperseg,
                    nfft=nfft,
                    noverlap=noverlap,
                    one_side=True,
                    apply_padding=False,
                )

                if pow2_downsample:
                    spectrum, f, t = downsample_complex_to_pow2(spectrum, f, t)

                torch.save(
                    spectrum,
                    os.path.join(dirs["cfg"][cfg_name]["data"], base_name + ".pt")
                )

                # PSNRs et injection dans pulses
                psnrs = compute_spectrum_psnr_per_pulse(
                    pulses, fe=fe, nfft=nfft, nperseg=nperseg, noverlap=noverlap
                )
                for pulse_data, psnr in zip(pulses, psnrs):
                    if psnr is not None:
                        pulse_data['psnr_db'][cfg_name] = psnr['psnr']
                    pulse_data['snr_db'] = psnr['snr']

            # Save emitters & pulses enrichis
            np.save(os.path.join(dirs["emitters"], base_name + ".npy"), emitters)
            np.save(os.path.join(dirs["pulses"], base_name + ".npy"), pulses)

    print("✅ Multi‑résolution dataset generation complete (PSNR & SNR injectés, labels non enregistrés)")




def generate_noise_spectra(
    *,
    record_length: float = 1e-3,       # seconds
    fe: float = 4.0e9,                 # sample rate [Hz]
    stft_cfgs: Optional[List[Dict[str, int]]] = None,
    seed: Optional[int] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    spectra[rec_idx][nperseg] → complex64 NumPy array (freq × time)
    """
    # ---------- default STFT resolutions -------------------------------
    if stft_cfgs is None:
        stft_cfgs = [
            {"nperseg": 128, "nfft": 128, "noverlap": 0},
            {"nperseg": 256, "nfft": 256, "noverlap": 0},
            {"nperseg": 512, "nfft": 512, "noverlap": 0},
            {"nperseg": 1024, "nfft": 1024, "noverlap": 0},
        ]

    # ---------- reproducibility ----------------------------------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ---------- thermal noise variance per *real* sample ---------------
    k_B   = 1.38064852e-23        # J·K⁻¹
    T_sys = 290.0                   # K
    B     = fe / 2.0                # Hz
    sigma2 = k_B * T_sys * B / fe   # V²

    n_samples = int(record_length * fe)
    x = torch.randn(n_samples) * np.sqrt(sigma2)

    # 2. STFTs through your own wrapper
    SA = SignalAnalysis(x.numpy(), fe)       # assumes SignalAnalysis is in scope
    spectra = {}
    for cfg in stft_cfgs:
        nperseg  = cfg["nperseg"]
        nfft     = cfg["nfft"]
        noverlap = cfg["noverlap"]

        f, t, S = SA.stft_pytorch(
            nperseg=nperseg,
            nfft=nfft,
            noverlap=noverlap,
            one_side=True,
            apply_padding=False,
        )
        spectra[nperseg] = S

    return spectra
