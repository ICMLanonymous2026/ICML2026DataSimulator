import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft, fftshift
from scipy.signal import spectrogram, stft, welch, cwt, morlet
import torch
from tftb.processing import WignerVilleDistribution

SAMPLING_FREQUENCY = 4e9

class SignalAnalysis:
    """
    Class for performing various radar signal analysis operations.
    """

    def __init__(self, signal, fe=None, time=None):
        """
        Initialize the RadarAnalysis object with the input signal, sampling frequency, and time array (optional).
        Args:
            signal: numpy.ndarray
                The input radar signal.
            fe: float, optional
                The sampling frequency of the signal. If not provided, it will be determined from the time array.
            time: numpy.ndarray, optional
                The time array corresponding to the signal. If not provided, it will be generated based on the signal length.
        """
        self.signal = signal

        # Check that the sampling frequency exists, otherwise compute according to the time sample
        if fe is not None:
            self.fe = fe
        else:
            if time is not None:
                self.fe = 1 / np.mean(np.diff(time))
            else:
                self.fe = SAMPLING_FREQUENCY

        if time is not None:
            self.time = time
        else:
            self.time = np.arange(len(signal)) * (1 / self.fe)

    def plot_time_domain(self):
        """
        Plot the radar signal in the time domain.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.time, self.signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Radar Signal in Time Domain')
        plt.show(block=False)

    def plot_fft(self):
        """
        Plot the radar signal in the frequency domain.
        """
        spectrum = fftshift(fft(self.signal))
        frequencies = np.linspace(-self.fe / 2, self.fe / 2, len(spectrum))

        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, np.abs(spectrum))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Radar Signal in Frequency Domain')
        plt.show(block=False)

    def plot_spectrogram(self, nperseg=64):
        """
        Plot the spectrogram of the radar signal.
        Args:
            nperseg: int, optional
                Number of samples per segment in the spectrogram.
        """
        frequencies, times, Sxx = spectrogram(self.signal, self.fe, nperseg=nperseg)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))
        plt.colorbar(label='Power Spectral Density (dB)')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Radar Signal Spectrogram')
        plt.show(block=False)

    def plot_autocorrelation(self):
        """
        Plot the autocorrelation function of the radar signal.
        """
        T = 1 / self.fe
        autocorr = np.correlate(self.signal, self.signal, mode='full')
        time_lags = np.linspace(-T * (len(self.signal) - 1), T * (len(self.signal) - 1), len(autocorr))

        plt.figure(figsize=(10, 4))
        plt.plot(time_lags, autocorr)
        plt.xlabel('Time Lag (s)')
        plt.ylabel('Autocorrelation')
        plt.title('Radar Signal Autocorrelation Function')
        plt.show(block=False)

    def plot_power_spectrum(self, window='hann', nfft=1024):
        """
                Plot the power spectrum of the radar signal.

                Args:
                    window (str): The window function to use.
                    nfft (int): The number of FFT points.
        """
        frequencies, power_spectrum = welch(self.signal, self.fe, window=window, nfft=nfft)

        plt.plot(frequencies, 10 * np.log10(power_spectrum))
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density (dB)')
        plt.title('Radar Signal Power Spectrum')
        plt.show(block=False)

    def plot_stft(self, window='hann', nperseg=64, noverlap=None, show=True):
        """
        Plot the short-time Fourier transform (STFT) of the radar signal.
        Args:
            window: str or tuple or array_like, optional
                Desired window to use. Defaults to 'hann'.
            nperseg: int, optional
                Number of samples per segment in the STFT.
            noverlap: int, optional
                Number of samples to overlap between segments in the STFT.
        """
        frequencies, time_steps, stft_data = stft(self.signal, self.fe, window=window, nperseg=nperseg,
                                                  noverlap=noverlap)

        if show:
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(time_steps, frequencies, np.abs(stft_data))
            plt.colorbar(label='Magnitude')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title('Radar Signal Short-Time Fourier Transform (STFT)')
            plt.show()

        return frequencies, time_steps, stft_data
    
    def stft_pytorch(self, nperseg, noverlap=0, nfft=None, window_array=None, device="cpu", one_side=False, scaling=None, apply_padding=True):
        """
        Returns the STFT of a signal, along with frequency and time vectors, with an option to return one-sided spectrum.
        
        Parameters:
            sig (torch.Tensor): The input signal.
            fs (int): The sampling frequency of the signal.
            nperseg (int): Length of each segment.
            noverlap (int): Number of points to overlap between segments.
            nfft (int, optional): Number of FFT points. Defaults to nperseg.
            window_array (torch.Tensor, optional): The window to be applied to each segment.
            device (str, optional): The device to perform calculations on. Defaults to "cpu".
            one_side (bool, optional): If True, return only the positive frequencies of the spectrum. Defaults to True.
            scaling (str,optional): 'psd' or 'spectrum', for 'psd', scaling_factor = fe*nperseg*max(nperseg,nfft) (*2 if one_side)
        Returns:
            f (torch.Tensor): The frequency vector.
            t (torch.Tensor): The time vector.
            sxx (torch.Tensor): The STFT of the signal.
        """
        if not isinstance(self.signal, torch.Tensor):
            dtype = torch.complex64 if np.iscomplexobj(self.signal) else torch.float32
            sig   = torch.tensor(self.signal, dtype=dtype, device=device)
        else: 
            sig = self.signal

        if torch.is_complex(sig):
            one_side = False

        if nfft is None: 
            nfft = nperseg

        if apply_padding:
            # Calculate the total length needed to make signal length a multiple of nperseg
            num_segments = (len(sig) - noverlap) // (nperseg - noverlap) + 1
            pad_total = nperseg-noverlap - len(sig)%(nperseg-noverlap)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            sig = torch.nn.functional.pad(sig, (pad_before, pad_after), mode='constant', value=0)
        else:
            # Calculate the number of segments that can be created without padding
            num_segments = (len(sig) - noverlap) // (nperseg - noverlap)
            total_length_used = num_segments * nperseg + (num_segments - 1) * noverlap
            sig = sig[:total_length_used]

        # Slicing
        sig = sig.unfold(0, nperseg, nperseg - noverlap)

        # Windowing
        if window_array is not None:
            sig = sig * window_array[None, :]

        # zero padding for FFT if nfft is greater than the segment length
        if nfft > nperseg:
            pad_left = (nfft - nperseg) // 2
            pad_right = nfft - nperseg - pad_left
            zeros_left = torch.zeros(sig.size(0), pad_left, device=device)
            zeros_right = torch.zeros(sig.size(0), pad_right, device=device)
            sig = torch.cat((zeros_left, sig, zeros_right), dim=1)
        
        # Vérification de la variance
        # print(torch.std(sig)/np.sqrt(nperseg/nfft)/2)

        if one_side:
            # FFT spectrum
            sxx = torch.fft.rfft(sig, n=nfft, dim=1, norm='forward')
            f = torch.fft.rfftfreq(nfft, 1 / self.fe).to(device)
            # Vérification de la conservation de l'énergie
            # e_col = []
            # for i in range(sxx.shape[1]):
            #     e_col.append(torch.sum(torch.abs(sxx[:,i])**2)*nfft)
            # e_col = np.sum(e_col)
            # print('E col = ',e_col)

            # Vérification de la variance
            # print(torch.std(sxx)/np.sqrt(nperseg/max(nfft,nperseg))/2*np.sqrt(nfft))
            
            if scaling == 'spectrum':
                sxx *= np.sqrt(2)
        else:
            # FFT spectrum
            sxx = torch.fft.fft(sig, n=nfft, dim=1, norm='forward')
            f   = torch.fft.fftshift(torch.fft.fftfreq(nfft, 1/self.fe)).to(device)
            sxx = torch.fft.fftshift(sxx, dim=1)

            # Vérification de la variance
            # print('torch.std(sig),torch.std(sxx)',torch.std(sig),torch.std(sxx)*np.sqrt(nfft))

            # Vérification de la conservation de l'énergie
            # e_col = []
            # for i in range(sxx.shape[1]):
            #     e_col.append(torch.sum(torch.abs(sxx[:,i])**2)*nfft)
            # e_col = np.sum(e_col)
            # print('E col = ',e_col)

        # Apply scaling for PSD
        if scaling == 'psd':

            if window_array is not None:
                win_power = window_array.pow(2).sum()  # Power of the window
            else:
                win_power = nperseg
            sxx /= np.sqrt(self.fe * win_power)

            # Vérification de la variance
            # print(torch.std(sxx)/np.sqrt(nperseg/max(nfft,nperseg))/2*np.sqrt(nfft*self.fe * win_power))

            # Vérification de la conservation de l'énergie
            # e_col = []
            # for i in range(sxx.shape[1]):
            #     e_col.append(torch.sum(torch.abs(sxx[:,i])**2)*nfft*self.fe * win_power)
            # e_col = np.sum(e_col)
            # print('E col = ',e_col)


        # Time vector
        t = torch.arange(sig.size(0)) * (nperseg - noverlap) / self.fe

        return f, t, sxx.T
    
    
    def wigner_ville_transform_torch(self, nfft=None, device="cpu", winlength=None, window=None):
        """
        Calcule la transformation de Wigner-Ville d'un signal en utilisant PyTorch avec une fenêtre optionnelle.

        Parameters:
            nfft (int, optional): Nombre de points FFT. Par défaut, la longueur du signal.
            device (str, optional): Dispositif de calcul ("cpu" ou "cuda").
            winlength (int, optional): Taille de la fenêtre pour limiter les calculs.
            window (torch.Tensor, optional): Fenêtre de pondération (fwindow). Si None, aucune fenêtre n'est utilisée.
            
        Returns:
            f (torch.Tensor): Vecteur des fréquences positives.
            t (torch.Tensor): Vecteur des temps.
            wvd (torch.Tensor): Matrice de la transformation de Wigner-Ville (fréquences positives).
        """
        if not isinstance(self.signal, torch.Tensor):
            signal = torch.tensor(self.signal, dtype=torch.complex64, device=device)
        else:
            signal = self.signal.to(device, dtype=torch.complex64)

        n = len(signal)
        if nfft is None:
            nfft = n  # Taille par défaut si non spécifié
        if winlength is None:
            winlength = n // 2 - 1  # Taille par défaut de la fenêtre 

        # Préparation de la fenêtre
        if window is not None:
            window = torch.tensor(window, dtype=torch.float32, device=device)
            lh = (window.shape[0] - 1) // 2
        else:
            lh = n // 2

        # Initialisation de la matrice WVD
        wvd = torch.zeros((nfft, n), dtype=torch.complex64, device=device)

        # Signal conjugué
        conj_signal = signal.conj()

        # Calcul Wigner-Ville
        for icol in range(n):
            # Calcul du maximum du décalage tau en fonction des contraintes
            taumax = min(icol, n - icol - 1, lh)
            tau = torch.arange(-taumax, taumax + 1, device=device, dtype=torch.long)

            # Produit avec fenêtre (si définie)
            if window is not None:
                weighted_product = window[lh + tau] * signal[icol + tau] * conj_signal[icol - tau]
            else:
                weighted_product = signal[icol + tau] * conj_signal[icol - tau]

            indices = (nfft + tau) % nfft
            wvd[indices, icol] = weighted_product

            # Cas spécial pour la limite tausec (si la fenêtre est définie)
            if window is not None and lh >= winlength // 2:
                special_tau = winlength // 2
                if special_tau <= icol < n - special_tau:
                    wvd[special_tau, icol] = (
                        window[lh + special_tau] * signal[icol + special_tau] * conj_signal[icol - special_tau]
                        + window[lh - special_tau] * signal[icol - special_tau] * conj_signal[icol + special_tau]
                    )
                    wvd[special_tau, icol] *= 0.5

        # Transformée de Fourier pour chaque colonne
        wvd = torch.fft.fft(wvd, dim=0)
        wvd = torch.real(wvd)

        # Garder uniquement les fréquences positives
        positive_freqs = slice(0, nfft // 2 + 1)  # Indices des fréquences positives
        wvd = wvd[positive_freqs, :]  # Filtrer la matrice pour ne garder que les fréquences positives

        # Générer les fréquences positives normalisées
        freqs = torch.arange(nfft // 2 + 1, device=device, dtype=torch.float32) * (self.fe / 2 / nfft)
        time = torch.arange(0, n, device=device, dtype=torch.float32) / self.fe

        return freqs, time, wvd


    
    def choi_williams_transform_torch(self, nfft=None, device="cpu", winlength=None, sigma=1.0):
        """
        Calcule la transformation de Choi-Williams (CWD) d'un signal en utilisant PyTorch.

        Parameters:
            nfft (int, optional): Nombre de points FFT. Par défaut, la longueur du signal.
            device (str, optional): Dispositif de calcul ("cpu" ou "cuda").
            winlength (int, optional): Taille de la fenêtre pour limiter les calculs.
            sigma (float, optional): Facteur de lissage pour le noyau exponentiel. Par défaut : 1.0.
            
        Returns:
            f (torch.Tensor): Vecteur des fréquences positives.
            t (torch.Tensor): Vecteur des temps.
            cwd (torch.Tensor): Matrice de la transformation de Choi-Williams (fréquences positives).
        """
        if not isinstance(self.signal, torch.Tensor):
            signal = torch.tensor(self.signal, dtype=torch.complex64, device=device)
        else:
            signal = self.signal.to(device, dtype=torch.complex64)

        n = len(signal)
        if nfft is None:
            nfft = n  # Taille par défaut si non spécifié
        if winlength is None:
            winlength = n // 10  # Taille par défaut de la fenêtre (10% de la longueur du signal)

        tausec = winlength // 2  # Décalage maximal pour tau

        # Calcul vectorisé des tailles de fenêtre
        indices = torch.arange(n, device=device)
        taulens = torch.min(
            torch.stack([
                indices,                      # Indices disponibles avant chaque point temporel
                n - indices - 1,              # Indices disponibles après chaque point temporel
                tausec * torch.ones(n, device=device)  # Limite imposée par tausec
            ]),
            dim=0
        )[0]

        # Initialisation de la matrice CWD
        cwd = torch.zeros((nfft, n), dtype=torch.complex64, device=device)

        # Signal conjugué
        conj_signal = signal.conj()

        # Calcul Choi-Williams
        for t_idx in range(n):
            taumax = taulens[t_idx].item()
            tau = torch.arange(-taumax, taumax + 1, device=device, dtype=torch.long)  # Convertir tau en long

            for u in range(max(0, t_idx - tausec), min(n, t_idx + tausec + 1)):
                # Calcul du noyau exponentiel
                kernel = torch.exp(-sigma * ((u - t_idx)**2) / (4.0 * tau.abs().float() + 1e-10))

                # Produit signal avancé/retardé
                x_tau = signal[(u + tau).long()]
                x_minus_tau = conj_signal[(u - tau).long()]

                # Application du noyau
                weighted_product = kernel * x_tau * x_minus_tau
                indices = (nfft + tau) % nfft
                cwd[indices, t_idx] += torch.sum(weighted_product)

        # Transformée de Fourier pour chaque colonne
        cwd = torch.fft.fft(cwd, dim=0)
        cwd = torch.real(cwd)

        # Garder uniquement les fréquences positives
        positive_freqs = slice(0, nfft // 2 + 1)  # Indices des fréquences positives
        cwd = cwd[positive_freqs, :]  # Filtrer la matrice pour ne garder que les fréquences positives

        # Générer les fréquences positives normalisées
        freqs = torch.arange(nfft // 2 + 1, device=device, dtype=torch.float32) * (self.fe / 2 / nfft)
        time = torch.arange(0, n, device=device, dtype=torch.float32) / self.fe

        return freqs, time, cwd
    

    def cyclostationary_transform_torch(self, nfft=None, device="cpu", window=None):
        """
        Calcule la densité spectrale cyclostationnaire (SCD) d'un signal basé sur : "Estimation of Modulation Parameters of LPI Radar Using
        Cyclostationary Method
        Raja Kumari Chilukuri1,2  · Hari Kishore Kakarla1
        · K. Subbarao3"

        Parameters:
            signal (torch.Tensor): Signal d'entrée (1D).
            fs (float): Fréquence d'échantillonnage.
            nfft (int, optional): Nombre de points FFT. Par défaut, la longueur du signal.
            device (str, optional): Dispositif de calcul ("cpu" ou "cuda").
            window (torch.Tensor, optional): Fenêtre de pondération. Si None, une fenêtre de Hamming est utilisée.

        Returns:
            alpha (torch.Tensor): Fréquences cycliques.
            f (torch.Tensor): Fréquences normales.
            scd (torch.Tensor): Matrice de la densité spectrale cyclostationnaire.
        """
        fs = self.fe
        if not isinstance(self.signal, torch.Tensor):
            signal = torch.tensor(self.signal, dtype=torch.complex64, device=device)
        else:
            signal = self.signal.to(device, dtype=torch.complex64)

        n = len(signal)
        if nfft is None:
            nfft = n  # Taille par défaut si non spécifié

        # Préparation de la fenêtre
        if window is None:
            window = torch.hamming_window(n, periodic=True, device=device, dtype=torch.float32)
        else:
            window = torch.tensor(window, dtype=torch.float32, device=device)

        # Appliquer la fenêtre au signal
        weighted_signal = signal * window

        # Calcul FFT du signal pondéré
        fft_signal = torch.fft.fft(weighted_signal, n=nfft)

        # Initialisation des variables
        scd = torch.zeros((nfft, nfft), dtype=torch.complex64, device=device)

        # Fréquences cycliques (alpha) et normales (f)
        alpha = torch.fft.fftfreq(nfft, d=1 / fs).to(device)
        f = alpha.clone()

        # Calcul des coefficients SCD
        for alpha_idx, alpha_val in enumerate(alpha):
            for k in range(nfft):
                # Indices décalés pour k + α/2 et k - α/2
                f_plus = (k + int(alpha_val * nfft / fs / 2)) % nfft
                f_minus = (k - int(alpha_val * nfft / fs / 2)) % nfft

                # Produit des termes décalés
                scd[alpha_idx, k] = fft_signal[f_plus] * fft_signal[f_minus].conj()

        return alpha, f, scd


    def wigner_ville_transform_tftb(self):
        """
        Calcule la transformation de Wigner-Ville d'un signal à l'aide de tftb.

        Returns:
            f (np.ndarray): Vecteur des fréquences.
            t (np.ndarray): Vecteur des temps.
            wvd (np.ndarray): Matrice de la transformation de Wigner-Ville.
        """
        from tftb.processing import WignerVilleDistribution

        # Vérifiez si le signal est sous forme de numpy array
        if not isinstance(self.signal, np.ndarray):
            signal = self.signal.numpy()  # Convertir de torch à numpy si nécessaire
        else:
            signal = self.signal

        # Calcul de la Wigner-Ville avec tftb
        wvd = WignerVilleDistribution(signal)
        wvd_output = wvd.run()

        # Déballer les résultats
        wvd_matrix, _, time_vector = wvd_output  # Ignorer les indices temporels inutiles

        # Générer l'axe des fréquences
        f = np.linspace(-self.fe / 2, self.fe / 2, wvd_matrix.shape[0])
        t = time_vector  # Utiliser directement le vecteur de temps retourné par `run`

        return f, t, wvd_matrix


    def plot_cwt(self, scales):
        """
        Plot the continuous wavelet transform (CWT) of the radar signal.
        Args:
            scales: array_like
                The scales (or widths) of the wavelet function.
        """
        wavelet_transform = cwt(self.signal, morlet, scales)

        plt.imshow(np.abs(wavelet_transform), aspect='auto', extent=[0, len(self.signal), scales[-1], scales[0]])
        plt.colorbar(label='Wavelet Amplitude')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.title('Radar Signal Continuous Wavelet Transform (CWT)')
        plt.show(block=False)

    def plot_all(self):
        """
        Plot all the analysis plots in a single page.
        """
        self.plot_time_domain()

        self.plot_fft()

        self.plot_spectrogram()

        self.plot_autocorrelation()

        self.plot_power_spectrum()

        self.plot_stft()

        plt.show()

    @staticmethod
    def return_plt_image():
        # Create a buffer to hold the image data
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Close the plot to free up resources
        plt.close()

        # Read the image from the buffer
        image = Image.open(buffer)

        return image
