import numpy as np

def generate_ofdm(fc, B, T, Fs, N_sub=64, cp_ratio=0.1):

    N = int(T * Fs)
    t = np.arange(N) / Fs

    delta_f = B / N_sub
    Ts = 1 / delta_f
    Ns = int(Ts * Fs)
    Ncp = int(cp_ratio * Ns)

    sym_len = Ns + Ncp
    n_symbols = max(1, N // sym_len)

    xc = np.zeros(n_symbols * sym_len, dtype=complex)

    # Time axes for interpolation
    t_small = np.linspace(0, 1, N_sub)
    t_big   = np.linspace(0, 1, Ns)

    for k in range(n_symbols):

        bits = np.random.randint(0, 2, (N_sub, 2))
        qpsk = (2 * bits[:,0] - 1) + 1j*(2 * bits[:,1] - 1)

        X = np.fft.ifftshift(qpsk)
        x_raw = np.fft.ifft(X)

        # Interpolation to match sample rate
        x_interp_real = np.interp(t_big, t_small, x_raw.real)
        x_interp_imag = np.interp(t_big, t_small, x_raw.imag)
        x_interp = x_interp_real + 1j * x_interp_imag

        x_interp /= np.sqrt(np.mean(np.abs(x_interp)**2))

        x_cp = np.concatenate([x_interp[-Ncp:], x_interp])

        start = k * sym_len
        xc[start:start+sym_len] = x_cp

    # Trim xc and t to match (OFDM block length)
    xc = xc[:n_symbols * sym_len]
    t  = t[:len(xc)]

    # Passband modulation
    x = np.real(xc * np.exp(1j * 2 * np.pi * fc * t))

    return t, x, xc
