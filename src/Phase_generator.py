import random
import numpy as np

class PhaseGenerator:
    """
    Public-safe phase (and symbol) generator.

    Supported public types:
    - 'barker_biphasique' : deterministic BPSK Barker sequences (phases in {0, π})
    - 'random_biphasique' : random BPSK sequence (phases in {0, π})
    - 'QPSK', 'QAM16', 'QAM64', 'QAM' : random complex QAM symbols

    All other phase types are intended to be unavailable in the public release.
    """

    def __init__(self, phase_type: str = 'barker_biphasique',
                 momentsNumber: int = 10,
                 qam_order: int | None = None) -> None:
        """
        Parameters
        ----------
        phase_type : str
            Type of code or modulation ('barker_biphasique', 'random_biphasique',
            'QPSK', 'QAM16', 'QAM64', or 'QAM').
        momentsNumber : int
            Length of the sequence (number of chips / symbols).
        qam_order : int or None
            Constellation order M for 'QAM'. If phase_type is 'QPSK', 'QAM16',
            or 'QAM64', this argument is optional and inferred from the name.
        """
        self.phase_type = phase_type
        self.momentsNumber = momentsNumber
        self.qam_order = qam_order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_phase_list(self):
        """
        Returns either:
        - a list of phases in radians (for BPSK Barker / random_biphasique), or
        - a NumPy array of complex QAM symbols (for QPSK/QAM).

        This unified interface is kept for backward compatibility.
        """
        if self.phase_type == 'barker_biphasique':
            return self.barker_biphasique()

        if self.phase_type == 'random_biphasique':
            return self.random_biphasique()

        if self.phase_type in {'QPSK', 'QAM16', 'QAM64', 'QAM'}:
            return self.qam_symbols()

        # All other forms are unavailable in the public version
        raise NotImplementedError(
            f"Phase type '{self.phase_type}' is not available in the public release."
        )

    # ------------------------------------------------------------------
    # Supported public BPSK codes
    # ------------------------------------------------------------------
    def barker_biphasique(self):
        """
        Returns the classical bipolar Barker sequences in BPSK form
        (phases in {0, π}).
        """
        barker_codes = {
            2:  [np.pi, 0],
            3:  [np.pi, np.pi, 0],
            4:  [np.pi, np.pi, 0, np.pi],
            5:  [np.pi, np.pi, np.pi, 0, np.pi],
            7:  [np.pi, np.pi, np.pi, 0, 0, np.pi, 0],
            11: [np.pi, np.pi, np.pi, 0, 0, 0, np.pi, 0, 0, np.pi, 0],
            13: [np.pi, np.pi, np.pi, np.pi, np.pi, 0, 0,
                 np.pi, np.pi, 0, np.pi, 0, np.pi],
        }

        if self.momentsNumber not in barker_codes:
            raise ValueError(
                f"Barker code of length {self.momentsNumber} is not supported."
            )

        return barker_codes[self.momentsNumber]

    def random_biphasique(self):
        """
        Returns a random BPSK sequence with phases in {0, π}.
        """
        return [random.choice((0, np.pi)) for _ in range(self.momentsNumber)]

    # ------------------------------------------------------------------
    # QAM symbol generation
    # ------------------------------------------------------------------
    def qam_symbols(self) -> np.ndarray:
        """
        Generates a random sequence of complex QAM symbols.

        For phase_type:
        - 'QPSK'  → M = 4
        - 'QAM16' → M = 16
        - 'QAM64' → M = 64
        - 'QAM'   → M = self.qam_order (must be provided)

        The constellation is assumed square (M = m^2), built on
        levels ±(1, 3, ..., m-1), and normalized to unit average power.
        """
        M = self._infer_qam_order()
        m = int(np.sqrt(M))

        if m * m != M:
            raise ValueError(
                f"QAM order M={M} is not a perfect square (required for square QAM)."
            )

        # Square grid: levels = {-m+1, ..., -1, 1, ..., m-1}
        levels = np.arange(-(m - 1), m, 2, dtype=float)
        xv, yv = np.meshgrid(levels, levels)
        constellation = xv.flatten() + 1j * yv.flatten()

        # Normalize to unit average power
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))

        # Draw a random symbol sequence of length momentsNumber
        indices = np.random.randint(0, M, size=self.momentsNumber)
        return constellation[indices]

    def _infer_qam_order(self) -> int:
        """
        Infer QAM order from phase_type or qam_order.
        """
        if self.phase_type == 'QPSK':
            return 4
        if self.phase_type == 'QAM16':
            return 16
        if self.phase_type == 'QAM64':
            return 64
        if self.phase_type == 'QAM':
            if self.qam_order is None:
                raise ValueError(
                    "For phase_type='QAM', 'qam_order' must be specified."
                )
            return int(self.qam_order)

        raise ValueError(
            f"Cannot infer QAM order for phase_type='{self.phase_type}'."
        )
