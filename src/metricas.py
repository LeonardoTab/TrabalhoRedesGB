"""Metricas: BER, localizacao de erros e eficiencia espectral nominal."""

import numpy as np


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    n = min(len(bits_tx), len(bits_rx))
    if n == 0:
        return float("nan")
    erros = np.sum(bits_tx[:n] != bits_rx[:n])
    return erros / n


def localizar_erros(bits_tx: np.ndarray, bits_rx: np.ndarray) -> np.ndarray:
    n = min(len(bits_tx), len(bits_rx))
    return np.where(bits_tx[:n] != bits_rx[:n])[0]


def eficiencia_espectral_nominal(bits_por_simbolo: float) -> float:
    """bits/s/Hz nominal, assumindo taxa de simbolo = banda (Nyquist ideal),
    mesma convencao usada na Atividade 10 (Nyquist/Shannon)."""
    return float(bits_por_simbolo)
