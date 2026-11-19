"""
Curvas teoricas de BER, usadas para validar a simulacao (sugestao de
aprofundamento do enunciado). Todas recebem Eb/N0 em dB e devolvem a
probabilidade de erro de bit.

Notacao: g = Eb/N0 linear; Q(x) = 0.5 * erfc(x / sqrt(2)).

AWGN
    BPSK e QPSK (Gray):   Pb = Q(sqrt(2 g))
    16-QAM (Gray):        Pb = 3/4 Q(sqrt(4g/5)) + 1/2 Q(3 sqrt(4g/5)) - 1/4 Q(5 sqrt(4g/5))
    Manchester (decisao pela diferenca das duas metades): sinal antipodal,
                          Pb = Q(sqrt(2 g))
    AMI (limiar em A/2):  Pb = 3/2 Q(sqrt(g)) - 1/2 Q(3 sqrt(g))
        (o bit 0 erra para os dois lados; o bit 1 so erra para dentro)

Rayleigh plano com CSI perfeito (deteccao coerente)
    A media de Q(sqrt(2 c g |h|^2)) sobre |h|^2 exponencial vale
        1/2 (1 - sqrt(c g / (1 + c g)))
    BPSK e QPSK: c = 1
    16-QAM: mesma combinacao do AWGN, com c = 2/5, 18/5 e 10
"""

import numpy as np
from scipy.special import erfc


def _q(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def _g(snr_db):
    return 10 ** (np.asarray(snr_db, dtype=float) / 10)


def _ray(c, g):
    return 0.5 * (1 - np.sqrt(c * g / (1 + c * g)))


def ber_bpsk_awgn(snr_db):
    return _q(np.sqrt(2 * _g(snr_db)))


ber_qpsk_awgn = ber_bpsk_awgn
ber_manchester_awgn = ber_bpsk_awgn


def ber_16qam_awgn(snr_db):
    a = np.sqrt(0.8 * _g(snr_db))
    return 0.75 * _q(a) + 0.5 * _q(3 * a) - 0.25 * _q(5 * a)


def ber_ami_awgn(snr_db):
    a = np.sqrt(_g(snr_db))
    return 1.5 * _q(a) - 0.5 * _q(3 * a)


def ber_bpsk_rayleigh(snr_db):
    return _ray(1.0, _g(snr_db))


ber_qpsk_rayleigh = ber_bpsk_rayleigh


def ber_16qam_rayleigh(snr_db):
    g = _g(snr_db)
    return 0.75 * _ray(0.4, g) + 0.5 * _ray(3.6, g) - 0.25 * _ray(10.0, g)


TEORIA = {
    ("bpsk", "awgn"): ber_bpsk_awgn,
    ("qpsk", "awgn"): ber_qpsk_awgn,
    ("16qam", "awgn"): ber_16qam_awgn,
    ("bpsk", "rayleigh"): ber_bpsk_rayleigh,
    ("qpsk", "rayleigh"): ber_qpsk_rayleigh,
    ("16qam", "rayleigh"): ber_16qam_rayleigh,
    ("ami", "awgn"): ber_ami_awgn,
    ("manchester", "awgn"): ber_manchester_awgn,
}


if __name__ == "__main__":
    print("Eb/N0   BPSK/QPSK AWGN   16-QAM AWGN   BPSK/QPSK Rayleigh   16-QAM Rayleigh")
    for s in range(0, 16, 2):
        print(f"{s:3d} dB   {ber_bpsk_awgn(s):.3e}      {ber_16qam_awgn(s):.3e}    "
              f"{ber_bpsk_rayleigh(s):.3e}            {ber_16qam_rayleigh(s):.3e}")
    assert abs(ber_bpsk_awgn(0) - 0.0786) < 1e-3
    assert abs(ber_bpsk_rayleigh(0) - 0.1464) < 1e-3
    print("OK")
