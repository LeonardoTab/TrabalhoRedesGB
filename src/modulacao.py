"""
Modulacoes digitais: BPSK, QPSK e 16-QAM.

Todas normalizam a energia MEDIA do simbolo para 1 (Es=1), o que permite
comparar as modulacoes de forma justa em termos de Eb/N0 (energia por BIT),
que e o eixo padrao das curvas de BER teoricas/praticas - ver canal.py para
a conversao de Eb/N0 para a variancia de ruido em cada caso.

Cada modulacao expoe:
    modulate(bits) -> simbolos (np.ndarray complexo)
    demodulate(simbolos) -> bits (np.ndarray de 0/1), por distancia minima
        (deteccao de maxima verossimilhanca para ruido gaussiano branco)
    bits_por_simbolo -> quantos bits cada simbolo carrega (eficiencia espectral
        nominal, em bits/s/Hz, assumindo taxa de simbolo = banda disponivel)
"""

import numpy as np


# ---------------------------------------------------------------------------
# BPSK - 1 bit/simbolo
# ---------------------------------------------------------------------------
class BPSK:
    bits_por_simbolo = 1
    constelacao = np.array([-1.0, 1.0], dtype=complex)

    @staticmethod
    def modulate(bits: np.ndarray) -> np.ndarray:
        return np.where(bits == 1, 1.0, -1.0).astype(complex)

    @staticmethod
    def demodulate(simbolos: np.ndarray) -> np.ndarray:
        return (simbolos.real > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# QPSK - 2 bits/simbolo, Gray coding, energia media normalizada a 1
#   bits (b0 b1) -> I = (1 se b0==0 senao -1)/sqrt(2); Q = (1 se b1==0 senao -1)/sqrt(2)
#   Gray: pontos vizinhos na constelacao diferem em so 1 bit, minimizando o
#   numero de bits errados quando o ruido empurra o simbolo pro vizinho mais
#   proximo (o erro mais provavel).
# ---------------------------------------------------------------------------
class QPSK:
    bits_por_simbolo = 2
    _norm = 1 / np.sqrt(2)

    @classmethod
    def modulate(cls, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(-1, 2)
        I = np.where(bits[:, 0] == 0, 1.0, -1.0)
        Q = np.where(bits[:, 1] == 0, 1.0, -1.0)
        return (I + 1j * Q) * cls._norm

    @staticmethod
    def demodulate(simbolos: np.ndarray) -> np.ndarray:
        b0 = (simbolos.real < 0).astype(np.uint8)
        b1 = (simbolos.imag < 0).astype(np.uint8)
        bits = np.empty(len(simbolos) * 2, dtype=np.uint8)
        bits[0::2] = b0
        bits[1::2] = b1
        return bits


# ---------------------------------------------------------------------------
# 16-QAM - 4 bits/simbolo (2 para I, 2 para Q), Gray coding em cada eixo,
# niveis {-3,-1,1,3} normalizados para energia media 1 (dividido por sqrt(10),
# que e a energia media de um 16-QAM quadrado com esses niveis).
# ---------------------------------------------------------------------------
class QAM16:
    bits_por_simbolo = 4
    _norm = 1 / np.sqrt(10)
    # mapeamento Gray de 2 bits -> nivel {-3,-1,1,3}
    _gray_para_nivel = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
    _nivel_para_gray = {v: k for k, v in _gray_para_nivel.items()}
    _niveis_ordenados = np.array([-3.0, -1.0, 1.0, 3.0])

    @classmethod
    def modulate(cls, bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(-1, 4)
        I = np.array([cls._gray_para_nivel[(b0, b1)] for b0, b1 in bits[:, 0:2]])
        Q = np.array([cls._gray_para_nivel[(b0, b1)] for b0, b1 in bits[:, 2:4]])
        return (I + 1j * Q) * cls._norm

    @classmethod
    def demodulate(cls, simbolos: np.ndarray) -> np.ndarray:
        I = simbolos.real / cls._norm
        Q = simbolos.imag / cls._norm
        # decisao por nivel mais proximo (deteccao por distancia minima em cada eixo)
        idx_I = np.argmin(np.abs(I[:, None] - cls._niveis_ordenados[None, :]), axis=1)
        idx_Q = np.argmin(np.abs(Q[:, None] - cls._niveis_ordenados[None, :]), axis=1)
        niveis_I = cls._niveis_ordenados[idx_I]
        niveis_Q = cls._niveis_ordenados[idx_Q]
        bits = np.empty((len(simbolos), 4), dtype=np.uint8)
        for i, (nI, nQ) in enumerate(zip(niveis_I, niveis_Q)):
            b0, b1 = cls._nivel_para_gray[nI]
            b2, b3 = cls._nivel_para_gray[nQ]
            bits[i] = (b0, b1, b2, b3)
        return bits.reshape(-1)


MODULACOES = {"bpsk": BPSK, "qpsk": QPSK, "16qam": QAM16}


def energia_media(mod_cls, n_teste=20000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_teste * mod_cls.bits_por_simbolo, dtype=np.uint8)
    simbolos = mod_cls.modulate(bits)
    return float(np.mean(np.abs(simbolos) ** 2))


if __name__ == "__main__":
    from geracao import bits_aleatorios

    for nome, cls in MODULACOES.items():
        n_bits = cls.bits_por_simbolo * 20000
        bits = bits_aleatorios(n_bits, seed=1)
        simbolos = cls.modulate(bits)
        bits_rx = cls.demodulate(simbolos)
        ok = np.array_equal(bits, bits_rx)
        Es = energia_media(cls)
        print(f"{nome:6s} bits/simbolo={cls.bits_por_simbolo}  Es_medio={Es:.4f}  "
              f"round-trip sem ruido: {'OK' if ok else 'FALHOU'}")
        assert ok
        assert abs(Es - 1.0) < 0.05, f"energia media deveria ser ~1.0, deu {Es}"
    print("Todas as modulacoes OK (round-trip exato e Es=1 confirmados).")
