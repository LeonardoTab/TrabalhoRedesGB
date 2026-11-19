"""
Modelos de canal: ideal, AWGN e desvanecimento Rayleigh (com AWGN por cima).

Convencao de SNR usada: Eb/N0 (energia por BIT sobre densidade espectral de
ruido), em dB - e o eixo padrao das curvas de BER na literatura, e o unico
jeito justo de comparar modulacoes com bits/simbolo diferentes (BPSK, QPSK,
16-QAM) "com parametros equivalentes", como pede o enunciado.

Sendo Es a energia media por simbolo (ou por amostra, na codificacao de linha):
    Eb = Es / bits_por_simbolo
    Es/N0 (linear) = (Eb/N0 linear) * bits_por_simbolo
    N0 = Es / (Es/N0)
    variancia do ruido complexo total = N0
    variancia por dimensao (I e Q) = N0/2  ->  desvio padrao = sqrt(N0/2)

Nas modulacoes os simbolos sao normalizados para Es = 1 (ver modulacao.py),
que e o valor padrao do parametro `es`.

Na codificacao de linha (sinais reais, 1 dimensao) "bits_por_simbolo" e na
verdade bits_por_amostra (1 para AMI, 0.5 para Manchester, que usa 2 amostras
por bit) e `es` e a energia media por amostra de cada codigo (0.5 no AMI,
porque metade das amostras e zero; 1.0 no Manchester). Passar o `es` correto
e o que garante que o Eb/N0 nominal seja o Eb/N0 real tambem para o AMI.
"""

import numpy as np


def _ruido_std(snr_db: float, bits_por_simbolo: float, es: float = 1.0) -> float:
    snr_linear = 10 ** (snr_db / 10)       # Eb/N0 linear
    es_n0 = snr_linear * bits_por_simbolo  # Es/N0 linear
    n0 = es / es_n0
    return np.sqrt(n0 / 2)                 # desvio padrao por dimensao (I ou Q)


def awgn(simbolos: np.ndarray, snr_db: float, bits_por_simbolo: float,
         rng: np.random.Generator, real_apenas: bool = False,
         es: float = 1.0) -> np.ndarray:
    """Adiciona ruido gaussiano branco aditivo. Se real_apenas=True (usado na
    codificacao de linha, sinal real em banda base), o ruido so e adicionado
    na parte real. `es` e a energia media por simbolo/amostra do sinal."""
    std = _ruido_std(snr_db, bits_por_simbolo, es)
    ruido_real = rng.normal(0, std, size=len(simbolos))
    if real_apenas:
        return simbolos + ruido_real
    ruido_imag = rng.normal(0, std, size=len(simbolos))
    return simbolos + ruido_real + 1j * ruido_imag


def rayleigh(simbolos: np.ndarray, snr_db: float, bits_por_simbolo: float,
             rng: np.random.Generator, es: float = 1.0):
    """Desvanecimento Rayleigh plano por simbolo (flat fading, i.i.d. por
    simbolo - 'fast fading'), com AWGN somado por cima. Assume-se conhecimento
    perfeito do canal no receptor (CSI perfeito) e equalizacao por divisao
    (zero-forcing de 1 tap), premissa comum em simulacoes didaticas de BER sob
    desvanecimento.

    Efeito colateral conhecido da equalizacao por divisao: quando |h| e muito
    pequeno (desvanecimento profundo), dividir por h amplifica o ruido
    ("noise enhancement"). E isso que explica erros isolados mesmo em SNR
    alta - ver tests/test_consistency.py.

    Retorna (simbolos_equalizados, h) para quem quiser inspecionar o canal.
    """
    h_real = rng.normal(0, 1 / np.sqrt(2), size=len(simbolos))
    h_imag = rng.normal(0, 1 / np.sqrt(2), size=len(simbolos))
    h = h_real + 1j * h_imag  # E[|h|^2] = 1

    std = _ruido_std(snr_db, bits_por_simbolo, es)
    ruido = rng.normal(0, std, size=len(simbolos)) + 1j * rng.normal(0, std, size=len(simbolos))

    recebido = h * simbolos + ruido
    equalizado = recebido / h
    return equalizado, h


CANAIS = ("ideal", "awgn", "rayleigh")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    simbolos = np.array([1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j], dtype=complex) / np.sqrt(2)

    for snr in [0, 10, 20]:
        ruidosos = awgn(simbolos, snr, bits_por_simbolo=2, rng=rng)
        print(f"AWGN  SNR={snr:3d}dB  erro medio |rx-tx| = {np.mean(np.abs(ruidosos - simbolos)):.4f}")

    for snr in [0, 10, 20]:
        eq, h = rayleigh(simbolos, snr, bits_por_simbolo=2, rng=rng)
        print(f"Rayleigh SNR={snr:3d}dB  erro medio |rx_eq-tx| = {np.mean(np.abs(eq - simbolos)):.4f}  "
              f"|h| medio = {np.mean(np.abs(h)):.3f}")

    # calibracao: o desvio padrao medido do ruido deve bater com sqrt(N0/2)
    n = 200_000
    zeros = np.zeros(n, dtype=complex)
    for bps, es in [(1, 1.0), (2, 1.0), (4, 1.0), (1, 0.5), (0.5, 1.0)]:
        r = awgn(zeros, 6.0, bps, rng, es=es)
        esperado = np.sqrt(es / (10 ** 0.6 * bps) / 2)
        assert abs(r.real.std() - esperado) / esperado < 0.02
    print("Erro cai com SNR crescente e o ruido medido bate com sqrt(N0/2): OK.")
