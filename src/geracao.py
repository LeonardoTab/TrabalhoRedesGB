"""
Geracao de dados: mensagem ASCII <-> bits, e geracao de fluxos pseudoaleatorios
de bits para os experimentos estatisticos (PRBS - pseudo-random bit sequence).

Por que dois modos?
- Para a VALIDACAO FUNCIONAL / teste de consistencia, usamos uma mensagem ASCII
  real, curta, legivel, para demonstrar a cadeia completa (texto -> bits ->
  linha/modulacao -> canal -> demod -> decod -> bits -> texto) e mostrar os
  bits/caracteres que vieram errados.
- Para os EXPERIMENTOS ESTATISTICOS (>=100k bits por combinacao, 5 repeticoes
  com seeds diferentes), usamos fluxos de bits pseudoaleatorios com seed
  controlada. Isso e pratica padrao em medicao de BER (sequencias PRBS) e
  equivale, para fins de medicao de taxa de erro, a transmitir uma mensagem
  ASCII arbitrariamente longa - mas nos da controle exato sobre a quantidade
  de bits e reprodutibilidade via seed.
"""

import numpy as np


def mensagem_para_bits(texto: str) -> np.ndarray:
    """Converte uma string (ASCII/UTF-8) em um vetor de bits (0/1), MSB primeiro por byte."""
    dados = texto.encode("utf-8")
    bits = np.unpackbits(np.frombuffer(dados, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_para_mensagem(bits: np.ndarray) -> str:
    """Converte um vetor de bits (0/1) de volta em string. Bits incompletos (nao
    multiplos de 8) no final sao descartados. Bytes invalidos em UTF-8 (por causa
    de erros de transmissao) sao mantidos como caracteres de substituicao, para
    que a mensagem corrompida ainda possa ser inspecionada."""
    n_bytes = len(bits) // 8
    bits = bits[: n_bytes * 8]
    if n_bytes == 0:
        return ""
    byte_arr = np.packbits(bits.astype(np.uint8))
    return byte_arr.tobytes().decode("utf-8", errors="replace")


def bits_aleatorios(n_bits: int, seed: int) -> np.ndarray:
    """Gera um fluxo de bits pseudoaleatorios (PRBS) reprodutivel via seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


if __name__ == "__main__":
    msg = "UNISINOS Redes 2026"
    bits = mensagem_para_bits(msg)
    print(f"Mensagem: {msg!r}")
    print(f"Bits ({len(bits)}): {''.join(map(str, bits[:40]))}...")
    de_volta = bits_para_mensagem(bits)
    print(f"Reconstruida: {de_volta!r}")
    assert de_volta == msg
    print("OK: round-trip sem erros.")
