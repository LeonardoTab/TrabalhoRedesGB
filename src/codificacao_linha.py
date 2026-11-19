"""
Codificacoes de linha (baseband line coding): AMI e Manchester.

IMPORTANTE - diferenca entre codificacao de linha e codificacao de correcao
de erros (ver tambem o texto no README):
    Codificacao de linha define como os bits 0/1 sao representados fisicamente
    como niveis/pulsos de tensao no meio de transmissao (aqui, simulados como
    amostras reais). Ela NAO adiciona redundancia para detectar/corrigir erros
    - e so uma representacao do mesmo bit. Codificacao de correcao de erros
    (ex. Hamming, visto na Atividade 08) adiciona bits extras (paridade) para
    que o receptor consiga detectar e, em alguns casos, corrigir bits que o
    canal corrompeu. Uma linha codificada em AMI/Manchester ainda pode chegar
    com erros se o canal for ruidoso o bastante; so a codificacao de correcao
    de erros resolve isso.

Cada codificacao expoe:
    encode(bits) -> sinal (np.ndarray de reais, "amostras")
    decode(sinal) -> bits (np.ndarray de 0/1)
    amostras_por_bit -> quantas amostras do sinal representam 1 bit
    energia_media -> energia media por AMOSTRA para bits equiprovaveis; o
        canal usa esse valor para calibrar o ruido em Eb/N0 (ver canal.py).
        No AMI metade das amostras e zero, entao a energia media e 0.5 e nao 1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# AMI (Alternate Mark Inversion) - 1 amostra por bit
#   bit 0 -> nivel 0 (sem pulso)
#   bit 1 -> alterna entre +1 e -1 a cada novo "1" (balanceamento DC)
# ---------------------------------------------------------------------------
class AMI:
    amostras_por_bit = 1
    energia_media = 0.5  # bits equiprovaveis: metade das amostras e 0, metade e +-1

    @staticmethod
    def encode(bits: np.ndarray) -> np.ndarray:
        # o k-esimo bit 1 recebe +1 se k e impar e -1 se k e par (polaridade alternada)
        ordem = np.cumsum(bits == 1)
        return np.where(bits == 1, np.where(ordem % 2 == 1, 1.0, -1.0), 0.0)

    @staticmethod
    def decode(sinal: np.ndarray) -> np.ndarray:
        # qualquer pulso (positivo ou negativo) com magnitude relevante = bit 1
        limiar = 0.5
        return (np.abs(sinal) > limiar).astype(np.uint8)


# ---------------------------------------------------------------------------
# Manchester (convencao de G. E. Thomas) - 2 amostras por bit, com transicao no meio
#   bit 0 -> [-1, +1]  (transicao baixo->alto)
#   bit 1 -> [+1, -1]  (transicao alto->baixo)
#   A convencao IEEE 802.3 usa o mapeamento inverso (bit 1 = baixo->alto). As
#   duas sao equivalentes em desempenho; so muda qual transicao representa o 1.
#   Autossincronizante: sempre ha uma transicao no meio de cada bit, o que
#   permite ao receptor recuperar o clock mesmo sem uma referencia externa -
#   diferente do AMI, que fica "parado" em longas sequencias de zeros.
# ---------------------------------------------------------------------------
class Manchester:
    amostras_por_bit = 2
    energia_media = 1.0  # toda amostra e +-1

    @staticmethod
    def encode(bits: np.ndarray) -> np.ndarray:
        sinal = np.empty(len(bits) * 2, dtype=float)
        sinal[0::2] = np.where(bits == 1, 1.0, -1.0)
        sinal[1::2] = np.where(bits == 1, -1.0, 1.0)
        return sinal

    @staticmethod
    def decode(sinal: np.ndarray) -> np.ndarray:
        primeira_metade = sinal[0::2]
        segunda_metade = sinal[1::2]
        # bit = 1 se a primeira metade ficou "mais alta" que a segunda
        return (primeira_metade > segunda_metade).astype(np.uint8)


CODIFICACOES = {"ami": AMI, "manchester": Manchester}


if __name__ == "__main__":
    from geracao import mensagem_para_bits, bits_para_mensagem

    bits = mensagem_para_bits("Teste 101")
    for nome, cls in CODIFICACOES.items():
        sinal = cls.encode(bits)
        bits_rx = cls.decode(sinal)
        ok = np.array_equal(bits, bits_rx)
        print(f"{nome:12s} amostras/bit={cls.amostras_por_bit}  "
              f"len(sinal)={len(sinal):5d}  round-trip sem ruido: {'OK' if ok else 'FALHOU'}")
        assert ok
