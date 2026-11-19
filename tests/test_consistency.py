"""
Teste de consistencia / validacao funcional (item "Realizar um teste de
consistencia" do plano experimental), em duas etapas:

1. CANAL IDEAL (sem ruido): todas as codificacoes e modulacoes devem
   reconstruir a mensagem exatamente. Qualquer erro aqui e bug de mapeamento
   ou sincronismo no pipeline, e o teste falha.

2. SNR ELEVADA (Eb/N0 = 30 dB) nos dois canais: espera-se BER = 0. Quando ha
   divergencia, o teste a DIAGNOSTICA em vez de so acusar: para cada bit
   errado mostra o valor transmitido, o valor recebido e, no Rayleigh, o
   ganho |h| do canal naquele instante.

   Divergencia explicada: erro em canal Rayleigh num instante de
   desvanecimento profundo (|h| < 0.1, ou seja, mais de 20 dB abaixo do ganho
   medio). A equalizacao divide por h e, com |h| pequeno, amplifica o ruido
   ("noise enhancement" do zero-forcing). E um efeito fisico do canal
   modelado, nao um defeito do codigo, e nenhuma codificacao de linha
   protege contra ele - so uma codificacao de correcao de erros protegeria.

   Qualquer outra divergencia e tratada como falha.

Codigo de saida: 0 se nao houver falha nao explicada, 1 caso contrario.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from geracao import mensagem_para_bits, bits_para_mensagem
from codificacao_linha import CODIFICACOES
from modulacao import MODULACOES
from simulate import simular, indices_do_sinal

MENSAGEM = "Teste de consistencia UNISINOS Redes de Computadores 2026 - dupla X."
SNR_ALTA_DB = 30.0
SEED = 42
LIMIAR_FADE = 0.1  # |h| abaixo disso = desvanecimento profundo (ganho < -20 dB)

ESQUEMAS = [("modulacao", n) for n in MODULACOES] + [("codificacao", n) for n in CODIFICACOES]


def _rodar(tipo, nome, canal, bits):
    r = simular(bits, canal=canal, snr_db=SNR_ALTA_DB, seed=SEED, **{tipo: nome})
    msg_rx = bits_para_mensagem(r["bits_rx"])
    return r, msg_rx


def _diagnosticar(r, canal):
    """Devolve (explicada, linhas de texto) para os bits errados de r."""
    linhas, explicada = [], canal == "rayleigh"
    for bit in r["erros_idx"].tolist():
        idx = indices_do_sinal(r, bit)
        tx = ", ".join(f"{v:.3f}" for v in np.atleast_1d(r["sinal_tx"][idx]))
        rx = ", ".join(f"{v:.3f}" for v in np.atleast_1d(r["sinal_rx"][idx]))
        txt = (f"      bit {bit}: enviado {int(r['bits_tx'][bit])}, recebido {int(r['bits_rx'][bit])} | "
               f"sinal tx [{tx}] -> rx [{rx}]")
        if r["h"] is not None:
            h_min = float(np.min(np.abs(r["h"][idx])))
            txt += f" | |h| = {h_min:.4f} (ruido amplificado ~{1 / h_min:.0f}x)"
            if h_min >= LIMIAR_FADE:
                explicada = False
        linhas.append(txt)
    return explicada, linhas


def rodar_testes():
    bits = mensagem_para_bits(MENSAGEM)
    print(f"Mensagem de teste ({len(bits)} bits): {MENSAGEM!r}\n")

    falhas = 0

    print("Etapa 1 - canal ideal (sem ruido)")
    for tipo, nome in ESQUEMAS:
        r, msg_rx = _rodar(tipo, nome, "ideal", bits)
        ok = r["n_erros"] == 0 and msg_rx == MENSAGEM
        falhas += not ok
        print(f"  [{'OK' if ok else 'FALHOU'}] {tipo}={nome:11s} n_erros={r['n_erros']}")

    print(f"\nEtapa 2 - SNR elevada (Eb/N0 = {SNR_ALTA_DB} dB, seed {SEED})")
    perfeitas, explicadas, total = 0, 0, 0
    for tipo, nome in ESQUEMAS:
        for canal in ["awgn", "rayleigh"]:
            total += 1
            r, msg_rx = _rodar(tipo, nome, canal, bits)
            if r["n_erros"] == 0 and msg_rx == MENSAGEM:
                perfeitas += 1
                print(f"  [OK] {tipo}={nome:11s} canal={canal:9s} BER=0  n_erros=0")
                continue
            explicada, linhas = _diagnosticar(r, canal)
            status = "DIVERGENCIA EXPLICADA" if explicada else "FALHOU"
            explicadas += explicada
            falhas += not explicada
            print(f"  [{status}] {tipo}={nome:11s} canal={canal:9s} BER={r['ber']:.6f}  n_erros={r['n_erros']}")
            print(f"      mensagem recebida: {msg_rx!r}")
            print("\n".join(linhas))

    print(f"\nResumo: {perfeitas}/{total} combinacoes reconstruiram a mensagem sem nenhum bit incorreto; "
          f"{explicadas} divergencia(s) explicada(s); {falhas} falha(s).")
    if explicadas:
        print("Divergencia explicada = erro em canal Rayleigh num instante de desvanecimento profundo "
              f"(|h| < {LIMIAR_FADE}): a equalizacao por divisao amplifica o ruido por 1/|h|. "
              "Efeito fisico do canal, nao bug do pipeline (a etapa 1 prova que o mapeamento esta correto).")
    return falhas == 0


if __name__ == "__main__":
    sys.exit(0 if rodar_testes() else 1)
