"""
Grid experimental completo. Gera todos os resultados brutos e todas as
figuras do trabalho:

    python3 experiments/run_experiment.py

1. Modulacoes (BPSK, QPSK, 16-QAM) x canais (AWGN, Rayleigh) x 8 valores de
   Eb/N0 x 5 repeticoes (seeds) x 100.000 bits  -> 240 simulacoes.
2. Codificacoes de linha (AMI, Manchester) no mesmo grid -> 160 simulacoes.
3. Curvas BER x SNR com barras de erro (desvio padrao amostral entre as
   repeticoes) e as curvas teoricas tracejadas (src/teoria.py).
4. Comparacao AWGN x Rayleigh por modulacao.
5. Tabela de compromisso BER x eficiencia espectral.
6. Diagramas de constelacao antes e depois de cada canal.
7. Exemplo de mensagem de texto recuperada em cada canal.
8. configuracao.json com todos os parametros usados.

Seeds fixas e reaproveitadas entre combinacoes (common random numbers):
reduz a variancia ao comparar curvas diferentes nas mesmas condicoes de
ruido "sorteado". O grid e deterministico: rodar de novo reproduz os CSVs
byte a byte.

Pontos com zero erros: quando nenhuma das repeticoes registra erro, a BER
medida e 0 e nao pode ser desenhada em escala log. Esses pontos aparecem
como um triangulo no piso do grafico e significam "nenhum erro em
N_BITS x len(SEEDS) bits", isto e, BER < 3 / (bits totais) com 95% de
confianca (regra do 3) - e NAO uma BER medida igual ao piso.
"""

import os
import sys
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geracao import bits_aleatorios, mensagem_para_bits, bits_para_mensagem
from modulacao import MODULACOES
from codificacao_linha import CODIFICACOES
from simulate import simular
from teoria import TEORIA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_BITS = 100_000
SNR_LIST = [0, 2, 4, 6, 8, 10, 12, 14]   # Eb/N0 em dB, 8 pontos
SEEDS = [100, 101, 102, 103, 104]         # 5 repeticoes, reaproveitadas entre combinacoes
MODULACOES_LISTA = ["bpsk", "qpsk", "16qam"]
CODIFICACOES_LISTA = ["ami", "manchester"]
CANAIS = ["awgn", "rayleigh"]
SNR_TABELA_DB = 8                         # Eb/N0 da tabela de compromisso
SNR_CONSTELACAO_DB = 10                   # Eb/N0 dos diagramas de constelacao
N_SIMBOLOS_CONSTELACAO = 400
SEED_CONSTELACAO = 7
MENSAGEM_EXEMPLO = "UNISINOS Redes de Computadores 2026"
SNR_EXEMPLO_DB = [4, 8, 14]
SEED_EXEMPLO = 1

BER_PISO = 1e-6                            # limite inferior do eixo dos graficos
BITS_POR_PONTO = N_BITS * len(SEEDS)
LIMITE_SEM_ERROS = 3 / BITS_POR_PONTO      # regra do 3: BER < 3/n com 95% de confianca

CORES = {"bpsk": "#1f77b4", "qpsk": "#2e7d32", "16qam": "#b45309",
         "ami": "#6a1b9a", "manchester": "#00838f"}
MARCADORES = {"bpsk": "o", "qpsk": "s", "16qam": "D", "ami": "o", "manchester": "s"}
# BPSK e QPSK tem a mesma BER por bit: o BPSK e desenhado maior e por baixo
# para que as duas curvas continuem visiveis quando coincidem.
TAMANHOS = {"bpsk": 12, "qpsk": 6, "16qam": 7, "ami": 7, "manchester": 7}
ESPESSURAS = {"bpsk": 5, "qpsk": 2, "16qam": 2, "ami": 2, "manchester": 2}
ROTULOS = {"bpsk": "BPSK", "qpsk": "QPSK", "16qam": "16-QAM", "ami": "AMI", "manchester": "Manchester"}


# ---------------------------------------------------------------------------
# Simulacao
# ---------------------------------------------------------------------------
def rodar_grid(esquemas, tipo, nome_csv):
    """tipo: 'modulacao' ou 'codificacao' (nome do argumento de simular)."""
    linhas = []
    total = len(esquemas) * len(CANAIS) * len(SNR_LIST) * len(SEEDS)
    i = 0
    for esquema in esquemas:
        for canal in CANAIS:
            for snr in SNR_LIST:
                for seed in SEEDS:
                    i += 1
                    bits = bits_aleatorios(N_BITS, seed=seed)
                    r = simular(bits, canal=canal, snr_db=snr, seed=seed, **{tipo: esquema})
                    linhas.append({
                        tipo: esquema, "canal": canal, "snr_db": snr, "seed": seed,
                        "n_bits": r["n_bits"], "n_erros": r["n_erros"], "ber": r["ber"],
                        "eficiencia_bits_s_hz": r["eficiencia_bits_s_hz"],
                    })
                    if i % 40 == 0 or i == total:
                        print(f"  [{i}/{total}] {esquema:10s} {canal:9s} SNR={snr:3d}dB seed={seed} -> BER={r['ber']:.6f}")

    caminho = os.path.join(RAW_DIR, nome_csv)
    _salvar_csv(caminho, linhas)
    print(f"Resultados brutos salvos em {caminho} ({len(linhas)} linhas)\n")
    return linhas


def agregar(linhas, tipo, nome_csv):
    """Agrega por (esquema, canal, snr): media e desvio padrao amostral do BER entre repeticoes."""
    grupos = defaultdict(list)
    for l in linhas:
        grupos[(l[tipo], l["canal"], l["snr_db"])].append(l["ber"])

    agregados = []
    for (esquema, canal, snr), bers in grupos.items():
        bers = np.array(bers)
        agregados.append({
            tipo: esquema, "canal": canal, "snr_db": snr,
            "ber_media": bers.mean(), "ber_std": bers.std(ddof=1),
            "ber_min": bers.min(), "ber_max": bers.max(), "n_repeticoes": len(bers),
        })
    caminho = os.path.join(RAW_DIR, nome_csv)
    _salvar_csv(caminho, agregados)
    print(f"Resultados agregados salvos em {caminho}")
    return agregados


def _salvar_csv(caminho, linhas):
    with open(caminho, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def _pontos(agregados, tipo, esquema, canal):
    return sorted([a for a in agregados if a[tipo] == esquema and a["canal"] == canal],
                  key=lambda a: a["snr_db"])


def _plotar_serie(ax, pontos, cor, marcador, rotulo, vazado=False, linha="-", tamanho=7, espessura=2):
    """Desenha media +- desvio padrao. Pontos com BER medida = 0 nao entram na
    curva: viram um triangulo no piso do grafico (nenhum erro observado)."""
    medidos = [p for p in pontos if p["ber_media"] > 0]
    zeros = [p for p in pontos if p["ber_media"] == 0]
    xs = [p["snr_db"] for p in medidos]
    ys = np.array([p["ber_media"] for p in medidos])
    std = np.array([p["ber_std"] for p in medidos])
    # a barra inferior nao pode cruzar o piso do eixo log
    yerr = [np.minimum(std, ys - BER_PISO).clip(min=0), std]
    ax.errorbar(xs, ys, yerr=yerr, fmt=marcador, linestyle=linha, linewidth=espessura, markersize=tamanho,
                capsize=4, color=cor, ecolor="#333333", label=rotulo,
                markerfacecolor="white" if vazado else cor, markeredgewidth=2)
    if zeros:
        ax.plot([p["snr_db"] for p in zeros], [BER_PISO * 1.25] * len(zeros), linestyle="none",
                marker="v", markersize=tamanho + 2, color=cor, markerfacecolor="white" if vazado else cor,
                markeredgewidth=2, clip_on=False)
    return bool(zeros)


def _plotar_teoria(ax, esquema, canal):
    f = TEORIA.get((esquema, canal))
    if f is None:
        return
    x = np.linspace(SNR_LIST[0], SNR_LIST[-1], 200)
    y = f(x)
    ax.plot(x[y >= BER_PISO], y[y >= BER_PISO], linestyle="--", linewidth=1.2, color="#222222", zorder=5)


def _finalizar(ax, titulo, tem_zeros, tem_teoria=True):
    ax.set_yscale("log")
    ax.set_ylim(BER_PISO, 1)
    ax.set_xlim(SNR_LIST[0] - 0.5, SNR_LIST[-1] + 0.5)
    ax.set_xlabel("SNR — Eb/N0 (dB)")
    ax.set_ylabel("BER (escala log)")
    ax.set_title(titulo)
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if tem_teoria:
        handles.append(Line2D([], [], linestyle="--", color="#222222", linewidth=1.2))
        labels.append("teórico")
    if tem_zeros:
        handles.append(Line2D([], [], linestyle="none", marker="v", color="#555555", markersize=8))
        labels.append(f"nenhum erro em {BITS_POR_PONTO:,} bits".replace(",", ".")
                      + f" (BER < {LIMITE_SEM_ERROS:.0e})")
    ax.legend(handles, labels, loc="lower left", fontsize=9)


def plotar_ber_vs_snr(agregados, tipo, esquemas, canal, titulo, caminho_png):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tem_zeros = False
    for esquema in esquemas:
        _plotar_teoria(ax, esquema, canal)
        tem_zeros |= _plotar_serie(ax, _pontos(agregados, tipo, esquema, canal), CORES[esquema],
                                   MARCADORES[esquema], ROTULOS[esquema],
                                   tamanho=TAMANHOS[esquema], espessura=ESPESSURAS[esquema])
    tem_teoria = any((e, canal) in TEORIA for e in esquemas)
    _finalizar(ax, titulo, tem_zeros, tem_teoria)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=180)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_png}")


def plotar_awgn_vs_rayleigh(agregados, tipo, esquema, caminho_png):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tem_zeros = False
    for canal, vazado, linha in [("awgn", False, "-"), ("rayleigh", True, "-")]:
        _plotar_teoria(ax, esquema, canal)
        tem_zeros |= _plotar_serie(ax, _pontos(agregados, tipo, esquema, canal), CORES[esquema],
                                   MARCADORES[esquema], canal.upper() if canal == "awgn" else "Rayleigh",
                                   vazado=vazado, linha=linha)
    tem_teoria = any((esquema, c) in TEORIA for c in CANAIS)
    _finalizar(ax, f"AWGN vs. Rayleigh — {ROTULOS[esquema]}", tem_zeros, tem_teoria)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=180)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_png}")


def plotar_linha(agregados, caminho_png):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tem_zeros = False
    for esquema in CODIFICACOES_LISTA:
        _plotar_teoria(ax, esquema, "awgn")
        tem_zeros |= _plotar_serie(ax, _pontos(agregados, "codificacao", esquema, "awgn"), CORES[esquema],
                                   MARCADORES[esquema], f"{ROTULOS[esquema]} — AWGN")
        tem_zeros |= _plotar_serie(ax, _pontos(agregados, "codificacao", esquema, "rayleigh"), CORES[esquema],
                                   MARCADORES[esquema], f"{ROTULOS[esquema]} — Rayleigh", vazado=True)
    _finalizar(ax, "BER x SNR — Codificações de linha (AMI e Manchester)", tem_zeros)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=180)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_png}")


def plotar_constelacoes(caminho_png):
    """Constelacao transmitida e recebida (AWGN e Rayleigh equalizado) para
    cada modulacao, no mesmo Eb/N0."""
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    colunas = [("ideal", "Transmitida"), ("awgn", f"AWGN, {SNR_CONSTELACAO_DB} dB"),
               ("rayleigh", f"Rayleigh equalizado, {SNR_CONSTELACAO_DB} dB")]
    lim = 2.0
    for i, mod in enumerate(MODULACOES_LISTA):
        bps = MODULACOES[mod].bits_por_simbolo
        bits = bits_aleatorios(N_SIMBOLOS_CONSTELACAO * bps, seed=SEED_CONSTELACAO)
        ideal = np.unique(np.round(MODULACOES[mod].modulate(bits), 6))
        for j, (canal, nome) in enumerate(colunas):
            ax = axes[i, j]
            r = simular(bits, modulacao=mod, canal=canal, snr_db=SNR_CONSTELACAO_DB, seed=SEED_CONSTELACAO)
            rx = r["sinal_rx"]
            dentro = (np.abs(rx.real) <= lim) & (np.abs(rx.imag) <= lim)
            ax.scatter(rx.real[dentro], rx.imag[dentro], s=10, alpha=0.45, color=CORES[mod])
            ax.scatter(ideal.real, ideal.imag, s=70, facecolors="none", edgecolors="black", linewidths=1.5)
            ax.axhline(0, color="#999999", linewidth=0.6)
            ax.axvline(0, color="#999999", linewidth=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            fora = int((~dentro).sum())
            titulo = f"{ROTULOS[mod]} — {nome}"
            if canal != "ideal":
                titulo += f"\nBER = {r['ber']:.4f}" + (f" ({fora} pontos fora do quadro)" if fora else "")
            ax.set_title(titulo, fontsize=10)
            if j == 0:
                ax.set_ylabel("Q (quadratura)")
            if i == 2:
                ax.set_xlabel("I (em fase)")
    fig.suptitle(f"Constelações antes e depois do canal — {N_SIMBOLOS_CONSTELACAO} símbolos, seed {SEED_CONSTELACAO}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=160)
    plt.close(fig)
    print(f"Grafico salvo: {caminho_png}")


# ---------------------------------------------------------------------------
# Tabelas e exemplos
# ---------------------------------------------------------------------------
def tabela_compromisso(agregados, snr_referencia, caminho_csv):
    linhas = []
    for mod in MODULACOES_LISTA:
        for canal in CANAIS:
            p = [a for a in agregados if a["modulacao"] == mod and a["canal"] == canal
                 and a["snr_db"] == snr_referencia][0]
            linhas.append({
                "modulacao": mod, "canal": canal, "snr_db": snr_referencia,
                "ber_media": p["ber_media"], "eficiencia_bits_s_hz": MODULACOES[mod].bits_por_simbolo,
            })
    _salvar_csv(caminho_csv, linhas)
    print(f"Tabela de compromisso (SNR={snr_referencia}dB) salva em {caminho_csv}")
    return linhas


def tabela_teoria(agregados_mod, agregados_linha, caminho_csv):
    """Simulado x teorico, ponto a ponto, para todas as curvas com formula fechada."""
    linhas = []
    for tipo, agregados in [("modulacao", agregados_mod), ("codificacao", agregados_linha)]:
        for a in agregados:
            f = TEORIA.get((a[tipo], a["canal"]))
            if f is None:
                continue
            teo = float(f(a["snr_db"]))
            linhas.append({
                "esquema": a[tipo], "canal": a["canal"], "snr_db": a["snr_db"],
                "ber_simulada": a["ber_media"], "ber_std": a["ber_std"], "ber_teorica": teo,
                "erros_esperados_por_execucao": teo * N_BITS,
            })
    _salvar_csv(caminho_csv, linhas)
    print(f"Comparacao simulado x teorico salva em {caminho_csv}")
    return linhas


def exemplo_mensagem(caminho_txt):
    """Mostra como cada canal afeta a recuperacao de uma mensagem de texto."""
    bits = mensagem_para_bits(MENSAGEM_EXEMPLO)
    out = [f"Mensagem enviada ({len(bits)} bits): {MENSAGEM_EXEMPLO!r}",
           f"Modulacao QPSK, seed {SEED_EXEMPLO}", ""]
    for snr in SNR_EXEMPLO_DB:
        for canal in CANAIS:
            r = simular(bits, modulacao="qpsk", canal=canal, snr_db=snr, seed=SEED_EXEMPLO)
            out.append(f"Eb/N0 = {snr:2d} dB  {canal:9s} {r['n_erros']:3d} bits errados  "
                       f"recebida: {bits_para_mensagem(r['bits_rx'])!r}")
            if r["n_erros"]:
                out.append(f"{'':28s}posicoes dos bits incorretos: {r['erros_idx'].tolist()}")
        out.append("")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Exemplo de mensagem salvo em {caminho_txt}")


def salvar_configuracao(caminho_json):
    cfg = {
        "n_bits_por_execucao": N_BITS, "snr_db_eb_n0": SNR_LIST, "seeds": SEEDS,
        "modulacoes": MODULACOES_LISTA, "codificacoes_de_linha": CODIFICACOES_LISTA, "canais": CANAIS,
        "bits_por_ponto_da_curva": BITS_POR_PONTO,
        "limite_95_sem_erros": LIMITE_SEM_ERROS,
        "variabilidade": "desvio padrao amostral (ddof=1) da BER entre as repeticoes",
        "snr_tabela_compromisso_db": SNR_TABELA_DB,
        "constelacoes": {"snr_db": SNR_CONSTELACAO_DB, "n_simbolos": N_SIMBOLOS_CONSTELACAO,
                         "seed": SEED_CONSTELACAO},
        "exemplo_mensagem": {"mensagem": MENSAGEM_EXEMPLO, "snr_db": SNR_EXEMPLO_DB, "seed": SEED_EXEMPLO},
        "rayleigh": "plano, i.i.d. por simbolo, E[|h|^2]=1, CSI perfeito, equalizacao zero-forcing",
    }
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Configuracao salva em {caminho_json}")


if __name__ == "__main__":
    print(f"Grid de modulacoes: {len(MODULACOES_LISTA)} x {len(CANAIS)} canais x "
          f"{len(SNR_LIST)} SNRs x {len(SEEDS)} repeticoes x {N_BITS} bits")
    brutos_mod = rodar_grid(MODULACOES_LISTA, "modulacao", "resultados_brutos.csv")
    print(f"Grid de codificacoes de linha: {len(CODIFICACOES_LISTA)} x {len(CANAIS)} canais x "
          f"{len(SNR_LIST)} SNRs x {len(SEEDS)} repeticoes x {N_BITS} bits")
    brutos_linha = rodar_grid(CODIFICACOES_LISTA, "codificacao", "resultados_linha_brutos.csv")

    ag_mod = agregar(brutos_mod, "modulacao", "resultados_agregados.csv")
    ag_linha = agregar(brutos_linha, "codificacao", "resultados_linha_agregados.csv")

    plotar_ber_vs_snr(ag_mod, "modulacao", MODULACOES_LISTA, "awgn", "BER x SNR — Canal AWGN",
                      os.path.join(FIG_DIR, "ber_vs_snr_awgn.png"))
    plotar_ber_vs_snr(ag_mod, "modulacao", MODULACOES_LISTA, "rayleigh",
                      "BER x SNR — Canal com Desvanecimento Rayleigh",
                      os.path.join(FIG_DIR, "ber_vs_snr_rayleigh.png"))
    for mod in MODULACOES_LISTA:
        plotar_awgn_vs_rayleigh(ag_mod, "modulacao", mod, os.path.join(FIG_DIR, f"awgn_vs_rayleigh_{mod}.png"))
    plotar_linha(ag_linha, os.path.join(FIG_DIR, "ber_vs_snr_linha.png"))
    plotar_constelacoes(os.path.join(FIG_DIR, "constelacoes.png"))

    tabela_compromisso(ag_mod, snr_referencia=SNR_TABELA_DB,
                       caminho_csv=os.path.join(RAW_DIR, f"tabela_compromisso_snr{SNR_TABELA_DB}db.csv"))
    tabela_teoria(ag_mod, ag_linha, os.path.join(RAW_DIR, "simulado_vs_teorico.csv"))
    exemplo_mensagem(os.path.join(RAW_DIR, "exemplo_mensagem.txt"))
    salvar_configuracao(os.path.join(RAW_DIR, "configuracao.json"))

    print("\nGrid experimental concluido.")
