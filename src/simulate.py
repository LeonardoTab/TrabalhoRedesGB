"""
Pipeline ponta a ponta: bits -> (codificacao de linha OU modulacao) -> canal
-> demod/decod -> bits, com calculo de BER e eficiencia espectral nominal.

Uso como biblioteca (ver experiments/run_experiment.py) ou via CLI, a partir
da pasta src/:

    python3 simulate.py --mensagem "Ola, UNISINOS!" --modulacao qpsk --canal awgn --snr-db 8
    python3 simulate.py --n-bits 100000 --seed 1 --modulacao 16qam --canal rayleigh --snr-db 12
    python3 simulate.py --mensagem "Teste" --codificacao manchester --canal awgn --snr-db 5
    python3 simulate.py --mensagem "Teste" --codificacao ami --canal ideal
"""

import argparse
import numpy as np

from geracao import mensagem_para_bits, bits_para_mensagem, bits_aleatorios
from codificacao_linha import CODIFICACOES
from modulacao import MODULACOES
from canal import awgn, rayleigh, CANAIS
from metricas import calcular_ber, localizar_erros, eficiencia_espectral_nominal


def simular(bits: np.ndarray, *, modulacao: str = None, codificacao: str = None,
            canal: str = "awgn", snr_db: float = 10.0, seed: int = 0) -> dict:
    """Roda um unico ponto de simulacao (uma combinacao de parametros) e
    devolve um dicionario com bits recebidos, BER, eficiencia espectral, o
    sinal antes e depois do canal e, se usando desvanecimento, o canal h
    amostrado.

    canal: "ideal" (sem ruido, usado no teste de consistencia), "awgn" ou
    "rayleigh". snr_db e sempre Eb/N0 em dB e e ignorado no canal ideal.
    """
    if (modulacao is None) == (codificacao is None):
        raise ValueError("Especifique exatamente um entre 'modulacao' ou 'codificacao'.")
    if canal not in CANAIS:
        raise ValueError(f"canal desconhecido: {canal}")

    rng = np.random.default_rng(seed)
    h_info = None

    if modulacao is not None:
        mod_cls = MODULACOES[modulacao]
        bps = mod_cls.bits_por_simbolo
        amostras_por_bit = 1.0 / bps
        # garante que o numero de bits e multiplo de bits_por_simbolo
        n_validos = (len(bits) // bps) * bps
        bits_tx = bits[:n_validos]
        sinal_tx = mod_cls.modulate(bits_tx)

        if canal == "ideal":
            sinal_rx = sinal_tx.copy()
        elif canal == "awgn":
            sinal_rx = awgn(sinal_tx, snr_db, bps, rng)
        else:
            sinal_rx, h_info = rayleigh(sinal_tx, snr_db, bps, rng)

        bits_rx = mod_cls.demodulate(sinal_rx)
        eficiencia = eficiencia_espectral_nominal(bps)

    else:
        linha_cls = CODIFICACOES[codificacao]
        amostras_por_bit = linha_cls.amostras_por_bit  # 1 para AMI, 2 para Manchester
        bits_tx = bits
        sinal_tx = linha_cls.encode(bits_tx)
        bits_por_amostra = 1.0 / amostras_por_bit
        es = linha_cls.energia_media  # energia media por amostra (0.5 no AMI, 1.0 no Manchester)

        if canal == "ideal":
            sinal_rx = sinal_tx.copy()
        elif canal == "awgn":
            sinal_rx = awgn(sinal_tx.astype(complex), snr_db, bits_por_amostra, rng,
                            real_apenas=True, es=es).real
        else:
            sinal_rx_c, h_info = rayleigh(sinal_tx.astype(complex), snr_db, bits_por_amostra, rng, es=es)
            sinal_rx = sinal_rx_c.real

        bits_rx = linha_cls.decode(sinal_rx)
        eficiencia = eficiencia_espectral_nominal(bits_por_amostra)

    ber = calcular_ber(bits_tx, bits_rx)
    erros_idx = localizar_erros(bits_tx, bits_rx)

    return {
        "bits_tx": bits_tx,
        "bits_rx": bits_rx,
        "ber": ber,
        "n_bits": len(bits_tx),
        "n_erros": len(erros_idx),
        "erros_idx": erros_idx,
        "eficiencia_bits_s_hz": eficiencia,
        "h": h_info,
        "sinal_tx": sinal_tx,          # simbolos (modulacao) ou amostras (linha) transmitidos
        "sinal_rx": sinal_rx,          # o mesmo, depois do canal (e da equalizacao no Rayleigh)
        "amostras_por_bit": amostras_por_bit,
    }


def indices_do_sinal(r: dict, bit_idx: int) -> list:
    """Indices de simbolo/amostra de r["sinal_tx"] que carregam o bit bit_idx."""
    apb = r["amostras_por_bit"]
    if apb >= 1:
        apb = int(apb)
        return list(range(bit_idx * apb, bit_idx * apb + apb))
    return [int(bit_idx * apb)]


def _main():
    p = argparse.ArgumentParser(description="Simulador de transmissao digital ponta a ponta.")
    grupo_bits = p.add_mutually_exclusive_group(required=True)
    grupo_bits.add_argument("--mensagem", type=str, help="Mensagem ASCII/UTF-8 a transmitir")
    grupo_bits.add_argument("--n-bits", type=int, help="Numero de bits pseudoaleatorios a transmitir")

    grupo_tx = p.add_mutually_exclusive_group(required=True)
    grupo_tx.add_argument("--modulacao", choices=MODULACOES.keys())
    grupo_tx.add_argument("--codificacao", choices=CODIFICACOES.keys())

    p.add_argument("--canal", choices=CANAIS, default="awgn")
    p.add_argument("--snr-db", type=float, default=10.0, help="Eb/N0 em dB (ignorado no canal ideal)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.mensagem is not None:
        bits = mensagem_para_bits(args.mensagem)
    else:
        bits = bits_aleatorios(args.n_bits, seed=args.seed)

    r = simular(bits, modulacao=args.modulacao, codificacao=args.codificacao,
                canal=args.canal, snr_db=args.snr_db, seed=args.seed)

    snr_txt = "-" if args.canal == "ideal" else f"{args.snr_db} dB"
    print(f"Esquema: {args.modulacao or args.codificacao}  Canal: {args.canal}  "
          f"SNR(Eb/N0): {snr_txt}  Seed: {args.seed}")
    print(f"Bits transmitidos: {r['n_bits']}  Bits errados: {r['n_erros']}  BER: {r['ber']:.6f}")
    print(f"Eficiencia espectral nominal: {r['eficiencia_bits_s_hz']} bits/s/Hz")
    if args.mensagem is not None:
        msg_rx = bits_para_mensagem(r["bits_rx"])
        print(f"Mensagem original:   {args.mensagem!r}")
        print(f"Mensagem recebida:   {msg_rx!r}")
        if len(r["erros_idx"]) > 0:
            print(f"Posicoes dos bits incorretos (primeiros 20): {r['erros_idx'][:20].tolist()}")


if __name__ == "__main__":
    _main()
