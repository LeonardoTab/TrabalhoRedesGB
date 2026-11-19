#!/usr/bin/env bash
# Reproduz tudo: autotestes dos modulos, teste de consistencia e grid experimental.
set -e
cd "$(dirname "$0")"

echo "== Autotestes dos modulos =="
for m in geracao codificacao_linha modulacao canal teoria; do
    (cd src && python3 "$m.py")
done

echo
echo "== Teste de consistencia =="
python3 tests/test_consistency.py

echo
echo "== Grid experimental =="
python3 experiments/run_experiment.py

echo
echo "== md5 dos resultados =="
md5sum results/raw/*.csv
