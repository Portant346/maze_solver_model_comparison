#!/bin/bash
# Recorre TODOS los modelos (familias + submodelos CNN) en UN mapa, reseteando el
# mundo entre cada run. Pensado para la comparativa de navegacion del TFG.
#
# Uso (WSL2):
#   Terminal 1:  gz sim MundosTest/world_test<N>.sdf   (y DALE A PLAY ▶)
#   Terminal 2:  bash Robot/run_all_models.sh <N>      (N = 1..5, el mismo mapa)
#   Repite para los 5 mapas. Al final:  python Modelos/src/analyze_log.py --compare
#
# Cada run:
#   - imprime MODELO y MAPA que se esta probando,
#   - se etiqueta como <modelo>_w<N> en logs/nav_comparison.csv (fila por modelo y mapa),
#   - se auto-analiza al resolver (o al cortarse por timeout, que envia SIGINT y tambien analiza).
#
# Variables de entorno opcionales:
#   TIMEOUT=600 bash Robot/run_all_models.sh 1   # segundos maximos por run (def 420)
#   ONLY="fcn unet"  bash ...                     # correr solo esos labels base (def: todos)
#   SHOW=1 bash ...                               # abre la ventana de debug de pared (--show) en cada run
#   EXTRA="--wall-lower-frac 0.4" bash ...        # pasa argumentos extra a navigate.py en cada run
set -u

MAP="${1:?Uso: bash Robot/run_all_models.sh <N_mapa 1..5>  (con world_test<N> lanzado y en PLAY)}"
TIMEOUT="${TIMEOUT:-420}"
ONLY="${ONLY:-}"
# Franja central en ANCHO que ve el robot (recorta los laterales). Override: VIEW=X bash ...
# Historia (ver README §5/§9.1): 0.5 -> 0.66 -> 1.0 (completa) -> 0.6 (2026-06-11). 1.0 da mejor
# percepcion de pared pero re-expone a senales laterales; 0.6 acota los laterales conservando
# franja central amplia. 1.0 = sin recorte.
VIEW="${VIEW:-0.6}"
SHOW="${SHOW:-}"       # SHOW=1 -> añade --show (ventana de debug de pared) a cada run
EXTRA="${EXTRA:-}"     # argumentos extra que se pasan tal cual a navigate.py en cada run
WORLD="Moving_robot"   # los 5 world_test*.sdf comparten este nombre interno
[ -n "$SHOW" ] && EXTRA="$EXTRA --show"

# repo root = carpeta padre de este script (Robot/..)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

CUSTOM="Modelos/runs/compare_20260610_132232"   # mains de submodelos desde cero
TL="Modelos/runs/compare_20260610_135553"       # mains preentrenados

reset_world() {
  gz service -s "/world/${WORLD}/control" \
    --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'reset: {all: true}' >/dev/null 2>&1 \
    && echo "   [reset] mundo reiniciado" || echo "   [reset] AVISO: fallo el reset (¿nombre de world?)"
}

run() {
  local base="$1" rundir="$2" arrow="$3" thr="$4"
  if [ -n "$ONLY" ] && ! grep -qw "$base" <<<"$ONLY"; then return 0; fi
  echo ""
  echo "############################################################"
  echo "### MODELO = ${base}        MAPA = world_test${MAP}"
  echo "### main=${rundir}"
  echo "### arrow=${arrow}  thr=${thr}  view_w_frac=${VIEW}"
  echo "############################################################"
  reset_world
  sleep 2
  timeout -s INT "${TIMEOUT}" bash Robot/run_navigate.sh \
    --run-dir "$rundir" --arrow-run "$arrow" --arrow-conf-thresh "$thr" \
    --view-w-frac "$VIEW" $EXTRA \
    --analyze --analyze-label "${base}_w${MAP}"
  echo "### FIN ${base} en world_test${MAP}"
}

# ---- Familias (clasificador general + su specialist) ----
# Umbral de flecha UNIFORME 0.95 para los 11 modelos en este barrido grande (decision de
# desarrollo, ver README v6 y §9.1): igualar el bar hace la comparativa directa. El MLP se
# deja TAMBIEN en 0.95 (no en su 0.68 calibrado): que con el umbral comun no comprometa los
# giros de flecha es un RESULTADO a registrar, no un sesgo (su confianza vive mas abajo).
run cnn         Modelos/runs/cnn_unificado         Modelos/runs/cnn_arrows         0.95
run mlp         Modelos/runs/mlp_unificado         Modelos/runs/mlp_arrows         0.95
run rnn         Modelos/runs/rnn_unificado         Modelos/runs/rnn_arrows         0.95
run transformer Modelos/runs/transformer_unificado Modelos/runs/transformer_arrows 0.95
# ---- Submodelos CNN desde cero (paso 7) ----
run fcn    "$CUSTOM/fcn"    Modelos/runs/fcn_arrows    0.95
run unet   "$CUSTOM/unet"   Modelos/runs/unet_arrows   0.95
run yolo   "$CUSTOM/yolo"   Modelos/runs/yolo_arrows   0.95
run vgg    "$CUSTOM/vgg"    Modelos/runs/vgg_arrows    0.95
run resnet "$CUSTOM/resnet" Modelos/runs/resnet_arrows 0.95
# ---- Submodelos CNN preentrenados (necesitan torchvision+pillow en WSL) ----
run resnet_tl "$TL/resnet_tl" Modelos/runs/resnet_tl_arrows 0.95
run vgg_tl    "$TL/vgg_tl"    Modelos/runs/vgg_tl_arrows    0.95

echo ""
echo "============================================================"
echo "=== MAPA world_test${MAP} COMPLETADO ==="
echo "=== Lanza el siguiente mapa y repite: bash Robot/run_all_models.sh <N> ==="
echo "============================================================"
