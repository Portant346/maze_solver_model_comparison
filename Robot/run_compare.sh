#!/bin/bash
# Dispatcher UNICO de los modos de ejecucion de la comparativa de navegacion (TFG).
# Un solo punto de entrada con 4 modos; reutiliza por debajo run_all_models.sh, run_all_worlds.sh
# y run_navigate.sh, y AUTO-LANZA Gazebo (gz_lib.sh) para que todo corra desatendido.
#
# Uso (WSL2, UNA terminal):
#   bash Robot/run_compare.sh all-models <N>          # los 11 modelos (4 familias + 7 submodelos CNN) en el mundo N
#   bash Robot/run_compare.sh families  <N>           # solo las 4 familias (cnn/mlp/rnn/transformer) en el mundo N
#   bash Robot/run_compare.sh model-all-maps <modelo> # un modelo en los mundos 1..5
#   bash Robot/run_compare.sh model-map <modelo> <N>  # un modelo en el mundo N
#
#   modelo ∈ cnn|mlp|rnn|transformer|fcn|unet|yolo|vgg|resnet|resnet_tl|vgg_tl ;  N ∈ 1..5
#   Al terminar regenera los mapas semanticos y recuerda como comparar.
#
# Variables opcionales (se pasan a los scripts delegados):
#   HEADLESS=1   Gazebo sin ventana (mas rapido)        WALL_SOURCE=camera  detector de pared por camara
#   MANUAL=1     no auto-lanzar Gazebo (tu lo lanzas)    VIEW=1.0  TIMEOUT=600  EXTRA="--show-semantic"
#   SEMANTIC=0   no regenerar los mapas semanticos al final
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
SELF="$(dirname "$0")"

MODELS="cnn mlp rnn transformer fcn unet yolo vgg resnet resnet_tl vgg_tl"
FAMILIES="cnn mlp rnn transformer"

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

is_model() { grep -qw "$1" <<<"$MODELS"; }
is_world() { [[ "$1" =~ ^[1-5]$ ]]; }

regen_semantic() {
  if [ "${SEMANTIC:-1}" != "0" ]; then
    echo ""
    echo ">>> Regenerando mapas semanticos (path coloreado por señal)..."
    python Robot/semantic_map.py --logs-dir logs \
      || echo ">>> (aviso: no se pudieron regenerar los mapas semanticos)"
  fi
}

final_hint() {
  echo ""
  echo "============================================================"
  echo "=== HECHO. Compara:   python Modelos/src/analyze_log.py --compare"
  echo "=== Mapas semanticos: logs/semantic_map_*.png  +  logs/semantic_mosaic_w*.png"
  echo "============================================================"
}

# corre run_all_models.sh en el mundo N (auto-lanza Gazebo salvo MANUAL=1). $1=N, $2=ONLY (opcional)
run_models_on_map() {
  local N="$1" only="${2:-}"
  if [ -n "${MANUAL:-}" ]; then
    echo ">>> MANUAL: lanza 'gz sim MundosTest/world_test${N}.sdf' y dale a PLAY; luego Enter."
    read -r -p ">>> Enter cuando el mundo ${N} este en PLAY... " _ </dev/tty
    ONLY="$only" bash Robot/run_all_models.sh "$N"
    return $?
  fi
  # shellcheck disable=SC1090
  source "$SELF/gz_lib.sh"
  trap 'gz_kill' EXIT
  gz_launch "MundosTest/world_test${N}.sdf"
  if ! gz_wait_ready; then gz_kill; trap - EXIT; return 1; fi
  ONLY="$only" bash Robot/run_all_models.sh "$N"
  local rc=$?
  gz_kill; trap - EXIT
  return $rc
}

MODE="${1:-}"; shift || true
case "$MODE" in
  all-models)
    N="${1:-}"; is_world "$N" || { echo "N de mundo invalido (1..5): '${N:-}'"; usage; }
    echo "=== MODO all-models · 11 modelos en el mundo ${N} ==="
    run_models_on_map "$N" ""
    regen_semantic; final_hint
    ;;
  families)
    N="${1:-}"; is_world "$N" || { echo "N de mundo invalido (1..5): '${N:-}'"; usage; }
    echo "=== MODO families · 4 familias en el mundo ${N} ==="
    run_models_on_map "$N" "$FAMILIES"
    regen_semantic; final_hint
    ;;
  model-all-maps)
    M="${1:-}"; is_model "$M" || { echo "Modelo desconocido: '${M:-}'"; usage; }
    echo "=== MODO model-all-maps · ${M} en los mundos 1..5 ==="
    bash Robot/run_all_worlds.sh "$M"
    regen_semantic; final_hint
    ;;
  model-map)
    M="${1:-}"; N="${2:-}"
    is_model "$M" || { echo "Modelo desconocido: '${M:-}'"; usage; }
    is_world "$N" || { echo "N de mundo invalido (1..5): '${N:-}'"; usage; }
    echo "=== MODO model-map · ${M} en el mundo ${N} ==="
    WORLDS="$N" bash Robot/run_all_worlds.sh "$M"
    regen_semantic; final_hint
    ;;
  -h|--help|help|"") usage 0 ;;
  *) echo "Modo desconocido: '${MODE}'"; usage ;;
esac
