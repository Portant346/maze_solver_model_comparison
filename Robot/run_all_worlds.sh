#!/bin/bash
# Lanza UN modelo en TODOS los mapas. Lo inverso de run_all_models.sh (todos los modelos en un mapa).
#
# MODO AUTOMATICO (por defecto): UNA sola terminal, SIN intervencion manual. Por cada mapa el script
# lanza Gazebo el solo (gz sim -r = arranca ya en PLAY), navega, y cierra Gazebo antes del siguiente.
# Cada mapa tiene un TIMEOUT de seguridad por si se cuelga. Tu solo intervienes con Ctrl+C si ves que
# un mapa se ha bloqueado o no va a resolver:
#   - 1x Ctrl+C  -> corta ESE mapa (se analiza igual = se registra el fallo) y pasa al siguiente,
#   - 2x Ctrl+C (pulsa, espera el mensaje "cortando", pulsa otra vez) -> aborta TODO el barrido.
#
# MODO MANUAL (MANUAL=1): tu lanzas gz sim en otra terminal y el script pausa pidiendo Enter entre mapas.
#
# Uso (WSL2, UNA terminal):
#   bash Robot/run_all_worlds.sh cnn        # cnn por LIDAR en los 5 mapas, automatico
#   Repite con otro modelo:  bash Robot/run_all_worlds.sh mlp   (mlp usa thr de flecha 0.68; resto 0.95)
#   Al final:  python Modelos/src/analyze_log.py --compare
#
# Variables opcionales:
#   WALL_SOURCE=camera bash ...   # detector de pared por camara (def: lidar)
#   WORLDS="1 3 5"     bash ...   # subconjunto/orden de mapas (def: 1 2 3 4 5)
#   TIMEOUT=600        bash ...   # segundos maximos por mapa (def 420)
#   HEADLESS=1         bash ...   # Gazebo SIN ventana (mas rapido, pero no lo ves)
#   MANUAL=1           bash ...   # modo manual (tu lanzas gz y pulsas Enter entre mapas)
#   VIEW=1.0 / SHOWMAP=1 / EXTRA="--show"   bash ...
set -u

MODEL="${1:?Uso: bash Robot/run_all_worlds.sh <modelo>  (cnn|mlp|rnn|transformer|fcn|unet|yolo|vgg|resnet|resnet_tl|vgg_tl)}"
WALL_SOURCE="${WALL_SOURCE:-lidar}"
WORLDS="${WORLDS:-1 2 3 4 5}"
TIMEOUT="${TIMEOUT:-420}"
VIEW="${VIEW:-0.6}"
MANUAL="${MANUAL:-}"
HEADLESS="${HEADLESS:-}"
SHOWMAP="${SHOWMAP:-}"
EXTRA="${EXTRA:-}"
[ -n "$SHOWMAP" ] && EXTRA="$EXTRA --show-map"

# repo root = carpeta padre de este script (Robot/..)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

CUSTOM="Modelos/runs/compare_20260610_132232"   # mains de submodelos desde cero (paso 7)
TL="Modelos/runs/compare_20260610_135553"       # mains preentrenados (transfer learning)

# modelo -> (main, specialist de flechas, umbral de flecha). Umbral por FAMILIA: mlp 0.68; resto 0.95.
case "$MODEL" in
  cnn)         RUNDIR="Modelos/runs/cnn_unificado";         ARROW="Modelos/runs/cnn_arrows";         THR=0.95 ;;
  mlp)         RUNDIR="Modelos/runs/mlp_unificado";         ARROW="Modelos/runs/mlp_arrows";         THR=0.68 ;;
  rnn)         RUNDIR="Modelos/runs/rnn_unificado";         ARROW="Modelos/runs/rnn_arrows";         THR=0.95 ;;
  transformer) RUNDIR="Modelos/runs/transformer_unificado"; ARROW="Modelos/runs/transformer_arrows"; THR=0.95 ;;
  fcn)         RUNDIR="$CUSTOM/fcn";    ARROW="Modelos/runs/fcn_arrows";    THR=0.95 ;;
  unet)        RUNDIR="$CUSTOM/unet";   ARROW="Modelos/runs/unet_arrows";   THR=0.95 ;;
  yolo)        RUNDIR="$CUSTOM/yolo";   ARROW="Modelos/runs/yolo_arrows";   THR=0.95 ;;
  vgg)         RUNDIR="$CUSTOM/vgg";    ARROW="Modelos/runs/vgg_arrows";    THR=0.95 ;;
  resnet)      RUNDIR="$CUSTOM/resnet"; ARROW="Modelos/runs/resnet_arrows"; THR=0.95 ;;
  resnet_tl)   RUNDIR="$TL/resnet_tl";  ARROW="Modelos/runs/resnet_tl_arrows"; THR=0.95 ;;
  vgg_tl)      RUNDIR="$TL/vgg_tl";     ARROW="Modelos/runs/vgg_tl_arrows";    THR=0.95 ;;
  *) echo "Modelo desconocido: '$MODEL'. Opciones: cnn mlp rnn transformer fcn unet yolo vgg resnet resnet_tl vgg_tl"; exit 1 ;;
esac

navigate_map() {   # navega el mapa $1 (Gazebo ya debe estar publicando)
  local N="$1"
  echo ">>> Navegando en el mundo ${N} (TIMEOUT ${TIMEOUT}s; Ctrl+C para cortar este mapa)..."
  timeout -s INT "${TIMEOUT}" bash Robot/run_navigate.sh \
    --run-dir "$RUNDIR" --arrow-run "$ARROW" --arrow-conf-thresh "$THR" \
    --view-w-frac "$VIEW" --wall-source "$WALL_SOURCE" $EXTRA \
    --analyze --analyze-label "${MODEL}_${WALL_SOURCE}_w${N}"
}

# ------------------------------------------------------------------ #
echo "============================================================"
echo "=== BARRIDO DE MAPAS  ·  modelo=${MODEL}  fuente_pared=${WALL_SOURCE}  modo=$([ -n "$MANUAL" ] && echo MANUAL || echo AUTO)"
echo "=== mapas=[${WORLDS}]  view=${VIEW}  timeout=${TIMEOUT}s  thr_flecha=${THR}"
echo "=== main=${RUNDIR}  arrow=${ARROW}"
echo "============================================================"

# ============================ MODO MANUAL ============================ #
if [ -n "$MANUAL" ]; then
  for N in $WORLDS; do
    WORLD="MundosTest/world_test${N}.sdf"
    echo ""
    echo "############################################################"
    echo "### ${MODEL} (${WALL_SOURCE})  ·  MUNDO ${N}   (${WORLD})"
    echo "############################################################"
    echo ">>> En la TERMINAL DE GAZEBO: cierra el mundo anterior, lanza 'gz sim ${WORLD}' y dale a PLAY (▶)."
    read -r -p ">>> Cuando el MUNDO ${N} este en PLAY, pulsa Enter para navegar... " _ </dev/tty
    if ! gz topic -l 2>/dev/null | grep -q "/camera/image_raw"; then
      echo ">>> AVISO: no veo /camera/image_raw. ¿Lanzaste el mundo y diste a PLAY?"
      read -r -p ">>> Enter para intentarlo igual, o Ctrl+C para abortar... " _ </dev/tty
    fi
    navigate_map "$N"
    echo "### FIN mundo ${N}."
  done
  echo ""
  echo "=== COMPLETADOS los mapas [${WORLDS}] con ${MODEL} (${WALL_SOURCE}). Compara: python Modelos/src/analyze_log.py --compare"
  exit 0
fi

# ============================ MODO AUTO ============================ #
GZ_PID=""
kill_gz() {                       # cierra el Gazebo lanzado por el script (y su descendencia)
  if [ -n "${GZ_PID:-}" ]; then
    pkill -TERM -P "$GZ_PID" 2>/dev/null
    kill -TERM "$GZ_PID" 2>/dev/null
    sleep 2
    pkill -KILL -P "$GZ_PID" 2>/dev/null
    kill -KILL "$GZ_PID" 2>/dev/null
  fi
  pkill -f "gz sim" 2>/dev/null   # red de seguridad (servidor/gui que queden sueltos)
  GZ_PID=""
  sleep 1
}
trap 'kill_gz' EXIT               # nunca dejar Gazebo huerfano si el script termina

LAST_INT=-100
on_int() {                        # Ctrl+C: 1x corta el mapa (sigue), 2x (<3s) aborta todo
  local now=$SECONDS
  if [ $((now - LAST_INT)) -le 3 ]; then
    echo ""; echo ">>> Ctrl+C x2 -> ABORTANDO todo el barrido."
    kill_gz; trap - INT EXIT; exit 130
  fi
  LAST_INT=$now
  echo ""; echo ">>> (Ctrl+C) cortando el mapa actual; pulsa Ctrl+C OTRA VEZ en <3s para abortar TODO."
}
trap 'on_int' INT

for N in $WORLDS; do
  WORLD="MundosTest/world_test${N}.sdf"
  GZLOG="/tmp/gz_world_test${N}.log"
  echo ""
  echo "############################################################"
  echo "### ${MODEL} (${WALL_SOURCE})  ·  MUNDO ${N}   (${WORLD})"
  echo "############################################################"
  echo ">>> Lanzando Gazebo$([ -n "$HEADLESS" ] && echo ' (headless)') en PLAY..."
  if [ -n "$HEADLESS" ]; then
    gz sim -r -s "$WORLD" >"$GZLOG" 2>&1 </dev/null &
  else
    gz sim -r "$WORLD" >"$GZLOG" 2>&1 </dev/null &
  fi
  GZ_PID=$!

  # esperar a que el mundo publique la camara (señal de que el servidor esta listo)
  ok=""
  for _i in $(seq 1 90); do
    if gz topic -l 2>/dev/null | grep -q "/camera/image_raw"; then ok=1; break; fi
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
      echo ">>> ERROR: Gazebo se cerro al arrancar (log: ${GZLOG}). Salto el mapa ${N}."; break
    fi
    sleep 1
  done
  if [ -z "$ok" ]; then
    echo ">>> AVISO: el mundo ${N} no publico /camera/image_raw a tiempo (log: ${GZLOG}). Salto el mapa."
    kill_gz; continue
  fi
  sleep 2                         # margen para que camara/LIDAR esten estables

  navigate_map "$N"

  echo "### FIN mundo ${N}. Cerrando Gazebo..."
  kill_gz
done

trap - INT
echo ""
echo "============================================================"
echo "=== COMPLETADOS los mapas [${WORLDS}] con ${MODEL} (${WALL_SOURCE}) ==="
echo "=== Tabla comparativa:   python Modelos/src/analyze_log.py --compare ==="
echo "============================================================"
