#!/bin/bash
# Helpers reutilizables para lanzar/cerrar Gazebo desde un script (modo AUTO, SIN intervencion).
# Extraido de run_all_worlds.sh para que run_compare.sh lo comparta. Se usa con `source`:
#   source "$(dirname "$0")/gz_lib.sh"
#   gz_launch "MundosTest/world_test3.sdf"   # arranca en PLAY (gz sim -r), en background -> GZ_PID
#   gz_wait_ready || { gz_kill; exit 1; }    # espera a que publique /camera/image_raw
#   ... navegar ...
#   gz_kill                                  # cierra Gazebo y su descendencia
#
# Variables que honra: HEADLESS=1 (sin ventana, mas rapido).
# El que hace `source` deberia poner:  trap 'gz_kill' EXIT   (para no dejar Gazebo huerfano).

GZ_PID=""

gz_launch() {                      # $1 = ruta al .sdf del mundo
  local world="$1"
  local gzlog="/tmp/gz_$(basename "$world" .sdf).log"
  echo ">>> Lanzando Gazebo$([ -n "${HEADLESS:-}" ] && echo ' (headless)') en PLAY: ${world}"
  if [ -n "${HEADLESS:-}" ]; then
    gz sim -r -s "$world" >"$gzlog" 2>&1 </dev/null &
  else
    gz sim -r "$world" >"$gzlog" 2>&1 </dev/null &
  fi
  GZ_PID=$!
  GZ_LOG="$gzlog"
}

gz_wait_ready() {                  # espera (hasta ~90 s) a que aparezca /camera/image_raw
  local i
  for i in $(seq 1 90); do
    if gz topic -l 2>/dev/null | grep -q "/camera/image_raw"; then
      sleep 2                      # margen para que camara/LIDAR esten estables
      return 0
    fi
    if ! kill -0 "$GZ_PID" 2>/dev/null; then
      echo ">>> ERROR: Gazebo se cerro al arrancar (log: ${GZ_LOG:-?})."
      return 1
    fi
    sleep 1
  done
  echo ">>> AVISO: no se publico /camera/image_raw a tiempo (log: ${GZ_LOG:-?})."
  return 1
}

gz_kill() {                        # cierra el Gazebo lanzado por gz_launch (y su descendencia)
  if [ -n "${GZ_PID:-}" ]; then
    pkill -TERM -P "$GZ_PID" 2>/dev/null
    kill -TERM "$GZ_PID" 2>/dev/null
    sleep 2
    pkill -KILL -P "$GZ_PID" 2>/dev/null
    kill -KILL "$GZ_PID" 2>/dev/null
  fi
  pkill -f "gz sim" 2>/dev/null    # red de seguridad (servidor/gui sueltos)
  GZ_PID=""
  sleep 1
}
