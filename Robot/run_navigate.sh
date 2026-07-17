#!/bin/bash
# Lanzador del controlador de navegacion (lado WSL2/Ubuntu).
# Exporta el workaround de protobuf necesario para las bindings de gz.msgs.
# Lanza primero el mundo en otra terminal:  gz sim ../MundosTest/world_test1.sdf
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 navigate.py "$@"
