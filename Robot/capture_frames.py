"""Captura de frames desde la camara del robot en Gazebo (lado WSL2).

Sirve para cerrar el "domain gap": graba lo que REALMENTE ve la camara en el
simulador (pasillos, flechas en angulo, etc.) para ampliar el dataset y
reentrenar. Asi el modelo aprende la distribucion real, no solo los recortes
manuales bien encuadrados.

Flujo recomendado:
  1. Lanza un mundo:           gz sim ../MundosPrueba/world1_robot_0_0.sdf  (dale a play)
  2. Conduce el robot a una vista clara de una clase (con move_robot.py, la GUI, etc.)
  3. Captura una rafaga etiquetada:
        python3 capture_frames.py --label FREE_PATH --max 30
     Los frames se guardan en  Modelos/dataset_live/FREE_PATH/
  4. Repite para cada clase/zona, sobre todo las debiles (FREE_PATH, ARROW_*, UNKNOWN).
  5. Luego mezclamos dataset_live con el dataset y reentrenamos.

Sin --label, guarda en  dataset_live/unsorted/  para ordenar a mano despues.
"""
import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path

# Necesario ANTES de importar gz.msgs (conflicto con protobuf nuevo en ~/.local)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

CLASSES = ["ARROW_LEFT", "ARROW_RIGHT", "ARROW_UP", "CROSS", "FREE_PATH",
           "GOAL", "UNKNOWN", "WALL", "WALL_LEFT", "WALL_RIGHT"]

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "Modelos" / "dataset_live"


class Capturer:
    def __init__(self, args):
        self.args = args
        label = args.label if args.label else "unsorted"
        self.out_dir = Path(args.out_dir) / label
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.latest = None
        self.frame_id = 0
        self.node = Node()
        self.node.subscribe(Image, args.image_topic, self._on_image)
        print(f"Suscrito a {args.image_topic}")
        print(f"Guardando en: {self.out_dir}")

    def _on_image(self, msg):
        w, h = msg.width, msg.height
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if buf.size == h * w * 3:                  # RGB_INT8
            with self.lock:
                self.latest = buf.reshape(h, w, 3).copy()
                self.frame_id += 1

    def run(self):
        period = 1.0 / self.args.rate
        saved, last_proc = 0, 0
        print(f"Capturando hasta {self.args.max} frames a <= {self.args.rate} Hz. "
              f"Ctrl+C para parar.\n")
        try:
            while saved < self.args.max:
                t0 = time.time()
                with self.lock:
                    fid, frame = self.frame_id, self.latest
                if frame is not None and fid != last_proc:
                    last_proc = fid
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fn = self.out_dir / f"live_{ts}.png"
                    # gz da RGB; cv2.imwrite espera BGR
                    cv2.imwrite(str(fn), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    saved += 1
                    print(f"[{saved}/{self.args.max}] {fn.name}", flush=True)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            pass
        print(f"\nGuardados {saved} frames en {self.out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Captura frames de la camara del robot")
    p.add_argument("--label", default=None, choices=CLASSES,
                   help="Clase de los frames (carpeta destino). Sin esto -> unsorted/")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--rate", type=float, default=2.0, help="Maximo de frames/s a guardar")
    p.add_argument("--max", type=int, default=30, help="Numero de frames a capturar")
    return p.parse_args()


if __name__ == "__main__":
    Capturer(parse_args()).run()
