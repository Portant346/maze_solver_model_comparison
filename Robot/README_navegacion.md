# Navegación del robot por clasificación de imágenes

Controlador en tiempo real que resuelve el laberinto en Gazebo:
**cámara → CNN (`cnn_unificado`) → máquina de estados → `cmd_vel`**.

Se ejecuta en **WSL2/Ubuntu** junto a Gazebo Harmonic.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `cnn_infer.py` | Carga el modelo y clasifica frames RGB (preprocesado con cv2, igual que el entrenamiento). |
| `navigate.py` | Bucle de control: suscribe la cámara, decide la acción y publica `cmd_vel`. |
| `run_navigate.sh` | Lanzador (exporta el workaround de protobuf necesario para gz.msgs). |

## Cómo probarlo

**1. Lanza un mundo** (terminal 1, en WSL2):
```bash
cd Robot
gz sim ../MundosTest/world_test1.sdf
```
Dale al play (▶) en la GUI de Gazebo.

**2. Comprueba los nombres de los topics** (terminal 2):
```bash
gz topic -l
```
Busca el de la cámara (esperado `/camera/image_raw`) y el de movimiento
(`/cmd_vel`). Si difieren, pásalos con `--image-topic` / `--cmd-topic`.

**3. Primero en seco** (ver decisiones SIN mover el robot):
```bash
bash run_navigate.sh --dry-run
```
Verás por consola la clase detectada, la confianza y la acción que tomaría.

**4. Navegación real**:
```bash
bash run_navigate.sh
```

## Política de navegación

| Clase | Acción |
|---|---|
| `GOAL` | dirigirse hacia la diana (azul) y parar solo al **llegar** (cerca y centrada) |
| `CROSS` | media vuelta |
| `ARROW_LEFT` / `RIGHT` | girar a ese lado |
| `ARROW_UP` / `FREE_PATH` | avanzar recto |
| `WALL` | girar al **lado abierto** (detector geométrico de negro por franjas); si no es claro, hacia la última flecha vista o derecha |
| `WALL_LEFT` / `RIGHT` | avanzar corrigiendo al lado contrario (el modelo en vivo casi no las emite; la dirección de giro la decide el detector geométrico) |
| baja confianza / `UNKNOWN` | girar despacio buscando una señal |

Giro: `angular.z > 0` = izquierda, `< 0` = derecha (igual que `move_robot.py`).

## Parámetros (tuning)

```
--rate 10              Hz del bucle de control (cámara a 10 Hz desde nav-v6)
--vote 3               voto mayoritario sobre N frames (devuelve la conf media de la clase ganadora)
--conf-thresh 0.4      umbral de confianza para ACTUAR (avanzar/corregir)
--sign-conf-thresh 0.55 umbral MAYOR para COMPROMETER una direccional (flecha/cruz); filtra ruido L/R ~0.5
--cross-interrupt-conf 0.85 conf mínima para que una CRUZ interrumpa un giro en curso y lo reconvierta a 180°
--pending-lost-frames 3 frames seguidos sin ver el cartel para girar una flecha pendiente (al perderla de vista)
--forward 0.3          velocidad lineal al avanzar
--turn 0.6             velocidad angular al girar en sitio
--steer 0.3            corrección angular al esquivar pared lateral
--search 0.4           velocidad angular al buscar (UNKNOWN persistente)
--crop-frac 1.0        recorte central del frame en vivo (p.ej. 0.6); 1.0 = sin recorte
                       (filtro de visión: ignora señales en los bordes; solo en ejecución)
--arrow-deg 90         grados a girar con una flecha (NOMINAL/fallback si no se usa el vector)
--arrow-vector         ángulo de giro de flecha por su VECTOR (verde→PCA, magnitud adaptativa; on por defecto; --no-arrow-vector = fijo)
--arrow-min-deg 25     clamp inferior del giro de flecha por vector
--arrow-max-deg 135    clamp superior del giro de flecha por vector
--cross-deg 180        grados a girar con una cruz
--wall-deg 90          grados a girar al toparse una pared
--turn-tol-deg 20      margen EXTRA por encima del objetivo: gira de obj a obj+tol (nunca menos), completa al despejar
--turn-stop-on-open    parar el giro al abrirse un pasillo por delante (giro adaptativo; on por defecto)
--turn-open-thresh 0.10 negro del centro por debajo del cual el frente se considera despejado (más bajo = más alineado)
--turn-open-drop 0.20  caída mínima del negro central vs inicio del giro para considerar que se abre un pasillo nuevo
--turn-min-open-deg 20 giro mínimo antes de permitir parar por apertura
--post-turn-s 1.5      enfriamiento tras girar (ignora señales para alejarse del cartel)
--wall-probe-s 2.0     ante una pared sin intención, avanza despacio buscando señal N s
--wall-probe-speed 0.3 fracción de la velocidad de avance durante el sondeo de pared
--turn-scale 1.0       calibración del giro por TIEMPO (solo fallback sin odometría)
--odom-turn            giros por ÁNGULO REAL medido con odometría (on por defecto; --no-odom-turn = por tiempo)
--odom-topic /model/vehicle_blue/odometry   topic gz de odometría del robot
--wall-dir             detector geométrico de dirección de pared (on por defecto; --no-wall-dir para apagar)
--wall-dark-thresh 50  brillo (0-255) por debajo del cual un píxel cuenta como pared/negro
--wall-center-block 0.5 fracción de negro en la franja central para considerar pared de frente
--goal-homing          dirigirse hacia la diana en vez de parar al verla (on por defecto; --no-goal-homing = parar al verla)
--goal-reach-frac 0.08 fracción de azul para dar la META por alcanzada (calíbralo con el area% del log)
--goal-center-tol 0.30 |offset| máximo de la diana para considerarla centrada
--log                  guarda un CSV por ejecución (on por defecto; --no-log para desactivar)
--log-dir <ruta>       carpeta de los logs (def: TFG_v2/logs)
```

**Log de ejecución**: cada run guarda `logs/nav_<fecha>.csv` con una fila por decisión
(`t, frame, cls, conf, state, lin, ang, yaw_deg, pos_x, pos_y, arrow_ang, open_side,
dark_l/c/r, goal_off, goal_area, note`). Sirve para analizar el histórico del clasificador
y reconstruir la trayectoria/giros desde la odometría (`pos_x,pos_y,yaw_deg`) y cotejarla con
los mapas de Gazebo. Se escribe con flush por fila (no se pierde si hay segfault al salir).

**Diana / META (nav-v11)**: la diana se detecta por su color AZUL (`goal_in_view`): centroide
(hacia dónde girar) y área (cercanía). El robot **se dirige** hacia ella —si está a un lado gira
sin avanzar, si está centrada avanza, y si hay una pared negra de por medio la esquiva— y **solo
para al llegar** (azul ≥ `--goal-reach-frac` y centrado). Así una falsa detección de GOAL no
termina la ejecución. Requiere que la CNN diga GOAL y que haya azul.

**Detector geométrico de pared (nav-v8)**: al margen del clasificador, divide el frame en
3 franjas (izq/centro/der) y mide el % de píxeles negros (pared cercana ≈ negra; suelo gris).
Si el centro está bloqueado, gira hacia el lado menos negro (más abierto). Solo decide el
giro de una pared **sin señal**; las flechas/cruz tienen prioridad. El log muestra
`negro LCR=dl/dc/dr` en frames de pared para calibrar el umbral.

Los giros se completan por **ángulo REAL medido con odometría** (nav-v9): el robot gira
**al menos el ángulo objetivo** (`--arrow-deg` 90, `--cross-deg` 180, `--wall-deg` 90) y
sigue **hasta que el pasillo de delante queda despejado**, con tope `objetivo + --turn-tol-deg`.
Ventana `[obj, obj+tol]` → **nunca gira de menos** (una flecha siempre ≥90°). Al medir el yaw
real, es **independiente del RTF** de Gazebo. En el log verás el progreso `girando … 47/90°`.
Si no hay odometría (`--no-odom-turn` o sin binding), cae a giro por **tiempo**
(`t = ángulo / --turn`, asume RTF≈1) y entonces `--turn-scale` compensa (≈ 1/RTF); el log
muestra `(sin odom)`.

**Intención diferida (flechas, nav-v7)**: al ver una flecha NO gira al instante; memoriza
la dirección, se **acerca mientras ve el cartel** y gira **en cuanto lo pierde de vista**
(`--pending-lost-frames`), o antes si topa una pared (vía rápida). Así también gira con
flechas "flotando" sin pared detrás. La `CROSS` no se difiere: gira 180° en el sitio al
verla con confianza, y una `CROSS` muy confiable vista a mitad de un giro lo reconvierte a
180° (giro interrumpible).

**Giro adaptativo: parar al abrirse el pasillo (nav-v12)**: durante un giro, el robot **para
en cuanto el pasillo de delante se despeja** (el negro central baja respecto al inicio del giro
y queda < `--turn-open-thresh`), en vez de fiarse de un ángulo fijo. Así un cruce que necesita
~45° no se convierte en 90°/120°. Distingue: si el frente **se abre** durante el giro → para
ahí; si **ya estaba abierto** al empezar (la flecha te redirige) → hace el ángulo nominal. La
CRUZ queda excluida (media vuelta de 180°). La odometría mide el ángulo real. El vector de la
flecha (nav-v10) y `--arrow-deg` quedan como objetivo máximo/fallback.

**Ángulo de flecha por su vector (nav-v10)**: la **dirección** (izq/der) la decide la CNN; el
vector verde (PCA, punta = franja más ancha) aporta una **magnitud objetivo** cuando concuerda
en signo (si no, nominal `--arrow-deg`). OJO: el vector mide la dirección *dibujada* de la
flecha, no el ángulo al pasillo, por eso nav-v12 lo complementa parando al despejarse el frente.

## Tasa de la cámara (ya a 10 Hz desde nav-v6)

La cámara está a **10 Hz**. ⚠️ El robot está **embebido en cada mundo**, así que el
`<update_rate>` vive en cada `MundosTest/*.sdf` y `MundosPrueba/*.sdf` (8 ficheros), no en
`Robot/moving_robot_camera.sdf` (esa es solo la copia de referencia y no la que carga
`gz sim`). Si cambias la tasa, hazlo en los mundos y ajusta `--vote` en consecuencia.
