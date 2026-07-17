"""Registro de modelos del TFG.

Todos los modelos comparten la misma firma de constructor
    __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
             img_h=..., img_w=..., **kwargs)
para que el resto del pipeline (train.py, engine.py, sweep.py, compare_models.py)
los trate de forma intercambiable: basta con definir la clase aqui y registrarla
en MODEL_REGISTRY.

`base_channels` y `dropout` son los dos hiperparametros que se barren en sweep.py;
cada modelo los interpreta segun su arquitectura (canales de conv, ancho de las
capas densas, tamano de estado oculto, dimension del modelo del transformer).
`img_h`/`img_w` los inyecta train.py para los modelos que necesitan conocer el
tamano de entrada (MLP, RNN, transformer); la CNN los ignora (es convolucional y
usa AdaptiveAvgPool).

Tipos incluidos (objetivo del TFG: comparar familias de modelo sobre el mismo
dataset de 10 clases):
  - cnn         SimpleCNN  -> referencia convolucional (explota estructura 2D)
  - mlp         MLP        -> perceptron multicapa sobre el pixel aplanado
  - rnn         ImageRNN   -> GRU bidireccional leyendo la imagen por filas
  - transformer ImageTransformer -> encoder transformer con filas como tokens

Arquitecturas CNN comparadas EN CLASIFICACION (paso 7 del README). Las que en
origen NO son clasificadores (deteccion/segmentacion) se reducen aqui a su parte
clasificadora (backbone/encoder) + una cabeza de 10 clases, para compararlas en
igualdad sobre la MISMA tarea. Se documenta como tal (no es "YOLO detectando"):
  - vgg     VGGStyle    -> bloques conv 3x3 apilados + cabeza FC (receta VGG).
  - resnet  ResNetStyle -> bloques residuales con atajos (skip connections).
  - yolo    DarknetStyle-> backbone tipo Darknet (el de YOLO) + GAP (sin cabeza de deteccion).
  - fcn     FCNStyle    -> totalmente convolucional, SIN FC: clasifica por GAP del mapa de clases.
  - unet    UNetEncoder -> solo el encoder (camino contractivo) de U-Net + GAP (sin decoder).
  (R-CNN se omite: es un pipeline de deteccion por regiones, sin forma propia de clasificador.)
Variantes PREENTRENADAS (transfer learning, pesos ImageNet de torchvision), como
"techo" de lo alcanzable; NO comparables apples-to-apples con las de arriba (parten
con ventaja). Requieren descargar pesos la primera vez:
  - vgg_tl     VGGPretrained    -> torchvision vgg11_bn preentrenada, cabeza reemplazada.
  - resnet_tl  ResNetPretrained -> torchvision resnet18 preentrenada, cabeza reemplazada.
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN de referencia (baseline)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """CNN de referencia (baseline).

    Bloques Conv-BN-ReLU-MaxPool apilados + cabeza totalmente conectada.
    Se usa AdaptiveAvgPool para ser robusto al tamano exacto de entrada
    (por eso ignora img_h/img_w).

    base_channels y dropout se exponen como hiperparametros para el barrido.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 **kwargs):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, c1),   # /2
            block(c1, c2),            # /4
            block(c2, c3),            # /8
            block(c3, c3),            # /16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# MLP (perceptron multicapa) sobre el pixel aplanado
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Perceptron multicapa: aplana la imagen e ignora la estructura espacial.

    Sirve de contraste con la CNN: muchos parametros (la primera capa densa
    conecta cada pixel con cada neurona) y sin sesgo inductivo de localidad.

    base_channels controla el ancho (h1 = base_channels * hidden_mult);
    usamos LayerNorm (no BatchNorm) para no depender del tamano del batch.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 img_h=100, img_w=160, hidden_mult=16, **kwargs):
        super().__init__()
        in_dim = in_channels * img_h * img_w
        h1 = base_channels * hidden_mult
        h2 = h1 // 2
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# RNN: GRU bidireccional leyendo la imagen como secuencia de filas
# ---------------------------------------------------------------------------
class ImageRNN(nn.Module):
    """Lee la imagen de arriba a abajo: cada fila es un paso temporal.

    Cada fila aporta in_channels*img_w caracteristicas. Una GRU bidireccional
    recorre las img_h filas y se hace pooling temporal (media) de las salidas
    antes de la cabeza lineal. Captura dependencias verticales en la escena.

    base_channels controla el tamano del estado oculto (hidden = base_channels*4).
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 img_h=100, img_w=160, num_layers=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        input_size = in_channels * img_w
        hidden = base_channels * 4
        self.rnn = nn.GRU(
            input_size, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),   # *2 por bidireccional
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, H, C*W): cada fila (timestep) lleva todos sus canales
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        out, _ = self.rnn(x)
        feat = out.mean(dim=1)            # pooling temporal sobre las filas
        return self.head(feat)


# ---------------------------------------------------------------------------
# Transformer: encoder con cada fila de la imagen como token
# ---------------------------------------------------------------------------
def _sinusoidal_pos(n, d):
    """Codificacion posicional sinusoidal (n_tokens, d_model)."""
    pos = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(0, d, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ImageTransformer(nn.Module):
    """Encoder transformer tratando cada fila de la imagen como un token.

    Cada fila (in_channels*img_w) se proyecta a d_model, se le suma una
    codificacion posicional sinusoidal y pasa por un TransformerEncoder.
    Se hace media de los tokens (LayerNorm) antes de la cabeza lineal.

    base_channels controla la dimension del modelo (d_model = base_channels*4),
    que debe ser divisible por nhead (con base_channels=32 -> d_model=128).
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 img_h=100, img_w=160, nhead=4, num_layers=3, **kwargs):
        super().__init__()
        d_model = base_channels * 4
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} (base_channels*4) no es divisible por nhead={nhead}"
            )
        self.proj = nn.Linear(in_channels * img_w, d_model)
        # buffer no persistente: depende solo de (img_h, d_model), se reconstruye
        self.register_buffer("pos", _sinusoidal_pos(img_h, d_model), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        x = self.proj(x) + self.pos[:h].unsqueeze(0)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))      # pooling de tokens
        return self.head(x)


# ===========================================================================
# Arquitecturas CNN comparadas en CLASIFICACION (paso 7)
# Todas comparten la firma estandar e ignoran img_h/img_w (son convolucionales
# + AdaptiveAvgPool, robustas al tamano de entrada). base_channels escala el
# ancho; dropout regulariza la cabeza.
# ===========================================================================

class VGGStyle(nn.Module):
    """VGG: profundidad con filtros pequenos (3x3) y una cabeza FC grande.

    Bloques de 2-3 convoluciones 3x3 (Conv-BN-ReLU) seguidos de MaxPool, con los
    canales creciendo base->2x->4x->8x, y una cabeza densa de 2 capas (lo
    caracteristico de VGG: el grueso de los parametros vive en las FC).
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)

        def vgg_block(cin, cout, n):
            layers = []
            for k in range(n):
                layers += [nn.Conv2d(cin if k == 0 else cout, cout, 3, padding=1),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            vgg_block(in_channels, c1, 2),   # /2
            vgg_block(c1, c2, 2),            # /4
            vgg_block(c2, c3, 3),            # /8
            vgg_block(c3, c4, 3),            # /16
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 * 2 * 2, c4 * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(c4 * 4, c4 * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(c4 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class _BasicBlock(nn.Module):
    """Bloque residual de ResNet: dos conv 3x3 + atajo (identidad o proyeccion 1x1)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return torch.relu(out)


class ResNetStyle(nn.Module):
    """ResNet: bloques residuales con atajos (skip connections).

    Estructura tipo ResNet-18 (stem + 4 etapas de 2 bloques basicos, duplicando
    canales y reduciendo resolucion por stride). El atajo deja pasar el gradiente
    y permite redes profundas sin degradacion. GAP + cabeza lineal.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(_BasicBlock(c1, c1), _BasicBlock(c1, c1))
        self.layer2 = nn.Sequential(_BasicBlock(c1, c2, stride=2), _BasicBlock(c2, c2))
        self.layer3 = nn.Sequential(_BasicBlock(c2, c3, stride=2), _BasicBlock(c3, c3))
        self.layer4 = nn.Sequential(_BasicBlock(c3, c4, stride=2), _BasicBlock(c4, c4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x))


class DarknetStyle(nn.Module):
    """Backbone tipo Darknet (el de YOLO) usado SOLO como clasificador.

    Alterna conv 3x3 que expande canales y conv 1x1 que los comprime (BN +
    LeakyReLU, la activacion caracteristica de Darknet), con MaxPool entre etapas.
    Como Darknet-19, clasifica con una conv 1x1 a num_classes y global average
    pooling -> NO lleva la cabeza de deteccion (cajas/anclas) de YOLO.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()

        def cbl(cin, cout, k):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                nn.BatchNorm2d(cout), nn.LeakyReLU(0.1, inplace=True),
            )

        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        self.features = nn.Sequential(
            cbl(in_channels, c1, 3), nn.MaxPool2d(2),                  # /2
            cbl(c1, c2, 3), nn.MaxPool2d(2),                          # /4
            cbl(c2, c3, 3), cbl(c3, c2, 1), cbl(c2, c3, 3), nn.MaxPool2d(2),  # /8
            cbl(c3, c4, 3), cbl(c4, c3, 1), cbl(c3, c4, 3), nn.MaxPool2d(2),  # /16
        )
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c4, num_classes, 1)   # 1x1 a clases (estilo Darknet-19)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(self.drop(x))
        return self.pool(x).flatten(1)


class FCNStyle(nn.Module):
    """FCN: red totalmente convolucional, SIN capas densas.

    En vez de una cabeza FC, una conv 1x1 produce un mapa de puntuaciones por
    clase (B, num_classes, h, w) y se hace global average pooling para obtener un
    logit por clase (la idea de FCN/Network-in-Network llevada a clasificacion).
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, c1), block(c1, c2), block(c2, c3), block(c3, c4),
        )
        self.score = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(c4, num_classes, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.score(x)                 # mapa de clases
        return self.pool(x).flatten(1)    # GAP -> un logit por clase


class UNetEncoder(nn.Module):
    """U-Net reducida a clasificacion: solo el camino CONTRACTIVO (encoder).

    Bloques doble-conv 3x3 + downsample (MaxPool) y un bottleneck, como el encoder
    de U-Net; despues global average pooling + cabeza lineal. El decoder (la
    reconstruccion pixel a pixel y los skips al decoder) se omite porque no aporta
    a clasificar la imagen entera.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, **kwargs):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)

        def double_conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.enc1 = double_conv(in_channels, c1)
        self.enc2 = double_conv(c1, c2)
        self.enc3 = double_conv(c2, c3)
        self.bottleneck = double_conv(c3, c4)
        self.down = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        x = self.down(self.enc1(x))
        x = self.down(self.enc2(x))
        x = self.down(self.enc3(x))
        x = self.bottleneck(x)
        return self.classifier(self.pool(x))


# ---------------------------------------------------------------------------
# Variantes PREENTRENADAS (transfer learning con pesos ImageNet de torchvision)
# ---------------------------------------------------------------------------
def _build_torchvision(arch, num_classes, dropout, pretrained, freeze_backbone):
    """Construye una arquitectura de torchvision con (o sin) pesos ImageNet y le
    reemplaza la cabeza por una de num_classes. Maneja la API nueva (weights=) y
    la antigua (pretrained=) de torchvision."""
    from torchvision import models as tvm

    builders = {
        "vgg11_bn": (tvm.vgg11_bn, "VGG11_BN_Weights"),
        "resnet18": (tvm.resnet18, "ResNet18_Weights"),
    }
    builder, weights_attr = builders[arch]
    weights = None
    if pretrained:
        try:                                  # torchvision >= 0.13 (API weights=)
            weights = getattr(tvm, weights_attr).IMAGENET1K_V1
            net = builder(weights=weights)
        except (AttributeError, TypeError):   # torchvision antigua (pretrained=)
            net = builder(pretrained=True)
    else:
        try:
            net = builder(weights=None)
        except TypeError:
            net = builder(pretrained=False)

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    # Reemplaza la cabeza (siempre entrenable) por una de num_classes
    if arch.startswith("vgg"):
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:  # resnet
        in_f = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return net


class VGGPretrained(nn.Module):
    """torchvision vgg11_bn preentrenada en ImageNet, con la cabeza reemplazada.

    Transfer learning: parte de pesos ImageNet (ignora base_channels). Por defecto
    afina toda la red; freeze_backbone=True congela el backbone y solo entrena la
    cabeza. NO es comparable apples-to-apples con las versiones desde cero.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 pretrained=True, freeze_backbone=False, **kwargs):
        super().__init__()
        self.net = _build_torchvision("vgg11_bn", num_classes, dropout,
                                      pretrained, freeze_backbone)

    def forward(self, x):
        return self.net(x)


class ResNetPretrained(nn.Module):
    """torchvision resnet18 preentrenada en ImageNet, con la cabeza reemplazada.

    Igual que VGGPretrained pero con resnet18 (mucho mas ligera). Transfer learning,
    no comparable apples-to-apples con las versiones desde cero.
    """

    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3,
                 pretrained=True, freeze_backbone=False, **kwargs):
        super().__init__()
        self.net = _build_torchvision("resnet18", num_classes, dropout,
                                      pretrained, freeze_backbone)

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "cnn": SimpleCNN,
    "mlp": MLP,
    "rnn": ImageRNN,
    "transformer": ImageTransformer,
    # Arquitecturas CNN comparadas en clasificacion (paso 7), desde cero:
    "vgg": VGGStyle,
    "resnet": ResNetStyle,
    "yolo": DarknetStyle,
    "fcn": FCNStyle,
    "unet": UNetEncoder,
    # Variantes preentrenadas (transfer learning):
    "vgg_tl": VGGPretrained,
    "resnet_tl": ResNetPretrained,
}


def build_model(name, num_classes, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido: {name!r}. Disponibles: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
