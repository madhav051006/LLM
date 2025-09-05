# mixed_qat_litert_keras.py
# Static mixed-precision QAT (per-layer INT8 + FP32 fallback) -> LiteRT (.tflite)
# Two export modes:
#  - Mode A (default): float32 I/O (no representative dataset)
#  - Mode B (optional): int8 I/O (requires representative dataset)

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from ai_edge_litert.interpreter import Interpreter

tf.random.set_seed(42)

# ==============================
# Config
# ==============================
INTEGER_ONLY_IO = False   # False = Mode A (no rep dataset). True = Mode B (int8 I/O with rep dataset).
REP_SAMPLES = 128         # used only if INTEGER_ONLY_IO=True
EPOCHS = 1
BATCH = 64

# ==============================
# 1) Build model (NHWC)
# ==============================
def build_float_model(num_classes=10):
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv_first")(inputs)   # FLOAT
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv_32_q")(x)        # INT8
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_64_q")(x)        # INT8
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_64_float")(x)    # FLOAT
    x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv_128_q")(x)      # INT8
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_float")(x)                         # FLOAT
    outputs = keras.layers.Dense(num_classes, name="logits")(x)                                   # FLOAT
    return keras.Model(inputs, outputs, name="mixed_qat_cifar10")

float_model = build_float_model()
float_model.summary()

# ==============================
# 2) Annotate quantized layers only
# ==============================
QuantAnnot = tfmot.quantization.keras.quantize_annotate_layer
SENSITIVE_LAYERS = {"conv_first", "conv_64_float", "dense_float", "logits"}  # keep these float

def clone_fn(layer: keras.layers.Layer):
    if isinstance(layer, keras.layers.Conv2D) and layer.name not in SENSITIVE_LAYERS:
        return QuantAnnot(layer)   # quantize selected Conv2D
    return layer

annotated = keras.models.clone_model(float_model, clone_function=clone_fn)

# ==============================
# 3) Apply QAT transforms
# ==============================
with tfmot.quantization.keras.quantize_scope():
    qat_model = tfmot.quantization.keras.quantize_apply(annotated)

qat_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Tiny demo train (replace with real CIFAR-10 for accuracy)
x_dummy = np.random.rand(256, 32, 32, 3).astype("float32")
y_dummy = np.random.randint(0, 10, size=(256,))
qat_model.fit(x_dummy, y_dummy, epochs=EPOCHS, batch_size=BATCH, verbose=1)

# ==============================
# 4) Convert to LiteRT/TFLite
# ==============================
def make_rep_dataset():
    # Use *real* preprocessed inputs here for best calibration when INTEGER_ONLY_IO=True
    x = (x_dummy).astype("float32")
    def gen():
        n = min(REP_SAMPLES, len(x))
        for i in range(n):
            yield [x[i:i+1]]
    return gen

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)

if INTEGER_ONLY_IO:
    # Mode B: integer-only I/O requires representative dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = make_rep_dataset()
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path = "mixed_qat_cifar10_int8_io.tflite"
else:
    # Mode A: ***IMPORTANT*** Do NOT set int8 I/O or PTQ-only flags.
    # Let the converter consume the QAT graph as-is.
    # (No representative dataset needed; float32 I/O, int8 internals where annotated.)
    # NOTE: Do NOT set `converter.optimizations` or `supported_ops` here.
    tflite_path = "mixed_qat_cifar10_int8_partial.tflite"

tflite_bytes = converter.convert()
with open(tflite_path, "wb") as f:
    f.write(tflite_bytes)
print(f"✅ Exported LiteRT model → {tflite_path}")

# ==============================
# 5) Quick LiteRT inference check
# ==============================
interp = Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("Input details:", inp)
print("Output details:", out)

# Prepare one dummy input
if inp["dtype"].__name__ == "int8":
    scale, zp = inp["quantization"]
    x = np.random.rand(1, 32, 32, 3).astype("float32")
    x_q = np.clip(np.round(x / (scale if scale else 1.0) + zp), -128, 127).astype(np.int8)
    feed = x_q
else:
    feed = np.random.rand(1, 32, 32, 3).astype(inp["dtype"].__name__)

interp.set_tensor(inp["index"], feed)
interp.invoke()
y = interp.get_tensor(out["index"])
print("LiteRT output shape:", y.shape)

# ==============================
# 6) Show which layers were intended to be quantized
# ==============================
print("\n=== Layer precision summary (requested) ===")
for l in float_model.layers:
    if isinstance(l, keras.layers.Conv2D):
        status = "INT8 (QAT)" if l.name not in SENSITIVE_LAYERS else "FLOAT32"
        print(f"{l.name:15s} -> {status}")
    elif isinstance(l, keras.layers.Dense):
        print(f"{l.name:15s} -> FLOAT32")
