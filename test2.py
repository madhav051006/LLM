# adabits_qat_cifar10_to_litert.py
# AdaBits-style QAT in Keras -> freeze to 8-bit -> official QAT (TF-MOT) -> LiteRT (.tflite) export + quick run

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from ai_edge_litert.interpreter import Interpreter

tf.random.set_seed(1234)

# ==============================
# Config
# ==============================
BITWIDTHS = (8, 6, 4)         # AdaBits training bitwidth set
EPOCHS_ADABITS = 1            # increase for real training
EPOCHS_QAT = 1                # increase for real training
BATCH = 128
INTEGER_ONLY_IO = False       # False: float32 I/O (no rep dataset). True: int8 I/O (needs rep dataset)
REP_SAMPLES = 256             # representative samples when INTEGER_ONLY_IO=True

# ==============================
# Data: CIFAR-10
# ==============================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = y_train.flatten().astype("int32")
y_test  = y_test.flatten().astype("int32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10_000).batch(BATCH).prefetch(2)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH).prefetch(2)

# ==============================
# AdaBits blocks: Fake-quant + switchable bitwidths
# ==============================
def fake_quant_sym(x, num_bits, clip):
    qmin = -(2 ** (num_bits - 1))
    qmax =  (2 ** (num_bits - 1)) - 1
    clip = tf.maximum(clip, tf.cast(1e-6, x.dtype))
    x = tf.clip_by_value(x, -clip, clip)
    scale = clip / qmax
    q = tf.round(x / scale) * scale
    # STE
    return x + tf.stop_gradient(q - x)

class SwitchableActQuant(keras.layers.Layer):
    """Activation fake-quant with learnable clip per bitwidth (AdaBits S-CL)."""
    def __init__(self, bitwidths=(8,6,4), **kw):
        super().__init__(**kw)
        self.bitwidths = tuple(sorted(set(bitwidths), reverse=True))
        self.current_bits = tf.Variable(self.bitwidths[0], trainable=False, dtype=tf.int32)
        self.alpha = {}  # bw -> learnable clip

    def build(self, input_shape):
        for b in self.bitwidths:
            self.alpha[b] = self.add_weight(
                name=f"alpha_{b}", shape=(), dtype=self.dtype, trainable=True,
                initializer=keras.initializers.Constant(6.0)
            )

    def call(self, x, training=None):
        x = tf.nn.relu(x)
        out = 0.0
        for bw in self.bitwidths:
            mask = tf.cast(tf.equal(self.current_bits, bw), x.dtype)
            out += mask * fake_quant_sym(x, int(bw), tf.abs(self.alpha[bw]))
        return out

class AdaBitsConv2D(keras.layers.Layer):
    """Conv2D with AdaBits-style quantization for weights and activations."""
    def __init__(self, filters, kernel_size, strides=1, padding="same", use_bias=True, bitwidths=(8,6,4), name=None):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.strides = (1, strides, strides, 1) if isinstance(strides, int) else (1, *strides, 1)
        self.pad = padding.upper()
        self.use_bias = use_bias
        self.bitwidths = tuple(sorted(set(bitwidths), reverse=True))
        self.actq = SwitchableActQuant(self.bitwidths)
        self.current_bits = self.actq.current_bits

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(kh, kw, in_ch, self.filters),
            initializer="he_normal", trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)
        else:
            self.bias = None

    def call(self, x, training=None):
        # Weight clip = max-abs
        w_clip = tf.reduce_max(tf.abs(self.kernel))
        out = 0.0
        for bw in self.bitwidths:
            mask = tf.cast(tf.equal(self.current_bits, bw), x.dtype)
            qw = fake_quant_sym(self.kernel, int(bw), w_clip)
            y = tf.nn.conv2d(x, qw, strides=self.strides, padding=self.pad)
            if self.bias is not None:
                y = tf.nn.bias_add(y, self.bias)
            y = self.actq(y, training=training)
            out += mask * y
        return out

# ==============================
# AdaBits model (same topology as later float/QAT model)
# ==============================
def build_adabits_model(num_classes=10, bitwidths=BITWIDTHS):
    inputs = keras.Input((32,32,3))
    x = AdaBitsConv2D(32, 3, padding="same", bitwidths=bitwidths, name="ab_conv1")(inputs)
    x = AdaBitsConv2D(32, 3, padding="same", bitwidths=bitwidths, name="ab_conv2")(x)
    x = keras.layers.MaxPool2D()(x)

    x = AdaBitsConv2D(64, 3, padding="same", bitwidths=bitwidths, name="ab_conv3")(x)
    x = AdaBitsConv2D(64, 3, padding="same", bitwidths=bitwidths, name="ab_conv4")(x)
    x = keras.layers.MaxPool2D()(x)

    x = AdaBitsConv2D(128, 3, padding="same", bitwidths=bitwidths, name="ab_conv5")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu", name="ab_dense")(x)
    outputs = keras.layers.Dense(10, name="ab_logits")(x)
    return keras.Model(inputs, outputs, name="adabits_net")

adabits_model = build_adabits_model()
optimizer = keras.optimizers.Adam(1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc = keras.metrics.SparseCategoricalAccuracy()
val_acc = keras.metrics.SparseCategoricalAccuracy()

# Utility: set current bitwidth across all AdaBits layers
BW_CHOICES = tf.constant(BITWIDTHS, dtype=tf.int32)
def set_current_bitwidth(model, bw):
    bw = tf.cast(bw, tf.int32)
    for l in model.layers:
        if hasattr(l, "current_bits") and isinstance(l.current_bits, tf.Variable):
            l.current_bits.assign(bw)
        if hasattr(l, "actq") and hasattr(l.actq, "current_bits"):
            l.actq.current_bits.assign(bw)

@tf.function
def train_step(x, y, bw):
    set_current_bitwidth(adabits_model, bw)
    with tf.GradientTape() as tape:
        logits = adabits_model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, adabits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, adabits_model.trainable_variables))
    train_acc.update_state(y, logits)
    return loss

@tf.function
def eval_step(x, y):
    set_current_bitwidth(adabits_model, tf.reduce_max(BW_CHOICES))  # evaluate at 8b
    logits = adabits_model(x, training=False)
    val_acc.update_state(y, logits)

print("\n=== AdaBits phase ===")
for epoch in range(EPOCHS_ADABITS):
    train_acc.reset_state()
    val_acc.reset_state()
    for xb, yb in train_ds:
        bw = tf.random.shuffle(BW_CHOICES)[0]  # tf.int32
        _loss = train_step(xb, yb, bw)
    for xb, yb in test_ds.take(50):
        eval_step(xb, yb)
    print(f"Epoch {epoch+1}/{EPOCHS_ADABITS}  TrainAcc={train_acc.result():.3f}  ValAcc@8b={val_acc.result():.3f}")

# ==============================
# Freeze to 8-bit: transfer to a float Keras model
# ==============================
def build_float_model(num_classes=10):
    inputs = keras.Input((32,32,3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv4")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv5")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu", name="dense")(x)
    outputs = keras.layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="float_net")

float_model = build_float_model()

# Map AdaBits weights -> Float model
name_map = {
    "ab_conv1": "conv1",
    "ab_conv2": "conv2",
    "ab_conv3": "conv3",
    "ab_conv4": "conv4",
    "ab_conv5": "conv5",
    "ab_dense": "dense",
    "ab_logits": "logits",
}
for ab_name, fl_name in name_map.items():
    ab_layer = adabits_model.get_layer(ab_name)
    fl_layer = float_model.get_layer(fl_name)
    if "conv" in ab_name:
        fl_layer.set_weights([ab_layer.kernel.numpy(), ab_layer.bias.numpy()])
    else:  # dense / logits
        fl_layer.set_weights(ab_layer.get_weights())

print("\nTransferred AdaBits weights -> float model (frozen at 8-bit behavior).")

# ==============================
# Official QAT (TF-MOT) on selected layers (INT8), others float
# ==============================
QuantAnnot = tfmot.quantization.keras.quantize_annotate_layer
SENSITIVE = {"conv1", "dense", "logits"}  # keep these float; quantize conv2..conv5

def clone_fn(layer):
    if isinstance(layer, keras.layers.Conv2D) and layer.name not in SENSITIVE:
        return QuantAnnot(layer)
    return layer

annotated = keras.models.clone_model(float_model, clone_function=clone_fn)
annotated.set_weights(float_model.get_weights())  # <— keep weights you just transferred

with tfmot.quantization.keras.quantize_scope():
    qat_model = tfmot.quantization.keras.quantize_apply(annotated)

qat_model.compile(
    optimizer=keras.optimizers.Adam(5e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("\n=== Short TF-MOT QAT finetune ===")
qat_model.fit(train_ds.take(200), epochs=EPOCHS_QAT, validation_data=test_ds.take(50), verbose=1)

# ==============================
# Convert to LiteRT (.tflite)
# ==============================
def make_rep_dataset():
    def gen():
        for i in range(min(REP_SAMPLES, len(x_train))):
            yield [x_train[i:i+1]]
    return gen

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
if INTEGER_ONLY_IO:
    # Strict int8 I/O
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = make_rep_dataset()
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path = "adabits_qat_cifar10_int8_io.tflite"
else:
    # Float32 I/O; int8 internals where annotated (no rep dataset needed)
    # Do NOT set int8 I/O here.
    tflite_path = "adabits_qat_cifar10_int8_partial.tflite"

tflite_bytes = converter.convert()
with open(tflite_path, "wb") as f:
    f.write(tflite_bytes)
print(f"\n✅ Exported LiteRT model → {tflite_path}")

# ==============================
# LiteRT interpreter sanity check
# ==============================
interp = Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("Input:", {"shape": inp["shape"].tolist(), "dtype": str(inp["dtype"]), "quant": inp["quantization"]})
print("Output:", {"shape": out["shape"].tolist(), "dtype": str(out["dtype"]), "quant": out["quantization"]})

sample = x_test[:1].copy()
if inp["dtype"].__name__ == "int8":
    scale, zp = inp["quantization"]
    sample_q = np.clip(np.round(sample / (scale if scale else 1.0) + zp), -128, 127).astype(np.int8)
    feed = sample_q
else:
    feed = sample.astype(inp["dtype"].__name__)

interp.set_tensor(inp["index"], feed)
interp.invoke()
pred = interp.get_tensor(out["index"])
print("LiteRT logits (first sample):", pred[0])
