"""
Week 5 - Deep Learning Application in Data Science
Fashion-MNIST image classification with TensorFlow/Keras

Three models are trained and compared:
  1. MLP baseline               (no spatial awareness)
  2. Plain CNN                  (no regularisation -> demonstrates overfitting)
  3. Residual CNN (final model) (BatchNorm + residual blocks + augmentation +
                                 dropout + label smoothing + cosine LR decay)

Usage:  python fashion_dl.py mlp | plain | v1 | final   (final = v2, the best model)
"""
import sys, gzip, json, time, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)
DATA = "data"; OUT = "results"; os.makedirs(OUT, exist_ok=True)
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ------------------------------------------------------------------ data
def load_idx(name, image=True):
    with gzip.open(f"{DATA}/{name}.gz", "rb") as f:
        buf = f.read()
    if image:
        return np.frombuffer(buf, np.uint8, offset=16).reshape(-1, 28, 28)
    return np.frombuffer(buf, np.uint8, offset=8)


def get_data():
    x_tr = load_idx("train-images-idx3-ubyte").astype("float32") / 255.0
    y_tr = load_idx("train-labels-idx1-ubyte", False).astype("int64")
    x_te = load_idx("t10k-images-idx3-ubyte").astype("float32") / 255.0
    y_te = load_idx("t10k-labels-idx1-ubyte", False).astype("int64")
    x_tr, x_te = x_tr[..., None], x_te[..., None]
    # stratified-ish random 90/10 train/validation split (test set untouched)
    idx = np.random.RandomState(SEED).permutation(len(x_tr))
    n_val = 6000
    va, tr = idx[:n_val], idx[n_val:]
    return (x_tr[tr], y_tr[tr]), (x_tr[va], y_tr[va]), (x_te, y_te)


# ---------------------------------------------------------------- models
def build_mlp():
    return keras.Sequential([
        layers.Input((28, 28, 1)), layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax")], name="mlp_baseline")


def build_plain_cnn():
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax")], name="plain_cnn")


def conv_bn_relu(x, f, k=3, s=1, l2=1e-4):
    x = layers.Conv2D(f, k, strides=s, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2) if l2 else None)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def res_block(x, f, l2=1e-4):
    """Two 3x3 conv-BN layers with an identity skip connection."""
    s = x
    y = conv_bn_relu(x, f, l2=l2)
    y = layers.Conv2D(f, 3, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2) if l2 else None)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Add()([s, y])
    return layers.ReLU()(y)


def build_v1():
    inp = layers.Input((28, 28, 1))
    # on-the-fly augmentation (only active during training)
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomTranslation(0.08, 0.08, fill_mode="nearest")(x)
    x = layers.RandomRotation(0.05, fill_mode="nearest")(x)
    x = conv_bn_relu(x, 16)
    x = res_block(x, 16)
    x = layers.MaxPooling2D()(x)               # 14x14
    x = layers.SpatialDropout2D(0.1)(x)
    x = conv_bn_relu(x, 32)
    x = res_block(x, 32)
    x = layers.MaxPooling2D()(x)               # 7x7
    x = layers.SpatialDropout2D(0.2)(x)
    x = conv_bn_relu(x, 64)
    x = layers.GlobalAveragePooling2D()(x)     # far fewer params than Flatten+Dense
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out, name="residual_cnn")


def build_final():
    """v2: lighter augmentation, no double weight-decay, residual blocks moved to
    the low-resolution stages (14x14, 7x7) where they are cheap and most useful."""
    inp = layers.Input((28, 28, 1))
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomTranslation(0.06, 0.06, fill_mode="nearest")(x)
    x = conv_bn_relu(x, 16, l2=0)
    x = layers.MaxPooling2D()(x)               # 14x14
    x = conv_bn_relu(x, 32, l2=0)
    x = res_block(x, 32, l2=0)
    x = layers.MaxPooling2D()(x)               # 7x7
    x = layers.SpatialDropout2D(0.1)(x)
    x = conv_bn_relu(x, 64, l2=0)
    x = res_block(x, 64, l2=0)
    x = conv_bn_relu(x, 128, l2=0)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out, name="residual_cnn_v2")


# -------------------------------------------------------------- training
def train(name):
    (xtr, ytr), (xva, yva), (xte, yte) = get_data()
    if name == "mlp":
        model, epochs, bs = build_mlp(), 20, 128
        opt, loss = keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy"
    elif name == "plain":
        model, epochs, bs = build_plain_cnn(), 20, 128
        opt, loss = keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy"
    else:
        model, epochs, bs = (build_v1() if name == "v1" else build_final()), (20 if name == "v1" else 15), 128
        steps = epochs * int(np.ceil(len(xtr) / bs))
        lr = keras.optimizers.schedules.CosineDecay(2e-3 if name == "final" else 3e-3, steps, alpha=0.01)
        opt = keras.optimizers.AdamW(lr, weight_decay=5e-4 if name == "final" else 1e-4)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05 if name == "final" else 0.1)
        ytr_oh = keras.utils.to_categorical(ytr, 10)
        yva_oh = keras.utils.to_categorical(yva, 10)
    model.summary()
    model.compile(opt, loss, metrics=["accuracy"])
    t0 = time.time()
    if name in ("v1", "final"):
        cbs = [keras.callbacks.ModelCheckpoint(f"{OUT}/{name}.keras", monitor="val_accuracy",
                                               save_best_only=True, verbose=0)]
        h = model.fit(xtr, ytr_oh, validation_data=(xva, yva_oh), epochs=epochs,
                      batch_size=bs, callbacks=cbs, verbose=2)
    else:
        h = model.fit(xtr, ytr, validation_data=(xva, yva), epochs=epochs,
                      batch_size=bs, verbose=2)
        model.save(f"{OUT}/{name}.keras")
    secs = time.time() - t0
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    hist["train_seconds"] = secs
    hist["params"] = int(model.count_params())
    json.dump(hist, open(f"{OUT}/{name}_history.json", "w"))
    print(f"[{name}] done in {secs/60:.1f} min")


if __name__ == "__main__":
    train(sys.argv[1])
