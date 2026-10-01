import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from fashion_dl import get_data, CLASSES, OUT, load_idx

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
(xtr, ytr), (xva, yva), (xte, yte) = get_data()
res = {}

# ---------------- sample images
raw = load_idx("train-images-idx3-ubyte"); rawy = load_idx("train-labels-idx1-ubyte", False)
fig, axs = plt.subplots(2, 10, figsize=(13, 3.2))
for c in range(10):
    ims = np.where(rawy == c)[0][:2]
    for r, i in enumerate(ims):
        axs[r, c].imshow(raw[i], cmap="gray"); axs[r, c].axis("off")
        if r == 0: axs[r, c].set_title(CLASSES[c], fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/samples.png", dpi=150); plt.close()

# class balance
cnt = np.bincount(ytr, minlength=10)
plt.figure(figsize=(7, 3)); plt.bar(CLASSES, cnt, color="#3b6ea5"); plt.xticks(rotation=45, ha="right")
plt.ylabel("Training images"); plt.tight_layout(); plt.savefig(f"{OUT}/class_balance.png", dpi=150); plt.close()

# ---------------- evaluate each model on the untouched test set
probs = {}
for name in ["mlp", "plain", "v1", "final"]:
    m = keras.models.load_model(f"{OUT}/{name}.keras")
    p = m.predict(xte, batch_size=256, verbose=0)
    probs[name] = p
    pred = p.argmax(1)
    h = json.load(open(f"{OUT}/{name}_history.json"))
    res[name] = dict(
        params=h["params"], train_min=h["train_seconds"] / 60,
        test_acc=float((pred == yte).mean()),
        macro_f1=float(f1_score(yte, pred, average="macro")),
        final_train_acc=h["accuracy"][-1], final_val_acc=h["val_accuracy"][-1],
        best_val_acc=max(h["val_accuracy"]),
        final_train_loss=h["loss"][-1], final_val_loss=h["val_loss"][-1],
        min_val_loss=min(h["val_loss"]), min_val_loss_epoch=int(np.argmin(h["val_loss"])) + 1,
        epochs=len(h["loss"]))
final_pred = probs["final"].argmax(1)
rep = classification_report(yte, final_pred, target_names=CLASSES, output_dict=True, digits=4)
res["report"] = rep
cm = confusion_matrix(yte, final_pred)
res["cm"] = cm.tolist()

# ---------------- training curves (all models)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {"mlp": "#d98c2b", "plain": "#c0392b", "v1": "#7d5ba6", "final": "#2e8b57"}
labels = {"mlp": "MLP", "plain": "Plain CNN", "v1": "Residual CNN v1", "final": "Residual CNN v2"}
for n in ["mlp", "plain", "v1", "final"]:
    h = json.load(open(f"{OUT}/{n}_history.json"))
    e = np.arange(1, len(h["loss"]) + 1)
    axs[0].plot(e, h["accuracy"], "--", c=cols[n], alpha=.7)
    axs[0].plot(e, h["val_accuracy"], c=cols[n], label=labels[n])
    if n in ("mlp", "plain"):
        axs[1].plot(e, h["loss"], "--", c=cols[n], alpha=.7)
        axs[1].plot(e, h["val_loss"], c=cols[n], label=labels[n])
axs[0].set_title("Accuracy (dashed = train, solid = validation)"); axs[0].set_xlabel("Epoch"); axs[0].legend()
axs[0].set_ylim(.84, 1.0)
axs[1].set_title("Loss, unregularised models (dashed = train)"); axs[1].set_xlabel("Epoch"); axs[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/curves_compare.png", dpi=150); plt.close()

# final model curves
h = json.load(open(f"{OUT}/final_history.json")); e = np.arange(1, len(h["loss"]) + 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(e, h["accuracy"], label="Train"); axs[0].plot(e, h["val_accuracy"], label="Validation")
axs[0].set_title("Residual CNN v2 - accuracy"); axs[0].set_xlabel("Epoch"); axs[0].legend()
axs[1].plot(e, h["loss"], label="Train"); axs[1].plot(e, h["val_loss"], label="Validation")
axs[1].set_title("Residual CNN v2 - loss (label-smoothed CE)"); axs[1].set_xlabel("Epoch"); axs[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/curves_final.png", dpi=150); plt.close()

# ---------------- confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues"); ax.grid(False)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.colorbar(im, fraction=.046)
plt.tight_layout(); plt.savefig(f"{OUT}/confusion.png", dpi=150); plt.close()

# ---------------- per-class F1 bar chart
f1s = [rep[c]["f1-score"] for c in CLASSES]
order = np.argsort(f1s)
plt.figure(figsize=(7, 3.6))
plt.barh([CLASSES[i] for i in order], [f1s[i] for i in order], color="#3b6ea5")
plt.xlim(.7, 1.0); plt.xlabel("F1-score"); plt.tight_layout()
plt.savefig(f"{OUT}/f1_per_class.png", dpi=150); plt.close()

# ---------------- misclassified examples
wrong = np.where(final_pred != yte)[0]
conf = probs["final"].max(1)[wrong]
pick = wrong[np.argsort(-conf)[:10]]
fig, axs = plt.subplots(2, 5, figsize=(10, 4.6))
for a, i in zip(axs.ravel(), pick):
    a.imshow(xte[i, ..., 0], cmap="gray"); a.axis("off")
    a.set_title(f"true: {CLASSES[yte[i]]}\npred: {CLASSES[final_pred[i]]} ({probs['final'][i].max():.2f})", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/misclassified.png", dpi=150); plt.close()

# ---------------- learned feature maps of first conv layer
m = keras.models.load_model(f"{OUT}/final.keras")
first = [l for l in m.layers if isinstance(l, keras.layers.Conv2D)][0]
w = first.get_weights()[0]  
fig, axs = plt.subplots(2, 8, figsize=(8, 2.4))
for k, a in enumerate(axs.ravel()):
    a.imshow(w[:, :, 0, k], cmap="RdBu"); a.axis("off")
plt.suptitle("Learned 3x3 filters of the first convolution layer", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/filters.png", dpi=150); plt.close()

# ---------------- architecture diagram
fig, ax = plt.subplots(figsize=(13.5, 3.3)); ax.axis("off"); ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.3)
blocks = [("Input\n28x28x1", "#dfe6ee"), ("Augment\nflip+shift", "#f3e3c3"),
          ("Conv3x3\n16+BN+ReLU", "#b7d3ee"), ("MaxPool\n14x14", "#e6c9c9"),
          ("Conv3x3\n32+BN+ReLU", "#b7d3ee"), ("Residual\nblock 32", "#8fb9e6"),
          ("MaxPool\n7x7\n+SpDrop .1", "#e6c9c9"), ("Conv3x3\n64+BN+ReLU", "#b7d3ee"),
          ("Residual\nblock 64", "#8fb9e6"), ("Conv3x3\n128+BN+ReLU", "#b7d3ee"),
          ("Global\nAvgPool", "#cfe5cf"), ("Dropout .2\nDense 10\nSoftmax", "#c9e0c0")]
w_, gap = 0.95, 0.12
for i, (t, c) in enumerate(blocks):
    x = 0.1 + i * (w_ + gap)
    ax.add_patch(FancyBboxPatch((x, 0.9), w_, 1.5, boxstyle="round,pad=0.02", fc=c, ec="#333"))
    ax.text(x + w_ / 2, 1.65, t, ha="center", va="center", fontsize=7)
    if i < len(blocks) - 1:
        ax.add_patch(FancyArrowPatch((x + w_ + .02, 1.65), (x + w_ + gap - .02, 1.65), arrowstyle="->", mutation_scale=8))
for i in (5, 8):
    x = 0.1 + i * (w_ + gap)
    ax.add_patch(FancyArrowPatch((x + .15, 2.4), (x + w_ - .15, 2.4), connectionstyle="arc3,rad=-.9",
                                 arrowstyle="->", mutation_scale=9, color="#b03030"))
    ax.text(x + w_ / 2, 2.95, "skip (+)", ha="center", fontsize=7, color="#b03030")
ax.text(6.75, 0.35, "Feature maps: 28x28x16  ->  14x14x32  ->  7x7x64  ->  7x7x128  ->  128-d vector  ->  10 classes",
        ha="center", fontsize=9)
plt.savefig(f"{OUT}/architecture.png", dpi=170, bbox_inches="tight"); plt.close()

# residual block detail diagram
fig, ax = plt.subplots(figsize=(5, 4)); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 5)
def bx(y, t, c="#b7d3ee"):
    ax.add_patch(FancyBboxPatch((1.4, y), 2.2, .55, boxstyle="round,pad=0.02", fc=c, ec="#333"))
    ax.text(2.5, y + .27, t, ha="center", va="center", fontsize=8.5)
bx(4.3, "x (input)", "#dfe6ee"); bx(3.4, "Conv3x3 + BN + ReLU"); bx(2.5, "Conv3x3 + BN")
bx(1.6, "Add  (x + F(x))", "#f3e3c3"); bx(.7, "ReLU", "#cfe5cf")
for y0, y1 in [(4.3, 3.95), (3.4, 3.05), (2.5, 2.15), (1.6, 1.25)]:
    ax.annotate("", xy=(2.5, y1), xytext=(2.5, y0), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(1.4, 1.87), xytext=(1.4, 4.55), arrowprops=dict(arrowstyle="->", color="#b03030",
            connectionstyle="arc3,rad=.7"))
ax.text(.35, 3.2, "identity\nskip", color="#b03030", fontsize=8, ha="center")
plt.savefig(f"{OUT}/resblock.png", dpi=170, bbox_inches="tight"); plt.close()

json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("report", "cm")}, indent=1))
print("final macro F1:", rep["macro avg"]["f1-score"])
for c in CLASSES: print(f"{c:12s} P={rep[c]['precision']:.3f} R={rep[c]['recall']:.3f} F1={rep[c]['f1-score']:.3f}")
