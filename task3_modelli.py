# -*- coding: utf-8 -*-
"""
Fase 3 - Modelli
=================
Transfer learning (EfficientNetB0, fine-tuning a due fasi) + valutazione,
training e aggregazione su 5 fold, modello finale valutato sul test, e infine
una CNN custom (ricerca iperparametri + training su 5 fold, due dimensioni).

Richiede che task1_eda.py e task2_preprocessing.py siano gia' stati eseguiti
(servono outputs/trainval.csv, outputs/test.csv, outputs/class_weight.json).

Esegui `python task3_modelli.py` per far girare l'intera fase. ATTENZIONE:
allena piu' modelli (k-fold x2 architetture) -> puo' richiedere ore su CPU.
Riduci EPOCHS_* qui sotto per una prova rapida in locale.
"""

import gc
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (confusion_matrix, classification_report,
                              balanced_accuracy_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from task1_eda import (
    CLASS_NAMES, N_CLASSES, IMG_SIZE, BATCH_SIZE, SEED, N_FOLDS,
    TRAINVAL_CSV, TEST_CSV, CLASS_WEIGHT_JSON, GDRIVE_MEDICINE_DIR,
    KFOLD_TL_DIR, KFOLD_CNN_DIR, FINAL_MODEL_PATH, AGGREGATED_CM_PNG, FIGURES_DIR,
    set_all_seeds,
)
from task2_preprocessing import clip_and_zscore

FIGURES_DIR_TL = FIGURES_DIR / "task3_modelli" / "transfer_learning"
FIGURES_DIR_CNN = FIGURES_DIR / "task3_modelli" / "cnn_custom"
FIGURES_DIR_TL.mkdir(parents=True, exist_ok=True)
FIGURES_DIR_CNN.mkdir(parents=True, exist_ok=True)

# Numero di epoche: alzali per una run "vera", abbassali per una prova rapida.
EPOCHS_A = 8       # fase A (feature extraction, backbone congelato)
EPOCHS_B = 8       # fase B (fine-tuning)
EPOCHS_CNN = 30    # training CNN custom


# =============================================================================
# Caricamento split canonico (prodotto da task2_preprocessing.py)
# =============================================================================
def load_splits():
    trainval_df = pd.read_csv(TRAINVAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    with open(CLASS_WEIGHT_JSON) as f:
        class_weight = {int(k): v for k, v in json.load(f).items()}

    folds = []
    for fold_id in sorted(trainval_df["fold"].unique()):
        fva = trainval_df[trainval_df["fold"] == fold_id].reset_index(drop=True)
        ftr = trainval_df[trainval_df["fold"] != fold_id].reset_index(drop=True)
        folds.append((int(fold_id), ftr, fva))
    return trainval_df, test_df, class_weight, folds


# =============================================================================
# Transfer learning: generatori (preprocess_input, non z-score)
# =============================================================================
def _tl_aug_kwargs():
    return dict(rotation_range=12, width_shift_range=0.05, height_shift_range=0.05,
                zoom_range=0.10, shear_range=8, horizontal_flip=True,
                vertical_flip=False, brightness_range=(0.9, 1.1), fill_mode="nearest")


def make_tl_generators(fold_train_df, fold_val_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **_tl_aug_kwargs())
    val_datagen   = ImageDataGenerator(preprocessing_function=preprocess_input)

    common = dict(x_col="path", y_col="label", target_size=img_size, color_mode="rgb",
                  classes=CLASS_NAMES, class_mode="categorical",
                  batch_size=batch_size, seed=SEED)
    train_gen = train_datagen.flow_from_dataframe(fold_train_df, directory=None, shuffle=True, **common)
    val_gen   = val_datagen.flow_from_dataframe(fold_val_df, directory=None, shuffle=False, **common)

    ok = all(train_gen.class_indices.get(n) == i for i, n in enumerate(CLASS_NAMES))
    print("class_indices:", train_gen.class_indices,
          "\n[ok] ordine allineato ai class_weights" if ok else "\n[!] ordine DISALLINEATO")
    return train_gen, val_gen


def make_final_generators(trainval_df, test_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Come make_tl_generators, ma per il training finale (trainval completo vs test)."""
    return make_tl_generators(trainval_df, test_df, img_size=img_size, batch_size=batch_size)


# =============================================================================
# Modello di transfer learning: backbone EfficientNetB0 + testa a 4 classi
# =============================================================================
def build_model(img_size=IMG_SIZE, n_classes=N_CLASSES, dropout=0.3):
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(*img_size, 3), pooling="avg")
    base.trainable = False                       # FASE A: backbone congelato

    inputs = layers.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base


def load_final_tl_model(save_path=FINAL_MODEL_PATH, unfreeze_from=-20):
    """
    Ricostruisce il modello TL e ricarica i pesi salvati da train_final()/fine_tune().
    IMPORTANTE: bisogna rimettere lo stesso stato "trainable" (base sbloccata,
    ultimi `unfreeze_from` layer scongelati) che c'era al momento del save_weights,
    altrimenti load_weights fallisce con "ValueError: axes don't match array"
    (cambiare trainable cambia l'ordine train/non-train dei pesi di BatchNorm).
    """
    model, base = build_model()
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    model.load_weights(save_path)
    return model, base


def train_head(model, train_gen, val_gen, class_weight, epochs=EPOCHS_A):
    """FASE A - feature extraction (backbone congelato, si allena solo la testa)."""
    cbs = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
           ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)]
    print("\n=== FASE A: feature extraction (backbone congelato) ===")
    return model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                     class_weight=class_weight, callbacks=cbs)


def fine_tune(model, base, train_gen, val_gen, class_weight,
              unfreeze_from=-20, epochs=EPOCHS_B, lr=1e-5):
    """FASE B - fine-tuning (scongela gli ultimi blocchi, LR molto basso)."""
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:      # scongela solo gli ULTIMI blocchi
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cbs = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
           ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1)]
    print(f"\n=== FASE B: fine-tuning (ultimi {abs(unfreeze_from)} layer, lr={lr}) ===")
    return model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                     class_weight=class_weight, callbacks=cbs)


def plot_history(*histories, save_as=None):
    """Disegna accuracy e loss (train/val) concatenando le fasi A e B."""
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]; val_acc += h.history["val_accuracy"]
        loss += h.history["loss"];    val_loss += h.history["val_loss"]
    ep = range(1, len(acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ep, acc, label="train"); ax[0].plot(ep, val_acc, label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoca"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, loss, label="train"); ax[1].plot(ep, val_loss, label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoca"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()


# =============================================================================
# Valutazione (comune a TL e CNN)
# =============================================================================
def get_predictions(model, gen):
    """gen deve avere shuffle=False -> ordine allineato alle etichette vere."""
    gen.reset()
    y_prob = model.predict(gen, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    y_true = gen.classes
    return y_true, y_pred, y_prob


def plot_confusion(y_true, y_pred, save_as=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (conteggi)")
    axes[0].set_xlabel("Predetto"); axes[0].set_ylabel("Reale")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Normalizzata per riga (sensibilita' per classe)")
    axes[1].set_xlabel("Predetto"); axes[1].set_ylabel("Reale")
    plt.tight_layout()
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return cm


def full_report(y_true, y_pred, y_prob=None):
    print("=== Metriche per classe ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Balanced accuracy : {bal_acc:.3f}   (media delle sensibilita' per classe)")
    print(f"Macro-F1          : {macro_f1:.3f}   (media degli F1, non pesata)")

    cm = confusion_matrix(y_true, y_pred)
    print("\nSensibilita' e specificita' per classe:")
    total = cm.sum()
    for i, name in enumerate(CLASS_NAMES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) else 0
        spec = TN / (TN + FP) if (TN + FP) else 0
        print(f"  {name:11s}: sensibilita'={sens:.3f}  specificita'={spec:.3f}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(pd.get_dummies(y_true), y_prob, average="macro", multi_class="ovr")
            print(f"\nROC-AUC (macro, one-vs-rest): {auc:.3f}")
        except Exception as e:
            print(f"\n[ROC-AUC non calcolabile: {e}]")

    return {"balanced_accuracy": bal_acc, "macro_f1": macro_f1}


def evaluate(model, gen, save_cm=None):
    y_true, y_pred, y_prob = get_predictions(model, gen)
    plot_confusion(y_true, y_pred, save_as=save_cm)
    return full_report(y_true, y_pred, y_prob)


# =============================================================================
# Training + valutazione su 5 fold (transfer learning)
# =============================================================================
def run_kfold(folds, class_weight, epochs_A=EPOCHS_A, epochs_B=EPOCHS_B,
              save_dir=KFOLD_TL_DIR, save_models=False):
    """
    Esegue il training completo sui fold e raccoglie le metriche per fold.
    Ritorna (df, summary): un DataFrame con una riga per fold + media/std.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for fold_id, fold_train, fold_val in folds:
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_id}   (train={len(fold_train)}  val={len(fold_val)})")
        print("=" * 70)

        train_gen, val_gen = make_tl_generators(fold_train, fold_val)
        model, base = build_model()

        train_head(model, train_gen, val_gen, class_weight, epochs=epochs_A)
        fine_tune(model, base, train_gen, val_gen, class_weight, epochs=epochs_B)

        y_true, y_pred, y_prob = get_predictions(model, val_gen)
        m = full_report(y_true, y_pred, y_prob)
        m["fold"] = fold_id
        results.append(m)

        pd.DataFrame(results).to_csv(save_dir / "metrics_folds.csv", index=False)
        np.savez(save_dir / f"fold{fold_id}_preds.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        if save_models:
            # .weights.h5: model.save() pieno rompe su EfficientNetB0 (bug TF, vedi FINAL_MODEL_PATH)
            model.save_weights(save_dir / f"model_fold{fold_id}.weights.h5")

        del model, base, train_gen, val_gen
        tf.keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(results).set_index("fold")
    summary = pd.DataFrame({"media": df.mean(), "std": df.std()})
    print("\n" + "=" * 70)
    print("  RISULTATO K-FOLD (media +/- deviazione standard sui 5 fold)")
    print("=" * 70)
    for metric in df.columns:
        print(f"  {metric:20s}: {summary.loc[metric,'media']:.3f} +/- {summary.loc[metric,'std']:.3f}")
    df.to_csv(save_dir / "metrics_folds.csv")
    summary.to_csv(save_dir / "metrics_summary.csv")
    print(f"\n[ok] metriche salvate in {save_dir}/")
    return df, summary


def load_all_predictions(save_dir):
    """Carica e concatena tutti i fold*_preds.npz salvati dal ciclo k-fold."""
    files = sorted(glob.glob(str(Path(save_dir) / "fold*_preds.npz")))
    if not files:
        raise FileNotFoundError(f"Nessun file fold*_preds.npz in {save_dir}")
    y_true, y_pred, y_prob = [], [], []
    for f in files:
        d = np.load(f)
        y_true.append(d["y_true"]); y_pred.append(d["y_pred"]); y_prob.append(d["y_prob"])
        print(f"  {Path(f).name}: {len(d['y_true'])} esempi")
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    print(f"[ok] totale esempi aggregati: {len(y_true)}")
    return y_true, y_pred, y_prob


def aggregate_confusion(save_dir, save_fig=None):
    y_true, y_pred, y_prob = load_all_predictions(save_dir)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix aggregata (5 fold, conteggi)")
    axes[0].set_xlabel("Predetto"); axes[0].set_ylabel("Reale")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Normalizzata per riga (sensibilita' per classe)")
    axes[1].set_xlabel("Predetto"); axes[1].set_ylabel("Reale")
    plt.tight_layout()
    if save_fig: plt.savefig(save_fig, dpi=120, bbox_inches="tight")
    plt.show()

    print("\n=== Metriche aggregate (su tutti i fold) ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro-F1         : {f1_score(y_true, y_pred, average='macro'):.3f}")

    print("\nErrori principali per classe (dove vengono confuse):")
    for i, name in enumerate(CLASS_NAMES):
        row = cm[i].copy(); row[i] = 0
        if row.sum() > 0:
            j = row.argmax()
            pct = 100 * row[j] / cm[i].sum()
            print(f"  {name:11s}: soprattutto -> {CLASS_NAMES[j]} ({pct:.0f}% dei {name})")
    return cm


# =============================================================================
# Modello finale (con validation vera per l'early stopping) - il TEST non
# viene mai toccato qui: si usa una sola volta, alla fine, per confrontare
# questo modello con la CNN finale (vedi make_test_generator_tl/cnn sotto).
# =============================================================================
def make_final_train_val_split(trainval_df, val_size=0.15, seed=SEED):
    """
    Split dedicato SOLO al training finale (TL o CNN): una fetta di
    validation per un early stopping vero. E' uno split fresco, diverso dai
    5 fold della k-fold - il test resta comunque fuori da tutto questo.
    """
    final_train, final_val = train_test_split(
        trainval_df, test_size=val_size, stratify=trainval_df["label"], random_state=seed,
    )
    final_train = final_train.reset_index(drop=True)
    final_val = final_val.reset_index(drop=True)
    print(f"Training finale: train={len(final_train)}  val={len(final_val)}")
    return final_train, final_val


def train_final(trainval_df, class_weight, val_size=0.15,
                epochs_A=EPOCHS_A, epochs_B=EPOCHS_B, save_path=FINAL_MODEL_PATH):
    """
    Allena il modello finale TL con un vero validation set (ricavato dal
    trainval) -> early stopping reale, non epoche fisse alla cieca.
    Il test NON viene toccato qui. Ritorna (model, hA, hB).
    """
    final_train, final_val = make_final_train_val_split(trainval_df, val_size=val_size)
    train_gen, val_gen = make_tl_generators(final_train, final_val)
    model, base = build_model()

    hA = train_head(model, train_gen, val_gen, class_weight, epochs=epochs_A)
    hB = fine_tune(model, base, train_gen, val_gen, class_weight, epochs=epochs_B)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # solo le weights: model.save() pieno rompe su EfficientNetB0 (bug TF 2.10-2.12,
        # normalizzazione interna diventa un EagerTensor non serializzabile in JSON).
        # Per ricaricare: build_model() poi model.load_weights(save_path).
        model.save_weights(save_path)
        print(f"\n[ok] pesi del modello finale salvati in {save_path}")
    return model, hA, hB


def make_test_generator_tl(test_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Generatore SOLO test per il modello TL (preprocess_input, shuffle=False)."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    common = dict(x_col="path", y_col="label", target_size=img_size, color_mode="rgb",
                  classes=CLASS_NAMES, class_mode="categorical", batch_size=batch_size, seed=SEED)
    return datagen.flow_from_dataframe(test_df, directory=None, shuffle=False, **common)


# =============================================================================
# CNN custom (architettura semplice, allenata da zero)
# =============================================================================
def make_cnn_generators(fold_train_df, fold_val_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Generatori per la CNN custom: usa clip_and_zscore (task2_preprocessing), non preprocess_input."""
    aug = dict(rotation_range=12, width_shift_range=0.05, height_shift_range=0.05,
               zoom_range=0.10, shear_range=8, horizontal_flip=True,
               vertical_flip=False, brightness_range=(0.9, 1.1), fill_mode="nearest")
    train_datagen = ImageDataGenerator(preprocessing_function=clip_and_zscore, **aug)
    val_datagen   = ImageDataGenerator(preprocessing_function=clip_and_zscore)
    common = dict(x_col="path", y_col="label", target_size=img_size, color_mode="rgb",
                  classes=CLASS_NAMES, class_mode="categorical", batch_size=batch_size, seed=SEED)
    train_gen = train_datagen.flow_from_dataframe(fold_train_df, directory=None, shuffle=True, **common)
    val_gen   = val_datagen.flow_from_dataframe(fold_val_df, directory=None, shuffle=False, **common)
    ok = all(train_gen.class_indices.get(n) == i for i, n in enumerate(CLASS_NAMES))
    print("class_indices:", train_gen.class_indices, "\n[ok] allineato" if ok else "\n[!] DISALLINEATO")
    return train_gen, val_gen


def build_cnn(img_size=IMG_SIZE, n_classes=N_CLASSES, lr=1e-3, dropout=0.5):
    """
    4 blocchi Conv+BatchNorm+MaxPool(2x2) -> GlobalAveragePooling -> Dense -> softmax.
    """
    model = models.Sequential([
        layers.Input(shape=(*img_size, 3)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout, seed=SEED),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(model, train_gen, val_gen, class_weight, epochs=EPOCHS_CNN):
    cbs = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
           ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)]
    return model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                     class_weight=class_weight, callbacks=cbs)


def build_cnn_sized(size="base", img_size=IMG_SIZE, n_classes=N_CLASSES, lr=1e-3, dropout=0.5):
    """
    Tre varianti di dimensione:
      small : 16-32-64-64   filtri, Dense 128   (rete leggera)
      base  : 32-64-128-128 filtri, Dense 256   (quella di default)
      big   : 32-64-128-256 filtri, Dense 512   (rete piu' capace)
    """
    filters = {"small": [16, 32, 64, 64],
               "base":  [32, 64, 128, 128],
               "big":   [32, 64, 128, 256]}[size]
    dense = {"small": 128, "base": 256, "big": 512}[size]

    model = models.Sequential([layers.Input(shape=(*img_size, 3))])
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization(momentum=0.9))
        model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense, activation="relu"))
    model.add(layers.Dropout(dropout, seed=SEED))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def search_lr_then_size(fold_train_df, fold_val_df, class_weight,
                        lrs=(1e-3, 3e-4, 1e-4), sizes=("small", "base", "big"),
                        epochs=25, img_size=(128, 128)):
    """
    Ricerca coordinata su un fold, in due passi:
      PASSO 1: fissa size='base', prova gli lr -> sceglie il lr migliore
      PASSO 2: fissa il lr migliore, prova le dimensioni -> sceglie la size migliore
    Ritorna (best_lr, best_size).
    """
    train_gen, val_gen = make_cnn_generators(fold_train_df, fold_val_df, img_size=img_size)

    def _run(build_fn):
        tf.keras.backend.clear_session(); gc.collect()
        set_all_seeds(SEED)
        model = build_fn()
        cbs = [EarlyStopping(monitor="val_loss", patience=5,
                             restore_best_weights=True, verbose=0),
               ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=0)]
        model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                  class_weight=class_weight, callbacks=cbs, verbose=2)
        score = float(model.evaluate(val_gen, verbose=0)[1])
        del model
        return score

    print("\n########## PASSO 1: learning rate (size=base) ##########")
    lr_res = {}
    for lr in lrs:
        print(f"\n===== lr={lr} =====")
        lr_res[lr] = _run(lambda lr=lr: build_cnn_sized("base", lr=lr, img_size=img_size))
    best_lr = max(lr_res, key=lr_res.get)
    print("\nval_accuracy per lr:", {k: round(v, 3) for k, v in lr_res.items()})
    print(f"-> lr migliore: {best_lr}")

    print(f"\n########## PASSO 2: dimensione rete (lr={best_lr}) ##########")
    size_res = {}
    for size in sizes:
        print(f"\n===== size={size} =====")
        size_res[size] = _run(lambda size=size: build_cnn_sized(size, lr=best_lr, img_size=img_size))
    best_size = max(size_res, key=size_res.get)
    print("\nval_accuracy per size:", {k: round(v, 3) for k, v in size_res.items()})
    print(f"-> size migliore: {best_size}")

    print(f"\n=== CONFIGURAZIONE SCELTA: lr={best_lr}, size={best_size} ===")

    ctrl = _run(lambda: build_cnn_sized("base", lr=1e-3, img_size=img_size))
    print(f"[controllo] lr=1e-3/base ripetuto: {ctrl:.3f} vs {lr_res[1e-3]:.3f} "
          f"(delta {abs(ctrl - lr_res[1e-3]):.3f})")

    return best_lr, best_size


def train_kfold_cnn(folds, class_weight, lr=1e-3, size="base",
                    epochs=EPOCHS_CNN, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    save_root=KFOLD_CNN_DIR):
    save_dir = Path(save_root) / f"{size}_lr{lr}_bs{batch_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, fold_train, fold_val in folds:
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_id}  (train={len(fold_train)}  val={len(fold_val)})  "
              f"lr={lr}  size={size}  bs={batch_size}")
        print("=" * 70)

        tf.keras.backend.clear_session(); gc.collect()
        set_all_seeds(SEED + fold_id)

        train_gen, val_gen = make_cnn_generators(fold_train, fold_val,
                                                 img_size=img_size, batch_size=batch_size)
        model = build_cnn_sized(size, lr=lr, img_size=img_size)
        train_cnn(model, train_gen, val_gen, class_weight, epochs=epochs)

        model.save(save_dir / f"model_fold{fold_id}.keras")
        pd.DataFrame(model.history.history).to_csv(save_dir / f"fold{fold_id}_history.csv", index=False)
        print(f"[ok] fold {fold_id} salvato in {save_dir}/")

        del model, train_gen, val_gen
        tf.keras.backend.clear_session(); gc.collect()

    print(f"\n[ok] training completato: {save_dir}/")


def evaluate_kfold_cnn(folds, size="base", lr=1e-3, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                       save_root=KFOLD_CNN_DIR):
    """
    Ricarica i modelli salvati da train_kfold_cnn per una config (size, lr) e
    calcola le metriche aggregate (balanced accuracy, macro-F1) fold per fold,
    con media +/- std - stesso formato di run_kfold, per confrontare
    onestamente "base" vs "big" (non solo l'accuracy delle history.csv).
    """
    save_dir = Path(save_root) / f"{size}_lr{lr}_bs{batch_size}"
    results = []

    for fold_id, fold_train, fold_val in folds:
        model_path = save_dir / f"model_fold{fold_id}.keras"
        if not model_path.exists():
            print(f"[!] modello mancante per fold {fold_id}: {model_path}"); continue

        _, val_gen = make_cnn_generators(fold_train, fold_val, img_size=img_size, batch_size=batch_size)
        model = tf.keras.models.load_model(model_path)

        print(f"\n--- fold {fold_id} ({size}, lr={lr}) ---")
        y_true, y_pred, y_prob = get_predictions(model, val_gen)
        m = full_report(y_true, y_pred, y_prob)
        m["fold"] = fold_id
        results.append(m)

        del model, val_gen
        tf.keras.backend.clear_session(); gc.collect()

    df = pd.DataFrame(results).set_index("fold")
    summary = pd.DataFrame({"media": df.mean(), "std": df.std()})
    print("\n" + "=" * 70)
    print(f"  RISULTATO K-FOLD CNN '{size}' (lr={lr}) - media +/- std sui {len(df)} fold")
    print("=" * 70)
    for metric in df.columns:
        print(f"  {metric:20s}: {summary.loc[metric,'media']:.3f} +/- {summary.loc[metric,'std']:.3f}")
    df.to_csv(save_dir / "metrics_folds_eval.csv")
    summary.to_csv(save_dir / "metrics_summary_eval.csv")
    return df, summary


def train_final_cnn(trainval_df, class_weight, size="base", lr=1e-3, val_size=0.15,
                    epochs=EPOCHS_CNN, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    save_path=None):
    """
    Equivalente di train_final() ma per la CNN custom: usa un vero validation
    set (early stopping reale) ricavato dal trainval, con la 'size' (small/base/big)
    vincitrice della k-fold. Il test NON viene toccato qui.
    """
    final_train, final_val = make_final_train_val_split(trainval_df, val_size=val_size)
    train_gen, val_gen = make_cnn_generators(final_train, final_val,
                                             img_size=img_size, batch_size=batch_size)
    set_all_seeds(SEED)
    model = build_cnn_sized(size, lr=lr, img_size=img_size)

    print(f"\n=== CNN FINALE: training con validation (size={size}, lr={lr}) ===")
    hist = train_cnn(model, train_gen, val_gen, class_weight, epochs=epochs)

    if save_path is None:
        save_path = GDRIVE_MEDICINE_DIR / "modellok" / f"model_cnn_{size}_finale.keras"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\n[ok] CNN finale salvata in {save_path}")
    return model, hist


def make_test_generator_cnn(test_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Generatore SOLO test per la CNN custom (clip_and_zscore, shuffle=False)."""
    datagen = ImageDataGenerator(preprocessing_function=clip_and_zscore)
    common = dict(x_col="path", y_col="label", target_size=img_size, color_mode="rgb",
                  classes=CLASS_NAMES, class_mode="categorical", batch_size=batch_size, seed=SEED)
    return datagen.flow_from_dataframe(test_df, directory=None, shuffle=False, **common)


# =============================================================================
# USO (esegui in ordine)
# =============================================================================
if __name__ == "__main__":
    print("GPU disponibili:", tf.config.list_physical_devices('GPU'))

    trainval_df, test_df, class_weight, folds = load_splits()

    # --- Transfer learning: prova su un fold ---
    fold_id, fold_train, fold_val = folds[0]
    train_gen, val_gen = make_tl_generators(fold_train, fold_val)
    model, base = build_model()
    hA = train_head(model, train_gen, val_gen, class_weight, epochs=EPOCHS_A)
    hB = fine_tune(model, base, train_gen, val_gen, class_weight, epochs=EPOCHS_B)
    plot_history(hA, hB, save_as=FIGURES_DIR_TL / "tl_demo_fold0_accuracy_loss.png")

    loss, acc = model.evaluate(val_gen)
    print(f"val accuracy: {acc:.4f}")
    evaluate(model, val_gen, save_cm=FIGURES_DIR_TL / "tl_demo_fold0_confusion_matrix.png")
    del model, base, train_gen, val_gen
    tf.keras.backend.clear_session(); gc.collect()

    # --- Transfer learning: k-fold completa + aggregazione ---
    df_tl, summary_tl = run_kfold(folds, class_weight, epochs_A=EPOCHS_A, epochs_B=EPOCHS_B,
                                  save_dir=KFOLD_TL_DIR, save_models=False)
    aggregate_confusion(KFOLD_TL_DIR, save_fig=AGGREGATED_CM_PNG)

    # --- Modello finale TL: training con vero validation set (niente test qui) ---
    model_tl, hA_final, hB_final = train_final(trainval_df, class_weight,
                                               epochs_A=EPOCHS_A, epochs_B=EPOCHS_B,
                                               save_path=FINAL_MODEL_PATH)
    plot_history(hA_final, hB_final, save_as=FIGURES_DIR_TL / "tl_finale_accuracy_loss_train_val.png")
    del model_tl
    tf.keras.backend.clear_session(); gc.collect()

    # --- CNN custom: ricerca iperparametri su un fold (sottocampionato) ---
    fold_id, fold_train, fold_val = folds[0]
    sub_train = (fold_train.groupby("label", group_keys=False)
                           .sample(frac=0.35, random_state=SEED))
    best_lr, best_size = search_lr_then_size(sub_train, fold_val, class_weight)

    # --- CNN custom: training k-fold completo, size "small" e "base" (le piu' vicine
    #     nella ricerca rapida: 0.815 vs 0.806; "big" era piu' indietro a 0.788) ---
    train_kfold_cnn(folds, class_weight, lr=1e-3, size="small", batch_size=BATCH_SIZE)
    train_kfold_cnn(folds, class_weight, lr=1e-3, size="base", batch_size=BATCH_SIZE)

    # --- Confronto small vs base (k-fold) -> scelta della size vincente ---
    _, summary_small = evaluate_kfold_cnn(folds, size="small", lr=1e-3, batch_size=BATCH_SIZE)
    _, summary_base = evaluate_kfold_cnn(folds, size="base", lr=1e-3, batch_size=BATCH_SIZE)
    ba_small = summary_small.loc["balanced_accuracy", "media"]
    ba_base = summary_base.loc["balanced_accuracy", "media"]
    winner = "base" if ba_base > ba_small else "small"
    print(f"\nbalanced_accuracy media: small={ba_small:.4f}  base={ba_base:.4f}  -> vince '{winner}'")

    # --- CNN finale: training con vero validation set (niente test qui) ---
    model_cnn, hist_cnn = train_final_cnn(trainval_df, class_weight, size=winner, lr=1e-3)
    tf.keras.backend.clear_session(); gc.collect()

    # --- CONFRONTO FINALE: il test si tocca UNA SOLA VOLTA qui, per entrambi ---
    print("\n" + "=" * 70)
    print("  CONFRONTO FINALE SUL TEST (TL vs CNN)")
    print("=" * 70)

    model_tl, _ = load_final_tl_model(FINAL_MODEL_PATH)
    test_gen_tl = make_test_generator_tl(test_df)
    print("\n--- Transfer learning (finale) ---")
    metrics_test_tl = evaluate(model_tl, test_gen_tl, save_cm=FIGURES_DIR_TL / "tl_finale_test_confusion_matrix.png")

    test_gen_cnn = make_test_generator_cnn(test_df)
    print(f"\n--- CNN custom ({winner}, finale) ---")
    metrics_test_cnn = evaluate(model_cnn, test_gen_cnn,
                                save_cm=FIGURES_DIR_CNN / f"cnn_{winner}_finale_test_confusion_matrix.png")

    print("\n=== RIEPILOGO ===")
    print("TL su test :", metrics_test_tl)
    print(f"CNN ({winner}) su test:", metrics_test_cnn)