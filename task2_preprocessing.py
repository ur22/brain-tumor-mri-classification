# -*- coding: utf-8 -*-
"""
Fase 2 - Preprocessing
=======================
2.3  Split TEST da parte + StratifiedKFold sul TRAINVAL (schema a due livelli)
2.1  Normalizzazione: clip -> z-score (per-immagine e globale)
2.2  Esperimento di normalizzazione (confronto tra schemi)
2.4  Generatori Keras con augmentation (pipeline finale di preprocessing)

Esegui `python task2_preprocessing.py` per far girare l'intera fase: legge
outputs/manifest_dedup.csv (prodotto da task1_eda.py) e produce
outputs/trainval.csv (con colonna "fold"), outputs/test.csv e
outputs/class_weight.json, usati poi da task3_modelli.py.

NOTA sull'adattamento: nel notebook originale la sezione 2.2 (esperimento di
normalizzazione) ricreava un secondo split trainval/test "al volo" per fare i
grafici, sovrascrivendo le variabili trainval_df/test_df usate poi in Fase 3
per il training finale e la valutazione sul test. Questo introduceva
un'incoerenza (rischio di data leakage) tra i fold usati per la k-fold e lo
split usato per il modello finale. Qui lo split canonico e' UNO SOLO (2.3,
seed=SEED, test_size=TEST_SIZE), salvato su disco, e riusato ovunque -
anche dall'esperimento di normalizzazione.
"""

import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from task1_eda import (
    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED, TEST_SIZE, N_FOLDS,
    MANIFEST_DEDUP_CSV, TRAINVAL_CSV, TEST_CSV, CLASS_WEIGHT_JSON, FIGURES_DIR,
)

FIGURES_DIR_TASK2 = FIGURES_DIR / "task2_preprocessing"
FIGURES_DIR_TASK2.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]     # una per classe
SOURCE_COLORS = {"figshare": "#3C00FF", "sartaj": "#FF0000", "br35h": "#FF00B7"}


# =============================================================================
# 2.3 Split TEST da parte (stratificato) + StratifiedKFold sul TRAINVAL
# =============================================================================
def make_train_test_split(manifest, test_size=TEST_SIZE, seed=SEED):
    trainval_df, test_df = train_test_split(
        manifest,
        test_size=test_size,
        stratify=manifest["label"],
        random_state=seed,
    )
    trainval_df = trainval_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"TRAINVAL: {len(trainval_df)}   TEST: {len(test_df)}\n")
    comp = pd.DataFrame({
        "trainval": trainval_df["label"].value_counts().reindex(CLASS_NAMES),
        "test":     test_df["label"].value_counts().reindex(CLASS_NAMES),
    })
    print("Classi nei due insiemi:")
    print(comp.to_string())

    if "source" in manifest.columns:
        src = pd.DataFrame({
            "trainval": trainval_df["source"].value_counts(normalize=True).round(3),
            "test":     test_df["source"].value_counts(normalize=True).round(3),
        }).fillna(0)
        print("\nQuota sorgenti (controllo confond):")
        print(src.to_string())
    return trainval_df, test_df


def make_class_weights(trainval_df):
    y = trainval_df["label"].to_numpy()
    weights = compute_class_weight("balanced", classes=np.array(CLASS_NAMES), y=y)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Class weights (indice -> peso):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} = {name:11s}: {class_weight[i]:.3f}")
    return class_weight


def make_folds(trainval_df, n_folds=N_FOLDS, seed=SEED):
    """
    Aggiunge al trainval_df la colonna "fold": per ogni riga, l'indice del
    fold in cui quella riga fa da VALIDATION (le altre righe fanno da train
    per quel fold). Ritorna (trainval_df_con_fold, folds) dove folds e' la
    lista di tuple (fold_id, fold_train_df, fold_val_df) pronta all'uso.
    """
    trainval_df = trainval_df.reset_index(drop=True).copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = trainval_df["label"].to_numpy()

    fold_col = np.full(len(trainval_df), -1, dtype=int)
    folds = []
    print(f"\nStratifiedKFold: {n_folds} fold sul trainval")
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_col[va_idx] = fold_id
        ftr = trainval_df.iloc[tr_idx].reset_index(drop=True)
        fva = trainval_df.iloc[va_idx].reset_index(drop=True)
        folds.append((fold_id, ftr, fva))
        print(f"  fold {fold_id}: train={len(ftr)}  val={len(fva)}")

    trainval_df["fold"] = fold_col
    return trainval_df, folds


def check_class_weight_alignment(class_weight, generator=None):
    """
    Verifica che gli INDICI dei class_weight corrispondano all'ordine delle classi.
    REGOLA: nei generatori usa sempre  classes=CLASS_NAMES  cosi' l'ordine e'
    'glioma, meningioma, pituitary, no_tumor' e combacia con i pesi.
    """
    print("Mappa attesa (CLASS_NAMES):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} = {name:11s} -> peso {class_weight[i]:.3f}")

    if generator is not None:
        gen_order = generator.class_indices
        print("\nOrdine reale del generatore (class_indices):")
        print(f"  {gen_order}")
        ok = all(gen_order.get(name) == i for i, name in enumerate(CLASS_NAMES))
        if ok:
            print("\n[ok] indici allineati: i class_weights vanno sulle classi giuste.")
        else:
            print("\n[!] DISALLINEATO: il generatore usa un ordine diverso da CLASS_NAMES.")
            print("    -> aggiungi  classes=CLASS_NAMES  nella flow_from_dataframe.")
        return ok
    else:
        print("\n(passa anche un generatore per verificare l'allineamento reale)")
        return None


def _autopct(values):
    """Mostra sia percentuale sia conteggio nella fetta."""
    vals = list(values)
    def fmt(pct):
        n = int(round(pct / 100.0 * sum(vals)))
        return f"{pct:.1f}%\n({n})"
    return fmt


def pie_classes(manifest, save_as=None):
    counts = manifest["label"].value_counts().reindex(CLASS_NAMES)
    plt.figure(figsize=(7, 7))
    plt.pie(counts.values, labels=counts.index, colors=CLASS_COLORS,
            autopct=_autopct(counts.values), startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11))
    plt.title(f"Distribuzione delle classi (totale {int(counts.sum())} immagini)",
              fontsize=13, fontweight="bold")
    plt.axis("equal")
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return counts


def pie_sources(manifest, save_as=None):
    counts = manifest["source"].value_counts()
    colors = [SOURCE_COLORS.get(s, "#999999") for s in counts.index]
    plt.figure(figsize=(7, 7))
    plt.pie(counts.values, labels=counts.index, colors=colors,
            autopct=_autopct(counts.values), startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11))
    plt.title(f"Distribuzione delle sorgenti (totale {int(counts.sum())} immagini)",
              fontsize=13, fontweight="bold")
    plt.axis("equal")
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return counts


def pie_source_per_class(manifest, save_as=None):
    """Una torta per ogni classe: da quali sorgenti e' composta (mostra il confond)."""
    ct = pd.crosstab(manifest["label"], manifest["source"]).reindex(CLASS_NAMES).fillna(0)
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 4.5))
    for ax, cls in zip(axes, CLASS_NAMES):
        row = ct.loc[cls]
        row = row[row > 0]
        colors = [SOURCE_COLORS.get(s, "#999999") for s in row.index]
        ax.pie(row.values, labels=row.index, colors=colors,
               autopct=lambda p: f"{p:.0f}%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=1.5),
               textprops=dict(fontsize=9))
        ax.set_title(f"{cls}\n({int(row.sum())} img)", fontsize=11, fontweight="bold")
        ax.axis("equal")
    fig.suptitle("Composizione di ogni classe per sorgente", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return ct


def pie_compare_splits(trainval_df, test_df, by="label", save_as=None):
    """
    Torte affiancate di TRAINVAL e TEST per confrontarne la distribuzione.
    by="label"  -> proporzioni delle classi (verifica la stratificazione)
    by="source" -> proporzioni delle sorgenti (verifica che il confond sia bilanciato)
    """
    order = CLASS_NAMES if by == "label" else None
    palette = CLASS_COLORS if by == "label" else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (name, df) in zip(axes, [("TRAINVAL", trainval_df), ("TEST", test_df)]):
        counts = df[by].value_counts()
        if order is not None:
            counts = counts.reindex(order).fillna(0)
        colors = palette if by == "label" else [SOURCE_COLORS.get(s, "#999999") for s in counts.index]
        ax.pie(counts.values, labels=counts.index, colors=colors,
               autopct=lambda p: f"{p:.1f}%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2),
               textprops=dict(fontsize=10))
        ax.set_title(f"{name}\n({int(counts.sum())} immagini)", fontsize=12, fontweight="bold")
        ax.axis("equal")
    fig.suptitle(f"Confronto distribuzione per '{by}': TRAINVAL vs TEST",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()

    comp = pd.DataFrame({
        "trainval %": (trainval_df[by].value_counts(normalize=True) * 100).round(1),
        "test %":     (test_df[by].value_counts(normalize=True) * 100).round(1),
    })
    if order is not None:
        comp = comp.reindex(order)
    print(f"Percentuali per '{by}' (devono essere quasi identiche):")
    print(comp.to_string())


def save_splits(trainval_df, test_df, class_weight):
    """Persiste lo split canonico su disco, cosi' task3_modelli.py lo puo' ricaricare."""
    trainval_df.to_csv(TRAINVAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    with open(CLASS_WEIGHT_JSON, "w") as f:
        json.dump({str(k): v for k, v in class_weight.items()}, f, indent=2)
    print(f"\n[ok] split salvati in {TRAINVAL_CSV}, {TEST_CSV}, {CLASS_WEIGHT_JSON}")


# =============================================================================
# 2.1 Normalizzazione: clip -> z-score
# =============================================================================
def clip_and_zscore(img):
    """Clip ai percentili [1,99] + z-score sui soli pixel del 'cervello' (> lo)."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip(x, lo, hi)
    brain = x > lo                       # statistiche solo sul cervello, non sullo sfondo
    if brain.sum() > 10:
        mu, sd = x[brain].mean(), x[brain].std() + 1e-8
    else:
        mu, sd = x.mean(), x.std() + 1e-8
    return (x - mu) / sd


def calculate_global_stats(image_paths, clip_percentiles=(1, 99)):
    """
    Media e deviazione standard globali (sui pixel 'cervello') calcolate SOLO sul train.
    Accumula somma e somma dei quadrati immagine per immagine (in float64): stesso
    risultato di np.mean/np.std su tutti i pixel insieme, ma senza tenere in
    memoria centinaia di milioni di valori.
    """
    n_pix, s1, s2 = 0, 0.0, 0.0
    print(f"Calculating global statistics from {len(image_paths)} images...")
    for i, path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i} images...")
        try:
            with Image.open(path) as img:
                x = np.asarray(img.convert('L')).astype(np.float32)
                lo, hi = np.percentile(x, clip_percentiles)
                x_clipped = np.clip(x, lo, hi)
                brain_pixels = x_clipped[x_clipped > lo].astype(np.float64)
                n_pix += brain_pixels.size
                s1 += brain_pixels.sum()
                s2 += np.square(brain_pixels).sum()
        except Exception as e:
            print(f"Warning: Could not process image {path} for global stats: {e}")

    if n_pix == 0:
        raise ValueError("No valid pixels found to calculate global statistics.")

    global_mu = s1 / n_pix
    global_sd = np.sqrt(max(s2 / n_pix - global_mu ** 2, 0.0)) + 1e-8
    print(f"Global statistics calculated: Mean = {global_mu:.4f}, Std Dev = {global_sd:.4f}")
    return global_mu, global_sd


def apply_global_zscore(img, global_mu, global_sd, clip_percentiles=(1, 99)):
    """Clip per-immagine + z-score con statistiche GLOBALI (calcolate sul train)."""
    x = np.asarray(img.convert('L')).astype(np.float32)
    lo, hi = np.percentile(x, clip_percentiles)
    x_clipped = np.clip(x, lo, hi)
    return (x_clipped - global_mu) / global_sd


def preview_single_image_normalization(manifest_row_path, save_as=None):
    """Confronto visivo rapido: immagine originale vs normalizzata per-immagine."""
    image_sample = Image.open(manifest_row_path)
    normalized = clip_and_zscore(np.asarray(image_sample.convert('L')))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_sample.convert('L'), cmap='gray')
    plt.title('Originale (scala di grigi)'); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(normalized, cmap='gray')
    plt.title('Z-score per-immagine'); plt.axis('off')
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()

    orig = np.asarray(image_sample.convert('L'))
    print(f"Originale min/max: {orig.min():.2f}/{orig.max():.2f}")
    print(f"Normalizzata min/max: {normalized.min():.2f}/{normalized.max():.2f}")
    print(f"Normalizzata mean/std: {normalized.mean():.2f}/{normalized.std():.2f}")


# =============================================================================
# 2.2 Esperimento di normalizzazione (confronto tra schemi)
# =============================================================================
def compare_normalizations(df_to_plot, train_df, color_by="source",
                           n_samples_img=300, pixels_per_img=800, seed=123, save_as=None):
    """
    Estrae un campione di immagini, applica i 4 schemi di normalizzazione e
    plotta i KDE. df_to_plot: da cui campionare. train_df: SOLO per calcolare
    le statistiche globali (mai il test).
    """
    print(f"\nGenerazione istogrammi colorati per '{color_by}'...")

    train_sample = train_df.sample(n=min(800, len(train_df)), random_state=SEED)
    paths_for_global = train_sample['path'].tolist()
    g_mu, g_sd = calculate_global_stats(paths_for_global)

    plot_sample = df_to_plot.sample(n=min(n_samples_img, len(df_to_plot)), random_state=seed)

    results = {"Originale (Clipped)": [], "Min-Max (0-1)": [],
               "Z-score (Per-immagine)": [], "Z-score (Globale)": []}
    metadata = []

    for _, row in plot_sample.iterrows():
        try:
            with Image.open(row["path"]) as im:
                x = np.asarray(im.convert('L')).astype(np.float32)

                lo, hi = np.percentile(x, [1, 99])
                x_clip = np.clip(x, lo, hi)

                brain_mask = x_clip > lo
                if not brain_mask.any(): continue
                brain_pixels = x_clip[brain_mask]

                orig = brain_pixels

                min_v, max_v = orig.min(), orig.max()
                minmax = (orig - min_v) / (max_v - min_v + 1e-8)

                mu_img, sd_img = orig.mean(), orig.std() + 1e-8
                z_img = (orig - mu_img) / sd_img

                z_glob = (orig - g_mu) / g_sd

                if len(orig) > pixels_per_img:
                    idx = np.random.choice(len(orig), pixels_per_img, replace=False)
                else:
                    idx = np.arange(len(orig))

                results["Originale (Clipped)"].extend(orig[idx])
                results["Min-Max (0-1)"].extend(minmax[idx])
                results["Z-score (Per-immagine)"].extend(z_img[idx])
                results["Z-score (Globale)"].extend(z_glob[idx])

                metadata.extend([row[color_by]] * len(idx))

        except Exception:
            pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Confronto Normalizzazioni (Colorato per {color_by.upper()})", fontsize=16, fontweight='bold')

    df_plot = pd.DataFrame({"Gruppo": metadata})

    for (title, data), ax in zip(results.items(), axes.flatten()):
        df_plot["Valore"] = data
        sns.kdeplot(data=df_plot, x="Valore", hue="Gruppo", common_norm=False, fill=True, alpha=0.3, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Intensita' Pixel (solo cervello)")
        ax.set_ylabel("Densita'")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_as: plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()


def alignment_score(df_to_eval, train_df, n_samples_img=300):
    """
    Variabilita' (coefficiente di variazione) del 75-esimo percentile tra le
    immagini: piu' e' basso, piu' le immagini sono allineate tra loro.
    """
    print("\n--- Calcolo Alignment Score (Variabilita' tra le immagini) ---")

    train_sample = train_df.sample(n=min(500, len(train_df)), random_state=SEED)
    paths_for_global = train_sample['path'].tolist()
    g_mu, g_sd = calculate_global_stats(paths_for_global)

    eval_sample = df_to_eval.sample(n=min(n_samples_img, len(df_to_eval)), random_state=99)
    p75_orig, p75_minmax, p75_zimg, p75_zglob = [], [], [], []

    for _, row in eval_sample.iterrows():
        try:
            with Image.open(row["path"]) as im:
                x = np.asarray(im.convert('L')).astype(np.float32)
                lo, hi = np.percentile(x, [1, 99])
                x_clip = np.clip(x, lo, hi)

                mask = x_clip > lo
                if not mask.any(): continue
                orig = x_clip[mask]

                p75_orig.append(np.percentile(orig, 75))

                min_v, max_v = orig.min(), orig.max()
                p75_minmax.append(np.percentile((orig - min_v) / (max_v - min_v + 1e-8), 75))

                mu_img, sd_img = orig.mean(), orig.std() + 1e-8
                p75_zimg.append(np.percentile((orig - mu_img) / sd_img, 75))

                p75_zglob.append(np.percentile((orig - g_mu) / g_sd, 75))
        except Exception:
            pass

    def cv(arr):
        return np.std(arr) / abs(np.mean(arr) + 1e-8)

    print(f"Originale (Clipped):       CV = {cv(p75_orig):.4f}")
    print(f"Min-Max (0-1):             CV = {cv(p75_minmax):.4f}")
    print(f"Z-score (Per-immagine):    CV = {cv(p75_zimg):.4f}  <-- Spesso il piu' basso (miglior allineamento)")
    print(f"Z-score (Globale):         CV = {cv(p75_zglob):.4f}")
    print("Nota: un CV piu' basso indica intensita' dei tessuti piu' stabili e allineate.")


# =============================================================================
# 2.4 / 2.5 / 2.6 Generatori Keras con augmentation (pipeline finale)
#
#  clip + z-score PER-IMMAGINE -> resize 224x224 -> 3 canali RGB
#  -> [solo train] augmentation
#
#  IMPORTANTE:
#   - augmentation SOLO sul train (mai su validation/test)
#   - classes=CLASS_NAMES  -> ordine coerente con i class_weights
#   - niente rescale 1/255: normalizziamo noi con clip + z-score
# =============================================================================
def _aug_kwargs():
    return dict(
        rotation_range=12,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        shear_range=8,
        horizontal_flip=True,       # il cervello e' ~simmetrico -> ok
        vertical_flip=False,        # NO: anatomia innaturale
        brightness_range=(0.9, 1.1),
        fill_mode="nearest",
    )


def make_generators(fold_train_df, fold_val_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Ritorna (train_gen, val_gen).
      train_gen: augmentation + clip/z-score, shuffle=True
      val_gen:   solo clip/z-score,           shuffle=False
    """
    train_datagen = ImageDataGenerator(preprocessing_function=clip_and_zscore, **_aug_kwargs())
    val_datagen   = ImageDataGenerator(preprocessing_function=clip_and_zscore)

    common = dict(x_col="path", y_col="label", target_size=img_size,
                  color_mode="rgb", classes=CLASS_NAMES,
                  class_mode="categorical", batch_size=batch_size, seed=SEED)

    train_gen = train_datagen.flow_from_dataframe(fold_train_df, directory=None, shuffle=True, **common)
    val_gen   = val_datagen.flow_from_dataframe(fold_val_df, directory=None, shuffle=False, **common)

    print("class_indices:", train_gen.class_indices)
    ok = all(train_gen.class_indices.get(n) == i for i, n in enumerate(CLASS_NAMES))
    print("[ok] ordine classi allineato ai class_weights" if ok
          else "[!] ordine classi DISALLINEATO -> controlla classes=CLASS_NAMES")
    return train_gen, val_gen


def preview_augmentation(fold_train_df, n=6, img_size=IMG_SIZE):
    """Mostra n versioni augmentate della stessa immagine (SENZA z-score, per vedere l'effetto)."""
    datagen = ImageDataGenerator(**_aug_kwargs())
    img = img_to_array(load_img(fold_train_df.iloc[0]["path"], target_size=img_size))
    img = np.expand_dims(img, 0)
    it = datagen.flow(img, batch_size=1, seed=SEED)
    plt.figure(figsize=(3 * n, 3))
    for k in range(n):
        batch = next(it)
        plt.subplot(1, n, k + 1)
        plt.imshow(batch[0].astype("uint8"))
        plt.axis("off")
    plt.suptitle("Esempi di augmentation (solo train)")
    plt.tight_layout(); plt.show()


def preview_preprocessed(train_gen, n=6):
    """Mostra un batch DOPO tutta la pipeline (augmentation + z-score) per verifica finale."""
    xb, yb = next(train_gen)
    plt.figure(figsize=(3 * n, 3))
    for k in range(min(n, len(xb))):
        plt.subplot(1, n, k + 1)
        im = xb[k]
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        plt.imshow(im)
        plt.title(CLASS_NAMES[int(np.argmax(yb[k]))], fontsize=9); plt.axis("off")
    plt.suptitle("Batch dopo la pipeline completa (augmentation + clip/z-score)")
    plt.tight_layout(); plt.show()


# =============================================================================
# USO (esegui in ordine)
# =============================================================================
if __name__ == "__main__":
    manifest_dedup = pd.read_csv(MANIFEST_DEDUP_CSV)

    # 2.3 split canonico + fold + class weight (UNICO split usato ovunque)
    trainval_df, test_df = make_train_test_split(manifest_dedup)
    trainval_df, folds = make_folds(trainval_df)
    class_weight = make_class_weights(trainval_df)
    check_class_weight_alignment(class_weight)

    # torte descrittive
    pie_classes(manifest_dedup, save_as=FIGURES_DIR_TASK2 / "torta_distribuzione_classi.png")
    pie_sources(manifest_dedup, save_as=FIGURES_DIR_TASK2 / "torta_distribuzione_sorgenti.png")
    pie_source_per_class(manifest_dedup, save_as=FIGURES_DIR_TASK2 / "torta_composizione_classe_per_sorgente.png")
    pie_compare_splits(trainval_df, test_df, by="label",
                       save_as=FIGURES_DIR_TASK2 / "confronto_trainval_test_per_classe.png")
    pie_compare_splits(trainval_df, test_df, by="source",
                       save_as=FIGURES_DIR_TASK2 / "confronto_trainval_test_per_sorgente.png")

    # persiste lo split canonico per task3_modelli.py
    save_splits(trainval_df, test_df, class_weight)

    # 2.1 normalizzazione: esempio su una singola immagine
    preview_single_image_normalization(trainval_df.loc[0, "path"],
                                       save_as=FIGURES_DIR_TASK2 / "normalizzazione_esempio_singola_immagine.png")

    # 2.2 esperimento di normalizzazione (riusa il trainval canonico, non ne crea uno nuovo)
    compare_normalizations(trainval_df, train_df=trainval_df, color_by="source",
                           save_as=FIGURES_DIR_TASK2 / "confronto_normalizzazione_per_sorgente.png")
    compare_normalizations(trainval_df, train_df=trainval_df, color_by="label",
                           save_as=FIGURES_DIR_TASK2 / "confronto_normalizzazione_per_classe.png")
    alignment_score(trainval_df, train_df=trainval_df)

    # 2.4/2.5/2.6 pipeline finale: generatori + controllo visivo
    fold_id, fold_train, fold_val = folds[0]
    train_gen, val_gen = make_generators(fold_train, fold_val)
    preview_augmentation(fold_train)
    preview_preprocessed(train_gen)