# -*- coding: utf-8 -*-
"""
Fase 1 - Analisi Esplorativa dei Dati (EDA)
============================================
1.1 Costruzione del manifest (id, path, label, source) dai dataset locali
1.2 Controlli di integrita' delle immagini
1.3 Distribuzione delle classi + label x source
1.4 Intensita' per sorgente (domain shift)
1.5 Dimensioni, canali, piani anatomici
1.6 Deduplicazione dentro e tra i dataset (pHash)

Questo file contiene anche la CONFIGURAZIONE CONDIVISA del progetto (percorsi,
classi, iperparametri, seed), importata da task2_preprocessing.py e
task3_modelli.py, cosi' tutto e' definito in un unico posto.

Esegui `python task1_eda.py` per far girare l'intera fase: produce
outputs/manifest.csv e outputs/manifest_dedup.csv, usati poi da task2.
"""

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# CONFIGURAZIONE CONDIVISA (importata anche da task2 e task3)
# Percorsi locali (al posto di Google Drive / /content/...)
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Cartella che contiene gli zip dei dataset (figshare.zip, SARTAJ.zip,
# Br35H_no.zip) e in cui vengono salvati modelli e risultati della k-fold.
# Di default e' `data/` dentro il repo. Per usare un'altra cartella (per esempio
# quella di Google Drive sincronizzata da Drive Desktop) imposta la variabile
# d'ambiente MRI_DATA_DIR prima di lanciare gli script:
#   Windows (PowerShell): $env:MRI_DATA_DIR = "G:\My Drive\medicine"
#   Linux / WSL:          export MRI_DATA_DIR="/mnt/g/My Drive/medicine"
GDRIVE_MEDICINE_DIR = Path(os.environ.get("MRI_DATA_DIR", PROJECT_DIR / "data")).expanduser()
DATA_DIR = GDRIVE_MEDICINE_DIR  # alias con un nome piu' generico

# Cartella locale dove vengono scompattati gli zip (equivalente di
# /content/extracted_datasets/ in Colab). Verificata contro il contenuto reale
# degli zip: figshare.zip -> figshare/figshare/<classe>, SARTAJ.zip ->
# SARTAJ/SARTAJ/<classe>, Br35H_no.zip -> Br35H_no/<img>.jpg (nessuna sottocartella).
EXTRACTED_DIR = PROJECT_DIR / "extracted_datasets"

DATASETS = [
    {"root": EXTRACTED_DIR / "figshare" / "figshare", "source": "figshare"},
    {"root": EXTRACTED_DIR / "SARTAJ" / "SARTAJ",      "source": "sartaj"},
    {"root": EXTRACTED_DIR / "Br35H_no",               "source": "br35h"},
]

# Output "leggeri" e riproducibili (manifest, split, figure EDA) restano locali.
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
for _d in (OUTPUT_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
MANIFEST_DEDUP_CSV = OUTPUT_DIR / "manifest_dedup.csv"
TRAINVAL_CSV = OUTPUT_DIR / "trainval.csv"   # include la colonna "fold"
TEST_CSV = OUTPUT_DIR / "test.csv"
CLASS_WEIGHT_JSON = OUTPUT_DIR / "class_weight.json"

# Modelli/risultati di training (Fase 3): vanno nella cartella dati
# (MRI_DATA_DIR), che puo' essere anche una cartella sincronizzata con Google
# Drive, cosi' i pesi restano al sicuro e fuori dal repository.
KFOLD_TL_DIR = GDRIVE_MEDICINE_DIR / "kfold_out"          # risultati k-fold transfer learning
KFOLD_CNN_DIR = GDRIVE_MEDICINE_DIR / "kfold_cnn"         # risultati k-fold CNN custom
# .weights.h5 e non .keras: bug noto di TF 2.10/2.11/2.12 con EfficientNetB0,
# model.save() fallisce (EagerTensor non serializzabile in JSON). Le sole
# weights si salvano/ricaricano senza problemi (serve pero' ricostruire
# l'architettura con build_model() prima di load_weights()).
FINAL_MODEL_PATH = GDRIVE_MEDICINE_DIR / "modellok" / "model_finale.weights.h5"
AGGREGATED_CM_PNG = GDRIVE_MEDICINE_DIR / "cm_aggregata.png"

# -----------------------------------------------------------------------------
# Classi canoniche e mappa delle etichette grezze -> classe canonica
# -----------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
N_CLASSES = len(CLASS_NAMES)

# 'yes' del Br35H NON e' usabile a 4 classi (sottotipo ignoto) -> None (scartato).
# Se una cartella ha un nome diverso, aggiungilo qui.
RAW_LABEL_MAP = {
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
    "notumor": "no_tumor", "no_tumor": "no_tumor", "br35h_no": "no_tumor",
    "yes": None,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -----------------------------------------------------------------------------
# Parametri di training / preprocessing condivisi tra le fasi
# -----------------------------------------------------------------------------
IMG_SIZE = (224, 224)      # per backbone ImageNet
BATCH_SIZE = 32
SEED = 111
TEST_SIZE = 0.15
N_FOLDS = 5


def set_all_seeds(seed=SEED):
    """Fissa il seed di os/random/numpy/tensorflow per la riproducibilita'."""
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


FIGURES_DIR_TASK1 = FIGURES_DIR / "task1_eda"
FIGURES_DIR_TASK1.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1.1 Decomprimi i dataset dalla cartella dati (es. una cartella sincronizzata con
#     Google Drive Desktop, vedi MRI_DATA_DIR) e costruisci il manifest
# =============================================================================
def extract_all_zips(main_folder=GDRIVE_MEDICINE_DIR, base_unzip_path=EXTRACTED_DIR):
    """
    Scompatta ogni file .zip trovato direttamente dentro `main_folder` (la
    cartella "medicine" sincronizzata da Google Drive Desktop) in una
    sottocartella di `base_unzip_path` con lo stesso nome del file zip.
    Salta gli zip gia' estratti in precedenza.
    """
    base_unzip_path.mkdir(parents=True, exist_ok=True)

    if not main_folder.exists():
        raise FileNotFoundError(
            f"Cartella dei dati non trovata: {main_folder}\n"
            "Mettici dentro figshare.zip, SARTAJ.zip e Br35H_no.zip, oppure "
            "imposta MRI_DATA_DIR sulla cartella che li contiene (vedi README)."
        )

    print(f"Ricerca di file .zip in: {main_folder}")
    n_extracted = 0
    for item_path in sorted(main_folder.iterdir()):
        if not (item_path.is_file() and item_path.suffix.lower() == ".zip"):
            continue

        extract_folder = base_unzip_path / item_path.stem
        if extract_folder.exists() and any(extract_folder.iterdir()):
            print(f"[skip] '{item_path.name}' gia' estratto in '{extract_folder}'")
            continue

        extract_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nDecompressione di '{item_path.name}' in '{extract_folder}'...")
        with zipfile.ZipFile(item_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Decompressione completata per '{item_path.name}'.")
        n_extracted += 1

    print(f"\n[ok] {n_extracted} archivi estratti (gli altri erano gia' pronti).")


# =============================================================================
# 1.1 Costruzione del manifest
# =============================================================================
def canonicalize_label(raw_name):
    """Nome grezzo di una cartella -> classe canonica (o None se da scartare)."""
    key = raw_name.strip().lower().replace(" ", "_")
    return RAW_LABEL_MAP.get(key, None)


def inspect_class_folders(datasets=DATASETS):
    """
    Stampa, per ogni dataset, le sottocartelle e come vengono mappate.
    LANCIALA PRIMA di costruire il manifest: se qualcosa e' 'SCARTATA' ma e'
    una vera classe, aggiungi la variante del nome a RAW_LABEL_MAP.
    """
    for spec in datasets:
        root = Path(spec["root"])
        print(f"\n[{spec['source']}] {root}")
        if not root.exists():
            print("  (!) percorso inesistente")
            continue
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not subdirs:
            print("  (!) nessuna sottocartella: il percorso punta davvero alle classi?")
        for d in subdirs:
            label = canonicalize_label(d.name)
            n_img = sum(1 for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS)
            status = f"-> {label}" if label else "-> SCARTATA (nome non in RAW_LABEL_MAP)"
            print(f"  {d.name:22s} {status:38s} ({n_img} immagini)")


def build_manifest(datasets=DATASETS, save_to=MANIFEST_CSV):
    """
    Costruisce il manifest unico: una riga per immagine, colonne
    [id, path, label, source]. Salva anche il CSV.
    """
    rows = []
    for spec in datasets:
        root = Path(spec["root"])
        source = spec["source"]
        if not root.exists():
            print(f"[ATTENZIONE] percorso inesistente, salto: {root}")
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = canonicalize_label(class_dir.name)
            if label is None:
                print(f"[info] scarto cartella non mappabile: {class_dir.name} ({source})")
                continue
            # Ordine FISSO (nome senza distinzione maiuscole/minuscole, come
            # Esplora risorse di Windows): rglob restituisce i file in un ordine
            # che dipende dal sistema operativo, e da quest'ordine dipendono la
            # deduplicazione e lo split. Cosi' i risultati sono riproducibili ovunque.
            img_paths = sorted(class_dir.rglob("*"),
                               key=lambda q: str(q.relative_to(class_dir)).upper())
            for img_path in img_paths:
                if img_path.suffix.lower() in IMG_EXTS:
                    rows.append({
                        "path": str(img_path.resolve()),
                        "label": label,
                        "source": source,
                    })

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise ValueError("Manifest vuoto: controlla i percorsi in DATASETS.")
    manifest.insert(0, "id", np.arange(len(manifest)))

    if save_to:
        manifest.to_csv(save_to, index=False)
        print(f"\n[ok] manifest salvato in {save_to}  (righe: {len(manifest)})")
    return manifest


def summary(manifest):
    """Riepilogo rapido: conteggi per classe e tabella incrociata label x source."""
    print("\nConteggi per classe:")
    print(manifest["label"].value_counts().reindex(CLASS_NAMES).to_string())
    print("\nlabel x source:")
    print(pd.crosstab(manifest["label"], manifest["source"])
            .reindex(CLASS_NAMES).fillna(0).astype(int).to_string())


# =============================================================================
# 1.2 Controlli di integrita' delle immagini
# =============================================================================
def check_image_integrity(image_paths):
    """
    Controlli di base sui file immagine per identificare file corrotti, vuoti
    o con un numero inatteso di canali (es. grigio triplicato in RGB).
    Ritorna (corrupted_images, empty_images, grayscale_rgb_images, valid_count).
    """
    corrupted_images = []
    empty_images = []
    grayscale_rgb_images = []
    valid_images_count = 0

    print(f"Esecuzione controlli di integrita' su {len(image_paths)} immagini...")

    for i, image_path in enumerate(image_paths):
        if i % 1000 == 0 and i > 0:
            print(f"Elaborate {i} immagini...")

        if os.path.getsize(image_path) == 0:
            empty_images.append(image_path)
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            img = Image.open(image_path)
            if img.mode == 'RGB':
                img_np = np.array(img)
                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    if np.all(img_np[:, :, 0] == img_np[:, :, 1]) and np.all(img_np[:, :, 1] == img_np[:, :, 2]):
                        grayscale_rgb_images.append(image_path)

            valid_images_count += 1

        except Exception as e:
            corrupted_images.append((image_path, str(e)))

    print("\n--- Riepilogo Controlli di Integrita' ---")
    print(f"Immagini totali elaborate: {len(image_paths)}")
    print(f"Immagini valide: {valid_images_count}")
    print(f"Immagini corrotte/non apribili: {len(corrupted_images)}")
    if corrupted_images:
        print("Primi 5 percorsi immagini corrotte:")
        for path, error in corrupted_images[:5]:
            print(f"  - {path} (Errore: {error})")

    print(f"Immagini vuote (dimensione 0 byte): {len(empty_images)}")
    if empty_images:
        print("Primi 5 percorsi immagini vuote:")
        for path in empty_images[:5]:
            print(f"  - {path}")

    print(f"Immagini in scala di grigi con 3 canali (RGB triplicato): {len(grayscale_rgb_images)}")
    if grayscale_rgb_images:
        print("Primi 5 percorsi immagini in scala di grigi triplicate RGB:")
        for path in grayscale_rgb_images[:5]:
            print(f"  - {path}")

    if not corrupted_images and not empty_images and not grayscale_rgb_images:
        print("Tutti i controlli di integrita' di base sono stati superati con successo.")
    else:
        print("Alcuni problemi di integrita' sono stati rilevati. Rivedi i dettagli sopra.")

    return corrupted_images, empty_images, grayscale_rgb_images, valid_images_count


# =============================================================================
# 1.3 Distribuzione classi + label x source
# =============================================================================
def plot_class_distribution(manifest, kind="stacked", save_as=None):
    """
    kind = "stacked"  -> barre impilate: totale per classe + composizione per sorgente
    kind = "grouped"  -> barre affiancate: confronto diretto tra sorgenti
    """
    ct = (pd.crosstab(manifest["label"], manifest["source"])
            .reindex(CLASS_NAMES).fillna(0).astype(int))
    sources = list(ct.columns)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(sources)}
    x = np.arange(len(ct)); totals = ct.sum(axis=1).values

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if kind == "stacked":
        bottom = np.zeros(len(ct))
        for s in sources:
            vals = ct[s].values
            ax.bar(x, vals, bottom=bottom, width=0.62, label=s,
                   color=colors[s], edgecolor="white", linewidth=1.2)
            for xi, v, b in zip(x, vals, bottom):
                if v > totals.max() * 0.03:
                    ax.text(xi, b + v / 2, f"{v}", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            bottom += vals
        for xi, t in zip(x, totals):
            ax.text(xi, t + totals.max() * 0.015, f"{t}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold", color="#333")
    else:  # grouped
        w = 0.8 / len(sources)
        for i, s in enumerate(sources):
            off = (i - (len(sources) - 1) / 2) * w
            bars = ax.bar(x + off, ct[s].values, width=w * 0.95, label=s,
                          color=colors[s], edgecolor="white")
            ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(x); ax.set_xticklabels(ct.index, fontsize=11)
    ax.set_ylabel("numero di immagini", fontsize=11)
    ax.set_title("Distribuzione delle classi per sorgente", fontsize=13, fontweight="bold")
    ax.legend(title="sorgente", frameon=False, fontsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=0.25)
    ax.margins(y=0.12)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()

    print(ct.to_string())
    tot = ct.sum(axis=1)
    print(f"\nSbilanciamento (classe piu' / meno numerosa): {tot.max() / tot.min():.2f}")
    return ct


# =============================================================================
# 1.4 Intensita' per sorgente (domain shift)
# =============================================================================
def plot_intensity_by_source(manifest, n_sample_per_source=500, seed=0, save_as=None):
    rng = np.random.default_rng(seed)
    rows = []
    for src, g in manifest.groupby("source"):
        idx = g.index.to_numpy()
        if len(idx) > n_sample_per_source:
            idx = rng.choice(idx, n_sample_per_source, replace=False)
        for i in idx:
            try:
                with Image.open(manifest.loc[i, "path"]) as im:
                    rows.append({"source": src,
                                 "avg_intensity": np.asarray(im.convert("L")).mean()})
            except Exception:
                pass
    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5.5))
    palette = {s: c for s, c in zip(sorted(df["source"].unique()),
                                    ["#4C72B0", "#DD8452", "#55A868", "#C44E52"])}
    sns.kdeplot(data=df, x="avg_intensity", hue="source", fill=True,
                alpha=0.25, common_norm=False, palette=palette)
    plt.title("Distribuzione dell'intensita' media per sorgente", fontweight="bold")
    plt.xlabel("intensita' media (0-255)"); plt.ylabel("densita'")
    for sp in ["top", "right"]:
        plt.gca().spines[sp].set_visible(False)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return df


# =============================================================================
# 1.5 Dimensioni, canali, piani
# =============================================================================
def scan_dimensions(manifest):
    """Ritorna il manifest arricchito con: width, height, channels, mode, readable."""
    ws, hs, chs, modes, ok = [], [], [], [], []
    for p in manifest["path"]:
        try:
            with Image.open(p) as im:          # .size legge l'header, non carica i pixel
                w, h = im.size
                ch = len(im.getbands())
                mode = im.mode
            ws.append(w); hs.append(h); chs.append(ch); modes.append(mode); ok.append(True)
        except Exception as e:
            ws.append(np.nan); hs.append(np.nan); chs.append(np.nan)
            modes.append(f"ERR:{e}"); ok.append(False)
    df = manifest.copy()
    df["width"] = ws; df["height"] = hs
    df["channels"] = chs; df["mode"] = modes; df["readable"] = ok
    return df


def report_dimensions(df, save_as=None):
    good = df[df["readable"]].copy()
    n_bad = int((~df["readable"]).sum())
    if n_bad:
        print(f"[!] immagini illeggibili: {n_bad}")

    good["size"] = (good["width"].astype(int).astype(str) + "x" +
                    good["height"].astype(int).astype(str))
    print(f"Immagini leggibili: {len(good)}")
    print(f"Dimensioni uniche: {good['size'].nunique()}")
    print(f"Range larghezza: {int(good['width'].min())} - {int(good['width'].max())}")
    print(f"Range altezza:   {int(good['height'].min())} - {int(good['height'].max())}\n")

    print("Dimensioni piu' comuni per sorgente:")
    for src, g in good.groupby("source"):
        top = g["size"].value_counts().head(3)
        tops = ", ".join(f"{s} ({n})" for s, n in top.items())
        print(f"  {src:10s}: {tops}")

    plt.figure(figsize=(8, 6))
    for src, g in good.groupby("source"):
        plt.scatter(g["width"], g["height"], s=8, alpha=0.3, label=src)
    plt.xlabel("larghezza (px)"); plt.ylabel("altezza (px)")
    plt.title("Risoluzioni delle immagini per sorgente")
    plt.legend(); plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return good


def report_channels(df):
    good = df[df["readable"]]
    ct = pd.crosstab(good["source"], good["channels"])
    print("Canali per sorgente (1 = grigio, 3 = RGB triplicato):")
    print(ct.to_string())
    print("\n-> nel preprocessing uniformeremo tutto a 1 o 3 canali secondo il modello.")
    return ct


def show_planes_montage(manifest, source=None, n=12, ncols=4, seed=0):
    """
    Mostra una griglia di immagini (opzionalmente di una sola sorgente) per
    controllare a occhio se i piani (assiale/sagittale/coronale) sono mescolati.
    """
    sub = manifest if source is None else manifest[manifest["source"] == source]
    if len(sub) == 0:
        print(f"[montage] nessuna immagine per source={source}"); return
    rng = np.random.default_rng(seed)
    idx = rng.choice(sub.index.to_numpy(), min(n, len(sub)), replace=False)
    nrows = int(np.ceil(len(idx) / ncols))
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for k, i in enumerate(idx):
        plt.subplot(nrows, ncols, k + 1)
        try:
            with Image.open(manifest.loc[i, "path"]) as im:
                plt.imshow(np.asarray(im.convert("L")), cmap="gray")
        except Exception:
            pass
        plt.axis("off")
    plt.suptitle(f"Piani anatomici - source={source or 'tutte'}", fontsize=14)
    plt.tight_layout(); plt.show()


def channel_diff(path):
    """
    Ritorna la differenza media tra i canali R,G,B di un'immagine.
    ~0  -> i tre canali sono identici (grigio triplicato, nessun colore reale)
    >0  -> c'e' colore reale (i canali differiscono)
    Ritorna None se l'immagine ha meno di 3 canali.
    """
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float32)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float((np.abs(r - g) + np.abs(r - b) + np.abs(g - b)).mean() / 3.0)


def verify_rgb(manifest, n_sample_per_source=300, seed=0, color_threshold=1.0):
    """
    Campiona immagini a >=3 canali per sorgente e misura la differenza tra canali.
    Stampa quante sono 'grigio triplicato' vs 'colore reale'.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for src, g in manifest.groupby("source"):
        idx = g.index.to_numpy()
        if len(idx) > n_sample_per_source:
            idx = rng.choice(idx, n_sample_per_source, replace=False)
        for i in idx:
            d = channel_diff(manifest.loc[i, "path"])
            if d is not None:
                rows.append({"source": src, "path": manifest.loc[i, "path"], "chan_diff": d})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("Nessuna immagine a 3+ canali nel campione."); return df

    print("Differenza media tra canali R/G/B (campione), per sorgente:")
    print("(~0 = grigio triplicato ; >0 = colore reale)\n")
    for src, g in df.groupby("source"):
        real = int((g["chan_diff"] > color_threshold).sum())
        print(f"  {src:10s}: diff media {g['chan_diff'].mean():6.3f}  |  "
              f"con colore reale: {real}/{len(g)}")

    plt.figure(figsize=(8, 4.5))
    plt.hist(df["chan_diff"], bins=50, color="#4C72B0")
    plt.axvline(color_threshold, color="red", ls="--", label=f"soglia colore ({color_threshold})")
    plt.title("Differenza tra canali R/G/B"); plt.xlabel("differenza media"); plt.ylabel("conteggio")
    plt.legend(); plt.tight_layout(); plt.show()

    colored = df[df["chan_diff"] > color_threshold]
    if len(colored):
        print(f"\n[!] {len(colored)} immagini con colore reale nel campione: le mostro.")
        _show_paths(colored["path"].head(6).tolist(), "Immagini con COLORE reale (canali diversi)")
    else:
        print("\n[ok] tutte grigio triplicato: nessun colore reale -> la conversione a grigio non perde nulla.")
    return df


def show_rgba(manifest, n=6):
    """Trova e mostra le immagini a 4 canali (RGBA), separando i canali per capire cosa sono."""
    paths = []
    for p in manifest["path"]:
        try:
            with Image.open(p) as im:
                if len(im.getbands()) == 4:
                    paths.append(p)
        except Exception:
            pass
    print(f"Immagini a 4 canali (RGBA) trovate: {len(paths)}")
    if not paths:
        return paths
    for p in paths[:n]:
        with Image.open(p) as im:
            arr = np.asarray(im)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(arr[..., :3]); axes[0].set_title("RGB")
        for k, name in enumerate(["R", "G", "B", "Alpha"]):
            axes[k + 1].imshow(arr[..., k], cmap="gray"); axes[k + 1].set_title(name)
        for a in axes:
            a.axis("off")
        plt.suptitle(Path(p).name); plt.tight_layout(); plt.show()
    return paths


def _show_paths(paths, title):
    plt.figure(figsize=(3 * len(paths), 3))
    for k, p in enumerate(paths):
        plt.subplot(1, len(paths), k + 1)
        with Image.open(p) as im:
            plt.imshow(im)
        plt.axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.show()


# =============================================================================
# 1.6 Deduplicazione dentro e tra i dataset (pHash)
# =============================================================================
HASH_SIZE = 8               # 8x8 = 64 bit
HIGHFREQ_FACTOR = 4         # il pHash lavora su hash_size*4 e prende le basse frequenze

THRESHOLD_PRESETS = {"very-strict": 3, "strict": 5, "medium": 8, "loose": 12}

# ordine di preferenza della sorgente da TENERE quando due immagini sono duplicate
# (Figshare prima: e' la fonte piu' pulita per i tumori)
SOURCE_PRIORITY = {"figshare": 0, "br35h": 1, "sartaj": 2}

_DCT_CACHE = {}


def _dct_matrix(n):
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    return np.cos(np.pi / n * (i + 0.5) * k)


def _phash_bits(path, hash_size=HASH_SIZE, highfreq=HIGHFREQ_FACTOR):
    img_size = hash_size * highfreq
    if img_size not in _DCT_CACHE:
        _DCT_CACHE[img_size] = _dct_matrix(img_size).astype(np.float32)
    D = _DCT_CACHE[img_size]
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((img_size, img_size)),
                         dtype=np.float32)
    dct = D @ arr @ D.T
    low = dct[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    return (low > med).astype(np.uint8)


def add_phash(manifest, hash_size=HASH_SIZE):
    bits, first_err = [], None
    for p in manifest["path"]:
        try:
            bits.append(_phash_bits(p, hash_size))
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}  (es: {p})"
            bits.append(None)
    out = manifest.copy()
    out["_phash"] = bits
    out["phash_ok"] = [b is not None for b in bits]
    n_fail = int((~out["phash_ok"]).sum())
    if n_fail:
        print(f"[dedup] hash non calcolabile per {n_fail} immagini. Primo errore -> {first_err}")
    else:
        print(f"[dedup] pHash calcolato per tutte le {len(out)} immagini.")
    return out


def _stack_bits(m):
    idx = np.where(m["phash_ok"].to_numpy())[0]
    B = np.stack(m.iloc[idx]["_phash"].to_list()).astype(np.int8)
    return idx, B


def diagnose_threshold(m, sample=3000, seed=0, chunk=300, max_x=30):
    idx, B = _stack_bits(m)
    rng = np.random.default_rng(seed)
    if len(B) > sample:
        B = B[rng.choice(len(B), sample, replace=False)]
    n = len(B)
    nn = np.full(n, 10**9)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = (B[s:e, None, :] != B[None, :, :]).sum(axis=2)
        for bi in range(e - s):
            d[bi, s + bi] = 10**9
        nn[s:e] = d.min(axis=1)

    plt.figure(figsize=(9, 5))
    plt.hist(np.clip(nn, 0, max_x), bins=range(0, max_x + 2), color="#4C72B0")
    for t, c in [(3, "red"), (5, "orange")]:
        plt.axvline(t, color=c, ls="--", label=f"soglia {t}")
    plt.xlabel("distanza di Hamming dal vicino piu' prossimo (0-64 bit)")
    plt.ylabel("numero di immagini")
    plt.title("Diagnostico dedup (pHash)\nspike a sinistra = duplicati ; blocco a destra = immagini diverse")
    plt.legend(); plt.tight_layout(); plt.show()

    pct = np.percentile(nn, [1, 5, 10, 25, 50])
    print("Distanza al vicino piu' prossimo (percentili):")
    print(f"  1%={pct[0]:.0f}  5%={pct[1]:.0f}  10%={pct[2]:.0f}  25%={pct[3]:.0f}  mediana={pct[4]:.0f}")
    print("Per 'identiche o quasi' resta basso: soglia 3 (very-strict) o 5 (strict).")
    return nn


def dedup_greedy(m, threshold=3, chunk=300):
    """
    threshold: intero, oppure una chiave di THRESHOLD_PRESETS ('very-strict', ...).
    Ordina le immagini per priorita' di sorgente (Figshare prima), cosi' il
    RAPPRESENTANTE tenuto e' quello della fonte piu' pulita.
    Aggiunge le colonne: is_dup_keep, dup_of (id del rappresentante).
    """
    if isinstance(threshold, str):
        threshold = THRESHOLD_PRESETS[threshold]

    m = m.reset_index(drop=True)
    idx, B = _stack_bits(m)

    order = sorted(range(len(idx)),
                   key=lambda k: SOURCE_PRIORITY.get(m.iloc[idx[k]]["source"], 99))
    B = B[order]
    global_idx = idx[order]

    n = len(B)
    keep_bits = np.empty((n, B.shape[1]), dtype=np.int8)
    keep_rows = np.empty(n, dtype=int)
    kept = 0
    is_keep = np.ones(len(m), dtype=bool)
    dup_of = np.full(len(m), -1, dtype=int)

    for i in range(n):
        q = B[i]
        if kept == 0:
            is_dup = False
        else:
            dist = (keep_bits[:kept] != q).sum(axis=1)
            jmin = int(dist.argmin())
            is_dup = dist[jmin] <= threshold

        row = global_idx[i]
        if is_dup:
            is_keep[row] = False
            dup_of[row] = keep_rows[jmin]
        else:
            keep_bits[kept] = q
            keep_rows[kept] = row
            kept += 1

    m["is_dup_keep"] = is_keep
    m["dup_of"] = dup_of
    _report(m, threshold)
    return m


def _report(m, threshold):
    n_dups = int((~m["is_dup_keep"]).sum())
    print(f"\n[dedup] soglia={threshold} -> duplicati da rimuovere: {n_dups} / {len(m)}")

    dups = m[~m["is_dup_keep"]]
    cross = same_class = diff_class = 0
    for _, r in dups.iterrows():
        rep = m[m.index == r["dup_of"]]
        if len(rep) == 0:
            continue
        rep = rep.iloc[0]
        if rep["source"] != r["source"]:
            cross += 1
        if rep["label"] != r["label"]:
            diff_class += 1
        else:
            same_class += 1
    print(f"[dedup] di cui cross-source: {cross}")
    if diff_class:
        print(f"[!] {diff_class} duplicati hanno CLASSE DIVERSA dal loro rappresentante: "
              f"sospetti falsi positivi -> abbassa la soglia e ricontrolla.")
    else:
        print("[ok] nessun duplicato attraversa classi diverse: soglia sensata.")


def show_duplicate_examples(m, n=5, prefer_cross=True, seed=0):
    dups = m[(~m["is_dup_keep"]) & (m["dup_of"] >= 0)].copy()
    if len(dups) == 0:
        print("Nessun duplicato da mostrare."); return
    if prefer_cross:
        rep_src = m.loc[dups["dup_of"].values, "source"].values
        dups["_cross"] = rep_src != dups["source"].values
        dups = dups.sort_values("_cross", ascending=False)
    rng = np.random.default_rng(seed)
    chosen = dups.head(30).sample(min(n, len(dups)), random_state=int(rng.integers(1e6)))

    for _, r in chosen.iterrows():
        rep = m.loc[r["dup_of"]]
        plt.figure(figsize=(6, 3))
        for k, row in enumerate([rep, r]):
            plt.subplot(1, 2, k + 1)
            with Image.open(row["path"]) as im:
                plt.imshow(np.asarray(im.convert("L")), cmap="gray")
            tag = "TENUTA" if k == 0 else "rimossa"
            plt.title(f"{tag}\n{row['source']} / {row['label']}", fontsize=9); plt.axis("off")
        plt.tight_layout(); plt.show()


def apply_dedup(m, save_to=MANIFEST_DEDUP_CSV):
    keep = m[m["is_dup_keep"]].drop(columns=[c for c in ["_phash"] if c in m.columns]).reset_index(drop=True)
    print(f"\n[dedup] immagini dopo dedup: {len(keep)} / {len(m)}")
    print("\nDistribuzione classi DOPO dedup:")
    print(pd.crosstab(keep["label"], keep["source"])
            .reindex(CLASS_NAMES).fillna(0).astype(int).to_string())
    if save_to:
        keep.to_csv(save_to, index=False)
        print(f"\n[ok] manifest deduplicato salvato in {save_to}")
    return keep


def _show_pairs(m, dups, title, n, seed):
    """Disegna n coppie (rappresentante TENUTO | duplicato RIMOSSO)."""
    if len(dups) == 0:
        print(f"[{title}] nessuna coppia da mostrare.")
        return
    rng = np.random.default_rng(seed)
    take = dups.sample(min(n, len(dups)), random_state=int(rng.integers(1e6)))
    print(f"\n===== {title}  (mostro {len(take)} di {len(dups)}) =====")
    for _, r in take.iterrows():
        rep = m.loc[r["dup_of"]]
        dist = ""
        if "_phash" in m.columns and rep["_phash"] is not None and r["_phash"] is not None:
            dist = f"  |  Hamming={int((np.asarray(rep['_phash']) != np.asarray(r['_phash'])).sum())}"
        plt.figure(figsize=(6, 3.2))
        for k, row in enumerate([rep, r]):
            plt.subplot(1, 2, k + 1)
            with Image.open(row["path"]) as im:
                plt.imshow(np.asarray(im.convert("L")), cmap="gray")
            tag = "TENUTA" if k == 0 else "RIMOSSA"
            plt.title(f"{tag}\n{row['source']} / {row['label']}", fontsize=9)
            plt.axis("off")
        plt.suptitle(f"{title}{dist}", fontsize=10)
        plt.tight_layout()
        plt.show()


def inspect_dedup(m, n_cross=6, n_internal=6, seed=0):
    """
    m: il DataFrame restituito da dedup_greedy (ha is_dup_keep e dup_of).
    Mostra coppie cross-source e coppie interne separatamente, con conteggi.
    """
    dups = m[(~m["is_dup_keep"]) & (m["dup_of"] >= 0)].copy()

    rep_source = m.loc[dups["dup_of"].values, "source"].values
    rep_label = m.loc[dups["dup_of"].values, "label"].values
    dups["rep_source"] = rep_source
    dups["rep_label"] = rep_label

    cross = dups[dups["rep_source"] != dups["source"]]
    internal = dups[dups["rep_source"] == dups["source"]]

    print(f"Totale duplicati rimossi: {len(dups)}")
    print(f"  - CROSS-SOURCE (tra dataset diversi): {len(cross)}")
    print(f"  - INTERNI (stesso dataset):           {len(internal)}")

    if len(cross):
        pair = (cross["rep_source"] + "  <->  " + cross["source"]).value_counts()
        print("\nSovrapposizioni cross-source per coppia di dataset:")
        print(pair.to_string())

    _show_pairs(m, cross, "CROSS-SOURCE (stessa immagine in due dataset)", n_cross, seed)
    _show_pairs(m, internal, "INTERNE (slice quasi identici nello stesso dataset)", n_internal, seed + 1)


# =============================================================================
# USO (esegui in ordine)
# =============================================================================
if __name__ == "__main__":
    # 1.1 scompatta i dataset da Google Drive (skip automatico se gia' estratti)
    extract_all_zips()

    inspect_class_folders()
    manifest = build_manifest()
    summary(manifest)
    print(manifest.head())

    # 1.2 integrita' (sulle immagini effettivamente usate nel manifest)
    check_image_integrity(manifest["path"].tolist())

    # 1.3 distribuzione
    plot_class_distribution(manifest, kind="stacked",
                             save_as=FIGURES_DIR_TASK1 / "distribuzione_classi_per_sorgente.png")

    # 1.4 intensita' per sorgente
    plot_intensity_by_source(manifest, save_as=FIGURES_DIR_TASK1 / "distribuzione_intensita_per_sorgente.png")

    # 1.5 dimensioni / canali / piani
    df_dim = scan_dimensions(manifest)
    report_dimensions(df_dim, save_as=FIGURES_DIR_TASK1 / "distribuzione_risoluzioni_immagini.png")
    report_channels(df_dim)
    show_planes_montage(manifest, source="figshare")
    show_planes_montage(manifest, source="sartaj")
    verify_rgb(manifest)
    show_rgba(manifest)

    # 1.6 deduplicazione
    m_hash = add_phash(manifest)
    diagnose_threshold(m_hash)

    for t in [2, 3, 5]:
        print(f"===== soglia {t} =====")
        dedup_greedy(m_hash.copy(), threshold=t)
        print()

    m1 = dedup_greedy(m_hash.copy(), threshold=1)   # soglia scelta dopo l'ispezione
    inspect_dedup(m1, n_cross=6, n_internal=6)

    manifest_dedup = apply_dedup(m1)   # salva anche outputs/manifest_dedup.csv

    print(len(manifest_dedup))
    print(manifest_dedup["label"].value_counts())