#!/usr/bin/env python3
"""Calcule les distances sémantiques entre prédictions et réponses de référence."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
EPSILON = 1e-12


@dataclass(frozen=True)
class SemanticDistanceResult:
    model: str
    category: str
    distances: np.ndarray

    @property
    def stats(self) -> Dict[str, float]:
        values = self.distances
        return {
            "count": float(values.size),
            "mean": float(values.mean()) if values.size else math.nan,
            "median": float(np.median(values)) if values.size else math.nan,
            "std": float(values.std(ddof=0)) if values.size else math.nan,
            "min": float(values.min()) if values.size else math.nan,
            "max": float(values.max()) if values.size else math.nan,
        }


class EmbeddingGemmaEncoder:
    """Encode les textes avec EmbeddingGemma et met en cache les vecteurs."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        auth_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache: MutableMapping[str, Tensor] = {}

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logging.info("Chargement du tokenizer %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=auth_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.info("Chargement du modèle %s sur %s", model_id, self.device)
        self.model = AutoModel.from_pretrained(
            model_id,
            token=auth_token,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        vocab_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(vocab_limit, int) and vocab_limit > 0:
            self.max_length = min(self.max_length, vocab_limit)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> Tensor:
        if not texts:
            return torch.empty((0, self.model.config.hidden_size), dtype=torch.float32)

        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length

        outputs: List[Tensor] = []
        missing_texts: List[str] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                outputs.append(cached.clone())
            else:
                outputs.append(None)  # type: ignore[arg-type]
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            new_vectors = self._compute_embeddings(missing_texts, batch_size, max_length)
            for local_idx, tensor in enumerate(new_vectors):
                text = missing_texts[local_idx]
                tensor = tensor.detach().cpu()
                self.cache[text] = tensor
                outputs[missing_indices[local_idx]] = tensor.clone()

        stacked = torch.stack(outputs, dim=0)
        return stacked

    def _compute_embeddings(self, texts: Sequence[str], batch_size: int, max_length: int) -> Tensor:
        batches = list(_batched(texts, batch_size))
        vectors: List[Tensor] = []
        total_batches = len(batches)
        for batch_idx, batch in enumerate(batches, start=1):
            print(f"[Embeddings] Batch {batch_idx}/{total_batches} (size={len(batch)})")
            tokens = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.model(**tokens)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                last_hidden = outputs.last_hidden_state
                attention_mask = tokens.get("attention_mask")
                if attention_mask is None:
                    raise RuntimeError("Le modèle ne fournit pas d'attention_mask.")
                mask = attention_mask.unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(EPSILON)
                pooled = summed / counts

            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu())

        return torch.vstack(vectors)

    @staticmethod
    def cosine_distance(left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("Les tenseurs doivent avoir la même forme pour calculer la distance.")
        left_norm = torch.nn.functional.normalize(left, dim=-1)
        right_norm = torch.nn.functional.normalize(right, dim=-1)
        similarities = (left_norm * right_norm).sum(dim=-1)
        distances = 1.0 - similarities
        return distances.clamp_min(0.0)


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalise_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value)
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower()
    if lowered in {"nan", "none", "null", ""}:
        return ""
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)
    for i in range(1, len_a + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, len_b + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev
    return prev[len_b]


def normalized_levenshtein_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    denom = max(len(a), len(b))
    return float(levenshtein_distance(a, b)) / float(denom)


def jaro_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0
    match_distance = max(len_a, len_b) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    a_matches = [False] * len_a
    b_matches = [False] * len_b

    matches = 0
    transpositions = 0

    # Count matches
    for i in range(len_a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_b)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    return (
        (matches / len_a)
        + (matches / len_b)
        + ((matches - transpositions) / matches)
    ) / 3.0


def jaro_winkler_similarity(a: str, b: str, prefix_scaling: float = 0.1) -> float:
    j = jaro_similarity(a, b)
    # common prefix length up to 4
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
            if prefix_len == 4:
                break
        else:
            break
    return j + (prefix_len * prefix_scaling * (1.0 - j))


def jaro_distance(a: str, b: str) -> float:
    return 1.0 - jaro_similarity(a, b)


def jaro_winkler_distance(a: str, b: str) -> float:
    return 1.0 - jaro_winkler_similarity(a, b)


def load_results(path: Path) -> Mapping[str, Mapping[str, Mapping[str, List[object]]]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_semantic_distances(
    data: Mapping[str, Mapping[str, Mapping[str, List[object]]]],
    embedder: EmbeddingGemmaEncoder,
    batch_size: int,
    max_length: int,
) -> Tuple[List[SemanticDistanceResult], Dict[str, Dict[str, np.ndarray]]]:
    results: List[SemanticDistanceResult] = []
    distance_map: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, categories in data.items():
        print(f"\n[Model] Début: {model_name} (categories={len(categories)})")
        for category_name, payload in categories.items():
            print(f"[Category] Début: {model_name}/{category_name}")
            if not isinstance(payload, dict):
                logging.debug("Ignoré %s/%s : format inattendu", model_name, category_name)
                continue
            true = payload.get("true")
            predicted = payload.get("predicted")
            if not isinstance(true, list) or not isinstance(predicted, list):
                logging.warning("%s/%s ne contient pas de listes true/predicted.", model_name, category_name)
                continue
            if len(true) != len(predicted):
                logging.warning(
                    "%s/%s : tailles différentes (true=%d, predicted=%d).",
                    model_name,
                    category_name,
                    len(true),
                    len(predicted),
                )
                continue
            cleaned_true = [normalise_cell(item) for item in true]
            cleaned_pred = [normalise_cell(item) for item in predicted]

            print(f"[Encode] {model_name}/{category_name}: {len(cleaned_true)} paires")
            true_vectors = embedder.encode(cleaned_true, batch_size=batch_size, max_length=max_length)
            pred_vectors = embedder.encode(cleaned_pred, batch_size=batch_size, max_length=max_length)
            distances_tensor = embedder.cosine_distance(true_vectors, pred_vectors)
            distances = distances_tensor.cpu().numpy()
            # Affiche tous les 10 exemples
            for i, d in enumerate(distances, start=1):
                if i % 10 == 0:
                    print(f"[Semantic {model_name}/{category_name}] i={i} distance={float(d):.4f}")

            result = SemanticDistanceResult(model=model_name, category=category_name, distances=distances)
            results.append(result)

            distance_map.setdefault(model_name, {})[category_name] = distances

            logging.info(
                "%s/%s : %d exemples, distance moyenne %.4f",
                model_name,
                category_name,
                distances.size,
                float(distances.mean()),
            )

    return results, distance_map


def build_summary_dataframe(results: Sequence[SemanticDistanceResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        stats = item.stats
        rows.append(
            {
                "model": item.model,
                "category": item.category,
                "count": int(stats["count"]),
                "mean_distance": stats["mean"],
                "median_distance": stats["median"],
                "std_distance": stats["std"],
                "min_distance": stats["min"],
                "max_distance": stats["max"],
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["model", "category"], inplace=True)
    return df


def plot_distributions(
    distance_map: Mapping[str, Mapping[str, np.ndarray]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logging.warning("Matplotlib est requis pour tracer les distributions: %s", exc)
        return

    if not distance_map:
        logging.warning("Aucune donnée de distance à tracer.")
        return

    ordered_models = list(distance_map.keys())
    ordered_categories = list(next(iter(distance_map.values())).keys()) if ordered_models else []
    max_categories = max((len(categories) for categories in distance_map.values()), default=0)

    n_rows = len(ordered_models)
    n_cols = max_categories
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for row_idx, model_name in enumerate(ordered_models):
        categories = distance_map[model_name]
        col_names = list(categories.keys())
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(col_names):
                ax.axis("off")
                continue
            category_name = col_names[col_idx]
            distances = categories[category_name]
            if distances.size == 0:
                ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.hist(distances, bins=40, range=(0.0, 2.0), color="#286CD3", alpha=0.85)
            mean_value = float(distances.mean())
            ax.axvline(mean_value, color="#D33F49", linestyle="--", linewidth=1.5, label=f"μ={mean_value:.3f}")
            ax.set_title(f"{model_name}\n{category_name}")
            ax.set_xlabel("Distance cosinus")
            ax.set_ylabel("Fréquence")
            ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("Distributions sauvegardées dans %s", output_path)


def _write_category_progress_csv(
    model_name: str,
    category_name: str,
    distances: np.ndarray,
    output_dir: Path,
) -> None:
    """Crée/écrit un CSV par catégorie au fur et à mesure.

    Fichier: <output_dir>/<model>_<category>.csv
    Colonnes: index, distance, rolling_mean
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}_{category_name}.csv"

    # Prépare le DataFrame pour la catégorie courante
    df = pd.DataFrame({
        "index": np.arange(1, distances.size + 1, dtype=int),
        "distance": distances.astype(float),
    })
    if not df.empty:
        df["rolling_mean"] = df["distance"].expanding(min_periods=1).mean()
    else:
        df["rolling_mean"] = []

    # Écrit en écrasant pour garantir la cohérence de l'entête
    df.to_csv(file_path, index=False)
    logging.info("CSV catégorie mis à jour: %s", file_path)


def _write_char_category_csv(
    model_name: str,
    category_name: str,
    true_values: Sequence[str],
    pred_values: Sequence[str],
    metrics: Sequence[str],
    base_output_dir: Path,
) -> None:
    """Écrit un CSV par catégorie avec distances caractère pour chaque métrique.

    Fichier: <base_output_dir>/<char_output_subdir>/<model>_<category>_char.csv
    Colonnes: index, true, predicted, <metric> pour chaque métrique, et <metric>_rolling_mean
    """
    if len(true_values) != len(pred_values):
        return

    # Prépare répertoires
    char_dir = base_output_dir
    char_dir.mkdir(parents=True, exist_ok=True)

    file_path = char_dir / f"{model_name}_{category_name}_char.csv"

    rows: List[Dict[str, object]] = []
    # Map nom->fonction
    metric_funcs = {
        "levenshtein": normalized_levenshtein_distance,
        "jaro": jaro_distance,
        "jaro_winkler": jaro_winkler_distance,
    }
    use_metrics = [m for m in metrics if m in metric_funcs]

    # Calcule lignes
    for idx, (t, p) in enumerate(zip(true_values, pred_values), start=1):
        row: Dict[str, object] = {"index": idx, "true": t, "predicted": p}
        for m in use_metrics:
            row[m] = float(metric_funcs[m](t, p))
        rows.append(row)
        if idx % 10 == 0:
            preview = {m: f"{row[m]:.4f}" for m in use_metrics}
            print(f"[Char {model_name}/{category_name}] i={idx} metrics={preview}")

    if not rows:
        pd.DataFrame(columns=["index", "true", "predicted", *use_metrics]).to_csv(file_path, index=False)
        logging.info("CSV char catégorie mis à jour (vide): %s", file_path)
        return

    df = pd.DataFrame(rows)
    for m in use_metrics:
        mean_col = f"{m}_rolling_mean"
        df[mean_col] = df[m].expanding(min_periods=1).mean()

    df.to_csv(file_path, index=False)
    logging.info("CSV char catégorie mis à jour: %s", file_path)


def _write_model_aggregated_csv(
    model_name: str,
    model_categories: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Crée/écrit un CSV agrégé par modèle au fur et à mesure.

    Fichier: <output_dir>/<model>.csv
    - Une colonne par catégorie contenant les distances par ligne (exemple)
    - Une colonne supplémentaire par catégorie '<category>_mean' (moyenne cumulée)
    """
    if not model_categories:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}.csv"

    # Détermine le nombre maximal de lignes parmi les catégories
    category_names = list(model_categories.keys())
    max_len = max((arr.size for arr in model_categories.values()), default=0)
    if max_len == 0:
        # Écrit un fichier vide si nécessaire
        pd.DataFrame().to_csv(file_path, index=False)
        logging.info("CSV modèle mis à jour (vide): %s", file_path)
        return

    # Construit le DataFrame aligné sur max_len
    data: Dict[str, List[float]] = {}
    for category in category_names:
        arr = model_categories[category].astype(float)
        padded = np.full((max_len,), np.nan, dtype=float)
        padded[: arr.size] = arr
        data[category] = padded.tolist()

    df = pd.DataFrame(data)
    # Ajoute les moyennes cumulées par catégorie
    for category in category_names:
        mean_col = f"{category}_mean"
        if df[category].notna().any():
            df[mean_col] = df[category].expanding(min_periods=1).mean()
        else:
            df[mean_col] = np.nan

    df.to_csv(file_path, index=False)
    logging.info("CSV modèle mis à jour: %s", file_path)


def _update_summary_csv(summary_path: Path, result: SemanticDistanceResult) -> None:
    """Met à jour (upsert) le CSV de synthèse après chaque catégorie."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Construit une ligne de synthèse pour ce couple (modèle, catégorie)
    stats = result.stats
    row = {
        "model": result.model,
        "category": result.category,
        "count": int(stats["count"]),
        "mean_distance": stats["mean"],
        "median_distance": stats["median"],
        "std_distance": stats["std"],
        "min_distance": stats["min"],
        "max_distance": stats["max"],
    }

    if summary_path.exists():
        try:
            existing = pd.read_csv(summary_path)
        except Exception:
            existing = pd.DataFrame(columns=list(row.keys()))
    else:
        existing = pd.DataFrame(columns=list(row.keys()))

    # Upsert
    mask = (existing.get("model") == row["model"]) & (existing.get("category") == row["category"]) if not existing.empty else None
    if existing.empty:
        updated = pd.DataFrame([row])
    else:
        if mask is not None and mask.any():
            existing.loc[mask, list(row.keys())] = list(row.values())
            updated = existing
        else:
            updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

    # Tri pour stabilité
    if not updated.empty and {"model", "category"}.issubset(updated.columns):
        updated.sort_values(["model", "category"], inplace=True)

    updated.to_csv(summary_path, index=False)
    logging.info("Synthèse mise à jour: %s", summary_path)


def _update_char_summary_csv(
    summary_path: Path,
    model_name: str,
    category_name: str,
    rows: pd.DataFrame,
    metrics: Sequence[str],
) -> None:
    """Met à jour (upsert) le CSV de synthèse caractère par (modèle, catégorie)."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Calcule stats par métrique
    summary_row: Dict[str, object] = {
        "model": model_name,
        "category": category_name,
        "count": int(rows.shape[0]),
    }
    for m in metrics:
        if m in rows.columns and not rows[m].empty:
            values = rows[m].astype(float).to_numpy()
            summary_row[f"{m}_mean"] = float(np.mean(values))
            summary_row[f"{m}_median"] = float(np.median(values))
            summary_row[f"{m}_std"] = float(np.std(values, ddof=0))
            summary_row[f"{m}_min"] = float(np.min(values))
            summary_row[f"{m}_max"] = float(np.max(values))
        else:
            summary_row[f"{m}_mean"] = math.nan
            summary_row[f"{m}_median"] = math.nan
            summary_row[f"{m}_std"] = math.nan
            summary_row[f"{m}_min"] = math.nan
            summary_row[f"{m}_max"] = math.nan

    # Charge existant
    if summary_path.exists():
        try:
            existing = pd.read_csv(summary_path)
        except Exception:
            existing = pd.DataFrame(columns=list(summary_row.keys()))
    else:
        existing = pd.DataFrame(columns=list(summary_row.keys()))

    # Upsert
    if existing.empty:
        updated = pd.DataFrame([summary_row])
    else:
        # Assure colonnes
        for col in summary_row.keys():
            if col not in existing.columns:
                existing[col] = math.nan
        mask = (existing.get("model") == summary_row["model"]) & (existing.get("category") == summary_row["category"]) if not existing.empty else None
        if mask is not None and mask.any():
            existing.loc[mask, list(summary_row.keys())] = list(summary_row.values())
            updated = existing
        else:
            updated = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)

    if not updated.empty and {"model", "category"}.issubset(updated.columns):
        updated.sort_values(["model", "category"], inplace=True)

    updated.to_csv(summary_path, index=False)
    logging.info("Synthèse char mise à jour: %s", summary_path)


def process_and_write_incrementally(
    data: Mapping[str, Mapping[str, Mapping[str, List[object]]]],
    embedder: EmbeddingGemmaEncoder,
    batch_size: int,
    max_length: int,
    output_dir: Path,
    summary_path: Path,
    plot_path: Path,
    char_output_dir: Path,
    char_summary_path: Path,
    char_metrics: Sequence[str],
) -> Tuple[List[SemanticDistanceResult], Dict[str, Dict[str, np.ndarray]]]:
    """Boucle principale avec écritures CSV et plots au fur et à mesure."""
    results: List[SemanticDistanceResult] = []
    distance_map: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, categories in data.items():
        for category_name, payload in categories.items():
            if not isinstance(payload, dict):
                logging.debug("Ignoré %s/%s : format inattendu", model_name, category_name)
                continue
            true = payload.get("true")
            predicted = payload.get("predicted")
            if not isinstance(true, list) or not isinstance(predicted, list):
                logging.warning("%s/%s ne contient pas de listes true/predicted.", model_name, category_name)
                continue
            if len(true) != len(predicted):
                logging.warning(
                    "%s/%s : tailles différentes (true=%d, predicted=%d).",
                    model_name,
                    category_name,
                    len(true),
                    len(predicted),
                )
                continue

            cleaned_true = [normalise_cell(item) for item in true]
            cleaned_pred = [normalise_cell(item) for item in predicted]

            true_vectors = embedder.encode(cleaned_true, batch_size=batch_size, max_length=max_length)
            pred_vectors = embedder.encode(cleaned_pred, batch_size=batch_size, max_length=max_length)
            distances_tensor = embedder.cosine_distance(true_vectors, pred_vectors)
            distances = distances_tensor.cpu().numpy()

            result = SemanticDistanceResult(model=model_name, category=category_name, distances=distances)
            results.append(result)

            # Met à jour la carte de distances
            distance_map.setdefault(model_name, {})[category_name] = distances

            # Écritures/plots incrémentaux
            _write_category_progress_csv(model_name, category_name, distances, output_dir)
            print(f"[File] CSV catégorie écrit pour {model_name}/{category_name}")
            _write_model_aggregated_csv(model_name, distance_map[model_name], output_dir)
            print(f"[File] CSV modèle mis à jour: {model_name}.csv")
            _update_summary_csv(summary_path, result)
            print(f"[File] Synthèse mise à jour: {summary_path.name}")
            plot_distributions(distance_map, plot_path)
            print(f"[Plot] Distributions sauvegardées: {plot_path.name}")

            # Distances caractère (basé sur all_results.json, pas sur les CSV existants)
            _write_char_category_csv(
                model_name=model_name,
                category_name=category_name,
                true_values=cleaned_true,
                pred_values=cleaned_pred,
                metrics=char_metrics,
                base_output_dir=char_output_dir,
            )
            print(f"[File] CSV char catégorie écrit pour {model_name}/{category_name}")
            # Recharge le CSV écrit pour calculer la synthèse
            char_file = char_output_dir / f"{model_name}_{category_name}_char.csv"
            try:
                char_rows = pd.read_csv(char_file)
            except Exception:
                char_rows = pd.DataFrame(columns=["index", "true", "predicted", *char_metrics])
            _update_char_summary_csv(
                summary_path=char_summary_path,
                model_name=model_name,
                category_name=category_name,
                rows=char_rows,
                metrics=char_metrics,
            )
            print(f"[File] Synthèse char mise à jour: {char_summary_path.name}")

            logging.info(
                "%s/%s : %d exemples, distance moyenne %.4f",
                model_name,
                category_name,
                distances.size,
                float(distances.mean()) if distances.size else float("nan"),
            )

    return results, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("results/020925/all_results.json"),
        help="Chemin vers le fichier JSON contenant les prédictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/020925"),
        help="Dossier de sortie pour les graphiques et tableaux.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Identifiant Hugging Face du modèle EmbeddingGemma à utiliser.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Périphérique PyTorch (ex: 'cpu', 'cuda', 'cuda:0'). 'auto' choisit automatiquement.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Taille de lot pour le calcul des embeddings.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Longueur maximale des séquences tokenisées.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="semantic_distance_summary.csv",
        help="Nom du fichier CSV de synthèse.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="semantic_distance_histograms.png",
        help="Nom du fichier image pour les distributions.",
    )
    parser.add_argument(
        "--char-output-subdir",
        type=str,
        default="char_metrics",
        help="Sous-dossier pour enregistrer les CSV et graphiques de distances caractère.",
    )
    parser.add_argument(
        "--char-summary-csv",
        type=str,
        default="char_distance_summary.csv",
        help="Nom du fichier CSV de synthèse pour les distances caractère.",
    )
    parser.add_argument(
        "--char-metrics",
        nargs="*",
        default=["levenshtein", "jaro", "jaro_winkler"],
        help="Liste des métriques caractère à calculer (levenshtein, jaro, jaro_winkler).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    auth_token = (
        os.getenv("HF_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    if not args.json_path.exists():
        raise SystemExit(f"Fichier introuvable: {args.json_path}")

    logging.info("Chargement des résultats depuis %s", args.json_path)
    data = load_results(args.json_path)

    embedder = EmbeddingGemmaEncoder(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        auth_token=auth_token,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / args.summary_csv
    plot_path = output_dir / args.plot_name
    char_output_dir = output_dir / args.char_output_subdir
    char_output_dir.mkdir(parents=True, exist_ok=True)
    char_summary_path = char_output_dir / args.char_summary_csv

    # Traitement incrémental: écrit CSV et plots à chaque catégorie
    results, distance_map = process_and_write_incrementally(
        data=data,
        embedder=embedder,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=output_dir,
        summary_path=summary_path,
        plot_path=plot_path,
        char_output_dir=char_output_dir,
        char_summary_path=char_summary_path,
        char_metrics=args.char_metrics,
    )

    # Affiche une synthèse finale (le CSV de synthèse a déjà été mis à jour au fil de l'eau)
    summary_df = build_summary_dataframe(results)
    if summary_df.empty:
        logging.warning("Aucun résultat calculé.")
    else:
        logging.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
