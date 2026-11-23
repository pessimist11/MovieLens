"""Fit pointwise rankers and choose checkpoints with full-pipeline VALID metrics."""

import copy
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits

from ..eval.metrics import evaluate_all
from ..pipeline import MMR_CANDIDATE_LIMIT, Recommender
from .base import checkpoint_path, save_checkpoint
from .features import RankFeatures
from .samples import candidate_groups, make_snapshot


def _pack(features, groups):
    rows = [features.encode(g.user, g.candidates) for g in groups]
    inputs = tuple(torch.cat(column) for column in zip(*rows))
    labels = torch.tensor([
        c.item_idx in g.truth for g in groups for c in g.candidates
    ], dtype=torch.float32)
    weights = torch.from_numpy(np.concatenate([g.weights for g in groups]))
    return inputs, labels, weights


@torch.inference_mode()
def _validate(model, packed, groups, rec):
    inputs, labels, _ = packed
    logits = torch.cat([
        model(*batch) for batch in zip(*(x.split(rec.cfg.rank.batch_size) for x in inputs))
    ])
    probabilities = logits.sigmoid().numpy()
    auc = roc_auc_score(labels, probabilities) if labels.min() != labels.max() else None
    loss = log_loss(labels, probabilities.astype(np.float64), labels=[0, 1])
    recommendations = {}
    offset = 0
    k = rec.cfg.eval.k
    for group in groups:
        end = offset + len(group.candidates)
        candidates = group.candidates
        if rec.dataset.seen_for(group.user, "valid"):
            candidates = [
                replace(c, rank_score=float(p))
                for c, p in zip(candidates, probabilities[offset:end])
            ]
            candidates.sort(key=lambda c: c.rank_score, reverse=True)
        offset = end
        result = (
            rec.reranker.rerank(candidates[:MMR_CANDIDATE_LIMIT], k)
            if rec.reranker is not None else candidates[:k]
        )
        recommendations[group.user] = [c.item_idx for c in result]

    ds = rec.dataset
    metrics = evaluate_all(
        recommendations, ds.ground_truth["valid"],
        n_items=ds.n_items, tail_items=ds.tail_items,
        item_popularity=ds.item_popularity, genre_matrix=ds.genre_matrix, k=k,
    )
    return {"auc": auc, "logloss": loss, **metrics.as_dict()}


def train_rankers(dataset, cfg, ranker_classes, *, force=False):
    paths = {cls.name: checkpoint_path(cls.name, dataset, cfg) for cls in ranker_classes}
    missing = [cls for cls in ranker_classes if force or not paths[cls.name].exists()]
    if not missing:
        return paths
    options = cfg.rank
    if min(options.epochs, options.patience, options.batch_size, options.history_size,
           options.negatives_per_user, options.snapshot_bpr_epochs,
           options.snapshot_two_tower_epochs, options.popularity_half_life_days) <= 0:
        raise ValueError("Ranking batch, history, sampling, epochs and decay settings must be positive")

    snapshot, truth, unavailable, cutoff = make_snapshot(dataset, cfg)
    training = candidate_groups(
        snapshot, truth, snapshot.dataset.seen["valid"], unavailable=unavailable,
        negative_limit=options.negatives_per_user,
    )
    rec = Recommender(dataset, replace(cfg, rank=replace(options, model="none")))
    validation = candidate_groups(rec, dataset.ground_truth["valid"], dataset.seen["valid"])
    if not training or not validation:
        raise ValueError("Training and validation must both have recalled candidates")

    inputs, labels, weights = _pack(RankFeatures(snapshot.dataset, cfg), training)
    if labels.min() == labels.max():
        raise ValueError("The ranking snapshot must contain both positive and negative candidates")
    packed_valid = _pack(RankFeatures(dataset, cfg), validation)
    print(
        f"[rank] snapshot cutoff={cutoff:.0f}, train rows={len(labels)}, "
        f"positives={int(labels.sum())}, valid rows={len(packed_valid[1])}", flush=True,
    )
    for cls in missing:
        torch.manual_seed(cfg.project.seed)
        generator = torch.Generator().manual_seed(cfg.project.seed)
        model = cls.network(dataset, cfg)
        dense = inputs[3]
        model.centre.copy_(dense.mean(0))
        model.scale.copy_(dense.std(0, unbiased=False).clamp_min(1e-3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr)
        best, best_state, history = (-1.0, -1.0), None, []
        best_epoch = 0
        for epoch in range(1, options.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in torch.randperm(len(labels), generator=generator).split(options.batch_size):
                logits = model(*(inputs_column[batch] for inputs_column in inputs))
                loss = binary_cross_entropy_with_logits(
                    logits, labels[batch], weight=weights[batch], reduction="sum"
                )
                optimizer.zero_grad()
                (loss / weights[batch].sum()).backward()
                optimizer.step()
                total_loss += loss.detach().item()

            model.eval()
            valid = _validate(model, packed_valid, validation, rec)
            row = {"epoch": epoch, "train_loss": total_loss / weights.sum().item(), **valid}
            history.append(row)
            print(
                f"[{cls.name}] epoch={epoch:2d} auc={valid['auc']} "
                f"logloss={valid['logloss']:.4f} recall={valid['recall@k']:.4f} "
                f"ndcg={valid['ndcg@k']:.4f}", flush=True,
            )
            key = (valid["recall@k"], valid["ndcg@k"])
            if key > best:
                best, best_epoch = key, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= options.patience:
                break
        model.load_state_dict(best_state)
        save_checkpoint(paths[cls.name], model, {
            "best_epoch": best_epoch,
            "valid": history[best_epoch - 1],
            "history": history,
            "snapshot_cutoff": cutoff,
            "train_rows": len(labels),
            "train_positives": int(labels.sum()),
        })
    return paths
