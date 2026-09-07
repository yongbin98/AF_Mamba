import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RRIDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def get_episode_rri(data, sid, eid):
    return torch.as_tensor(data[sid]["episodes"][eid]["rri"], dtype=torch.float32).squeeze().reshape(-1)

def has_valid_af_episode(data, sid, input_segment_size=3600, prediction_horizon=3600):
    required = input_segment_size + prediction_horizon
    return any(len(get_episode_rri(data, sid, eid)) >= required for eid in data[sid]["episodes"])

def has_valid_nsr_episode(data, sid, input_segment_size=3600):
    return any(len(get_episode_rri(data, sid, eid)) >= input_segment_size for eid in data[sid]["episodes"])

def build_windows_for_subject(data, sid, input_segment_size=3600, prediction_horizon=3600):
    label = int(data[sid]["label"])
    windows = []
    for eid in sorted(data[sid]["episodes"]):
        rri = get_episode_rri(data, sid, eid)
        if label == 1:
            required = input_segment_size + prediction_horizon
            if len(rri) < required:
                continue
            w = rri[-required:-prediction_horizon]
        else:
            if len(rri) < input_segment_size:
                continue
            w = rri[:input_segment_size]
        if len(w) != input_segment_size:
            continue
        windows.append((w.clone().detach().float().unsqueeze(0), torch.tensor(label, dtype=torch.long)))
    return windows

def split_ids_balanced(id_list, fold=5):
    ids = list(id_list)
    random.shuffle(ids)
    folds = [[] for _ in range(fold)]
    if not ids:
        return tuple(folds)
    start = random.randrange(fold)
    for i, sid in enumerate(ids):
        folds[(start + i) % fold].append(sid)
    return tuple(folds)

def sub_wise_train_valid_split(data, input_segment_size=3600, prediction_horizon=3600, n_folds=5):
    dataset_names = sorted({data[sid]["dataset"] for sid in data})
    af_sets, nsr_sets = [[] for _ in range(n_folds)], [[] for _ in range(n_folds)]
    print("\n=== DATASET-STRATIFIED SUBJECT-WISE SPLITTING ===")
    for ds in dataset_names:
        af_ids = [sid for sid in data if data[sid]["dataset"] == ds and int(data[sid]["label"]) == 1
                  and has_valid_af_episode(data, sid, input_segment_size, prediction_horizon)]
        nsr_ids = [sid for sid in data if data[sid]["dataset"] == ds and int(data[sid]["label"]) == 0
                   and has_valid_nsr_episode(data, sid, input_segment_size)]
        af_folds, nsr_folds = split_ids_balanced(af_ids, n_folds), split_ids_balanced(nsr_ids, n_folds)
        for i in range(n_folds):
            af_sets[i].extend(af_folds[i])
            nsr_sets[i].extend(nsr_folds[i])
        print(f"\n{ds}: AF={len(af_ids)}, NSR={len(nsr_ids)}")
        print("  AF per fold: ", [len(x) for x in af_folds])
        print("  NSR per fold:", [len(x) for x in nsr_folds])
        if 0 < len(af_ids) < n_folds:
            print(f"  WARNING: only {len(af_ids)} AF subjects; AF cannot appear in every fold.")
        if 0 < len(nsr_ids) < n_folds:
            print(f"  WARNING: only {len(nsr_ids)} NSR subjects; NSR cannot appear in every fold.")
    print("\n=== FINAL FOLD DISTRIBUTION ===")
    for i in range(n_folds):
        print(f"Fold {i}: AF={len(af_sets[i])}, NSR={len(nsr_sets[i])}, Total={len(af_sets[i]) + len(nsr_sets[i])}")
    total_af, total_nsr = sum(map(len, af_sets)), sum(map(len, nsr_sets))
    print(f"AF subjects: {total_af}\nNSR subjects: {total_nsr}\nTotal: {total_af + total_nsr}\n")
    return tuple(af_sets), tuple(nsr_sets)

def _build_split(data, subject_ids, input_segment_size, prediction_horizon):
    windows, sids = [], []
    for sid in subject_ids:
        ws = build_windows_for_subject(data, sid, input_segment_size, prediction_horizon)
        windows.extend(ws)
        sids.extend([sid] * len(ws))
    return windows, sids

def build_datasets(data, af_sets, nsr_sets, fold_idx, input_segment_size=3600, prediction_horizon=3600, batch_size=16):
    n_folds = len(af_sets)
    if n_folds != len(nsr_sets):
        raise ValueError("af_sets and nsr_sets must have the same number of folds.")
    if not 0 <= fold_idx < n_folds:
        raise ValueError(f"fold_idx must be between 0 and {n_folds - 1}.")
    test_i, val_i = fold_idx, (fold_idx + 1) % n_folds
    train_i = [i for i in range(n_folds) if i not in (test_i, val_i)]
    train_ids = [sid for i in train_i for sid in (af_sets[i] + nsr_sets[i])]
    val_ids = af_sets[val_i] + nsr_sets[val_i]
    test_ids = af_sets[test_i] + nsr_sets[test_i]
    print(f"\n=== FOLD {fold_idx} ===")
    print(f"Subjects: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    train_windows, _ = _build_split(data, train_ids, input_segment_size, prediction_horizon)
    val_windows, val_sids = _build_split(data, val_ids, input_segment_size, prediction_horizon)
    test_windows, test_sids = _build_split(data, test_ids, input_segment_size, prediction_horizon)
    train_loader = DataLoader(RRIDataset(train_windows), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(RRIDataset(val_windows), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(RRIDataset(test_windows), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, train_ids, val_ids, test_ids, val_sids, test_sids

def pairout_split(data, heldout_af, heldout_nsr, input_segment_size, prediction_horizon, seed=42):
    af_ids = [
        sid for sid in data
        if int(data[sid]["label"]) == 1
        and has_valid_af_episode(data, sid, input_segment_size, prediction_horizon)
    ]
    nsr_ids = [
        sid for sid in data
        if int(data[sid]["label"]) == 0
        and has_valid_nsr_episode(data, sid, input_segment_size)
    ]

    test_ids = (
        [sid for sid in af_ids if data[sid]["dataset"] == heldout_af]
        + [sid for sid in nsr_ids if data[sid]["dataset"] == heldout_nsr]
    )

    af_remain = [sid for sid in af_ids if data[sid]["dataset"] != heldout_af]
    nsr_remain = [sid for sid in nsr_ids if data[sid]["dataset"] != heldout_nsr]

    rng = random.Random(seed)
    rng.shuffle(af_remain)
    rng.shuffle(nsr_remain)

    n_af = int(0.8 * len(af_remain))
    n_nsr = int(0.8 * len(nsr_remain))

    train_ids = af_remain[:n_af] + nsr_remain[:n_nsr]
    val_ids = af_remain[n_af:] + nsr_remain[n_nsr:]

    return train_ids, val_ids, test_ids


def build_datasets_pairout(data, train_ids, val_ids, test_ids, input_segment_size=3600, prediction_horizon=3600, batch_size=16):
    train_windows, val_windows, test_windows = [], [], []
    val_sids, test_sids = [], []

    for sid in train_ids:
        train_windows.extend(
            build_windows_for_subject(data, sid, input_segment_size, prediction_horizon)
        )

    for sid in val_ids:
        windows = build_windows_for_subject(data, sid, input_segment_size, prediction_horizon)
        val_windows.extend(windows)
        val_sids.extend([sid] * len(windows))

    for sid in test_ids:
        windows = build_windows_for_subject(data, sid, input_segment_size, prediction_horizon)
        test_windows.extend(windows)
        test_sids.extend([sid] * len(windows))

    train_loader = DataLoader(
        RRIDataset(train_windows),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        RRIDataset(val_windows),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        RRIDataset(test_windows),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, val_sids, test_sids