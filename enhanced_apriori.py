"""
Enhanced Apriori Algorithm (EAA)

Features implemented:
- Entropy-Guided Candidate Pruning (EGCP)
- Adaptive Dynamic Support (ADS)
- Vectorized Bitwise Apriori (VBA)

This file is a self-contained Python script. It generates synthetic
transaction datasets, runs both a baseline Apriori (set-based) and the
Enhanced Apriori, measures execution time across varying dataset sizes,
and plots efficiency comparison graphs.

Author: Hithesh
"""

import math
import time
import random
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

# Utility functions

def generate_random_transactions(num_transactions, num_items, min_per_tx=1, max_per_tx=30, seed=None):
    """Generate random transactions as lists of items (strings)."""
    if seed is not None:
        random.seed(seed)
    items = [f'item{i}' for i in range(num_items)]
    transactions = []
    for _ in range(num_transactions):
        k = random.randint(min_per_tx, min(max_per_tx, num_items))
        transaction = random.sample(items, k)
        transactions.append(transaction)
    return transactions

# Entropy-Guided Pruning (EGCP)

def compute_entropies(transactions):
    """Compute Shannon entropy for each item over transactions."""
    total = len(transactions)
    flat = [item for t in transactions for item in set(t)]  # count per transaction presence
    counts = Counter(flat)
    entropies = {}
    for item, cnt in counts.items():
        p = cnt / total
        if p <= 0 or p >= 1:
            H = 0.0
        else:
            H = - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
        entropies[item] = H
    return entropies


def entropy_pruning(transactions, threshold=0.0):
    """Prune items with entropy <= threshold. Returns pruned transactions.

    threshold: default 0.0 keeps all items. Typical use: 0.1-0.5
    """
    ent = compute_entropies(transactions)
    informative = {item for item, H in ent.items() if H > threshold}
    if not informative:
        # if nothing passes, return original transactions
        return transactions, ent
    pruned = [[item for item in t if item in informative] for t in transactions]
    # Also remove empty transactions
    pruned = [t for t in pruned if t]
    return pruned, ent

# Bit encoding utilities (VBA)

def encode_transactions_bit(transactions):
    """Map each item to a bit position and encode transactions as integers.

    Returns:
      encoded: list[int]  -- bitmask per transaction
      item_to_bit: dict   -- item->bit (1<<pos)
      bit_to_item: dict   -- bit->item
    """
    items = sorted({itm for t in transactions for itm in t})
    item_to_bit = {item: 1 << idx for idx, item in enumerate(items)}
    bit_to_item = {v: k for k, v in item_to_bit.items()}
    encoded = [sum(item_to_bit[i] for i in t) for t in transactions]
    return encoded, item_to_bit, bit_to_item


def bit_count(x):
    """Count number of 1-bits in integer x."""
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

# Bitwise support counting

def bitwise_support_count(encoded, candidates, min_support, total_transactions):
    """Count supports for integer bitmask candidates.

    candidates: iterable of int bitmasks
    min_support: proportion in (0,1]
    total_transactions: int

    Returns dict: {candidate_bitmask: count}
    """
    min_count = math.ceil(min_support * total_transactions)
    freq = {}
    # small optimization: iterate transactions once per candidate is unavoidable here
    for cand in candidates:
        c = 0
        # count how many transactions contain candidate (t & cand) == cand
        for t in encoded:
            if (t & cand) == cand:
                c += 1
                # optional early stop
                if c >= min_count:
                    break
        if c >= min_count:
            freq[cand] = c
    return freq


# -------------------------
# Candidate generation (bitmasks)
# -------------------------

def generate_candidates_from_prev(prev_freq_keys, k_bits_count=None):
    """Generate candidate bitmasks from previous frequent bitmasks.

    prev_freq_keys: iterable of ints
    k_bits_count: desired number of items (optional). If None, inferred
    """
    prev = list(prev_freq_keys)
    candidates = set()
    n = len(prev)
    for i in range(n):
        for j in range(i + 1, n):
            union = prev[i] | prev[j]
            # ensure we grew by combining different itemsets
            if union != prev[i] and union != prev[j]:
                if k_bits_count is None or bit_count(union) == k_bits_count:
                    candidates.add(union)
    return candidates


# -------------------------
# Adaptive Dynamic Support (ADS)
# -------------------------

def adaptive_support(base_support, k, prev_avg_count, prev_max_count, total_trans):
    """Compute adaptive min_support for level k.

    base_support: proportion (e.g., 0.05)
    prev_avg_count, prev_max_count: counts from previous level
    total_trans: total number of transactions

    Returns a proportion in (0,1].
    """
    # Avoid division by zero
    if prev_max_count <= 0:
        return base_support
    prev_avg = prev_avg_count / total_trans
    prev_max = prev_max_count / total_trans
    alpha = 0.35
    # exponential decay with level k but scaled by previous level's distribution
    factor = (prev_avg / max(prev_max, 1e-9)) * math.exp(-alpha * (k - 1))
    # clamp factor to [0.2, 2.0] to avoid extreme values
    factor = max(0.2, min(2.0, factor))
    new_support = base_support * factor
    # make sure support is not above 0.999 and not below a small floor
    new_support = max(0.001, min(0.999, new_support))
    return new_support


# -------------------------
# Enhanced Apriori (EAA): EGCP + ADS + VBA
# -------------------------

def enhanced_apriori(transactions, base_support=0.05, entropy_threshold=0.0, verbose=True):
    """Enhanced Apriori implementation returning frequent itemsets and counts.

    transactions: list of list of items
    base_support: proportion (0-1)
    entropy_threshold: prune items with entropy <= threshold

    Returns: list of dicts per level: [{bitmask: count}, ...], item maps
    """
    if verbose:
        print("[EAA] Starting. Original tx count:", len(transactions))

    # 1) Entropy-based pruning
    pruned_tx, ent = entropy_pruning(transactions, threshold=entropy_threshold)
    if verbose:
        print("[EAA] After entropy pruning tx count:", len(pruned_tx))

    if not pruned_tx:
        return [], {}, {}

    # 2) Encode into bit masks
    encoded, item_to_bit, bit_to_item = encode_transactions_bit(pruned_tx)
    total = len(encoded)

    # 3) Initial 1-item candidates
    initial_candidates = list(item_to_bit.values())
    # Count supports for single items
    level = 1
    min_support = base_support
    freq_levels = []

    freq1 = bitwise_support_count(encoded, initial_candidates, min_support, total)
    if not freq1:
        return freq_levels, item_to_bit, bit_to_item
    freq_levels.append(freq1)
    if verbose:
        print(f"[EAA] Level {level} frequent items: {len(freq1)}")

    prev_freq = freq1
    level += 1

    # 4) Iterate generating candidates and counting using bitwise ops
    while True:
        # Generate candidate bitmasks for this level
        desired_bits = level  # number of items in candidate
        candidates = generate_candidates_from_prev(list(prev_freq.keys()), k_bits_count=desired_bits)
        if not candidates:
            break

        # Adaptive min_support using counts from previous level
        prev_counts = list(prev_freq.values())
        prev_avg = sum(prev_counts) / len(prev_counts)
        prev_max = max(prev_counts)
        min_support = adaptive_support(base_support, level, prev_avg, prev_max, total)
        if verbose:
            print(f"[EAA] Level {level} candidates: {len(candidates)}, adaptive min_support: {min_support:.4f}")

        # Count supports
        freq_k = bitwise_support_count(encoded, candidates, min_support, total)
        if not freq_k:
            break

        freq_levels.append(freq_k)
        prev_freq = freq_k
        level += 1

    return freq_levels, item_to_bit, bit_to_item


# -------------------------
# Baseline Apriori (simple set-based) for comparison
# -------------------------

def support_count_set(transactions, itemset):
    return sum(1 for t in transactions if set(itemset).issubset(set(t)))


def baseline_apriori(transactions, min_support=0.05, verbose=True):
    """Simple Apriori implementation using Python sets.

    Returns list of dicts per level: [{frozenset: count}, ...]
    """
    total = len(transactions)
    items = sorted({i for t in transactions for i in t})
    # Level 1
    C1 = [frozenset([i]) for i in items]
    L1 = {}
    min_count = math.ceil(min_support * total)
    for c in C1:
        cnt = support_count_set(transactions, c)
        if cnt >= min_count:
            L1[c] = cnt
    if verbose:
        print(f"[BASE] Level 1 frequent: {len(L1)}")
    levels = []
    if not L1:
        return levels
    levels.append(L1)
    k = 2
    prev_L = L1
    while True:
        prev_list = list(prev_L.keys())
        candidates = set()
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k:
                    candidates.add(union)
        if not candidates:
            break
        Lk = {}
        for cand in candidates:
            cnt = support_count_set(transactions, cand)
            if cnt >= min_count:
                Lk[cand] = cnt
        if not Lk:
            break
        levels.append(Lk)
        prev_L = Lk
        k += 1
        if verbose:
            print(f"[BASE] Level {k-1} frequent: {len(Lk)}")
    return levels


# -------------------------
# Benchmarking & Plotting
# -------------------------

def measure_and_plot(sizes, num_items=20, base_support=0.05, entropy_threshold=0.0, seed=42):
    baseline_times = []
    enhanced_times = []
    # For reproducibility
    random.seed(seed)

    for n in sizes:
        data = generate_random_transactions(n, num_items, min_per_tx=1, max_per_tx=min(6, num_items), seed=seed)

        # Baseline
        t0 = time.time()
        baseline_apriori(data, min_support=base_support)
        t1 = time.time()
        baseline_times.append(t1 - t0)

        # Enhanced
        t0 = time.time()
        enhanced_apriori(data, base_support=base_support, entropy_threshold=entropy_threshold)
        t1 = time.time()
        enhanced_times.append(t1 - t0)

        print(f"n={n:6d} | baseline={baseline_times[-1]:.4f}s | enhanced={enhanced_times[-1]:.4f}s")

    # Plotting
    plt.figure(figsize=(9, 6))
    plt.plot(sizes, baseline_times, marker='o', label='Baseline Apriori (set-based)')
    plt.plot(sizes, enhanced_times, marker='s', label='Enhanced Apriori (EAA)')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Apriori vs Enhanced Apriori Performance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------
# Demo / main
# -------------------------

if __name__ == '__main__':
    # Parameters you can tweak
    sizes = [200,400,800,1600]  # increase if you have more CPU/time
    num_items = 25
    base_support = 0.05  # 5% support
    entropy_threshold = 0.01  # small pruning; set to 0 for no pruning

    print("Running benchmarks... (this may take a while for larger sizes)")
    measure_and_plot(sizes, num_items=num_items, base_support=base_support, entropy_threshold=entropy_threshold)
