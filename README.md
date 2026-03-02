# Enhanced Apriori Algorithm (EAA)

**Performance: 8x speedup over traditional Apriori**

## Author
Hithesh Reddy Satti  
B.E. Computer Science, Osmania University  
Guide: Prof. M. A. Hameed

## Overview
Implementation of three novel optimizations to the classical Apriori frequent 
itemset mining algorithm, achieving significant performance improvements through 
information-theoretic pruning, adaptive thresholds, and bitwise operations.

## Key Innovations

### 1. Entropy-Guided Candidate Pruning (EGCP)
- Uses Shannon entropy to filter low-information items
- Reduces candidate generation overhead
- Maintains high-quality frequent itemsets

### 2. Adaptive Dynamic Support (ADS)
- Dynamically adjusts min_support thresholds per level
- Uses exponential decay scaled by previous level statistics
- Prevents premature termination while maintaining efficiency

### 3. Vectorized Bitwise Apriori (VBA)
- Replaces set-based operations with bitwise integer operations
- Achieves O(1) subset checking via bitmasks
- Significantly reduces memory access patterns

## Performance Results

| Dataset Size | Baseline | Enhanced | Speedup |
|--------------|----------|----------|---------|
| 1,600 txns   | 0.200s   | 0.025s   | **8x**  |

**Configuration:** 25 items, 5% min_support, synthetic transactions

## Usage
```python
python enhanced_apriori.py
```

## Requirements
- Python 3.8+
- matplotlib
- Standard library only (collections, itertools, math)

## Technical Details
- **Time Complexity:** Improved from O(n·k²) to O(n·k) for candidate generation
- **Space Complexity:** Reduced through bitwise encoding
- **Scalability:** Tested on datasets up to 6,400 transactions

## Citation
If you use this work, please cite:
```
Satti, Hithesh Reddy(2024). Enhanced Apriori Algorithm with Entropy-Guided Pruning
and Adaptive Support. University College of Engineering, Osmania University.
```
