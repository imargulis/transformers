## Overview

The cache storage abstraction now supports two caching strategies:

### 1. **K/V Pair Caching** (Traditional - Default)
- Caches the final key and value tensors after projection
- Zero recomputation cost
- Higher memory usage
- **Best for**: Standard attention, most use cases

### 2. **Hidden State Caching** (New)
- Caches pre-projection hidden states  
- Materializes to K/V pairs on-demand
- Lower memory usage, higher compute
- **Best for**: Memory-constrained scenarios


## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Attention Layer                          │
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Hidden    │                                                │
│  │   States    │                                                │
│  │ [B, S, H]   │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         │  Cache here? ────┐                                    │
│         │                  │                                    │
│         ├──────────────────┼─────────────┐                      │
│         │                  │             │                      │
│    ┌────▼────┐        ┌────▼────┐   ┌───▼────┐                  │
│    │ Q Proj  │        │ K Proj  │   │ V Proj │                  │
│    │         │        │         │   │        │                  │
│    └────┬────┘        └────┬────┘   └───┬────┘                  │
│         │                  │            │                       │
│    ┌────▼────┐        ┌────▼────┐   ┌───▼────┐                  │
│    │ Queries │        │  Keys   │   │ Values │                  │
│    │[B,H,S,D]│        │[B,H,S,D]│   │[B,H,S,D]                  │
│    └────┬────┘        └────┬────┘   └───┬────┘                  │
│         │                  │            │                       │
│         │                  └────────────┘                       │
│         │                       │                               │
│         │              Or cache here? (traditional)             │
│         │                       │                               │
│         └───────────────────────┘                               │
│                      │                                          │
│                 ┌────▼──────┐                                   │
│                 │ Attention │                                   │
│                 │  Scores   │                                   │
│                 └───────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Storage Strategy Comparison

### Strategy 1: K/V Pair Caching (Traditional)

```
                                                ┌──────────────┐
                                                │ Hidden State │  (Not cached)
                                                └──────┬───────┘
                                                       │ Project (once)
                            ---------------------------------------------------------
                            |                                                       |
                            ▼                                                       ▼                         
                    ┌──────────────┐                                         ┌──────────────┐
                    │  Key Tensor  │ ◄─── CACHED                             │ Value Tensor │ ◄─── CACHED  
                    └──────────────┘                                         └──────────────┘
                            │                                                       │
                            ▼                                                       ▼
                                     (Used for attention - no recomputation)
```

**Characteristics:**
- ✅ Zero recomputation
- ❌ Higher memory (stores full K/V tensors)
- ✅ Fast access
- **Best for:** Standard self-attention

---

### Strategy 2: Hidden State Caching (New)

```
                                                ┌──────────────┐
                                                │ Hidden State │  ◄─── CACHED
                                                └──────┬───────┘
                                                       │ Project (on every retrieve!)
                            ---------------------------------------------------------
                            |                                                       |
                            ▼                                                       ▼                         
                    ┌──────────────┐                                         ┌──────────────┐
                    │  Key Tensor  │ ◄─── (Computed on-demand)               │ Value Tensor │ ◄─── (Computed on-demand)  
                    └──────────────┘                                         └──────────────┘
                            │                                                       │
                            ▼                                                       ▼
                                     (Used for attention - recomputation needed)

```

**Characteristics:**
- ❌ Requires recomputation (projection)
- ✅ Lower memory (stores only hidden states)
- ❌ Slower access (must materialize)
- **Best for:** memory-constrained enviroments

---

## Architecture

```
CacheStorage (Abstract Base)
├── cache_stage: "kv_pairs" or "hidden_states"
├── retrieve(materialize: bool)
└── materialize_to_kv(**kwargs)

KVStorage (K/V Pair Caching)
├── keys: Tensor
└── values: Tensor

HiddenStateStorage (Hidden State Caching)
├── hidden_states: Tensor
├── k_proj: Projection layer
├── v_proj: Projection layer
└── materialize_to_kv() → (keys, values)

StaticTensorStorage (Static K/V with torch.compile support)
QuantizedStorage (Quantized K/V storage)
├── QuantoQuantizedStorage
└── HQQQuantizedStorage
```


## Implementation Architecture

```
CacheStorage (Abstract Base Class)
├── cache_stage: str
│   ├── "kv_pairs" ──► Stores final K/V tensors
│   └── "hidden_states" ──► Stores pre-projection hidden states
│
├── initialize(key_states, value_states, **kwargs)
├── store(key_states, value_states, **kwargs)
├── retrieve(materialize: bool, **kwargs) ──► Returns (keys, values)
├── materialize_to_kv(hidden_states, **kwargs) ──► Projects to K/V
├── get_seq_length() → int
├── reset()
├── offload()
├── prefetch(device)
└── reorder(beam_idx)

┌────────────────────────┐    ┌──────────────────────────┐
│   KVStorage            │    │  HiddenStateStorage      │
├────────────────────────┤    ├──────────────────────────┤
│ cache_stage="kv_pairs" │    │ cache_stage="hidden_     │
│                        │    │            states"       │
│ keys: Tensor           │    │ hidden_states: Tensor    │
│ values: Tensor         │    │ k_proj: Layer/Weight     │
│                        │    │ v_proj: Layer/Weight     │
│ retrieve():            │    │                          │
│   return keys, values  │    │ retrieve(materialize):   │
│                        │    │   if materialize:        │
│                        │    │     return proj(hidden)  │
│                        │    │   else:                  │
│                        │    │     return hidden, None  │
└────────────────────────┘    └──────────────────────────┘
          │                              │
          └───────────┬──────────────────┘
                      │
          ┌───────────▼─────────────┐
          │  CacheLayerMixin        │
          │  (Uses storage)         │
          └─────────────────────────┘
                      │
          ┌───────────┴──────────────┐
          │                          │
     ┌────▼─────┐             ┌──────▼─────┐
     │ Dynamic  │             │   Static   │
     │ Layer    │             │   Layer    │
     └──────────┘             └────────────┘
```

---

## Data Flow Examples

### Example 1: K/V Caching (DynamicLayer)

```
Step 1: Initialize
┌─────────────┐
│ layer.update│
│ (keys, vals)│
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  KVStorage   │
│ .initialize()│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ keys: Tensor │ Stored in memory
│ values:      │
│   Tensor     │
└──────────────┘

Step 2: Retrieve
┌──────────────┐
│ layer.update │
│ (new keys)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  KVStorage   │
│ .retrieve()  │ ──► Direct return (fast!)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Return cached│
│ keys, values │
└──────────────┘
```

### Example 2: Hidden State Caching

```
Step 1: Initialize
┌───────────────┐
│ storage.init  │
│ (hidden,      │
│  k_proj,      │
│  v_proj)      │
└───────┬───────┘
        │
        ▼
┌────────────────┐
│ HiddenState    │
│ Storage        │
│ .initialize()  │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ hidden_states: │ Stored
│   Tensor       │
│ k_proj: Layer  │ Reference stored
│ v_proj: Layer  │ Reference stored
└────────────────┘

Step 2: Retrieve (Materialize)
┌────────────────┐
│ storage.       │
│ retrieve(      │
│   materialize  │
│   =True)       │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ HiddenState    │
│ Storage        │
│ .materialize() │ ──► Compute projections!
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ keys = k_proj  │ Computed on-the-fly
│   (hidden)     │
│ values = v_proj│
│   (hidden)     │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Return         │
│ (keys, values) │
└────────────────┘
```

---

## Memory Layout Comparison (TODO: review)

### Example: Llama-2 Style (GQA - 8 KV heads, 32 Q heads)

```
Configuration:
- Batch size: 1
- Sequence length: 1024 tokens
- Hidden dimension: 4096
- Q heads: 32, KV heads: 8
- Head dimension: 128

┌────────────────────────────────────────────────────┐
│ K/V Pair Caching:                                  │
├────────────────────────────────────────────────────┤
│ Keys:   [1, 8, 1024, 128]  = 1,048,576 values      │
│ Values: [1, 8, 1024, 128]  = 1,048,576 values      │
│ Total:  2,097,152 values × 4 bytes = 8.00 MB       │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Hidden State Caching:                              │
├────────────────────────────────────────────────────┤
│ Hidden: [1, 1024, 4096]    = 4,194,304 values      │
│ Total:  4,194,304 values × 4 bytes = 16.00 MB      │
└────────────────────────────────────────────────────┘

Memory Ratio: 16 / 8 = 2x MORE for hidden state caching!
❌ Not beneficial for this case
```

---

## Performance Characteristics

| Metric | KVStorage | HiddenStateStorage |
|--------|-----------|-------------------|
| Memory | Higher | Lower |
| Access Speed | Fast (O(1)) | Slower (O(n) for projection) |
| Recomputation | None | Every retrieve() call |
| Best For | Self-attention | Cross-attention, memory-constrained |
| Complexity | Simple | More complex |
| torch.compile | ✅ Supported | ✅ Supported |

---

## Use Cases

| Scenario | Recommended Storage | Reason |
|----------|-------------------|---------|
| Standard Self-Attention | `KVStorage` | Most efficient |
| Multi-Query Attention (MQA) | `KVStorage` | K/V already compressed |
| Grouped-Query Attention (GQA) | `KVStorage` | K/V already compressed |
| Cross-Attention (Encoder-Decoder) | `HiddenStateStorage` | Encoder states reused many times |
| Memory-Constrained | `HiddenStateStorage` | Trade compute for memory |
| Very Long Context | Hybrid (future) | Different strategies for different ranges |

## Future Enhancements

1. **Hybrid Caching**: Combine strategies for different context ranges
   - Recent tokens: K/V cache (fast)
   - Distant tokens: Hidden state cache (compressed)

2. **Quantized Hidden States**: Further memory compression

3. **Fused Kernels**: Optimize materialization overhead

4. **Adaptive Strategy**: Automatically choose based on:
   - Available memory
   - Reuse frequency
   - Model architecture (MQA/GQA/MHA)

## Summary

The multi-stage cache architecture provides **flexibility** to choose the right caching strategy based on:
- 📊 **Memory constraints**
- ⚡ **Compute availability**  
- 🔄 **Reuse frequency**
- 🏗️ **Model architecture** (MHA, MQA, GQA)

**Default recommendation**: Use `KVStorage` unless you have specific memory constraints or are working with cross-attention.
