# Frozen mixed-page functional stress selection

The [input lock](functional-batch-v1-lock.json) preserves the original benchmark's
content-selected prose page plus the named white blank, sparse line and cafe
receipt. Existing model output lengths, agreement or timings did not select a
replacement. Initial execution is one mixed batch of four, at FP32/AVX2,
four threads, minimum dimension 64, maximum dimension 1536 and cap 4096.

| Input | Source pixels | Prepared pixels | Prefix tokens | Prefix + cap |
| --- | ---: | ---: | ---: | ---: |
| Prose `3f294b5e60a0c2d4` | 1653 x 2339 | 1088 x 1536 | 6544 | 10640 |
| `blank-white` | 1024 x 1280 | 1024 x 1280 | 5136 | 9232 |
| `sparse-room` | 1024 x 768 | 1024 x 768 | 3088 | 7184 |
| `receipt-cafe` | 640 x 960 | 640 x 960 | 2416 | 6512 |

All budgets fit the 16384-token model limit. Full source-lock page-object hashes
bind category/annotation membership without copying source text. Source files,
canonical PNG bytes, decoded RGB bytes and ground-truth hashes were rechecked.
The three existing supplement replay records also passed 225 lineage/result
checks; their shapes support these expectations, but their cap 512 outputs are
context only.

Every selected input requires a **new sequential expanded/unpacked baseline
from the same captured binary** with the exact 4-thread/1536/4096 contract.
Earlier EOS does not waive cap, binary or source equivalence. Joint candidates
must compare actual literal text, IDs, stops, counts, dimensions, precision and
free-running status against those fresh baselines. No inference was started by
this selection task.

At batch 4, analytic cache payload reservation is 6,049,759,232 bytes for expanded
KV and 3,799,121,920 bytes for compact KV. Largest shared prefill workspace payload
is 381,960,192 bytes; largest hidden state and all prepared patches add further
payload. These estimates exclude allocator overhead, native scratch, thread
stacks, image buffers, logits/positions, optional packed weight copies and
mapped-versus-owned model residency. Reservation is not measured RSS or peak
memory. The harness must require at least 12 GiB available physical memory before
each process; that guard is not a maximum-memory guarantee.

This is a functional comparison of one natural page and three authored fixtures.
It establishes neither representative throughput nor corpus OCR quality. The
input lock contains no timing results, and the natural source retains its
research/noncommercial restrictions.

Lock SHA256: `c9ba1bcbadb7ffe8865aa8a149a75fb239053fb79955d86083c7a28b4de3fb72`.
