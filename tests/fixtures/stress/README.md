# Stress-page provenance

The historical free-running comparison used four locally supplied images:
a wiki screenshot (`liber_primus_wiki_screen.png`), a scanned book page
(`scanned-book-page_ballooning.png`), a desktop mail screenshot
(`smoke_screenshot.png`) and a grocery advertisement
(`kroger-bench-09_fried.jpg`), plus deep-fried versions. Their original
sources and redistribution licences were not recorded. They are no longer
bundled in this repository, and no automated test depends on them.

The immutable
[free-running receipt](../../../reference/fast-defaults-free-running-v1.json)
retains each image's SHA-256, historical path, per-page results and the
comparison procedure. Its path names describe the inputs at run time;
they are not download locations. The `classic` and `classic-2026-11-02`
variants are separate unseeded draws, not byte-identical copies. Both sets
were removed with the original images.

[Mode fidelity](../../../docs/MODES.md#free-running-validation) summarizes
that comparison. Repeating it requires independently obtaining the exact
inputs matching the receipt's hashes; the measurements remain historical
evidence, not a runnable fixture suite.

## Historical image preparation

The original input dimensions and contents were:

| File | Size | Content |
|---|---|---|
| `liber_primus_wiki_screen.png` | 1184 × 912 | Wiki screenshot: runes, hexadecimal lines, a code block |
| `scanned-book-page_ballooning.png` | 812 × 1299 | Scanned book page with margin labels |
| `smoke_screenshot.png` | 2572 × 1874 | Desktop mail client screenshot |
| `kroger-bench-09_fried.jpg` | 800 × 465 | Grocery advertisement photo, already deep-fried |

The first three images were degraded by a local port of
deepfriedmemes.com's filter chain. That tool is not in this repository.
Every invocation used `--max-dimension 0`, preserving the input dimensions:

| Historical folder | Options | Effect |
|---|---|---|
| `fried/mild/` | `--contrast 60 --sharpen 40 --noise 8 --jpeg-quality 0.3 --jpeg-iterations-after 6` | Brightness 15, contrast 60, sharpen 40, noise 8, then 6 JPEG re-encodes at quality 30-32 |
| `fried/classic/` | Tool defaults | Brightness 15, contrast 155, sharpen 83, noise 18, then 28 JPEG re-encodes at quality 17-19 |
| `fried/enhanced/` | `--mode enhanced` | Sigmoid contrast and boosted HSV saturation, the same brightness, sharpening and noise, then the same re-encodes |
| `fried/classic-2026-11-02/` | Tool defaults | An earlier classic draw of the same three images |

Noise and JPEG-quality jitter were unseeded, so those options describe the
preparation without reproducing the recorded bytes. Comparisons used one
fresh process per arm, the same page order and flags, and a 4,096-token cap
where specified. Changed token sequences and finish reasons were read
against the image; differences confined to illegible regions did not
establish an improvement.
