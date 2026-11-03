# Stress pages

Inputs for free-running before/after comparisons of numeric changes,
including out-of-distribution input: screenshots, a scanned page, a photo
and deep-fried copies of them. No test reads them, and they hold no
expected output: a comparison runs both arms on them and reads the pages
whose tokens differ against the images.

| File | Size | Content |
|---|---|---|
| `liber_primus_wiki_screen.png` | 1184 × 912 | Screenshot of a wiki page: runes, hexadecimal lines, a code block |
| `scanned-book-page_ballooning.png` | 812 × 1299 | Scanned book page with margin labels |
| `smoke_screenshot.png` | 2572 × 1874 | Screenshot of a desktop mail client |
| `kroger-bench-09_fried.jpg` | 800 × 465 | Photo of a grocery advertisement, already deep-fried |

`fried/` holds deep-fried versions of the first three, made with the
user's local deep-fryer, a port of deepfriedmemes.com's filter chain that
is not in this repository. It ran as-is through its command line with
`--max-dimension 0` (no resize, so each file keeps its source's size) and
these options per level:

| Folder | Options | Effect |
|---|---|---|
| `mild/` | `--contrast 60 --sharpen 40 --noise 8 --jpeg-quality 0.3 --jpeg-iterations-after 6` | Weaker filters, then 6 JPEG re-encodes at quality 30-32 |
| `classic/` | none (the tool's defaults) | Brightness 15, contrast 155, sharpen 83, noise 18, then 28 JPEG re-encodes at quality 17-19 |
| `enhanced/` | `--mode enhanced` | Sigmoid contrast and boosted saturation in HSV space, the same brightness, sharpening and noise, then the same re-encodes |
| `classic-2026-11-02/` | none | An earlier classic draw of the same three images |

The tool's noise and the per-pass jitter of its JPEG quality are unseeded,
so these files are the record: rerunning the tool gives other images, as
`classic/` and `classic-2026-11-02/` show.

Run a comparison with one fresh process per arm, the same pages and the
same flags, changing only the binary or the knob under test:

```sh
target/release/falcon-ocr --model artifacts/packed --mode fast run \
  tests/fixtures/stress/*.png tests/fixtures/stress/*.jpg \
  tests/fixtures/stress/fried/*/*.jpg --max-new-tokens 4096 > before.jsonl
```

Then run the changed build into `after.jsonl` and compare the two files
line by line (one line per page, in input order): `token_ids`, the first
differing token, `finish_reason`, and the text of every page that differs,
read against its image. `--max-new-tokens 4096` bounds a loop that the
repetition stop does not catch. Two readings that differ only where the
image itself is illegible say nothing about the change.
