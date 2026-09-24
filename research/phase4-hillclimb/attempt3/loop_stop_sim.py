"""Simulate a repetition stop on recorded outputs.

Rule: stop after the step where, for some period p <= PMAX, the last
`window` tokens are p-periodic with window >= max(MIN_WINDOW, REPS * p).
Reports firing position, tokens saved, and CER vs ground truth of the
truncated text (decoded with the model tokenizer).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import levenshtein  # noqa: E402

PMAX, MIN_WINDOW, REPS = 128, 256, 4


def fire(ids):
    run = [0] * (PMAX + 1)
    for i in range(1, len(ids)):
        for p in range(1, min(PMAX, i) + 1):
            run[p] = run[p] + 1 if ids[i] == ids[i - p] else 0
            if run[p] + p >= max(MIN_WINDOW, REPS * p):
                return i + 1, p  # tokens kept (through step i), period
    return None, None


def main():
    for f in sys.argv[1:]:
        report = json.loads(Path(f).read_text(encoding="utf-8"))
        paths = [i["path"] for i in report["inputs"]]
        outs = report["samples"][0]["outputs"]
        saved = total = fired = fired_eos = 0
        c_before = c_after = chars = 0
        lines = []
        for path, o in zip(paths, outs):
            ids = o["token_ids"]
            total += len(ids)
            keep, p = fire(ids)
            truth_file = Path(path).parent / "ground-truth.txt"
            truth = truth_file.read_text(encoding="utf-8") if truth_file.is_file() else None
            text_after = o["text"]
            if keep is not None:
                fired += 1
                fired_eos += o["finish_reason"] == "eos"
                saved += len(ids) - keep
                # Approximate: loop text is uniform, so truncate in proportion.
                text_after = o["text"][: round(len(o["text"]) * keep / len(ids))]
                lines.append(f"  fired {Path(path).parent.name} stop={o['finish_reason']} tokens {len(ids)} -> {keep} period {p}")
            if truth is not None:
                chars += len(truth)
                c_before += levenshtein(truth, o["text"])
                c_after += levenshtein(truth, text_after)
        print(f"{report.get('profile')}: fired on {fired} pages ({fired_eos} that ended at EOS); tokens {total} -> {total - saved} "
              f"(-{saved / total:.1%}); micro CER vs truth {c_before / chars:.2%} -> {c_after / chars:.2%}")
        print("\n".join(lines))


main()
