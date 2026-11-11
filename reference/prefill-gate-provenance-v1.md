# Prefill gate provenance

The budgets and results retain the commit names used when the gates ran.
The integration branch rebased that history. These corresponding commits
are ancestors of the integrated branch:

| Receipt commit | Integrated commit | Verified correspondence |
|---|---|---|
| `72ba98ef12b3c3017977d79cb722fc16f833a7f2` | `eb889ca4fc61efd0d6085215980b58fc3da2d43d` | Identical `fast-defaults-gate-v1-budget.json` |
| `deff0bd284eeb4183f5aef5bd603341aec747823` | `36f717bbe1acc48c888a818440114daabbd3cffa` | Identical `near-exact-softmax-gate-v1-budget.json` |
| `a8375271001ee67f2121e3a264e52932d0105204` | `d4a9deb2d3e2fe1fe8afa21e6b5015d1820da6ba` | Identical `src/`, `Cargo.toml`, `Cargo.lock`, `build.rs` and `examples/` |
| `fce8b5c44b20e54652d74c6e038e9c74d3ad66de` | `f9ad01ec19c704e84857c5bf40cebc699c8177dd` | Identical `src/`, `Cargo.toml`, `Cargo.lock`, `build.rs` and `examples/` |

These comparisons establish the budget contents and Rust build inputs,
not equality of the entire trees: the integrated history also includes
fixture-fetching and documentation changes. The results retain the pinned
binary hashes, timestamps and historical repetition thresholds. No gate
was rerun during integration, and its verdict has not been changed.

[prefill-gate-data-v1.sha256](prefill-gate-data-v1.sha256) pins the
[archive](prefill-gate-data-v1.zip) and all six JSON members. Verify from
the repository root:

```sh
python -m zipfile -e reference/prefill-gate-data-v1.zip reference
(cd reference && sha256sum -c prefill-gate-data-v1.sha256)
```

The extracted JSON files are ignored by Git. They are the page manifests,
evaluation locks and masked locks named by the two budgets and results.
