# Prefill gate provenance

This note supplies the complete commit mapping for the fast-defaults and
near-exact-softmax v1 receipts. It supersedes
[prefill-gate-provenance-v2.md](prefill-gate-provenance-v2.md). The original
receipts and their commit names remain unchanged.

The mapping covers all ten project commit identities cited in the eleven
tracked `fast-defaults-*.json` and `near-exact-softmax-*.json` receipts,
including nested strings and binary paths, and all six JSON members of
[prefill-gate-data-v1.zip](prefill-gate-data-v1.zip). Page identifiers,
image hashes, binary hashes and external dataset/model revisions are not
project commits. They retain their recorded meaning.

Each integrated commit below is an ancestor of the integration branch.
“Build inputs” means the complete `src/` and `examples/` trees plus
`Cargo.toml`, `Cargo.lock` and `rust-toolchain.toml`, compared byte for byte
between the two commits. “Reference” means the complete `reference/` tree
at those commits. The implementation and registration patches marked as
equal also have identical stable patch IDs over the source and reference
paths.

| Receipt commit | Integrated commit | Role | Verified equality |
|---|---|---|---|
| `72ba98ef12b3c3017977d79cb722fc16f833a7f2` | `eb889ca4fc61efd0d6085215980b58fc3da2d43d` | Fast gate budget registration | Build inputs and reference |
| `deff0bd284eeb4183f5aef5bd603341aec747823` | `36f717bbe1acc48c888a818440114daabbd3cffa` | Near-exact gate budget registration | Build inputs and reference |
| `a8375271001ee67f2121e3a264e52932d0105204` | `d4a9deb2d3e2fe1fe8afa21e6b5015d1820da6ba` | Fast gate and free-running binary source | Build inputs and reference |
| `fce8b5c44b20e54652d74c6e038e9c74d3ad66de` | `f9ad01ec19c704e84857c5bf40cebc699c8177dd` | Near-exact gate binary source | Build inputs; reference differs as described below |
| `70dcefef4b6e1a70b016555057c9e923abbd99c2` | `a4fe7996b63f1b05627c47569730d6243be87439` | Fast prefill default selection | Build inputs and reference; same implementation patch |
| `c859bd2b0147be09de4d196f21cb8c4d46f8673a` | `3fc38b49a629c87f1a90209a45656925a28d72fc` | Counting-stop introduction | Build inputs and reference; same implementation patch |
| `c7c64f94ca3b7743ad31aae33e0e63c54d156c2c` | `100beb2d709b1f5dde4ddd586e1cd76ad1e4515f` | Counting-stop threshold of 64 | Build inputs and reference; same implementation patch |
| `533469aa4fbc55e1b25867b2c233b740e069cfc1` | `3ac8b5ccaaa7a6a4525cd2ccd681ad880355f375` | Near-exact gate binary name and equivalent source | Build inputs and reference |
| `48d2c828155f39c408e0d6279eb704532e6855e0` | `b60481f30f61d4bddde9a4b5e5700ab3a5d5664e` | Near-exact page/lock registration | Build inputs and reference; same registration patch |
| `5ee86b9e4376d9261c853bdb42c737e80aa7b327` | `92138438a644578176d81b8576e56c1cc3816d45` | BF16 projection removal | Build inputs and reference; same implementation patch |

The two budget files themselves are unchanged between the receipt commit,
the integrated commit and the current tree. The three archived fast-gate
JSON members are byte-identical to their files at `72ba98e`; the three
near-exact members are byte-identical to their files at `48d2c82`.

`fce8b5c` and `533469a` have identical build inputs, as the near-exact
budget records. Their mapped commits preserve those inputs. The
`fce8b5c` → `f9ad01e` comparison differs under `reference/` because the
integrated history already contains the near-exact calibration lock,
duplicates, evaluation lock, manifest, masked lock and smoke lock. The old
source commit precedes those six files. This is a registration-order
difference, not an equality claim for the full trees.

Budget registration, binary source and later policy decisions are
separate events. The fast-gate binary predates the counting rule. The
near-exact gate binary has the 32-cycle rule; the cited 64-cycle change is
a later policy commit, not the source of that binary. The fast-default
selection and BF16 projection removal likewise describe decisions rather
than changes to a pinned binary. The receipts retain their original
binary hashes, timestamps, thresholds and verdicts. These source
comparisons do not rebuild the historical binaries or rerun a gate.

The historical receipt commits are retained by local `checkpoint/*` tags;
they are not supplied by a normal clone. The source comparisons below
require those old-side objects, so an ordinary clone cannot independently
recompute the correspondence. Current integrated ancestry and the archive
checksums can be checked independently from an ordinary clone of the
integration branch. With the historical objects available, use the commit
pairs in the table to verify build-input equality from the repository root.
For example:

```sh
git merge-base --is-ancestor 3ac8b5c HEAD
git diff --exit-code 533469a 3ac8b5c -- \
  src Cargo.toml Cargo.lock rust-toolchain.toml examples reference
```

The archive and all six extracted members are pinned by
[prefill-gate-data-v1.sha256](prefill-gate-data-v1.sha256). Verify from the
repository root:

```sh
python -m zipfile -e reference/prefill-gate-data-v1.zip reference
(cd reference && sha256sum -c prefill-gate-data-v1.sha256)
```

The extracted JSON files are ignored by Git. The checksums, original
receipts and this source map serve different purposes: file identity,
recorded gate evidence and correspondence of source states.
