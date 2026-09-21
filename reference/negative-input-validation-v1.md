# Public negative-input validation

Both bounded gaps in `image-api-requirements-audit-v1` now have executed regression evidence. The new `tests/negative_inputs.rs` suite passed **5 tests, 0 failures, 0 ignored** on native Windows and Ubuntu-24.04 under WSL, using one test thread and one scalar runner thread. The two model-dependent tests remain explicitly ignored by default; these validation runs selected them with `--include-ignored` and loaded the verified pinned assets.

The suite covers missing/empty files, a recognizable unsupported GIF header, incomplete PNG/JPEG headers, six invalid generation-option combinations, zero batch size, empty image/file collections, zero-sized RGB inputs and mixed valid/invalid first chunks. It exercises the public preparation and Runner APIs. CLI checks require nonzero error exits and empty stdout, including a valid file followed by a missing file in the first batch chunk. Missing CLI image arguments are rejected before model loading. Traced invalid RGB requests fail the test if they reach a model tensor or decode callback.

No successful OCR generation was requested or observed. No implementation defect was found, and no production source, Cargo file, existing fixture, build wrapper or captured helper was changed. All 24 recorded source/build-input hashes remained unchanged across the builds and tests.

The [receipt](negative-input-validation-v1.json), SHA256 `20266dfc48470bf8a475d16f602f5ae62535625dfe2e8358241f9b993a2ee285`, contains commands, source identities, post-test binary hashes and log hashes. Completed logs are `negative-input-validation-windows-v2.log` and `negative-input-validation-wsl-v1.log`. The preserved Windows v1 log records a wrapper invocation error before test execution; passing the literal Cargo separator through an explicit string array resolved it without source changes.

Scope remains bounded: malformed headers are not a universal policy for every imperfect but decodable image. Mixed failures are tested **within the first chunk only**; later-chunk rollback or transactionality is not claimed. These checks establish no new OCR quality, numerical parity, memory-pressure, performance, PDF, layout or HTTP result. Binary hashes were collected after completion; the source before/after check is not a hermetic startup binary attestation.

Windows reproduction:

```powershell
./scripts/build_windows.ps1 -CargoArguments @('test','--locked','--release','--test','negative_inputs','--jobs','2','--','--include-ignored','--test-threads=1','--nocapture')
```

WSL reproduction, using the existing ext4 build target and local tools:

```bash
CUDA_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/home/amazi/falcon-ocr-rust-reference/rust-target FOCR_TOOL_DIR=/home/amazi/falcon-ocr-rust-reference/build-tools bash scripts/build_linux.sh test --locked --release --test negative_inputs --jobs 2 -- --include-ignored --test-threads=1 --nocapture
```
