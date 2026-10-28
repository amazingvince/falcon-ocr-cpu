# Pending GPU reference work

The resident 200-page v3 FP32 reference completed successfully in
`artifacts/reference/corpus-v3-fp32-4096`: 275,903 tokens, 181 EOS stops and 19
length stops, with a longest natural output of 3733 tokens. The validated,
source-bound record summary is `gpu-corpus-v3-fp32-summary.json`; the original
run and its after-launch source archive remain unchanged. All GPU work below uses the isolated RTX 4090 UUID
`GPU-2efafa74-a255-add8-9c6c-ad80b6b8cb58`, with the existing project preflight.
Large runtime/cache files stay in the D-backed WSL ext4 environment.

Completed subsequent work:

- The reviewed startup source/runtime identity, exact completed-record checks,
  actual cuBLAS environment validation and atomic publication are active for new
  runs. Historical runs and their explicitly after-launch archives are preserved.
- Original supplement: all 15 pages, 1,056 IDs, text and EOS stops match Rust.
  See `original-supplement-fp32-redecoded-gpu-comparison-v1.json`; blank markers
  and the separately labeled CPU saved-token replay are preserved.
- Actual fused RMSNorm outputs match all 14 frozen operators exactly. All 1,444
  observed inverse-standard-deviation rows equal the independently inferred
  scales. Same-argument CUDA rsqrt on saved CUDA-shaped variance arguments also
  matches all 1,444 scales, without observing or inferring the GPU variance.
- Four projection profiler cases match their frozen outputs. A subsequent
  supported logger replay preserves both prefill outputs and kernel launches;
  actual execution logs report W13 split-K 2/in-place reduction and W2 split-K
  14/compute-type reduction. That logger capture alone did not establish
  partition membership; the later owned-workspace probe below adds conditional
  observations. See
  `linear-cublas-logged-algorithms-fp32.json`.
- Three functional originals at the exact 4096-token contract are complete:
  102 IDs, all EOS. `gpu-functional-originals-fp32-4096-validation.json` binds
  records and startup sources. Joined with the existing prose reference, all
  five saved functional invocations match: 20/20 page-results, 1,242 IDs per
  mode. See `functional-batch-v1-gpu-parity.json`; GPU prepared dimensions were
  not recorded, and this concurrent functional work is not a benchmark.
- The canonical CUDA rsqrt diagnostic passed all 1,266,679,808 positive normal
  arguments at biased exponents 104–254, with zero observed-function or integer
  exponent-mapping mismatches. `artifacts/reference/rsqrt-canonical-table-fp32-v1`
  preserves the table and exhaustive report. The completed isolated width-768
  intervention was rejected: frozen tensor failures increased from 10 to 18,
  with all 17 argmax IDs unchanged. See
  `rms-gpu-rsqrt-width768-intervention-v1.json`. No production lookup-table design
  or arithmetic change was accepted.
- Official plain-engine and direct-vLLM full-page checks both passed all three
  frozen pages: 4,881 IDs, raw text, prefixes and EOS exact. The direct serving
  audit preserves four compiled PTX files and finds no TF32 attention code.
  The project-owned server is stopped; reports are `official-fullpages-fp32.json`
  and `vllm-fullpages-fp32.json`. The failed pre-forward official harness attempt
  remains preserved separately.
- The reviewed caller-owned W2 workspace diagnostic completed with matching real
  control output, supported execution logs and kernel launches. All three coded
  basis probes agree with the one predetermined layout hypothesis. Conditional
  K membership covers thirteen ranges of 165 and a final range of 159; neither
  within-partition nor final reduction order is inferred. The complete evidence
  is `artifacts/reference/linear-owned-workspace-fp32-v1`; no search or retry was
  used to make the interpretation pass.
- The exact-context GPU run reached its requested boundary: 8,100 prefix tokens
  plus 8,284 emitted tokens, a length stop, capacity 16,384 and directly observed
  final KV cursor 16,383. Its first 8,192 IDs match the earlier reference.
  `gpu-exact-context-boundary-fp32-summary.json` binds the validated new source
  archive and records. CPU parity and natural OCR quality are separate.

- The fused BF16 observer completed with all compiled artifacts preserved, but
  was rejected under its original gate. All three complete raw BF16 outputs
  matched; natural LSE differed in 7/9/12 elements for layers 0/17/19, and
  log2 LSE in 7/11/13. The uninstrumented outputs exactly matched the frozen
  fixture. All requested stores and masked key-192 iterations were present.
  `bf16-fused-observer-v4-summary.json` binds this negative evidence; captured
  intermediates are not accepted native observations. Earlier v2/v3 artifact
  collection failures remain preserved and unqualified.
- Standalone CUDA exp2 replay completed on all 1,876 frozen arguments. Native
  `ex2.approx.ftz.f32` and `torch.exp2` agree in every FP32 and BF16 result.
  Rust differs in 1,180 FP32 results and one BF16 result at a rounding midpoint.
  The rejected fused capture was explicitly excluded. See
  `bf16-exp2-gpu-replay-v1-summary.json`; this operator result does not establish
  the arguments actually used inside the uninstrumented fused kernel.
  `bf16-exp2-midpoint-origin-v1.json` identifies the lone BF16 difference as a
  mixed CPU-score/oracle-maximum counterfactual. The corresponding actual
  saved oracle and CPU argument results each match CUDA exactly.

The single constrained follow-up also failed. Removing only the debug-only sum
consumer under the reviewed v3 partial-observation specification retained the
changed masked reduction and the same LSE mismatch counts. The new mandatory
MMA/PTX reduction-structure gate rejected it as well. Local-sum observation is
explicitly unavailable, and v2 completion is not claimed. See
`bf16-fused-observer-v5-summary.json`. No BF16 bounds or production arithmetic
were changed. The subsequent authorized unmodified PTX roundtrip passed with
exactly one pinned assembler invocation: complete cubin/SASS bytes, function
attributes, raw BF16, natural LSE and log2 LSE all match for layers 0/17/19.
`bf16-ptx-roundtrip-v1-summary.json` records this control checkpoint. One sparse
stores-only PTX observer under `bf16-ptx-observer-spec-v1.json` passed independent
source/ABI/store-coverage review and 12 host tests. Its one permitted assembly
was rejected before any observer kernel launch: the candidate has one additional
FP32 FADD under the unchanged machine-inventory gate. Source and input closure
passed; all PTX/cubin/SASS evidence is preserved. No intermediate values were
captured, no full-output acceptance is claimed, and no retry or variant was run.
See `bf16-ptx-observer-v1-rejection.json`. The SASS opcode/immediate inventory
remains a necessary check, explicitly not proof of identical machine dependency
structure. No further observer execution is authorized.

The bounded read-only diagnosis localizes the extra FADD to a duplicated exp2
argument used only by observer stores. Both additions read the same unchanged
operands; the first feeds native exp2/reduction and the second feeds debug copies.
This does not overturn the gate or establish whole-machine equivalence. See
`bf16-ptx-observer-extra-fadd-diagnosis-v1.json` and its companion note; no further
candidate was prepared or executed during that audit.

The separately authorized prospective v2 is now prepared in versioned files.
It removes exactly four masked-loop stores of `%r2099`, preserving every other
PTX byte and gate. Thirty-two argument values are explicitly unavailable across
the three cases, with boolean masks and host NaN placeholders; no missing value
is reconstructed. `bf16-ptx-observer-spec-v2.json` binds this single candidate and
the preserved v1 evidence. Ten host tests and independent frozen-source review
passed. After root release, the CRLF launcher failed before Python or CUDA;
the reviewed launcher is preserved, and root authorized an archived LF-only
execution wrapper. The one numerical assembly then failed the unchanged
machine-inventory gate with one additional FADD. No observer/control/pointwise
kernel launched, and the complete raw/natural/log2 output, mapping and runtime
coverage gates were not reached. No native intermediate was captured or
accepted. `bf16-ptx-observer-v2-rejection.json` and
`bf16-ptx-observer-v2-operational-recovery-v1.json` bind these separate events.
The RTX 4090 is idle; no further observer, diagnosis sweep, assembly variant or
retry is authorized. The BF16 numerical gate remains open.
