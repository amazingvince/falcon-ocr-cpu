# Read-only diagnosis of the rejected PTX observer

The extra FADD is a repeated exp2-argument subtraction used by debug stores.
This is a static finding in the preserved assembly. The observer remains rejected
before launch; no native intermediate or full-output acceptance was obtained.
The source-bound details are in `bf16-ptx-observer-extra-fadd-diagnosis-v1.json`.

All FADD count differences localize to generated source line 416: 128 in the
original and 129 in the observer. The candidate computes
`FADD R146, R62, -R228` at `0x1e390`, then consumes and overwrites that result
with `MUFU.EX2 R146, R146` at `0x1e400`. The resulting probability feeds the
native reduction at `0x1e4a0`.

At `0x1e840`, it repeats the same subtraction into `R110`. The source registers
have identical preceding definitions and neither is overwritten between the two
adds; no control transfer intervenes. The second result feeds four debug stores,
directly or through integer copies. No floating-point consumer of that repeated
result was found before its aliases are overwritten.

The debug offsets join this value unambiguously to original virtual register
`%r2099`, defined by `sub.f32 %r2099, %r1795, %r2086`. These operands are the
masked-loop log2 score and zero-protected maximum. The native exp2 consumes that
same virtual register. The four observer destinations concern rows 13/head 1,
13/head 2 in layer 0 and row 41/heads 6 and 5 in layer 19; each selected thread
contributes one of local key columns 0, 2, 4 and 6.

This supports compiler rematerialization for observation, rather than an added
operation in the original denominator recurrence. It does not establish complete
dependency equivalence across the two machine-code allocations, and it does not
relax the failed inventory gate.

If separately authorized, one narrowly reduced prospective observation could
omit just the masked-loop debug stores of `%r2099`, explicitly marking those
argument values unavailable. That could remove this observed pressure to
recompute, but a changed allocation may behave differently. Any future candidate
would still require independent review and the unchanged inventory and complete
output gates. No such candidate, assembly or launch was performed in this audit.
