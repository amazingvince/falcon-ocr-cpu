"""Inspect one staged compact attention head; no builds/model/timing execution.

This execution-only helper binds its own source at inspection startup/end.
Its identity is separate from the previously frozen build preparation.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import uuid

import capture
from capture import ROOT, closure, read, sha, write, need, digest
from patch import TEST_NAMES, TEST_FILTER, INPUT_PINS

NAME = 'falcon_ocr::kernels::attention64_staged::compact_head'
PDBUTIL = Path('C:/Program Files/LLVM/bin/llvm-pdbutil.exe')
OBJDUMP = Path('C:/Program Files/LLVM/bin/llvm-objdump.exe')
UTILITY_SOURCE = 'research/benchmarks/experiments/attention64_temporal/inspect_compiled.py'
UTILITY_SOURCE_SHA = 'f26676f14edd9a6a1a38fd974483c2e72f271e000b3bdc260e496004ae8da24b'
# PE parsing and instruction summarization copied from the pinned source above.
# It is not imported at runtime. These local bytes are bound by this file hash.

def save(output, name, text):
    with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def command(args):
    return subprocess.check_output([str(v) for v in args], text=True, timeout=180)


def pe_identity(raw):
    need(len(raw) >= 64 and raw[:2] == b'MZ', 'Invalid DOS header')
    pe = struct.unpack_from('<I', raw, 0x3c)[0]
    need(raw[pe:pe + 4] == b'PE\0\0', 'Invalid PE')
    count = struct.unpack_from('<H', raw, pe + 6)[0]
    opt_size = struct.unpack_from('<H', raw, pe + 20)[0]
    opt = pe + 24
    need(struct.unpack_from('<H', raw, opt)[0] == 0x20b, 'Expected PE32+')
    base = struct.unpack_from('<Q', raw, opt + 24)[0]
    sections = []
    for i in range(count):
        pos = opt + opt_size + 40 * i
        virtual_size, rva, rawsize, fileoffset = struct.unpack_from('<IIII', raw, pos + 8)
        need(fileoffset + rawsize <= len(raw), 'Truncated PE section')
        sections.append((rva, rawsize, fileoffset, virtual_size))

    def file_offset(rva, length):
        found = [off + rva - start for start, size, off, _ in sections
                 if start <= rva and rva + length <= start + size]
        need(len(found) == 1, 'RVA extent unmapped/ambiguous')
        return found[0]

    debug_rva, debug_size = struct.unpack_from('<II', raw, opt + 112 + 6 * 8)
    need(debug_size > 0 and debug_size % 28 == 0, 'Invalid debug directory extent')
    debug_offset = file_offset(debug_rva, debug_size)
    rsds = []
    for pos in range(debug_offset, debug_offset + debug_size, 28):
        kind, length, _, pointer = struct.unpack_from('<IIII', raw, pos + 12)
        need(pointer + length <= len(raw), 'Truncated debug record')
        if kind == 2 and raw[pointer:pointer + 4] == b'RSDS':
            need(length >= 25, 'Truncated CodeView RSDS')
            rsds.append({'guid': str(uuid.UUID(bytes_le=raw[pointer + 4:pointer + 20])).upper(),
                         'age': struct.unpack_from('<I', raw, pointer + 20)[0],
                         'original_path': raw[pointer + 24:pointer + length].rstrip(b'\0').decode('utf-8')})
    need(len(rsds) == 1, 'Expected one CodeView RSDS record')
    return base, sections, rsds[0]


def summarize_instructions(instructions):
    """Descriptive evidence only; do not infer a reduction from opcode counts."""
    calls = [{'address': hex(a), 'instruction': m, 'operand': op}
             for a, m, op in instructions if m.startswith('call')]
    indirect = [{'address': hex(a), 'instruction': m, 'operand': op}
                for a, m, op in instructions if (m.startswith('call') or m.startswith('jmp'))
                and op.lstrip().startswith('*')]
    fmas = [{'address': hex(a), 'instruction': m, 'operand': op}
            for a, m, op in instructions if m.startswith('vfmadd')]
    horizontal = [{'address': hex(a), 'instruction': m, 'operand': op}
                  for a, m, op in instructions if m.startswith(('vhadd', 'vextractf128', 'vaddps'))]
    # RBP can be a general-purpose pointer; even RSP vector accesses can be
    # callee-save prologue/epilogue traffic. Preserve addresses for human
    # dataflow review instead of mislabeling every access as an accumulator spill.
    stack_candidates = [{'address': hex(a), 'instruction': m, 'operand': op}
                        for a, m, op in instructions if m.startswith('v')
                        and re.search(r'%[er](?:sp|bp)\b', op)]
    loops = []
    for address, mnemonic, operand in instructions:
        target = re.match(r'0x([0-9a-fA-F]+)\b', operand)
        if not mnemonic.startswith('j') or not target:
            continue
        start = int(target[1], 16)
        if start >= address:
            continue
        body = [(a, m, op) for a, m, op in instructions if start <= a <= address]
        count = sum(m.startswith('vfmadd') for _, m, _ in body)
        if count:
            loops.append({'start': hex(start), 'back_edge': hex(address),
                          'branch': mnemonic, 'fma_instruction_count': count,
                          'calls': [c for c in calls if start <= int(c['address'], 16) <= address],
                          'vector_stack_address_candidates': [v for v in stack_candidates
                              if start <= int(v['address'], 16) <= address]})
    return {'calls': calls, 'indirect_call_or_tail_transfers_among_parsed_instructions': indirect,
            'no_indirect_call_or_tail_transfer_among_parsed_instructions': not indirect,
            'complete_decoded_byte_coverage': 'unverified',
            'fma_instructions': fmas, 'horizontal_reduction_instruction_candidates': horizontal,
            'vector_stack_address_candidates': stack_candidates,
            'backward_branch_ranges_containing_fma': loops,
            'spill_classification': 'unresolved_requires_register_dataflow_review',
            'staged_pv_recurrence_equivalence_from_disassembly': 'not_automatically_claimed'}


def check_bound(bound):
    for path, expected in bound.items():
        need(sha(path) == expected, 'Input/source/tool changed: ' + str(path))


def inputs(args):
    directory = args.prepared.resolve()
    need(directory.is_relative_to(ROOT) and directory != ROOT, 'Prepared directory must be inside workspace')
    bound = {Path(__file__).resolve(): sha(Path(__file__).resolve()),
             directory / 'preparation.json': args.preparation_sha256,
             directory / 'build.json': args.build_sha256,
             directory / 'operators.json': args.operators_sha256}
    prep = read(directory / 'preparation.json', args.preparation_sha256)
    build = read(directory / 'build.json', args.build_sha256)
    ops = read(directory / 'operators.json', args.operators_sha256)
    need(prep['kind'] == 'attention64-staged-preparation-v1'
         and build['kind'] == 'attention64-staged-build-v1', 'Wrong experiment receipts')
    need(build['status'] == 'built_not_executed'
         and build['preparation_sha256'] == args.preparation_sha256, 'Build/preparation mismatch')
    need(ops['kind'] == 'attention64-staged-operators-v1'
         and ops['status'] == 'passed' and ops['exit_code'] == 0
         and ops['source_closure'] is True
         and ops['all_selected_tests_actually_passed'] is True
         and ops['operator_fixture_sha256'] == INPUT_PINS
         and ops['full_model_or_benchmark_executed'] is False, 'Operators not accepted')
    need(ops['preparation_sha256'] == args.preparation_sha256
         and ops['build_sha256'] == args.build_sha256, 'Operator evidence belongs to another build')
    need(len(TEST_NAMES) == len(set(TEST_NAMES)) == 17
         and all(name.startswith(TEST_FILTER) for name in TEST_NAMES)
         and ops['tests'] == sorted(TEST_NAMES) and ops['selected_test_count'] == len(TEST_NAMES)
         and prep['selected_tests'] == list(TEST_NAMES), 'Exact operator inventory differs')
    operator = build['executables']['operators']
    operator_path = (directory / operator['path']).resolve()
    need(operator_path.parent == directory, 'Unexpected operator executable path')
    need(ops['binary_sha256'] == operator['sha256'], 'Operators used another binary')
    bound[operator_path] = operator['sha256']
    path = directory / 'operator.log'
    raw = path.read_bytes()
    need(digest(raw) == ops['log_sha256'], 'Operator log changed')
    log = raw.decode('utf-8')
    need('17 passed; 0 failed' in log
         and all(log.splitlines().count(f'test {name} ... ok') == 1 for name in TEST_NAMES),
         'Missing actual operator completions')
    bound[path] = ops['log_sha256']
    closure(directory, prep)
    for name, value in prep['experiment_source_sha256'].items():
        bound[ROOT / name] = value
    bound[directory / 'source.zip'] = prep['source_archive_sha256']
    bound[capture.CONTROL / 'build.json'] = capture.CONTROL_BUILD
    bound[capture.CONTROL / 'source.zip'] = capture.CONTROL_ARCHIVE
    benchmark_dir = directory / 'benchmark-build'
    benchmark_path = benchmark_dir / 'build.json'
    benchmark = read(benchmark_path, build['benchmark_build_sha256'])
    need(benchmark['status'] == 'complete' and benchmark['source_unchanged_during_build'] is True
         and benchmark['build_exit_code'] == 0 and benchmark['example'] == 'ocr_bench', 'Benchmark build incomplete')
    control = read(capture.CONTROL / 'build.json', capture.CONTROL_BUILD)
    names = set(control['source_sha256'])
    need(benchmark['source_sha256'] == {n: prep['project_source_sha256'][n] for n in names},
         'Benchmark source inventory differs from candidate')
    changed = sorted(n for n in names if benchmark['source_sha256'][n] != control['source_sha256'][n])
    need(changed == ['src/kernels.rs'], 'Staged benchmark changed sources other than kernels')
    exe = (benchmark_dir / benchmark['binary']).resolve()
    archive = (benchmark_dir / benchmark['source_archive']).resolve()
    need(exe.parent == benchmark_dir and archive.parent == benchmark_dir, 'Unexpected preserved build paths')
    bound.update({benchmark_path: build['benchmark_build_sha256'], exe: benchmark['binary_sha256'],
                  archive: benchmark['source_archive_sha256']})
    check_bound(bound)
    return directory, prep, build, benchmark, exe, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--operators-sha256', required=True)
    parser.add_argument('--pdb', type=Path, required=True)
    parser.add_argument('--pdb-sha256', required=True)
    args = parser.parse_args()
    need(sys.platform == 'win32', 'Native Windows PE/PDB inspection expected')
    directory, prep, build, benchmark, exe, bound = inputs(args)
    pdb = args.pdb.resolve()
    target = Path(build['target_directory']).resolve()
    need(pdb == target / 'release/examples/ocr_bench.pdb', 'Expected matching emitted benchmark PDB path')
    bound.update({pdb: args.pdb_sha256, PDBUTIL: sha(PDBUTIL), OBJDUMP: sha(OBJDUMP)})
    check_bound(bound)
    raw = exe.read_bytes()
    need(digest(raw) == benchmark['binary_sha256'], 'Executable changed during read')
    benchmark_dir = directory / 'benchmark-build'
    output = benchmark_dir / 'compiled-staged-head'
    output.mkdir(exist_ok=False)
    report = {'kind': 'attention64-staged-compiled-head-v1', 'status': 'failed',
              'function': NAME, 'preparation_sha256': args.preparation_sha256,
              'build_sha256': args.build_sha256, 'operators_sha256': args.operators_sha256,
              'benchmark_build_sha256': build['benchmark_build_sha256'],
              'input_sha256': {str(p): h for p, h in bound.items()},
              'copied_utility_provenance': {'path': UTILITY_SOURCE, 'sha256': UTILITY_SOURCE_SHA,
                  'runtime_import': False,
                  'change': 'Staged single-union operator/build validation, exact staged symbol and PV-specific descriptive labels; PE/PDB parser and opcode summary limitations retained.'},
              'helper_provenance': 'Separate inspection-start/end closure; not retroactively included in build preparation.',
              'source_and_input_closure': False, 'performance_claim': False,
              'new_build_model_or_benchmark_execution': False,
              'limitations': [
                  'Only one exact PDB-resolved function is inspected; missing/inlined/ambiguous is unresolved, without alternate searches.',
                  'Opcode counts do not prove register-resident outputs, absence of per-key output traffic, or unchanged FMA recurrence; manual review is required.',
                  'Parsed instruction addresses are checked against the PDB extent; complete decoded byte coverage is unverified, so transfer summaries concern only parsed instructions.',
                  'Vector stack references may be Windows callee-save traffic or pointer addressing, not arithmetic spills.',
                  'PDB section/offset/extent identifies the function; objdump nearest-export labels are not authoritative.',
                  'Source/operator/full-model parity and performance remain separate evidence.']}
    try:
        preserved = benchmark_dir / 'ocr_bench.pdb'
        if preserved.exists():
            need(sha(preserved) == args.pdb_sha256, 'Preserve existing different symbol file')
        else:
            pdb_bytes = pdb.read_bytes()
            need(digest(pdb_bytes) == args.pdb_sha256, 'PDB changed during read')
            with preserved.open('xb') as stream:
                stream.write(pdb_bytes)
        bound[preserved] = args.pdb_sha256
        report.update(preserved_pdb=str(preserved), pdb_sha256=args.pdb_sha256)
        base, sections, identity = pe_identity(raw)
        summary = command([PDBUTIL, 'dump', '-summary', preserved])
        save(output, 'pdb-summary.txt', summary)
        need('GUID: {' + identity['guid'] + '}' in summary
             and re.search(r'Age:\s*' + str(identity['age']) + r'\b', summary), 'PE/PDB identity mismatch')
        report.update(pe_codeview=identity, pdb_identity_matches_pe=True)
        symbols = command([PDBUTIL, 'dump', '-symbols', preserved])
        matches = list(re.finditer(r'S_[LG]PROC32(?:_ID)?[^\n]*`' + re.escape(NAME) + r'`\n([^\n]+)', symbols))
        if len(matches) != 1:
            save(output, 'symbol-search.txt', '\n'.join(m.group() for m in matches)
                 + '\nExact matching symbol count: ' + str(len(matches)) + '\n')
            report.update(status='unresolved_symbol', exact_symbol_matches=len(matches))
        else:
            symbol = matches[0].group()
            save(output, 'staged-head-symbol.txt', symbol + '\n')
            address = re.search(r'addr = (\d+):(\d+), code size = (\d+)', symbol)
            need(address is not None, 'No function extent')
            section, offset, size = map(int, address.groups())
            need(1 <= section <= len(sections) and size > 0, 'Invalid symbol section/size')
            rva, rawsize, _, _ = sections[section - 1]
            need(offset + size <= rawsize, 'Function exceeds materialized section')
            start, stop = base + rva + offset, base + rva + offset + size
            dump_args = [OBJDUMP, '-d', '--no-show-raw-insn', '--start-address=' + hex(start),
                         '--stop-address=' + hex(stop), exe]
            disassembly = command(dump_args)
            save(output, 'staged-head-disassembly.txt', disassembly)
            instructions = []
            for line in disassembly.splitlines():
                match = re.match(r'^\s*([0-9a-fA-F]+):\s+(\w+)\s*(.*)', line)
                if match:
                    instructions.append((int(match[1], 16), match[2], match[3]))
            need(instructions and instructions[0][0] == start
                 and all(start <= a < stop for a, _, _ in instructions), 'Unexpected disassembly extent')
            report.update(status='inspected_pending_dataflow_review', section=section,
                          section_offset_decimal=offset, code_bytes=size,
                          virtual_address=hex(start), stop_address=hex(stop),
                          objdump_command=[str(v) for v in dump_args],
                          output_register_residency='unresolved_requires_register_dataflow_review',
                          absence_of_per_key_output_memory_traffic='unresolved_requires_register_dataflow_review',
                          probability_calls_outside_pv_loop='unresolved_requires_register_dataflow_review',
                          **summarize_instructions(instructions))
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        closure(directory, prep)
        check_bound(bound)
        for name, expected in artifacts.items():
            need(sha(output / name) == expected, 'Saved inspection artifact changed: ' + name)
        need({p.name for p in output.iterdir() if p.is_file()} == set(artifacts), 'Unexpected saved inspection files')
        report.update(source_and_input_closure=True,
                      input_sha256={str(p): h for p, h in bound.items()}, output_sha256=artifacts)
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        if 'output_sha256' not in report:
            report['output_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write(output / 'symbol-receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(output / 'symbol-receipt.json')}))


if __name__ == '__main__':
    main()
