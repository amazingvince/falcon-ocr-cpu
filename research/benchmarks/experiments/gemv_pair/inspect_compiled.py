"""Offline inspection of one PDB-resolved GEMV block; never builds or runs a model."""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import uuid

from run_smoke import (read, sha, write, need, parser, helper_sources, validated_inputs,
                       check_bound, close_inputs, output_closure, digest)

NAME = 'falcon_ocr::kernels::gemv_pair_candidate::channel_block'
PDBUTIL = Path('C:/Program Files/LLVM/bin/llvm-pdbutil.exe')
OBJDUMP = Path('C:/Program Files/LLVM/bin/llvm-objdump.exe')


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
    return {'calls': calls, 'indirect_call_or_tail_transfers': indirect,
            'no_indirect_call_or_tail_transfer_in_function': not indirect,
            'fma_instructions': fmas, 'horizontal_reduction_instructions': horizontal,
            'vector_stack_address_candidates': stack_candidates,
            'backward_branch_ranges_containing_fma': loops,
            'spill_classification': 'unresolved_requires_register_dataflow_review',
            'paired_reduction_equivalence_from_disassembly': 'not_automatically_claimed'}


def main():
    p = parser(__doc__)
    p.add_argument('--pdb', type=Path, required=True, help='Matching emitted benchmark PDB; never rebuilt here')
    p.add_argument('--pdb-sha256', required=True, help='Prospectively selected symbol identity')
    args = p.parse_args()
    helpers = helper_sources(Path(__file__).resolve())
    directory, prep, build, bound = validated_inputs(args, helpers)
    need(sys.platform == 'win32', 'Native PE/PDB tools expected')
    benchmark_dir = directory / 'benchmark-build'
    benchmark_path = benchmark_dir / 'build.json'
    benchmark = read(benchmark_path, build['benchmark_build_sha256'])
    need(benchmark['status'] == 'complete' and benchmark['source_unchanged_during_build'] is True,
         'Incomplete benchmark binary capture')
    need(all(prep['project_source_sha256'].get(n) == h for n, h in benchmark['source_sha256'].items()),
         'Benchmark sources differ from prepared candidate')
    exe = benchmark_dir / benchmark['binary']
    need(exe.resolve().parent == benchmark_dir.resolve(), 'Unexpected preserved executable path')
    pdb = args.pdb.resolve()
    target = Path(build['target_directory']).resolve()
    need(pdb.is_relative_to(target) and pdb.name == 'ocr_bench.pdb', 'PDB must belong to this fresh build target')
    bound.update({benchmark_path: build['benchmark_build_sha256'], exe: benchmark['binary_sha256'],
                  pdb: args.pdb_sha256, PDBUTIL: sha(PDBUTIL), OBJDUMP: sha(OBJDUMP)})
    check_bound(bound)
    raw = exe.read_bytes()
    need(digest(raw) == benchmark['binary_sha256'], 'Executable changed during read')
    output = benchmark_dir / 'compiled-channel-block'
    output.mkdir(exist_ok=False)
    report = {'kind': 'gemv-pair-compiled-channel-block-v1', 'status': 'failed',
              'function': NAME, 'preparation_sha256': args.preparation_sha256,
              'build_sha256': args.build_sha256, 'operators_sha256': args.operators_sha256,
              'benchmark_build_sha256': build['benchmark_build_sha256'],
              'input_sha256': {str(p): h for p, h in bound.items()},
              'helper_source_sha256': {str(p): h for p, h in helpers.items()},
              'source_and_input_closure': False, 'performance_claim': False,
              'limitations': [
                  'Only one PDB-resolved function is inspected; no build, model or benchmark executes.',
                  'FMA/opcode counts alone do not prove paired reduction equivalence or absence of all spills.',
                  'Vector stack references may include Windows nonvolatile-register saves, not accumulator spills.',
                  'Objdump nearest-export labels are not authoritative; PDB section/offset/extent identifies the function.',
                  'If the function is inlined, missing or ambiguous, evidence is unresolved; no alternate function search.']}
    try:
        preserved = benchmark_dir / 'ocr_bench.pdb'
        if preserved.exists():
            need(sha(preserved) == args.pdb_sha256, 'Existing symbol file has a different identity; preserve it')
        else:
            pdb_bytes = pdb.read_bytes()
            need(digest(pdb_bytes) == args.pdb_sha256, 'PDB changed during read')
            with preserved.open('xb') as stream:
                stream.write(pdb_bytes)
        bound[preserved] = args.pdb_sha256
        report['preserved_pdb'] = str(preserved)
        report['pdb_sha256'] = args.pdb_sha256
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
            save(output, 'channel-block-symbol.txt', symbol + '\n')
            address = re.search(r'addr = (\d+):(\d+), code size = (\d+)', symbol)
            need(address is not None, 'No function extent')
            section, offset, size = map(int, address.groups())
            need(1 <= section <= len(sections) and size > 0, 'Invalid symbol section/size')
            rva, rawsize, _, _ = sections[section - 1]
            need(offset + size <= rawsize, 'Function exceeds materialized section')
            start, stop = base + rva + offset, base + rva + offset + size
            args_dump = [OBJDUMP, '-d', '--no-show-raw-insn', '--start-address=' + hex(start),
                         '--stop-address=' + hex(stop), exe]
            disassembly = command(args_dump)
            save(output, 'channel-block-disassembly.txt', disassembly)
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
                          objdump_command=[str(v) for v in args_dump],
                          **summarize_instructions(instructions))
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        close_inputs(directory, prep, bound)
        output_closure(output, artifacts)
        report.update(source_and_input_closure=True,
                      input_sha256={str(p): h for p, h in bound.items()}, output_sha256=artifacts)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if 'output_sha256' not in report:
            report['output_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write(output / 'symbol-receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(output / 'symbol-receipt.json')}))


if __name__ == '__main__':
    main()
