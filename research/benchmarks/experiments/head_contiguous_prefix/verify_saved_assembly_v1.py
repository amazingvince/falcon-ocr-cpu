"""One supplementary rerender; never rewrite original inspection artifacts."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = ROOT / 'artifacts/diagnostics/head-contiguous-prefix-v1/benchmark-build/compiled-prefix-head/symbol-receipt.json'
ORIGINAL_SHA = '20874cba1c991caeaee0cf1589e1b3a0c0d9a684b5ff3d180c9acf5543eebec2'
SOURCE = ROOT / 'research/benchmarks/experiments/head_contiguous_prefix/inspect_compiled.py'
SOURCE_SHA = '3c2670ce91fb7905205d91ea07de90f0cc1886ace116e72f90f7902147f90e96'
OUTPUT = ROOT / 'reference/head-contiguous-prefix-assembly-artifact-verification-v1.json'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    need(not OUTPUT.exists(), 'Preserve previous supplementary attempt')
    raw = ORIGINAL.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'Original receipt changed')
    original = json.loads(raw)
    bound = {Path(p): h for p, h in original['input_sha256'].items()}
    bound.update({ORIGINAL: ORIGINAL_SHA, Path(__file__).resolve(): sha(__file__)})
    for name, h in original['output_sha256'].items():
        bound[ORIGINAL.parent / name] = h
    report = {'kind': 'head-contiguous-prefix-assembly-artifact-verification-v1', 'status': 'started',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'original_receipt_sha256': ORIGINAL_SHA,
              'inspector_source_sha256': SOURCE_SHA, 'commands': [], 'artifacts': [],
              'original_files_rewritten': False, 'model_build_benchmark_or_profile_execution': False,
              'source_input_tool_output_closure': False,
              'limits': ['Supplementary post-inspection regeneration checks exact saved bytes; it does not retroactively establish the original write-time observation window.',
                         'This reuses three pure functions extracted from the exact pinned inspector source; it tests saved-output consistency, not an independent PE parser or machine-dataflow proof.',
                         'Original parsed-instruction, complete-byte-coverage and address/dataflow limitations remain.']}
    failure = None
    try:
        need(original['status'] == 'inspected_pending_dataflow_review' and original['source_and_input_closure'] is True,
             'Original inspection incomplete')
        need(bound[SOURCE] == SOURCE_SHA, 'Inspector source identity differs')
        for path, expected in bound.items():
            need(sha(path) == expected, 'Before identity mismatch: ' + str(path))
        source = SOURCE.read_bytes()
        need(hashlib.sha256(source).hexdigest() == SOURCE_SHA, 'Source changed during read')
        names = {'pe_identity', 'summarize_instructions', 'direct_call_targets'}
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
        need({node.name for node in nodes} == names and len(nodes) == 3, 'Pure helper inventory differs')
        namespace = {'need': need, 'struct': struct, 'uuid': uuid, 're': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
        pdb = Path(original['preserved_pdb'])
        dump = original['objdump_command']
        exe, objdump = Path(dump[-1]), Path(dump[0])
        pdbutil = Path('C:/Program Files/LLVM/bin/llvm-pdbutil.exe')
        need(all(path in bound for path in (pdb, exe, objdump, pdbutil)), 'Unbound regeneration input')
        exe_bytes = exe.read_bytes()
        need(hashlib.sha256(exe_bytes).hexdigest() == bound[exe], 'Executable changed during read')
        base, sections, identity = namespace['pe_identity'](exe_bytes)
        need(identity == original['pe_codeview'], 'PE identity differs')

        def tool(command):
            report['commands'].append([str(v) for v in command])
            return subprocess.check_output([str(v) for v in command], text=True, timeout=180)

        summary = tool([pdbutil, 'dump', '-summary', pdb])
        need('GUID: {' + identity['guid'] + '}' in summary
             and re.search(r'Age:\s*' + str(identity['age']) + r'\b', summary), 'PDB GUID/age differs')
        symbols = tool([pdbutil, 'dump', '-symbols', pdb])
        found = list(re.finditer(r'S_[LG]PROC32(?:_ID)?[^\n]*`' + re.escape(original['function']) + r'`\n([^\n]+)', symbols))
        need(len(found) == 1, 'Exact symbol count differs')
        symbol = found[0].group()
        extent = re.search(r'addr = (\d+):(\d+), code size = (\d+)', symbol)
        need(extent is not None, 'Missing exact PDB extent')
        section, offset, size = map(int, extent.groups())
        need((section, offset, size) == (original['section'], original['section_offset_decimal'], original['code_bytes']),
             'PDB extent differs')
        need(1 <= section <= len(sections) and size > 0 and offset + size <= sections[section - 1][1], 'Invalid PE extent')
        start = base + sections[section - 1][0] + offset
        stop = start + size
        need((hex(start), hex(stop)) == (original['virtual_address'], original['stop_address']), 'VA extent differs')
        need(dump == [str(objdump), '-d', '--no-show-raw-insn', '--start-address=' + hex(start),
                      '--stop-address=' + hex(stop), str(exe)], 'Objdump command differs')
        disassembly = tool(dump)
        instructions = []
        for line in disassembly.splitlines():
            match = re.match(r'^\s*([0-9a-fA-F]+):\s+(\w+)\s*(.*)', line)
            if match:
                instructions.append((int(match[1], 16), match[2], match[3]))
        need(instructions and instructions[0][0] == start
             and all(start <= a < stop for a, _, _ in instructions), 'Parsed disassembly extent differs')
        publics = tool([pdbutil, 'dump', '-publics', pdb])
        targets = namespace['direct_call_targets'](instructions, symbols, publics, base, sections)
        need(targets == original['direct_call_target_resolution'], 'Recorded target interpretation differs')
        for key, value in namespace['summarize_instructions'](instructions).items():
            need(original[key] == value, 'Instruction summary differs: ' + key)
        generated = {'pdb-summary.txt': summary, 'prefix-head-symbol.txt': symbol + '\n',
                     'prefix-head-disassembly.txt': disassembly,
                     'direct-call-targets.json': json.dumps(targets, indent=2) + '\n'}
        need(set(generated) == set(original['output_sha256']), 'Original output inventory differs')
        for name, text in generated.items():
            encoded = text.encode('utf-8')
            expected = original['output_sha256'][name]
            saved = (ORIGINAL.parent / name).read_bytes()
            need(hashlib.sha256(encoded).hexdigest() == expected == hashlib.sha256(saved).hexdigest()
                 and saved == encoded, 'Saved/regenerated UTF8 differs: ' + name)
            report['artifacts'].append({'name': name, 'sha256': expected, 'bytes': len(encoded), 'exact_bytes_equal': True})
        report.update(status='all_four_saved_artifacts_exactly_regenerated',
                      pe_pdb_identity_and_extent_equal=True, parsed_instruction_summaries_equal=True,
                      direct_call_target_interpretation_equal=True)
    except BaseException as error:
        failure = error
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        try:
            for path, expected in bound.items():
                need(sha(path) == expected, 'End identity mismatch: ' + str(path))
            report['source_input_tool_output_closure'] = True
        except BaseException as error:
            failure = failure or error
            report.update(status='failed', closure_error=repr(error))
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      bound_sha256={str(p):h for p,h in bound.items()},
                      all_started_tool_processes_completed_or_reaped=True)
        with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': report['status'], 'commands': len(report['commands']),
                      'bound_files': len(bound), 'receipt_sha256': sha(OUTPUT)}))
    if failure:
        raise failure


if __name__ == '__main__':
    main()
