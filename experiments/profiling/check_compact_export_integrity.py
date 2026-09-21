#!/usr/bin/env python3
"""Independently check saved compact PerfView XML; never open ETL or run tools.

Adapted from the preserved perfview-export-review-v1/inspect_xml.py. This helper
is bound separately at execution, not retroactively included in the capture plan.
Export integrity does not turn a failed exporter or lifetime audit into success.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
PROCESS_ROOT = re.compile(r'^Process(?:32|64)?\s+.+? \((\d+)\)(?: Args:.*)?$', re.I)
COMPACT_HEAD = re.compile(r'(?<![\w:])falcon_ocr::kernels::attention64_candidate::(?:compact_head(?=$|[^a-z0-9_])|compact::(?:\{\{closure\}\}|closure(?:_env)?\$0)(?=$|[^a-z0-9_]))')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def category(names):
    # Independent copy of the frozen, prospective attribution rules; no import
    # or execution of the analyzer being corroborated.
    value = '\n'.join(names).lower()
    if 'attention_gemm' in value:
        return 'attention_gemm_call_path'
    if 'attention_with_simd' in value or 'attention_compact_with_simd' in value or COMPACT_HEAD.search(value):
        return 'attention_call_path'
    for needle, label in [
        ('linear_with_simd', 'linear_call_path'), ('gemm', 'gemm_call_path'),
        ('rms_norm', 'normalization_call_path'), ('sum_squares_pairwise', 'normalization_call_path'),
        ('squared_relu_gate', 'gate_call_path'), ('dot_avx', 'dot_without_resolved_operator_caller'),
        ('axpy_avx', 'axpy_without_resolved_operator_caller'), ('rayon', 'rayon_without_resolved_operator_caller')]:
        if needle in value:
            return label
    return 'other_or_unresolved'


def inspect(path, pid, summary):
    frames, stacks, headers, ancestry = {}, {}, {}, {}
    exclusive, inclusive, categories, roots = Counter(), Counter(), Counter(), Counter()
    cross, questionable_frames = Counter(), Counter()
    time_bins = defaultdict(Counter)
    count = sample_ids = 0
    total = unknown_leaf = project_unknown = project_known = questionable = broken = 0.0
    first, last = math.inf, -math.inf
    gui = None
    parents = []

    def chain(sid):
        if sid in ancestry:
            return ancestry[sid]
        current, seen, result = sid, set(), []
        while current != -1:
            require(current not in seen, 'Cyclic stack chain')
            require(current in stacks, 'Missing stack reference')
            seen.add(current)
            fid, current = stacks[current]
            require(fid in frames, 'Missing frame reference')
            result.append(frames[fid])
        ancestry[sid] = tuple(result)
        return ancestry[sid]

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == 1 and infos[0].filename.lower().endswith('.xml'), 'Exactly one XML ZIP member required')
        info = infos[0]
        require(not info.is_dir(), 'XML member is a directory')
        with archive.open(info) as source:
            # Reaching EOF validates ZipExtFile's CRC. Parsing and removing each
            # record retains only frame/stack inventories and aggregate metrics.
            for event, element in ET.iterparse(source, events=('start', 'end')):
                if event == 'start':
                    parents.append(element)
                    if element.tag in ('Frames', 'Stacks', 'Samples'):
                        require(element.tag not in headers, 'Duplicate inventory header')
                        headers[element.tag] = dict(element.attrib)
                    continue
                if element.tag == 'Frame':
                    require(parents[-2].tag == 'Frames', 'Frame outside inventory')
                    fid = int(element.attrib['ID'])
                    require(fid == len(frames), 'Noncontiguous/duplicate frame ID')
                    frames[fid] = element.text or ''
                elif element.tag == 'Stack':
                    require(parents[-2].tag == 'Stacks', 'Stack outside inventory')
                    sid = int(element.attrib['ID'])
                    require(sid == len(stacks), 'Noncontiguous/duplicate stack ID')
                    stacks[sid] = (int(element.attrib['FrameID']), int(element.attrib['CallerID']))
                elif element.tag == 'Sample':
                    require(parents[-2].tag == 'Samples', 'Sample outside inventory')
                    if 'ID' in element.attrib:
                        require(int(element.attrib['ID']) == count, 'Noncontiguous sample ID')
                        sample_ids += 1
                    sid = int(element.attrib['StackID'])
                    require(sid in stacks, 'Sample references missing stack')
                    weight = float(element.attrib.get('Metric', '1'))
                    time = float(element.attrib['Time'])
                    require(math.isfinite(weight) and weight > 0 and math.isfinite(time), 'Nonfinite/nonpositive sample')
                    names = chain(sid)
                    process = [name for name in names if PROCESS_ROOT.fullmatch(name)]
                    require(len(process) == 1 and int(PROCESS_ROOT.fullmatch(process[0]).group(1)) == pid,
                            'Sample lacks one exact-PID process root')
                    count += 1
                    total += weight
                    first, last = min(first, time), max(last, time)
                    roots[process[0]] += weight
                    exclusive[names[0]] += weight
                    for name in set(names):
                        inclusive[name] += weight
                    group = category(names)
                    categories[group] += weight
                    time_bins[int(time // 1000)][group] += weight
                    cross[(names[0], group)] += weight
                    leaf = names[0]
                    unresolved = '!?' in leaf or re.search(r'!(?:0x)?[0-9a-fA-F]+$', leaf)
                    if unresolved:
                        unknown_leaf += weight
                    if 'ocr_bench!' in leaf.lower():
                        if unresolved:
                            project_unknown += weight
                        else:
                            project_known += weight
                    if any('??' in name for name in names):
                        questionable += weight
                    for name in set(names):
                        if '??' in name:
                            questionable_frames[name] += weight
                    if any(name in ('BROKEN', 'ROOT', 'OVERHEAD') for name in names):
                        broken += weight
                elif element.tag == 'StackWindowGuiState':
                    require(gui is None, 'Duplicate GUI metadata')
                    gui = ET.tostring(element, encoding='unicode')
                if element.tag in ('Frame', 'Stack', 'Sample'):
                    parents[-2].remove(element)
                    element.clear()
                parents.pop()
            require(source.read(1) == b'', 'XML ZIP member was not fully consumed')
        require(not parents, 'Incomplete XML nesting')

    actual = {'Frames': len(frames), 'Stacks': len(stacks), 'Samples': count}
    require({key: int(value['Count']) for key, value in headers.items()} == actual, 'Declared/actual inventory mismatch')
    require(count > 0 and math.isfinite(total) and total > 0, 'No valid sample inventory')
    require(sample_ids in (0, count), 'Partially populated sample IDs')
    # Include unsampled chains: corrupt unused references must not disappear.
    for fid, caller in stacks.values():
        require(fid in frames and (caller == -1 or caller in stacks), 'Invalid unused stack/frame reference')
    for sid in stacks:
        chain(sid)

    require(summary.get('kind') == 'perfview-compact-cpu-stack-summary-v1'
            and summary.get('status') == 'export_parsed_requires_capture_audit', 'Not the compact analyzer summary')
    expected_fields = {
        'expected_pid': pid, 'input': str(path.resolve()), 'xml_member': info.filename,
        'frame_count': len(frames), 'stack_count': len(stacks), 'sample_count': count,
        'sampled_cpu_metric_total': total, 'first_sample_relative_ms': first, 'last_sample_relative_ms': last,
        'process_roots': dict(roots), 'empty_stack_metric': 0.0,
        'project_unresolved_leaf_metric': project_unknown, 'project_resolved_leaf_metric': project_known,
        'one_second_bins': [{'trace_second': k, 'categories': dict(v)} for k, v in sorted(time_bins.items())]}
    for key, expected in expected_fields.items():
        require(summary.get(key) == expected, 'Analyzer mismatch: ' + key)
    for key, counter in [('exclusive_by_name', exclusive), ('inclusive_by_name', inclusive),
                         ('exclusive_call_path_categories', categories)]:
        rows = summary[key]
        require(len(rows) == len({row['name'] for row in rows}), 'Duplicate analyzer category/name')
        require({row['name']: row['sampled_cpu_metric'] for row in rows} == dict(counter), 'Analyzer totals mismatch: ' + key)
        require(all(row['percent_of_export_metric'] == 100 * counter[row['name']] / total for row in rows), 'Analyzer percentage mismatch')

    return {
        'xml_member': {'name': info.filename, 'uncompressed_bytes': info.file_size,
                       'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}',
                       'crc_validated_by_complete_read': True},
        'headers': headers, 'actual_counts': actual, 'sample_ids_present': sample_ids,
        'all_declared_counts_exact': True, 'frame_stack_ids_contiguous': True,
        'all_references_valid_and_all_stack_chains_acyclic': True,
        'every_sample_has_exactly_one_process_root_pid': pid,
        'sampled_cpu_metric_total': total, 'sample_time_range_ms': [first, last],
        'analyzer_counts_times_roots_categories_names_and_bins_reproduced': True,
        'gui_metadata_xml': gui, 'questionable_double_question_mark_frames': dict(questionable_frames),
        'any_questionable_frame_in_sample_metric': questionable, 'unknown_leaf_metric': unknown_leaf,
        'project_unknown_leaf_metric': project_unknown, 'sentinel_root_broken_overhead_ancestry_metric': broken,
        'exclusive_leaf_call_path_cross_tab': [
            {'leaf': leaf, 'call_path': group, 'metric': weight, 'percent_of_all_samples': 100 * weight / total}
            for (leaf, group), weight in cross.most_common()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output required')
    require(args.pid > 0, 'Positive captured PID required')
    started = datetime.now(timezone.utc).isoformat()
    paths = [Path(__file__).resolve(), args.input.resolve(), args.summary.resolve(), HERE / 'analyze_compact_stacks.py']
    require(len(set(paths)) == 4 and args.output.resolve() not in paths, 'Input/output aliases forbidden')
    summary_bytes = args.summary.read_bytes()
    summary = json.loads(summary_bytes)
    bindings = {str(path): digest(path) for path in paths if path != args.summary.resolve()}
    bindings[str(args.summary.resolve())] = hashlib.sha256(summary_bytes).hexdigest()
    require(summary['input_sha256'] == bindings[str(args.input.resolve())], 'Summary bound to a different export')
    require(summary['analyzer_sha256'] == bindings[str(HERE / 'analyze_compact_stacks.py')], 'Analyzer source binding differs')
    report = inspect(args.input, args.pid, summary)
    for file, expected in bindings.items():
        require(digest(Path(file)) == expected, 'Changed input/helper/analyzer: ' + file)
    report.update({
        'kind': 'independent-compact-perfview-xml-integrity-v1',
        'status': 'export_integrity_verified_capture_audit_still_required',
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'checked_sources_before_after': bindings, 'source_and_input_window_unchanged': True,
        'scope': 'Saved XML only; no ETL read, conversion, inference, recorder or exporter rerun.',
        'limitations': [
            'Exporter exit/log outcome is separate and must remain preserved, including any nonzero exit.',
            'No ETL lifetime/event-loss/stack-coverage acceptance follows from XML integrity.',
            'Summary agreement does not recover missing/export-filtered samples; compare the independent exact-PID ETL audit.',
            'Function-only call paths cannot split inlined dot/PV instructions or infer exact model phase boundaries.',
            'Sample metrics are not wall time, bandwidth, cache-miss counters or an unprofiled performance result.']})
    with args.output.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, ensure_ascii=True, allow_nan=False)
        target.write('\n')
    print(json.dumps({'status': report['status'], 'actual_counts': report['actual_counts'],
                      'output': str(args.output.resolve()), 'sha256': digest(args.output)}))


if __name__ == '__main__':
    main()
