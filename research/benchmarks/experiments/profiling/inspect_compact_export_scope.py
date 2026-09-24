#!/usr/bin/env python3
"""Diagnose a rejected saved XML export without weakening the frozen analyzer.

No ETL conversion, profiler, model, or external command is executed. Every
rootless/wrong-PID/ambiguous sample is retained in an adjacent fresh JSONL file.
Target-only categories remain a labeled subset; they never accept the capture.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

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


def inspect(path, pid, rejected_output):
    frames, stacks, headers, ancestry = {}, {}, {}, {}
    labels = ('valid_target', 'wrong_pid', 'rootless', 'ambiguous_roots')
    counts = Counter({key: 0 for key in labels})
    metrics = Counter({key: 0.0 for key in labels})
    root_counts, root_metrics, cardinalities = Counter(), Counter(), Counter()
    signature_counts, signature_metrics = Counter(), Counter()
    target_categories, target_category_counts = Counter(), Counter()
    rejected_digest = hashlib.sha256()
    total_count = sample_ids = 0
    total_metric = 0.0
    first, last = math.inf, -math.inf
    target_first, target_last = math.inf, -math.inf
    parents = []

    def chain(sid):
        if sid in ancestry:
            return ancestry[sid]
        current, seen, names = sid, set(), []
        while current != -1:
            require(current not in seen, 'Cyclic stack chain')
            require(current in stacks, 'Missing stack reference')
            seen.add(current)
            fid, current = stacks[current]
            require(fid in frames, 'Missing frame reference')
            names.append(frames[fid])
        ancestry[sid] = tuple(names)
        return ancestry[sid]

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == 1 and infos[0].filename.lower().endswith('.xml') and not infos[0].is_dir(),
                'Exactly one XML ZIP member required')
        info = infos[0]
        with archive.open(info) as source:
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
                    sample_id = int(element.attrib['ID']) if 'ID' in element.attrib else None
                    if sample_id is not None:
                        require(sample_id == total_count, 'Noncontiguous sample ID')
                        sample_ids += 1
                    sid = int(element.attrib['StackID'])
                    require(sid in stacks, 'Sample references missing stack')
                    weight = float(element.attrib.get('Metric', '1'))
                    timestamp = float(element.attrib['Time'])
                    require(math.isfinite(weight) and weight > 0 and math.isfinite(timestamp), 'Nonfinite/nonpositive sample')
                    names = chain(sid)
                    roots = tuple(name for name in names if PROCESS_ROOT.fullmatch(name))
                    root_pids = [int(PROCESS_ROOT.fullmatch(name).group(1)) for name in roots]
                    if len(roots) == 0:
                        label = 'rootless'
                    elif len(roots) > 1:
                        label = 'ambiguous_roots'
                    elif root_pids[0] != pid:
                        label = 'wrong_pid'
                    else:
                        label = 'valid_target'
                    counts[label] += 1
                    metrics[label] += weight
                    cardinalities[len(roots)] += 1
                    signature_counts[roots] += 1
                    signature_metrics[roots] += weight
                    for name in set(roots):
                        root_counts[name] += 1
                        root_metrics[name] += weight
                    if label == 'valid_target':
                        group = category(names)
                        target_categories[group] += weight
                        target_category_counts[group] += 1
                        target_first, target_last = min(target_first, timestamp), max(target_last, timestamp)
                    else:
                        # No sampling/cap/truncation of rejected records. JSONL
                        # keeps memory bounded even if an entire export is wrong.
                        record = {'ordinal': total_count, 'sample_id': sample_id,
                                  'time_relative_ms': timestamp, 'stack_id': sid, 'metric': weight,
                                  'classification': label, 'process_roots': list(roots), 'root_pids': root_pids,
                                  'complete_chain_leaf_to_root': list(names)}
                        serialized = json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n'
                        rejected_output.write(serialized)
                        rejected_digest.update(serialized.encode('utf-8'))
                    total_count += 1
                    total_metric += weight
                    first, last = min(first, timestamp), max(last, timestamp)
                if element.tag in ('Frame', 'Stack', 'Sample'):
                    parents[-2].remove(element)
                    element.clear()
                parents.pop()
            require(source.read(1) == b'', 'XML member not fully consumed')
        require(not parents, 'Incomplete XML nesting')
    actual = {'Frames': len(frames), 'Stacks': len(stacks), 'Samples': total_count}
    require({key: int(value['Count']) for key, value in headers.items()} == actual, 'Declared/actual inventory mismatch')
    require(total_count > 0 and math.isfinite(total_metric) and total_metric > 0, 'No valid sample inventory')
    require(sample_ids in (0, total_count), 'Partially populated sample IDs')
    for fid, caller in stacks.values():
        require(fid in frames and (caller == -1 or caller in stacks), 'Invalid unused stack/frame reference')
    for sid in stacks:
        chain(sid)
    require(sum(counts.values()) == total_count, 'Classification count mismatch')
    require(all(math.isfinite(value) for value in metrics.values()), 'Nonfinite aggregate metric')
    rejected_count = total_count - counts['valid_target']
    target_metric = metrics['valid_target']
    return {
        'status': 'rejected_process_scope' if rejected_count else 'process_scope_verified_capture_audit_still_required',
        'strict_frozen_analyzer_scope_would_pass': rejected_count == 0,
        'complete_capture_accepted': False, 'expected_pid': pid,
        'xml_member': {'name': info.filename, 'uncompressed_bytes': info.file_size,
                       'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}',
                       'crc_validated_by_complete_read': True},
        'headers': headers, 'actual_counts': actual, 'sample_ids_present': sample_ids,
        'all_declared_counts_exact': True, 'frame_stack_ids_contiguous': True,
        'all_references_valid_and_all_stack_chains_acyclic': True,
        'sample_counts': dict(counts), 'sample_metrics': dict(metrics),
        'all_sample_count': total_count, 'all_sample_metric': total_metric,
        'all_sample_time_range_ms': [first, last], 'rejected_sample_count': rejected_count,
        'emitted_rejected_records_sha256': rejected_digest.hexdigest(),
        'root_cardinality_sample_counts': dict(cardinalities),
        'process_root_distribution': [{'name': name, 'pid': int(PROCESS_ROOT.fullmatch(name).group(1)),
                                       'sample_count': count, 'sample_metric': root_metrics[name]}
                                      for name, count in root_counts.most_common()],
        'process_root_distribution_semantics': 'Each distinct exact root name per sample; ambiguous samples can contribute to several roots. Full root-chain signatures preserve duplicates.',
        'root_chain_signature_distribution': [{'roots_leaf_to_root': list(names), 'sample_count': count,
                                               'sample_metric': signature_metrics[names]}
                                              for names, count in signature_counts.most_common()],
        'valid_target_subset': {
            'sample_count_denominator': counts['valid_target'], 'sample_metric_denominator': target_metric,
            'sample_time_range_ms': [target_first, target_last] if counts['valid_target'] else None,
            'categories': [{'name': name, 'sample_count': target_category_counts[name], 'sample_metric': weight,
                            'percent_of_valid_target_metric': 100 * weight / target_metric,
                            'percent_of_all_export_metric': 100 * weight / total_metric}
                           for name, weight in target_categories.most_common()],
            'qualification': 'Positively attributed subset only. Rejected samples remain fully enumerated; no complete-profile or phase-timing claim.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(args.pid > 0, 'Positive captured PID required')
    rejected = args.output.with_name(args.output.stem + '.rejected-samples.jsonl')
    require(not args.output.exists() and not rejected.exists(), 'Fresh report and rejected-record paths required')
    sources = [Path(__file__).resolve(), args.input.resolve()]
    require(args.output.resolve() not in sources and rejected.resolve() not in sources, 'Input/output aliases forbidden')
    started = datetime.now(timezone.utc).isoformat()
    bindings = {str(path): digest(path) for path in sources}
    with rejected.open('x', encoding='utf-8', newline='\n') as records:
        report = inspect(args.input, args.pid, records)
    rejected_hash = digest(rejected)
    require(rejected_hash == report['emitted_rejected_records_sha256'], 'Rejected records differ from emitted bytes')
    for file, expected in bindings.items():
        require(digest(Path(file)) == expected, 'Changed source/export: ' + file)
    require(digest(rejected) == rejected_hash, 'Rejected records changed during closure')
    report.update({
        'kind': 'rejected-compact-export-process-scope-diagnostic-v1',
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'checked_source_and_export_before_after': bindings, 'source_and_export_unchanged': True,
        'rejected_records': {'path': str(rejected.resolve()), 'sha256': rejected_hash,
                             'count': report['rejected_sample_count'], 'complete_chains': True,
                             'records_truncated_or_omitted': False},
        'scope': 'Saved XML only. No ETL, model, recording, conversion or exporter rerun; frozen analyzer and acceptance gates unchanged.',
        'limitations': ['Exporter exit/log failure remains a separate preserved outcome.',
                       'Exact-PID ETL lifetime, loss and attached-stack coverage gates are not evaluated here.',
                       'An exported chain cannot prove that a rootless sample is harmless teardown or identify its true owner.',
                       'Function-only categories and subset percentages are not wall-time phases, hardware counters or speed results.']})
    with args.output.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, ensure_ascii=True, allow_nan=False)
        target.write('\n')
    print(json.dumps({'status': report['status'], 'counts': report['sample_counts'],
                      'output': str(args.output.resolve()), 'sha256': digest(args.output)}))
    return 1 if report['rejected_sample_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
