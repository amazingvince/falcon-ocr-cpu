"""Synthetic functional-protocol tests; no model files, tensors, builds or inference."""
import copy
from pathlib import Path
import unittest

import run_model


def identities(count, prefix):
    return {f'{prefix}.{i}': {'dtype': 'F32-le', 'shape': [1], 'elements': 1,
                            'sha256': f'{i:064x}'} for i in range(count)}


def fixture(layout='compact'):
    canonical = identities(1904, 'synthetic.canonical')
    gpu_ids = list(range(100, 116)) + [263]
    metadata = {'model_revision': 'a' * 40, 'weights_sha256': 'b' * 64, 'token_ids': gpu_ids}
    sequences = [gpu_ids, [261, 263], [200, 201, 202, 203, 204, 263]]

    def result(ids, teacher=False):
        return {'token_ids': ids.copy(), 'text': 'synthetic text',
                'finish_reason': 'length' if teacher else 'eos',
                'teacher_forced': teacher, 'output_tokens': len(ids),
                'width': 256, 'height': 128, 'input_tokens': 144,
                'precision': 'fp32', 'backend': 'rust-gemm/avx2',
                'weight_layout': 'unpacked', 'packed_weight_bytes': 0}

    trace = {'tensors': identities(2144, 'synthetic.mixed'),
             'logits_argmax': [[f'request.{i}.prefill.logits', [ids[0]]]
                               for i, ids in enumerate(sequences)],
             'decode_rows': [], 'active_request_indices': []}
    for step in range(16):
        active = [i for i, ids in enumerate(sequences) if len(ids) > step + 1]
        trace['decode_rows'].append(len(active))
        trace['active_request_indices'].append([f'batch.0.decode.{step}.request_indices', active])
        trace['logits_argmax'].append([f'batch.0.decode.{step}.logits',
                                      [sequences[i][step + 1] for i in active]])
    value = {'schema_version': 1, 'status': 'completed', 'cache_layout': layout,
             'runtime': copy.deepcopy(run_model.RUNTIME), 'performance_measurement': False,
             'model_revision': metadata['model_revision'], 'weights_sha256': metadata['weights_sha256'],
             'canonical': {'path_kind': 'teacher_forced_same_prefix', 'result': result(gpu_ids, True),
                           'trace': {'tensors': copy.deepcopy(canonical),
                                     'logits_argmax': [[name, [token]] for name, token in zip(
                                         ['prefill.logits'] + [f'decode.{i}.logits' for i in range(16)], gpu_ids)]}},
             'independent_free_single': [result(ids) for ids in sequences],
             'independent_free_mixed': [result(ids) for ids in sequences],
             'mixed_trace': trace,
             'allocation': {k: copy.deepcopy(run_model.ALLOCATIONS) for k in ('single', 'mixed')}}
    return value, canonical, metadata


def operator_group():
    exe = Path('synthetic/operators.exe')
    selected = 'synthetic::tests::'
    names = [selected + 'case_a', selected + 'case_b']
    group = {'filter': selected, 'tests': names.copy(), 'passed': True,
             'exit_code': 0, 'log': 'operator-0.log',
             'command': [str(exe), selected, '--show-output', '--test-threads', '1']}
    return group, selected, names, exe, 0


class ModelProtocol(unittest.TestCase):
    def test_valid_complete_synthetic_modes(self):
        before, expected, metadata = fixture()
        after = copy.deepcopy(before)
        after['cache_layout'] = 'head_contiguous_prefix'
        self.assertEqual(len(expected), 1904)
        self.assertEqual(len(before['mixed_trace']['tensors']), 2144)
        self.assertEqual(set(before['mixed_trace']['decode_rows']), {1, 2, 3})
        run_model.validate_result(before, 'compact', expected, metadata)
        run_model.validate_result(after, 'head_contiguous_prefix', expected, metadata)
        run_model.compare_modes(before, after)

    def test_teacher_length_accepted_eos_rejected(self):
        value, expected, metadata = fixture()
        run_model.validate_result(value, 'compact', expected, metadata)
        value['canonical']['result']['finish_reason'] = 'eos'
        with self.assertRaisesRegex(ValueError, 'Wrong teacher output'):
            run_model.validate_result(value, 'compact', expected, metadata)

    def test_canonical_digest_change_rejected(self):
        value, expected, metadata = fixture()
        value['canonical']['trace']['tensors']['synthetic.canonical.901']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, '1904 CPU tensor identities differ'):
            run_model.validate_result(value, 'compact', expected, metadata)

    def test_mixed_digest_change_rejected_between_modes(self):
        before, expected, metadata = fixture()
        after = copy.deepcopy(before)
        after['cache_layout'] = 'head_contiguous_prefix'
        after['mixed_trace']['tensors']['synthetic.mixed.1700']['sha256'] = 'f' * 64
        # The mixed identity anchor is the fresh Compact result, so this belongs
        # to the cross-mode gate rather than the independent canonical anchor.
        run_model.validate_result(before, 'compact', expected, metadata)
        run_model.validate_result(after, 'head_contiguous_prefix', expected, metadata)
        with self.assertRaisesRegex(ValueError, 'differ beyond layout label'):
            run_model.compare_modes(before, after)

    def test_wrong_active_request_order_rejected(self):
        value, expected, metadata = fixture()
        value['mixed_trace']['active_request_indices'][0][1] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, 'Active session mapping differs'):
            run_model.validate_result(value, 'compact', expected, metadata)

    def test_nonzero_allocations_and_extra_intervals_rejected(self):
        for phase in ('single', 'mixed'):
            for field, changed in (('allocation_calls', 1), ('requested_bytes', 64),
                                   ('decode_starts', 2), ('decode_ends', 0)):
                with self.subTest(phase=phase, field=field):
                    value, expected, metadata = fixture()
                    value['allocation'][phase][field] = changed
                    with self.assertRaisesRegex(ValueError, 'Warmed allocation interval failed'):
                        run_model.validate_result(value, 'compact', expected, metadata)

    def test_free_length_and_mismatched_argmax_rejected(self):
        value, expected, metadata = fixture()
        for records in ('independent_free_single', 'independent_free_mixed'):
            value[records][1]['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'Free stop/count differs'):
            run_model.validate_result(value, 'compact', expected, metadata)
        value, expected, metadata = fixture()
        value['mixed_trace']['logits_argmax'][3][1][0] += 1
        with self.assertRaisesRegex(ValueError, 'Mixed argmax/output disagreement'):
            run_model.validate_result(value, 'compact', expected, metadata)

    def test_layout_order_and_common_field_mutation_rejected(self):
        before, _, _ = fixture()
        after = copy.deepcopy(before)
        after['cache_layout'] = 'head_contiguous_prefix'
        with self.assertRaisesRegex(ValueError, 'Wrong ordered layout'):
            run_model.compare_modes(after, before)
        after['independent_free_single'][0]['text'] = 'changed text'
        with self.assertRaisesRegex(ValueError, 'differ beyond layout label'):
            run_model.compare_modes(before, after)

    def test_valid_operator_command_and_group(self):
        run_model.validate_operator_group(*operator_group())

    def test_wrong_operator_command_rejected(self):
        for index, changed in ((0, 'different/operators.exe'), (1, 'different::filter'),
                               (2, '--ignored'), (3, '--skip'), (4, '2')):
            with self.subTest(argument=index):
                args = operator_group()
                args[0]['command'][index] = changed
                with self.assertRaisesRegex(ValueError, 'Wrong operator command'):
                    run_model.validate_operator_group(*args)

    def test_missing_or_failed_operator_group_rejected(self):
        for field, changed in (('tests', ['synthetic::tests::case_a']), ('passed', False),
                               ('exit_code', 1), ('log', 'operator-1.log')):
            with self.subTest(field=field):
                args = operator_group()
                args[0][field] = changed
                with self.assertRaisesRegex(ValueError, 'Wrong operator group'):
                    run_model.validate_operator_group(*args)


if __name__ == '__main__':
    unittest.main()
