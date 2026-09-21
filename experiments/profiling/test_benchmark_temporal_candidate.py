"""Offline temporal protocol tests. All process and model-input work is mocked."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest import mock

SPEC = importlib.util.spec_from_file_location("temporal_candidate_protocol", Path(__file__).with_name("benchmark_temporal_candidate.py"))
bc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bc)


class FakeChild:
    pid = 12345

    def __init__(self, first_error=None):
        self.first_error = first_error
        self.waits = 0
        self.killed = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1 and self.first_error is not None:
            raise self.first_error
        return -9 if self.killed else 0

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True


def synthetic_builds(folder):
    original = {n: "1" * 64 for n in bc.SOURCE_CHANGES}
    original.update({"examples/ocr_bench.rs":"2"*64, "src/runner.rs":"3"*64})
    control = {"binary_sha256":bc.CONTROL_EXE_SHA, "source_sha256":original, "source_archive":"source.zip",
               "command":["control-wrapper", "build", "--locked", "--release", "--jobs", "2"],
               "rustc_version":"same-rustc", "cargo_version":"same-cargo",
               "environment_overrides":{"CARGO_TARGET_DIR":str(folder/'control-target')}}
    candidate = copy.deepcopy(control)
    candidate['binary_sha256'] = 'b' * 64
    candidate['environment_overrides']['CARGO_TARGET_DIR'] = str(folder/'candidate-target')
    candidate['source_sha256'].update({n:'4'*64 for n in bc.SOURCE_CHANGES})
    candidate['source_sha256'].update({n:'5'*64 for n in bc.SOURCE_ADDITIONS})
    return {'control':(control,folder/'control.exe'), 'candidate':(candidate,folder/'candidate.exe')}


class TemporalProtocolTests(unittest.TestCase):
    def test_explicit_cli_and_report_spellings_for_each_job(self):
        workload = bc.rb.read(bc.rb.DEFAULT_WORKLOAD)  # Small frozen JSON only.
        plan = {'binary':'not-executed.exe', 'repetitions':3, 'cpu_label':'cpu', 'environment_label':'native'}
        for name,cli,report in [('control-before','compact','compact'),
                                ('candidate','temporal-candidate','temporal_candidate'),
                                ('control-after','compact','compact')]:
            command = bc.command_for(plan,workload,name,Path('unused.json'))
            self.assertEqual(command[command.index('--cache-layout')+1],cli)
            self.assertEqual(bc.job(name)['mode']['cache_layout'],report)
            self.assertEqual(command[command.index('--batches')+1],'1')
            self.assertEqual(command[command.index('--warmup')+1],'2')
            self.assertEqual(command[command.index('--repetitions')+1],'3')
            self.assertEqual(command[-1],str(bc.ROOT/workload['inputs']['prose']['canonical_path']))
        with self.assertRaises(ValueError):bc.job('unplanned-case')

    def test_real_compact_baseline_and_explicit_candidate_report_contract(self):
        workload=bc.rb.read(bc.rb.DEFAULT_WORKLOAD)
        baseline,_=bc.checked_json(bc.BASELINE,bc.BASELINE_SHA)
        control,_=bc.checked_json(bc.CONTROL,bc.CONTROL_SHA)
        signatures=bc.baseline_signatures(baseline,workload,control)
        self.assertEqual(len(signatures[0]['token_ids']),1140)
        self.assertEqual(signatures[0]['finish_reason'],'eos')
        plan={'repetitions':3,'cpu_label':'AMD Ryzen 9 7950X','environment_label':'Native Windows'}
        candidate=copy.deepcopy(baseline)
        candidate['cache_layout']='temporal_candidate'
        self.assertEqual(bc.report_signatures(candidate,'candidate',plan,workload,control),signatures)
        for value in ('compact','temporal-candidate'):
            candidate['cache_layout']=value
            with self.assertRaisesRegex(ValueError,'cache_layout mismatch'):
                bc.report_signatures(candidate,'candidate',plan,workload,control)

    def test_actual_os_arch_and_embedded_inventory_required(self):
        workload=bc.rb.read(bc.rb.DEFAULT_WORKLOAD)
        baseline,_=bc.checked_json(bc.BASELINE,bc.BASELINE_SHA)
        control,_=bc.checked_json(bc.CONTROL,bc.CONTROL_SHA)
        for field,value in [('os','linux'),('arch','aarch64')]:
            changed=copy.deepcopy(baseline);changed[field]=value
            with self.assertRaisesRegex(ValueError,'platform'):
                bc.baseline_signatures(changed,workload,control)
        baseline['source_sha256'].pop('runner')
        with self.assertRaisesRegex(ValueError,'inventory'):
            bc.baseline_signatures(baseline,workload,control)

    def test_source_allowlist_exact_with_no_harness_change_or_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            builds=synthetic_builds(Path(temp));control,candidate=(builds[k][0] for k in ('control','candidate'))
            self.assertEqual(bc.compare_builds(control,candidate),(sorted(bc.SOURCE_CHANGES),sorted(bc.SOURCE_ADDITIONS)))
            mutations=[lambda c:c['source_sha256'].update({'src/unreviewed.rs':'6'*64}),
                       lambda c:c['source_sha256'].update({'examples/ocr_bench.rs':'6'*64}),
                       lambda c:c['source_sha256'].pop('src/runner.rs'),
                       lambda c:c['source_sha256'].pop('src/temporal_model_tests.rs'),
                       lambda c:c['source_sha256'].update({'src/config.rs':control['source_sha256']['src/config.rs']})]
            for mutate in mutations:
                changed=copy.deepcopy(candidate);mutate(changed)
                with self.assertRaises(ValueError):bc.compare_builds(control,changed)

    def test_same_compiler_flags_distinct_binary_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            builds=synthetic_builds(Path(temp));control,candidate=(builds[k][0] for k in ('control','candidate'))
            mutations=[lambda c:c.update(binary_sha256=control['binary_sha256']),
                       lambda c:c['environment_overrides'].update(CARGO_TARGET_DIR=control['environment_overrides']['CARGO_TARGET_DIR']),
                       lambda c:c['environment_overrides'].update(RUSTFLAGS='-C target-cpu=native'),
                       lambda c:c.update(rustc_version='different'),
                       lambda c:c['command'].append('--features=extra')]
            for mutate in mutations:
                changed=copy.deepcopy(candidate);mutate(changed)
                with self.assertRaises(ValueError):bc.compare_builds(control,changed)

    def protocol(self,folder):
        builds=synthetic_builds(folder)
        workload=bc.rb.read(bc.rb.DEFAULT_WORKLOAD)
        plan={'kind':bc.KIND,'output_directory':str(folder),'host_platform':platform.platform(),
              'builds':{'control':str(bc.CONTROL),'candidate':str(folder/'candidate/build.json')},
              'workload':str(bc.rb.DEFAULT_WORKLOAD),'historical_baseline':str(bc.BASELINE),
              'jobs':copy.deepcopy(bc.JOBS),'cache_layouts':copy.deepcopy(bc.LAYOUTS),
              'source_changes':sorted(bc.SOURCE_CHANGES),'source_additions':sorted(bc.SOURCE_ADDITIONS),
              'repetitions':3,'max_process_seconds':900,'control_drift_limit_percent':5,
              'target_latency_reduction_percent':5,'default_promotion':False,'expected_signatures':[]}
        files={str(Path(bc.__file__).resolve()),str(bc.ROOT/'scripts/realistic_benchmark.py'),
               str(folder/'protocol-source.zip'),plan['workload'],plan['historical_baseline'],*plan['builds'].values()}
        for key,(build,binary) in builds.items():
            files.update({str(binary),str(Path(plan['builds'][key]).parent/build['source_archive'])})
        plan['files_sha256']={p:'f'*64 for p in files}
        return plan,workload,builds

    def save(self,path,plan):
        raw=json.dumps(plan).encode();path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def mocked_inputs(self,stack,workload,builds):
        def read(path,expected=None):
            if Path(path).resolve()==bc.rb.DEFAULT_WORKLOAD.resolve():return workload,'f'*64
            return {},'f'*64
        stack.enter_context(mock.patch.object(bc,'checked_json',side_effect=read))
        stack.enter_context(mock.patch.object(bc,'baseline_signatures',return_value=[]))
        stack.enter_context(mock.patch.object(bc.rb,'verify'))
        stack.enter_context(mock.patch.object(bc.rb,'validate_inputs'))
        stack.enter_context(mock.patch.object(bc.rb,'validate_build',side_effect=[builds['control'],builds['candidate']]))

    def test_fixed_plan_accepts_offline_and_mutations_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json';plan,workload,builds=self.protocol(path.parent)
            digest=self.save(path,plan)
            with ExitStack() as stack:
                self.mocked_inputs(stack,workload,builds)
                self.assertEqual(bc.validate(path,digest)[0],plan)
            mutations=[lambda p:p['jobs'].reverse(),lambda p:p.update(repetitions=1),
                       lambda p:p.update(max_process_seconds=901),lambda p:p.update(control_drift_limit_percent=6),
                       lambda p:p['cache_layouts']['candidate'].update(report='compact'),
                       lambda p:p['source_additions'].pop(),lambda p:p.update(files_sha256={})]
            for mutate in mutations:
                changed=copy.deepcopy(plan);mutate(changed);digest=self.save(path,changed)
                with ExitStack() as stack:
                    self.mocked_inputs(stack,workload,builds)
                    with self.assertRaises(ValueError):bc.validate(path,digest)

    def test_wrong_plan_digest_fails_before_input_work(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json';plan,_,_=self.protocol(path.parent);self.save(path,plan)
            with mock.patch.object(bc.rb,'validate_inputs') as input_work:
                with self.assertRaisesRegex(ValueError,'Plan hash mismatch'):bc.validate(path,'0'*64)
                input_work.assert_not_called()

    def test_checked_json_hashes_the_bytes_it_parses(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'record.json';value={'actual':42};raw=json.dumps(value).encode();path.write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            self.assertEqual(bc.checked_json(path,digest),(value,digest))
            path.write_bytes(b'not JSON')
            with self.assertRaisesRegex(ValueError,'identity changed'):bc.checked_json(path,digest)

    def lifecycle(self,kind):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);plan,workload,builds=self.protocol(folder)
            error=KeyboardInterrupt() if kind=='interrupt' else (subprocess.TimeoutExpired(['unused.exe'],900) if kind=='timeout' else None)
            child=FakeChild(error)
            args=argparse.Namespace(plan=folder/'plan.json',plan_sha256='f'*64,quiet_attestation='offline mocked test')
            original_write=bc.rb.write_new
            def write(path,value):
                if kind=='start_receipt' and str(path).endswith('.start.json'):raise OSError('Synthetic receipt failure')
                original_write(path,value)
            with mock.patch.object(bc,'validate',return_value=(plan,workload,builds)), \
                 mock.patch.object(bc,'checked_json',return_value=({},'f'*64)), \
                 mock.patch.object(bc,'baseline_signatures',return_value=[]), \
                 mock.patch.object(bc,'command_for',return_value=['unused.exe']), \
                 mock.patch.object(bc.rb,'sha',return_value='f'*64), \
                 mock.patch.object(bc.rb,'write_new',side_effect=write), \
                 mock.patch.object(bc.subprocess,'Popen',return_value=child), redirect_stdout(io.StringIO()):
                with self.assertRaises((KeyboardInterrupt,OSError,ValueError)):bc.run(args)
            self.assertTrue(child.killed)
            self.assertGreaterEqual(child.waits,2 if kind in ('timeout','interrupt') else 1)
            self.assertTrue((folder/'execution-failed.json').is_file())

    def test_timeout_reaps_only_owned_child(self):self.lifecycle('timeout')
    def test_interrupt_reaps_only_owned_child(self):self.lifecycle('interrupt')
    def test_start_receipt_failure_reaps_owned_child(self):self.lifecycle('start_receipt')


if __name__=='__main__':unittest.main()
