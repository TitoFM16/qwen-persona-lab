import sys
import threading
import time
import unittest
from unittest import mock

from lab.steering import ModelRuntimePool, MODELS

A, B, C = list(MODELS)


class OutOfMemoryError(RuntimeError):
    pass


class FakeRuntime:
    log = []
    lock = threading.Lock()
    active = 0
    max_active = 0
    block = {}
    oom_on_load = set()
    oom_on_compare = set()
    seq = 0

    def __init__(self, device='cpu', norms_dir=''):
        self.model = self.tokenizer = self.torch = None
        self.model_id = None
        self.model_revision = None
        FakeRuntime.seq += 1
        n = FakeRuntime.seq
        self._baseline_cache = {'k': ('baseline-%d' % n, n)}
        self._norm_sources = {'layer0': [float(n), n + 0.5]}
        self.loads = 0
        self.generates = 0

    @classmethod
    def reset(cls):
        cls.log, cls.active, cls.max_active = [], 0, 0
        cls.block, cls.oom_on_load, cls.oom_on_compare = {}, set(), set()
        cls.seq = 0

    def _enter(self):
        with FakeRuntime.lock:
            FakeRuntime.active += 1
            FakeRuntime.max_active = max(FakeRuntime.max_active, FakeRuntime.active)

    def _exit(self):
        with FakeRuntime.lock:
            FakeRuntime.active -= 1

    def load(self, model_id):
        self._enter()
        try:
            gate = FakeRuntime.block.get(model_id)
            if gate:
                gate['started'].set()
                gate['release'].wait(5)
            if model_id in FakeRuntime.oom_on_load:
                self.model = object()  # partial load
                raise OutOfMemoryError('CUDA out of memory')
            self.model, self.tokenizer, self.model_id = object(), object(), model_id
            self.model_revision = 'rev-' + model_id
            self.loads += 1
            FakeRuntime.log.append(('load', model_id))
        finally:
            self._exit()

    def unload(self):
        self.model = self.tokenizer = self.model_id = None
        self._baseline_cache.clear()
        self._norm_sources.clear()
        FakeRuntime.log.append(('unload',))

    def _encode(self, prompt, messages):
        return prompt

    def _generate(self, ids, max_new, intervention=None):
        self._enter()
        try:
            self.generates += 1
            FakeRuntime.log.append(('generate', self.model_id, ids, max_new))
            time.sleep(0.01)
            return 'x', 'length', [1]
        finally:
            self._exit()

    def compare(self, model_id, prompt, components, **kw):
        self._enter()
        try:
            if model_id in FakeRuntime.oom_on_compare:
                raise OutOfMemoryError('CUDA out of memory')
            time.sleep(0.01)
            return {'baseline': 'a', 'steered': 'b', 'metrics': {}, 'provenance': {'model': model_id}}
        finally:
            self._exit()


class PoolTests(unittest.TestCase):
    def setUp(self):
        FakeRuntime.reset()
        self.pool = ModelRuntimePool('cpu', runtime_factory=FakeRuntime)

    def test_all_three_resident_identities_retained(self):
        self.pool.warm([A, B, C])
        ids = {m: (self.pool._runtimes[m], self.pool._runtimes[m].model, self.pool._runtimes[m].tokenizer,
                   self.pool._runtimes[m]._baseline_cache) for m in (A, B, C)}
        values = {m: (dict(self.pool._runtimes[m]._baseline_cache),
                      {k: list(v) for k, v in self.pool._runtimes[m]._norm_sources.items()}) for m in (A, B, C)}
        self.assertEqual(len({repr(v) for v in values.values()}), 3)
        for m in (A, B, C, A, C, B):
            result = self.pool.compare(m, 'p', [])
            self.assertEqual(result['provenance']['pool']['resident_models'], sorted([A, B, C]))
        for m in (A, B, C):
            rt = self.pool._runtimes[m]
            self.assertEqual((rt, rt.model, rt.tokenizer, rt._baseline_cache), ids[m])
            self.assertEqual(rt._baseline_cache, values[m][0])
            self.assertEqual(rt._norm_sources, values[m][1])
            self.assertEqual(rt.model_revision, 'rev-' + m)
            self.assertEqual(self.pool.status['models'][m]['load_count'], 1)
        self.assertEqual(len([e for e in FakeRuntime.log if e[0] == 'load']), 3)
        self.assertNotIn(('unload',), FakeRuntime.log)

    def test_warm_hello_once_and_idempotent(self):
        self.pool.warm([A, B])
        gens = [e for e in FakeRuntime.log if e[0] == 'generate']
        self.assertEqual([(g[1], g[2]) for g in gens], [(A, 'Hello'), (B, 'Hello')])
        self.assertTrue(all(1 <= g[3] <= 4 for g in gens))
        st = self.pool.status
        self.assertEqual(st['ready_models'], [A, B])
        for m in (A, B):
            info = st['models'][m]
            self.assertLessEqual(info['weights_loaded_at'], info['warm_complete_at'])
            self.assertIsNotNone(info['warmup_seconds'])
        self.assertNotEqual(st['models'][A]['runtime_id'], st['models'][B]['runtime_id'])
        before = (list(FakeRuntime.log), st['models'][A]['warm_complete_at'])
        self.pool.warm([A, B])
        self.assertEqual(FakeRuntime.log, before[0])
        self.assertEqual(self.pool.status['models'][A]['warm_complete_at'], before[1])

    def test_serialization_and_nonblocking_status(self):
        gate = {'started': threading.Event(), 'release': threading.Event()}
        FakeRuntime.block[A] = gate
        t1 = threading.Thread(target=self.pool.warm, args=([A],))
        t1.start()
        self.assertTrue(gate['started'].wait(2))
        t2 = threading.Thread(target=self.pool.warm, args=([B],))
        t3 = threading.Thread(target=self.pool.compare, args=(C, 'p', []))
        t2.start(); t3.start()
        started = time.monotonic()
        st = self.pool.status
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(st['warming_models'], [A])
        self.assertEqual(st['ready_models'], [])
        self.assertEqual(st['resident_models'], [])
        gate['release'].set()
        for t in (t1, t2, t3):
            t.join(5)
        self.assertEqual(FakeRuntime.max_active, 1)
        self.assertEqual(self.pool.status['warming_models'], [])

    def test_oom_isolated_and_explicit_retry(self):
        self.pool.warm([A, B])
        keep = self.pool._runtimes[A], self.pool._runtimes[A].model
        FakeRuntime.oom_on_load.add(C)
        st = self.pool.warm([C])
        self.assertNotIn(C, st['resident_models'])
        self.assertNotIn(C, st['ready_models'])
        self.assertEqual(st['model_errors'][C]['type'], 'OutOfMemoryError')
        self.assertTrue(st['model_errors'][C]['oom'])
        self.assertEqual(st['ready_models'], [A, B])
        self.assertEqual((self.pool._runtimes[A], self.pool._runtimes[A].model), keep)
        # No automatic retry
        n = len(FakeRuntime.log)
        self.pool.warm([C])
        self.assertEqual(len(FakeRuntime.log), n)
        with self.assertRaises(RuntimeError):
            self.pool.compare(C, 'p', [])
        self.assertEqual(self.pool.compare(A, 'p', [])['baseline'], 'a')
        FakeRuntime.oom_on_load.clear()
        st = self.pool.warm([C], retry_failed=True)
        self.assertIn(C, st['ready_models'])
        self.assertEqual(st['model_errors'], {})
        # compare OOM
        FakeRuntime.oom_on_compare.add(B)
        with self.assertRaises(OutOfMemoryError):
            self.pool.compare(B, 'p', [])
        st = self.pool.status
        self.assertNotIn(B, st['ready_models'])
        self.assertIn(B, st['model_errors'])
        self.assertEqual(st['ready_models'], [A, C])

    def _snapshot(self, models):
        out = {}
        st = self.pool.status
        for m in models:
            rt = self.pool._runtimes[m]
            out[m] = {'rt': rt, 'model': rt.model, 'tok': rt.tokenizer, 'torch': rt.torch,
                      'rev': rt.model_revision, 'bc': rt._baseline_cache, 'bc_val': dict(rt._baseline_cache),
                      'ns': rt._norm_sources, 'ns_val': {k: list(v) for k, v in rt._norm_sources.items()},
                      'loads': st['models'][m]['load_count'], 'warm_at': st['models'][m]['warm_complete_at']}
        return out

    def _assert_same(self, snap):
        st = self.pool.status
        for m, o in snap.items():
            rt = self.pool._runtimes[m]
            self.assertIs(rt, o['rt'])
            self.assertIs(rt.model, o['model'])
            self.assertIs(rt.tokenizer, o['tok'])
            self.assertIs(rt.torch, o['torch'])
            self.assertEqual(rt.model_revision, o['rev'])
            self.assertEqual(rt.model_id, m)
            self.assertIs(rt._baseline_cache, o['bc'])
            self.assertEqual(rt._baseline_cache, o['bc_val'])
            self.assertIs(rt._norm_sources, o['ns'])
            self.assertEqual(rt._norm_sources, o['ns_val'])
            self.assertEqual(st['models'][m]['load_count'], o['loads'])
            self.assertEqual(st['models'][m]['warm_complete_at'], o['warm_at'])
            self.assertIn(m, st['ready_models'])

    def test_real_torch_cuda_oom_load_and_compare_on_cpu(self):
        try:
            import torch
            real_oom = torch.cuda.OutOfMemoryError
        except (ImportError, AttributeError):
            self.skipTest('torch with cuda.OutOfMemoryError unavailable')
        this = sys.modules[__name__]
        with mock.patch.object(this, 'OutOfMemoryError', real_oom):
            self.assertIs(FakeRuntime.load.__globals__['OutOfMemoryError'], real_oom)
            self.pool.warm([A, B])
            snap = self._snapshot([A, B])
            # load OOM on C
            FakeRuntime.oom_on_load.add(C)
            st = self.pool.warm([C])
            err = st['model_errors'][C]
            self.assertEqual(err['type'], 'OutOfMemoryError')
            self.assertTrue(err['oom'])
            self.assertNotIn(C, st['ready_models'])
            self.assertNotIn(C, st['resident_models'])
            self.assertFalse(st['models'][C]['loaded'])
            self.assertEqual(st['ready_models'], [A, B])
            self._assert_same(snap)
            n = len(FakeRuntime.log)
            self.pool.warm([C])  # no automatic retry
            self.assertEqual(len(FakeRuntime.log), n)
            with self.assertRaises(RuntimeError) as ctx:
                self.pool.compare(C, 'p', [])
            self.assertNotIsInstance(ctx.exception, real_oom)
            self.assertNotIn(('load', C), FakeRuntime.log)
            self.assertEqual(self.pool.compare(A, 'p', [])['provenance']['model'], A)
            self._assert_same(snap)
            # explicit retry restores C
            FakeRuntime.oom_on_load.clear()
            st = self.pool.warm([C], retry_failed=True)
            self.assertEqual(st['ready_models'], [A, B, C])
            self.assertEqual(st['model_errors'], {})
            snap = self._snapshot([A, C])
            # compare OOM on B: real torch exception propagates, others untouched
            FakeRuntime.oom_on_compare.add(B)
            with self.assertRaises(real_oom):
                self.pool.compare(B, 'p', [])
            st = self.pool.status
            self.assertTrue(st['model_errors'][B]['oom'])
            self.assertEqual(st['model_errors'][B]['type'], 'OutOfMemoryError')
            self.assertNotIn(B, st['ready_models'])
            self.assertEqual(st['ready_models'], [A, C])
            self.assertEqual(st['resident_models'], [A, C])
            self._assert_same(snap)
            before = len(FakeRuntime.log)
            with self.assertRaises(RuntimeError):  # no implicit retry of B
                self.pool.compare(B, 'p', [])
            self.assertEqual(len(FakeRuntime.log), before)
            self.assertEqual(self.pool.compare(C, 'p', [])['provenance']['model'], C)
            self._assert_same(snap)

    def test_unload_one_all_and_unsupported(self):
        self.pool.warm([A, B, C])
        st = self.pool.unload(B)
        self.assertEqual(st['resident_models'], [A, C])
        self.assertEqual(st['ready_models'], [A, C])
        st = self.pool.unload()
        self.assertEqual(st['resident_models'], [])
        self.assertFalse(st['loaded'])
        with self.assertRaises(ValueError):
            self.pool.warm(['Qwen/Nope'])
        with self.assertRaises(ValueError):
            self.pool.compare('Qwen/Nope', 'p', [])
        with self.assertRaises(ValueError):
            self.pool.unload('Qwen/Nope')


if __name__ == '__main__':
    unittest.main()
