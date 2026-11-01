"""The persistent namespace must distinguish every compilation input."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('compile_cache', Path(__file__).with_name('compile_cache.py'))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CacheNamespaceTests(unittest.TestCase):
    def test_fingerprint_uses_contents_and_relative_names(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for root in (Path(first), Path(second)):
                (root / 'compiler.py').write_text('first')
                (root / 'rules.td').write_text('rules')
            initial = cache.source_fingerprint({'src': first})
            self.assertEqual(initial, cache.source_fingerprint({'src': second}))
            (Path(second) / 'compiler.py').write_text('changed')
            self.assertNotEqual(initial, cache.source_fingerprint({'src': second}))
            (Path(second) / 'compiler.py').write_text('first')
            (Path(second) / 'rules.td').rename(Path(second) / 'renamed.td')
            self.assertNotEqual(initial, cache.source_fingerprint({'src': second}))

    def test_namespace_tracks_profile_options_and_versions(self):
        def key(env, versions=None, options=None):
            return cache.namespace('source', env, versions or {'torch': '1'}, options or {'tile': 16})
        first = key({'TNCA_PROFILE': 'dp', 'TNCA_COMPILE_CACHE_DIR': '/a',
                     'TNCA_TLM_OUT': '/run-a', 'TNCA_DUMP_DIR': '/dump-a'})
        self.assertEqual(first, key({'TNCA_PROFILE': 'dp', 'TNCA_COMPILE_CACHE_DIR': '/b',
                                     'TNCA_TLM_OUT': '/run-b', 'TNCA_DUMP_DIR': '/dump-b',
                                     'TNCA_SUBMIT': 'single'}))
        self.assertNotEqual(first, key({'TNCA_PROFILE': 'other'}))
        self.assertNotEqual(first, key({'TNCA_PROFILE': 'dp', 'TNCA_NEW_TARGET_KNOB': '1'}))
        self.assertNotEqual(first, key({'TNCA_PROFILE': 'dp'}, {'torch': '2'}))
        self.assertNotEqual(first, key({'TNCA_PROFILE': 'dp'}, options={'tile': 32}))

    def test_runtime_cache_reset_and_test_edits_preserve_compiled_graphs(self):
        base = {'TNCA_PROFILE': 'dp'}
        runtime = dict(base, TNCA_DEVICE_ZERO='1', TNCA_QUEUE_STEPS='1',
                       TNCA_REUSE_WARM_PREFILL='1', TNCA_REUSE_HYBRID_PREFILL='1')
        self.assertEqual(cache.namespace('source', base, {}, {}), cache.namespace('source', runtime, {}, {}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'compiler.py').write_text('compiler')
            (root / 'test_compiler.py').write_text('first')
            (root / 'tests').mkdir()
            (root / 'tests' / 'helper.py').write_text('first')
            before = cache.source_fingerprint({'tnca': root})
            (root / 'test_compiler.py').write_text('changed')
            (root / 'tests' / 'helper.py').write_text('changed')
            self.assertEqual(before, cache.source_fingerprint({'tnca': root}))
            (root / 'compiler.py').write_text('changed')
            self.assertNotEqual(before, cache.source_fingerprint({'tnca': root}))

    def test_untuned_cache_survives_a_runtime_image_path_change(self):
        def key(env):
            return cache.namespace('source', env, {'torch': '1'}, {'tile': 16})
        before = {'TNCA_PROFILE': 'dp', 'TNCA_BACKEND': 'fpga-u280-multicore',
                  'TNCA_BACKEND_ARGS': '{"xclbin":"/old/image"}',
                  'TNCA_OPERATING_CLOCK_EVIDENCE': '/old/clock', 'TNCA_COUNTERS': '1'}
        after = dict(before, TNCA_BACKEND_ARGS='{"xclbin":"/new/image"}',
                     TNCA_OPERATING_CLOCK_EVIDENCE='/new/clock', TNCA_COUNTERS='0')
        self.assertEqual(key(before), key(after))
        # Backend selection itself determines physical-grid lowering.
        self.assertNotEqual(key(before), key(dict(after, TNCA_BACKEND='sim-file')))

    def test_tuned_cache_remains_bound_to_runtime_hardware(self):
        def key(env):
            return cache.namespace('source', env, {'torch': '1'}, {'tile': 16})
        before = {'TNCA_PROFILE': 'dp', 'TNCA_TUNE': '1',
                  'TNCA_BACKEND_ARGS': '{"xclbin":"/old/image"}',
                  'TNCA_OPERATING_CLOCK_EVIDENCE': '/old/clock'}
        self.assertNotEqual(key(before), key(dict(before, TNCA_BACKEND_ARGS='{"xclbin":"/new/image"}')))
        self.assertNotEqual(key(before), key(dict(before, TNCA_OPERATING_CLOCK_EVIDENCE='/new/clock')))
        self.assertNotEqual(key(before), key(dict(before, TNCA_COUNTERS='1')))

    def test_missing_compiler_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                cache.source_fingerprint({'src': Path(root) / 'missing'})
            with self.assertRaises(FileNotFoundError):
                cache.source_fingerprint({'src': root}, {'rules': Path(root) / 'missing'})

    def test_staged_wheel_and_overridden_data_affect_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, compiler, override = root / 'site/tnca', root / 'site/tnca_cc', root / 'override'
            for relative in ('spec', 'isa', 'compiler/targets'):
                (package / 'data' / relative).mkdir(parents=True)
            compiler.mkdir(parents=True)
            (override / 'spec').mkdir(parents=True)
            (override / 'spec/model.json').write_text('first')
            for name in ('r2_abi_v1.json', 'r2_manifest_v1.schema.json'):
                (package / 'data/spec' / name).write_text('{}')
            assembler = [root / 'tnca_asm.py', root / 'tnca_isa.py']
            for path in assembler:
                path.write_text('pass')
            def data_path(*parts, env=None):
                return package / 'data' / Path(*parts)
            trees, files = cache.dependency_paths(package, compiler, data_path, assembler, [override])
            first = cache.source_fingerprint(trees, files)
            (override / 'spec/model.json').write_text('changed')
            self.assertNotEqual(first, cache.source_fingerprint(trees, files))


if __name__ == '__main__':
    unittest.main()
