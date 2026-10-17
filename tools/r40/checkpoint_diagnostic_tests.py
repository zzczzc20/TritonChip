#!/usr/bin/env python3
"""Check diagnostic storage accounting without opening a Vivado checkpoint."""
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'flow/fpga/diagnose_r40_checkpoint.tcl'


class StorageAccounting(unittest.TestCase):
    def report(self, rows, out):
        # Only the pure Tcl reporting procedures precede this argument guard.
        procedures = SCRIPT.read_text().split('\nif {[llength $argv] ni {2 3}}', 1)[0]
        data = '\n'.join('{' + ' '.join(map(str, row)) + '}' for row in rows)
        body = (procedures + '\nset rows {\n' + data + '\n}\n' +
                'if {[catch {r40_tag_storage_report $rows {' + str(out/'cells.tsv') +
                '} {' + str(out/'summary.tsv') + '}} message]} ' +
                '{puts stderr $message; exit 1}\n')
        return subprocess.run(['tclsh'], input=body, text=True,
                              capture_output=True, timeout=15)

    def test_macro_children_are_not_added_to_storage_roots(self):
        # Hierarchical RAM ratios observed in the retained three-cache OOC
        # screen: 96 RAM32M16 roots, 1,344 RAMD32 and 192 RAMS32 children.
        # From evidence/r40/v2/convergence/attempt02/preflight.json. Keep this
        # small regression fixture usable in source exports without evidence/.
        observed = {'RAM32M16': 96, 'RAMD32': 1344, 'RAMS32': 192}
        rows = []
        for core in range(3):
            cache = f'inst/u_system/g_tile[{core}].u_tile/u_slice'
            for way in range(8):
                for bank in range(4):
                    name = f'{cache}/g_sync_tags.g_way[{way}].bank_reg_{bank}'
                    rows.append((name, 'RAM32M16', 0))
                    rows.extend((f'{name}/RAMD{i}', 'RAMD32', 1) for i in range(14))
                    rows.extend((f'{name}/RAMS{i}', 'RAMS32', 1) for i in range(2))
                    rows.append((f'{name}/decode', 'LUT1', 1))
        rows.append(rows[0])  # Repeated canonical cells must not inflate counts.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.report(rows, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out/'summary.tsv').open() as stream:
                summary = list(csv.DictReader(stream, delimiter='\t'))
            totals = {}
            for row in summary:
                key = row['scope'], row['primitive']
                totals[key] = totals.get(key, 0) + int(row['unique_cells_DO_NOT_SUM_scopes'])
            self.assertEqual(totals['storage_root', 'RAM32M16'], observed['RAM32M16'])
            self.assertEqual(totals['storage_child', 'RAMD32'], observed['RAMD32'])
            self.assertEqual(totals['storage_child', 'RAMS32'], observed['RAMS32'])
            self.assertEqual(len({row['cache'] for row in summary}), 3)
            with (out/'cells.tsv').open() as stream:
                inventory = list(csv.DictReader(stream, delimiter='\t'))
            for row in inventory:
                if row['scope'] == 'storage_child':
                    self.assertTrue(row['storage_parent'].endswith(tuple(f'bank_reg_{i}' for i in range(4))))

    def test_exploded_ram_and_register_fallback_remain_visible(self):
        rows = [('g_sync_tags.g_way[0].bank_reg', 'RAMD32', 1),
                ('g_sync_tags.g_way[1].bank_reg[0]', 'FDRE', 1),
                ('g_sync_tags.g_way[1].bank_reg_mux', 'LUT6', 1)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.report(rows, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out/'cells.tsv').open() as stream:
                inventory = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual({(row['scope'], row['primitive']) for row in inventory},
                             {('storage_root', 'RAMD32'), ('storage_flop', 'FDRE'), ('helper', 'LUT6')})
            self.assertTrue(all(row['cache'] == '<top>' for row in inventory))

    def test_missing_or_conflicting_storage_is_refused(self):
        fixtures = [[], [('g_sync_tags.bank_reg_mux', 'LUT6', 1)],
                    [('g_sync_tags.bank_reg', 'FDRE', 1),
                     ('g_sync_tags.bank_reg', 'RAMD32', 1)]]
        for rows in fixtures:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self.assertNotEqual(self.report(rows, out).returncode, 0)
                self.assertFalse((out/'cells.tsv').exists())


if __name__ == '__main__':
    unittest.main()
