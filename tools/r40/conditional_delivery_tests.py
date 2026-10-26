#!/usr/bin/env python3
"""Bucket selection must preserve complete original inventories and identity."""
from pathlib import Path
import unittest
import conditional_delivery as delivery


class DeliveryPlanTests(unittest.TestCase):
    def test_bucket_changes_only_the_complete_model_policy(self):
        args=('python',Path('/image.xclbin'),'a'*64,'files:original',Path('/clock.json'),Path('/fresh'),4<<30)
        original=delivery.plan(*args)
        bucket=delivery.plan(*args,bucket=True)
        self.assertEqual(original[:-1],bucket[:-1])
        self.assertEqual(original[-1]['name'],bucket[-1]['name'])
        command=bucket[-1]['command'][:]
        self.assertEqual(command.count('--bucket'),1)
        command.remove('--bucket')
        self.assertEqual(command,original[-1]['command'])
        native=[pid for stage in bucket if stage['name'].startswith('native77-')
                for pid in stage.get('point_ids',[])]
        self.assertEqual(len(native),77)
        self.assertEqual(set(native),{p['id'] for p in delivery.frozen('existing.json')})
        hbm=next(s for s in bucket if s['name']=='original-hbm14')
        self.assertEqual(len(hbm['point_ids']),14)
        self.assertEqual(set(hbm['point_ids']),{p['id'] for p in delivery.frozen('hbm-core.json')})


if __name__=='__main__':unittest.main()
