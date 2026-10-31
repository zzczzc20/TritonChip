#!/usr/bin/env python3
"""Budget/cache/event policy tests; synthetic records cannot sign off performance."""
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.autotune import DeviceSamples, GraphBudget, SharedGraphBudget, PersistentCache, select, identity_key, event_seconds


def identity():
    return dict(kernel_source='source-sha',shape=[16,16,64],stride=[64,1],dtype='float16',
                software_tree='tree',compiler='compiler',profile='v5',capabilities=['mma'],
                rtl={'cores':3},memory_map={'bytes':1048576},image='image-sha',clock_hz=100000000)


def events(cycles):
    return [dict(clock='mmio-device-clock/1',epoch=1,session_id=2,start=10,end=10+cycles)]


def publish_cache(directory, number):
    PersistentCache(directory)['a'*64]={'candidate':number}


def shared_admission(path):
    return SharedGraphBudget(path).admit('same-kernel')


class PolicyTests(unittest.TestCase):
    def test_independent_samples_exclude_host_gaps_and_preserve_raw_median(self):
        samples=[events(9), [dict(events(3)[0],start=1000000,end=1000003)], events(6)]
        result=self.choose([dict(knobs={'tile':16})],lambda _:DeviceSamples(samples))
        row=result['rows'][0]
        self.assertEqual(row['device_records'],samples[2])
        self.assertEqual(row['seconds'],6/100000000)
        self.assertEqual(row['sample_range_seconds'],[3/100000000,9/100000000])

    def test_independent_samples_reject_empty_excess_and_changed_session(self):
        for samples in ([], [events(1)]*17,
                        [events(1),[dict(events(1)[0],epoch=2)]],
                        [events(1),[dict(events(1)[0],session_id=3)]]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.choose([dict(knobs={})],lambda _:DeviceSamples(samples))

    def test_uncertainty_retains_baseline_and_cache_preserves_decision(self):
        base,wide=dict(knobs={'tile':16}),dict(knobs={'tile':32})
        def run(durations,cache=None,measure=None):
            return select(identity(),[base,wide],capacities={},capabilities=['mma'],
                          budget=GraphBudget(),cache=cache,conservative=base,
                          measure=measure or (lambda c:DeviceSamples([events(n) for n in durations[c==wide]])))
        cache={}
        result=run([[100,105,110],[90,95,101]],cache)
        self.assertEqual(result['selected'],base)
        self.assertIn('inconclusive',result['decision'])
        hit=run(None,cache,lambda _:self.fail('cache remeasured'))
        self.assertTrue(hit['cache_hit'])
        self.assertEqual(hit['rows'],result['rows'])
        self.assertEqual(hit['selected'],base)
        self.assertEqual(run([[100,105,110],[90,95,99]])['selected'],wide)
        self.assertEqual(run([[100,105,110],[90,95,100]])['selected'],base)
        self.assertEqual(run([[100],[90]])['selected'],base)

    def test_reopening_expired_worker_budget_does_not_reset_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'expired.sqlite'
            first=SharedGraphBudget(path,max_seconds=.02)
            time.sleep(.03)
            reopened=SharedGraphBudget(path,max_seconds=.02)
            self.assertEqual(first.started,reopened.started)
            self.assertFalse(reopened.admit('new-kernel'))
            self.assertTrue(reopened.exceeded)

    def test_worker_processes_share_candidate_limit_and_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'graph.sqlite'
            first=SharedGraphBudget(path)
            with ProcessPoolExecutor(max_workers=4) as workers:
                accepted=list(workers.map(shared_admission,[path]*64))
            self.assertEqual(sum(accepted),16)
            reopened=SharedGraphBudget(path)
            self.assertEqual(first.started,reopened.started)
            self.assertFalse(reopened.admit('same-kernel'))
            self.assertTrue(reopened.admit('other-kernel'))
            with self.assertRaisesRegex(ValueError,'configuration'):
                SharedGraphBudget(path,max_seconds=300)
    def test_persistent_cache_reopens_and_rejects_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates=[dict(knobs={'tile':16})]
            self.choose(candidates,lambda _:events(7),cache=PersistentCache(directory))
            reopened=PersistentCache(directory)
            self.assertTrue(self.choose(candidates,lambda _:self.fail('cache reran'),cache=reopened)['cache_hit'])
            next(Path(directory).glob('*.json')).write_text('{broken')
            self.assertFalse(self.choose(candidates,lambda _:events(8),cache=reopened)['cache_hit'])
            changed=identity();changed['image']='other-board-image'
            self.assertFalse(self.choose(candidates,lambda _:events(9),cache=reopened,ident=changed)['cache_hit'])

    def test_persistent_concurrent_publication_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=PersistentCache(directory);key='a'*64
            with ThreadPoolExecutor(max_workers=8) as workers:
                list(workers.map(lambda n:cache.__setitem__(key,{'candidate':n}),range(32)))
            self.assertIn(cache[key]['candidate'],range(32))
            self.assertEqual(len(list(Path(directory).iterdir())),1)
            with self.assertRaises(ValueError):cache['../unsafe']={}

    def test_independent_processes_share_atomic_records(self):
        with tempfile.TemporaryDirectory() as directory:
            with ProcessPoolExecutor(max_workers=4) as workers:
                futures=[workers.submit(publish_cache,directory,n) for n in range(12)]
                for future in futures:future.result()
            self.assertIn(PersistentCache(directory)['a'*64]['candidate'],range(12))
            self.assertEqual(len(list(Path(directory).iterdir())),1)

    def choose(self,candidates,measure,budget=None,cache=None,ident=None,caps=None):
        return select(ident or identity(),candidates,capacities={'vrf':64},
                      capabilities=caps or ['mma'],budget=budget or GraphBudget(),
                      measure=measure,cache=cache)

    def test_actual_event_span_wins_over_active_cycle_sum(self):
        candidates=[dict(knobs={'tile':16}),dict(knobs={'tile':32})]
        def measure(candidate):
            if candidate['knobs']['tile']==32:return events(50)
            a=events(2)[0];b=dict(a,start=110,end=112)
            return [a,b]
        result=self.choose(candidates,measure)
        self.assertEqual(result['selected'],candidates[1])
        self.assertEqual(result['rows'][0]['seconds'],102/100000000)

    def test_resource_and_false_capability_refused_before_callback(self):
        called=[]
        c=[dict(resources={'vrf':65}),dict(requires=['mma']),dict(knobs={'legal':True})]
        result=self.choose(c,lambda x:(called.append(x) or events(5)),caps={'mma':False})
        self.assertEqual(called,[c[-1]])
        self.assertEqual([r['status'] for r in result['rows']],['refused','refused','measured'])

    def test_sixteen_measured_candidates_across_repeated_searches(self):
        budget=GraphBudget();calls=[]
        c=[dict(knobs={'n':n}) for n in range(20)]
        result=self.choose(c,lambda x:(calls.append(x) or events(1)),budget=budget)
        self.assertEqual(len(calls),16)
        self.assertEqual(sum(r['status']=='not-measured' for r in result['rows']),4)
        with self.assertRaises(ValueError):self.choose(c,lambda x:events(1),budget=budget)

    def test_graph_time_budget_counts_compile_and_measure_overrun(self):
        now=[0];budget=GraphBudget(max_seconds=10,clock=lambda:now[0])
        def measure(candidate):now[0]+=6;return events(candidate['n'])
        result=self.choose([{'n':1},{'n':2},{'n':3}],measure,budget=budget,cache={})
        self.assertTrue(result['budget_exceeded'])
        self.assertEqual([r['status'] for r in result['rows']],['measured','over-budget','not-measured'])

    def test_cache_key_covers_every_identity_field(self):
        c=[dict(knobs={'tile':16})];cache={}
        self.choose(c,lambda x:events(4),cache=cache)
        self.assertTrue(self.choose(c,lambda x:self.fail('cache reran'),cache=cache)['cache_hit'])
        for field in identity():
            changed=copy.deepcopy(identity())
            changed[field]=200000000 if field=='clock_hz' else {'changed':True}
            self.assertNotEqual(identity_key(changed),identity_key(identity()))
        changed=identity();changed['image']='new-image'
        self.assertFalse(self.choose(c,lambda x:events(5),cache=cache,ident=changed)['cache_hit'])

    def test_host_time_and_mixed_epoch_are_refused(self):
        c=[dict(knobs={})]
        for measurement in ({'host_seconds':.001},[dict(events(1)[0],clock='host-wall')],
                            [events(1)[0],dict(events(1)[0],epoch=2)]):
            with self.assertRaises(ValueError):self.choose(c,lambda x:measurement)

    def test_missing_identity_and_relaxed_budgets_are_refused(self):
        with self.assertRaises(ValueError):identity_key({'shape':[16]})
        for kwargs in ({'max_seconds':601},{'max_candidates':17},{'max_seconds':0},
                       {'max_seconds':True},{'max_candidates':1.5},{'max_seconds':float('nan')}):
            with self.assertRaises(ValueError):GraphBudget(**kwargs)

    def test_invalid_endpoints_and_frequency_refused(self):
        for value in (-1, 1 << 64, True, 1.5):
            with self.assertRaises(ValueError):
                event_seconds([dict(events(5)[0],start=value)],100)
        for frequency in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):event_seconds(events(5),frequency)

    def test_parallel_requests_share_kernel_and_graph_budget(self):
        budget=GraphBudget()
        with ThreadPoolExecutor(max_workers=8) as workers:
            accepted=list(workers.map(lambda _:budget.admit('same-kernel'),range(64)))
        self.assertEqual(sum(accepted),16)
        self.assertEqual(budget.counts['same-kernel'],16)
        self.assertTrue(budget.admit('other-kernel'))


if __name__=='__main__':unittest.main()
