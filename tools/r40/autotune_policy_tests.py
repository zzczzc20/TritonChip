#!/usr/bin/env python3
"""Budget/cache/event policy tests; synthetic records cannot sign off performance."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.autotune import GraphBudget, select, identity_key, event_seconds


def identity():
    return dict(kernel_source='source-sha',shape=[16,16,64],stride=[64,1],dtype='float16',
                software_tree='tree',compiler='compiler',profile='v5',capabilities=['mma'],
                rtl={'cores':3},memory_map={'bytes':1048576},image='image-sha',clock_hz=100000000)


def events(cycles):
    return [dict(clock='mmio-device-clock/1',epoch=1,session_id=2,start=10,end=10+cycles)]


class PolicyTests(unittest.TestCase):
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
