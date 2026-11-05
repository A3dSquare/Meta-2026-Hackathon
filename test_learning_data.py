import copy
import unittest
from collections import Counter
from test_route_controller import row,config
from route_controller import BalancedReplay,CONTROLLER_DIM
from route_learner import RouteLearner


class LearningDataTests(unittest.TestCase):
    def test_manual_hold_reserve_survives_many_releases_and_gameplay(self):
        replay=BalancedReplay(320)
        for i in range(40):replay.add(dict(row(1),source='human'))
        for i in range(4000):
            replay.add(dict(row(0,teacher=0),source='human'))
            replay.add(row(0,teacher=0))
        human=Counter(r['action'] for r in replay.rows() if r.get('source')=='human')
        self.assertEqual(human,{0:10,1:10})
        clone=BalancedReplay(320);clone.load(copy.deepcopy(replay.state()))
        self.assertEqual(len(clone),len(replay))
        clone.clear_human()
        self.assertTrue(clone.rows())
        self.assertFalse(any(r.get('source')=='human' for r in clone.rows()))

    def test_old_replay_migration_protects_surviving_human_examples(self):
        replay=BalancedReplay(320)
        state=replay.state();state.pop('version')
        state['buckets']={(0,True):[dict(row(),source='human'),row()]}
        state['seen']={(0,True):10000}
        replay.load(state)
        self.assertEqual(len(replay.buckets[('human',0,1)]),1)
        self.assertEqual(len(replay.buckets[(0,True)]),1)
        self.assertEqual(replay.seen[('human',0,1)],1)

    def test_planner_guesses_do_not_count_as_witnessed_accuracy(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        rows=[dict(row(1,teacher=-1),planner=1) for _ in range(30)]
        rows += [dict(row(0,teacher=0),source='human') for _ in range(6)]
        metrics=learner._metrics(learner.candidate,rows)
        self.assertEqual(metrics['action_label_counts'],[6,0])
        self.assertEqual(metrics['human_action_label_counts'],[6,0])
        self.assertIsNone(metrics['action_recall'][1])
        self.assertEqual(metrics['planner_labels'],30)

    def test_coaching_balances_actions_then_sources(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.replay.add(dict(row(1),source='human'))
        learner.replay.add(dict(row(0,teacher=0),source='human'))
        for i in range(500):learner.replay.add(dict(row(1,teacher=-1),planner=1))
        draws=learner._coaching_rows(4000)
        holds=[r for r in draws if (r['teacher'] if r['teacher']>=0 else r['planner'])==1]
        self.assertTrue(1700<len(holds)<2300)
        human=sum(r.get('source')=='human' for r in holds)
        self.assertTrue(.4<human/len(holds)<.6)


if __name__=='__main__':unittest.main()
