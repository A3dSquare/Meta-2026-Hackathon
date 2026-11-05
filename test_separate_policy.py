import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from route_learner import RouteLearner
from route_controller import CONTROLLER_DIM
from test_route_controller import config,observation,row


class SeparatePolicyTests(unittest.TestCase):
    def test_losses_have_disjoint_gradients(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config());net=learner.candidate
        obs=torch.tensor(np.array([observation(),observation(1)]))
        net(obs).square().sum().backward()
        self.assertTrue(all(p.grad is None for n,p in net.named_parameters() if n.startswith('policy_')))
        net.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(net.policy(obs),torch.tensor([0,1])).backward()
        self.assertTrue(any(p.grad is not None for n,p in net.named_parameters() if n.startswith('policy_')))
        self.assertTrue(all(p.grad is None for n,p in net.named_parameters() if not n.startswith('policy_')))

    def test_old_checkpoint_migration_preserves_actions_weights_and_optimizer(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        for i in range(32):learner.replay.add(row(i%2,teacher=i%2))
        learner.train(8,queue_live_trial=False)
        obs=torch.tensor(np.array([observation(),observation(1)]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.pt';learner.save(str(path))
            ck=torch.load(path,weights_only=False)
            ck.pop('action_policy_version');ck['evaluation_policy_version']=3
            for key in ('q','candidate','target'):
                ck[key]={n:v for n,v in ck[key].items() if not n.startswith('policy_')}
            old_count=sum(not n.startswith('policy_') for n,_ in learner.candidate.named_parameters())
            ck['opt']['param_groups'][0]['params']=list(range(old_count))
            ck['opt']['state']={i:s for i,s in ck['opt']['state'].items() if i<old_count}
            torch.save(ck,path);source=path.read_bytes()
            restored=RouteLearner(CONTROLLER_DIM,2,config());restored.load(str(path))
            for key,net in [('q',restored.q),('candidate',restored.candidate),('target',restored.target)]:
                for name,value in ck[key].items():torch.testing.assert_close(net.state_dict()[name],value)
                with torch.no_grad():
                    self.assertTrue(torch.equal(net(obs).argmax(1),net.policy(obs).argmax(1)))
            for i,state in ck['opt']['state'].items():
                for key,value in state.items():torch.testing.assert_close(restored.opt.state_dict()['state'][i][key],value)
            self.assertIsNone(restored.trial)
            restored.train(8,queue_live_trial=False)
            self.assertEqual(path.read_bytes(),source)
            updated=Path(tmp)/'new.pt';restored.save(str(updated))
            again=RouteLearner(CONTROLLER_DIM,2,config());again.load(str(updated))
            with torch.no_grad():torch.testing.assert_close(again.candidate.policy(obs),restored.candidate.policy(obs))

    def test_action_and_validation_use_policy_not_value_argmax(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        for net in (learner.q,learner.candidate):
            with torch.no_grad():
                for p in net.parameters():p.zero_()
                net.advantages[0].bias[0]=10.
                net.policy_heads[0].bias[1]=10.
        learner.trusted_modes={0}
        action,_=learner.act(observation(),deterministic=True)
        self.assertEqual(action,1)
        self.assertEqual(learner._metrics(learner.candidate,[row(1)])['accuracy'],1.)


if __name__=='__main__':unittest.main()
