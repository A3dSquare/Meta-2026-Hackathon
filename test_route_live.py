"""Full route-controller stack on rendered frames with a fake keyboard/capture."""
import tempfile
from pathlib import Path
import sys


def main():
    import test_live_stack as fake
    import sim_gd as sim
    import winfocus
    winfocus.gd_is_foreground=lambda:None
    winfocus.focus_gd=lambda:False
    from env import GymGDEnv,EnvConfig
    from env_boxes import EntityBoxEnv
    from gd import GDInput
    from route_controller import RouteObservationEnv
    from trainer import Trainer,TrainConfig
    class Capture(fake.FakeCap):
        def __init__(self,game):super().__init__(game);self._ticks=0;self._overruns=0;self.last_frame_ts=0.
        def read(self):
            result=super().read();self._ticks+=1;self.last_frame_ts=self._ticks/60.
            return result
    fake.KEYS.down.clear();fake.KEYS.events.clear()
    game=fake.FakeGame(([sim.Obj('spike',16,0,1,.95),sim.Obj('wall',28,0,2,1),sim.Obj('spike',38,0,1,.95)],300.))
    capture=Capture(game)
    base=GymGDEnv(EnvConfig(max_steps=10**9,pct_hook=lambda _:game.pct),cap=capture,inp=GDInput(backend=fake.KEYS))
    base.settle_s=0.
    entity=EntityBoxEnv(base,frame_skip=2,motion_path=None)
    env=RouteObservationEnv(entity)
    with tempfile.TemporaryDirectory() as directory:
        cfg=TrainConfig(route_controller=True,total_steps=600,save_dir=directory,log_dir=directory,
            name='route_controller',route=False,verbose=0,net_arch=(32,32),buffer_size=512,
            warmup=16,batch_size=16,update_steps=32,max_rollout=128,max_grad_steps=4,replay_ratio=.1,hold_cost=0.)
        trainer=Trainer(env,cfg);trainer.run()
        assert trainer.steps==600 and trainer.learner.size()>0 and trainer.updates>0
        assert (Path(directory)/'route_controller.pt').exists()
        print('PASS: route observations, planner bootstrap, measured replay, training and atomic save;',
              'steps',trainer.steps,'updates',trainer.updates,'replay',trainer.learner.size(),
              'heldout',len(trainer.learner.validation),'promotions',trainer.learner.promotion_count)
    base.inp.close()


if __name__=='__main__':main()
