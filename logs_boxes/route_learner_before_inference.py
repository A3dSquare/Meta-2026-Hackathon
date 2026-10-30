"""A separately checkpointed, mode-specific controller with a learned world model.

Candidate weights train in shadow. Deployment is gated on episode-held-out
action supervision, safety scores and measured multi-frame motion error.
This is an offline validation gate, not proof of improved live completion.
"""
import copy
import math
import os
import numpy as np
from route_controller import (VERSION,HORIZON,HEADER,TARGET_TIMES,TARGET_START,ANALYTIC_START,
                              CONTROLLER_DIM,VALID_START,CONTROLLER_MODE_START,MODES,LEGACY_MODES,BalancedReplay,mode_indices)
from env_boxes import OBS_DIM,OFF_MODE,OFF_SEARCH,UNIT


def build_network(obs_dim,arch):
    import torch as th
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            # Action SEQUENCES and their actual durations condition motion;
            # changing inputs are not mislabeled as "hold this input forever".
            self.motion_encoder=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.motion_cell=nn.GRUCell(3,128)
            self.motion_heads=nn.ModuleList([nn.Linear(128,2) for _ in MODES])
            layers=[];d=obs_dim+HORIZON*4+1
            for width in arch:layers.extend((nn.Linear(d,width),nn.ReLU()));d=width
            self.trunk=nn.Sequential(*layers)
            self.values=nn.ModuleList([nn.Linear(d,1) for _ in MODES])
            self.advantages=nn.ModuleList([nn.Linear(d,2) for _ in MODES])
            self.register_buffer('motion_trust',th.zeros(len(MODES)))

        def modes(self,o):
            modes=o[:,CONTROLLER_MODE_START:CONTROLLER_MODE_START+len(MODES)]
            legacy=o[:,OFF_MODE:OFF_MODE+len(LEGACY_MODES)].argmax(1)
            return th.where(modes.sum(1)>.5,modes.argmax(1),legacy)

        def dynamics(self,o,actions,durations):
            z=self.motion_encoder(o);modes=self.modes(o)
            lag=(o[:,OBS_DIM+5]*8).round().long().clamp(0,8)
            previous=o[:,OBS_DIM+30]
            points=[]
            for j in range(HORIZON):
                index=(j-lag).clamp(0,HORIZON-1)
                commanded=actions.gather(1,index[:,None]).squeeze(1)
                effective=th.where(j<lag,o[:,OBS_DIM+14+j],commanded)
                # Only inputs that have reached this interval are visible.
                # Future feedback actions cannot leak information backward
                # into the prediction of an earlier state.
                z=self.motion_cell(th.stack((effective,previous,durations[:,j]),dim=1),z)
                values=th.stack([head(z) for head in self.motion_heads],dim=1)
                points.append(values[th.arange(len(o),device=o.device),modes])
                previous=effective
            return th.stack(points,dim=1)

        def forward(self,o):
            dt=o[:,OBS_DIM+2:OBS_DIM+3].clamp(.01,.1).expand(-1,HORIZON)
            trust=self.motion_trust[self.modes(o)][:,None]
            predictions=[]
            for action in (0.,1.):
                begin=ANALYTIC_START+int(action)*HORIZON*2
                prior=o[:,begin:begin+HORIZON*2]
                valid=o[:,VALID_START+int(action)*HORIZON:VALID_START+(int(action)+1)*HORIZON]
                used=prior.clone()
                selected=(trust[:,0]>.5) & (valid.max(1).values>.5)
                if bool(selected.any()):
                    with th.no_grad():
                        predicted=self.dynamics(o[selected],th.full_like(dt[selected],action),dt[selected]).flatten(1)
                    used[selected]=predicted.clamp(-2.,2.)
                predictions.append(used*valid.repeat_interleave(2,dim=1))
            z=self.trunk(th.cat((o,*predictions,trust),dim=1))
            qs=[]
            for value,advantage in zip(self.values,self.advantages):
                a=advantage(z);qs.append(value(z)+a-a.mean(1,keepdim=True))
            return th.stack(qs,dim=1)[th.arange(len(o),device=o.device),self.modes(o)]
    return Network()


class RouteLearner:
    def __init__(self,obs_dim,n_act,cfg):
        import torch as th
        if obs_dim!=CONTROLLER_DIM or n_act!=2:raise ValueError('route controller requires RouteObservationEnv and two inputs')
        self.th,self.cfg,self.obs_dim=th,cfg,obs_dim
        self.dev=th.device(cfg.device)
        if self.dev.type=='cpu' and cfg.cpu_threads>0:th.set_num_threads(cfg.cpu_threads)
        self.q=build_network(obs_dim,cfg.net_arch).to(self.dev)
        self.candidate=copy.deepcopy(self.q);self.target=copy.deepcopy(self.q)
        self.opt=th.optim.Adam(self.candidate.parameters(),lr=cfg.lr)
        self.replay=BalancedReplay(cfg.buffer_size)
        self.validation=BalancedReplay(max(256,cfg.buffer_size//5),seed=1)
        self.rng=np.random.default_rng(0);self.steps_seen=self.grad_steps=0
        self.episode_id=1;self.promotion_count=0;self.promotion_pending=False
        self.episode_open=False
        self.trusted_modes=set();self.model_verified_modes=set();self.context={}
        self.last_stats={};self.last_evaluation={};self.last_role='bootstrap'
        self.data_migration={};self._migration_source=None
        self.trial_model=None;self.trial=None;self.trial_modes=set();self.trial_skip_until_reset=False
        self.trial_failures=0;self.next_trial_episode=13;self.next_trial_grad=512

    @property
    def evaluating(self):return self.trial is not None

    def _active_network(self):
        return self.trial_model if self.evaluating and not self.cfg.play and self.trial.arm=='candidate' and not self.trial_skip_until_reset else self.q

    def size(self):return len(self.replay)

    def epsilon(self):
        f=min(1.,self.steps_seen/max(1,self.cfg.eps_decay_steps))
        return self.cfg.eps_start+(self.cfg.eps_end-self.cfg.eps_start)*f

    def set_context(self,env):
        self.context=getattr(env,'route_context',{})
        env.model_verified_modes={MODES[i] for i in self.model_verified_modes}
        source='candidate' if self.last_evaluation.get('promoted') else 'incumbent'
        env.controller_uncertainty={mode:(report[source]['position_p95'],report[source]['velocity_p95'])
            for mode,report in self.last_evaluation.get('modes',{}).items() if MODES.index(mode) in self.trusted_modes}

    def q_values(self,obs):
        with self.th.no_grad():
            return self._active_network()(self.th.as_tensor(obs[None],device=self.dev,dtype=self.th.float32))[0].cpu().numpy()

    def act(self,obs,deterministic=False):
        q=self.q_values(obs);mode=int(mode_indices(obs))
        active_modes=self.trusted_modes | (self.trial_modes if self.evaluating and not self.cfg.play and not self.trial_skip_until_reset and self.trial.arm=='candidate' else set())
        self.last_role='learned' if mode in active_modes else ('planner bootstrap' if self.context.get('verified') else 'planner: limited forecast')
        if self.evaluating:self.last_role='evaluation '+self.trial.arm+' / '+self.last_role
        if not deterministic and not self.evaluating:
            self.steps_seen+=1
            if self.rng.random()<self.epsilon():return int(self.rng.integers(2)),dict(q=q,controller_role='exploration',motion_prediction=self.predict_actions(obs))
        if mode not in active_modes:
            if self.context.get('first_action') is not None:action=int(self.context['first_action'])
            elif obs[OFF_SEARCH+3]>.5:action=int(obs[OFF_SEARCH]>max(obs[OFF_SEARCH+1],obs[OFF_SEARCH+2]))
            else:action=0
        else:action=int(q[1]>q[0])
        return action,dict(q=q,controller_role=self.last_role,motion_prediction=self.predict_actions(obs))

    def predict_actions(self,obs):
        net=self._active_network();th=self.th;mode=int(mode_indices(obs))
        tensor=th.as_tensor(obs[None],device=self.dev,dtype=th.float32)
        dt=float(np.clip(obs[OBS_DIM+2],.01,.1));durations=th.full((1,HORIZON),dt,device=self.dev)
        learned=bool(net.motion_trust[mode]>.5)
        result=dict(source='learned' if learned else 'analytical prior',origin_y=float(obs[0]*8.),
                    times=[(j+1)*dt for j in range(HORIZON)])
        with th.no_grad():
            for action,name in enumerate(('release','hold')):
                values=(net.dynamics(tensor,th.full_like(durations,float(action)),durations)[0].clamp(-2.,2.).cpu().numpy()
                        if learned else obs[ANALYTIC_START+action*HORIZON*2:ANALYTIC_START+(action+1)*HORIZON*2].reshape(HORIZON,2))
                result[name]=(values*np.array([8.,20.])).tolist()
                result[name+'_valid']=obs[VALID_START+action*HORIZON:VALID_START+(action+1)*HORIZON].astype(bool).tolist()
        return result

    def hold_prob(self,obs):
        q=self.q_values(obs)
        return float(1/(1+math.exp(float(np.clip(q[0]-q[1],-60,60)))))

    def planner_label(self):
        c=self.context
        return int(c['first_action']) if c.get('verified') and c.get('tracking_valid') and c.get('first_action') is not None else -1

    @staticmethod
    def _observed_label(obs,nxt,label,action):
        if label<0 or label!=action:return -1
        if min(obs[OBS_DIM+10],nxt[OBS_DIM+10])<.5:return -1
        dt=float(nxt[OBS_DIM+3])
        if not .005<=dt<=.1:return -1
        # First observed interval validates the pending-input model. A label
        # is still only a candidate; death-tail revocation happens upstream.
        points=obs[TARGET_START:ANALYTIC_START].reshape(-1,4)
        mask=points[:,3]>.5
        if not mask.any():return -1
        times=np.r_[0.,TARGET_TIMES[mask]]
        heights=np.r_[obs[OBS_DIM+11]*8.,points[mask,1]*8.]
        if dt>times[-1]:return -1
        expected=np.interp(dt,times,heights)
        observed=float(nxt[0]-obs[0])*UNIT
        return label if abs(observed-expected)<=.25 else -1

    def add_segment(self,obs,act,rew,next_obs_last,terminal,upto=None,teacher=None,route_records=None,completed=False):
        T=len(act);upto=T if upto is None else min(T,upto)
        states=list(obs)+([next_obs_last] if next_obs_last is not None else [])
        # Episode split is stable across max-rollout flushes and restarts.
        held_out=self.episode_id%5==0
        dest=self.validation if held_out else self.replay
        remaining=np.cumsum([float(s[OBS_DIM+3]) for s in states[:T]][::-1])[::-1]
        for t in range(upto):
            m=min(self.cfg.n_step,T-t);end=t+m
            nxt=states[end] if end<len(states) else None
            if nxt is None and not (terminal and end==T):continue
            if nxt is None:nxt=states[t]
            o=np.array(obs[t],np.float32,copy=True)
            sequence=np.zeros(HORIZON,np.float32);durations=np.zeros(HORIZON,np.float32)
            outcomes=np.zeros((HORIZON,2),np.float32);mask=np.zeros(HORIZON,np.float32)
            mode=int(mode_indices(o))
            for j in range(HORIZON):
                k=t+j
                if k>=T or k+1>=len(states) or (terminal and k>=T-1):break
                nx=states[k+1]
                if (int(mode_indices(nx))!=mode or min(o[OBS_DIM+10],nx[OBS_DIM+10])<.5
                        or o[OBS_DIM+6]!=nx[OBS_DIM+6] or o[OBS_DIM+7]!=nx[OBS_DIM+7]):break
                dt=float(nx[OBS_DIM+3])
                if not .005<=dt<=.1:break
                sequence[j]=act[k];durations[j]=dt
                previous=states[k]
                # Targets must be OBSERVATIONS, not another model's fitted
                # velocity. A stale arc estimate used to teach "falling"
                # while the same sample's recorded position was rising.
                measured_velocity=(float(nx[0])-float(previous[0]))*8./dt
                outcomes[j]=[(float(nx[0])-float(o[0])),measured_velocity/20.]
                mask[j]=1.
            label=int(teacher[t]) if teacher is not None else -1
            label=self._observed_label(o,states[t+1],label,act[t]) if t+1<len(states) else -1
            if label>=0:
                # Validate several REAL frames, including time after the
                # delayed first command can actually affect the player.
                record=route_records[t] if route_records else {}
                plan=record.get('plan',{});controls={c['decision']:int(c['hold']) for c in plan.get('controls',[])}
                required=max(2,int(round(o[OBS_DIM+5]*8))+1)
                elapsed=0.;matched=0;contradicted=False
                from route_controller import route_point
                for j in range(HORIZON):
                    if not mask[j] or controls.get(j)!=int(sequence[j]):break
                    elapsed+=durations[j];point=route_point(plan,elapsed)
                    if point is None:break
                    before=route_point(plan,elapsed-durations[j])
                    expected_velocity=(point[1]-before[1])/durations[j] if before is not None else point[2]
                    if abs(outcomes[j,0]*8.-point[1])>.3 or abs(outcomes[j,1]*20.-expected_velocity)>max(5.,.15/durations[j]):
                        contradicted=True;break
                    matched+=1
                if matched<required or contradicted:label=-1
            survival_seconds=remaining[t+1] if t+1<T else 0.
            discounted_return=0.;discount=1.
            for j in range(m):
                discounted_return+=discount*rew[t+j]
                interval=states[t+j+1][OBS_DIM+3] if t+j+1<len(states) else states[t+j][OBS_DIM+2]
                discount*=self.cfg.gamma**(float(np.clip(interval,.005,.1))/(1/30))
            row=dict(obs=o,nxt=np.array(nxt,np.float32,copy=True),action=int(act[t]),
                ret=discounted_return,discount=0. if terminal and end==T else discount,
                mode=mode,success=bool(completed or not terminal or survival_seconds>=1.5),teacher=label,
                actions=sequence,durations=durations,outcomes=outcomes,mask=mask,episode=self.episode_id)
            # Planner distillation is a recommendation, NOT a claim that an
            # unexecuted action succeeded. Keep it separate from witnessed labels.
            record=route_records[t] if route_records else {}
            row['planner']=(int(record['first_action']) if record.get('verified') and record.get('tracking_valid')
                and record.get('first_action') in (0,1) and row['success'] and o[OBS_DIM+10]>.5 else -1)
            dest.add(row)
        self.episode_open=not terminal
        if terminal:self.episode_id+=1
        return upto

    def _batch(self,rows):
        th=self.th
        def tensor(key,dtype=th.float32):return th.as_tensor(np.array([r[key] for r in rows]),device=self.dev,dtype=dtype)
        result={key:tensor(key,th.long if key in ('action','teacher') else th.float32)
                for key in ('obs','nxt','action','ret','discount','teacher','actions','durations','outcomes','mask')}
        result['guide']=th.as_tensor([r['teacher'] if r['teacher']>=0 else r.get('planner',-1) for r in rows],device=self.dev,dtype=th.long)
        return result

    def _coaching_buckets(self):
        buckets={}
        for row in self.replay.rows():
            label=row['teacher'] if row['teacher']>=0 else row.get('planner',-1)
            if label>=0:buckets.setdefault((row['mode'],label),[]).append(row)
        return buckets

    def _coaching_rows(self,n,buckets=None):
        # Preserve balanced sampling/RNG order. Replay stays fixed during a
        # paused training cycle, so partition it once rather than per gradient.
        if buckets is None:buckets=self._coaching_buckets()
        keys=list(buckets)
        if not keys:return []
        return [buckets[key][int(self.rng.integers(len(buckets[key])))]
                for key in (keys[int(self.rng.integers(len(keys)))] for _ in range(n))]

    def train(self,new_steps,tick=None):
        th,c=self.th,self.cfg
        if self.evaluating:return {}
        if len(self.replay)<c.warmup:return {}
        self.candidate.train();losses=[];motion_losses=[];teacher_draws=0
        count=min(c.max_grad_steps,max(1,int(c.replay_ratio*new_steps)))
        coaching_buckets=self._coaching_buckets()
        for j in range(count):
            if tick is not None and j%25==0:tick()
            b=self._batch(self.replay.sample(c.batch_size))
            with th.no_grad():
                action=self.candidate(b['nxt']).argmax(1)
                target=b['ret']+b['discount']*self.target(b['nxt']).gather(1,action[:,None]).squeeze(1)
            q=self.candidate(b['obs']);pred=q.gather(1,b['action'][:,None]).squeeze(1)
            td=th.nn.functional.smooth_l1_loss(pred,target)
            coaching=self._coaching_rows(c.batch_size,coaching_buckets)
            valid=b['teacher']>=0
            if coaching:
                guide=self._batch(coaching);guide_q=self.candidate(guide['obs'])
                per_row=th.nn.functional.cross_entropy(guide_q,guide['guide'],reduction='none')
                weights=th.where(guide['teacher']>=0,1.,.25)
                imitation=(per_row*weights).sum()/weights.sum().clamp_min(1.)
            else:imitation=q.sum()*0.
            motion=self.candidate.dynamics(b['obs'],b['actions'],b['durations'])
            error=th.nn.functional.smooth_l1_loss(motion,b['outcomes'],reduction='none').mean(2)
            dynamics=(error*b['mask']).sum()/b['mask'].sum().clamp_min(1.)
            # The velocity target is the measured interval-average velocity.
            # Enforce consistency with displacement over each actual interval.
            deltas=motion[:,:,0]-th.cat((th.zeros_like(motion[:,:1,0]),motion[:,:-1,0]),dim=1)
            consistency=((deltas-motion[:,:,1]*20./8.*b['durations'])**2*b['mask']).sum()/b['mask'].sum().clamp_min(1.)
            loss=td+imitation+dynamics+.05*consistency
            self.opt.zero_grad();loss.backward();th.nn.utils.clip_grad_norm_(self.candidate.parameters(),c.max_grad_norm)
            self.opt.step();self.grad_steps+=1
            if self.grad_steps%c.target_every==0:self.target.load_state_dict(self.candidate.state_dict())
            losses.append(float(loss.detach()));motion_losses.append(float(dynamics.detach()))
            teacher_draws+=int(valid.sum())
        self.candidate.eval()
        self.last_evaluation=self.evaluate_candidate()
        if self.last_evaluation.get('offline_passed') and self.episode_id>=self.next_trial_episode and self.grad_steps>=self.next_trial_grad:
            from controller_evaluation import LiveComparison
            self.trial_model=copy.deepcopy(self.candidate).eval()
            self.trial_modes=set(self.last_evaluation['eligible_modes'])
            self.trial=LiveComparison()
            self.last_evaluation['reason']='offline checks passed; frozen live comparison queued'
        elif self.last_evaluation.get('offline_passed'):
            self.last_evaluation.update(reason='offline checks passed; collecting more training before another live comparison',
                training_episodes_remaining=max(0,self.next_trial_episode-self.episode_id),
                gradient_steps_remaining=max(0,self.next_trial_grad-self.grad_steps))
        promoted=self.last_evaluation['promoted']
        self.last_stats=dict(loss=float(np.mean(losses)),motion_loss=float(np.mean(motion_losses)),grad_steps=count,
            teacher_draws=teacher_draws,buffer=len(self.replay),validation=len(self.validation),
            eps=self.epsilon(),promoted=promoted,promotions=self.promotion_count)
        return self.last_stats

    def _metrics(self,network,rows):
        th=self.th;b=self._batch(rows)
        with th.no_grad():
            qs=network(b['obs']);action=qs.argmax(1)
            predictions=network.dynamics(b['obs'],b['actions'],b['durations'])
            squared=(predictions-b['outcomes'])**2
            labels=b['teacher']>=0
            accuracy=float((action[labels]==b['teacher'][labels]).float().mean()) if labels.any() else None
            counts=[int((b['guide']==a).sum()) for a in (0,1)]
            recalls=[float((action[b['guide']==a]==a).float().mean()) if counts[a] else None for a in (0,1)]
            mask=b['mask'];n=mask.sum().clamp_min(1.)
            dy=float(th.sqrt((squared[:,:,0]*mask).sum()/n)*8.)
            interval_velocity=float(th.sqrt((squared[:,:,1]*mask).sum()/n)*20.)
            elapsed=b['durations'].cumsum(1).clamp_min(.005)
            predicted_velocity=(predictions[:,:,1]*20.*b['durations']).cumsum(1)/elapsed
            observed_velocity=b['outcomes'][:,:,0]*8./elapsed
            velocity_error=predicted_velocity-observed_velocity
            vy=float(th.sqrt((velocity_error**2*mask).sum()/n))
            observed=mask>.5
            p95=float(th.quantile(th.sqrt(squared[:,:,0][observed]),.95)*8.) if observed.any() else None
            v95=float(th.quantile(velocity_error[observed].abs(),.95)) if observed.any() else None
            survival=b['obs'][:,OFF_SEARCH:OFF_SEARCH+3]
            press=survival[:,0];release=survival[:,1:].max(1).values
            unsafe=th.where(action.bool(),press,release)<.895
            avoidable=th.maximum(press,release)>=1.-1e-6
            safety=float((unsafe & avoidable).float().mean())
            target=b['ret']+b['discount']*network(b['nxt']).max(1).values
            td=float(th.nn.functional.smooth_l1_loss(qs.gather(1,b['action'][:,None]).squeeze(1),target))
        return dict(n=len(rows),labels=int(labels.sum()),accuracy=accuracy,position_rmse=dy,velocity_rmse=vy,
                    position_p95=p95,velocity_p95=v95,
                    interval_velocity_rmse=interval_velocity,action_label_counts=counts,action_recall=recalls,
                    unsafe_rate=safety,td_error=td,motion_samples=int(mask.sum()))

    def evaluate_candidate(self):
        rows=self.validation.rows();report=dict(promoted=False,modes={},reason='insufficient held-out episodes')
        eligible=[];improved=False;passed=True
        for mode in sorted({r['mode'] for r in rows}):
            group=[r for r in rows if r['mode']==mode]
            if len(group)<self.cfg.controller_eval_min:continue
            old=self._metrics(self.q,group);new=self._metrics(self.candidate,group)
            witnessed={r['action'] for r in group if r['mask'][0]>.5}
            motion_ok=(new['motion_samples']>=self.cfg.controller_eval_min and witnessed=={0,1}
                       and new['position_rmse']<=.3 and new['velocity_rmse']<=4.)
            if motion_ok:
                # Evaluate the EXACT policy input representation that would
                # be deployed, including the newly trusted motion features.
                with self.th.no_grad():self.candidate.motion_trust[mode]=1.
                new=self._metrics(self.candidate,group)
            supervised=new['labels']>=max(8,self.cfg.controller_eval_min//4)
            # Overall agreement on only releases (or only holds) cannot show
            # that a policy can time jumps. Check both on episode-held-out data.
            balanced=(min(new['action_label_counts'])>=4 and all(x is not None and x>=.85 for x in new['action_recall']))
            if mode not in self.trusted_modes:
                ok=motion_ok and supervised and balanced and new['accuracy']>=.75 and new['unsafe_rate']<=.05
                improved=improved or ok
            else:
                ok=(motion_ok and balanced and new['unsafe_rate']<=old['unsafe_rate']+.005
                    and new['position_rmse']<=old['position_rmse']*1.05+.01
                    and new['velocity_rmse']<=old['velocity_rmse']*1.05+.1
                    and (not supervised or new['accuracy']>=old['accuracy']-.01)
                    and new['td_error']<=old['td_error']*1.1+.01)
                improved=improved or (supervised and new['accuracy']>old['accuracy']+.005) or new['position_rmse']<old['position_rmse']-.01
            report['modes'][MODES[mode]]=dict(incumbent=old,candidate=new,passed=bool(ok),motion_verified=motion_ok,
                balanced_action_supervision=balanced)
            if mode in self.trusted_modes or ok:
                eligible.append(mode);passed=passed and ok
        # A shared encoder update cannot silently bypass a previously trusted
        # mode's evaluation when that mode lacks validation coverage.
        passed=passed and self.trusted_modes.issubset(set(eligible))
        if eligible and passed and improved:
            report.update(offline_passed=True,eligible_modes=eligible,
                          reason='passed offline checks; live comparison required before deployment')
        elif report['modes']:report['reason']='candidate did not pass all incumbent protection checks'
        report['trusted_modes']=[MODES[i] for i in sorted(self.trusted_modes)]
        return report

    def finish_evaluation_attempt(self,progress,survived,complete=False):
        if self.trial_skip_until_reset:
            self.trial_skip_until_reset=False;self.trial.prints=[];self.trial.steps=self.trial.interventions=0
            return dict(complete=False,passed=False,reason='restart partial attempt discarded')
        live=self.trial.record(progress,survived,complete)
        self.last_evaluation['live']=live
        if live['complete']:
            if live['passed']:
                self.q.load_state_dict(self.trial_model.state_dict())
                self.trusted_modes.update(self.trial_modes);self.model_verified_modes.update(self.trial_modes)
                self.promotion_count+=1;self.promotion_pending=True
                self.last_evaluation.update(promoted=True,reason=live['reason'])
            else:self.last_evaluation.update(promoted=False,reason=live['reason'])
            self.trial_failures=0 if live['passed'] else self.trial_failures+1
            # At most six evaluation attempts after 24/48/96 learning episodes,
            # rather than six evaluation attempts after every single update.
            wait=24*2**min(2,max(0,self.trial_failures-1))
            self.next_trial_episode=self.episode_id+wait
            self.next_trial_grad=self.grad_steps+512
            self.last_evaluation['trusted_modes']=[MODES[i] for i in sorted(self.trusted_modes)]
            self.trial=self.trial_model=None;self.trial_modes=set()
        return live

    def save(self,path,extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
        if self._migration_source and os.path.normcase(os.path.abspath(path))==self._migration_source and os.path.isfile(path):
            backup=path+'.before_cube_targets'
            if not os.path.exists(backup):
                import shutil
                shutil.copy2(path,backup)
            self._migration_source=None
        state=dict(route_controller_version=VERSION,obs_dim=self.obs_dim,arch=list(self.cfg.net_arch),
            motion_targets_version=2,data_migration=self.data_migration,
            evaluation_policy_version=2,trial_schedule=dict(failures=self.trial_failures,
                next_episode=self.next_trial_episode,next_grad=self.next_trial_grad),
            q=self.q.state_dict(),candidate=self.candidate.state_dict(),target=self.target.state_dict(),opt=self.opt.state_dict(),
            steps_seen=self.steps_seen,grad_steps=self.grad_steps,episode_id=self.episode_id,episode_open=self.episode_open,
            replay=self.replay.state(),validation=self.validation.state(),rng=self.rng.bit_generator.state,
            trusted_modes=sorted(self.trusted_modes),model_verified_modes=sorted(self.model_verified_modes),
            promotion_count=self.promotion_count,promotion_pending=self.promotion_pending,evaluation=self.last_evaluation,**(extra or {}))
        if self.evaluating:
            state['live_trial']=dict(model=self.trial_model.state_dict(),modes=list(self.trial_modes),comparison=self.trial.__dict__)
        temporary=path+'.tmp'
        self.th.save(state,temporary);os.replace(temporary,path)

    def load(self,path):
        ck=self.th.load(path,map_location=self.dev,weights_only=False)
        if 'route_controller_version' not in ck:
            if ck.get('obs_dim')!=OBS_DIM:raise ValueError('legacy checkpoint observation layout does not match')
            # Preserve the useful base encoder; new route inputs initially
            # have zero weights. The original checkpoint is never rewritten.
            with self.th.no_grad():
                state=self.q.state_dict()
                for key,value in ck['q'].items():
                    if key.startswith('trunk.') and key in state:
                        if key=='trunk.0.weight':
                            state[key].zero_();state[key][:,:OBS_DIM].copy_(value)
                        elif state[key].shape==value.shape:state[key].copy_(value)
                for i in range(len(MODES)):
                    for suffix in ('weight','bias'):
                        state[f'values.{i}.{suffix}'].copy_(ck['q'][f'v.{suffix}'])
                        state[f'advantages.{i}.{suffix}'].copy_(ck['q'][f'a.{suffix}'])
                self.q.load_state_dict(state)
            self.candidate.load_state_dict(self.q.state_dict());self.target.load_state_dict(self.q.state_dict())
            print('[controller] imported legacy encoder; new route inputs, mode heads and motion model train separately')
            return
        if ck['route_controller_version']!=VERSION or ck['obs_dim']!=self.obs_dim:
            raise ValueError('unsupported route-controller checkpoint version/layout')
        for key,network in (('q',self.q),('candidate',self.candidate),('target',self.target)):
            network.load_state_dict(ck[key])
        self.opt.load_state_dict(ck['opt']);self.steps_seen=ck['steps_seen'];self.grad_steps=ck['grad_steps']
        self.episode_id=ck['episode_id'];self.replay.load(ck['replay']);self.validation.load(ck['validation'])
        if ck.get('episode_open',False):self.episode_id+=1
        self.rng.bit_generator.state=ck['rng'];self.trusted_modes=set(ck['trusted_modes'])
        self.model_verified_modes=set(ck['model_verified_modes']);self.promotion_count=ck['promotion_count']
        self.promotion_pending=ck.get('promotion_pending',False)
        self.last_evaluation=ck.get('evaluation',{})
        self.data_migration=ck.get('data_migration',{})
        if ck.get('motion_targets_version',1)<2:
            discarded=0;retained=0
            for replay in (self.replay,self.validation):
                for key,bucket in list(replay.buckets.items()):
                    clean=[]
                    for row in bucket:
                        target=np.asarray(row['outcomes']);duration=np.asarray(row['durations'])
                        observed=np.diff(np.r_[0.,target[:,0]])*8./np.maximum(duration,.005)
                        difference=np.abs(observed-target[:,1]*20.)
                        conflict=(row['mask']>.5) & (difference>5.) & (difference*duration>.2)
                        if conflict.any():discarded+=1;continue
                        target[:,1]=np.where(row['mask']>.5,observed/20.,0.)
                        row['outcomes']=target;clean.append(row);retained+=1
                    replay.buckets[key]=clean;replay.seen[key]=len(clean)
            self.data_migration=dict(discarded_inconsistent=discarded,retained=retained,
                velocity_targets='measured displacement / elapsed time',pending_trial_cancelled=bool(ck.get('live_trial')))
            # No approved controller is reset. Only an unapproved shadow
            # candidate trained on contradictory targets is restarted.
            if discarded and not self.trusted_modes:
                self.candidate.load_state_dict(self.q.state_dict());self.target.load_state_dict(self.q.state_dict())
                self.opt=self.th.optim.Adam(self.candidate.parameters(),lr=self.cfg.lr)
                self.data_migration['shadow_candidate_restarted']=True
            self.last_evaluation={};self._migration_source=os.path.normcase(os.path.abspath(path))
            print(f'[controller data] measured-velocity migration: kept {retained}, excluded {discarded} inconsistent examples; source is backed up before overwrite')
        schedule=ck.get('trial_schedule',{})
        self.trial_failures=schedule.get('failures',0)
        self.next_trial_episode=schedule.get('next_episode',self.episode_id+24)
        self.next_trial_grad=schedule.get('next_grad',self.grad_steps+512)
        if ck.get('evaluation_policy_version',1)<2:
            self.last_evaluation={}
            print('[controller] old one-action evaluation cancelled; collecting balanced action supervision before live comparison')
        if ck.get('live_trial') and not self._migration_source and ck.get('evaluation_policy_version',1)>=2:
            from controller_evaluation import LiveComparison
            trial=ck['live_trial'];self.trial=LiveComparison();self.trial.__dict__.update(trial['comparison'])
            self.trial_model=copy.deepcopy(self.candidate);self.trial_model.load_state_dict(trial['model'])
            self.trial_modes=set(trial['modes']);self.trial_skip_until_reset=True
