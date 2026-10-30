"""Frozen incumbent/candidate gameplay comparison, separate from training data."""
import statistics
import numpy as np


class LiveComparison:
    def __init__(self,attempts=6):
        if attempts<6 or attempts%2:raise ValueError('evaluation needs an even number of at least six attempts')
        self.attempts=attempts;self.records=[];self.reference=None;self.prints=[]
        self.steps=self.interventions=0;self.level_changed=False

    @property
    def arm(self):return 'incumbent' if len(self.records)%2==0 else 'candidate'

    def observe(self,frame,seconds):
        if frame is not None and len(self.prints)<3 and seconds >= (.15,.3,.45)[len(self.prints)]:
            from route import level_print
            self.prints.append(level_print(frame))

    def note_action(self,intervened):
        self.steps+=1;self.interventions+=int(intervened)

    def record(self,progress,survived,complete=False):
        if self.reference is None and self.prints:self.reference=list(self.prints)
        elif self.reference is not None and self.prints:
            from route import print_distance
            if statistics.median(print_distance(a,b) for a,b in zip(self.reference,self.prints))>8.:
                self.level_changed=True
        self.records.append(dict(arm=self.arm,progress=float(progress),survived=int(survived),
            completed=bool(complete),shield_rate=self.interventions/max(1,self.steps),identified=bool(self.prints)))
        self.prints=[];self.steps=self.interventions=0
        return self.summary()

    def summary(self):
        done=len(self.records)>=self.attempts
        report=dict(complete=done,passed=False,attempts=len(self.records),records=list(self.records))
        if not done:return dict(report,reason='collecting frozen alternating evaluation attempts')
        for arm in ('incumbent','candidate'):
            rows=[r for r in self.records if r['arm']==arm]
            values=[r['progress'] for r in rows]
            report[arm]=dict(median=statistics.median(values),lower_quartile=float(np.quantile(values,.25)),
                completion_count=sum(r['completed'] for r in rows),shield_rate=statistics.mean(r['shield_rate'] for r in rows))
        old,new=report['incumbent'],report['candidate']
        comparable=not self.level_changed and all(r['identified'] for r in self.records)
        no_regression=(new['median']>=old['median'] and new['lower_quartile']>=old['lower_quartile']
            and new['completion_count']>=old['completion_count'] and new['shield_rate']<=old['shield_rate']+.02)
        better=(new['median']>=old['median']+.01 or new['completion_count']>old['completion_count']
                or new['shield_rate']<old['shield_rate']-.05)
        report['passed']=bool(comparable and no_regression and better)
        report['reason']=('passed live consistency and intervention comparison' if report['passed'] else
            'inconclusive: level identity changed or unavailable' if not comparable else 'live candidate did not improve without regression')
        return report
