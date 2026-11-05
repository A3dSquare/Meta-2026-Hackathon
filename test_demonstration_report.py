import copy
import unittest
from demonstration_report import annotate_controller_thresholds


class ReportTests(unittest.TestCase):
    def test_annotations_preserve_values_and_are_adjacent_and_repeatable(self):
        metrics=dict(accuracy=.68,action_recall=[.69,.75],unsafe_rate=.024,
                     position_rmse=.307,velocity_rmse=3.,td_error=1.)
        report={'controller_after':{'trusted_modes':[], 'modes':{'cube':{
            'candidate':copy.deepcopy(metrics),'incumbent':copy.deepcopy(metrics),'passed':False}}}}
        annotate_controller_thresholds(report)
        candidate=report['controller_after']['modes']['cube']['candidate']
        self.assertEqual({k:candidate[k] for k in metrics},metrics)
        keys=list(candidate)
        self.assertEqual(keys[keys.index('accuracy')+1],'accuracy_comment')
        self.assertIn('68.0%',candidate['accuracy_comment'])
        self.assertIn('75%',candidate['accuracy_comment'])
        self.assertIn('85%',candidate['action_recall_comment'])
        self.assertIn('0.300',candidate['position_rmse_comment'])
        snapshot=copy.deepcopy(report)
        self.assertEqual(annotate_controller_thresholds(report),snapshot)
        report['controller_after']['trusted_modes']=['cube']
        annotate_controller_thresholds(report)
        candidate=report['controller_after']['modes']['cube']['candidate']
        self.assertIn('67.0%',candidate['accuracy_comment'])
        self.assertIn('2.9%',candidate['unsafe_rate_comment'])


if __name__=='__main__':unittest.main()
