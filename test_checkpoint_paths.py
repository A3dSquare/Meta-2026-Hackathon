import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class CheckpointPathTests(unittest.TestCase):
    def test_import_missing_physics_fails_before_reading_recordings_or_writing(self):
        import train_demonstrations
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'new'
            argv=['train_demonstrations.py','missing-recording','--motion-model',str(Path(root)/'missing.npz'),
                  '--out',str(out)]
            error=io.StringIO()
            with patch.object(sys,'argv',argv),contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    train_demonstrations.main()
            self.assertEqual(raised.exception.code,2)
            self.assertIn('Physics calibration not found',error.getvalue())
            self.assertFalse(out.exists())

    def test_live_missing_physics_fails_before_capture(self):
        from train_boxes import build
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'Physics calibration not found'):
                build(SimpleNamespace(motion_model=str(Path(root)/'missing.npz')))

    def test_controller_saves_to_loaded_file_and_best_sibling(self):
        from trainer import Trainer,TrainConfig,checkpoint_path
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            loaded=root/'manual_v3'/'custom_brain.pt'
            loaded.parent.mkdir();loaded.write_bytes(b'initial')
            unrelated=root/'route_controller.pt';unrelated.write_bytes(b'keep')
            def save(path,metadata):
                Path(path).write_bytes(b'updated')
            learner=SimpleNamespace(save=save,promotion_pending=True)
            config=TrainConfig(route_controller=True,route=False,save_dir=str(root),log_dir=str(root/'logs'))
            trainer=Trainer(SimpleNamespace(),config,learner=learner,model_path=str(loaded))
            try:
                trainer.save();trainer.save(best=True)
                self.assertEqual(loaded.read_bytes(),b'updated')
                self.assertEqual(loaded.with_name('custom_brain_best.pt').read_bytes(),b'updated')
                self.assertEqual(unrelated.read_bytes(),b'keep')
                self.assertFalse(learner.promotion_pending)
                self.assertEqual(checkpoint_path(trainer.cfg),str(loaded.resolve()))
                self.assertIsNone(config.checkpoint_file)
            finally:
                trainer._csv.close()

    def test_fresh_and_legacy_paths_unchanged(self):
        from trainer import TrainConfig,checkpoint_path
        cfg=TrainConfig(name='route_controller',route_controller=True)
        self.assertEqual(Path(checkpoint_path(cfg)),Path('checkpoints/route_controller.pt'))
        cfg=TrainConfig(name='preview_boxes')
        self.assertEqual(Path(checkpoint_path(cfg,best=True)),Path('checkpoints/best_boxes.pt'))

    def test_importing_legacy_encoder_does_not_overwrite_legacy_brain(self):
        from trainer import Trainer,TrainConfig
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);legacy=root/'best_boxes.pt';legacy.write_bytes(b'legacy')
            learner=SimpleNamespace(imported_legacy=True,save=lambda p,m:Path(p).write_bytes(b'new controller'))
            cfg=TrainConfig(route_controller=True,route=False,name='route_controller',save_dir=str(root),log_dir=str(root/'logs'))
            trainer=Trainer(SimpleNamespace(),cfg,learner=learner,model_path=str(legacy))
            try:
                trainer.save()
                self.assertEqual(legacy.read_bytes(),b'legacy')
                self.assertEqual((root/'route_controller.pt').read_bytes(),b'new controller')
            finally:
                trainer._csv.close()


if __name__=='__main__':unittest.main()
