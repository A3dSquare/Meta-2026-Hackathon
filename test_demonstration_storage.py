import json
from pathlib import Path
import tempfile
import unittest

from demonstration_storage import create_session, discover_recordings, grouped_clips, level_slug


class StorageTests(unittest.TestCase):
    def session(self, root, name, level, count):
        folder = root / name
        folder.mkdir()
        meta = dict(id=name, level=level, clips=[dict(id=str(i), result='success') for i in range(count)])
        (folder / 'meta.json').write_text(json.dumps(meta))
        return folder, meta, [], None

    def test_level_folder_reused_without_overwriting_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = create_session(tmp, 'Base After Base')
            marker = a / 'frames' / 'keep.jpg'
            marker.write_bytes(b'original')
            b = create_session(tmp, 'base-after-base')
            self.assertEqual(a.parent, b.parent)
            self.assertNotEqual(a, b)
            self.assertEqual(marker.read_bytes(), b'original')
            self.assertEqual(level_slug('CON'), 'level-con')
            self.assertNotIn('/', level_slug('../Dry Out'))

    def test_seven_plus_three_and_stable_split_on_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.session(root, 'a', 'Base After Base', 7)
            b = self.session(root, 'b', 'base-after-base', 3)
            first = grouped_clips([b, a])
            roles = {(r[1]['id'], r[2]['id']): r[-1] for r in first}
            self.assertEqual(sum(roles.values()), 2)
            self.assertTrue(roles['a', '4'])
            self.assertTrue(roles['b', '2'])
            c = self.session(root, 'c', 'base-after-base', 2)
            for r in grouped_clips([c, b, a]):
                key = r[1]['id'], r[2]['id']
                if key in roles:
                    self.assertEqual(r[-1], roles[key])
            self.assertEqual(len(discover_recordings([root, a[0]])), 3)

    def test_short_sessions_combine_but_different_levels_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.session(root, 'a', 'jumper', 3)
            b = self.session(root, 'b', 'jumper', 2)
            self.assertEqual(sum(r[-1] for r in grouped_clips([a, b])), 1)
            b[1]['level'] = 'dry-out'
            with self.assertRaisesRegex(ValueError, 'at least five'):
                grouped_clips([a, b])
            with self.assertRaisesRegex(ValueError, 'Duplicate recording ID'):
                grouped_clips([a, a])


if __name__ == '__main__':
    unittest.main()
