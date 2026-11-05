"""Recording layout and level-wide, whole-clip validation splits."""
import json
import re
import time
import uuid
from pathlib import Path


def level_slug(name):
    slug = re.sub(r'[^\w]+', '-', name.strip().casefold(), flags=re.UNICODE).strip('-_')
    if not slug:
        raise ValueError('Level name must contain letters or numbers')
    # Prefix Windows device names so every slug is a usable directory name.
    if slug in {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}:
        slug = 'level-' + slug
    return slug


def create_session(root, level):
    session_id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    folder = Path(root) / level_slug(level) / session_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'frames').mkdir()
    return folder


def discover_recordings(paths):
    """Accept legacy sessions, level folders, or the entire recording library."""
    found = {}
    for value in paths:
        path = Path(value)
        files = [path / 'meta.json'] if (path / 'meta.json').is_file() else sorted(path.rglob('meta.json'))
        if not files:
            raise ValueError(f'{path}: no recording meta.json found')
        for meta in files:
            found[meta.parent.resolve()] = meta.parent
    return sorted(found.values(), key=lambda p: str(p.resolve()))


def grouped_clips(sessions):
    """Assign whole clips per level; persist roles separately from recorder metadata.

    Existing full legacy sessions retain their original every-fifth split.
    Persisted roles cannot change when more sessions are added or renamed.
    """
    groups = {}
    seen = set()
    for folder, meta, frames, timeline in sessions:
        if meta['id'] in seen:
            raise ValueError(f'Duplicate recording ID: {meta["id"]}')
        seen.add(meta['id'])
        key = level_slug(meta.get('level') or meta['id'])
        groups.setdefault(key, []).append((folder, meta, frames, timeline))
    result = []
    for level, group in sorted(groups.items()):
        group.sort(key=lambda s: s[1]['id'])
        total = sum(c['result'] == 'success' for _, m, _, _ in group for c in m['clips'])
        if total < 5:
            raise ValueError(f'{level}: record at least five successful clips across this level (found {total})')
        offset = 0
        for folder, meta, frames, timeline in group:
            successes = [c for c in meta['clips'] if c['result'] == 'success']
            path = Path(folder) / 'training_split.json'
            splits = json.loads(path.read_text()) if path.exists() else {}
            for i, clip in enumerate(successes):
                heldout = splits.setdefault(clip['id'], (i if len(successes) >= 5 else offset + i) % 5 == 4)
                if type(heldout) is not bool:
                    raise ValueError(f'{path}: invalid split for {clip["id"]}')
                result.append((folder, meta, clip, frames, timeline, heldout))
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(splits, indent=2) + '\n')
            temporary.replace(path)
            offset += len(successes)
    return result
