"""Every object type the YOLO detector can be taught, and what each one does.

One table, used by the labeler (label_boxes.py), the trainer (detect_yolo.py), the
detector (YoloOnlyDetector) and the physics (gd_physics.special_effect / portals).

Each class has a BASE class -- one of the 7 the brain has always seen (0 hazard,
1-3 orbs, 4 pad, 5 portal, 6 solid) or the icon (7) -- so the brain's inputs keep the
same shape and an existing brain keeps working. The exact type (red orb, 2x speed
portal...) rides along as `kind` and drives the blue line's physics.

Launch strengths are GD 2.2's, relative to the cube's own jump (11.18). The less
common ones (black / dash / spider orbs, mode scaling for ship / ufo / ball / swing)
are approximate and get checked against recordings.
"""
from __future__ import annotations

# (name, base class, label group, what it does)
#   orb / pad effects: ("jump", mult)  launch away from the floor, mult x cube jump
#                      ("down", mult)  slam toward the floor
#                      ("flip", push)  flip gravity, small push toward the new floor
#                      ("flipjump", mult)  flip gravity and launch in the new direction
#                      ("dash", flip)  dash straight ahead while held (flip: gravity first)
#                      ("teleport_flip",)  jump to the opposite surface, flip gravity
#                      ("none",)       no movement (toggle orb) / unknown target (teleport)
#   portals:           ("mode", m) ("g", +1 | -1 | 0=toggle) ("size", mini) ("speed", blocks/s)
#                      ("none",)  mirror / dual / teleport: not modelled
OBJECTS = [
    # --- the 7 classes the brain has always had (kept: old labels stay valid) -------
    ("hazard", 0, "base", None),
    ("orb_y", 1, "orb", ("jump", 1.0)),        # yellow orb
    ("orb_b", 2, "orb", ("flip", 0.4)),        # blue orb
    ("orb_p", 3, "orb", ("jump", 0.72)),       # pink orb
    ("pad", 4, "pad", None),                   # pad, type from its colour (old labels)
    ("portal", 5, "portal", None),             # portal, type from its colour (old labels)
    ("solid", 6, "base", None),
    # --- the icon, by game mode -------------------------------------------------------
    ("cube", 7, "icon", "cube"),
    ("ship", 7, "icon", "ship"),
    ("ball", 7, "icon", "ball"),
    ("ufo", 7, "icon", "ufo"),
    ("wave", 7, "icon", "wave"),
    ("robot", 7, "icon", "robot"),
    ("spider", 7, "icon", "spider"),
    ("swing", 7, "icon", "swing"),
    # --- every orb ---------------------------------------------------------------------
    ("orb_red", 1, "orb", ("jump", 1.38)),
    ("orb_green", 2, "orb", ("flipjump", 1.0)),
    ("orb_black", 3, "orb", ("down", 1.4)),
    ("orb_dash_green", 3, "orb", ("dash", False)),
    ("orb_dash_pink", 2, "orb", ("dash", True)),
    ("orb_spider", 2, "orb", ("teleport_flip",)),
    ("orb_teleport", 3, "orb", ("none",)),
    ("orb_toggle", 3, "orb", ("none",)),
    # --- every pad -----------------------------------------------------------------------
    ("pad_yellow", 4, "pad", ("jump", 1.43)),
    ("pad_pink", 4, "pad", ("jump", 0.93)),
    ("pad_red", 4, "pad", ("jump", 1.8)),
    ("pad_blue", 4, "pad", ("flip", 0.4)),
    ("pad_spider", 4, "pad", ("teleport_flip",)),
    # --- mode portals --------------------------------------------------------------------
    ("portal_cube", 5, "mode_portal", ("mode", "cube")),
    ("portal_ship", 5, "mode_portal", ("mode", "ship")),
    ("portal_ball", 5, "mode_portal", ("mode", "ball")),
    ("portal_ufo", 5, "mode_portal", ("mode", "ufo")),
    ("portal_wave", 5, "mode_portal", ("mode", "wave")),
    ("portal_robot", 5, "mode_portal", ("mode", "robot")),
    ("portal_spider", 5, "mode_portal", ("mode", "spider")),
    ("portal_swing", 5, "mode_portal", ("mode", "swing")),
    # --- gravity / size / mirror / dual / teleport portals --------------------------------
    ("portal_grav_flip", 5, "other_portal", ("g", -1)),      # yellow: upside down
    ("portal_grav_normal", 5, "other_portal", ("g", 1)),     # blue: right way up
    ("portal_grav_toggle", 5, "other_portal", ("g", 0)),     # green: switch
    ("portal_mini", 5, "other_portal", ("size", True)),
    ("portal_big", 5, "other_portal", ("size", False)),
    ("portal_mirror", 5, "other_portal", ("none",)),
    ("portal_mirror_off", 5, "other_portal", ("none",)),
    ("portal_dual", 5, "other_portal", ("none",)),
    ("portal_dual_off", 5, "other_portal", ("none",)),
    ("portal_teleport", 5, "other_portal", ("none",)),
    # --- speed portals (GD's speeds in blocks/s) ------------------------------------------
    ("speed_0.5x", 5, "speed", ("speed", 8.37)),
    ("speed_1x", 5, "speed", ("speed", 10.39)),
    ("speed_2x", 5, "speed", ("speed", 12.91)),
    ("speed_3x", 5, "speed", ("speed", 15.6)),
    ("speed_4x", 5, "speed", ("speed", 19.2)),
]

NAMES = [o[0] for o in OBJECTS]
BASE = [o[1] for o in OBJECTS]
GROUP = [o[2] for o in OBJECTS]
EFFECT = {o[0]: o[3] for o in OBJECTS}
ICON_CLASSES = {i: o[3] for i, o in enumerate(OBJECTS) if o[2] == "icon"}
GROUPS = {}
for _i, _o in enumerate(OBJECTS):
    GROUPS.setdefault(_o[2], []).append(_i)

# how strongly an orb / pad launch acts in each mode (cube = 1). GD's values differ per
# mode; these are approximate until checked against recordings.
MODE_SCALE = {"cube": 1.0, "robot": 1.0, "spider": 1.0, "ball": 0.7, "ufo": 0.7,
              "ship": 0.55, "swing": 0.6, "wave": 0.0}


def orb_pad_effect(name: str):
    """("orb" | "pad", effect tuple) for a class name, None if it isn't an orb / pad."""
    i = NAMES.index(name) if name in NAMES else -1
    if i < 0 or GROUP[i] not in ("orb", "pad") or EFFECT[name] is None:
        return None
    return ("pad" if GROUP[i] == "pad" else "orb", EFFECT[name])


def portal_fx(name: str):
    """A portal class's effect tuple, None for the generic 'portal' (colour decides)."""
    i = NAMES.index(name) if name in NAMES else -1
    if i < 0 or GROUP[i] not in ("mode_portal", "other_portal", "speed"):
        return None
    return EFFECT[name]


def pad_kind_from_hue(hue: float):
    """Generic pad (old labels / built-in detector): its type from its colour."""
    if hue is None or hue < 0:
        return None
    h = hue * 180.0
    if 12 <= h <= 42:
        return "pad_yellow"
    if 135 <= h <= 172:
        return "pad_pink"
    if h >= 172 or h <= 6:
        return "pad_red"
    if 85 <= h <= 120:
        return "pad_blue"
    if 120 < h < 135:
        return "pad_spider"
    return None
