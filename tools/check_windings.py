# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Anthony Charretier
"""Consistency metrics of the winding labels written by peaks_windings.

Usage: python tools/check_windings.py results/json/NAME/scroll_z05626.json [more files...]
       python tools/check_windings.py --pair results/json/NAME/scroll_z07601.json results/json/NAME/scroll_z07602.json

None of these metrics needs a ground truth: on a consistent labelling the per-slice ones are close to 0 and the
agreement of two nearby slices (--pair) close to 100 %.
"""
import collections
import json
import sys
from dataclasses import dataclass


@dataclass
class SliceMetrics:
    path: str
    crossings: int
    off_integer: float  # |u_continuous - round(u)| > 0.25: the solver could not reconcile its relations
    pulled: float  # pull > 3 sigma
    bad_steps: float  # consecutive crossings of a ray on different fragments whose labels differ by other than +-1
    multi_sheet: float  # crossings in a fragment that one ray crosses twice or more (the fragment spans several sheets)

    HEADER = f"{'file':<28}{'crossings':>10}{'off-int':>9}{'pull>3':>9}{'bad-step':>10}{'multi-sheet':>13}"

    def row(self) -> str:
        name = self.path.replace("\\", "/").rsplit("/", 1)[-1]
        return (f"{name:<28}{self.crossings:>10}{self.off_integer:>9.1%}{self.pulled:>9.1%}"
                f"{self.bad_steps:>10.1%}{self.multi_sheet:>13.1%}")


class SliceChecker:
    OFF_INTEGER = 0.25
    MAX_PULL = 3.0

    def __init__(self, path: str):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.lines = json.load(f)["lines"]

    def measure(self) -> SliceMetrics:
        crossings = off_integer = pulled = 0
        good_steps = bad_steps = 0
        fragment_size = collections.Counter()
        multi_fragments = set()
        for line in self.lines:
            points = line["crossings"]
            per_fragment = collections.Counter(p["fragment"] for p in points)
            for fragment, count in per_fragment.items():
                fragment_size[fragment] += count
                if count > 1:
                    multi_fragments.add(fragment)
            for p in points:
                crossings += 1
                off_integer += abs(p["u_continuous"] - round(p["u_continuous"])) > self.OFF_INTEGER
                pulled += p["pull"] > self.MAX_PULL
            for a, b in zip(points, points[1:]):
                if a["fragment"] == b["fragment"]:
                    continue
                if abs(b["winding_label"] - a["winding_label"]) == 1:
                    good_steps += 1
                else:
                    bad_steps += 1
        total = max(crossings, 1)
        return SliceMetrics(
            path=self.path,
            crossings=crossings,
            off_integer=off_integer / total,
            pulled=pulled / total,
            bad_steps=bad_steps / max(good_steps + bad_steps, 1),
            multi_sheet=sum(fragment_size[f] for f in multi_fragments) / total,
        )


class SlicePairChecker:
    """Agreement of the labels of two nearby slices: the points of one sheet barely move from one slice to the next,
    so the points of the second slice within MAX_DISTANCE px of a point of the first should carry the same label, up
    to one shift for the whole slice (what align_labels corrects)."""

    MAX_DISTANCE = 2.0

    def __init__(self, path_a: str, path_b: str):
        self.points_a = self._points(path_a)
        self.points_b = self._points(path_b)

    @staticmethod
    def _points(path):
        with open(path, encoding="utf-8") as f:
            return [(p["x"], p["y"], p["winding_label"]) for line in json.load(f)["lines"] for p in line["crossings"]]

    def measure(self):
        cell = self.MAX_DISTANCE
        grid = collections.defaultdict(list)
        for x, y, label in self.points_b:
            grid[(int(x // cell), int(y // cell))].append((x, y, label))
        shifts = collections.Counter()
        for x, y, label in self.points_a:
            gx, gy = int(x // cell), int(y // cell)
            best, best_d2 = None, cell * cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for bx, by, b_label in grid.get((gx + dx, gy + dy), ()):
                        d2 = (bx - x) ** 2 + (by - y) ** 2
                        if d2 <= best_d2:
                            best, best_d2 = b_label, d2
            if best is not None:
                shifts[best - label] += 1
        matched = sum(shifts.values())
        shift, agreeing = shifts.most_common(1)[0] if shifts else (0, 0)
        return matched, shift, agreeing / max(matched, 1)


def main(args):
    if not args:
        print(__doc__)
        return 1
    if args[0] == "--pair":
        matched, shift, agreement = SlicePairChecker(args[1], args[2]).measure()
        print(f"{matched} points matched within {SlicePairChecker.MAX_DISTANCE} px; shift {shift:+d}; "
              f"same label (after the shift): {agreement:.1%}")
        return 0
    print(SliceMetrics.HEADER)
    for path in args:
        print(SliceChecker(path).measure().row())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
