"""Defect severity rules used by the station, the log and the advisor."""

# critical: the board cannot work; major: likely to fail assembly or test; minor: cosmetic or reliability risk
SEVERITY = {
    "short": "critical",
    "open_circuit": "critical",
    "missing_hole": "major",
    "spurious_copper": "major",
    "spur": "minor",
    "mouse_bite": "minor",
}
RANK = {"none": 0, "minor": 1, "major": 2, "critical": 3}


def board_severity(defects):
    """defects: list of dicts with a 'cls' key. Returns the worst severity on the board."""
    worst = "none"
    for d in defects:
        s = SEVERITY.get(d["cls"], "minor")
        if RANK[s] > RANK[worst]:
            worst = s
    return worst
