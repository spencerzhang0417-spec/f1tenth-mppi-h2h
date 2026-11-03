
#!/usr/bin/env python3
"""Regenerate raceline speed with automatic timestamped backup."""

import io
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================
# Configuration
# ============================================================

CSV_NAME = "race2_raceline_remapped.csv"

V_MIN = 1.0
V_MAX = 4.0
MAX_LAT_ACCEL = 4.3
MAX_ACCEL = 2.5
MAX_DECEL = 3.0

CURVATURE_WINDOW = 5
SPEED_WINDOW = 2
CURVATURE_LOOKAHEAD = 0.3
SPEED_SCALE = 1.0


# ============================================================
# File path
# ============================================================

def find_csv():
    here = Path(__file__).resolve().parent

    for root in (here, here.parent, here.parent / "mppi"):
        csv = root / "waypoints" / CSV_NAME
        if csv.is_file():
            return csv

    raise FileNotFoundError(
        f"Cannot find waypoints/{CSV_NAME}"
    )


# ============================================================
# Speed calculation (closed-loop raceline)
# ============================================================

def smooth(a, half_window):
    if not half_window:
        return a.copy()

    padded = np.pad(a, half_window, mode="wrap")
    kernel = np.ones(2 * half_window + 1) / (
        2 * half_window + 1
    )

    return np.convolve(padded, kernel, mode="valid")


def enforce_accel(v, ds):
    """Apply acceleration and braking limits around the lap."""
    s = v * v
    n = len(s)

    for _ in range(n + 1):
        changed = False

        # Forward: acceleration
        for i in range(n):
            j = (i + 1) % n
            upper = s[i] + 2 * MAX_ACCEL * ds[i]

            if s[j] > upper:
                s[j] = upper
                changed = True

        # Backward: braking
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            upper = s[j] + 2 * MAX_DECEL * ds[i]

            if s[i] > upper:
                s[i] = upper
                changed = True

        if not changed:
            return np.sqrt(s)

    raise RuntimeError("Speed solver did not converge")


def generate_speeds(w):
    xy = w[:, :2]

    # Handle CSVs that repeat the first point at the end.
    duplicate_end = (
        np.linalg.norm(xy[0] - xy[-1]) < 1e-7
    )

    if duplicate_end:
        xy = xy[:-1]

    if len(xy) < 3:
        raise ValueError("Need at least 3 distinct points")

    prev = np.roll(xy, 1, axis=0)
    nxt = np.roll(xy, -1, axis=0)

    ds = np.linalg.norm(nxt - xy, axis=1)

    if np.any(ds <= 1e-9):
        raise ValueError("Duplicate consecutive points")

    # 1. Curvature
    a = xy - prev
    b = nxt - xy
    c = nxt - prev

    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )

    kappa = np.zeros(len(xy))
    np.divide(
        2 * cross,
        denom,
        out=kappa,
        where=denom > 1e-12,
    )

    # Smooth magnitude so opposite turns do not cancel.
    kappa = smooth(np.abs(kappa), CURVATURE_WINDOW)

    # 2. Look ahead by actual distance
    if CURVATURE_LOOKAHEAD > 0:
        ahead = kappa.copy()

        for i in range(len(xy)):
            j = i
            distance = 0.0

            while distance < CURVATURE_LOOKAHEAD:
                distance += ds[j]
                j = (j + 1) % len(xy)

                ahead[i] = max(ahead[i], kappa[j])

                if j == i:
                    break

        kappa = ahead

    # 3. Lateral acceleration limit
    cap = np.minimum(
        V_MAX,
        np.sqrt(
            MAX_LAT_ACCEL / np.maximum(kappa, 1e-12)
        ),
    )

    # 4. Longitudinal limits
    v = enforce_accel(cap, ds)

    # 5. Speed smoothing
    v = np.minimum(v, smooth(v, SPEED_WINDOW))

    # 6. Global scaling
    # V_MIN cannot override the cornering speed limit.
    v = np.minimum(
        cap,
        np.maximum(
            np.minimum(V_MIN, cap),
            v * SPEED_SCALE,
        ),
    )

    # Reapply acceleration/braking limits after smoothing.
    v = enforce_accel(v, ds)

    if duplicate_end:
        v = np.append(v, v[0])

    return v, ds.sum()


# ============================================================
# Main: generate -> backup -> replace
# ============================================================

def main():
    csv = find_csv()

    # Read original CSV while preserving newline formatting.
    with csv.open("r", encoding="utf-8-sig", newline="") as f:
        original = f.read()

    w = np.loadtxt(
        io.StringIO(original),
        delimiter=",",
        comments="#",
        ndmin=2,
    )

    if (
        w.shape[0] < 3
        or w.shape[1] != 4
        or not np.isfinite(w).all()
    ):
        raise ValueError(
            "CSV must contain finite x,y,yaw,v values"
        )

    if not (
        0 < V_MIN <= V_MAX
        and MAX_LAT_ACCEL > 0
        and MAX_ACCEL > 0
        and MAX_DECEL > 0
        and SPEED_SCALE > 0
        and CURVATURE_LOOKAHEAD >= 0
        and CURVATURE_WINDOW >= 0
        and SPEED_WINDOW >= 0
    ):
        raise ValueError("Invalid speed parameters")

    # Calculate before touching the original file.
    v, length = generate_speeds(w)

    # Change only column 4; preserve original x/y/yaw text.
    lines = []
    row = 0

    for line in original.splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue

        body = line.rstrip("\r\n")
        fields = body.split(",")

        if len(fields) != 4 or row >= len(v):
            raise ValueError("Invalid CSV row")

        fields[3] = f"{v[row]:.10f}"
        lines.append(",".join(fields) + line[len(body):])
        row += 1

    if row != len(v):
        raise ValueError("CSV row count mismatch")

    output = "".join(lines).encode("utf-8")

    # Create completed temporary output first.
    temp = None

    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=csv.parent,
            prefix=".raceline_",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp = Path(f.name)
            f.write(output)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(temp, csv.stat().st_mode)

        # Backup the ORIGINAL CSV before replacement.
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        backup = csv.with_name(
            f"{csv.stem}_backup_{stamp}{csv.suffix}"
        )

        with csv.open("rb") as src, backup.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())

        # Atomic replacement: ROS2 CSV filename stays unchanged.
        os.replace(temp, csv)

    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)

    print(f"Updated: {csv}")
    print(f"Backup:  {backup.name}")
    print(
        f"Points: {len(w)}, "
        f"length: {length:.2f} m, "
        f"speed: {v.min():.2f}..{v.max():.2f} m/s"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
