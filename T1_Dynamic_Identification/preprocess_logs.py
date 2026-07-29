#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess T1 excitation CSV logs into NPZ arrays for identification.

The regressor uses commanded q_ref/qd_ref/qdd_ref, matching the paper's choice.
Measured q/dq are saved only for tracking diagnostics. Torque comes from tau_est.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np

from t1_payload_id.common import load_json, resolve_project_path, save_json, lowpass_filter


def load_csv_log(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    def block(prefix):
        return np.column_stack([data[f"{prefix}_{i}"] for i in range(7)])
    return {
        "wall_time": data["wall_time"],
        "t": data["traj_time"],
        "Q": block("q_ref"),
        "V": block("qd_ref"),
        "A": block("qdd_ref"),
        "Q_MEAS": block("q_meas"),
        "V_MEAS": block("qd_meas"),
        "A_MEAS": block("ddq_meas"),
        "TAU": block("tau_est"),
    }


def merge_logs(paths: List[Path], mode: str):
    items = [load_csv_log(p) for p in paths]
    if len(items) == 1 or mode == "concat":
        out = {}
        for k in items[0].keys():
            out[k] = np.concatenate([it[k] for it in items], axis=0)
        return out
    # average repeated runs sample by sample after truncation
    n = min(len(it["t"]) for it in items)
    out = {"t": items[0]["t"][:n]}
    for k in ["Q", "V", "A", "Q_MEAS", "V_MEAS", "A_MEAS", "TAU"]:
        out[k] = np.mean(np.stack([it[k][:n] for it in items], axis=0), axis=0)
    out["wall_time"] = items[0]["wall_time"][:n]
    return out


def collect_publish_timing(paths: List[Path]):
    per_file = []
    total_missed = 0
    max_lateness = 0.0
    max_clip_error = 0.0
    has_timing_columns = False
    for path in paths:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        names = set(data.dtype.names or ())
        item = {"file": str(path), "timing_columns_present": False}
        if {"publish_lateness_s", "command_published"} <= names:
            has_timing_columns = True
            published = np.atleast_1d(data["command_published"]).astype(float)
            lateness = np.atleast_1d(data["publish_lateness_s"]).astype(float)
            missed = int(np.count_nonzero(~np.isfinite(published) | (published < 0.5)))
            finite_lateness = lateness[np.isfinite(lateness)]
            file_max = float(np.max(finite_lateness)) if len(finite_lateness) else float("nan")
            total_missed += missed
            if np.isfinite(file_max):
                max_lateness = max(max_lateness, file_max)
            item.update({
                "timing_columns_present": True,
                "missed_deadlines": missed,
                "max_publish_lateness_s": file_max,
            })
            if "command_clip_error_rad" in names:
                clip_error = np.atleast_1d(data["command_clip_error_rad"]).astype(float)
                finite_clip = clip_error[np.isfinite(clip_error)]
                file_clip_max = (
                    float(np.max(finite_clip)) if len(finite_clip) else float("nan")
                )
                if np.isfinite(file_clip_max):
                    max_clip_error = max(max_clip_error, file_clip_max)
                item["max_command_clip_error_rad"] = file_clip_max
        per_file.append(item)
    return {
        "timing_columns_present": has_timing_columns,
        "missed_deadlines": total_missed if has_timing_columns else None,
        "max_publish_lateness_s": max_lateness if has_timing_columns else None,
        "max_command_clip_error_rad": (
            max_clip_error if has_timing_columns else None
        ),
        "per_file": per_file,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/t1_payload_config.json")
    ap.add_argument("--input", nargs="+", required=True, help="CSV logs")
    ap.add_argument("--out", required=True, help="processed NPZ path")
    ap.add_argument("--side", choices=["left", "right"], default=None)
    ap.add_argument("--repeat-mode", choices=["average", "concat"], default="average")
    ap.add_argument("--no-filter-torque", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    paths = [resolve_project_path(x) for x in args.input]
    publish_timing = collect_publish_timing(paths)
    D = merge_logs(paths, args.repeat_mode)
    dt = float(np.median(np.diff(D["t"]))) if len(D["t"]) > 1 else cfg["robot"].get("control_dt", 0.002)
    fs = 1.0 / dt
    tau_raw = D["TAU"]
    if args.no_filter_torque:
        tau = tau_raw.copy()
        filter_meta = {"enabled": False}
    else:
        cutoff = float(cfg["identification"].get("torque_filter_cutoff_hz", 12.0))
        order = int(cfg["identification"].get("torque_filter_order", 4))
        tau = lowpass_filter(tau_raw, fs, cutoff, order)
        filter_meta = {"enabled": True, "cutoff_hz": cutoff, "order": order}
    out = resolve_project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        t=D["t"], dt=np.array(dt), fs_hz=np.array(fs),
        Q=D["Q"], V=D["V"], A=D["A"], TAU=tau, TAU_RAW=tau_raw,
        Q_MEAS=D["Q_MEAS"], V_MEAS=D["V_MEAS"], A_MEAS=D["A_MEAS"],
        side=np.array(args.side or cfg["identification"].get("side", "left")),
        source_files=np.array([str(p) for p in paths], dtype=object), repeat_mode=np.array(args.repeat_mode))
    track_rmse = np.sqrt(np.mean((D["Q_MEAS"] - D["Q"]) ** 2, axis=0))
    summary = {
        "out": str(out),
        "source_files": [str(p) for p in paths],
        "n_samples": int(len(D["t"])),
        "dt": dt,
        "fs_hz": fs,
        "publish_timing": publish_timing,
        "torque_filter": filter_meta,
        "tracking_rmse_rad_per_joint": track_rmse.tolist(),
        "tau_std_per_joint": np.std(tau, axis=0).tolist()
    }
    save_json(summary, out.with_suffix(".json"))
    if publish_timing["timing_columns_present"] and publish_timing["missed_deadlines"]:
        print(
            f"[WARN] {publish_timing['missed_deadlines']} command deadlines were missed; "
            "do not use this run for formal identification"
        )
    if (
        publish_timing["timing_columns_present"]
        and publish_timing["max_command_clip_error_rad"] is not None
        and publish_timing["max_command_clip_error_rad"] > 1e-4
    ):
        print(
            "[WARN] excitation commands were rate-limited; "
            "do not use this run for formal identification"
        )
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
