#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate an empty-load model and compare it with its training fit."""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from t1_payload_id.common import (
    T1RegressorBuilder,
    metrics,
    resolve_project_path,
    save_json,
)


ASSESSMENT_THRESHOLDS = {
    "overall_rmse_nm_max": 0.25,
    "validation_to_training_rmse_ratio_max": 1.25,
    "weak_joint_r2_min": 0.80,
    "weak_joint_nrmse_std_max": 0.50,
}


def compare_metrics(training, validation):
    ratio = float(
        validation["overall_rmse"] / max(training["overall_rmse"], 1e-12)
    )
    weak_joints = [
        item["joint"]
        for item in validation["per_joint"]
        if (
            not np.isfinite(item["r2"])
            or item["r2"] < ASSESSMENT_THRESHOLDS["weak_joint_r2_min"]
            or item["nrmse_std"]
            > ASSESSMENT_THRESHOLDS["weak_joint_nrmse_std_max"]
        )
    ]
    overall_pass = (
        validation["overall_rmse"]
        <= ASSESSMENT_THRESHOLDS["overall_rmse_nm_max"]
        and ratio
        <= ASSESSMENT_THRESHOLDS["validation_to_training_rmse_ratio_max"]
    )
    if overall_pass and not weak_joints:
        status = "good"
    elif overall_pass:
        status = "acceptable_with_caveats"
    else:
        status = "poor"
    return {
        "status": status,
        "overall_rmse_ratio_validation_to_training": ratio,
        "weak_joints": weak_joints,
        "thresholds": ASSESSMENT_THRESHOLDS,
    }


def same_reference_trajectory(model, validation) -> bool:
    return all(
        key in model.files
        and key in validation.files
        and model[key].shape == validation[key].shape
        and np.allclose(model[key], validation[key], rtol=0.0, atol=1e-12)
        for key in ("Q", "V", "A")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/empty_validation")
    args = parser.parse_args()

    model_path = resolve_project_path(args.model)
    model = np.load(model_path, allow_pickle=True)
    config = json.loads(str(model["config_json"]))
    side = str(model["side"])
    builder = T1RegressorBuilder(config, side=side)
    alpha = model["alpha"]
    beta = model["beta0"]
    keep = model["keep_mask"].astype(bool)
    indices = model["base_idx_in_keep"].astype(int)
    training_metrics = metrics(
        model["TAU"].reshape(-1),
        model["tau_hat"].reshape(-1),
        7,
    )

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for value in args.val:
        path = resolve_project_path(value)
        validation = np.load(path, allow_pickle=True)
        regressor = builder.stack_full_regressor(
            validation["Q"], validation["V"], validation["A"], alpha
        )
        selected = regressor[:, keep][:, indices]
        tau_hat = (selected @ beta).reshape(-1, 7)
        tau = validation["TAU"]
        validation_metrics = metrics(tau.reshape(-1), tau_hat.reshape(-1), 7)
        comparison = compare_metrics(training_metrics, validation_metrics)
        same_trajectory = same_reference_trajectory(model, validation)
        scope = (
            "same_trajectory_repeatability"
            if same_trajectory
            else "different_trajectory_generalization"
        )
        summaries.append({
            "file": str(path),
            "metrics": validation_metrics,
            "comparison_to_training": comparison,
            "validation_scope": scope,
            "same_reference_trajectory_as_training": same_trajectory,
        })

        t = (
            validation["t"]
            if "t" in validation.files
            else np.arange(tau.shape[0])
        )
        for joint in range(7):
            figure = plt.figure(figsize=(10, 3))
            plt.plot(t, tau[:, joint], label="measured tau_est")
            plt.plot(t, tau_hat[:, joint], label="empty model")
            plt.xlabel("time [s]")
            plt.ylabel(f"joint {joint + 1} torque [Nm]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            figure.savefig(
                out_dir / f"{path.stem}_joint{joint + 1}.png",
                dpi=150,
            )
            plt.close(figure)

        print(
            f"[VALIDATION] {path.name}: status={comparison['status']}, "
            f"RMSE={validation_metrics['overall_rmse']:.6f} Nm, "
            f"ratio={comparison['overall_rmse_ratio_validation_to_training']:.3f}, "
            f"scope={scope}"
        )

    summary = {
        "model": str(model_path),
        "training_metrics": training_metrics,
        "results": summaries,
    }
    save_json(summary, out_dir / "summary.json")
    print(f"[OK] saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
