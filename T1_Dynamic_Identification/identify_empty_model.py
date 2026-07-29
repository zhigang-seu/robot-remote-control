#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identify the empty-load dynamic prior used by online payload estimation.

Run this on empty-gripper excitation logs first.  The output NPZ contains the
empty-load torque model, the direct payload regressor metadata, and a virtual
calibration payload fitted from empty residuals for the momentum-observer online
estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from t1_payload_id.common import (
    T1RegressorBuilder, load_json, save_json, resolve_project_path,
    nonlinear_friction_regressor, prune_zero_columns, qr_select_columns,
    robust_scaled_ridge, solve_scaled_ridge, compute_payload_H, metrics,
    payload_report
)


def load_npz_list(paths):
    Qs=[]; Vs=[]; As=[]; Ts=[]; metas=[]
    for p in paths:
        D=np.load(p, allow_pickle=True)
        Qs.append(D["Q"]); Vs.append(D["V"]); As.append(D["A"]); Ts.append(D["TAU"])
        metas.append({"file": str(p), "n_samples": int(D["Q"].shape[0]), "side": str(D["side"]) if "side" in D.files else "unknown"})
    return np.vstack(Qs), np.vstack(Vs), np.vstack(As), np.vstack(Ts), metas


def optimize_alpha(builder, Q, V, A, tau_stack, cfg, alpha0):
    if not cfg.get("fit_alpha", True):
        return np.asarray(alpha0, dtype=float), {"enabled": False}
    lo, hi = cfg.get("alpha_bounds", [0.2, 1.5])
    # Subsample for speed.
    stride = max(1, Q.shape[0] // 350)
    Qs, Vs, As = Q[::stride], V[::stride], A[::stride]
    taus = tau_stack.reshape(-1, 7)[::stride].reshape(-1)
    alpha0 = np.clip(np.asarray(alpha0, dtype=float), lo, hi)
    cache = {"calls": 0, "best": float("inf"), "alpha": alpha0.copy()}
    def obj(a):
        Y = builder.stack_full_regressor(Qs, Vs, As, a)
        Yk, keep = prune_zero_columns(Y, cfg.get("zero_column_rel_threshold", 1e-12))
        piv = qr_select_columns(Yk, cfg.get("qr_tol", 1e-9), cfg.get("base_rank_max", 80))
        Yb = Yk[:, piv]
        beta = solve_scaled_ridge(Yb, taus, cfg.get("ridge_lambda", 1e-8), True)
        r = taus - Yb @ beta
        rmse = float(np.sqrt(np.mean(r*r)))
        cache["calls"] += 1
        if rmse < cache["best"]:
            cache["best"] = rmse; cache["alpha"] = np.asarray(a).copy()
        return rmse
    res = minimize(obj, alpha0, method="L-BFGS-B", bounds=[(lo, hi)]*7, options={"maxiter": 60, "ftol": 1e-7})
    return np.asarray(res.x, dtype=float), {"enabled": True, "success": bool(res.success), "message": str(res.message), "rmse": float(res.fun), "calls": cache["calls"]}


def fit_virtual_calibration_payload(builder, Q, V, A, TAU, tau_hat, cfg):
    """Fit the paper-style virtual calibration object from empty-load residuals.

    This object is not the physical payload.  It captures repeatable unmodeled
    effects of the empty robot and is subtracted from online loaded estimates.
    """
    if not cfg.get("fit_virtual_calibration_payload", True):
        return np.zeros(16), np.zeros(7), {"enabled": False}

    start_fraction = float(cfg.get("virtual_calibration_start_fraction", 0.5))
    start = int(np.clip(start_fraction, 0.0, 0.9) * Q.shape[0])
    Qs, Vs, As = Q[start:], V[start:], A[start:]
    residual = (TAU[start:] - tau_hat.reshape(-1, 7)[start:]).reshape(-1)
    Wp = builder.stack_payload_regressor(Qs, Vs, As, mode=cfg.get("payload_param_mode", "pcpdi16"))
    bias_cols = np.tile(np.eye(7), (Qs.shape[0], 1))
    Phi = np.hstack([Wp, bias_cols])
    theta_bias = robust_scaled_ridge(
        Phi,
        residual,
        cfg.get("virtual_calibration_ridge_lambda", cfg.get("ridge_lambda", 1e-8)),
        cfg.get("robust_huber_delta", 2.5),
        cfg.get("robust_irls_iters", 6),
    )
    theta0 = theta_bias[:16]
    bias0 = theta_bias[16:23]
    residual_hat = Phi @ theta_bias
    meta = {
        "enabled": True,
        "start_fraction": start_fraction,
        "start_sample": int(start),
        "num_samples": int(Qs.shape[0]),
        "residual_rmse_nm": float(np.sqrt(np.mean((residual - residual_hat) ** 2))),
        "theta_virtual0_report": payload_report(theta0),
        "bias0": bias0.tolist(),
        "note": "theta_virtual0 is a calibration correction object, not a real payload; it is intentionally not physically projected.",
    }
    return theta0, bias0, meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/t1_payload_config.json")
    ap.add_argument("--train", nargs="+", required=True, help="processed empty-load NPZ files")
    ap.add_argument("--side", choices=["left", "right"], default=None)
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    cfg=load_json(args.config)
    icfg=cfg["identification"]
    side=args.side or icfg.get("side", "left")
    train_paths=[resolve_project_path(p) for p in args.train]
    Q,V,A,TAU,metas=load_npz_list(train_paths)
    tau=TAU.reshape(-1)
    builder=T1RegressorBuilder(cfg, side=side)
    alpha0=np.asarray(icfg.get("initial_alpha", [0.8]*7), dtype=float)
    print("[1/5] optimizing nonlinear friction alpha ...")
    alpha, alpha_meta=optimize_alpha(builder, Q,V,A,tau,icfg,alpha0)
    print("alpha =", alpha)

    print("[2/5] building full regressor ...")
    Y=builder.stack_full_regressor(Q,V,A,alpha)
    Y_keep, keep=prune_zero_columns(Y, icfg.get("zero_column_rel_threshold",1e-12))
    base_idx_in_keep=qr_select_columns(Y_keep, icfg.get("qr_tol",1e-9), icfg.get("base_rank_max",80))
    Y_base=Y_keep[:, base_idx_in_keep]
    print(f"Y full={Y.shape}, keep={Y_keep.shape}, base={Y_base.shape}")

    print("[3/5] solving robust ridge LS for empty-load beta0 ...")
    beta=robust_scaled_ridge(Y_base, tau, icfg.get("ridge_lambda",1e-8), icfg.get("robust_huber_delta",2.5), icfg.get("robust_irls_iters",6))
    tau_hat=Y_base @ beta
    fit_metrics=metrics(tau, tau_hat, 7)
    print("fit RMSE =", fit_metrics["overall_rmse"])

    print("[4/5] building T1 numerical payload map H ...")
    Wp=builder.stack_payload_regressor(Q,V,A, mode=icfg.get("payload_param_mode","pcpdi16"))
    H_payload, H_meta=compute_payload_H(Y_base, Wp, ridge=icfg.get("payload_H_ridge",1e-7))
    print(
        "H rel err =", H_meta["relative_reconstruction_error"],
        "effective rank =", H_meta["H_numerical_rank"],
        "effective cond =", H_meta["H_effective_condition"],
    )

    print("[5/5] fitting virtual calibration payload from empty residual ...")
    theta_virtual0, virtual_bias0, virtual_meta = fit_virtual_calibration_payload(builder, Q, V, A, TAU, tau_hat.reshape(-1, 7), icfg)
    print("virtual mass correction =", float(theta_virtual0[0]), "kg")

    out=resolve_project_path(args.out or f"models/{side}_empty_payload_prior.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        side=np.array(side), alpha=alpha, beta0=beta, keep_mask=keep, base_idx_in_keep=base_idx_in_keep,
        H_payload=H_payload, H_payload_meta_json=np.array(json.dumps(H_meta, ensure_ascii=False)),
        theta_virtual0=theta_virtual0, virtual_bias0=virtual_bias0,
        virtual_payload_meta_json=np.array(json.dumps(virtual_meta, ensure_ascii=False)),
        payload_param_mode=np.array(icfg.get("payload_param_mode","pcpdi16")), include_motor_inertia=np.array(builder.include_motor),
        Q=Q, V=V, A=A, TAU=TAU, tau_hat=tau_hat.reshape(-1,7), config_json=np.array(json.dumps(cfg, ensure_ascii=False)),
        train_metas_json=np.array(json.dumps(metas, ensure_ascii=False)))
    alpha_lo, alpha_hi = icfg.get("alpha_bounds", [0.2, 1.5])
    alpha_at_bounds = [
        bool(np.isclose(value, alpha_lo) or np.isclose(value, alpha_hi))
        for value in alpha
    ]
    summary={
        "side": side, "model_file": str(out), "training": metas, "alpha": alpha.tolist(),
        "alpha_meta": alpha_meta, "alpha_at_bounds": alpha_at_bounds,
        "dimensions": {"Y_full": list(Y.shape), "Y_keep": list(Y_keep.shape), "Y_base": list(Y_base.shape), "H_payload": list(H_payload.shape)},
        "fit_metrics": fit_metrics, "H_payload_meta": H_meta,
        "virtual_calibration_payload": virtual_meta,
        "important": "Use this NPZ as the online prior. beta0 is the empty-load torque prior; theta_virtual0 is only a residual correction for online momentum-observer payload ID."
    }
    save_json(summary, out.with_suffix(".json"))
    print(f"[OK] saved {out}")

if __name__ == "__main__":
    main()
