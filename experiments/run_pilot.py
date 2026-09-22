#!/usr/bin/env python3
"""Reproducible pilot study for the surrogate-assisted rolling-horizon paper.

The script builds a synthetic three-stage, three-product chemical line, trains
surrogate ablations, fits a bounded-domain piecewise-affine (PWA) surrogate,
and evaluates four closed-loop policies on common out-of-sample paths.
It is deliberately self-contained; plant data should replace the calibrated
constants before the results are used as empirical evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import torch
from scipy.optimize import lsq_linear
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


S, P = 3, 3
STAGES = ("reaction", "pressing", "drying")
PRODUCTS = ("A", "B", "C")
BASE_DEMAND = np.array([3.2, 2.8, 2.4])
BASE_CAPACITY = np.array([10.5, 9.2, 8.4])
BASE_YIELD = np.array([[0.96, 0.95, 0.94], [0.94, 0.93, 0.92], [0.92, 0.91, 0.90]])
PROC_TIME = np.array([[0.90, 1.00, 1.10], [0.95, 1.05, 1.15], [1.00, 1.10, 1.20]])
CLEAN = np.array([[0.0, 1.1, 1.7], [1.3, 0.0, 1.0], [1.8, 1.2, 0.0]])
TARGET_WIP = np.array([[7.0, 6.5, 6.0], [6.0, 5.5, 5.0], [5.0, 4.5, 4.0]])
L = 12
F_W = slice(0, 3)
F_IN = slice(3, 6)
F_ACTIVE = slice(6, 9)
I_SAME, I_G, I_A = 9, 10, 11
F_RHO = slice(12, 15)
F_PREVQ = slice(15, 18)
N_FEATURES = 18


@dataclass
class State:
    wip: np.ndarray
    inventory: np.ndarray
    backlog: np.ndarray
    prev_product: np.ndarray
    prev_q: np.ndarray
    ramp: np.ndarray
    cleaning_carry: np.ndarray
    health: np.ndarray


def initial_state(scale: float = 1.0) -> State:
    return State(
        wip=TARGET_WIP * scale,
        inventory=np.array([2.0, 2.0, 2.0]) * scale,
        backlog=np.zeros(P),
        prev_product=np.array([0, 0, 0], dtype=int),
        prev_q=np.zeros((S, P)),
        ramp=np.full((S, P), 0.25),
        cleaning_carry=np.zeros(S),
        health=np.ones(S),
    )


def exogenous_path(length: int, seed: int, stress: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * t / 12.0)
    sigma = 0.28 if stress else 0.14
    bias = 1.12 if stress else 1.0
    demand = np.maximum(0.0, BASE_DEMAND[None, :] * seasonal[:, None] * bias)
    demand *= np.exp(rng.normal(-0.5 * sigma**2, sigma, size=(length, P)))
    if stress:
        shock_start = int(rng.integers(3, max(4, length - 4)))
        demand[shock_start : shock_start + 3] *= np.array([1.45, 1.20, 1.35])

    availability = np.tile(BASE_CAPACITY, (length, 1))
    down_prob = 0.18 if stress else 0.07
    down = rng.random((length, S)) < down_prob
    availability *= np.where(down, rng.uniform(0.42, 0.72, size=(length, S)), 1.0)
    availability *= rng.uniform(0.94, 1.03, size=(length, S))
    ysd = 0.025 if stress else 0.012
    yields = np.clip(BASE_YIELD[None, :, :] + rng.normal(0, ysd, (length, S, P)), 0.78, 0.99)
    clean_mult = rng.lognormal(mean=0.12 if stress else 0.0, sigma=0.10, size=(length, S))
    return {"demand": demand, "availability": availability, "yield": yields, "clean_mult": clean_mult}


def make_feature(wip: np.ndarray, inflow: np.ndarray, active: int, prev: int,
                 cleaning: float, availability: float, rho: np.ndarray,
                 prev_q: np.ndarray) -> np.ndarray:
    x = np.zeros(N_FEATURES, dtype=np.float32)
    x[F_W], x[F_IN] = wip, inflow
    if active >= 0:
        x[6 + active] = 1.0
    x[I_SAME] = float(active == prev)
    x[I_G], x[I_A] = cleaning, availability
    # Previous completion is intentionally not exposed. The temporal models
    # must infer ramp-up and carryover from the observable history in Eq. (5).
    x[F_RHO], x[F_PREVQ] = rho, 0.0
    return x


def execute_period(state: State, release: np.ndarray, active: np.ndarray,
                   exo: dict[str, np.ndarray], t: int, rng: np.random.Generator,
                   collect_features: bool = False) -> tuple[State, dict]:
    w = state.wip.copy()
    q = np.zeros((S, P))
    features = np.zeros((S, N_FEATURES), dtype=np.float32)
    cleaning = np.zeros(S)
    ramp_next = 0.80 * state.ramp
    carry_next = np.zeros(S)
    health_next = np.zeros(S)
    inflow = np.maximum(0.0, release.copy())
    action_reject = 0
    for s in range(S):
        p = int(active[s])
        if p < 0:
            features[s] = make_feature(w[s], inflow, -1, int(state.prev_product[s]), 0.0,
                                       exo["availability"][t, s], exo["yield"][t, s], state.prev_q[s])
            w[s] += inflow
            inflow = np.zeros(P)
            continue
        cleaning[s] = CLEAN[state.prev_product[s], p] * exo["clean_mult"][t, s]
        a = exo["availability"][t, s]
        features[s] = make_feature(w[s], inflow, p, int(state.prev_product[s]), cleaning[s],
                                   a, exo["yield"][t, s], state.prev_q[s])
        material_vec = w[s] + inflow
        material = material_vec[p]
        # Cleaning carryover and post-disruption recovery are latent states.
        # Their recent causes are observable in the temporal feature window.
        usable_time = max(0.0, a - cleaning[s] - 0.80 * state.cleaning_carry[s])
        nominal = usable_time / PROC_TIME[s, p]
        saturation = 1.0 - math.exp(-max(material, 0.0) / (4.5 + 0.7 * s))
        same = float(p == state.prev_product[s])
        other = max(0.0, material_vec.sum() - material)
        mix_eff = 1.0 - 0.13 * other / (material_vec.sum() + 1.0)
        progress = 0.22 + 0.58 * state.ramp[s, p] + 0.20 * state.health[s]
        process_noise = float(np.clip(rng.normal(1.0, 0.025), 0.92, 1.06))
        completed = min(material, max(0.0, nominal * saturation * progress * mix_eff * process_noise))
        if material < 1e-7 or usable_time < 1e-7:
            action_reject += 1
        q[s, p] = completed
        w[s] = np.maximum(0.0, material_vec - q[s])
        inflow = exo["yield"][t, s] * q[s]
        ramp_next[s, p] = (min(1.0, 0.72 * state.ramp[s, p] + 0.28) if same else 0.08)
        carry_next[s] = 0.76 * state.cleaning_carry[s] + 0.90 * cleaning[s]
        observed_availability = float(np.clip(a / BASE_CAPACITY[s], 0.0, 1.0))
        health_next[s] = 0.85 * state.health[s] + 0.15 * observed_availability

    usable_finished = inflow
    available_finished = state.inventory + usable_finished
    due = state.backlog + exo["demand"][t]
    shipment = np.minimum(available_finished, due)
    inventory = available_finished - shipment
    backlog = due - shipment
    prev_product = state.prev_product.copy()
    for s in range(S):
        if active[s] >= 0:
            prev_product[s] = int(active[s])
    next_state = State(w, inventory, backlog, prev_product, q, ramp_next, carry_next, health_next)
    cost = {
        "inventory_cost": float(inventory.sum()),
        "backlog_cost": float(12.0 * backlog.sum()),
        "wip_cost": float(0.35 * w.sum()),
        "cleaning_cost": float(0.60 * cleaning.sum()),
    }
    cost["total_cost"] = sum(cost.values())
    info = {**cost, "q": q, "shipment": shipment, "cleaning": cleaning,
            "features": features if collect_features else None, "action_reject": action_reject}
    return next_state, info


def training_policy(state: State, exo: dict[str, np.ndarray], t: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    forecast = BASE_DEMAND * (1.0 + 0.12 * math.sin(2 * math.pi * t / 12.0))
    release = np.clip(forecast * rng.uniform(0.75, 1.35, P) +
                      0.35 * (TARGET_WIP[0] - state.wip[0]), 0.0, 7.0)
    active = np.zeros(S, dtype=int)
    for s in range(S):
        score = state.wip[s] + 0.7 * state.backlog + rng.uniform(0, 3.0, P)
        prev = state.prev_product[s]
        score[prev] += 2.2 if rng.random() < 0.65 else 0.0
        active[s] = int(np.argmax(score))
    return release, active


def generate_trajectories(n: int, length: int, seed: int,
                          regime: str = "mixed") -> tuple[np.ndarray, np.ndarray]:
    all_x = np.zeros((n, S, length, N_FEATURES), dtype=np.float32)
    all_q = np.zeros((n, S, length, P), dtype=np.float32)
    for r in range(n):
        rng = np.random.default_rng(seed + 1009 * r)
        if regime == "nominal":
            stress = False
        elif regime == "stress":
            stress = True
        else:
            stress = bool(r % 2 == 0)
        exo = exogenous_path(length, seed + 7919 * r, stress=stress)
        state = initial_state(scale=float(rng.uniform(0.65, 1.35)))
        for t in range(length):
            release, active = training_policy(state, exo, t, rng)
            state, info = execute_period(state, release, active, exo, t, rng, True)
            all_x[r, :, t] = info["features"]
            all_q[r, :, t] = info["q"]
    return all_x, all_q


def windows(x: np.ndarray, q: np.ndarray, stage: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, meta = [], [], []
    for r in range(x.shape[0]):
        for t in range(L - 1, x.shape[2]):
            xx.append(x[r, stage, t - L + 1 : t + 1])
            yy.append(q[r, stage, t])
            meta.append((r, t))
    return np.asarray(xx, np.float32), np.asarray(yy, np.float32), np.asarray(meta, int)


class CurrentMLP(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 32), nn.ReLU(), nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, P), nn.Softplus())

    def forward(self, x):
        return self.net(x[:, -1])


class WindowMLP(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        # Width is chosen to keep the parameter count comparable with TinyTCN.
        self.net = nn.Sequential(nn.Linear(L * f, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, P), nn.Softplus())

    def forward(self, x):
        return self.net(x.flatten(1))


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.c1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)

    def _trim(self, z):
        return z[:, :, :-self.pad] if self.pad else z

    def forward(self, x):
        z = torch.relu(self._trim(self.c1(x)))
        z = self._trim(self.c2(z))
        return torch.relu(x + z)


class TinyTCN(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.inp = nn.Conv1d(f, 16, 1)
        self.blocks = nn.Sequential(CausalBlock(16, 1), CausalBlock(16, 2))
        self.out = nn.Sequential(nn.Linear(16, P), nn.Softplus())

    def forward(self, x):
        z = self.blocks(torch.relu(self.inp(x.transpose(1, 2))))
        return self.out(z[:, :, -1])


@dataclass
class FittedModel:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    qscale: np.ndarray
    stage: int
    name: str

    def predict(self, raw: np.ndarray, batch: int = 1024) -> np.ndarray:
        self.model.eval()
        z = (raw - self.mean) / self.std
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                raw_mask = raw[i : i + batch, -1, F_ACTIVE]
                value = self.model(torch.tensor(z[i : i + batch], dtype=torch.float32)).cpu().numpy()
                out.append(value * raw_mask)
        return np.vstack(out) * self.qscale


def train_model(kind: str, stage: int, tr_x: np.ndarray, tr_y: np.ndarray,
                va_x: np.ndarray, va_y: np.ndarray, epochs: int, seed: int) -> FittedModel:
    torch.manual_seed(seed + 31 * stage)
    np.random.seed(seed + 31 * stage)
    mean = tr_x.reshape(-1, N_FEATURES).mean(0)
    std = tr_x.reshape(-1, N_FEATURES).std(0) + 1e-4
    qscale = np.maximum(np.quantile(tr_y, 0.95, axis=0), 1.0)
    tx = torch.tensor((tr_x - mean) / std, dtype=torch.float32)
    ty = torch.tensor(tr_y / qscale, dtype=torch.float32)
    vx = torch.tensor((va_x - mean) / std, dtype=torch.float32)
    vy = torch.tensor(va_y / qscale, dtype=torch.float32)
    if kind == "current_mlp":
        model = CurrentMLP(N_FEATURES)
    elif kind == "window_mlp":
        model = WindowMLP(N_FEATURES)
    else:
        model = TinyTCN(N_FEATURES)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=256, shuffle=True)
    best, best_state, patience = float("inf"), None, 0
    pi = kind == "pi_tcn"
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            raw_last = xb[:, -1] * torch.tensor(std) + torch.tensor(mean)
            active_mask = raw_last[:, F_ACTIVE]
            pred = model(xb) * active_mask
            loss = ((pred - yb) ** 2).mean()
            if pi and epoch >= max(4, epochs // 3):
                pred_raw = pred * torch.tensor(qscale)
                mat = raw_last[:, F_W] + raw_last[:, F_IN]
                material_pen = (torch.relu(pred_raw - mat) / torch.tensor(qscale)).pow(2).mean()
                cap_used = (pred_raw * torch.tensor(PROC_TIME[stage])).sum(1) + raw_last[:, I_G]
                capacity_pen = (torch.relu(cap_used - raw_last[:, I_A]) / 5.0).pow(2).mean()
                xw, xa, xg = xb.clone(), xb.clone(), xb.clone()
                xw[:, -1, F_W] += torch.tensor(0.5 / std[F_W])
                xa[:, -1, I_A] += float(0.5 / std[I_A])
                xg[:, -1, I_G] += float(0.3 / std[I_G])
                mono = (torch.relu(pred - model(xw) * active_mask).pow(2).mean() +
                        torch.relu(pred - model(xa) * active_mask).pow(2).mean() +
                        torch.relu(model(xg) * active_mask - pred).pow(2).mean())
                # Unlabelled collocation points expose the model to physically
                # difficult low-WIP, low-availability, high-cleaning states.
                raw_phys = raw_last.clone()
                raw_phys[:, F_W] *= 0.20
                raw_phys[:, F_IN] *= 0.50
                raw_phys[:, I_A] *= 0.45
                raw_phys[:, I_G] = torch.minimum(2.0 * raw_phys[:, I_G] + 1.0,
                                                 0.80 * raw_phys[:, I_A])
                xphys = xb.clone()
                xphys[:, -1] = (raw_phys - torch.tensor(mean)) / torch.tensor(std)
                pred_phys = model(xphys) * active_mask * torch.tensor(qscale)
                mat_phys = raw_phys[:, F_W] + raw_phys[:, F_IN]
                phys_material = (torch.relu(pred_phys - mat_phys) / torch.tensor(qscale)).pow(2).mean()
                phys_cap_used = (pred_phys * torch.tensor(PROC_TIME[stage])).sum(1) + raw_phys[:, I_G]
                phys_capacity = (torch.relu(phys_cap_used - raw_phys[:, I_A]) / 5.0).pow(2).mean()
                loss = (loss + 0.30 * material_pen + 0.50 * capacity_pen + 0.25 * mono
                        + 0.20 * phys_material + 0.30 * phys_capacity)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vraw = vx[:, -1] * torch.tensor(std) + torch.tensor(mean)
            val = float(((model(vx) * vraw[:, F_ACTIVE] - vy) ** 2).mean())
        if val < best - 1e-5:
            best, patience = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return FittedModel(model, mean, std, qscale, stage, kind)


def surrogate_metrics(fit: FittedModel, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = fit.predict(x)
    denom = max(float(y.std()), 1e-3)
    nrmse = float(np.sqrt(np.mean((pred - y) ** 2)) / denom)
    last = x[:, -1]
    mat = last[:, F_W] + last[:, F_IN]
    material_violation = np.any(pred > mat + 1e-4, axis=1)
    cap = (pred * PROC_TIME[fit.stage]).sum(1) + last[:, I_G]
    capacity_violation = cap > last[:, I_A] + 1e-4
    optimistic = np.maximum(pred - y, 0.0) / max(float(np.std(y)), 1e-3)
    xw, xa, xg = x.copy(), x.copy(), x.copy()
    xw[:, -1, F_W] += 0.5
    xa[:, -1, I_A] += 0.5
    xg[:, -1, I_G] += 0.3
    pw, pa, pg = fit.predict(xw), fit.predict(xa), fit.predict(xg)
    mono_bad = np.concatenate([(pw < pred - 1e-4).ravel(), (pa < pred - 1e-4).ravel(),
                               (pg > pred + 1e-4).ravel()])
    ood = x.copy()
    ood[:, -1, F_W] *= 0.20
    ood[:, -1, F_IN] *= 0.50
    ood[:, -1, I_A] *= 0.45
    ood[:, -1, I_G] = np.minimum(2.0 * ood[:, -1, I_G] + 1.0,
                                 0.80 * ood[:, -1, I_A])
    ood_pred = fit.predict(ood)
    ood_last = ood[:, -1]
    ood_mat = ood_last[:, F_W] + ood_last[:, F_IN]
    ood_mat_bad = np.any(ood_pred > ood_mat + 1e-4, axis=1)
    ood_cap = (ood_pred * PROC_TIME[fit.stage]).sum(1) + ood_last[:, I_G]
    ood_cap_bad = ood_cap > ood_last[:, I_A] + 1e-4
    ood_bad = ood_mat_bad | ood_cap_bad
    return {"nrmse_q": nrmse,
            "material_violation_pct": 100 * float(material_violation.mean()),
            "capacity_violation_pct": 100 * float(capacity_violation.mean()),
            "ood_physical_violation_pct": 100 * float(ood_bad.mean()),
            "monotonicity_violation_pct": 100 * float(mono_bad.mean()),
            "optimistic_error_p95": float(np.quantile(optimistic, 0.95))}


def rollout_nrmse(fit: FittedModel, raw_x: np.ndarray, raw_q: np.ndarray, horizon: int = 4) -> float:
    errs, truths = [], []
    s = fit.stage
    for r in range(raw_x.shape[0]):
        for start in range(L - 1, raw_x.shape[2] - horizon, horizon):
            hist = raw_x[r, s, start - L + 1 : start + 1].copy()
            w_pred = hist[-1, F_W].copy()
            for h in range(horizon):
                t = start + h
                current = raw_x[r, s, t].copy()
                current[F_W] = w_pred
                if h == 0:
                    hist[-1] = current
                else:
                    hist = np.concatenate([hist[1:], current[None]], axis=0)
                qhat = fit.predict(hist[None])[0]
                w_pred = np.maximum(0.0, w_pred + current[F_IN] - qhat)
                w_true = np.maximum(0.0, raw_x[r, s, t, F_W] + raw_x[r, s, t, F_IN] - raw_q[r, s, t])
                errs.append(w_pred - w_true)
                truths.append(w_true)
    return float(np.sqrt(np.mean(np.square(errs))) / (np.std(truths) + 1e-3))


def rollout_curve(fit: FittedModel, raw_x: np.ndarray, raw_q: np.ndarray,
                  horizon: int = 6) -> list[float]:
    by_h = [[] for _ in range(horizon)]
    true_h = [[] for _ in range(horizon)]
    s = fit.stage
    for r in range(raw_x.shape[0]):
        for start in range(L - 1, raw_x.shape[2] - horizon, horizon):
            hist = raw_x[r, s, start - L + 1 : start + 1].copy()
            w_pred = hist[-1, F_W].copy()
            for h in range(horizon):
                t = start + h
                current = raw_x[r, s, t].copy()
                current[F_W] = w_pred
                hist[-1] = current if h == 0 else hist[-1]
                if h > 0:
                    hist = np.concatenate([hist[1:], current[None]], axis=0)
                qhat = fit.predict(hist[None])[0]
                w_pred = np.maximum(0.0, w_pred + current[F_IN] - qhat)
                w_true = np.maximum(0.0, raw_x[r, s, t, F_W] + raw_x[r, s, t, F_IN] - raw_q[r, s, t])
                by_h[h].append(w_pred - w_true)
                true_h[h].append(w_true)
    return [float(np.sqrt(np.mean(np.square(by_h[h]))) / (np.std(true_h[h]) + 1e-3))
            for h in range(horizon)]


@dataclass
class PWAModel:
    thresholds: np.ndarray  # S,P,2
    coef: np.ndarray        # S,P,3,6: 1, material, A, -G, same, prevQ

    def predict_samples(self, stage: int, raw: np.ndarray) -> np.ndarray:
        out = np.zeros((len(raw), P))
        cur = raw[:, -1]
        for p in range(P):
            mat = cur[:, p] + cur[:, 3 + p]
            reg = np.digitize(mat, self.thresholds[stage, p])
            z = np.column_stack([np.ones(len(raw)), mat, cur[:, I_A], -cur[:, I_G],
                                 cur[:, I_SAME], cur[:, 15 + p]])
            for k in range(3):
                idx = reg == k
                out[idx, p] = np.maximum(0.0, z[idx] @ self.coef[stage, p, k]) * cur[idx, 6 + p]
        return out


def fit_pwa(pi_models: list[FittedModel], train_windows: list[np.ndarray]) -> PWAModel:
    thresholds = np.zeros((S, P, 2))
    coef = np.zeros((S, P, 3, 6))
    for s in range(S):
        raw = train_windows[s]
        target = pi_models[s].predict(raw)
        cur = raw[:, -1]
        for p in range(P):
            mat = cur[:, p] + cur[:, 3 + p]
            thresholds[s, p] = np.quantile(mat, [0.33, 0.66])
            reg = np.digitize(mat, thresholds[s, p])
            active = cur[:, 6 + p] > 0.5
            z = np.column_stack([np.ones(len(raw)), mat, cur[:, I_A], -cur[:, I_G],
                                 cur[:, I_SAME], cur[:, 15 + p]])
            for k in range(3):
                idx = (reg == k) & active
                if idx.sum() < 20:
                    idx = active.copy()
                # Positive slopes for material/availability/same/carry and negative cleaning effect.
                res = lsq_linear(z[idx], target[idx, p], bounds=([-20, 0, 0, 0, 0, 0], np.inf))
                c = res.x
                raw_fit = z[idx] @ c
                # Remove the upper tail of optimistic approximation error.
                c[0] -= max(0.0, float(np.quantile(raw_fit - target[idx, p], 0.90)))
                # Keep the expression nonnegative over the observed domain.
                c[0] += max(0.0, -float(np.min(z[idx] @ c)) + 0.02)
                coef[s, p, k] = c
    return PWAModel(thresholds, coef)


def pwa_metrics(pwa: PWAModel, pi_models: list[FittedModel], test_windows: list[np.ndarray],
                test_targets: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for s in range(S):
        pwa_pred = pwa.predict_samples(s, test_windows[s])
        tcn_pred = pi_models[s].predict(test_windows[s])
        y = test_targets[s]
        cur = test_windows[s][:, -1]
        mat = cur[:, F_W] + cur[:, F_IN]
        mat_bad = np.any(pwa_pred > mat + 1e-4, axis=1)
        cap_bad = (pwa_pred * PROC_TIME[s]).sum(1) + cur[:, I_G] > cur[:, I_A] + 1e-4
        rows.append({"stage": STAGES[s],
                     "pwa_vs_tcn_nrmse": float(np.sqrt(np.mean((pwa_pred - tcn_pred) ** 2)) / (np.std(tcn_pred) + 1e-3)),
                     "pwa_vs_des_nrmse": float(np.sqrt(np.mean((pwa_pred - y) ** 2)) / (np.std(y) + 1e-3)),
                     "material_violation_pct": 100 * float(mat_bad.mean()),
                     "capacity_violation_pct": 100 * float(cap_bad.mean()),
                     "optimistic_error_p95": float(np.quantile(np.maximum(pwa_pred - y, 0) / (np.std(y) + 1e-3), 0.95))})
    return pd.DataFrame(rows)


def forecast_scenarios(t: int, horizon: int, n: int, stress: bool, seed: int) -> list[dict[str, np.ndarray]]:
    scenarios = []
    for j in range(n):
        ex = exogenous_path(t + horizon + 1, seed + 997 * j, stress)
        scenarios.append({k: v[t : t + horizon] for k, v in ex.items()})
    return scenarios


def deterministic_scenario(t: int, horizon: int, stress: bool) -> dict[str, np.ndarray]:
    tt = np.arange(t, t + horizon)
    seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * tt / 12.0)
    demand = BASE_DEMAND[None, :] * seasonal[:, None] * (1.12 if stress else 1.0)
    down_prob = 0.18 if stress else 0.07
    mean_down_factor = 0.57
    availability = np.tile(BASE_CAPACITY * (1 - down_prob * (1 - mean_down_factor)), (horizon, 1))
    yields = np.tile(BASE_YIELD, (horizon, 1, 1))
    clean_mult = np.full((horizon, S), math.exp(0.12 if stress else 0.0))
    return {"demand": demand, "availability": availability, "yield": yields, "clean_mult": clean_mult}


def solve_plan(state: State, pwa: PWAModel, scenarios: list[dict[str, np.ndarray]],
               horizon: int, time_limit: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    start = time.perf_counter()
    m = pulp.LpProblem("rolling_plan", pulp.LpMinimize)
    O = range(len(scenarios)); H = range(horizon)
    release = pulp.LpVariable.dicts("R", (range(P), H), lowBound=0, upBound=7)
    x = pulp.LpVariable.dicts("x", (range(S), range(P), H), cat="Binary")
    z = pulp.LpVariable.dicts("z", (range(S), range(P), range(P), H), cat="Binary")
    W = pulp.LpVariable.dicts("W", (O, range(S), range(P), range(horizon + 1)), lowBound=0)
    Q = pulp.LpVariable.dicts("Q", (O, range(S), range(P), H), lowBound=0)
    I = pulp.LpVariable.dicts("I", (O, range(P), range(horizon + 1)), lowBound=0)
    B = pulp.LpVariable.dicts("B", (O, range(P), range(horizon + 1)), lowBound=0)
    Ship = pulp.LpVariable.dicts("Ship", (O, range(P), H), lowBound=0)
    dlt = pulp.LpVariable.dicts("d", (O, range(S), range(P), range(3), H), cat="Binary")
    for s in range(S):
        for h in H:
            m += pulp.lpSum(x[s][p][h] for p in range(P)) == 1
            for p in range(P):
                for q in range(P):
                    prev_expr = 1.0 if (h == 0 and state.prev_product[s] == p) else (x[s][p][h - 1] if h > 0 else 0.0)
                    m += z[s][p][q][h] <= prev_expr
                    m += z[s][p][q][h] <= x[s][q][h]
                    m += z[s][p][q][h] >= prev_expr + x[s][q][h] - 1
    big_m = 100.0
    obj = []
    for o, sc in enumerate(scenarios):
        for s in range(S):
            for p in range(P):
                m += W[o][s][p][0] == float(state.wip[s, p])
        for p in range(P):
            m += I[o][p][0] == float(state.inventory[p])
            m += B[o][p][0] == float(state.backlog[p])
        for h in H:
            for s in range(S):
                clean_expr = pulp.lpSum(float(CLEAN[p, q] * sc["clean_mult"][h, s]) * z[s][p][q][h]
                                        for p in range(P) for q in range(P))
                m += pulp.lpSum(float(PROC_TIME[s, p]) * Q[o][s][p][h] for p in range(P)) + clean_expr <= float(sc["availability"][h, s])
                for p in range(P):
                    if s == 0:
                        inflow = release[p][h]
                    else:
                        inflow = float(sc["yield"][h, s - 1, p]) * Q[o][s - 1][p][h]
                    material = W[o][s][p][h] + inflow
                    m += Q[o][s][p][h] <= material
                    m += Q[o][s][p][h] <= big_m * x[s][p][h]
                    m += pulp.lpSum(dlt[o][s][p][k][h] for k in range(3)) == x[s][p][h]
                    for k in range(3):
                        lo = 0.0 if k == 0 else float(pwa.thresholds[s, p, k - 1])
                        hi = float(pwa.thresholds[s, p, k]) if k < 2 else big_m
                        m += material >= lo - big_m * (1 - dlt[o][s][p][k][h])
                        m += material <= hi + big_m * (1 - dlt[o][s][p][k][h])
                        c = pwa.coef[s, p, k]
                        same_expr = z[s][p][p][h]
                        prevq = float(state.prev_q[s, p]) if h == 0 else Q[o][s][p][h - 1]
                        cap_expr = (float(c[0]) + float(c[1]) * material + float(c[2]) * float(sc["availability"][h, s])
                                    - float(c[3]) * clean_expr + float(c[4]) * same_expr + float(c[5]) * prevq)
                        m += Q[o][s][p][h] <= cap_expr + big_m * (1 - dlt[o][s][p][k][h])
                    m += W[o][s][p][h + 1] == material - Q[o][s][p][h]
            for p in range(P):
                usable = float(sc["yield"][h, 2, p]) * Q[o][2][p][h]
                m += Ship[o][p][h] <= I[o][p][h] + usable
                m += Ship[o][p][h] <= B[o][p][h] + float(sc["demand"][h, p])
                m += I[o][p][h + 1] == I[o][p][h] + usable - Ship[o][p][h]
                m += B[o][p][h + 1] == B[o][p][h] + float(sc["demand"][h, p]) - Ship[o][p][h]
            inv_cost = pulp.lpSum(I[o][p][h + 1] for p in range(P))
            back_cost = 12.0 * pulp.lpSum(B[o][p][h + 1] for p in range(P))
            wip_cost = 0.35 * pulp.lpSum(W[o][s][p][h + 1] for s in range(S) for p in range(P))
            clean_cost = 0.60 * pulp.lpSum(float(CLEAN[p, q] * sc["clean_mult"][h, s]) * z[s][p][q][h]
                                           for s in range(S) for p in range(P) for q in range(P))
            obj.append((inv_cost + back_cost + wip_cost + clean_cost) / len(scenarios))
    m += pulp.lpSum(obj)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.03)
    m.solve(solver)
    status = pulp.LpStatus[m.status]
    rel = np.zeros((horizon, P)); act = np.zeros((horizon, S), dtype=int)
    qplan = np.zeros((horizon, S, P))
    for h in H:
        for p in range(P):
            rel[h, p] = max(0.0, float(pulp.value(release[p][h]) or 0.0))
        for s in range(S):
            vals = [float(pulp.value(x[s][p][h]) or 0.0) for p in range(P)]
            act[h, s] = int(np.argmax(vals))
            for p in range(P):
                qplan[h, s, p] = np.mean([max(0.0, float(pulp.value(Q[o][s][p][h]) or 0.0)) for o in O])
    return rel, act, qplan, time.perf_counter() - start, status


def heuristic_action(state: State, t: int) -> tuple[np.ndarray, np.ndarray]:
    forecast = BASE_DEMAND * (1.0 + 0.12 * math.sin(2 * math.pi * t / 12.0))
    release = np.clip(forecast + 0.45 * state.backlog + 0.35 * (TARGET_WIP[0] - state.wip[0]), 0.0, 7.0)
    active = np.zeros(S, dtype=int)
    for s in range(S):
        score = 1.7 * state.backlog + state.wip[s]
        score[state.prev_product[s]] += 1.4
        active[s] = int(np.argmax(score))
    return release, active


def evaluate_policy(policy: str, pwa: PWAModel, seed: int, stress: bool,
                    length: int, horizon: int, solver_time_limit: float) -> dict[str, float]:
    actual = exogenous_path(length, seed, stress)
    rng = np.random.default_rng(seed + 17)
    scale = [0.7, 1.0, 1.3][seed % 3]
    state = initial_state(scale)
    totals = {k: 0.0 for k in ["inventory_cost", "backlog_cost", "wip_cost", "cleaning_cost", "total_cost"]}
    total_demand = total_ship = total_clean = total_reject = total_wip = 0.0
    solve_times, timeouts = [], 0
    cached_rel = cached_act = cached_q = None
    shortfall_num = shortfall_den = 0.0
    for t in range(length):
        if policy == "heuristic":
            rel, act = heuristic_action(state, t)
        else:
            replan = policy != "scenario_block" or cached_rel is None or t % horizon == 0
            if replan:
                if policy == "deterministic_rh":
                    sc = [deterministic_scenario(t, horizon, stress)]
                else:
                    sc = forecast_scenarios(t, horizon, 2, stress, seed=500000 + seed * 101 + t)
                cached_rel, cached_act, cached_q, sec, status = solve_plan(
                    state, pwa, sc, horizon, time_limit=solver_time_limit)
                solve_times.append(sec)
                timeouts += int(status not in ("Optimal", "Integer Feasible"))
            idx = 0 if policy != "scenario_block" else t % horizon
            rel, act = cached_rel[idx], cached_act[idx]
        planned_active = np.asarray(act, dtype=int)
        state, info = execute_period(state, np.asarray(rel), planned_active, actual, t, rng)
        if policy != "heuristic":
            shortfall_num += float(np.maximum(cached_q[idx] - info["q"], 0.0).sum())
            shortfall_den += float(cached_q[idx].sum())
        for k in totals:
            totals[k] += info[k]
        total_demand += float(actual["demand"][t].sum())
        total_ship += float(info["shipment"].sum())
        total_clean += float(info["cleaning"].sum())
        total_reject += float(info["action_reject"])
        total_wip += float(state.wip.sum())
    return {"policy": policy, "stress": int(stress), "seed": seed, **totals,
            "fill_rate": total_ship / max(total_demand, 1e-6),
            "mean_backlog": totals["backlog_cost"] / (12.0 * length),
            "mean_wip": total_wip / length, "cleaning_time": total_clean,
            "action_reject_rate": total_reject / (length * S),
            "target_shortfall_ratio": shortfall_num / max(shortfall_den, 1e-6) if policy != "heuristic" else np.nan,
            "solve_time_median": float(np.median(solve_times)) if solve_times else 0.0,
            "solve_time_p95": float(np.quantile(solve_times, 0.95)) if solve_times else 0.0,
            "timeout_count": timeouts}


def plot_results(sur: pd.DataFrame, rollout: pd.DataFrame, pol: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    pivot = sur.groupby("model")[["nrmse_q", "rollout_wip_nrmse"]].mean().loc[
        ["current_mlp", "window_mlp", "data_tcn", "pi_tcn"]]
    pivot.plot(kind="bar", ax=ax, color=["#4472C4", "#ED7D31"])
    ax.set_ylabel("Normalized error"); ax.set_xlabel("")
    ax.legend(["Throughput NRMSE", "4-step WIP NRMSE"], frameon=False)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(out / "surrogate_ablation.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for name, color, marker in [("current_mlp", "#7F7F7F", "o"),
                                ("window_mlp", "#ED7D31", "s"),
                                ("data_tcn", "#4472C4", "^"),
                                ("pi_tcn", "#70AD47", "D")]:
        g = rollout[rollout.model == name].groupby("horizon").wip_nrmse.mean()
        axes[0].plot(g.index, g.values, marker=marker, color=color, label=name)
    axes[0].set_xlabel("Rollout horizon"); axes[0].set_ylabel("WIP NRMSE")
    axes[0].legend(frameon=False, fontsize=8)
    phys = sur.groupby("model")[["ood_physical_violation_pct", "monotonicity_violation_pct"]].mean().loc[["data_tcn", "pi_tcn"]]
    xloc = np.arange(2); width = 0.34
    axes[1].bar(xloc - width/2, phys.ood_physical_violation_pct, width, label="OOD physical violation (%)", color="#C00000")
    axes[1].bar(xloc + width/2, phys.monotonicity_violation_pct, width, label="Monotonicity violation (%)", color="#FFC000")
    axes[1].set_xticks(xloc, ["Data TCN", "PI-TCN"]); axes[1].set_ylabel("Physical-risk metric")
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) Temporal rollout"); axes[1].set_title("(b) Physical consistency")
    fig.tight_layout(); fig.savefig(out / "temporal_physics_evidence.png", dpi=220); plt.close(fig)

    if pol.empty:
        return
    order = ["heuristic", "deterministic_rh", "scenario_block", "scenario_rh"]
    means = pol.groupby("policy")[["inventory_cost", "backlog_cost", "wip_cost", "cleaning_cost"]].mean().loc[order]
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    means.plot(kind="bar", stacked=True, ax=ax, color=["#70AD47", "#C00000", "#5B9BD5", "#FFC000"])
    ax.set_ylabel("Mean cumulative cost"); ax.set_xlabel("")
    ax.legend(["Inventory", "Backlog", "WIP", "Cleaning"], ncol=4, frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(out / "closed_loop_cost.png", dpi=220); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiments/output")
    ap.add_argument("--train-trajectories", type=int, default=90)
    ap.add_argument("--val-trajectories", type=int, default=24)
    ap.add_argument("--test-trajectories", type=int, default=30)
    ap.add_argument("--trajectory-length", type=int, default=26)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--closed-loop-reps", type=int, default=5)
    ap.add_argument("--closed-loop-length", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--solver-time-limit", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--skip-closed-loop", action="store_true")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    print("Generating simulator trajectories...", flush=True)
    tr_raw = generate_trajectories(args.train_trajectories, args.trajectory_length, args.seed, "nominal")
    va_raw = generate_trajectories(args.val_trajectories, args.trajectory_length, args.seed + 100000, "mixed")
    te_raw = generate_trajectories(args.test_trajectories, args.trajectory_length, args.seed + 200000, "stress")
    kinds = ["current_mlp", "window_mlp", "data_tcn", "pi_tcn"]
    fitted: dict[str, list[FittedModel]] = {k: [] for k in kinds}
    sur_rows, rollout_rows, tr_windows, te_windows, te_targets = [], [], [], [], []
    for s in range(S):
        trx, try_, _ = windows(*tr_raw, s); vax, vay, _ = windows(*va_raw, s); tex, tey, _ = windows(*te_raw, s)
        tr_windows.append(trx); te_windows.append(tex); te_targets.append(tey)
        for kind in kinds:
            print(f"Training {kind}, stage={STAGES[s]}...", flush=True)
            fit = train_model(kind, s, trx, try_, vax, vay, args.epochs, args.seed)
            fitted[kind].append(fit)
            row = {"model": kind, "stage": STAGES[s], **surrogate_metrics(fit, tex, tey)}
            row["rollout_wip_nrmse"] = rollout_nrmse(fit, *te_raw)
            sur_rows.append(row)
            for h, value in enumerate(rollout_curve(fit, *te_raw), start=1):
                rollout_rows.append({"model": kind, "stage": STAGES[s], "horizon": h,
                                     "wip_nrmse": value})
    sur = pd.DataFrame(sur_rows)
    rollout_df = pd.DataFrame(rollout_rows)
    print("Fitting PWA approximation...", flush=True)
    pwa = fit_pwa(fitted["pi_tcn"], tr_windows)
    pwa_df = pwa_metrics(pwa, fitted["pi_tcn"], te_windows, te_targets)
    rows = []
    if not args.skip_closed_loop:
        print("Running closed-loop planning experiments...", flush=True)
        policies = ["heuristic", "deterministic_rh", "scenario_block", "scenario_rh"]
        for stress in (False, True):
            for rep in range(args.closed_loop_reps):
                path_seed = args.seed + 300000 + 1000 * int(stress) + rep
                for policy in policies:
                    print(f"  policy={policy}, stress={stress}, rep={rep + 1}", flush=True)
                    rows.append(evaluate_policy(policy, pwa, path_seed, stress,
                                                args.closed_loop_length, args.horizon,
                                                args.solver_time_limit))
    pol = pd.DataFrame(rows)
    sur.to_csv(out / "surrogate_metrics.csv", index=False)
    rollout_df.to_csv(out / "rollout_by_horizon.csv", index=False)
    pwa_df.to_csv(out / "pwa_metrics.csv", index=False)
    if not pol.empty:
        pol.to_csv(out / "policy_metrics.csv", index=False)
    plot_results(sur, rollout_df, pol, out)
    summary = {
        "synthetic_pilot": True,
        "config": vars(args),
        "surrogate_mean": sur.groupby("model").mean(numeric_only=True).round(4).to_dict("index"),
        "pwa_mean": pwa_df.mean(numeric_only=True).round(4).to_dict(),
        "policy_mean": pol.groupby("policy").mean(numeric_only=True).round(4).to_dict("index") if not pol.empty else {},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
