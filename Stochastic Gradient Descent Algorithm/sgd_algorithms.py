
"""
sgd_algorithms.py
Written by Jingwen Feng

A small, dependency-free module with:
- A generic SGD loop
- A linear regression (w, b) squared-loss example
- Deterministic "four-point" example that matches the user's handwritten notes
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

Number = float

@dataclass
class SampleGrad:
    x: Number
    y: Number
    y_hat: Number
    residual: Number
    grad_w: Number
    grad_b: Number

@dataclass
class StepTrace:
    step: int
    batch: List[Tuple[Number, Number]]
    eval_at: Tuple[Number, Number]           # (w, b) BEFORE update
    sample_grads: List[SampleGrad]
    batch_avg_grad: Tuple[Number, Number]    # (Dw, Db)
    eta: Number
    updated_params: Tuple[Number, Number]    # (w, b) AFTER update

def squared_loss_gradients(w: Number, b: Number, x: Number, y: Number) -> SampleGrad:
    """
    For loss L = (w*x + b - y)^2
    dL/dw = 2*(w*x + b - y)*x
    dL/db = 2*(w*x + b - y)
    """
    y_hat = w * x + b
    r = y_hat - y
    grad_w = 2.0 * r * x
    grad_b = 2.0 * r
    return SampleGrad(x=x, y=y, y_hat=y_hat, residual=r, grad_w=grad_w, grad_b=grad_b)

def sgd_linear_regression_fixed_batches(
    init_w: Number,
    init_b: Number,
    batches: List[List[Tuple[Number, Number]]],
    eta: Number,
) -> List[StepTrace]:
    """
    Runs SGD for linear regression with EXACT, user-specified batches.
    Returns a detailed trace per step that matches the handwritten notes.
    """
    w, b = init_w, init_b
    traces: List[StepTrace] = []
    for k, batch in enumerate(batches, start=1):
        # compute per-sample grads at current params
        grads = [squared_loss_gradients(w, b, x, y) for (x, y) in batch]
        # average
        Dw = sum(g.grad_w for g in grads) / len(grads)
        Db = sum(g.grad_b for g in grads) / len(grads)
        # update
        new_w = w - eta * Dw
        new_b = b - eta * Db

        traces.append(
            StepTrace(
                step=k,
                batch=batch,
                eval_at=(w, b),
                sample_grads=grads,
                batch_avg_grad=(Dw, Db),
                eta=eta,
                updated_params=(new_w, new_b),
            )
        )
        w, b = new_w, new_b

    return traces

def averaged_params_from_trace(traces: List[StepTrace]) -> Tuple[Number, Number]:
    """
    Computes the average of the parameter vectors AFTER each update:
    bar_w = (w^(1)+...+w^(K))/K  and same for b.
    """
    if not traces:
        return (0.0, 0.0)
    sum_w = sum(t.updated_params[0] for t in traces)
    sum_b = sum(t.updated_params[1] for t in traces)
    K = len(traces)
    return (sum_w / K, sum_b / K)

def four_point_example_trace(eta: Number = 0.1) -> Dict[str, Any]:
    """
    Reproduces the exact four-step example from the notes:

    Data points: (1,3), (2,5), (3,7), (4,9) on y = 2x + 1
    Batches (size 2) in order:
      1) {(1,3), (3,7)}
      2) {(2,5), (4,9)}
      3) {(1,3), (4,9)}
      4) {(2,5), (3,7)}
    Start at (w,b) = (0,0).
    """
    batches = [
        [(1.0, 3.0), (3.0, 7.0)],
        [(2.0, 5.0), (4.0, 9.0)],
        [(1.0, 3.0), (4.0, 9.0)],
        [(2.0, 5.0), (3.0, 7.0)],
    ]
    trace = sgd_linear_regression_fixed_batches(
        init_w=0.0, init_b=0.0, batches=batches, eta=eta
    )
    avg_w, avg_b = averaged_params_from_trace(trace)
    # Return a JSON-like dict for convenience
    return {
        "traces": [asdict(t) for t in trace],
        "averaged_params": {"w": avg_w, "b": avg_b},
    }
