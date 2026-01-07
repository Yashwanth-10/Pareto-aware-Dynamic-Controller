"""
early_exit.py

A generalized, architecture-agnostic early-exit framework for PyTorch backbones.
Works with arbitrary nn.Module backbones and tensor shapes (only assumption: leading batch dimension).

Implements:
- ActivationSummaryModule: computes per-sample activation summaries (non-image-specific).
- EarlyExitController: simple rule-based or MLP controller mapping summaries + budget -> actions.
- EarlyExitWrapper: orchestrates sequential stages, summary extraction, controller decisions, and early exits.

Requires: Python 3.10+, PyTorch.
Only imports: torch, torch.nn as nn, torch.nn.functional as F, typing.

Example usage at the bottom demonstrates an identity backbone with two linear stages.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ActivationSummaryModule
# -----------------------------------------------------------------------------
class ActivationSummaryModule(nn.Module):
    """
    Compute a compact, architecture-agnostic summary vector from activations (and optional logits).

    The module computes per-sample statistics over all non-batch dimensions of `activations`:
      - act_mean: mean(abs(activations))
      - act_std: std(activations)
      - act_norm: L2 norm over flattened activation dims
      - sparsity: fraction of elements with abs(value) < sparsity_threshold

    If `logits` is provided and `num_classes` was given:
      - confidence: max softmax probability (with temperature scaling)
      - entropy: Shannon entropy of the softmax distribution

    Always includes:
      - progress: layer_id / max(total_layers - 1, 1)

    The output is a fixed-size summary vector of shape (batch_size, summary_dim).
    By default summary_dim == 6 and the layout is:
      - logits present: [confidence, entropy, act_mean, act_std, sparsity, progress]
      - logits absent:  [act_mean, act_std, act_norm, sparsity, progress, 0.0]

    Parameters
    ----------
    num_classes : int | None
        If provided, enables computing softmax-derived features when logits are present.
    sparsity_threshold : float
        Threshold to consider an activation element "near-zero" for sparsity computation.
    temperature : float
        Temperature used when softmax-ing logits to compute confidence and entropy.
    summary_dim : int
        Output dimensionality of the summary vector. Must be >= 6; if larger, remaining dims are zero-padded.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        sparsity_threshold: float = 0.1,
        temperature: float = 1.0,
        summary_dim: int = 6,
    ) -> None:
        super().__init__()
        if summary_dim < 6:
            raise ValueError("summary_dim must be at least 6 to support default features.")
        self.num_classes = num_classes
        self.sparsity_threshold = float(sparsity_threshold)
        self.temperature = float(temperature)
        self.summary_dim = int(summary_dim)

    def forward(
        self,
        activations: torch.Tensor,
        logits: Optional[torch.Tensor],
        layer_id: int,
        total_layers: int,
    ) -> torch.Tensor:
        """
        Compute summary features for a batch of activations.

        Parameters
        ----------
        activations : torch.Tensor
            Arbitrary tensor with leading batch dimension (batch_size, ...).
        logits : Optional[torch.Tensor]
            Either None or tensor of shape (batch_size, num_classes). If present and num_classes
            was given at construction, compute confidence and entropy.
        layer_id : int
            Index of the current layer/stage (0-based).
        total_layers : int
            Total number of layers/stages used to compute progress.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, summary_dim) with per-sample summaries.
        """
        if activations.dim() < 1:
            raise ValueError("activations must have at least a batch dimension.")
        batch_size = activations.shape[0]

        # Flatten non-batch dims -> (batch_size, n_features)
        if activations.dim() == 1:
            flat = activations.unsqueeze(1)  # shape (batch, 1)
        else:
            flat = activations.reshape(batch_size, -1)

        # Basic activation stats
        abs_flat = flat.abs()
        act_mean = abs_flat.mean(dim=1)  # mean of absolute activations
        act_std = flat.std(dim=1, unbiased=False)  # population std
        act_norm = torch.norm(flat, p=2, dim=1)  # L2 norm
        num_elems = flat.shape[1] if flat.shape[1] > 0 else 1
        sparsity = (abs_flat < self.sparsity_threshold).sum(dim=1).to(dtype=flat.dtype) / float(
            num_elems
        )

        # Progress: normalized position in the network [0..1]
        denom = max(total_layers - 1, 1)
        progress_val = float(layer_id) / float(denom)
        progress = torch.full((batch_size,), progress_val, device=activations.device, dtype=flat.dtype)

        # Logit-derived features
        confidence = torch.zeros((batch_size,), device=activations.device, dtype=flat.dtype)
        entropy = torch.zeros((batch_size,), device=activations.device, dtype=flat.dtype)
        if logits is not None and self.num_classes is not None:
            # Temperature-scaled softmax
            scaled = logits / (self.temperature + 1e-12)
            probs = F.softmax(scaled, dim=1)
            # confidence: max class prob
            confidence = probs.max(dim=1).values
            # entropy: -sum p log p
            eps = 1e-12
            clamped = torch.clamp(probs, min=eps)
            entropy = -(clamped * torch.log(clamped)).sum(dim=1)
            # consistent dtype already

        # Compose default 6-element vector depending on availability of logits
        if logits is not None and self.num_classes is not None:
            base = torch.stack([confidence, entropy, act_mean, act_std, sparsity, progress], dim=1)
        else:
            # When logits absent, place act_norm in position 2 and final slot 0.0
            zero_last = torch.zeros((batch_size,), device=activations.device, dtype=flat.dtype)
            base = torch.stack([act_mean, act_std, act_norm, sparsity, progress, zero_last], dim=1)

        # If user requested a larger summary_dim, pad with zeros; if smaller (shouldn't happen), crop.
        if self.summary_dim == 6:
            summary = base
        elif self.summary_dim > 6:
            pad = torch.zeros((batch_size, self.summary_dim - 6), device=activations.device, dtype=flat.dtype)
            summary = torch.cat([base, pad], dim=1)
        else:  # guarded by __init__, but keep defensive code
            summary = base[:, : self.summary_dim]

        return summary


# -----------------------------------------------------------------------------
# EarlyExitController
# -----------------------------------------------------------------------------
class EarlyExitController(nn.Module):
    """
    Maps summary features and a remaining budget to per-sample actions.

    Modes:
      - "rule": simple threshold-based rule using summary[:, 0] as confidence/proxy.
      - "mlp": small feed-forward network that outputs logits over actions.

    Actions:
      EXIT = 0
      CONTINUE = 1
      SKIP = 2 (optional; used only if num_actions==3)

    Parameters
    ----------
    summary_dim : int
        Dimensionality of the summary vectors provided to forward.
    hidden_dim : int
        Hidden dimension for the MLP mode.
    num_actions : int
        2 or 3 actions. If 2, SKIP is not present.
    mode : str
        "rule" or "mlp".
    exit_conf_threshold : float
        Threshold used in "rule" mode on summary[:, 0] to decide exit.
    """

    EXIT = 0
    CONTINUE = 1
    SKIP = 2

    def __init__(
        self,
        summary_dim: int,
        hidden_dim: int = 64,
        num_actions: int = 2,
        mode: str = "mlp",
        exit_conf_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        if mode not in {"rule", "mlp"}:
            raise ValueError("mode must be 'rule' or 'mlp'")
        if num_actions not in {2, 3}:
            raise ValueError("num_actions must be 2 or 3")
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.mode = mode
        self.exit_conf_threshold = float(exit_conf_threshold)

        # MLP components (only used if mode == "mlp")
        if self.mode == "mlp":
            self.fc1 = nn.Linear(self.summary_dim + 1, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(
        self,
        summary: torch.Tensor,
        remaining_budget: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """
        Decide action for each sample.

        Parameters
        ----------
        summary : torch.Tensor
            (batch_size, summary_dim) summary features.
        remaining_budget : torch.Tensor | float
            Either a scalar float or a tensor (batch_size,). If scalar, it is broadcasted.

        Returns
        -------
        torch.Tensor
            (batch_size,) int64 tensor with actions in {EXIT, CONTINUE, SKIP}.
        """
        if summary.dim() != 2 or summary.shape[1] != self.summary_dim:
            raise ValueError(f"summary must be shape (batch, {self.summary_dim})")

        batch_size = summary.shape[0]
        device = summary.device
        dtype = summary.dtype

        # Normalize remaining_budget to tensor of shape (batch_size,)
        if isinstance(remaining_budget, torch.Tensor):
            rb = remaining_budget.to(device=device, dtype=dtype)
            if rb.dim() == 0:
                rb = rb.unsqueeze(0).expand(batch_size)
            elif rb.dim() == 1 and rb.shape[0] == batch_size:
                pass
            else:
                # try broadcasting
                rb = rb.reshape(-1).to(device=device, dtype=dtype)
                if rb.numel() == 1:
                    rb = rb.expand(batch_size)
                elif rb.numel() == batch_size:
                    rb = rb
                else:
                    raise ValueError("remaining_budget has incompatible shape")
        else:
            rb = torch.full((batch_size,), float(remaining_budget), device=device, dtype=dtype)

        if self.mode == "rule":
            first_feat = summary[:, 0]
            # Condition: high confidence OR budget depleted -> EXIT
            cond_exit = (first_feat >= self.exit_conf_threshold) | (rb <= 0.0)
            actions = torch.where(cond_exit, torch.full_like(first_feat, self.EXIT, dtype=torch.int64),
                                  torch.full_like(first_feat, self.CONTINUE, dtype=torch.int64))
            return actions.to(torch.int64)

        # MLP mode
        x = torch.cat([summary, rb.unsqueeze(1)], dim=1)  # (batch, summary_dim+1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)  # (batch, num_actions)
        actions = torch.argmax(logits, dim=1).to(torch.int64)
        return actions


# -----------------------------------------------------------------------------
# EarlyExitWrapper
# -----------------------------------------------------------------------------
class EarlyExitWrapper(nn.Module):
    """
    Wrap an arbitrary backbone and manage early-exit execution over a sequence of stages.

    The wrapper expects:
      - stages: List[Callable[[torch.Tensor], torch.Tensor]] -- sequence of processing functions.
      - logits_fns: List[Optional[Callable[[torch.Tensor], torch.Tensor]]] -- same length; if not None,
        logits_fn(activations) returns logits tensor (batch, num_classes) for that stage.

    Operation:
      For each stage:
        - x = stage(x)
        - logits = logits_fn(x) if provided else None
        - summary = summary_module(x, logits, layer_id, num_layers)
        - actions = controller(summary, remaining_budget)
        - remaining_budget -= per_layer_cost
        - if exit_on_first and all(actions == EXIT) and logits is not None: return logits (and optionally all_logits)

    Parameters
    ----------
    backbone : nn.Module
        The overall backbone model (kept for reference / optional use).
    stages : List[Callable[[torch.Tensor], torch.Tensor]]
        Callables that transform activations through the network.
    logits_fns : List[Optional[Callable[[torch.Tensor], torch.Tensor]]]
        Per-stage logits-producing callables or None.
    num_layers : int
        Total number of stages (used to compute progress).
    summary_module : ActivationSummaryModule
        Module to compute summaries from activations/logits.
    controller : EarlyExitController
        Module to decide actions from summaries + budget.
    per_layer_cost : float
        Amount to subtract from remaining budget after each stage execution.
    exit_on_first : bool
        If True and all samples vote EXIT and logits present, immediately exit the whole batch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        stages: List[Callable[[torch.Tensor], torch.Tensor]],
        logits_fns: List[Optional[Callable[[torch.Tensor], torch.Tensor]]],
        num_layers: int,
        summary_module: ActivationSummaryModule,
        controller: EarlyExitController,
        per_layer_cost: float = 1.0,
        exit_on_first: bool = True,
    ) -> None:
        super().__init__()
        if len(stages) != len(logits_fns):
            raise ValueError("stages and logits_fns must have the same length.")
        self.backbone = backbone
        self.stages = stages
        self.logits_fns = logits_fns
        self.num_layers = int(num_layers)
        self.summary_module = summary_module
        self.controller = controller
        self.per_layer_cost = float(per_layer_cost)
        self.exit_on_first = bool(exit_on_first)

    def forward(
        self,
        x: torch.Tensor,
        budget: float = 1.0,
        return_all_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Run the wrapped model with early-exit logic.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with leading batch dimension.
        budget : float
            Initial budget (scalar) applied per-batch.
        return_all_logits : bool
            If True, also return the list of logits produced at stages where logits_fn != None.

        Returns
        -------
        torch.Tensor or (torch.Tensor, List[torch.Tensor])
            The logits returned by the exit stage (or final stage). If return_all_logits is True,
            also returns the list of all logits seen (in chronological order).
        """
        if x.dim() < 1:
            raise ValueError("Input x must have a batch dimension.")
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        remaining_budget = torch.full((batch_size,), float(budget), device=device, dtype=torch.float32)
        all_logits: List[torch.Tensor] = []
        last_logits: Optional[torch.Tensor] = None

        activations = x
        for layer_id, (stage, logits_fn) in enumerate(zip(self.stages, self.logits_fns)):
            # Execute stage
            activations = stage(activations)

            # Compute logits if provided
            logits = logits_fn(activations) if logits_fn is not None else None
            if logits is not None:
                last_logits = logits

            # Compute summary and controller actions
            summary = self.summary_module(activations, logits, layer_id, self.num_layers)
            actions = self.controller(summary, remaining_budget)

            # Optionally track logits
            if logits is not None and return_all_logits:
                all_logits.append(logits)

            # Decrease budget after decision
            remaining_budget = remaining_budget - float(self.per_layer_cost)

            # Simple batch-level exit policy
            if logits is not None and self.exit_on_first:
                # If all samples voted for EXIT, return immediately
                if torch.all(actions == int(self.controller.EXIT)):
                    if return_all_logits:
                        return logits, all_logits
                    return logits

            # TODO: SKIP behavior and per-sample routing not implemented in this simple wrapper.

        # After all stages: if some logits were produced, return the last logits
        if last_logits is not None:
            if return_all_logits:
                # Ensure last logits are included if user requested all_logits but last stage's logits_fn was None:
                # we already appended logits where available.
                return last_logits, all_logits
            return last_logits

        # No logits ever produced -> ambiguous. Require that last stage provides logits.
        raise RuntimeError("No logits were produced by any stage. Ensure at least one logits_fn is not None.")


# -----------------------------------------------------------------------------
# Example usage (architecture-agnostic demo)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Small demonstration using vector inputs (batch, d) and two simple linear stages.
    torch.manual_seed(0)

    batch = 8
    d_in = 16
    hidden = 32
    num_classes = 4

    # Simple backbone reference (not used directly by stages in this demo)
    backbone = nn.Identity()

    # Define two stage modules operating on vectors
    stage1 = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
    stage2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

    # logits function only after stage2
    logits_head = nn.Linear(hidden, num_classes)

    # Wrap stages as callables
    stages = [lambda x, m=stage1: m(x), lambda x, m=stage2: m(x)]
    logits_fns = [None, lambda x, h=logits_head: h(x)]

    # Create summary and controller
    summary_module = ActivationSummaryModule(num_classes=num_classes, sparsity_threshold=0.05, temperature=1.0)
    controller = EarlyExitController(summary_dim=6, mode="rule", exit_conf_threshold=0.8)

    # Create wrapper
    wrapper = EarlyExitWrapper(
        backbone=backbone,
        stages=stages,
        logits_fns=logits_fns,
        num_layers=len(stages),
        summary_module=summary_module,
        controller=controller,
        per_layer_cost=0.5,
        exit_on_first=True,
    )

    # Dummy input
    x = torch.randn(batch, d_in)

    # Run with budget high (so rule depends on confidence)
    logits, all_logits = wrapper(x, budget=2.0, return_all_logits=True)
    print("Returned logits shape:", logits.shape)
    print("Number of logits snapshots collected:", len(all_logits))
    # Determine which stage produced the returned logits by index in all_logits
    if len(all_logits) > 0:
        produced_at = len(all_logits) - 1
        print(f"Exit produced at logits snapshot index: {produced_at}")
    else:
        print("No logits were collected (unexpected).")

    # Run with tiny budget to force exit via budget depletion rule
    logits2, all_logits2 = wrapper(x, budget=0.0, return_all_logits=True)
    print("Returned logits shape (low budget):", logits2.shape)
    print("Number of logits snapshots collected (low budget):", len(all_logits2))
