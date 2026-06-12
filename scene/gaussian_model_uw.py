"""
scene/gaussian_model_uw.py
--------------------------
UnderwaterGaussianModel — extends base 4DGS GaussianModel with one
per-Gaussian UW attribute:

    _sigma : (N, 1) transient score logit

CHANGES vs previous version:
    REMOVED: _beta_D and _beta_B per-Gaussian tensors (vestigial — they
             didn't drive rendering; physics_planes.PhysicsPlanes does).
             Their presence was confusing and the optimizer was wasting
             compute updating them.

    KEPT: _sigma machinery for transient handling (low weight, for ablation).
          Even if σ doesn't activate strongly, it doesn't hurt to keep.

The per-Gaussian beta values, if you ever decide you need them, can be
recovered by querying physics_planes at each Gaussian's camera-space depth.
But none of the new losses or rendering paths need this.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel


class UnderwaterGaussianModel(GaussianModel):

    def __init__(self, sh_degree, args, water_type="II", sigma_init=-2.0):
        super().__init__(sh_degree, args)
        self.water_type = water_type
        self.sigma_init = sigma_init
        self._uw_initialised = False
        self._uw_groups_added = False

    # ── helpers ────────────────────────────────────────────────────────────

    def _init_uw(self):
        """
        Initialize per-Gaussian σ tensor.
        Called after create_from_pcd or load_ply when N is known.
        """
        N = self._xyz.shape[0]
        device = self._xyz.device

        # σ logit initialized to sigma_init (default -2.0 → sigmoid ≈ 0.12).
        # That gives all Gaussians a low transient score at init: assume static.
        self._sigma = nn.Parameter(
            torch.full((N, 1), self.sigma_init, device=device))

        self._uw_initialised = True

    @staticmethod
    def _strip_uw_optimizer_groups(opt_state):
        if not isinstance(opt_state, dict) or "param_groups" not in opt_state:
            return opt_state
        keep_groups = []
        keep_params = set()
        for group in opt_state["param_groups"]:
            if group.get("name", "") in ("sigma", "physics_planes"):
                continue
            keep_groups.append(dict(group))
            keep_params.update(group.get("params", []))
        state = {
            k: v for k, v in opt_state.get("state", {}).items()
            if k in keep_params
        }
        stripped = dict(opt_state)
        stripped["param_groups"] = keep_groups
        stripped["state"] = state
        return stripped

    @classmethod
    def _strip_uw_from_base_capture(cls, model_args):
        if not isinstance(model_args, tuple) or len(model_args) < 13:
            return model_args
        model_args = list(model_args)
        model_args[12] = cls._strip_uw_optimizer_groups(model_args[12])
        return tuple(model_args)

    # ── overrides for setup hooks ──────────────────────────────────────────

    def create_from_pcd(self, pcd, spatial_lr_scale, maxtime):
        super().create_from_pcd(pcd, spatial_lr_scale, maxtime)
        self._init_uw()

    def load_ply(self, path):
        super().load_ply(path)
        sigma_path = os.path.join(os.path.dirname(path), "sigma.pth")
        if os.path.exists(sigma_path):
            sigma = torch.load(sigma_path, map_location="cuda")
            if sigma.shape[0] == self._xyz.shape[0]:
                self._sigma = nn.Parameter(
                    sigma.to(device=self._xyz.device, dtype=self._xyz.dtype)
                    .requires_grad_(True))
                self._uw_initialised = True
                return
        self._init_uw()

    def save_ply(self, path):
        super().save_ply(path)
        if (self._uw_initialised and hasattr(self, "_sigma")
                and self._sigma.shape[0] == self._xyz.shape[0]):
            torch.save(self._sigma.detach().cpu(),
                       os.path.join(os.path.dirname(path), "sigma.pth"))

    def capture(self):
        base = self._strip_uw_from_base_capture(super().capture())
        return {
            "base": base,
            "sigma": (self._sigma.detach().cpu()
                      if self._uw_initialised and hasattr(self, "_sigma")
                      else None),
            "uw_version": 1,
        }

    def restore(self, model_args, training_args):
        sigma = None
        if isinstance(model_args, dict) and "base" in model_args:
            sigma = model_args.get("sigma", None)
            model_args = model_args["base"]

        model_args = self._strip_uw_from_base_capture(model_args)
        super().restore(model_args, training_args)

        if sigma is not None and sigma.shape[0] == self._xyz.shape[0]:
            self._sigma = nn.Parameter(
                sigma.to(device=self._xyz.device, dtype=self._xyz.dtype)
                .requires_grad_(True))
            self._uw_initialised = True
        else:
            self._init_uw()

        self._uw_groups_added = False

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def get_sigma(self):
        """
        Activated transient score with a smooth numerical margin.

        Do not hard-clamp this value. A previous version used clamp(0.05, 0.95);
        when sparsity pushed logits below the lower clamp, gradients through σ
        vanished and every Gaussian stayed at exactly 0.05.

        Returns (N,) tensor.
        """
        sigma = torch.sigmoid(self._sigma).squeeze(-1)
        return 0.01 + 0.98 * sigma

    def deformation_norm(self, deformation):
        """
        Compute L2 norm of deformation per Gaussian.
        Args:
            deformation : (N, 3) deformation delta
        Returns:
            (N,) norms
        """
        return deformation.norm(dim=-1)

    # ── parameter groups for optimizer ─────────────────────────────────────

    def training_setup_uw(self, lr_sigma=5e-3):
        """
        Return extra optimizer param groups for UW-specific parameters.
        Call AFTER base training_setup(opt) has run.

        Args:
            lr_sigma : learning rate for σ params

        Returns:
            list of dicts ready to pass to optimizer.add_param_group
        """
        # Reinit if shape mismatch (after densification)
        if not self._uw_initialised or self._sigma.shape[0] != self._xyz.shape[0]:
            self._init_uw()
        return [
            {"params": [self._sigma], "lr": lr_sigma, "name": "sigma"},
        ]

    # ── densification ──────────────────────────────────────────────────────

    def densification_postfix(self, new_xyz, new_features_dc,
                                new_features_rest, new_opacities,
                                new_scaling, new_rotation,
                                new_deformation_table):
        """
        Extend σ tensor when new Gaussians are added during densification.
        Base class handles standard params + their optimizer states.
        We add σ entries for the new Gaussians.
        """
        # Base handles standard params + optimizer extension
        super().densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation,
            new_deformation_table)

        # If σ not initialized yet (shouldn't happen normally), init and return
        if not self._uw_initialised:
            self._init_uw()
            return

        N_new = new_xyz.shape[0]
        device = self._xyz.device

        # New σ entries: same init as the original (sigma_init = -2.0)
        new_sigma = torch.full((N_new, 1), self.sigma_init, device=device)

        # Concatenate old + new σ
        old_sigma = self._sigma.data
        new_sigma_full = torch.cat([old_sigma, new_sigma], dim=0)

        # Update the optimizer param group for σ
        found_sigma_group = False
        for group in self.optimizer.param_groups:
            if group.get("name") == "sigma":
                found_sigma_group = True
                stored = self.optimizer.state.get(group["params"][0], None)
                if stored is not None and "exp_avg" in stored:
                    # Extend Adam state for new σ entries (zero momentum)
                    stored["exp_avg"] = torch.cat(
                        [stored["exp_avg"], torch.zeros_like(new_sigma)], dim=0)
                    stored["exp_avg_sq"] = torch.cat(
                        [stored["exp_avg_sq"], torch.zeros_like(new_sigma)], dim=0)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    new_sigma_full.requires_grad_(True))
                if stored is not None:
                    self.optimizer.state[group["params"][0]] = stored
                self._sigma = group["params"][0]
                break
        if not found_sigma_group:
            self._sigma = nn.Parameter(new_sigma_full.requires_grad_(True))

    # ── pruning ────────────────────────────────────────────────────────────

    def prune_points(self, mask):
        """
        Prune Gaussians by mask (True = remove).
        Base class handles all standard + UW params since they're in optimizer.
        We update our reference to the pruned σ tensor.
        """
        super().prune_points(mask)

        if self._uw_initialised:
            found_sigma_group = False
            for group in self.optimizer.param_groups:
                if group.get("name") == "sigma":
                    found_sigma_group = True
                    self._sigma = group["params"][0]
                    break
            if (not found_sigma_group
                    and hasattr(self, "_sigma")
                    and self._sigma.shape[0] == mask.shape[0]):
                keep = ~mask
                self._sigma = nn.Parameter(
                    self._sigma.detach()[keep].clone().requires_grad_(True))

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        Underwater fine training can make opacity temporarily unreliable while
        the medium/color split is settling. Cap pruning so a bad interval cannot
        collapse the whole representation.
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        n_total = prune_mask.numel()
        n_prune = int(prune_mask.sum().item())
        min_remaining = int(getattr(self, "min_points_after_prune", 20000))
        max_fraction = float(getattr(self, "max_prune_fraction", 0.25))
        max_prune = max(0, int(max_fraction * n_total))
        allowed = min(n_prune, max_prune, max(0, n_total - min_remaining))

        if allowed <= 0:
            return

        if allowed < n_prune:
            candidate_idx = torch.nonzero(prune_mask, as_tuple=False).flatten()
            opacity = self.get_opacity.squeeze()[candidate_idx]
            selected = torch.topk(opacity, k=allowed, largest=False).indices
            capped_mask = torch.zeros_like(prune_mask)
            capped_mask[candidate_idx[selected]] = True
            prune_mask = capped_mask

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Override base to skip UW-specific param groups during base densification.
        σ is handled separately in densification_postfix above.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) != 1:
                continue
            name = group.get("name", "")
            # Skip UW-specific groups — handled separately
            if name in ("sigma", "physics_planes"):
                continue
            if name not in tensors_dict:
                continue

            extension_tensor = tensors_dict[name]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"],
                     torch.zeros_like(extension_tensor)),
                    dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0)
                    .requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0)
                    .requires_grad_(True))

            optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors
