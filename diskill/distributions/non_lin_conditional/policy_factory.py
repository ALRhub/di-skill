"""
https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/models/policy/policy_factory.py.
Only needed classes are kept and additional policy classes belonging to the Di-SkilL project are added.
"""

import torch as ch

from distributions.non_lin_conditional.gaussian_policy_diag import GaussianPolicyDiag
from distributions.non_lin_conditional.gaussian_policy_full import GaussianPolicyFull
from distributions.non_lin_conditional.softmax import GatingNetwork, ContextDistrNetwork


def get_policy_network(policy_type, proj_type, scaled_std: bool = False, device: ch.device = "cpu", dtype=ch.float32,
                       **kwargs):
    """
    Policy network factory for generating the required Gaussian policy model.
    Args:
        policy_type: 'full' or 'diag' covariance
        proj_type: Which projection is used.
        device: Torch device
        dtype: Torch dtype
        **kwargs: Policy arguments

    Returns:
        Gaussian Policy instance
    """

    if policy_type == "full":
        policy = GaussianPolicyFull(device=device, **kwargs)
    elif policy_type == "diag":
        policy = GaussianPolicyDiag(device=device, **kwargs)
    elif policy_type == 'softmax':
        policy = GatingNetwork(device=device, **kwargs)
    elif policy_type == 'ctxt_softmax':
        policy = ContextDistrNetwork(device=device, **kwargs)
    else:
        raise ValueError(f"Invalid policy type {policy_type}. Select one of 'full', 'diag'.")

    return policy.to(device, dtype)
