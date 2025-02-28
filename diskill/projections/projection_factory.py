"""
From https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/projections/projection_factory.py.
Adjusted in the sense that projections that are not used in Di-SkilL are removed
"""

from projections.base_projection_layer import BaseProjectionLayer
from projections.kl_projection_layer import KLProjectionLayer


def get_projection_layer(proj_type: str = "", **kwargs) -> BaseProjectionLayer:
    """
    Factory to generate the projection layers for all projections.
    Args:
        proj_type: One of None/' ', 'ppo', 'kl'
        **kwargs: arguments for projection layer

    Returns:

    """
    if not proj_type or proj_type.isspace() or proj_type.lower() in ["ppo"]:
        return BaseProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "kl":
        return KLProjectionLayer(proj_type, **kwargs)


    else:
        raise ValueError(
            f"Invalid projection type {proj_type}."
            f" Choose one of None/' ', 'ppo', 'kl'.")
