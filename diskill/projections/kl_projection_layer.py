"""
From https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/projections/kl_projection_layer.py.
Only Minor description changes and additional classes such ass KLProjectionGradFunctionCovOnly that are extending the
Projection opportunities. Other then mainly parameter renamings such that names are adjusted to the episodic RL case.
"""

import cpp_projection
import numpy as np
import torch as ch
from typing import Any, Tuple

from distributions.non_lin_conditional.abstract_gaussian_policy import AbstractGaussianPolicy
from projections.base_projection_layer import BaseProjectionLayer, mean_projection
from utils.projection_utils import gaussian_kl
from utils.torch_utils import get_numpy

MAX_EVAL = 1000


class KLProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, policy: AbstractGaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                 q: Tuple[ch.Tensor, ch.Tensor], eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Runs KL projection layer and constructs cholesky of covariance
        Args:
            **kwargs:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part

        Returns:
            mean, cov cholesky
        """
        mean, std = p
        old_mean, old_std = q

        mean_part, cov_part = gaussian_kl(policy, p, q)

        if not policy.contextual_std:
            # only project first one to reduce number of numerical optimizations
            std = std[:1]
            old_std = old_std[:1]
            cov_part = cov_part[:1]

        ################################################################################################################
        # project mean with closed form

        # proj_mean = ch.zeros_like(mean)
        # proj_mean_tmp = mean_projection(mean, old_mean, mean_part, eps)
        # is_nan = proj_mean_tmp.mean(-1).isnan()
        # if is_nan.any():
        #     proj_mean[is_nan] = old_mean[is_nan]
        # proj_mean[~is_nan] = proj_mean_tmp[~is_nan]
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ################################################################################################################
        # project cov with numeric optimization

        cov = policy.covariance(std)

        if policy.is_diag:
            old_cov = policy.covariance(old_std)
            try:
                proj_cov = KLProjectionGradFunctionDiagCovOnly.apply(cov.diagonal(dim1=-2, dim2=-1),
                                                                     old_cov.diagonal(dim1=-2, dim2=-1), eps_cov)
                proj_std = proj_cov.sqrt().diag_embed()
            except Exception as e:
                proj_std = old_std

        else:
            try:
                mask = cov_part > eps_cov
                proj_std = ch.zeros_like(std)
                proj_std[~mask] = std[~mask]
                if mask.any():
                    proj_cov = KLProjectionGradFunctionCovOnly.apply(cov, std.detach(), old_std, eps_cov)

                    # needs projection and projection failed
                    # mean propagates the nan values to the batch dimensions, in case any of entries is nan
                    is_nan = proj_cov.mean([-2, -1]).isnan() * mask
                    if is_nan.any():
                        proj_std[is_nan] = old_std[is_nan]
                        mask *= ~is_nan
                    proj_std[mask], failed_mask = ch.linalg.cholesky_ex(proj_cov[mask])
                    # check if any of the cholesky decompositions failed and keep old_std in that case
                    failed_mask = failed_mask.type(ch.bool)
                    if ch.any(failed_mask):
                        proj_std[failed_mask] = old_std[failed_mask]
            except Exception as e:
                # logging.error(e)
                # import traceback
                # import logging
                # logging.error(traceback.format_exc())
                print("Projection failed, taking old cholesky for projection.")
                proj_std = old_std
                raise e

        if not policy.contextual_std:
            # scale first std back to batchsize
            proj_std = proj_std.expand(mean.shape[0], -1, -1)

        return proj_mean, proj_std


class KLProjectionGradFunctionCovOnly(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=MAX_EVAL):
        if not KLProjectionGradFunctionCovOnly.projection_op:
            KLProjectionGradFunctionCovOnly.projection_op = \
                cpp_projection.BatchedCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        cov, chol, old_chol, eps_cov = args

        batch_shape = cov.shape[0]
        dim = cov.shape[-1]

        cov_np = get_numpy(cov)
        chol_np = get_numpy(chol)
        old_chol_np = get_numpy(old_chol)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        p_op = KLProjectionGradFunctionCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_chol_np, chol_np, cov_np)

        return cov.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_cov, = grad_outputs

        d_cov_np = get_numpy(d_cov)
        d_cov_np = np.atleast_2d(d_cov_np)

        df_stds = projection_op.backward(d_cov_np)
        df_stds = np.atleast_2d(df_stds)

        df_stds = d_cov.new(df_stds)

        # if (df_stds == 0.).all():
        # if ch.equal(df_stds, ch.eye(d_cov_np.shape[-1], dtype=df_stds.dtype)):
        return df_stds, None, None, None


class KLProjectionGradFunctionDiagCovOnly(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=MAX_EVAL):
        if not KLProjectionGradFunctionDiagCovOnly.projection_op:
            KLProjectionGradFunctionDiagCovOnly.projection_op = \
                cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        cov, old_cov_np, eps_cov = args

        batch_shape = cov.shape[0]
        dim = cov.shape[-1]

        cov_np = get_numpy(cov)
        old_cov_np = get_numpy(old_cov_np)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        # p_op = cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim)
        # ctx.proj = projection_op

        p_op = KLProjectionGradFunctionDiagCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_cov_np, cov_np)

        return cov.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_cov_np = get_numpy(d_std)
        d_cov_np = np.atleast_2d(d_cov_np)
        df_stds = projection_op.backward(d_cov_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None
