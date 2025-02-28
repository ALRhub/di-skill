from gym.envs.registration import register
from fancy_gym.envs import mujoco, DEFAULT_BB_DICT_DMP, DEFAULT_BB_DICT_ProMP, \
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS, DEFAULT_BB_DICT_ProDMP
from copy import deepcopy
import numpy as np

from contextual_envs.gym_wrappers.contextual_beerpong import ContextualBeerPongEnvWrapper
from contextual_envs.gym_wrappers.contextual_hj import ContextualHopperJumpEnvWrapper
from contextual_envs.gym_wrappers.contextual_mini_golf_wrapper import ContextualMiniGolfEnvWrapper, \
    ContextualMiniGolfOneObsEnvWrapper
from contextual_envs.gym_wrappers.contextual_reacher import Contextual5LinkReacherEnvWrapper, \
    Contextual5LinkLEFTReacherEnvWrapper
from contextual_envs.gym_wrappers.contextual_box_pushing_wrapper import ContextualBoxPushingEnvWrapper, \
    ContextualBoxPushingRotInvEnvWrapper, ContextualBoxPushingRndm2Rndm, ContextualBoxPushingRndm2RndmContextSampler, \
    ContextualBoxPushingObstacleEnvWrapper
from contextual_envs.gym_wrappers.contextual_table_tennis_wrapper import ContextualTableTennisEnvWrapper, \
    ContextualTableTennisVelEnvWrapper

## ReacherNd
_versions = ["Reacher5d-v0", "Reacher7d-v0", "Reacher5dSparse-v0", "Reacher7dSparse-v0"]
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}DMP-{_name[1]}'
    kwargs_dict_reacher_dmp = deepcopy(DEFAULT_BB_DICT_DMP)
    kwargs_dict_reacher_dmp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_reacher_dmp['wrappers'].append(Contextual5LinkReacherEnvWrapper)
    kwargs_dict_reacher_dmp['phase_generator_kwargs']['alpha_phase'] = 2
    kwargs_dict_reacher_dmp['name'] = _v
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_reacher_dmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["DMP"].append(_env_id)

    _env_id = f'VSL{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_reacher_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_reacher_promp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_reacher_promp['wrappers'].append(Contextual5LinkReacherEnvWrapper)
    kwargs_dict_reacher_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_reacher_promp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProDMP-{_name[1]}'
    kwargs_dict_reacher_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_reacher_prodmp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_reacher_prodmp['wrappers'].append(Contextual5LinkReacherEnvWrapper)
    kwargs_dict_reacher_prodmp['name'] = _v
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_reacher_prodmp['basis_generator_kwargs']['num_basis'] = 5
    kwargs_dict_reacher_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 3
    kwargs_dict_reacher_prodmp['phase_generator_kwargs']['alpha_phase'] = 3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['disable_goal'] = False
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['relative_goal'] = True
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_reacher_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

### Reacher 5d only on left side
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}LEFTProDMP-{_name[1]}'
    kwargs_dict_reacher_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_reacher_prodmp['wrappers'].append(mujoco.reacher.MPWrapper)
    kwargs_dict_reacher_prodmp['wrappers'].append(Contextual5LinkLEFTReacherEnvWrapper)
    kwargs_dict_reacher_prodmp['name'] = _v
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_reacher_prodmp['basis_generator_kwargs']['num_basis'] = 5
    kwargs_dict_reacher_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 3
    kwargs_dict_reacher_prodmp['phase_generator_kwargs']['alpha_phase'] = 3
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['disable_goal'] = False
    kwargs_dict_reacher_prodmp['trajectory_generator_kwargs']['relative_goal'] = True
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_reacher_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

## Box Pushing
_versions = ['BoxPushingDense-v0', 'BoxPushingDenseRotInv-v0',
             'BoxPushingTemporalSparse-v0', 'BoxPushingTemporalSparseRotInv-v0',
             'BoxPushingTemporalSpatialSparse-v0', 'BoxPushingTemporalSpatialSparseRotInv-v0',
             'BoxPushingTemporalSparseNoGuidanceRotInv-v0', 'BoxPushingTemporalSparseNoGuidanceAtAllRotInv-v0',
             'BoxPushingDenseRnd2Rnd-v0', 'BoxPushingTemporalSparseRnd2Rnd-v0',
             'BoxPushingTemporalSparseNotInclinedInit-v0',
             'BoxPushingObstacleDense-v0', 'BoxPushingObstacleTemporalSparse-v0',
             'BoxPushingObstacleTemporalSpatialSparse-v0',
             '_ContextSamplerBoxPushingRndm2Rndm-v0']

for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProDMP-{_name[1]}'
    kwargs_dict_box_pushing_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_box_pushing_prodmp['wrappers'].append(mujoco.box_pushing.MPWrapper)
    if _name[0][-6:] == 'RotInv':
        kwargs_dict_box_pushing_prodmp['wrappers'].append(ContextualBoxPushingRotInvEnvWrapper)
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    elif _name[0][-7:] == 'Rnd2Rnd':
        kwargs_dict_box_pushing_prodmp['wrappers'].append(ContextualBoxPushingRndm2Rndm)
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    elif _name[0][10:18] == 'Obstacle':
        kwargs_dict_box_pushing_prodmp['wrappers'].append(ContextualBoxPushingObstacleEnvWrapper)
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.8
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.8
    else:
        kwargs_dict_box_pushing_prodmp['wrappers'].append(ContextualBoxPushingEnvWrapper)
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    if _name[0][:15] == '_ContextSampler':
        kwargs_dict_box_pushing_prodmp['name'] = 'BoxPushingTemporalSparseRnd2Rnd-v0'
        kwargs_dict_box_pushing_prodmp['wrappers'][-1] = ContextualBoxPushingRndm2RndmContextSampler
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    else:
        kwargs_dict_box_pushing_prodmp['name'] = _v
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
        kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_scale'] = 0.3
    kwargs_dict_box_pushing_prodmp['controller_kwargs']['p_gains'] = 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.])
    kwargs_dict_box_pushing_prodmp['controller_kwargs']['d_gains'] = 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.])

    kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['goal_offset'] = 1.0
    kwargs_dict_box_pushing_prodmp['basis_generator_kwargs']['num_basis'] = 4
    kwargs_dict_box_pushing_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 3
    kwargs_dict_box_pushing_prodmp['phase_generator_kwargs']['alpha_phase'] = 3
    kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['disable_goal'] = False
    kwargs_dict_box_pushing_prodmp['trajectory_generator_kwargs']['relative_goal'] = False
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_box_pushing_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

_versions = ['TableTennis4D-v0', 'TableTennis5D-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProDMP-{_name[1]}'
    kwargs_dict_tt_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_tt_prodmp['wrappers'].append(mujoco.table_tennis.TT_MPWrapper)
    if _name[0] == 'TableTennis5D':
        kwargs_dict_tt_prodmp['wrappers'].append(ContextualTableTennisVelEnvWrapper)
    else:
        kwargs_dict_tt_prodmp['wrappers'].append(ContextualTableTennisEnvWrapper)
    kwargs_dict_tt_prodmp['name'] = _v
    kwargs_dict_tt_prodmp['controller_kwargs']['p_gains'] = 0.5 * np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0])
    kwargs_dict_tt_prodmp['controller_kwargs']['d_gains'] = 0.5 * np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1])
    kwargs_dict_tt_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.7
    kwargs_dict_tt_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_tt_prodmp['trajectory_generator_kwargs']['relative_goal'] = True
    kwargs_dict_tt_prodmp['trajectory_generator_kwargs']['disable_goal'] = True
    kwargs_dict_tt_prodmp['phase_generator_kwargs']['tau_bound'] = [0.56, 1.5]
    kwargs_dict_tt_prodmp['phase_generator_kwargs']['delay_bound'] = [0.05, 0.6]
    kwargs_dict_tt_prodmp['phase_generator_kwargs']['learn_tau'] = True
    kwargs_dict_tt_prodmp['phase_generator_kwargs']['learn_delay'] = True
    kwargs_dict_tt_prodmp['basis_generator_kwargs']['num_basis'] = 3
    kwargs_dict_tt_prodmp['basis_generator_kwargs']['alpha'] = 25.
    kwargs_dict_tt_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 3
    kwargs_dict_tt_prodmp['phase_generator_kwargs']['alpha_phase'] = 3
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_tt_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)


_versions = ['MiniGolf-v0', 'MiniGolf-v1', 'MiniGolf-v2']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProDMP-{_name[1]}'
    kwargs_dict_minigolf_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)

    if _name[1] == 'v2':
        kwargs_dict_minigolf_prodmp['wrappers'].append(mujoco.mini_golf.MiniGolfOneObsMPWrapper)
        kwargs_dict_minigolf_prodmp['wrappers'].append(ContextualMiniGolfOneObsEnvWrapper)
    else:
        kwargs_dict_minigolf_prodmp['wrappers'].append(mujoco.mini_golf.MiniGolfMPWrapper)
        kwargs_dict_minigolf_prodmp['wrappers'].append(ContextualMiniGolfEnvWrapper)
    kwargs_dict_minigolf_prodmp['name'] = _v
    kwargs_dict_minigolf_prodmp['controller_kwargs']['p_gains'] = 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.])
    kwargs_dict_minigolf_prodmp['controller_kwargs']['d_gains'] = 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.])
    kwargs_dict_minigolf_prodmp['trajectory_generator_kwargs']['weights_scale'] = 0.3
    kwargs_dict_minigolf_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_minigolf_prodmp['trajectory_generator_kwargs']['relative_goal'] = True
    kwargs_dict_minigolf_prodmp['trajectory_generator_kwargs']['disable_goal'] = True
    kwargs_dict_minigolf_prodmp['phase_generator_kwargs']['tau_bound'] = [0.45, 1.7]
    kwargs_dict_minigolf_prodmp['phase_generator_kwargs']['learn_tau'] = True
    kwargs_dict_minigolf_prodmp['phase_generator_kwargs']['learn_delay'] = False
    kwargs_dict_minigolf_prodmp['basis_generator_kwargs']['num_basis'] = 4
    kwargs_dict_minigolf_prodmp['basis_generator_kwargs']['alpha'] = 25.
    kwargs_dict_minigolf_prodmp['basis_generator_kwargs']['basis_bandwidth_factor'] = 3
    kwargs_dict_minigolf_prodmp['phase_generator_kwargs']['alpha_phase'] = 3
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_minigolf_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)


_versions = ['HopperJumpSparse-v0']
for _v in _versions:
    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProMP-{_name[1]}'
    kwargs_dict_hj_promp = deepcopy(DEFAULT_BB_DICT_ProMP)
    kwargs_dict_hj_promp['wrappers'].append(mujoco.hopper_jump.MPWrapper)
    kwargs_dict_hj_promp['wrappers'].append(ContextualHopperJumpEnvWrapper)
    kwargs_dict_hj_promp['name'] = _v
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_hj_promp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProMP"].append(_env_id)

    _name = _v.split("-")
    _env_id = f'VSL{_name[0]}ProDMP-{_name[1]}'
    kwargs_dict_hopper_jump_prodmp = deepcopy(DEFAULT_BB_DICT_ProDMP)
    kwargs_dict_hopper_jump_prodmp['wrappers'].append(mujoco.hopper_jump.MPWrapper)
    kwargs_dict_hopper_jump_prodmp['wrappers'].append(ContextualHopperJumpEnvWrapper)
    kwargs_dict_hopper_jump_prodmp['name'] = _v
    kwargs_dict_hopper_jump_prodmp['phase_generator_kwargs']['learn_tau'] = False
    kwargs_dict_hopper_jump_prodmp['phase_generator_kwargs']['tau_bound'] = [0.5, 2.0]
    kwargs_dict_hopper_jump_prodmp['phase_generator_kwargs']['tau'] = 0.8
    kwargs_dict_hopper_jump_prodmp['trajectory_generator_kwargs']['weights_scale'] = 1.0
    kwargs_dict_hopper_jump_prodmp['trajectory_generator_kwargs']['goal_scale'] = 1.0
    kwargs_dict_hopper_jump_prodmp['trajectory_generator_kwargs']['auto_scale_basis'] = True
    kwargs_dict_hopper_jump_prodmp['basis_generator_kwargs']['num_basis'] = 3
    register(
        id=_env_id,
        entry_point='fancy_gym.utils.make_env_helpers:make_bb_env_helper',
        kwargs=kwargs_dict_hopper_jump_prodmp
    )
    ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS["ProDMP"].append(_env_id)

