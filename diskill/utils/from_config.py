from distributions.non_lin_conditional.non_lin_ctxt_moe import CtxtDistrNonLinExpMOE
from model_updaters.pg_individual_context_softmax_updater import PGIndividualCtxtSoftmaxUpdater
from model_updaters.update_manager import BaseMoEUpdater

from model_updaters.proj_pg_updater import PGUpdater
from projections.projection_factory import get_projection_layer

from model_updaters.critic_updater import CriticUpdater

from critic_models.vf_net import VFNet
from fancy_gym import make

from utils.network_utils import get_optimizer
from utils.replay_buffer import PerCmpPGModReplayBuffer
from utils.sample_selector import NaiveSelector
from utils.sampler import EnvSampler, SampleManager
import torch as ch

from utils.schedulers import RandomCMPAdder, UpdateScheduler, LinBetaScheduler


def create_dummy_env(env_id, env_kwargs):
    try:
        dummy_env = make(env_id, 0, **env_kwargs)
    except:
        raise ValueError("Could not create environment. Is it defined?")
    return dummy_env


def build_critics(config):
    c_dim = config['environment']['c_dim']
    n_cmps = config['general']['n_init_cmps']
    device = ch.device("cuda:0" if not config['general']['cpu'] else "cpu")
    dtype = ch.float64 if config['general']['dtype'] == "float64" else ch.float32

    n_critics = n_cmps

    critic_config = config['critic']
    if critic_config['use_critic']:
        critics = [VFNet(c_dim, 1, init=critic_config['initialization'],
                         hidden_sizes=critic_config['hidden_sizes_critic'], activation=critic_config['activation'],
                         shared=False, global_critic=False)
                   for _ in range(n_critics)]
        critics = [critic.to(device, dtype) for critic in critics]
        optimizers_critic = [get_optimizer(critic_config['optimizer_critic'], critics[idx].parameters(),
                                           critic_config['lr_critic'], eps=1e-8) for idx in range(n_critics)]
        critic_updater = CriticUpdater(optimizers_critic, critic_config['val_epochs'],
                                       config['data']['num_minibatches'],
                                       config, critic_config['clip_vf'], device, dtype)
    else:
        critics = [None] * n_critics
        critic_updater = None
    return critics, critic_updater


def build_model(config, critics):
    device = ch.device("cuda:0" if not config['general']['cpu'] else "cpu")
    dtype = ch.float64 if config['general']['dtype'] == "float64" else ch.float32
    n_cmps = config['general']['n_init_cmps']
    c_dim = config['environment']['c_dim']
    a_dim = config['environment']['a_dim']
    dummy_env = create_dummy_env(config['environment']['env_id'], config['environment']['env_kwargs'])

    model = CtxtDistrNonLinExpMOE(c_dim, a_dim,
                                  n_cmps=n_cmps,
                                  hidden_sizes_expert=config['experts']['hidden_sizes_policy'],
                                  hidden_sizes_ctxt_distr=config['ctxt_distr']['hidden_sizes_ctxt_distr'],
                                  init_exp=config['experts']['initialization'],
                                  init_ctxt_distr=config['ctxt_distr']['initialization'],
                                  activation_exp=config['experts']['activation'],
                                  activation_ctxt_distr=config['ctxt_distr']['activation_ctxt_distr'],
                                  contextual_std=config['experts']['contextual_std'],
                                  init_std=config['experts']['init_std'],
                                  init_mean=config['experts']['init_mean'],
                                  policy_type=config['experts']['policy_type'],
                                  device=device,
                                  dtype=dtype,
                                  proj_type=config['trl']['proj_type'])

    optimizers_policy = [get_optimizer(config['experts']['optimizer_policy'], model.components[idx].parameters(),
                                       config['experts']['lr_policy'], eps=1e-8) for idx in range(n_cmps)]

    projection = get_projection_layer(proj_type=config['trl']['proj_type'],
                                      mean_bound=config['trl']['mean_bound'],
                                      cov_bound=config['trl']['cov_bound'],
                                      trust_region_coeff=config['trl']['trust_region_coeff'],
                                      scale_prec=config['trl']['scale_prec'],
                                      entropy_schedule=config['trl']['entropy_schedule'],
                                      action_dim=a_dim,
                                      total_train_steps=config['iterations'],
                                      target_entropy=config['trl']['target_entropy'],
                                      temperature=config['trl']['temperature'],
                                      entropy_eq=config['trl']['entropy_eq'],
                                      entropy_first=config['trl']['entropy_first'],
                                      do_regression=config['trl']['do_regression'],
                                      regression_iters=config['trl']['regression_iters'],
                                      regression_lr=config['trl']['lr_reg'],
                                      optimizer_type_reg=config['trl']['optimizer_reg'],
                                      cpu=config['general']['cpu'],
                                      dtype=dtype)
    expert_updater = PGUpdater(optimizers_policy, projection, config['experts']['epochs'],
                               config['general']['alpha'], config['data']['norm_advantages'],
                               config['data']['num_minibatches'],
                               config['experts']['clip_advantages'], device, dtype,
                               config['experts']['importance_ratio_clip'], config)
    optimizers_ctxt_distribution = [get_optimizer(config['ctxt_distr']['optimizer_ctxt_distr'],
                                                  model.ctxt_distribution[idx].parameters(),
                                                  config['ctxt_distr']['lr_ctxt_distr'], eps=1e-8)
                                    for idx in range(n_cmps)]

    ctxt_distr_updater = PGIndividualCtxtSoftmaxUpdater(optimizers_ctxt_distribution,
                                                        config['ctxt_distr']['epochs_ctxt_distr'],
                                                        config['general']['beta'],
                                                        config['data']['norm_advantages'],
                                                        config['ctxt_distr']['clip_advantages_ctxt_distr'],
                                                        device,
                                                        dtype,
                                                        config['ctxt_distr']['importance_ratio_clip_ctxt_distr'],
                                                        config,
                                                        config['ctxt_distr']['max_grad_norm'])

    # set the critics
    for i in range(model.num_components):
        model.components[i].set_critic(critics[i])
    return model, expert_updater, ctxt_distr_updater


def get_setup_from_config(config, seed, cpu_cores):
    # env related params
    env_id = config['environment']['env_id']
    env_kwargs = config['environment']['env_kwargs']
    num_envs = config['environment']['num_envs']
    num_test_envs = config['environment']['num_test_envs']

    n_cmps = config['general']['n_init_cmps']

    dummy_env = create_dummy_env(env_id, env_kwargs)
    c_dim = dummy_env.context_space.shape[0]
    a_dim = dummy_env.action_space.shape[0]
    config['environment']['c_dim'] = c_dim
    config['environment']['a_dim'] = a_dim

    cpu = True if config['general']['cpu'] else False
    dtype = ch.float64 if config['general']['dtype'] == "float64" else ch.float32
    n_env_ctxt_samples = config['data']['n_env_ctxt_samples']
    # EnvSampler
    sampler = EnvSampler(env_id, num_envs, num_test_envs, seed, use_ch=True, cpu=cpu, dtype=dtype,
                         n_env_ctxt_samples=n_env_ctxt_samples, env_kwargs=env_kwargs, cpu_cores=cpu_cores)
    sample_manager = SampleManager(sampler, NaiveSelector())
    cmp_add_scheduler = RandomCMPAdder(add_every_it=config['general']['add_every_it'],
                                       sample_dim=a_dim, ctxt_dim=c_dim, config=config,
                                       fine_tune_all_it=config['general']['fine_tune_all_it'],
                                       n_cmp_adds=config['general']['n_cmp_adds'])
    update_scheduler = UpdateScheduler(n_sim_cmp_updates=config['general']['n_init_cmps'],
                                       fine_tune_every_it=config['general']['fine_tune_every_it'],
                                       fine_tune_all_it=config['general']['fine_tune_all_it'])

    # ReplayBuffer
    replay_buffer = PerCmpPGModReplayBuffer(config['data']['buffer_size'], c_dim, a_dim, n_cmps,
                                            config['data']['norm_buffer_size'], dtype, cpu, use_ch=True)

    critics, critic_updater = build_critics(config)
    model, expert_updater, ctxt_distr_updater = build_model(config, critics)

    update_manager = BaseMoEUpdater(model, critics, expert_updater, None, None, ctxt_distr_updater, critic_updater)
    return sampler, replay_buffer, sample_manager, critics, model, update_manager, cmp_add_scheduler, \
        update_scheduler, dtype, cpu
