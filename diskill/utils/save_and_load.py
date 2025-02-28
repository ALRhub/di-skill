import numpy as np
import os

import torch as ch
import yaml, collections

from distributions.non_lin_conditional.non_lin_ctxt_moe import CtxtDistrNonLinExpMOE
from utils.from_config import build_critics


def save_model_non_lin_ctxt_distr_moe(model, save2path, optimizers_pol=None, optimizers_ctxt_distribution=None, it=None,
                                      config=None, value_functions=None, optimizers_critic=None):
    model_cmps_dict_list = [cmp.state_dict() for cmp in model.components]
    ctxt_cmps_dict_list = [cmp.state_dict() for cmp in model.ctxt_distribution]
    optimizers_pol_dict = [optimizer.state_dict() for optimizer in optimizers_pol]
    optimizers_ctxt_distr_dict = [optimizer.state_dict() for optimizer in optimizers_ctxt_distribution]
    if value_functions is not None:
        value_functions_dict = [val_fct.state_dict() for val_fct in value_functions]
        optimizers_critic_dict = [val_opt_critic.state_dict() for val_opt_critic in optimizers_critic]
    else:
        value_functions_dict = None
        optimizers_critic_dict = None
    model_dict = {'model_cmps_dict': model_cmps_dict_list, 'optimizers_pol_dict': optimizers_pol_dict,
                  'ctxt_cmps_dict': ctxt_cmps_dict_list,
                  'optimizers_ctxt_distr_dict': optimizers_ctxt_distr_dict,
                  'value_functions_dict': value_functions_dict, 'optimizers_critic_dict': optimizers_critic_dict}

    filename = 'nonlinmoe_ctxt_distr_model'

    if it is None:
        savepath = os.path.join(save2path, filename + '.npz')
    else:
        savepath = os.path.join(save2path, filename + '_' + str(it) + '.npz')

    savepath_config = os.path.join(save2path, 'config.yml')

    np.savez_compressed(savepath, **model_dict)
    with open(savepath_config, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def save_data(data_dict, save2path):
    filename = 'data'
    savepath = os.path.join(save2path, filename + '.npz')
    np.savez_compressed(savepath, **data_dict)


def load_nonlin_ctxt_distr_moe_dict(path2load, it=None):
    if it is None:
        model_path = os.path.join(path2load, 'nonlinmoe_ctxt_distr_model' + '.npz')
    else:
        model_path = os.path.join(path2load, 'nonlinmoe_ctxt_distr_model' + '_' + str(it) + '.npz')
    config_path = os.path.join(path2load, 'config.yml')
    model_dict = dict(np.load(model_path, allow_pickle=True))
    with open(config_path, 'r') as stream:
        con = yaml.safe_load(stream)
    return model_dict, con


def load_model_nonlin_ctxt_distr_moe(path2load, it=None):
    model_dict, con = load_nonlin_ctxt_distr_moe_dict(path2load, it=it)
    con['general']['n_init_cmps'] = model_dict['ctxt_cmps_dict'].shape[0]
    model, dtype, device = CtxtDistrNonLinExpMOE.create_from_config(con)
    use_critic = con['critic']['use_critic']
    if use_critic:
        critics = build_critics(con)[0]
    else:
        critics = None
    for i in range(model.num_components):
        c_model_dict = collections.OrderedDict()
        c_value_dict = collections.OrderedDict()
        for key in model_dict['model_cmps_dict'][i].keys():
            if 'critic' in key:
                c_value_dict[key.split('critic.')[1]] = model_dict['model_cmps_dict'][i][key]
            else:
                c_model_dict[key] = model_dict['model_cmps_dict'][i][key]
        model.components[i].load_state_dict(c_model_dict)
        if critics is not None:
            model.components[i].set_critic(critics[i])
            model.components[i].critic.load_state_dict(c_value_dict)
    for i in range(model.num_components):
        model.ctxt_distribution[i].load_state_dict(model_dict['ctxt_cmps_dict'][i])
    return model


def load_and_sort_all_models_nonlin_ctxt_distr_moe(path2load):
    all_data = os.listdir(path2load)
    models = {}
    last_it = 10000000
    for name in all_data:
        if name not in ['wandb', 'pngs']:
            if name.split('.')[1] == 'npz':
                it = name.split('_')[-1].split('.')[0]
                if it not in ['e', 'c', 'entropy', 'reward', 'config', 'model', 'interacts', 'executed', 'data']:
                    models[int(it)] = load_model_nonlin_ctxt_distr_moe(path2load, it=int(it))
    try:
        models[last_it] = load_model_nonlin_ctxt_distr_moe(path2load)
    except:
        print('last model was not saved!')
    keys_list = list(models.keys())
    sorted_iterations_idx = np.argsort(keys_list)
    sorted_keys_np_arr = np.array(keys_list)[sorted_iterations_idx]
    sorted_models = []
    for it in sorted_iterations_idx:
        c_key = keys_list[it]
        c_model = models[c_key]
        sorted_models.append(c_model)

    return models, sorted_models, sorted_keys_np_arr
