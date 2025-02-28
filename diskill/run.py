import cw2.cw_data.cw_wandb_logger
import numpy as np
import torch as ch

from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger
import os
import contextual_envs

from utils.from_config import get_setup_from_config
from utils.schedulers import LinBetaScheduler, ExpBetaScheduler
from vsl_base import VSLBase
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def deltree(target):
    print("deltree", target)
    for d in os.listdir(target):
        try:
            deltree(target + '/' + d)
        except OSError:
            os.remove(target + '/' + d)

    os.rmdir(target)


class VSLIterativeExperiment(experiment.AbstractIterativeExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        ch.set_num_threads(1)
        params = config["params"]
        # cpu_cores = config.get("cpu_cores", None)
        cpu_cores = None
        params.update({'iterations': config['iterations']})
        seed = rep
        np.random.seed(seed)
        ch.manual_seed(seed)

        sampler, replay_buffer, sample_manager, critics, model, update_manager, cmp_add_scheduler, update_scheduler, \
            dtype, cpu = get_setup_from_config(params, seed, cpu_cores)
        if params['ctxt_distr']['beta_scheduler']:
            # beta_scheduler = LinBetaScheduler(params['general']['beta'], params['general']['beta_target'],
            #                                   config['iterations'])
            beta_scheduler = ExpBetaScheduler(params['general']['beta'], params['general']['beta_target'],
                                              config['iterations'])
        else:
            beta_scheduler = None
        self.optimizer = VSLBase(seed,
                                 config['_rep_log_path'],
                                 params['general']['alpha'],
                                 params['general']['beta'],
                                 config['iterations'],
                                 model,
                                 sample_manager,
                                 params['data']['n_samples_p_cmp'],
                                 params['data']['batch_size'],
                                 replay_buffer,
                                 update_manager,
                                 params['general']['verbose'],
                                 cmp_add_scheduler,
                                 update_scheduler,
                                 params['general']['test_every_it'],
                                 params['general']['n_test_samples'],
                                 params['general']['save_model_every_it'],
                                 dtype,
                                 True,
                                 all_configs=params,
                                 log_verbosity=params['general'].get('log_verbosity', 1), beta_scheduler=beta_scheduler)
        self.savepath = config['_rep_log_path']
        self.logger = logger

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        return self.optimizer.train_iter(n)

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        self.delete_wandb_logs()

    def delete_wandb_logs(self):
        for logger in cw.logArray._logger_array:
            wandb_logger = logger if isinstance(logger, cw2.cw_data.cw_wandb_logger.WandBLogger) else None
        if wandb_logger is not None:
            wandb_logger.finalize()
            wandb_logger.run = None
            import os
            abs_path = wandb_logger.log_path + '/wandb'
            deltree(abs_path)
        else:
            print("No wandb error initialized")


if __name__ == "__main__":
    # ch.autograd.set_detect_anomaly(True)
    cw = cluster_work.ClusterWork(VSLIterativeExperiment)

    cw.add_logger(WandBLogger())
    # RUN
    cw.run()
