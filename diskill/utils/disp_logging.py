import logging
import sys


class _CWFormatter(logging.Formatter):
    def __init__(self):
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)


class BaseLogger:
    def __init__(self):
        self.logger = logging.getLogger('info')
        sh = logging.StreamHandler(sys.stdout)
        formatter = _CWFormatter()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def log_base_info(self, it, time, reward):
        try:
            self.logger.info('################# Iteration:{:.0f}   took {:.2f} s  Reward: {:.3f} #################'.
                             format(it, time, reward))
        except:
            self.logger.info('################# Iteration:{:.0f}   took {:.2f} s  Reward: ' + str(reward) +
                             ' #################'.format(it, time))

    def log_info(self, info_dict: dict, update_info: str):
        self.logger.info(f'----- {update_info} Distr. Updates -----')
        if len(info_dict.keys()) != 0:
            for key in info_dict.keys():
                log_string = f'{update_info} Cmp. {str(key)}: '
                for c_key in info_dict[key].keys():
                    try:
                        log_string += f' {str(c_key)}: ' + '{:.3f}'.format(info_dict[key][c_key])
                    except Exception:
                        pass
                self.logger.info(log_string)
            self.logger.info('')
        else:
            log_string = f'No {update_info} Comp. update'
            self.logger.info(log_string)
