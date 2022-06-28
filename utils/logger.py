"""
Log for training and testing.
"""

import os
import sys
import time
import logging
from tensorboardX import SummaryWriter
import torch.distributed as dist


# ================================================================
# API
# ================================================================


def configure(save, rank=0, is_distributed=False, is_write=True):
    """
    Make configuration for logger.
    :param save: str, directory for saving logs files.
    :param rank: int, device rank.
    :param is_distributed: bool, whether to use distributed machines.
    :param is_write: bool, whether to use tensorboard to save some results.
    :return: None
    """
    Logger.CURRENT = Logger(save, rank, is_distributed, is_write=is_write)


def log(string, *args):
    """
    Log one string to logs.txt. Similar to `print` function.
    """
    get_current().info(string, *args)


def print_kv():
    """
    print current logging variables.
    """
    get_current().print_kv()


def log_kv(key, val):
    """
    Log a new value (`val`) to variable (`key`) and update the average.
    """
    get_current().log_kv(key, val)


def write_kv(step):
    """
    Write current logging variables to tensorboard file at current step.
    """
    get_current().write_kv(step)


def clear_kv():
    """
    Clear current logging variables.
    """
    get_current().kvdict.clear()


def get_kv():
    """
    Get the dict that saves current logging variables.
    """
    return get_current().kvdict


def get_dir():
    """
    Get directory that logs files are being written to.
    Return object will be None if there is no output directory (i.e., if you didn't call `configure` function).
    """
    return get_current().get_dir()


# ================================================================
# Backend
# ================================================================

def get_current():
    assert Logger.CURRENT is not None
    return Logger.CURRENT


class Logger(object):
    CURRENT = None

    def __init__(self, save, rank=0, is_distributed=False, is_write=True):
        self.dir = save
        self.rank = rank
        self.is_distributed = is_distributed
        self.is_write = is_write
        self.kvdict = {}
        if not os.path.exists(save):
            os.makedirs(save, exist_ok=True)
        if rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(
                stream=sys.stdout,
                level=logging.INFO,
                format=log_format,
                datefmt='%m/%d %I:%M:%S %p'
            )
            fh = logging.FileHandler(os.path.join(save, 'logs.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()
            if is_write:
                self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime('(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)

    def print_kv(self):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(self.kvdict.items()):
            if hasattr(val.avg, "__float__"):
                valstr = "%-8.3g" % val.avg
            else:
                valstr = str(val.avg)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            self.info("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )

        for line in lines:
            self.info(line)

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def log_kv(self, key, value):
        if key in self.kvdict.keys():
            self.kvdict[key].update(value)
        else:
            self.kvdict[key] = AverageMeter()
            self.kvdict[key].update(value)

    def write_kv(self, step):
        for key, val in self.kvdict.items():
            average_tensor(val.avg, self.is_distributed)
            self.write_scalar(key, val.avg, step)

    def clear_kv(self):
        self.kvdict.clear()

    def get_dir(self):
        return self.dir

    def write_scalar(self, *args, **kwargs):
        assert self.is_write
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def write_figure(self, *args, **kwargs):
        assert self.is_write
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def write_image(self, *args, **kwargs):
        assert self.is_write
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def write_histogram(self, *args, **kwargs):
        assert self.is_write
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def close(self):
        if self.rank == 0 and self.is_write:
            self.writer.close()


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = None

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = None

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAverageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.avg = 0

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size
