import torch
import math
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
import re
from functools import partial
from misc import *
import pandas as pd
import matplotlib.pyplot as plt
import time

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class TestCallback(Callback):
    _order=1
    def after_step(self):
        print(self.n_iter)
        if self.n_iter>=10: raise CancelTrainException()

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0
    
    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
        
    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)] 
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)

class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        
    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()
        
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(self.epoch)

class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

class BatchTransformXCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.xb)

"""These functions assist with annealing parameters like LR and momentum"""

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)
@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = ((pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])).item()
        return scheds[idx](actual_pos)
    return _inner

def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]

def splice_anneal(midpoint=1e-6, max_lr=1e-4):
    return [sched_lin(0, midpoint), sched_cos(midpoint, max_lr)]

def one_cycle_schedule(max_lr=1e-3, pct_start=0.3, div=25, moms=[0.85,0.95]):
    lr_sched = combine_scheds([pct_start, 1-pct_start], cos_1cycle_anneal(max_lr/div, max_lr, max_lr/1e4))
    mom_sched = combine_scheds([pct_start, 1-pct_start], cos_1cycle_anneal(moms[1],moms[0],moms[1]))
    return [ParamScheduler('lr', lr_sched), ParamScheduler('mom', mom_sched)]

def splice_schedule(max_lr = 1e-3, midpoint=1e-6, pct_start=0.3, div=25, f_moms=[0.85,0.95], moms=[0.85,0.95]):
    """The corresponding splitter passed to learner should return a list [old_params, new_params]"""

    old_param_lr_sched = combine_scheds([pct_start, 1-pct_start], splice_anneal(midpoint, max_lr/10))
    new_param_lr_sched = combine_scheds([pct_start, 1-pct_start], cos_1cycle_anneal(max_lr/div, max_lr, max_lr/1e4))

    old_param_mom_sched = sched_cos(moms[0],moms[1])
    new_param_mom_sched = combine_scheds([pct_start, 1-pct_start], cos_1cycle_anneal(moms[1],moms[0],moms[1]))

    lr_sched = ParamScheduler('lr', [old_param_lr_sched, new_param_lr_sched])
    mom_sched = ParamScheduler('mom', [old_param_mom_sched, new_param_mom_sched])

    return [lr_sched, mom_sched]

class LayerActivations(Callback):
    _order = 0 
    def __init__(self, ms = None):
        self.ms = ms

    def begin_fit(self):
        if self.ms is None:
            self.ms = [m for m in self.model if len(list(m.parameters())) > 1]
        self.hooks = Hooks(self.ms, append_stats, fwd=True)

    def after_fit(self):
        self.hooks.remove()

    def plot_stats(self):
        for k,m in enumerate(self.ms):
            print('Block {}: {}'.format(k, m))

        fig, ax = plt.subplots(figsize=(12,8))
        act_means = pd.DataFrame({'Block_{}'.format(k): Smoother().process(hook.stats[0]) for k, hook in enumerate(self.hooks)})
        for act in act_means.columns:
            ax.plot(act_means[act])
        ax.legend()
        ax.set_xlabel('Iteration')
        plt.title('Mean of activations by block')

        fig, ax = plt.subplots(figsize=(12,8))
        act_stds = pd.DataFrame({'Block_{}'.format(k): Smoother().process(hook.stats[1]) for k, hook in enumerate(self.hooks)})
        for act in act_stds.columns:
            ax.plot(act_stds[act])
        ax.legend()
        ax.set_xlabel('Iteration')
        plt.title('Std of activations by block')

    def plot_distributions(self, num_batches=120):
        for k,h in enumerate(self.hooks):
            fig,ax = plt.subplots(figsize=(12,8))
            hist = get_hist(h)
            if hist[:8].sum().item() == 0:
                hist = hist[29:]
            ax.imshow(hist[:,:num_batches], origin='lower', cmap='RdYlGn')
            ax.axis('off')
            plt.title('Block {}'.format(k))

    def plot_percent_small(self):
        for k,h in enumerate(self.hooks):
            fig, ax = plt.subplots(figsize = (8,6))
            vals = torch.stack(h.stats[2]).t().float()
            vals = vals[29:31].sum(dim=0) / vals.sum(dim=0)
            ax.plot(vals)
            ax.set_xlabel('Iteration')
            plt.title('Percent activations near zero: Block {}'.format(k))
        
class GradientNorms(Callback):
    _order = 0 
    def __init__(self, ms = None):
        self.ms = ms

    def begin_fit(self):
        if self.ms is None:
            self.ms = [m for m in self.model if len(list(m.parameters())) > 1]
        self.hooks = Hooks(self.ms, append_gradient_stats, fwd=False)

    def after_fit(self):
        self.hooks.remove()

    def plot_stats(self):
        for k,m in enumerate(self.ms):
            print('Block {}: {}'.format(k, m))

        fig, ax = plt.subplots(figsize=(12,8))
        act_means = pd.DataFrame({'Block_{}'.format(k): Smoother().process(hook.stats[0]) for k, hook in enumerate(self.hooks)})
        for act in act_means.columns:
            ax.plot(act_means[act])
        ax.legend()
        ax.set_xlabel('Iteration')
        plt.title('Mean of gradient norm by block')

        fig, ax = plt.subplots(figsize=(12,8))
        act_stds = pd.DataFrame({'Block_{}'.format(k): Smoother().process(hook.stats[1]) for k, hook in enumerate(self.hooks)})
        for act in act_stds.columns:
            ax.plot(act_stds[act])
        ax.legend()
        ax.set_xlabel('Iteration')
        plt.title('Std of gradient norm by block')

    def plot_distributions(self, num_batches=100):
        for k,h in enumerate(self.hooks):
            fig,ax = plt.subplots(figsize=(12,8))
            ax.imshow(get_hist(h)[:,:num_batches], origin='lower', cmap='RdYlGn')
            ax.axis('off')
            plt.title('Block {}'.format(k))
        
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses = [[] for _ in self.opt.hypers],[]

    def after_batch(self):
        if not self.in_train: return
        for k, group in enumerate(self.opt.hypers):
            self.lrs[k].append(group['lr'])
        self.losses.append(self.loss.detach().cpu())        

    def plot_lr(self): 
        lrs = pd.DataFrame({'param_group_{}'.format(k): self.lrs[k] for k in range(len(self.lrs))})
        fig,ax = plt.subplots(figsize=(8,8))
        for group in lrs.columns:
            lrs[group].plot.line()
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')

    def plot_loss(self): plt.plot(self.losses)
        
    def plot(self, clip_pct=0.03):
        losses = Smoother().process([o.item() for o in self.losses])
        start = int(len(losses) * clip_pct)
        end = int(len(losses) * (1-clip_pct))
        fig,ax = plt.subplots(figsize=(6,6))
        ax.plot(self.lrs[-1][start:end], losses[start:end])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_xscale('log')

class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs):
        self.pname,self.sched_funcs = pname,listify(sched_funcs)

    def begin_batch(self): 
        if not self.in_train: return
        fs = self.sched_funcs
        if len(fs)==1: fs = fs*len(self.opt.param_groups)
        pos = self.n_epochs/self.epochs
        for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)
            
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=300, min_lr=1e-6, max_lr=1):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9
        
    def begin_batch(self): 
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers: pg['lr'] = lr
            
    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*100:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss

    def after_fit(self):
        self.recorder.plot()





