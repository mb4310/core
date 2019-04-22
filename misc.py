from typing import *
from torch.nn import init
from functools import partial
import torch

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        try: return self.items[idx]
        except TypeError:
            if isinstance(idx[0],bool):
                assert len(idx)==len(self) # bool mask
                return [o for m,o in zip(idx,self.items) if m]
            return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

class Hook():
    def __init__(self, m, f, fwd=True): 

        if fwd:
            self.hook = m.register_forward_hook(partial(f, self))
        else:
            self.hook = m.register_backward_hook(partial(f,self))

    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

def append_gradient_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds, hists = hook.stats
    if mod.training:
        means.append(outp[-1].data.abs().mean().item())
        stds.append(outp[-1].data.abs().std().item())
        max_bin = outp[-1].data.abs().max().item()
        hists.append(outp[-1].data.cpu().histc(40, -max_bin, max_bin))

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds, hists = hook.stats
    if mod.training:
        means.append(outp.data.mean().item())
        stds.append(outp.data.std().item())
        max_bin = outp.abs().max().item()
        hists.append(outp.data.cpu().histc(60,-max_bin,max_bin))

class Hooks(ListContainer):
    def __init__(self, ms, f, fwd=True): 
        super().__init__([Hook(m, f, fwd) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
        
    def remove(self):
        for h in self: h.remove()

class Smoother():
    def __init__(self, beta=0.95):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.vals = []

    def add_value(self, val):
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1-self.beta)*val
        self.vals.append(self.mov_avg/(1-self.beta**self.n))

    def process(self,array):
        for item in array:
            self.add_value(item)
        return self.vals

    def reset(self):
        self.n, self.mov_avg, self.vals = 0,0,[]


def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

def greyscale_tfm(xb):
    return xb.mean(dim=1, keepdim=True)

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

def get_batch(learn):
    learn.xb,learn.yb = next(iter(learn.data.valid_dl))
    learn.do_begin_fit(0)
    learn('begin_batch')
    learn('after_fit')
    return learn.xb,learn.yb

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(l, lin_layers)

def model_summary(learn, find_all=False, print_mod=False):
    xb,yb = get_batch(learn)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook,mod,inp,out: print(f"{mod}\n" if print_mod else "", out.shape)
    with Hooks(mods, f) as hooks: learn.model(xb)