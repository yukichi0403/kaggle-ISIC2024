import sys
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler


# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders] # [imbalanced_loader_iter, balanced_loader_iter]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

    
class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders # [imbalanced_loader, balanced_loader]

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

    
def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities


def modify_loader(loader, mode, ep=None, n_eps=None):
    class_count = np.unique(loader.dataset.dr, return_counts=True)[1] #クラスのuniqueごとの個数を返す
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps) #クラスごとの出現させる確率の配列
    sample_weights = sampling_probs[loader.dataset.dr] #元のデータセットにクラスごとの出現させる確率を適応

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader


def get_combo_loader(loader, base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling)

    balanced_loader = modify_loader(loader, mode='class')
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader