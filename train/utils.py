import torch
import numpy as np

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


# To save the checkpoint
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def plot_losses_train(res_dir, losses_train, title_plot):
    # Get some variables about the train
    ####################
    n_epochs_train = len(losses_train)
    keys_train = list(losses_train[0].keys())
    n_iter_train = len(losses_train[0][keys_train[0]])

    # Average losses
    ####################
    losses_train_mean = {key_: [] for key_ in keys_train}
    losses_train_std = {key_: [] for key_ in keys_train}
    for epoch_ in losses_train:
        for key_ in keys_train:
            losses_train_mean[key_].append(np.mean(epoch_[key_]))
            losses_train_std[key_].append(np.std(epoch_[key_]))

    # Plot losses
    ####################
    import matplotlib.pyplot as plt
    start_epoch = 2

    plt.figure(figsize=(30, 30))
    for i_, key_ in enumerate(keys_train):
        plt.subplot(6, 2, i_ + 1)
        plt.fill_between(np.arange(start_epoch, n_epochs_train),
                         [x - y for x, y in zip(losses_train_mean[key_][start_epoch:],
                                                losses_train_std[key_][start_epoch:])],
                         [x + y for x, y in zip(losses_train_mean[key_][start_epoch:],
                                                losses_train_std[key_][start_epoch:])],
                         alpha=0.2)
        plt.plot(np.arange(start_epoch, n_epochs_train), losses_train_mean[key_][start_epoch:])
        plt.xlabel('Epochs')
        plt.ylabel(key_)

        if i_ >= len(keys_train) - 1:
            break

    plt.savefig(res_dir + '/' + title_plot + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(res_dir + '/' + title_plot + '.svg',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close()
