import torch
import numpy as np


def add_softmax_labels(softmax_preds):
    """ Returns added multi-class foreground softmax predictions (background excluded)
        Assumes background in first channel

    Args:
        softmax_preds: multi-class softmax network segmentation predictions (shape BNH[WD])
    Returns: torch tensor with original image shape and two channels, background and foreground
    """

    added_preds = torch.sum(softmax_preds[:, 1:], dim=1)
    added_preds = torch.cat([softmax_preds[:, 0, ...].unsqueeze(1), added_preds.unsqueeze(1)], dim=1)

    return added_preds


# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


# To save the checkpoint
def save_checkpoint(ckpt_name,
                    ckpt_dir,
                    model,
                    optimizer,
                    iteration=None,
                    epoch=None,
                    losses_train=None,
                    losses_valid=None,
                    losses_train_joint=None,
                    losses_valid_joint=None,
                    lr_scheduler=None,
                    binary_seg_weight=None,
                    multi_seg_weight=None,
                    best_valid_loss=None,
                    best_metric_valid=None,
                    ):

    model = model.state_dict()
    optimizer = optimizer.state_dict()
    if lr_scheduler:
        lr_scheduler = lr_scheduler.state_dict()

    ckpt_dict = {'model': model, 'optimizer': optimizer, 'iteration': iteration,
                 'epoch': epoch, 'losses_train': losses_train, 'losses_valid': losses_valid,
                 'losses_train_joint': losses_train_joint, 'losses_valid_joint': losses_valid_joint,
                 'lr_scheduler': lr_scheduler, 'binary_seg_weight': binary_seg_weight,
                 'multi_seg_weight': multi_seg_weight,'best_valid_loss': best_valid_loss,
                 'best_metric_valid': best_metric_valid}

    torch.save(ckpt_dict, '{}{}.ckpt'.format(ckpt_dir, ckpt_name))


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
