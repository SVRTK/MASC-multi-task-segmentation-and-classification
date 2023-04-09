import torch
from networks.losses import DiceCEsoft
import torch.nn as nn
import numpy as np


class Trainer:
    """ Parent class with access to all training and validation utilities

    """

    def __init__(
            self,
            train_loader,
            val_loader,
            max_iterations,
            ckpt_dir,
            res_dir,
            experiment_type="segment",
            optimizer_seg=None,
            optimizer_class=None,
            lr_scheduler_seg=None,
            lr_scheduler_class=None,
            loss_function_seg=DiceCEsoft(),
            loss_function_class=nn.CrossEntropyLoss(),
            input_type_class="multi",
            eval_num=1
    ):
        """
        Args:
            train_loader: dataloader for training.
            val_loader: dataloader for validation.
            max_iterations: maximum number of training iterations.
            ckpt_dir: directory to save model checkpoints'.
            res_dir: directory to save inference predictions and test set metrics.
            experiment_type: (str) defines experiment type
                            one of:
                                    "segment" (only train segmenter),
                                    "classify" (only train classifier),
                                    "joint" (multi-task joint classifier + segmenter)
                                    "LP" (VoxelMorph Label Propagation)
                            default "segment"
            optimizer_seg: segmentation network optimizer.
            optimizer_class: classifier network optimizer.
            lr_scheduler_seg: learning rate scheduler for segmenter network.
            lr_scheduler_class: learning rate scheduler for classifier network.
            loss_function_seg: segmentation loss function (default DiceCE).
            loss_function_class: classification loss function (default CE).
            input_type_class: str defining expected input to classifier. One of:
                            "multi" = use multi-class segmentation labels,
                            "binary" = use binary segmentation labels (add multi-class preds)
                            "img" = use input volume image
            eval_num: number of epochs between each validation loop (default 1).
        """
        super().__init__()

        self.optimizer_seg = optimizer_seg
        self.optimizer_class = optimizer_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_iterations = max_iterations
        self.ckpt_dir = ckpt_dir
        self.res_dir = res_dir
        self.eval_num = eval_num
        self.experiment_type = experiment_type
        self.input_type_class = input_type_class
        self.loss_function_class = loss_function_class
        self.loss_function_seg = loss_function_seg
        self.lr_scheduler_seg = lr_scheduler_seg
        self.lr_scheduler_class = lr_scheduler_class

        exp_names = ["segment", "classify", "joint", "LP"]
        in_class_names = ["multi", "binary", "img"]

        if self.experiment_type not in exp_names:
            raise ValueError("experiment_type parameter"
                             "should be either {}".format(exp_names))

        if self.input_type_class not in in_class_names:
            raise ValueError("input_type_class parameter"
                             "should be either {}".format(exp_names))

    def get_training_dict(self, training=True):
        """ Returns an empty dictionary to store training and validation losses and metrics
            Args:
                training: boolean, set to False for validation metrics

        """
        metrics_train_seg = {
            'total_train_loss': [],
            'multi_train_loss': [],
            'binary_train_loss': []
        }
        metrics_valid_seg = {
            'dice_valid': [],
            'multi_valid_loss': [],
            'binary_valid_loss': []
        }
        metrics_train_class = {'total_train_loss': []}
        metrics_valid_class = {'total_valid_loss': [], 'accuracy': []}

        metrics_train_joint = {
            'total_train_loss_seg': [],
            'multi_train_loss': [],
            'binary_train_loss': [],
            'total_loss': []
        }
        metrics_valid_joint = {
            'dice_valid': [],
            'binary_valid_loss': [],
            'multi_valid_loss': [],
            'total_valid_loss': [],
            'accuracy': [],
        }

        if self.experiment_type == "classify":
            return metrics_train_class if training else metrics_valid_class

        elif self.experiment_type == "segment":
            return metrics_train_seg if training else metrics_valid_seg

        elif self.experiment_type == "joint":
            return metrics_train_joint if training else metrics_valid_joint

    def compute_seg_loss(self, logit_map, mask, LP, binary_seg_weight=1, multi_seg_weight=1):
        """Computes total segmentation loss combining multi-class propagated labels and binary
        Args:
            logit_map: segmentation network output logits
            mask: binary mask torch tensor
            LP: multi-class vessel mask torch tensor
            binary_seg_weight: weight for binary segmentation loss
            multi_seg_weight: weight for multi-class segmentation loss

        Returns: total segmentation loss, binary loss, multi-class loss
        """
        pred = torch.softmax(logit_map, dim=1)
        multi_loss = self.loss_function_seg(pred, LP)
        binary_loss = self.loss_function_seg(add_softmax_labels(pred), mask)
        total_loss_seg = binary_seg_weight * binary_loss + multi_seg_weight * multi_loss

        return total_loss_seg, multi_loss, binary_loss

    def get_input_classifier(self, img=None, segmenter=None):
        """ Generates input tensor to classifier based on input_type_class parameter
            Args:
                img: original image tensor - default
                segmenter: segmentation network
            Returns torch tensor to be used as input to classifier
        """
        if self.input_type_class == "img" or not segmenter:
            class_in = img
        elif self.input_type_class == "multi":
            class_in = torch.softmax(segmenter(img), dim=1)
        elif self.input_type_class == "binary":
            class_in = torch.softmax(segmenter(img), dim=1)
            class_in = add_softmax_labels(class_in)

        return class_in


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
