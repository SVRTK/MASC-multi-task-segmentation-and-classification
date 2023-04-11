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
            train_loader: dataloader for training (pytorch dataloader)
            val_loader: dataloader for validation (pytorch dataloader)
            max_iterations: maximum number of training iterations (int)
            ckpt_dir: directory to save model checkpoints' (str)
            res_dir: directory to save inference predictions and test set metrics (str)
            experiment_type: defines experiment type (str) 
                            one of:
                                    "segment" (only train segmenter),
                                    "classify" (only train classifier),
                                    "joint" (multi-task joint classifier + segmenter)
                                    "LP" (VoxelMorph Label Propagation)
                            default "segment"
            optimizer_seg: segmentation network optimizer (pytorch model)
            optimizer_class: classifier network optimizer (pytorch model) 
            lr_scheduler_seg: learning rate scheduler for segmenter network (pytorch LR scheduler)
            lr_scheduler_class: learning rate scheduler for classifier network (pytorch LR scheduler)
            loss_function_seg: segmentation loss function (default DiceCE) (pytorch loss function)
            loss_function_class: classification loss function (default CE) (pytorch loss function)
            input_type_class: str defining expected input to classifier (str).
                            One of:
                                "multi" = use multi-class segmentation labels,
                                "binary" = use binary segmentation labels (add multi-class preds)
                                "img" = use input volume image
            eval_num: number of epochs between each validation loop (default 1) (int)
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
        """ Returns an empty dictionary to store training and validation losses and metrics (for each epoch)
            Args:
                training: set to False for validation metrics (bool)

        """
        metrics_train_seg = {
            'total_train_loss_seg': [],
            'multi_train_loss_seg': [],
            'binary_train_loss_seg': []
        }
        metrics_valid_seg = {
            'dice_valid': [],
            'multi_valid_loss_seg': [],
            'binary_valid_loss_seg': []
        }
        metrics_train_class = {'total_train_loss_class': []}
        metrics_valid_class = {'total_valid_loss_class': [], 'accuracy': []}

        metrics_train_joint = [metrics_train_seg, metrics_train_class]
        metrics_valid_joint = [metrics_valid_seg, metrics_valid_class]

        if self.experiment_type == "classify":
            return metrics_train_class if training else metrics_valid_class

        elif self.experiment_type == "segment":
            return metrics_train_seg if training else metrics_valid_seg

        elif self.experiment_type == "joint":
            return metrics_train_joint if training else metrics_valid_joint

    def compute_seg_loss(self, logit_map, mask, LP, binary_seg_weight=1, multi_seg_weight=1):
        """Computes total segmentation loss combining multi-class propagated labels and binary
        Args:
            logit_map: segmentation network output logits (torch tensor)
            mask: binary mask (torch tensor)
            LP: multi-class vessel mask (torch tensor)
            binary_seg_weight: weight for binary segmentation loss (float)
            multi_seg_weight: weight for multi-class segmentation loss (float)

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
                img: original image tensor - default (torch tensor)
                segmenter: segmentation network (pytorch model)
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


def get_in_channels_class(config):
    """ Returns the number of input channels for a given classifier experiment
        Args:
            config: config dictionary containing training experiment parameters (dict)
        Returns:
            number of input classifier channels (int) 
    """
    if config['input_type_class']=='vol':
        in_channels = 1
    else:
        in_channels = config['N_seg_labels']
    return in_channels


def add_softmax_labels(softmax_preds):
    """ Returns added multi-class foreground softmax predictions (background excluded)
        Assumes background in first channel

    Args:
        softmax_preds: multi-class softmax network segmentation predictions (shape BNH[WD]) (torch tensor)
    Returns: torch tensor with original image shape and two channels, background and foreground
    """

    added_preds = torch.sum(softmax_preds[:, 1:], dim=1)
    added_preds = torch.cat([softmax_preds[:, 0, ...].unsqueeze(1), added_preds.unsqueeze(1)], dim=1)

    return added_preds


def cuda(xs, device_num=None):
    """ Sends torch tensor to cuda device
        Args:
            xs: torch tensor
        Returns:
            cuda tensor

    """
    if torch.cuda.is_available():
        if device_num:
            torch.cuda.set_device(device_num)
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def save_checkpoint(ckpt_name,
                    ckpt_dir,
                    model,
                    optimizer,
                    iteration=None,
                    epoch=None,
                    losses_train=None,
                    losses_valid=None,
                    lr_scheduler=None,
                    binary_seg_weight=None,
                    multi_seg_weight=None,
                    best_valid_loss=None,
                    best_metric_valid=None,
                    ):
    """ Saves network checkpoint as a dict, with option to save training and valid losses
        Args:
            ckpt_name: checkpoint file name (str)
            ckpt_dir: directory where checkpoint will be stored (str)
            model: latest model to checkpoint (pytorch model)
            optimizer: optimizer to checkpoint (pytorch optimizer)
            iteration: latest iteration (int)
            epoch: latest epoch (int)
            losses_train: list of dictionaries containing training losses (list)
            losses_valid: list of dictionaries containing valid losses (list)
            lr_scheduler: learning rate scheduler (pytorch LR scheduler)
            binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
            multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
            best_valid_loss: the best mean validation loss (float)
            best_metric_valid: the best mean validation metric (float)
    """

    model = model.state_dict()
    optimizer = optimizer.state_dict()
    if lr_scheduler:
        lr_scheduler = lr_scheduler.state_dict()

    ckpt_dict = {'model': model, 'optimizer': optimizer, 'iteration': iteration,
                 'epoch': epoch, 'losses_train': losses_train, 'losses_valid': losses_valid,
                 'lr_scheduler': lr_scheduler, 'binary_seg_weight': binary_seg_weight,
                 'multi_seg_weight': multi_seg_weight, 'best_valid_loss': best_valid_loss,
                 'best_metric_valid': best_metric_valid}

    torch.save(ckpt_dict, '{}{}.ckpt'.format(ckpt_dir, ckpt_name))


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    """ Loads network checkpoint from .ckpt file
        Args:
            ckpt_path: directory and checkpoint name (str)
            map_location: change the device of the tensors in the state dict
                            set to None for GPU training (str, e.g. 'cpu')

        Returns:
            Checkpoint
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def try_load_ckpt(ckpt_dir, ckpt_name, model, optimizer, lr_scheduler=None):
    """ Checks if model has been previously checkpointed, and if so load model weights and losses
        Args: 
            ckpt_dir: directory where checkpoint is saved (str)
            ckpt_name: name of the checkpoint file (str)
            model: model to load weights onto (pytorch model)
            optimizer: optimizer for loading state dict (pytorch model)
            lr_scheduler: learning rate scheduler for loading state dict (pytorch learning rate scheduler)
    :return: 
    """

    # Loading pretrained model
    try:
        ckpt = load_checkpoint(ckpt_path='{}{}.ckpt'.format(ckpt_dir, ckpt_name))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        losses_train_init_class = ckpt['losses_train']
        losses_valid_init_class = ckpt['losses_valid']
        best_metric = ckpt['best_metric']
        loss_val_best = ckpt['best_loss']
        iteration = ckpt['iteration']
        epoch = ckpt['epoch']
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    except:
        print("Starting training from scratch")
        losses_train_init_class = []
        losses_valid_init_class = []
        best_metric = 1e-5
        loss_val_best = 1e5
        iteration = 0
        epoch = 0

    return losses_train_init_class, losses_valid_init_class, best_metric, loss_val_best, iteration, epoch


class LambdaLR():
    """ Computes the LR decay rate
    
    """
    def __init__(self, epochs, offset, decay_epoch):
        """
        Args: 
            epochs: total number of epochs for training (int)
            offset: number of epochs to offset current epoch with (int)
            decay_epoch: final+1 LR decay multiplicative factor (int)
        """
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epochs - self.decay_epoch)

def plot_losses_train(res_dir, losses_train, title_plot):
    """ Plots and saves the training/validation losses as .svg & .eps files
        Args:
            res_dir: directory to save the plotted losses (str)
            losses_train: list of dicts containing losses for each epoch (list)
            title_plot: title of saved plot file (str)
    """
    n_epochs_train = len(losses_train)
    keys_train = list(losses_train[0].keys())
    n_iter_train = len(losses_train[0][keys_train[0]])

    # Average losses (over each epoch)
    losses_train_mean = {key_: [] for key_ in keys_train}
    losses_train_std = {key_: [] for key_ in keys_train}
    for epoch_ in losses_train:
        for key_ in keys_train:
            losses_train_mean[key_].append(np.mean(epoch_[key_]))
            losses_train_std[key_].append(np.std(epoch_[key_]))

    # Plot losses
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

    plt.savefig(res_dir + '/' + title_plot + '.svg',
                format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(res_dir + '/' + title_plot + '.eps',
                format='eps', bbox_inches='tight', transparent=True)
    plt.close()
