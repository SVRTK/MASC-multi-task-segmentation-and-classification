from train import train
from train import train_utils
import os
from monai.networks.nets import AttentionUnet, DenseNet121
import torch

# TODO READ CONFIG FILE
# TODO IMPLEMENT A DEFAULT CONFIG
# TODO IMPORT REQUIRED MODULES
# TODO DEFINE OPTIMIZER, NETWORK, LOSS FUNCS
# TODO - DEFAULT TO BOTH CLASSIFIER AND SEGMENTER
# TODO DEFINE DATALOADERS
# TODO LOAD CHECKPOINTS
# TODO SLOW INCREASE OF BINARY WEIGHT

config = {
    # Directories & filenames
    'ckpt_dir': './Checkpoints/',
    'res_dir': './Results/',
    'ckpt_name_seg': 'latest_segmenter',
    'ckpt_name_class': 'latest_classifier',

    # Experiment types
    'experiment_type': 'joint',
    'input_type_class': 'multi',
    'training': True,

    # Experiment parameters
    'eval_num': 3,
    'max_iterations': 10000,
    'batch_size': 12,

    # Classifier parameters
    'dropout_class': 0.2,
    'lr_class': 1e-4,
    'weight_decay_class': 1e-5,

    # Segmenter parameters
    'dropout_seg': 0.2,
    'lr_seg': 1e-3,
    'weight_decay_seg': 1e-5,
    'chann_segnet': (32, 64, 128, 256, 512),
    'strides_segnet': (2, 2, 2, 2),
    'ksize_segnet': 3,
    'up_ksize_segnet': 3,

    # Data parameters
    'spatial_dims': 3,
    'N_diagnosis': 3,
    'N_seg_labels': 12
}


def train(config):
    # Make checkpoint and results dir
    if not os.path.isdir(config['ckpt_dir']):
        os.makedirs(config['ckpt_dir'])

    if not os.path.isdir(config['res_dir']):
        os.makedirs(config['res_dir'])

    max_epochs = config['max_iterations'] / config['batch_size']

    # Define networks & optimizers for classification and load any pre-trained weights
    if config['experiment_type'] == "classify" or "joint":
        in_channels_class = train_utils.get_in_channels_class(config)

        classifier = train_utils.cuda(DenseNet121(spatial_dims=config['spatial_dims'],
                                                  in_channels=in_channels_class,
                                                  out_channels=config['N_diagnosis'],
                                                  dropout_prob=config['dropout_class']))

        optimizer_class = torch.optim.AdamW(classifier.parameters(),
                                            lr=config['lr_class'],
                                            weight_decay=config['weight_decay_class'])

        lr_scheduler_class = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_class,
                                                               lr_lambda=train_utils.LambdaLR(
                                                                   max_epochs, 0, 1).step)

        # Check if model has been pretrained, and load weights and losses
        losses_train_init_class, losses_valid_init_class, \
        best_metric_class, loss_val_best_class, \
        iteration_class, epoch_class = train_utils.try_load_ckpt(config['ckpt_dir'],
                                                                 config['ckpt_name_class'],
                                                                 classifier,
                                                                 optimizer_class,
                                                                 lr_scheduler_class=lr_scheduler_class)

    if config['experiment_type'] == "segment" or "joint":
        segmenter = train_utils.cuda(AttentionUnet(spatial_dims=config['spatial_dims'],
                                                   in_channels=1,
                                                   out_channels=config['N_seg_labels'],
                                                   channels=config['chann_segnet'],
                                                   strides=config['strides_segnet'],
                                                   kernel_size=config['ksize_segnet'],
                                                   up_kernel_size=config['up_ksize_segnet'],
                                                   dropout=config['dropout_seg']))

        optimizer_seg = torch.optim.AdamW(segmenter.parameters(),
                                          lr=config['lr_seg'],
                                          weight_decay=config['weight_decay_seg'])

        lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_seg,
                                                             lr_lambda=train_utils.LambdaLR(
                                                                 max_epochs, 0, 1).step)

        # Check if model has been pretrained, and load weights and losses
        losses_train_init_seg, losses_valid_init_seg, \
        best_metric_seg, loss_val_best_seg, \
        iteration_seg, epoch_seg = train_utils.try_load_ckpt(config['ckpt_dir'],
                                                             config['ckpt_name_seg'],
                                                             segmenter,
                                                             optimizer_seg,
                                                             lr_scheduler=lr_scheduler_seg)

    # Set up Trainer class
    trainer = train_utils.Trainer(
        train_loader=None,
        val_loader=None,
        max_iterations=config['max_iterations'],
        ckpt_dir=config['ckpt_dir'],
        res_dir=config['res_dir'],
        experiment_type=config['experiment_type'],
        optimizer_seg=optimizer_seg,
        optimizer_class=optimizer_class,
        lr_scheduler_seg=lr_scheduler_seg,
        lr_scheduler_class=lr_scheduler_class,
        input_type_class=config['input_type_class'],
        eval_num=config['eval_num']
    )

    if config['training']:
        # HERE THIS HAS TO BE ADAPTED FOR
        train.RunTrain()(
            iteration,
            max_iteration,
            epoch,
            losses_train,
            losses_valid,
            segmenter=None,
            best_valid_loss=None,
            best_metrics_valid_seg=None,
            binary_seg_weight=0,
            multi_seg_weight=1,
            classifier=None,
            best_metrics_valid_class=None,
            multi_task_weight=0
        )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('hi')
    # Read in args

    # Check what experiment

    # Run training loop
