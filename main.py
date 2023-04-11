from train.train import RunTrain
from networks.prepare_networks import get_nets
import os


# TODO Dataloaders
# TODO Inference and testing
# TODO Label propagation network
# TODO LP training
# TODO LP eval and inference
# TODO argparsing and reading config file
# TODO check everything is documented
# TODO Test all loops, that it trains correctly
# TODO Add a ReadME
# TODO Requirements.txt
# TODO setup.py

CONFIG = {
    # Directories & filenames
    'ckpt_dir': './Checkpoints/',
    'res_dir': './Results/',
    'data_dir': '/Data/',
    'data_JSON_file': 'data.json',
    'ckpt_name_seg': 'latest_segmenter',
    'ckpt_name_class': 'latest_classifier',

    # Experiment types
    'experiment_type': 'joint',
    'input_type_class': 'multi',
    'training': True,
    'infer': True,

    # Experiment parameters
    'eval_num': 3,
    'max_iterations': 10000,
    'batch_size': 12,
    'gpu_ids': 0,

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
    'binary_seg_weight': 1,
    'multi_seg_weight': 1,
    'multi_task_weight': 14,

    # Data parameters
    'spatial_dims': 3,
    'N_diagnosis': 3,
    'N_seg_labels': 12
}


def train(config):
    """ Runs training from config file
        Args:
            config: dictionary (main config file) with experiment parameter info (dict)
    """
    # Make checkpoint and results dir
    if not os.path.isdir(config['ckpt_dir']):
        os.makedirs(config['ckpt_dir'])

    if not os.path.isdir(config['res_dir']):
        os.makedirs(config['res_dir'])

    # Define networks, optimizers and load any existing checkpoints, prepare lists to store losses
    segmenter, optimizer_seg, lr_scheduler_seg, \
    classifier, optimizer_class, lr_scheduler_class, \
    iteration, epoch, max_epoch, \
    losses_train_init_seg, losses_valid_init_seg, best_metric_seg, binary_seg_weight, \
    losses_train_init_class, losses_valid_init_class, best_metric_class = get_nets(config)

    # Set up Trainer class
    runtrain = RunTrain(train_loader=None,
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
                        eval_num=config['eval_num'],
                        gpu_device=config['gpu_device']
                        )

    # Train experiment
    if config['training']:
        runtrain.train_experiment(iteration,
                                  max_epoch,
                                  epoch,
                                  segmenter=segmenter,
                                  losses_train_seg=losses_train_init_seg,
                                  losses_valid_seg=losses_valid_init_seg,
                                  best_metrics_valid_seg=best_metric_seg,
                                  binary_seg_weight=binary_seg_weight,
                                  multi_seg_weight=config['multi_seg_weight'],
                                  classifier=classifier,
                                  losses_train_class=losses_train_init_class,
                                  losses_valid_class=losses_valid_init_class,
                                  best_metrics_valid_class=best_metric_class,
                                  multi_task_weight=config['multi_task_weight']
                                  )


if __name__ == '__main__':
    train(CONFIG)
