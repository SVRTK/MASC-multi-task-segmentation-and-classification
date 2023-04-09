import train
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

    # Classifier parameters
    'dropout_class': 0.2,
    'lr_class': 1e-4,
    'weight_decay_class': 1e-5,

    # Segmenter parameters
    'dropout_seg': 0.2,
    'lr_seg': 1e-3,
    'weight_decay_seg': 1e-5,

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

        # Check if model has been pretrained, and load weights and losses
        losses_train_init_class, losses_valid_init_class, \
        iteration, epoch = train_utils.try_load_ckpt(config['ckpt_dir'],
                                                     config['ckpt_name_class'],
                                                     classifier,
                                                     optimizer_class)

    if config['experiment_type'] == "segment" or "joint":
        segmenter = train_utils.cuda(AttentionUnet(spatial_dims=config['spatial_dims'],
                                                   in_channels=1,
                                                   out_channels=config['N_seg_labels'],
                                                   channels=(32, 64, 128, 256, 512),
                                                   strides=(2,2,2,2),
                                                   kernel_size=3,
                                                   up_kernel_size=3,
                                                   dropout=config['dropout_seg']))

        optimizer_seg = torch.optim.AdamW(segmenter.parameters(),
                                      lr=config['lr_seg'],
                                      weight_decay=config['weight_decay_seg'])

        # Check if model has been pretrained, and load weights and losses
        losses_train_init_seg, losses_valid_init_seg, \
        iteration, epoch = train_utils.try_load_ckpt(config['ckpt_dir'],
                                                     config['ckpt_name_seg'],
                                                     segmenter,
                                                     optimizer_seg)

def load_experiment_from_config(config, classifier=None, segmenter=None,
                                optimizer_class=None, optimizer_seg=None):
    """ Loads experiment and losses depending on experiment type (defined in config)
        Args:
            config: dictionary containing experiment information (dict)
            classifier: classifier network to load (pytorch model)
            segmenter: segmenter network to load (pytorch model)
            optimizer_class: classifier optimizer (pytorch optimizer)
            optimizer_seg: segmenter optimizer (pytorch optimizer)
    """
    # HERE HAVE TO GET NETWORKS
    if classifier:
        ckpt_class = load_checkpoint(ckpt_path='{}latest_classifier.ckpt'.format(
            config['ckpt_dir']))
        classifier.load_state_dict(ckpt_class['model'])
        optimizer_class.load_state_dict(ckpt_class['optimizer'])
        losses_train_init_class = ckpt_class['losses_train']
        losses_valid_init_class = ckpt_class['losses_valid']
        iteration = ckpt_class['iteration']
        epoch = ckpt_class['epoch']
    if segmenter:
        ckpt_seg = load_checkpoint(ckpt_path='{}latest_segmenter.ckpt'.format(
            config['ckpt_dir']))
        segmenter.load_state_dict(ckpt_seg['model'])



    if args.scheduler_class == True:
        lr_scheduler_class.load_state_dict(ckpt['lr_scheduler'])
        print("LR SCHEDULER CLASS", optimizer_class.param_groups[0]['lr'])


best_valid_loss = best_valid_loss,
best_metric_valid = best_metrics_valid
        accuracy_val_best = 1  # ckpt['best_metric']
        loss_val_best = 10000  # ckpt['best_loss']

        # Loading latest model
        ckpt = utils.load_checkpoint(
            '%s/latest_classifier.ckpt' % (os.path.join(ckpt_dir)))

        classifier.load_state_dict(ckpt['model'])
        optimizer_class.load_state_dict(ckpt['optimizer'])
        losses_train_init_class = ckpt['losses_train']
        losses_valid_init_class = ckpt['losses_valid']
        iteration = ckpt['iteration']
        epoch = ckpt['epoch']

        if args.scheduler_class == True:
            lr_scheduler_class.load_state_dict(ckpt['lr_scheduler'])
            print("LR SCHEDULER CLASS", optimizer_class.param_groups[0]['lr'])

        print("loading pre-trained classifier model from iteration {} and epoch {}".format(iteration, epoch))

    except:
        print("starting training classifier from scratch")
        losses_train_init_class = []
        losses_valid_init_class = []
        accuracy_val_best = 1e-6
        loss_val_best = 1e6

    # Set up Trainer class
    trainer = train_utils.Trainer(
                                    train_loader=None,
                                    val_loader=None,
                                    max_iterations=config['max_iterations'],
                                    ckpt_dir=config['ckpt_dir'],
                                    res_dir=config['res_dir'],
                                    experiment_type=config['experiment_type'],
                                    optimizer_seg=None,
                                    optimizer_class=None,
                                    lr_scheduler_seg=None,
                                    lr_scheduler_class=None,
                                    input_type_class=config['input_type_class'],
                                    eval_num=config['eval_num']
                                )

    if config['training']:
        while iteration < config['max_iterations']:


            """
            iteration, losses_train_init, losses_valid_init, dice_val_best, loss_val_best, lr_scheduler = train_ae(
                model,
                loss_function,
                optimizer,
                iteration,
                epoch,
                train_loader_all,
                val_loader,
                loss_val_best,
                dice_val_best,
                args.max_iterations,
                args.eval_num,
                metrics_train,
                metrics_valid,
                ckpt_dir,
                res_dir,
                args.N_classes,
                lr_scheduler=lr_scheduler,
                losses_train_init=losses_train_init,
                losses_valid_init=losses_valid_init,
                input_type_class=args.input_type_class_AE,
                use_condition=args.use_condition
            )
            """

                epoch += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # Read in args

    # Check what experiment

    # Run training loop

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
