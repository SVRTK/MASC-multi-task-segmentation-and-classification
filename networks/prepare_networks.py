from train import train_utils
from monai.networks.nets import AttentionUnet, DenseNet121
import torch


def get_nets(config):
	""" Defines required networks, optimizers, and loads pre-trained weights and losses
		Args:
			config: dictionary (main config file) with experiment parameter info (dict)
		Returns:
			segmenter, segmenter optimizer and LR scheduler
			classifier, classifier optimizer and LR scheduler
			iteration, epoch, maximum epoch,
			segmentation training and validation losses list of dicts,
			best segmentation dice, latest binary segmentation loss weight,
			classifier training and validation losses list of dicts,
			best classifier accuracy
	"""

	# Define empty networks
	segmenter = None
	optimizer_seg = None
	lr_scheduler_seg = None
	classifier = None
	optimizer_class = None
	lr_scheduler_class = None

	losses_train_init_class, losses_valid_init_class = [], []
	losses_train_init_seg, losses_valid_init_seg = [], []
	binary_seg_weight = float(config['binary_seg_weight'])
	best_metric_seg = 1e-5
	best_metric_class = 1e-5
	iteration = 0
	epoch = 0

	max_epoch = int(config['max_iterations']) / int(config['batch_size'])

	# Load and define only required networks
	if config['input_type_class'] != 'img':
		segmenter = AttentionUnet(spatial_dims=int(config['spatial_dims']),
								  in_channels=1,
								  out_channels=int(config['N_seg_labels']),
								  channels=config['chann_segnet'],
								  strides=config['strides_segnet'],
								  kernel_size=int(config['ksize_segnet']),
								  up_kernel_size=int(config['up_ksize_segnet']),
								  dropout=float(config['dropout_seg']))

		segmenter = train_utils.init_network(segmenter, [config['gpu_ids']])
		optimizer_seg = torch.optim.AdamW(segmenter.parameters(),
										  lr=float(config['lr_seg']),
										  weight_decay=float(config['weight_decay_seg']))

		lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_seg,
															 lr_lambda=train_utils.LambdaLR(
																 max_epoch, 0, 1).step)

		# Check if model has been pretrained, and load weights and losses
		losses_train_init_seg, losses_valid_init_seg, \
		best_metric_seg,  \
		iteration, epoch, binary_seg_weight = train_utils.try_load_ckpt(config['ckpt_dir'],
																		config['ckpt_name_seg'],
																		segmenter,
																		optimizer_seg,
																		lr_scheduler=lr_scheduler_seg,
																		load_wbin=True)
		if not binary_seg_weight:
			binary_seg_weight = config['binary_seg_weight']

	if config['experiment_type'] == "classify" or config['experiment_type'] == "joint":
		in_channels_class = train_utils.get_in_channels_class(config)
		classifier = train_utils.cuda(DenseNet121(spatial_dims=int(config['spatial_dims']),
												  in_channels=in_channels_class,
												  out_channels=int(config['N_diagnosis']),
												  dropout_prob=float(config['dropout_class'])))

		classifier = train_utils.init_network(classifier, [config['gpu_ids']])

		optimizer_class = torch.optim.AdamW(classifier.parameters(),
											lr=float(config['lr_class']),
											weight_decay=float(config['weight_decay_class']))

		lr_scheduler_class = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_class,
															   lr_lambda=train_utils.LambdaLR(
																   max_epoch, 0, 1).step)

		# Check if model has been pretrained, and load weights and losses
		losses_train_init_class, losses_valid_init_class, \
		best_metric_class, \
		iteration_class, epoch_class, _ = train_utils.try_load_ckpt(config['ckpt_dir'],
																	config['ckpt_name_class'],
																	classifier,
																	optimizer_class,
																	lr_scheduler=lr_scheduler_class)

		if config['experiment_type'] == 'classify':
			iteration = iteration_class
			epoch = epoch_class

	return segmenter, optimizer_seg, lr_scheduler_seg, \
		   classifier, optimizer_class, lr_scheduler_class, \
		   iteration, epoch, max_epoch, \
		   losses_train_init_seg, losses_valid_init_seg, best_metric_seg, binary_seg_weight, \
		   losses_train_init_class, losses_valid_init_class, best_metric_class
