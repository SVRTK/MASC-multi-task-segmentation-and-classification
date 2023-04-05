import torch
import torch.nn as nn
import numpy as np
import tqdm

from networks.losses import DiceCEsoft
from train import utils


class Trainer:
    """ Parent class with access to all training and validation utilities and functions

    """
    def __init__(
            self,
            optimizer,
            train_loader,
            val_loader,
            max_iterations,
            ckpt_dir,
            res_dir,
            segmenter,
            classifier,
            experiment_type="segment",
            N_classes_segment=12,
            N_classes_class=3,
            loss_function_seg=DiceCEsoft(),
            loss_function_class=nn.CrossEntropyLoss(),
            binary_seg_weight=1.0,
            multi_seg_weight=1.0,
            input_type_class="multi",
            eval_num=1
    ):
        """
        Args:
            optimizer: network pytorch optimizer.
            train_loader: dataloader for training.
            val_loader: dataloader for validation.
            max_iterations: maximum number of training iterations.
            ckpt_dir: directory to save model checkpoints
            res_dir: directory to save inference predictions and test set metrics
            segmenter: segmentation network to train/evaluate
            classifier: classifier network to train/evaluate
            experiment_type: (str) defines experiment type
                            one of:
                                    "segment" (only train segmenter),
                                    "classify" (only train classifier),
                                    "joint" (multi-task joint classifier + segmenter)
                                    "LP" (VoxelMorph Label Propagation)
                            default "segment"
            N_classes_segment: number of segmentation labels (default 12)
            N_classes_class: number of diagnoses for classifier (default 3)
            loss_function_seg: segmentation loss function (default DiceCE)
            loss_function_class: classification loss function (default CE)
            binary_seg_weight: weight for binary segmentation loss 
            multi_seg_weight: weight for multi-class segmentation loss
            input_type_class: str defining expected input to classifier. One of:
                            "multi" = use multi-class segmentation labels,
                            "binary" = use binary segmentation labels (add multi-class preds)
                            "img" = use input volume image
            eval_num: number of epochs between each validation loop (default 1).
        """
        super().__init__()

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_iterations = max_iterations
        self.ckpt_dir = ckpt_dir
        self.res_dir = res_dir
        self.eval_num = eval_num
        self.segmenter = segmenter
        self.classifier = classifier
        self.experiment_type = experiment_type
        self.input_type_class = input_type_class
        self.N_classes_segment = N_classes_segment
        self.N_classes_class = N_classes_class
        self.loss_function_class = loss_function_class
        self.loss_function_seg = loss_function_seg
        self.binary_seg_weight = binary_seg_weight
        self.multi_seg_weight = multi_seg_weight

        exp_names = ["segment", "classify", "joint", "LP"]
        in_class_names = ["multi", "binary", "img"]

        if self.experiment_type not in exp_names:
            raise ValueError("experiment_type parameter"
                             "should be either {}".format(exp_names))

        if self.input_type_class not in in_class_names:
            raise ValueError("input_type_class parameter"
                             "should be either {}".format(exp_names))


    def add_softmax_labels(self, softmax_preds):
        """ Returns added multi-class foreground softmax predictions (background excluded)
            Assumes background in first channel

        Args:
            softmax_preds: multi-class softmax network segmentation predictions (shape BNH[WD])
        Returns: torch tensor with original image shape and two channels, background and foreground
        """

        added_preds = torch.sum(softmax_preds[:, 1:], dim=1)
        added_preds = torch.cat([softmax_preds[:, 0, ...].unsqueeze(1), added_preds.unsqueeze(1)], dim=1)

        return added_preds

    def compute_seg_loss(self, logit_map, mask, LP):
        """Computes total segmentation loss combining multi-class propagated labels and binary
        Args:
            mask: binary mask torch tensor
            LP: multi-class vessel mask torch tensor
        Returns: total segmentation loss, binary loss, multi-class loss
        """
        pred = torch.softmax(logit_map, dim=1)
        multi_loss = self.loss_function_seg(pred, LP)
        binary_loss = self.loss_function_seg(self.add_softmax_labels(pred), mask)
        total_loss_seg = self.binary_seg_weight*binary_loss + self.multi_seg_weight*multi_loss

        return total_loss_seg, multi_loss, binary_loss




    # POTENTIALLY TO DELETE
    def set_train(self):
        if self.experiment_type == "segment":
            self.segmenter.train()
        elif self.experiment_type == "classify":
            self.classifier.train()
        elif self.experiment_type == "joint":
            self.segmenter.train()
            self.classifier.train()

    def get_input_net(self, batch, segmenter=None):
        img, mask, LP, diagnosis = (
            batch["image"].cuda(), batch["mask"].cuda(), batch["LP"].cuda(), batch["label"].cuda())

        if self.experiment_type == "segment":
            input_net = img
            label = [LP, mask]

        elif self.experiment_type == "classify":
            input_net = self.get_input_classifier(img=img, segmenter=segmenter)
            label = diagnosis

        elif self.experiment_type == "joint":
            input_net = self.get_input_classifier(img=img, segmenter=segmenter)
            label = [LP, mask, diagnosis]

        return input_net, label

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
            class_in = self.add_softmax_labels(class_in)

        return class_in

    def train_segmenter(self):
        self.segmenter.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for step, batch in enumerate(epoch_iterator):
            step += 1
            img, LP, mask = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda())
            logit_map = self.segmenter(img)



            loss_train = self.loss_function_class(logit_map, diagnosis)
            loss_train.backward()

            if lr_scheduler is not None:
                metrics_class['lr'].append(self.optimizer.param_groups[0]['lr'])

            metrics_class['train_loss'].append(loss_train.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, loss_train)
            )

            iteration += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            losses_train.append(metrics_class)
            utils.plot_losses_train(self.res_dir, losses_train, 'fig_losses_train_')

    def get_loss_func(self):
        if self.experiment_type == "segment":
            loss_multi_seg = self.loss_function_seg()
            loss_bin_seg =
            total_loss = self.loss_function_seg()
            # RETURN LOSS BASED ON WEIGHTINGS FOR EACH PART (BINARY AND MULTI)
        elif self.experiment_type == "classify":



class TrainNet(Trainer):
    def train(self,
              iteration,
              epoch,
              metric_val_best,
              loss_val_best,
              metrics_class,
              metrics_valid,
              lr_scheduler=None,
              losses_train_init=None,
              losses_valid_init=None,
              ):

        # Setting values from previous epochs
        step = 0
        best_metric = metric_val_best
        best_loss = loss_val_best
        losses_train = losses_train_init
        losses_valid = losses_valid_init

        model = self.classifier if self.experiment_type == "classifier" else self.segmenter
        model.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for step, batch in enumerate(epoch_iterator):
            step += 1
            input_net, label = self.get_input_net(batch, segmenter=self.segmenter)
            logit_map = model(input_net)


            # HERE: FUNCTION TO GET APPROPRIATE LOSS FUNCTION AND LABEL FOR EACH EXPERIMENT
            # RETURNS TOTAL LOSS, AND BACKPROP HAPPENS

            loss_train = self.loss_function_class(logit_map, diagnosis)


            loss_train.backward()

            if lr_scheduler is not None:
                metrics_class['lr'].append(self.optimizer.param_groups[0]['lr'])

            metrics_class['train_loss'].append(loss_train.item())

            # HERE FUNCTION WHICH STEPS THROUGH APPROPRIATE OPTIMIZER
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, loss_train)
            )

            iteration += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            losses_train.append(metrics_class)
            utils.plot_losses_train(self.res_dir, losses_train, 'fig_losses_train_')


class TrainClassifier(Trainer):
    def get_input_classifier(self, img=None, segmenter=None):
        """ Generates input tensor to classifier based on input_type_class parameter
            Args:
                img: original image tensor - default
                pred_seg: softmax segmentation predictions (multi-class)
            Returns torch tensor to be used as input to classifier
        """
        if self.input_type_class == "img" or not segmenter:
            class_in = img
        elif self.input_type_class == "multi":
            class_in = torch.softmax(segmenter(img), dim=1)
        elif self.input_type_class == "binary":
            class_in = torch.softmax(segmenter(img), dim=1)
            class_in = self.add_softmax_labels(class_in)

        return class_in

    def train(self,
              model,
              iteration,
              epoch,
              metric_val_best,
              loss_val_best,
              metrics_class,
              metrics_valid,
              lr_scheduler=None,
              losses_train_init=None,
              losses_valid_init=None,
              segmenter=None,
              ):
        step = 0

        # Setting values from previous epochs
        best_metric = metric_val_best
        best_loss = loss_val_best
        losses_train = losses_train_init
        losses_valid = losses_valid_init

        model.train()
        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for step, batch in enumerate(epoch_iterator):
            step += 1
            img, diagnosis = (batch["image"].cuda(), batch["label"].cuda())

            class_in = self.get_input_classifier(img=img, segmenter=segmenter)
            logit_map = model(class_in)

            loss_train = self.loss_function_class(logit_map, diagnosis)
            loss_train.backward()

            if lr_scheduler is not None:
                metrics_class['lr'].append(self.optimizer.param_groups[0]['lr'])

            metrics_class['train_loss'].append(loss_train.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, loss_train)
            )

            iteration += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            losses_train.append(metrics_class)
            utils.plot_losses_train(self.res_dir, losses_train, 'fig_losses_train_')





