import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from monai import transforms

from networks.losses import DiceCEsoft
from train import utils


class Trainer:
    """ Parent class with access to all training and validation utilities and functions

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
            ckpt_dir: directory to save model checkpoints'
            res_dir: directory to save inference predictions and test set metrics
            experiment_type: (str) defines experiment type
                            one of:
                                    "segment" (only train segmenter),
                                    "classify" (only train classifier),
                                    "joint" (multi-task joint classifier + segmenter)
                                    "LP" (VoxelMorph Label Propagation)
                            default "segment"
            optimizer_seg: segmentation network optimizer.
            optimizer_class: classifier network oprimizer.
            loss_function_seg: segmentation loss function (default DiceCE)
            loss_function_class: classification loss function (default CE)
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
        binary_loss = self.loss_function_seg(self.add_softmax_labels(pred), mask)
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
            class_in = self.add_softmax_labels(class_in)

        return class_in

    def train_classifier(self,
                         classifier,
                         iteration,
                         epoch,
                         metrics_train,
                         metrics_valid=None,
                         best_metrics_valid=0.0,
                         best_valid_loss=1000,
                         lr_scheduler=None,
                         losses_train=None,
                         losses_valid=None,
                         segmenter=None
                         ):

        classifier.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, label = (batch["image"].cuda(), batch["label"].cuda())
            class_in = self.get_input_classifier(img=img, segmenter=segmenter)
            pred = classifier(class_in)

            total_loss = self.loss_function_class(pred, label)
            total_loss.backward()
            self.optimizer_class.step()
            self.optimizer_class.zero_grad()

            metrics_train['total_train_loss'].append(total_loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_class_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_valid_loss, \
            mean_accuracy = self.valid_classifier(classifier=classifier,
                                                  metrics_valid=metrics_valid,
                                                  losses_valid=losses_valid,
                                                  segmenter=segmenter
                                                  )

            # Checkpoint network
            utils.save_checkpoint(ckpt_name='latest_classifier',
                                  ckpt_dir=self.ckpt_dir,
                                  model=classifier,
                                  optimizer=self.optimizer_class,
                                  iteration=iteration,
                                  epoch=epoch,
                                  losses_train=losses_train,
                                  losses_valid=losses_valid,
                                  lr_scheduler=lr_scheduler,
                                  best_valid_loss=best_valid_loss,
                                  best_metric_valid=best_metrics_valid
                                  )
            if mean_accuracy > best_metrics_valid or mean_valid_loss < best_valid_loss:
                print(
                    "Classifier Model Was Saved ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}, current best loss: {}, current loss: {}".format(
                        best_metrics_valid, mean_accuracy, best_valid_loss, mean_valid_loss
                    )
                )

                utils.save_checkpoint(ckpt_name='best_metric_classifier',
                                      ckpt_dir=self.ckpt_dir,
                                      model=classifier,
                                      optimizer=self.optimizer_class,
                                      iteration=iteration,
                                      epoch=epoch,
                                      losses_train=losses_train,
                                      losses_valid=losses_valid,
                                      lr_scheduler=lr_scheduler,
                                      best_valid_loss=best_valid_loss,
                                      best_metric_valid=best_metrics_valid
                                      )
            else:
                print(
                    "Segmenter Model not best ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}, current best loss: {}, current loss: {}".format(
                        best_metrics_valid, mean_accuracy, best_valid_loss, mean_valid_loss
                    )
                )

        return iteration, losses_train, losses_valid, best_metrics_valid, best_valid_loss, lr_scheduler

    def valid_classifier(self,
                         classifier,
                         metrics_valid,
                         losses_valid=None,
                         segmenter=None
                         ):

        classifier.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        num_correct = 0.0
        metric_count = 0
        loss_vals = list()

        for batch in epoch_iterator_val:
            img, label = (batch["image"].cuda(), batch["label"].cuda())
            with torch.no_grad():
                class_in = self.get_input_classifier(img=img, segmenter=segmenter)
                pred = classifier(class_in)

            total_loss = self.loss_function_class(pred, label)
            value = torch.eq(pred.argmax(dim=1), label)
            metric_count += len(value)
            num_correct += value.sum().item()

            epoch_iterator_val.set_description(
                "Validate (loss=%2.5f)" % (total_loss)
            )

            metrics_valid['total_valid_loss'].append(total_loss.item())

        # Computing accuracy
        metric = num_correct / metric_count
        mean_losses = np.mean(loss_vals)
        metrics_valid['accuracy'].append(metric)
        losses_valid.append(metrics_valid)

        utils.plot_losses_train(self.res_dir, losses_valid, 'metrics_valid_class_')

        return losses_valid, mean_losses, metric

    def train_segmenter(self,
                        segmenter,
                        iteration,
                        epoch,
                        metrics_train,
                        metrics_valid=None,
                        lr_scheduler=None,
                        binary_seg_weight=1,
                        multi_seg_weight=1,
                        best_metrics_valid=0.0,
                        best_valid_loss=1000,
                        losses_train=None,
                        losses_valid=None,
                        ):

        segmenter.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda())
            logit_map = segmenter(img)

            total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                        multi_seg_weight=multi_seg_weight,
                                                                        binary_seg_weight=binary_seg_weight)
            total_loss.backward()

            metrics_train['total_train_loss'].append(total_loss.item())
            metrics_train['multi_train_loss'].append(multi_loss.item())
            metrics_train['binary_train_loss'].append(binary_loss.item())

            self.optimizer_seg.step()
            self.optimizer_seg.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_seg_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_multi_loss, \
            mean_dice_val = self.valid_segmenter(segmenter,
                                                 metrics_valid,
                                                 losses_valid_seg=losses_valid
                                                 )

            # Checkpoint network
            utils.save_checkpoint(ckpt_name='latest_segmenter',
                                  ckpt_dir=self.ckpt_dir,
                                  model=segmenter,
                                  optimizer=self.optimizer_seg,
                                  iteration=iteration,
                                  epoch=epoch,
                                  losses_train=losses_train,
                                  losses_valid=losses_valid,
                                  lr_scheduler=lr_scheduler,
                                  binary_seg_weight=binary_seg_weight,
                                  multi_seg_weight=multi_seg_weight,
                                  best_valid_loss=best_valid_loss,
                                  best_metric_valid=best_metrics_valid
                                  )
            if mean_dice_val > best_metrics_valid or mean_multi_loss < best_valid_loss:
                print(
                    "Segmenter Model Was Saved ! Current Best Dice: {} "
                    "Current Avg. Dice: {}, current best multi loss: {}, current multi loss: {}".format(
                        best_metrics_valid, mean_dice_val, best_valid_loss, mean_multi_loss
                    )
                )

                if mean_dice_val > best_metrics_valid:
                    best_metrics_valid = mean_dice_val

                    utils.save_checkpoint(ckpt_name='best_metric_segmenter',
                                          ckpt_dir=self.ckpt_dir,
                                          model=segmenter,
                                          optimizer=self.optimizer_seg,
                                          iteration=iteration,
                                          epoch=epoch,
                                          losses_train=losses_train,
                                          losses_valid=losses_valid,
                                          lr_scheduler=lr_scheduler,
                                          binary_seg_weight=binary_seg_weight,
                                          multi_seg_weight=multi_seg_weight,
                                          best_valid_loss=best_valid_loss,
                                          best_metric_valid=best_metrics_valid
                                          )

                if mean_multi_loss < best_valid_loss:
                    best_valid_loss = mean_multi_loss

                    utils.save_checkpoint(ckpt_name='best_valid_loss_segmenter',
                                          ckpt_dir=self.ckpt_dir,
                                          model=segmenter,
                                          optimizer=self.optimizer_seg,
                                          iteration=iteration,
                                          epoch=epoch,
                                          losses_train=losses_train,
                                          losses_valid=losses_valid,
                                          lr_scheduler=lr_scheduler,
                                          binary_seg_weight=binary_seg_weight,
                                          multi_seg_weight=multi_seg_weight,
                                          best_valid_loss=best_valid_loss,
                                          best_metric_valid=best_metrics_valid
                                          )

            else:
                print(
                    "Segmenter Model not best ! Current Best Dice: {} "
                    "Current Avg. Dice: {}, current best multi loss: {}, current multi loss: {}".format(
                        best_metrics_valid, mean_dice_val, best_valid_loss, mean_multi_loss
                    )
                )

        return iteration, losses_train, losses_valid, best_metrics_valid, best_valid_loss, lr_scheduler

    def valid_segmenter(self,
                        segmenter,
                        metrics_valid,
                        losses_valid=None, ):

        segmenter.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        post_label = transforms.AsDiscrete(to_onehot=2)
        dice_metric = transforms.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_vals = list()
        seg_multi_loss_vals = list()

        for batch in epoch_iterator_val:
            img, LP, mask = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda())

            with torch.no_grad():
                logit_map = segmenter(img)
                total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP)

            binary_out = self.add_softmax_labels(torch.softmax(logit_map, dim=1))
            val_labels_list = transforms.decollate_batch(mask)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = transforms.decollate_batch(binary_out)
            val_output_convert = [
                post_label(torch.argmax(val_pred_tensor, dim=0).unsqueeze(0))
                for val_pred_tensor in val_outputs_list
            ]

            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (dice=%2.5f)" % (dice)
            )

            metrics_valid['dice_valid'].append(dice)
            metrics_valid['total_valid_loss'].append(total_loss.item())
            metrics_valid['multi_valid_loss'].append(multi_loss.item())
            metrics_valid['binary_valid_loss'].append(binary_loss.item())

        dice_metric.reset()

        mean_seg_multi_loss = np.mean(seg_multi_loss_vals)
        mean_dice_val = np.mean(dice_vals)
        losses_valid.append(metrics_valid)

        utils.plot_losses_train(self.res_dir, losses_valid, 'metrics_valid_seg_')

        return losses_valid, mean_seg_multi_loss, mean_dice_val

    def train_joint(self,
                    classifier,
                    segmenter,
                    iteration,
                    epoch,
                    metrics_train,
                    metrics_valid=None,
                    lr_scheduler_seg=None,
                    lr_scheduler_class=None,
                    binary_seg_weight=1,
                    multi_seg_weight=1,
                    multi_task_weight=1,
                    best_metrics_valid_seg=0.0,
                    best_valid_loss_seg=1000,
                    best_metrics_valid_class=0.0,
                    best_valid_loss_class=1000,
                    losses_train=None,
                    losses_valid=None,
                    ):

        segmenter.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda())
            logit_map = segmenter(img)

            total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                        multi_seg_weight=multi_seg_weight,
                                                                        binary_seg_weight=binary_seg_weight)
            total_loss.backward()

            metrics_train['total_train_loss'].append(total_loss.item())
            metrics_train['multi_train_loss'].append(multi_loss.item())
            metrics_train['binary_train_loss'].append(binary_loss.item())

            self.optimizer_seg.step()
            self.optimizer_seg.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_seg_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_multi_loss, \
            mean_dice_val = self.valid_segmenter(segmenter,
                                                 metrics_valid_seg,
                                                 losses_valid_seg=losses_valid
                                                 )

            # Checkpoint network
            utils.save_checkpoint(ckpt_name='latest_segmenter',
                                  ckpt_dir=self.ckpt_dir,
                                  model=segmenter,
                                  optimizer=self.optimizer_seg,
                                  iteration=iteration,
                                  epoch=epoch,
                                  losses_train=losses_train,
                                  losses_valid=losses_valid,
                                  lr_scheduler=lr_scheduler,
                                  binary_seg_weight=binary_seg_weight,
                                  multi_seg_weight=multi_seg_weight,
                                  best_valid_loss=best_valid_loss,
                                  best_metric_valid=best_metrics_valid
                                  )
            if mean_dice_val > best_metrics_valid or mean_multi_loss < best_valid_loss:
                print(
                    "Segmenter Model Was Saved ! Current Best Dice: {} "
                    "Current Avg. Dice: {}, current best multi loss: {}, current multi loss: {}".format(
                        best_metrics_valid, mean_dice_val, best_valid_loss, mean_multi_loss
                    )
                )

                if mean_dice_val > best_metrics_valid:
                    best_metrics_valid = mean_dice_val

                    utils.save_checkpoint(ckpt_name='best_metric_segmenter',
                                          ckpt_dir=self.ckpt_dir,
                                          model=segmenter,
                                          optimizer=self.optimizer_seg,
                                          iteration=iteration,
                                          epoch=epoch,
                                          losses_train=losses_train,
                                          losses_valid=losses_valid,
                                          lr_scheduler=lr_scheduler,
                                          binary_seg_weight=binary_seg_weight,
                                          multi_seg_weight=multi_seg_weight,
                                          best_valid_loss=best_valid_loss,
                                          best_metric_valid=best_metrics_valid
                                          )

                if mean_multi_loss < best_valid_loss:
                    best_valid_loss = mean_multi_loss

                    utils.save_checkpoint(ckpt_name='best_valid_loss_segmenter',
                                          ckpt_dir=self.ckpt_dir,
                                          model=segmenter,
                                          optimizer=self.optimizer_seg,
                                          iteration=iteration,
                                          epoch=epoch,
                                          losses_train=losses_train,
                                          losses_valid=losses_valid,
                                          lr_scheduler=lr_scheduler,
                                          binary_seg_weight=binary_seg_weight,
                                          multi_seg_weight=multi_seg_weight,
                                          best_valid_loss=best_valid_loss,
                                          best_metric_valid=best_metrics_valid
                                          )

            else:
                print(
                    "Segmenter Model not best ! Current Best Dice: {} "
                    "Current Avg. Dice: {}, current best multi loss: {}, current multi loss: {}".format(
                        best_metrics_valid, mean_dice_val, best_valid_loss, mean_multi_loss
                    )
                )

        return iteration, losses_train, losses_valid, best_metrics_valid, best_valid_loss, lr_scheduler
        


