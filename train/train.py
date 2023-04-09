import torch
import numpy as np
from tqdm import tqdm
from monai import transforms

from train import train_utils as utils
from train_utils import Trainer, cuda


class RunTrain(Trainer):
    """ Training class with access to all training and validation loops, including
        exclusive training and validation of segmentation and classifier network,
        and joint multi-task training and validation

        Inherits from utils Trainer class
    """

    def train_classifier(self,
                         classifier,
                         iteration,
                         epoch,
                         best_metrics_valid=0.0,
                         best_valid_loss=1000,
                         losses_train=None,
                         losses_valid=None,
                         segmenter=None
                         ):
        """ Training loop for classifier only
        Args:
            classifier: classifier network (pytorch model)
            iteration: training iteration (int)
            epoch: training epoch (int)
            best_metrics_valid: the best accuracy on validation set (float)
            best_valid_loss: minimum validation loss recorded (float)
            losses_train: list of dicts containing training losses for each epoch (list)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
            segmenter: segmentation network (pytorch model)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics and losses
        """

        metrics_train = self.get_training_dict()
        classifier.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, label = (cuda(batch["image"]), cuda(batch["label"]))
            class_in = self.get_input_classifier(img=img, segmenter=segmenter)

            # Pass through classifier, compute loss and backprop
            pred = classifier(class_in)
            total_loss = self.loss_function_class(pred, label)
            total_loss.backward()
            self.optimizer_class.step()
            self.optimizer_class.zero_grad()

            # Append metrics for this epoch
            metrics_train['total_train_loss'].append(total_loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        # Step through scheduler and add epoch metrics to list of all training metrics
        if self.lr_scheduler_class:
            self.lr_scheduler_class.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_class_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_valid_loss, \
            mean_accuracy = self.valid_classifier(classifier=classifier,
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
                                  lr_scheduler=self.lr_scheduler_seg,
                                  best_valid_loss=best_valid_loss,
                                  best_metric_valid=best_metrics_valid
                                  )

            # Checkpoint best network and overwrite best metric
            if mean_accuracy > best_metrics_valid or mean_valid_loss < best_valid_loss:
                print(
                    "Classifier Model Was Saved ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}, current best loss: {}, current loss: {}".format(
                        best_metrics_valid, mean_accuracy, best_valid_loss, mean_valid_loss
                    )
                )

                best_metrics_valid = mean_accuracy
                best_valid_loss = mean_valid_loss

                utils.save_checkpoint(ckpt_name='best_metric_classifier',
                                      ckpt_dir=self.ckpt_dir,
                                      model=classifier,
                                      optimizer=self.optimizer_class,
                                      iteration=iteration,
                                      epoch=epoch,
                                      losses_train=losses_train,
                                      losses_valid=losses_valid,
                                      lr_scheduler=self.lr_scheduler_class,
                                      best_valid_loss=best_valid_loss,
                                      best_metric_valid=best_metrics_valid
                                      )
            else:
                print(
                    "Classifier Model not best ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}, current best loss: {}, current loss: {}".format(
                        best_metrics_valid, mean_accuracy, best_valid_loss, mean_valid_loss
                    )
                )

        return iteration, losses_train, losses_valid, best_metrics_valid, best_valid_loss

    def valid_classifier(self,
                         classifier,
                         losses_valid=None,
                         segmenter=None
                         ):
        """ Validation loop for classifier. Computes classifier loss and accuracy on validation set
        Args:
            classifier: classifier network (pytorch model)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
            segmenter: segmentation network (pytorch model)
        Returns:
            updated validation losses lists, mean loss, and accuracy
        """

        metrics_valid = self.get_training_dict(training=False)
        classifier.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        num_correct = 0.0
        metric_count = 0
        loss_vals = list()

        for batch in epoch_iterator_val:
            img, label = (cuda(batch["image"]), cuda(batch["label"]))

            # Forward pass and validation loss computation
            with torch.no_grad():
                class_in = self.get_input_classifier(img=img, segmenter=segmenter)
                pred = classifier(class_in)
                total_loss = self.loss_function_class(pred, label)

            # Count the number of correct classifications
            value = torch.eq(pred.argmax(dim=1), label)
            metric_count += len(value)
            num_correct += value.sum().item()

            epoch_iterator_val.set_description(
                "Validate (loss=%2.5f)" % (total_loss)
            )

            metrics_valid['total_valid_loss'].append(total_loss.item())

        # Compute epoch accuracy and append epoch metrics
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
                        binary_seg_weight=1,
                        multi_seg_weight=1,
                        best_metrics_valid=0.0,
                        best_valid_loss=1000,
                        losses_train=None,
                        losses_valid=None,
                        ):
        """ Training loop for segmenter only
        Args:
            segmenter: segmenter network (pytorch model)
            iteration: training iteration (int)
            epoch: training epoch (int)
            binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
            multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
            best_metrics_valid: the best dice on validation set (float)
            best_valid_loss: minimum validation loss recorded (float)
            losses_train: list of dicts containing training losses for each epoch (list)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics and losses
        """

        metrics_train = self.get_training_dict()
        segmenter.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda())

            # Forward pass, segmentation loss function computation and backprop
            logit_map = segmenter(img)
            total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                        multi_seg_weight=multi_seg_weight,
                                                                        binary_seg_weight=binary_seg_weight)
            total_loss.backward()
            self.optimizer_seg.step()
            self.optimizer_seg.zero_grad()

            # Append metrics for this epoch to empty dict
            metrics_train['total_train_loss'].append(total_loss.item())
            metrics_train['multi_train_loss'].append(multi_loss.item())
            metrics_train['binary_train_loss'].append(binary_loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        # Step through scheduler and append epoch metrics to overal metrics
        if self.lr_scheduler_seg:
            self.lr_scheduler_seg.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_seg_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_multi_loss, \
            mean_dice_val = self.valid_segmenter(segmenter,
                                                 losses_valid=losses_valid
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
                                  lr_scheduler=self.lr_scheduler_seg,
                                  binary_seg_weight=binary_seg_weight,
                                  multi_seg_weight=multi_seg_weight,
                                  best_valid_loss=best_valid_loss,
                                  best_metric_valid=best_metrics_valid
                                  )
            # Checkpoint best network and overwrite the best validation metrics
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
                                          lr_scheduler=self.lr_scheduler_seg,
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
                                          lr_scheduler=self.lr_scheduler_seg,
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

        return iteration, losses_train, losses_valid, best_metrics_valid, best_valid_loss

    def valid_segmenter(self, segmenter, losses_valid=None):
        """ Validation loop for segmenter. Computes segmentation loss and dice metric on validation set
        Args:
            segmenter: segmenter network (pytorch model)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            updated validation losses lists, mean multi-class loss and dice
        """

        metrics_valid = self.get_training_dict(training=False)
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

            # Forward pass and loss computation
            with torch.no_grad():
                logit_map = segmenter(img)
                total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP)

            # Add labels to for a single binary ROI mask
            binary_out = utils.add_softmax_labels(torch.softmax(logit_map, dim=1))

            # One-hot encode labels
            val_labels_list = transforms.decollate_batch(mask)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            # Argmax and one-hot encode preds
            val_outputs_list = transforms.decollate_batch(binary_out)
            val_output_convert = [
                post_label(torch.argmax(val_pred_tensor, dim=0).unsqueeze(0))
                for val_pred_tensor in val_outputs_list
            ]

            # Compute dice metric
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (dice=%2.5f)" % (dice)
            )

            metrics_valid['dice_valid'].append(dice)
            metrics_valid['multi_valid_loss'].append(multi_loss.item())
            metrics_valid['binary_valid_loss'].append(binary_loss.item())

        dice_metric.reset()

        # Compute mean validation metrics for this epoch
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
                    binary_seg_weight=1,
                    multi_seg_weight=1,
                    multi_task_weight=1,
                    best_metrics_valid_seg=0.0,
                    best_metrics_valid_class=0.0,
                    losses_train=None,
                    losses_valid=None,
                    ):
        """ Training loop for our multi-task framework (joint segmenter + classifier)
        Args:
            classifier: classifier network (pytorch model)
            segmenter: segmenter network (pytorch model)
            iteration: training iteration (int)
            epoch: training epoch (int)
            binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
            multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
            multi_task_weight: weight for our multi-task framework (balance between class and seg loss) (float)
            best_metrics_valid_seg: the best dice on validation set (float)
            best_metrics_valid_class: the best accuracy on validation set (float)
            losses_train: list of dicts containing training losses for each epoch (list)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics (accuracy and dice)

        """

        metrics_train = self.get_training_dict()
        segmenter.train()
        classifier.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask, label = (
                batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda(), batch["label"].cuda())

            # Pass through segmenter & segmenter loss computation
            logit_map = segmenter(img)
            total_loss_seg, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                            multi_seg_weight=multi_seg_weight,
                                                                            binary_seg_weight=binary_seg_weight)

            # Pass through classifier & classifier loss computation
            class_in = torch.softmax(logit_map, dim=1)
            pred = classifier(class_in)
            loss_class = self.loss_function_class(pred, label)

            # Compute total loss and backprop
            total_loss = total_loss_seg + multi_task_weight * loss_class
            total_loss.backward()
            self.optimizer_seg.step()
            self.optimizer_class.step()
            self.optimizer_class.zero_grad()
            self.optimizer_seg.zero_grad()

            # Append epoch training losses to empty dict
            metrics_train['total_train_loss_seg'].append(total_loss_seg.item())
            metrics_train['multi_train_loss'].append(multi_loss.item())
            metrics_train['binary_train_loss'].append(binary_loss.item())
            metrics_train['train_loss_class'].append(loss_class.item())
            metrics_train['total_loss'].append(total_loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        # Step through LR schedulers
        if self.lr_scheduler_seg:
            self.lr_scheduler_seg.step()

        if self.lr_scheduler_class:
            self.lr_scheduler_class.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_seg_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:
            losses_valid, mean_dice_val, mean_accuracy = self.valid_joint(
                segmenter,
                classifier,
                binary_seg_weight=binary_seg_weight,
                multi_seg_weight=multi_seg_weight,
                multi_task_weight=multi_task_weight,
                losses_valid=losses_valid,
            )

            # Checkpoint networks
            utils.save_checkpoint(ckpt_name='latest_segmenter',
                                  ckpt_dir=self.ckpt_dir,
                                  model=segmenter,
                                  optimizer=self.optimizer_seg,
                                  iteration=iteration,
                                  epoch=epoch,
                                  losses_train=losses_train,
                                  losses_valid=losses_valid,
                                  lr_scheduler=self.lr_scheduler_seg,
                                  binary_seg_weight=binary_seg_weight,
                                  multi_seg_weight=multi_seg_weight,
                                  best_metric_valid=best_metrics_valid_seg
                                  )

            # Checkpoint best segmenter and overwrite the best validation metric
            if mean_dice_val > best_metrics_valid_seg:
                print(
                    "Segmenter Model Was Saved ! Current Best Dice: {} Current Avg. Dice: {}".format(
                        best_metrics_valid_seg, mean_dice_val
                    )
                )

                best_metrics_valid_seg = mean_dice_val

                utils.save_checkpoint(ckpt_name='best_metric_segmenter',
                                      ckpt_dir=self.ckpt_dir,
                                      model=segmenter,
                                      optimizer=self.optimizer_seg,
                                      iteration=iteration,
                                      epoch=epoch,
                                      losses_train=losses_train,
                                      losses_valid=losses_valid,
                                      lr_scheduler=self.lr_scheduler_seg,
                                      binary_seg_weight=binary_seg_weight,
                                      multi_seg_weight=multi_seg_weight,
                                      best_metric_valid=best_metrics_valid_seg
                                      )

            else:
                print(
                    "Segmenter Model Not Best ! Current Best Dice: {} Current Avg. Dice: {}".format(
                        best_metrics_valid_seg, mean_dice_val
                    )
                )

            # Checkpoint best classifier and overwrite the best validation metric
            if mean_accuracy > best_metrics_valid_class:
                print(
                    "Classifier Model Was Saved ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}".format(
                        best_metrics_valid_class, mean_accuracy
                    )
                )

                best_metrics_valid_class = mean_accuracy

                utils.save_checkpoint(ckpt_name='best_metric_classifier',
                                      ckpt_dir=self.ckpt_dir,
                                      model=classifier,
                                      optimizer=self.optimizer_class,
                                      iteration=iteration,
                                      epoch=epoch,
                                      losses_train=losses_train,
                                      losses_valid=losses_valid,
                                      lr_scheduler=self.lr_scheduler_class,
                                      best_metric_valid=best_metrics_valid_class
                                      )
            else:
                print(
                    "Classifier Model not best ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}".format(
                        best_metrics_valid_class, mean_accuracy
                    )
                )

        return iteration, losses_train, losses_valid, \
               best_metrics_valid_class, best_metrics_valid_seg

    def valid_joint(self,
                    segmenter,
                    classifier,
                    binary_seg_weight=1,
                    multi_seg_weight=1,
                    multi_task_weight=1,
                    losses_valid=None,
                    ):
        """ Validation loop for our multi-task framework .
        Computes segmentation dice metric and classfier accuracy on validation set
        Args:
            segmenter: segmenter network (pytorch model)
            classifier: classifier network (pytorch model)
            binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
            multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
            multi_task_weight: weight for our multi-task framework (balance between class and seg loss) (float)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            updated validation losses lists, mean multi-class loss and dice
        """
        metrics_valid = self.get_training_dict(training=False)
        segmenter.eval()
        classifier.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        post_label = transforms.AsDiscrete(to_onehot=2)
        dice_metric = transforms.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_vals = list()
        num_correct = 0.0
        metric_count = 0

        for batch in epoch_iterator_val:
            img, LP, mask, label = (batch["image"].cuda(), batch["LP"].cuda(), batch["mask"].cuda(), batch["label"])

            with torch.no_grad():

                # Pass through segmenter & loss computation
                logit_map = segmenter(img)
                total_loss_seg, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                                multi_seg_weight=multi_seg_weight,
                                                                                binary_seg_weight=binary_seg_weight)
                # Pass through classifier & loss computation
                class_in = torch.softmax(logit_map, dim=1)
                pred = classifier(class_in)
                loss_class = self.loss_function_class(pred, label)

                total_loss = total_loss_seg + multi_task_weight * loss_class

                # Compute classifier metrics
                value = torch.eq(pred.argmax(dim=1), label)
                metric_count += len(value)
                num_correct += value.sum().item()

            # Compute segmentation metrics
            binary_out = utils.add_softmax_labels(torch.softmax(logit_map, dim=1))
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

            # Append validation metrics to empty epoch dict
            metrics_valid['dice_valid'].append(dice)
            metrics_valid['binary_valid_loss'].append(binary_loss)
            metrics_valid['multi_valid_loss'].append(multi_loss)
            metrics_valid['total_valid_loss'].append(total_loss.item())

        dice_metric.reset()

        # Computing accuracy
        accuracy = num_correct / metric_count
        metrics_valid['accuracy'].append(accuracy)

        # Mean epoch metrics
        losses_valid.append(metrics_valid)
        mean_dice_val = np.mean(dice_vals)
        losses_valid.append(metrics_valid)

        utils.plot_losses_train(self.res_dir, losses_valid, 'metrics_valid_joint')

        return losses_valid, mean_dice_val, accuracy

