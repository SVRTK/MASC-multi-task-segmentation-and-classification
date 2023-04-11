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

    def train_experiment(self,
                         iteration,
                         max_epoch,
                         epoch,
                         segmenter=None,
                         losses_train_seg=None,
                         losses_valid_seg=None,
                         best_metrics_valid_seg=None,
                         binary_seg_weight=0,
                         multi_seg_weight=1,
                         classifier=None,
                         losses_train_class=None,
                         losses_valid_class=None,
                         best_metrics_valid_class=None,
                         multi_task_weight=0
                         ):
        """ Performs training and validation until max_iterations is reached.
            Experiment training based on parameter experiment_type
            Args:
                iteration: current training iteration (int)
                max_epoch: maximum epoch to reach (int)
                epoch: current training epoch (int)
                segmenter: segmenter network (pytorch model)
                losses_train_seg: list of dicts containing segmentation training losses for each epoch (list)
                losses_valid_seg: list of dicts containing segmentation valid losses and metrics for each epoch (list)
                best_metrics_valid_seg: the best segmentation dice on validation set (float)
                binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
                multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
                classifier: classifier network (pytorch model)
                losses_train_class: list of dicts containing classification training losses for each epoch (list)
                losses_valid_class: list of dicts containing classification valid losses and metrics for each epoch (list)
                best_metrics_valid_class: the best classifier accuracy on validation set (float)
                multi_task_weight: weight for our multi-task framework (balance between class and seg loss) (float)
        """

        if self.experiment_type == "classify":
            while epoch < max_epoch:
                iteration, losses_train_class, losses_valid_class, \
                best_metrics_valid_class = self.train_classifier(classifier,
                                                                 iteration,
                                                                 epoch,
                                                                 best_metrics_valid=best_metrics_valid_class,
                                                                 losses_train=losses_train_class,
                                                                 losses_valid=losses_valid_class,
                                                                 segmenter=segmenter
                                                                 )
                epoch += 1
        elif self.experiment_type == "segmenter":
            while epoch < max_epoch:

                # Increase binary weight gradually
                if epoch % 50 == 0.0 and epoch > 0.0:
                    binary_seg_weight += 0.05
                    print("Increasing binary loss by 0.05 W=", binary_seg_weight)

                iteration, losses_train_seg, losses_valid_seg, \
                best_metrics_valid_seg = self.train_segmenter(segmenter,
                                                              iteration,
                                                              epoch,
                                                              binary_seg_weight=binary_seg_weight,
                                                              multi_seg_weight=multi_seg_weight,
                                                              best_metrics_valid=best_metrics_valid_seg,
                                                              losses_train=losses_train_seg,
                                                              losses_valid=losses_valid_seg,
                                                              )

                epoch += 1

        elif self.experiment_type == "joint":
            while epoch < max_epoch:
                iteration, losses_train_seg, losses_valid_seg, \
                losses_train_class, losses_valid_class, \
                best_metrics_valid_class, best_metrics_valid_seg = self.train_joint(classifier,
                                                                                    segmenter,
                                                                                    iteration,
                                                                                    epoch,
                                                                                    binary_seg_weight=binary_seg_weight,
                                                                                    multi_seg_weight=multi_seg_weight,
                                                                                    multi_task_weight=multi_task_weight,
                                                                                    losses_train_seg=losses_train_seg,
                                                                                    losses_valid_seg=losses_valid_seg,
                                                                                    best_metrics_valid_seg=best_metrics_valid_seg,
                                                                                    losses_train_class=losses_train_class,
                                                                                    losses_valid_class=losses_valid_class,
                                                                                    best_metrics_valid_class=best_metrics_valid_class
                                                                                    )
                epoch += 1

    def train_classifier(self,
                         classifier,
                         iteration,
                         epoch,
                         best_metrics_valid=0.0,
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
            losses_train: list of dicts containing training losses for each epoch (list)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
            segmenter: segmentation network (pytorch model)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics (accuracy)
        """

        metrics_train = self.get_training_dict()
        classifier.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, label = (cuda(batch["image"], device_num=self.gpu_device),
                          cuda(batch["label"], device_num=self.gpu_device))
            class_in = self.get_input_classifier(img=img, segmenter=segmenter)

            # Pass through classifier, compute loss and backprop
            pred = classifier(class_in)
            total_loss = self.loss_function_class(pred, label)
            total_loss.backward()
            self.optimizer_class.step()
            self.optimizer_class.zero_grad()

            # Append metrics for this epoch
            metrics_train['total_train_loss_class'].append(total_loss.item())

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

            losses_valid, mean_accuracy = self.valid_classifier(classifier=classifier,
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
                                  best_metric_valid=best_metrics_valid
                                  )

            # Checkpoint best network and overwrite best metric
            if mean_accuracy > best_metrics_valid:
                print(
                    "Classifier Model Was Saved ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}".format(best_metrics_valid, mean_accuracy)
                )

                best_metrics_valid = mean_accuracy

                utils.save_checkpoint(ckpt_name='best_metric_classifier',
                                      ckpt_dir=self.ckpt_dir,
                                      model=classifier,
                                      optimizer=self.optimizer_class,
                                      iteration=iteration,
                                      epoch=epoch,
                                      losses_train=losses_train,
                                      losses_valid=losses_valid,
                                      lr_scheduler=self.lr_scheduler_class,
                                      best_metric_valid=best_metrics_valid
                                      )
            else:
                print(
                    "Classifier Model not best ! Current Best Accuracy: {} "
                    "Current Avg. Accuracy: {}".format(best_metrics_valid, mean_accuracy)
                )

        return iteration, losses_train, losses_valid, best_metrics_valid

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
            updated validation losses lists and accuracy
        """

        metrics_valid = self.get_training_dict(training=False)
        classifier.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        num_correct = 0.0
        metric_count = 0

        for batch in epoch_iterator_val:
            img, label = (cuda(batch["image"], device_num=self.gpu_device),
                          cuda(batch["label"], device_num=self.gpu_device))

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

            metrics_valid['total_valid_loss_class'].append(total_loss.item())

        # Compute epoch accuracy and append epoch metrics
        metric = num_correct / metric_count
        metrics_valid['accuracy'].append(metric)
        losses_valid.append(metrics_valid)

        utils.plot_losses_train(self.res_dir, losses_valid, 'metrics_valid_class_')

        return losses_valid, metric

    def train_segmenter(self,
                        segmenter,
                        iteration,
                        epoch,
                        binary_seg_weight=1,
                        multi_seg_weight=1,
                        best_metrics_valid=0.0,
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
            losses_train: list of dicts containing training losses for each epoch (list)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics (dice)
        """

        metrics_train = self.get_training_dict()
        segmenter.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask = (cuda(batch["image"], device_num=self.gpu_device),
                             cuda(batch["LP"], device_num=self.gpu_device),
                             cuda(batch["mask"], device_num=self.gpu_device))

            # Forward pass, segmentation loss function computation and backprop
            logit_map = segmenter(img)
            total_loss, multi_loss, binary_loss = self.compute_seg_loss(logit_map, mask, LP,
                                                                        multi_seg_weight=multi_seg_weight,
                                                                        binary_seg_weight=binary_seg_weight)
            total_loss.backward()
            self.optimizer_seg.step()
            self.optimizer_seg.zero_grad()

            # Append metrics for this epoch to empty dict
            metrics_train['total_train_loss_seg'].append(total_loss.item())
            metrics_train['multi_train_loss_seg'].append(multi_loss.item())
            metrics_train['binary_train_loss_seg'].append(binary_loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        # Step through scheduler and append epoch metrics to overall metrics
        if self.lr_scheduler_seg:
            self.lr_scheduler_seg.step()

        losses_train.append(metrics_train)
        utils.plot_losses_train(self.res_dir, losses_train, 'losses_train_seg_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid, mean_dice_val = self.valid_segmenter(segmenter, losses_valid=losses_valid)

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
                                  best_metric_valid=best_metrics_valid
                                  )
            # Checkpoint best network and overwrite the best validation metrics
            if mean_dice_val > best_metrics_valid:
                print(
                    "Segmenter Model Was Saved ! Current Best Dice: {} "
                    "Current Avg. Dice: {}".format(
                        best_metrics_valid, mean_dice_val
                    )
                )
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
                                      best_metric_valid=best_metrics_valid
                                      )

            else:
                print(
                    "Segmenter Model not best ! Current Best Dice: {} "
                    "Current Avg. Dice: {}".format(best_metrics_valid, mean_dice_val)
                )

        return iteration, losses_train, losses_valid, best_metrics_valid

    def valid_segmenter(self, segmenter, losses_valid=None):
        """ Validation loop for segmenter. Computes segmentation loss and dice metric on validation set
        Args:
            segmenter: segmenter network (pytorch model)
            losses_valid: list of dicts containing valid losses and metrics for each epoch (list)
        Returns:
            updated validation losses lists and dice
        """

        metrics_valid = self.get_training_dict(training=False)
        segmenter.eval()

        epoch_iterator_val = tqdm(
            self.val_loader, desc="Validate (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        post_label = transforms.AsDiscrete(to_onehot=2)
        dice_metric = transforms.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_vals = list()

        for batch in epoch_iterator_val:
            img, LP, mask = (cuda(batch["image"], device_num=self.gpu_device),
                             cuda(batch["LP"], device_num=self.gpu_device),
                             cuda(batch["mask"], device_num=self.gpu_device))

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
            metrics_valid['multi_valid_loss_seg'].append(multi_loss.item())
            metrics_valid['binary_valid_loss_seg'].append(binary_loss.item())

        dice_metric.reset()

        # Compute mean validation metrics for this epoch
        mean_dice_val = np.mean(dice_vals)
        losses_valid.append(metrics_valid)

        utils.plot_losses_train(self.res_dir, losses_valid, 'metrics_valid_seg_')

        return losses_valid, mean_dice_val

    def train_joint(self,
                    classifier,
                    segmenter,
                    iteration,
                    epoch,
                    binary_seg_weight=1,
                    multi_seg_weight=1,
                    multi_task_weight=1,
                    losses_train_seg=None,
                    losses_valid_seg=None,
                    best_metrics_valid_seg=0.0,
                    losses_train_class=None,
                    losses_valid_class=None,
                    best_metrics_valid_class=0.0
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
            losses_train_seg: list of dicts containing segmentation training losses for each epoch (list)
            losses_valid_seg: list of dicts containing segmentation valid losses and metrics for each epoch (list)
            best_metrics_valid_seg: the best dice on validation set (float)
            losses_train_class: list of dicts containing classification training losses for each epoch (list)
            losses_valid_class: list of dicts containing classification valid losses and metrics for each epoch (list)
            best_metrics_valid_class: the best accuracy on validation set (float)
        Returns:
            latest training iteration, updated training and validation losses lists,
            the best validation metrics (accuracy and dice)

        """

        metrics_train_seg, metrics_train_class = self.get_training_dict()
        segmenter.train()
        classifier.train()

        epoch_iterator = tqdm(
            self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        for batch in epoch_iterator:
            img, LP, mask, label = (cuda(batch["image"], device_num=self.gpu_device),
                                    cuda(batch["LP"], device_num=self.gpu_device),
                                    cuda(batch["mask"], device_num=self.gpu_device),
                                    cuda(batch["label"], device_num=self.gpu_device))

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
            metrics_train_seg['total_train_loss_seg'].append(total_loss_seg.item())
            metrics_train_seg['multi_train_loss_seg'].append(multi_loss.item())
            metrics_train_seg['binary_train_loss_seg'].append(binary_loss.item())
            metrics_train_class['total_train_loss_class'].append(loss_class.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (iteration, self.max_iterations, total_loss)
            )

            iteration += 1

        # Step through LR schedulers
        if self.lr_scheduler_seg:
            self.lr_scheduler_seg.step()

        if self.lr_scheduler_class:
            self.lr_scheduler_class.step()

        losses_train_seg.append(metrics_train_seg)
        losses_train_class.append(metrics_train_class)
        utils.plot_losses_train(self.res_dir, losses_train_seg, 'losses_train_seg_')
        utils.plot_losses_train(self.res_dir, losses_train_class, 'losses_train_class_')

        # >>>>>>>>>>>>>> Validate <<<<<<<<<<<<<<< #
        if (
                epoch % self.eval_num == 0 and epoch != 0
        ) or iteration == self.max_iterations:

            losses_valid_seg, losses_valid_class, \
            mean_dice_val, mean_accuracy = self.valid_joint(segmenter,
                                                            classifier,
                                                            binary_seg_weight=binary_seg_weight,
                                                            multi_seg_weight=multi_seg_weight,
                                                            multi_task_weight=multi_task_weight,
                                                            losses_valid_seg=losses_valid_seg,
                                                            losses_valid_class=losses_valid_class
                                                            )

            # Checkpoint networks
            utils.save_checkpoint(ckpt_name='latest_segmenter',
                                  ckpt_dir=self.ckpt_dir,
                                  model=segmenter,
                                  optimizer=self.optimizer_seg,
                                  iteration=iteration,
                                  epoch=epoch,
                                  losses_train=losses_train_seg,
                                  losses_valid=losses_valid_seg,
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
                                      losses_train=losses_train_seg,
                                      losses_valid=losses_valid_seg,
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

            # Checkpoint the best classifier and overwrite the best validation metric
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
                                      losses_train=losses_train_class,
                                      losses_valid=losses_valid_class,
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

        return iteration, losses_train_seg, losses_valid_seg, \
               losses_train_class, losses_valid_class, \
               best_metrics_valid_class, best_metrics_valid_seg

    def valid_joint(self,
                    segmenter,
                    classifier,
                    binary_seg_weight=1,
                    multi_seg_weight=1,
                    multi_task_weight=1,
                    losses_valid_seg=None,
                    losses_valid_class=None
                    ):
        """ Validation loop for our multi-task framework .
        Computes segmentation dice metric and classifier accuracy on validation set
        Args:
            segmenter: segmenter network (pytorch model)
            classifier: classifier network (pytorch model)
            binary_seg_weight: weight for binary loss (manual labels and joined pred labels) (float)
            multi_seg_weight: weight for multi-class loss (LP and pred labels) (float)
            multi_task_weight: weight for our multi-task framework (balance between class and seg loss) (float)
            losses_valid_seg: list of dicts containing segmentation valid losses and metrics for each epoch (list)
            losses_valid_class: list of dicts containing classification valid losses and metrics for each epoch (list)
        Returns:
            updated validation losses lists, mean accuracy and dice
        """
        metrics_valid_seg, metrics_valid_class = self.get_training_dict(training=False)
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
            img, LP, mask, label = (cuda(batch["image"], device_num=self.gpu_device),
                                    cuda(batch["LP"], device_num=self.gpu_device),
                                    cuda(batch["mask"], device_num=self.gpu_device),
                                    cuda(batch["label"], device_num=self.gpu_device))

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
            metrics_valid_seg['dice_valid'].append(dice)
            metrics_valid_seg['binary_valid_loss_seg'].append(binary_loss)
            metrics_valid_seg['multi_valid_loss_seg'].append(multi_loss)
            metrics_valid_class['total_valid_loss_class'].append(loss_class.item())

        dice_metric.reset()

        # Computing accuracy
        accuracy = num_correct / metric_count
        metrics_valid_class['accuracy'].append(accuracy)

        # Mean epoch metrics
        losses_valid_seg.append(metrics_valid_seg)
        mean_dice_val = np.mean(dice_vals)
        losses_valid_class.append(metrics_valid_class)

        utils.plot_losses_train(self.res_dir, losses_valid_seg, 'metrics_valid_seg')
        utils.plot_losses_train(self.res_dir, losses_valid_class, 'metrics_valid_class')

        return losses_valid_seg, losses_valid_class, mean_dice_val, accuracy
