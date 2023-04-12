import torch
import numpy as np
from monai import transforms
from monai.metrics import compute_meandice, compute_hausdorff_distance, compute_average_surface_distance, \
    get_confusion_matrix, compute_confusion_matrix_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, \
    classification_report, balanced_accuracy_score
import train.train_utils as utils
from train.train_utils import Trainer, cuda
import os
import csv
import nibabel as nib
import matplotlib.pyplot as plt


# Testing loop
class RunTest(Trainer):
    """ Testing class with access to all testing and inference loops
        for both segmentation and classification networks

        Inherits from utils Trainer class
    """

    def test_experiment(self, test_files, test_ds, segmenter=None, classifier=None):
        """ Runs testing depending on experiment type (classifier, or segmenter, or both)
            Args:
                test_files: decathlon datalist test files (decathlon datalist)
                test_ds: pytorch dataset containing test files (pytorch Dataset)
                segmenter: segmenter model to be tested (pytorch model)
                classifier: classifier model to be tested (pytorch model)

        """
        if self.experiment_type == "segment" or "joint":
            self.test_segmenter(model=segmenter, test_files=test_files, test_ds=test_ds)
        if self.experiment_type == "classify" or "joint":
            self.test_classifier(model=classifier, test_files=test_files, test_ds=test_ds)

    def test_segmenter(self, model, test_files, test_ds):
        """ Performs testing on segmenter, and saves metrics (csv) and segmentation predictions in res_dir
            Args:
                model: segmenter model to be tested (pytorch model)
                test_files: decathlon datalist test files (decathlon datalist)
                test_ds: pytorch dataset containing test files (pytorch Dataset)
        """

        model.eval()
        post_label = transforms.AsDiscrete(to_onehot=self.N_seg_labels)
        post_label_binary = transforms.AsDiscrete(to_onehot=2)
        test_sub_metrics = []

        for x in range(len(test_files)):

            case_num = x
            img_name = test_files[case_num]["image"]
            case_name = os.path.split(test_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
            out_name = self.res_dir + "/cnn-lab-" + case_name

            img_tmp_info = nib.load(img_name)

            with torch.no_grad():

                img = test_ds[case_num]["image"]
                label = test_ds[case_num]["mask"]

                # Send to device, add batch size, and forward pass
                val_labels = cuda(label, device_num=self.gpu_device)
                val_inputs = torch.unsqueeze(cuda(img, device_num=self.gpu_device), 1)
                val_outputs = model(val_inputs)

                # Save the prediction
                out_label = torch.argmax(val_outputs, dim=1).detach().cpu()[0, ...]
                out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
                nib.save(out_lab_nii, out_name)

                # Convert predictions to binary ROI
                binary_out = utils.add_labels(torch.argmax(val_outputs, dim=1))
                binary_out = post_label_binary(binary_out).unsqueeze(0)
                val_labels_convert_binary = post_label_binary(utils.add_labels(val_labels)).unsqueeze(0)

                metrics_all = []

                # Compute ROI scores
                dice_ROI_unet = compute_meandice(binary_out, val_labels_convert_binary, include_background=False)[0].cpu().numpy()

                hausdorff_ROI = compute_hausdorff_distance(binary_out, val_labels_convert_binary,
                                                           include_background=False,
                                                           distance_metric='euclidean', percentile=95, directed=False)[0].cpu().numpy()

                avg_surface_dist_ROI = \
                    compute_average_surface_distance(binary_out, val_labels_convert_binary,
                                                     include_background=False,
                                                     symmetric=True, distance_metric='euclidean')[0].cpu().numpy()

                cm = get_confusion_matrix(binary_out.contiguous(),
                                          val_labels_convert_binary.contiguous(),
                                          include_background=False)
                sensitivity_ROI = compute_confusion_matrix_metric("sensitivity", cm)[0].numpy()
                specificity_ROI = compute_confusion_matrix_metric("specificity", cm)[0].numpy()
                precision_ROI = compute_confusion_matrix_metric("precision", cm)[0].numpy()

                metrics_all.extend([i for i in dice_ROI_unet])
                metrics_all.extend([i for i in hausdorff_ROI])
                metrics_all.extend([i for i in avg_surface_dist_ROI])
                metrics_all.extend([i for i in sensitivity_ROI])
                metrics_all.extend([i for i in specificity_ROI])
                metrics_all.extend([i for i in precision_ROI])

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
                #                          Multi-class metrics                       #
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

                val_labels_multi = post_label(val_labels).unsqueeze(0)
                val_outputs_multi = post_label(torch.argmax(val_outputs, dim=1).unsqueeze(0)).unsqueeze(0)

                # Computing dice scores
                dice_multi_unet = compute_meandice(val_outputs_multi, val_labels_multi, include_background=False)[0].cpu().numpy()

                hausdorff_multi = \
                    compute_hausdorff_distance(val_outputs_multi, val_labels_multi, include_background=False,
                                               distance_metric='euclidean', percentile=95, directed=False)[0].cpu().numpy()

                avg_surface_dist_multi = \
                    compute_average_surface_distance(val_outputs_multi, val_labels_multi,
                                                     include_background=False,
                                                     symmetric=True, distance_metric='euclidean')[0].cpu().numpy()

                metrics_all.extend([i for i in dice_multi_unet])
                metrics_all.extend([i for i in hausdorff_multi])
                metrics_all.extend([i for i in avg_surface_dist_multi])

                pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI", "precision ROI"]
                pred_names.extend(["Dice label " + str(i + 1) for i in range(self.N_seg_labels - 1)])
                pred_names.extend(["HD95 label " + str(i + 1) for i in range(self.N_seg_labels - 1)])
                pred_names.extend(["ASD label " + str(i + 1) for i in range(self.N_seg_labels - 1)])
                pred_names.append("casenum")

                metrics_all.append(case_name[:-7])

                # Save to csv
                if os.path.isfile(self.res_dir + 'seg_overlap_metrics.csv'):
                    with open(self.res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(metrics_all)

                else:
                    with open(self.res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(pred_names)
                        writer.writerow(metrics_all)

                test_sub_metrics.append(metrics_all[:-1])

        # Compute averages and save
        avg_metrics = np.mean(test_sub_metrics, axis=0)
        std_metrics = np.std(test_sub_metrics, axis=0)
        avg_names = ["Average scores"]
        std_names = ["STDEV scores"]

        with open(self.res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(avg_names)
            writer.writerow(avg_metrics)
            writer.writerow(std_names)
            writer.writerow(std_metrics)

    def test_classifier(self, model, test_files, test_ds, segmenter=None):
        """ Performs testing on classifier. Metrics are saved in a csv in res_dir
            Args:
                model: classifier model for testings (pytorch model)
                test_files: decathlon datalist testing files (decathlon datalist)
                test_ds: pytorch dataset containing testing files (pytorch Dataset)
                segmenter: segmenter model to use for input to classifier (pytorch model)
        """

        model.eval()
        y_true, y_preds, y_preds_softmax = [], [], []

        for x in range(len(test_files)):
            case_num = x
            label = test_ds[case_num]["label"]
            val_labels = cuda(label, device_num=self.gpu_device)
            img = test_ds[case_num]["image"]
            img = cuda(torch.unsqueeze(img, 1), device_num=self.gpu_device)
            y_true.append(val_labels.item())

            class_in = self.get_input_classifier(img=img, segmenter=segmenter)

            with torch.no_grad():
                logits = model(class_in)

            # Softmax preds
            outputs_softmax = torch.squeeze(logits.softmax(dim=1), 0)
            y_preds_softmax.append(outputs_softmax.detach().cpu().numpy())

            # Argmax outputs for metrics
            val_outputs = logits.argmax(dim=1)
            y_preds.append(val_outputs.item())

        y_preds_softmax = np.array(y_preds_softmax)

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_preds)
        labels = ["CoA", "RAA", "DAA"]

        # Generate classification metrics report and save
        all_metrics_report = classification_report(y_true, y_preds, target_names=labels, sample_weight=None,
                                                   output_dict=True)
        print(all_metrics_report)
        # open file for writing, "w" is writing
        w = csv.writer(open(self.res_dir + "classification_report.csv", "w"))

        # Convert labelled numbers to conditions
        y_preds_conds = list(map(utils.convert_num_to_cond, y_preds))
        y_true_conds = list(map(utils.convert_num_to_cond, y_true))

        # loop over dictionary keys and values
        for key, val in all_metrics_report.items():
            # write every key and value to file
            w.writerow([key, val])
        w.writerow(["Balanced accuracy", balanced_acc])
        w.writerow(["True labels", y_true])
        w.writerow(["Pred labels", y_preds])
        w.writerow(["softmax", y_preds_softmax])

        # Plotting the confusion matrix and saving plot
        cm = confusion_matrix(y_true_conds, y_preds_conds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        plt.figure()
        disp.plot()
        plt.savefig(self.res_dir + '/confusion_matrix.png', dpi=200, bbox_inches='tight', transparent=True)
        plt.savefig(self.res_dir + '/confusion_matrix.eps', format='eps')
        plt.show()

        # Multi-label confusion matrix (compute CM for each class or sample, one-vs-rest)
        cm_multi = multilabel_confusion_matrix(y_true_conds, y_preds_conds, labels=labels)

        for ind, cm_ind in enumerate(cm_multi):
            # Plotting the confusion matrix and saving plot
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_ind, display_labels=["all", labels[ind]])
            plt.figure()
            disp.plot()
            plt.savefig(self.res_dir + '/' + str(labels[ind]) + '_confusion_matrix.png', dpi=200, bbox_inches='tight',
                        transparent=True)
            plt.savefig(self.res_dir + '/' + str(labels[ind]) + '_confusion_matrix.eps', format='eps')
            plt.show()

    def infer(self, model, test_files, test_ds, classifier=None):
        """ Performs inference on segmenter and classifier.
            Segmentation predictions are saved in res_dir, with predicted label in the filename
            Args:
                model: segmenter model for inference (pytorch model)
                test_files: decathlon datalist inference files (decathlon datalist)
                test_ds: pytorch dataset containing inference files (pytorch Dataset)
                classifier: classifier model (pytorch model)
        """

        model.eval()
        post_label = transforms.AsDiscrete(to_onehot=self.N_seg_labels)

        for x in range(len(test_files)):

            case_num = x
            img_name = test_files[case_num]["image"]
            case_name = os.path.split(test_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]


            img_tmp_info = nib.load(img_name)

            with torch.no_grad():

                img = test_ds[case_num]["image"]
                # Send to device, add batch size, and forward pass
                val_inputs = torch.unsqueeze(cuda(img, device_num=self.gpu_device), 1)
                val_outputs = model(val_inputs)

                if classifier:
                    class_in = self.get_input_classifier(img=val_inputs, segmenter=model)

                    with torch.no_grad():
                        logits = classifier(class_in)
                        class_out = logits.argmax(dim=1)
                        class_out = utils.convert_num_to_cond(val_outputs.item())

                out_name = self.res_dir + "/cnn-lab-" + class_out + "-" + case_name
                # Save the prediction
                out_label = torch.argmax(val_outputs, dim=1).detach().cpu()[0, ...]
                out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
                nib.save(out_lab_nii, out_name)
