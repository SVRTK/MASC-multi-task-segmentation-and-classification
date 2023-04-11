import torch
import numpy as np
from monai import transforms
from monai.metrics import compute_meandice, compute_hausdorff_distance, compute_average_surface_distance, \
    get_confusion_matrix, compute_confusion_matrix_metric
import train_utils as utils
from train_utils import Trainer, cuda
import os
import csv
import nibabel as nib


# Testing loop
class RunTest(Trainer):
    """ Testing class with access to all testing and inference loops
        for both segmentation and classification networks

        Inherits from utils Trainer class
    """

    def test(self, model, test_files, test_ds):

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

                pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI",
                          "precision ROI", "casenum"]

                # >> Multi-class labels
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

                pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI",
                              "precision ROI"]
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

