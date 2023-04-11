import torch
import numpy as np
from monai import transforms
import train_utils as utils
from train_utils import Trainer, cuda
import os
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
        post_pred = transforms.AsDiscrete(argmax=True, to_onehot=self.N_seg_labels)
        argmax_only = transforms.AsDiscrete(argmax=True, to_onehot=self.N_seg_labels)

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

                cond = int(test_ds[case_num]["label"])

                # Send to device and adding batch size for forward pass
                val_labels = label.cuda()
                val_inputs = torch.unsqueeze(img.cuda(), 1)

                val_outputs = model(val_inputs)

                 # Save the prediction
                # CONTINUE FROM HERE
                    out_label = tf(torch.argmax(val_outputs, dim=1).detach().cpu())[0, :, :, :]
                    out_lab_nii = nib.Nifti1Image(out_label, img_tmp_info.affine, img_tmp_info.header)
                    nib.save(out_lab_nii, out_name)

                    # Convert predictions to binary ROI
                    val_outputs_convert_binary = tf(post_label_binary(
                        torch.from_numpy(add_labels(torch.argmax(val_outputs, dim=1).detach().cpu().numpy())))).unsqueeze(0)

                else:
                    val_outputs = test_ds[case_num]["LP"]
                    out_label = test_ds[case_num]["LP"]
                    val_outputs_convert_binary = post_label_binary(
                        torch.from_numpy(add_labels(val_outputs.detach().cpu().numpy()))).unsqueeze(0)

                # Get rid of batch size 1 for one hot encoding
                metrics_all = []

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
                #                           Binary ROI metrics                       #
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

                # Converting multi-class label to binary ROI
                val_labels_convert_binary = post_label_binary(
                    torch.from_numpy(add_labels(val_labels.detach().cpu().numpy()))).unsqueeze(0)

                # Compute ROI scores
                dice_ROI_unet = \
                    compute_meandice(val_outputs_convert_binary, val_labels_convert_binary, include_background=False)[
                        0].cpu().numpy()

                hausdorff_ROI = \
                    compute_hausdorff_distance(val_outputs_convert_binary, val_labels_convert_binary,
                                               include_background=False,
                                               distance_metric='euclidean', percentile=95, directed=False)[0].cpu().numpy()

                avg_surface_dist_ROI = \
                    compute_average_surface_distance(val_outputs_convert_binary, val_labels_convert_binary,
                                                     include_background=False,
                                                     symmetric=True, distance_metric='euclidean')[0].cpu().numpy()

                cm = get_confusion_matrix(val_outputs_convert_binary.contiguous(), val_labels_convert_binary.contiguous(),
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

                print("Dice ROI= {}".format(dice_ROI_unet))
                print(" ")
                print("HD95 ROI = {}".format(hausdorff_ROI))
                print(" ")
                print("Avg. surface distance ROI= {}".format(avg_surface_dist_ROI))
                print(" ")
                print(" ------------------------------------------------------ ")

                # Comment out when running multi
                # test_sub_metrics.append(metrics_all[:-1])

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
                #                          Multi-class metrics                       #
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

                if N_classes > 2:
                    print("multi-class")
                    pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI",
                                  "precision ROI", "casenum"]

                    # >> Multi-class labels
                    val_labels_convert = post_label(torch.from_numpy(val_labels.detach().cpu().numpy())).unsqueeze(0)

                    # Multi-class scores
                    if LP:
                        out_label_lp = test_ds[case_num]["LP"]

                        plt.figure()
                        plt.title("out_label_lp")
                        plt.imshow(out_label_lp.numpy()[0, :, 40, :])
                        plt.show()

                        val_output_convert = post_label(out_label_lp).unsqueeze(0)
                    else:
                        val_output_convert = post_label(out_label.unsqueeze(0)).unsqueeze(0).cpu()

                    # Computing dice scores
                    dice_multi_unet = compute_meandice(val_output_convert, val_labels_convert, include_background=False)[
                        0].cpu().numpy()

                    hausdorff_multi = \
                        compute_hausdorff_distance(val_output_convert, val_labels_convert, include_background=False,
                                                   distance_metric='euclidean', percentile=95, directed=False)[
                            0].cpu().numpy()

                    avg_surface_dist_multi = \
                        compute_average_surface_distance(val_output_convert, val_labels_convert, include_background=False,
                                                         symmetric=True, distance_metric='euclidean')[0].cpu().numpy()

                    print("Dice = {}, shape = {}".format(dice_multi_unet, dice_multi_unet.shape))
                    print(" ")
                    print("HD95 = {}, shape = {}".format(hausdorff_multi, hausdorff_multi.shape))
                    print(" ")
                    print("Avg. surface distance = {}, shape = {}".format(avg_surface_dist_multi,
                                                                          avg_surface_dist_multi.shape))
                    print(" ")
                    print(" ------------------------------------------------------ ")

                    metrics_all.extend([i for i in dice_multi_unet])
                    metrics_all.extend([i for i in hausdorff_multi])
                    metrics_all.extend([i for i in avg_surface_dist_multi])

                    pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI",
                                  "precision ROI"]
                    pred_names.extend(["Dice label " + str(i + 1) for i in range(N_classes - 1)])
                    pred_names.extend(["HD95 label " + str(i + 1) for i in range(N_classes - 1)])
                    pred_names.extend(["ASD label " + str(i + 1) for i in range(N_classes - 1)])
                    pred_names.append("casenum")


                else:

                    pred_names = ["Dice ROI", "HD95 ROI", "avg dist. ROI", "sensitivity ROI", "specificity ROI",
                                  "precision ROI", "casenum"]

                metrics_all.append(case_name[:-7])
                # Save to csv
                if os.path.isfile(res_dir + 'seg_overlap_metrics.csv'):
                    with open(res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(metrics_all)

                else:
                    with open(res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(pred_names)
                        writer.writerow(metrics_all)

                test_sub_metrics.append(metrics_all[:-1])

        # Compute averages and save
        avg_metrics = np.mean(test_sub_metrics, axis=0)
        std_metrics = np.std(test_sub_metrics, axis=0)
        avg_names = ["Average scores"]
        std_names = ["STDEV scores"]

        with open(res_dir + 'seg_overlap_metrics.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(avg_names)
            writer.writerow(avg_metrics)
            writer.writerow(std_names)
            writer.writerow(std_metrics)

        # Plot TSNE
        test_set_feature_rep = np.concatenate(test_set_feature_rep, axis=0)
        print("concatenated features shape", test_set_feature_rep.shape)
        # Change dir
        save_dir = res_dir
        # Save array of filenames and save concatenated BN features
        np.save(save_dir + "image_fns_test", image_fns)
        np.save(save_dir + "bn_test1", test_set_feature_rep)
        np.save(save_dir + "labels_test", labels)

        return 0