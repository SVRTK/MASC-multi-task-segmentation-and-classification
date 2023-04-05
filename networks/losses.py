import torch
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete


class DiceCEsoft(torch.nn.Module):
    """
    Compute the DiceCE loss taking as input the softmax preds, not raw logits
    This implementation avoids the softmax being computed twice (would be the case if using pytorch CE)
    Employs Monai DiceCE loss

    Args:
        weight_ce: (float) weight for cross entropy loss (default 1.0)
        weight_dice: (float) weight for dice loss (default 1.0)
        ignore_index: any index to be ignored (default -100, all ignored)
    """

    def __init__(self, weight_ce=1.0, weight_dice=1.0, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(softmax=False, to_onehot_y=True)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        """
        Args:
            pred: softmax prediction of shape BNH[WD], where N is the number of classses
            target: argmax target of shape B1H[WD]

        """

        loss = 0.
        n_batch, n_class = pred.shape[0], pred.shape[1]

        eps = 1e-10
        onehot_tf = AsDiscrete(to_onehot=n_class)

        # For each batch
        for y1, x1 in zip(pred, target):
            x_oh = onehot_tf(target.unsqueeze(0)).long()
            # For each class
            for class_index in range(n_class):
                if class_index == self.ignore_index:
                    n_batch -= 1
                    continue
                loss -= x_oh[class_index, ...] * torch.log(y1[class_index, ...] + eps)

        celoss = loss / n_batch
        celoss = torch.mean(celoss)
        diceloss = self.dice_loss(pred, target)
        loss = self.weight_ce*celoss + self.weight_dice*diceloss

        return loss

