import torch
from torch import nn
delta = 0.10


class AS_Softmax(nn.Module):
    """
    AS-Softmax:
        Training objective:pt − pi ≥ δ;
        δ ∈ [0, 1], if δ=1, AS-Softmax is equivalent to softmax.
    """

    def __init__(self, delta):

        super(AS_Softmax, self).__init__()
        self.delta = delta

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor):  Logits refers to the output of the model after passing through the classifier. Its size is (batch, labels_num).
            labels (torch.Tensor):  Labels refers to the gold labels of input, and its size is (batch).
        Returns:
            as_logits (torch.Tensor):  Logits after AS-Softmax algorithm processing
            labels (torch.Tensor)
        """
        active_logits = logits.view(-1, logits.size(-1))
        active_labels = labels.view(-1)
        logits_softmax = nn.Softmax(dim=-1)(active_logits)
        as_label_mask = active_labels != -100
        as_active_labels = torch.where(as_label_mask, active_labels, 0 * active_labels)
        gold_softmax = torch.gather(logits_softmax, dim=1, index=as_active_labels.view(-1).unsqueeze(1))

        is_lt = (gold_softmax.repeat(1, active_logits.shape[1]) - logits_softmax) <= self.delta
        as_logits = torch.where(is_lt, active_logits, torch.tensor(float('-inf')).type_as(active_logits))
        return as_logits, active_labels


class Multi_label_AS_Softmax(nn.Module):
    """
    AS-Softmax for multi-label classification:
        Training objective:(1)pt^{min} − pi ≥ δ and (2)pt − pi^{max} ≥ δ,
        pt^{min} represents the smallest of all target scores in one sample,
        pi^{max} represents the largest of all non-target scores in one samples;
        δ ∈ [0, 1], if δ=1, AS-Softmax is equivalent to multi-label softmax.
    """

    def __init__(self, delta):
        super(Multi_label_AS_Softmax, self).__init__()
        self.delta = delta

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor):  Logits refers to the output of the model after passing through the classifier. Its size is (batch, labels_num).
            labels (torch.Tensor):  Labels refers to the gold labels of input, and its size is (batch,labels_num). Here, labels are one-hot vectors.
        Returns:
            mask_neg (torch.Tensor):  Mask out the probabilities of meeting training objective (1). Size: (batch,labels_num).
            mask_pos (torch.Tensor):  Mask out the probabilities of meeting training objective (2). Size: (batch,labels_num).
        """
        logits_softmax = nn.Softmax(dim=-1)(logits.detach())
        gold_softmax = (logits_softmax - (1 - labels) * 1e12) > 0
        # get all target scores
        target_logits = torch.where(gold_softmax, logits_softmax, (torch.tensor(float('inf')).type_as(logits_softmax)).to(torch.float))
        # get all non-target scores
        non_target_logits = torch.where(gold_softmax, (torch.tensor(float('-inf')).type_as(logits_softmax)).to(torch.float), logits_softmax)
        # get pi^{max}
        max_non_target_logits = torch.max(non_target_logits, dim=-1)[0].unsqueeze(1)
        # get pt^{min}
        min_target_logits = torch.min(target_logits, dim=-1)[0].unsqueeze(1)
        # if pt^{min} − pi ≥ δ, zi = 0
        mask_neg = (min_target_logits.repeat(1, logits.shape[1]) - logits_softmax) <= self.delta
        # if pt − pi^{max} ≥ δ, zt = 0
        mask_pos = (logits_softmax - max_non_target_logits.repeat(1,logits.shape[1])) <= self.delta

        return mask_neg, mask_pos

    def as_multilabel_categorical_crossentropy(self, y_pred, y_true, mask_neg, mask_pos):
        """
        Args:
            y_pred (torch.Tensor):  Logits refers to the output of the model after passing through the classifier. Its size is (batch, labels_num).
            y_true (torch.Tensor):  Labels refers to the gold labels of input, and its size is (batch,labels_num). Here, y_true are one-hot vectors.
            mask_neg (torch.Tensor):  It means the first output of multi-label-as-softmax and its size is (batch,labels_num).
            mask_pos (torch.Tensor):  It means the second output of multi-label-as-softmax and its size is (batch,labels_num).
        Returns:
            loss (torch.Tensor):  Average loss value of multi-label classification.
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        mask_neg = torch.where(mask_neg, torch.tensor(1).to(mask_neg.device), torch.tensor(0).to(mask_neg.device))
        mask_pos = torch.where(mask_pos, torch.tensor(1).to(mask_pos.device), torch.tensor(0).to(mask_pos.device))
        y_pred_neg = torch.exp(y_pred_neg) * mask_neg
        y_pred_pos = torch.exp(y_pred_pos) * mask_pos
        neg_loss = torch.log(1 + torch.sum(y_pred_neg, dim=-1))
        pos_loss = torch.log(1 + torch.sum(y_pred_pos, dim=-1))

        return torch.mean(neg_loss + pos_loss)


def compute_accumulation_step(logits,lamb, speedup=True):
    """
    Args:
        logits : output
        speedup : control this module

    Returns:
        ignore samples, z_i_num, num of accumulation steps
    """
    logits = logits.view(-1, logits.shape[-1])
    bs_sql = logits.shape[0]
    classes = logits.shape[1]
    z_i_0 = logits == float("-inf")
    z_i_num = z_i_0.sum().item()
    ignore_samples = (torch.sum(z_i_0, dim=-1) ==
                      (classes - 1)).sum().cpu().item()
    if not speedup:
        return ignore_samples, z_i_num, 1
    arg = bs_sql / (bs_sql - ignore_samples + 1e-8)
    return ignore_samples, z_i_num, int(max(1, round(lamb * arg)))


if __name__ == '__main__':
    pass
