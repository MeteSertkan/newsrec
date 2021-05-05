import torch
from torchmetrics import Metric
from torchmetrics.functional import auroc


class AUC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("auc", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds.squeeze(dim=0), target.squeeze(dim=0)
        assert preds.shape == target.shape
        self.auc += auroc(preds, target)
        self.count += 1.0

    def compute(self):
        return self.auc / self.count


class NDCG(Metric):
    def __init__(self, dist_sync_on_step=False, k=10):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("ndcg", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def dcg_score(self, y_true, y_score):
        # if sequence smaller than k
        sequence_length = y_score.shape[1]
        if sequence_length < self.k:
            k = sequence_length
        else: 
            k = self.k
        _, order = torch.topk(input=y_score,
                              k=k,
                              largest=True)
        y_true = torch.take(y_true, order)
        gains = torch.pow(2, y_true) - 1
        discounts = torch.log2(torch.arange(y_true.shape[1]).type_as(y_score) + 2.0)
        return torch.sum(gains / discounts)

    def ndcg_score(self, y_true, y_score):
        best = self.dcg_score(y_true, y_true)
        actual = self.dcg_score(y_true, y_score)
        return actual / best

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.ndcg += self.ndcg_score(target, preds)
        self.count += 1.0

    def compute(self):
        return self.ndcg / self.count


class MRR(Metric):
    def __init__(self, dist_sync_on_step=False, k=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mrr", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        if self.k is None:
            order = torch.argsort(input=preds, descending=True)
        else:
            sequence_length = preds.shape[1]
            if sequence_length < self.k:
                k = sequence_length
            else: 
                k = self.k
            _, order = torch.topk(input=preds,
                                  k=k,
                                  largest=True)
        
        y_true = torch.take(target, order)
        rr_score = y_true / (torch.arange(y_true.shape[1]).type_as(preds) + 1)
            
        self.mrr += torch.sum(rr_score) / torch.sum(y_true)
        self.count += 1.0

    def compute(self):
        return self.mrr / self.count


class SentiMRR(Metric):
    def __init__(self, dist_sync_on_step=False, k=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("senti_mrr", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def update(self, y_pred: torch.Tensor, s_c: torch.Tensor, s_mean: torch.Tensor):
        assert y_pred.shape == s_c.shape
        if self.k is None:
            order = torch.argsort(input=y_pred, descending=True)
        else:
            sequence_length = y_pred.shape[0]
            if sequence_length < self.k:
                k = sequence_length
            else: 
                k = self.k
            _, order = torch.topk(input=y_pred,
                                  k=k,
                                  largest=True)
        
        s_c = torch.take(s_c, order)
        senti_rr_score = s_c / (torch.arange(s_c.shape[0]).type_as(s_c) + 1.0)
        senti_rr_score = s_mean * torch.sum(senti_rr_score)
        senti_rr_score = torch.nn.functional.relu(senti_rr_score)
            
        self.senti_mrr += senti_rr_score
        self.count += 1.0

    def compute(self):
        return self.senti_mrr / self.count


class Senti(Metric):
    def __init__(self, dist_sync_on_step=False, k=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("senti", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.k = k

    def update(self, y_pred: torch.Tensor, s_c: torch.Tensor, s_mean: torch.Tensor):
        assert y_pred.shape == s_c.shape
        if self.k is None:
            order = torch.argsort(input=y_pred, descending=True)
        else:
            sequence_length = y_pred.shape[0]
            if sequence_length < self.k:
                k = sequence_length
            else: 
                k = self.k
            _, order = torch.topk(input=y_pred,
                                  k=k,
                                  largest=True)
        
        senti_score = torch.take(s_c, order)
        senti_score = s_mean * torch.sum(senti_score)
        senti_score = torch.nn.functional.relu(senti_score)
            
        self.senti += senti_score
        self.count += 1.0

    def compute(self):
        return self.senti / self.count
