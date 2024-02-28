import torch
import math
from sklearn.metrics import precision_score, recall_score, f1_score


class EarlyStopping(object):
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 1):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.__value = -math.inf if mode == 'max' else math.inf
        self.__times = 0

    def state_dict(self) -> dict:
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'value': self.__value,
            'times': self.__times
        }

    def load_state_dict(self, state_dict: dict):
        self.monitor = state_dict['monitor']
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self.__value = state_dict['value']
        self.__times = state_dict['times']

    def reset(self):
        self.__times = 0

    def __call__(self, metrics) -> bool:
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        if (self.mode == 'min' and metrics <= self.__value) or (
                self.mode == 'max' and metrics >= self.__value):
            self.__value = metrics
            self.__times = 0
        else:
            self.__times += 1
        if self.__times >= self.patience:
            return True
        return False

def evaluate(decoder_model, dataloader, device):
    correct = 0
    total = 0
    all_y = []
    all_pred_y = []
    with torch.no_grad():
        decoder_model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.permute([0, 2, 1])
            x = x.float()
            y = y.long()
            all_y = all_y + y.cpu().tolist()
            y_pre = decoder_model(x)

            _, label_index = torch.max(y_pre.data, dim=-1)
            all_pred_y = all_pred_y + label_index.cpu().tolist()
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()


        macro_f1 = f1_score(all_y, all_pred_y, average='weighted')
        recall = recall_score(all_y, all_pred_y, average='weighted')
        precision = precision_score(all_y, all_pred_y, average='weighted')

        print(f'Macro-F1 score:%.4f' % macro_f1)
        print(f'Recall:%.4f' % recall)
        print(f'Precision:%.4f' % precision)


def cal_best_model_evaluating_indicator(model_path, model, testDataloader, device):
    model.load_state_dict(torch.load(model_path))
    evaluate(model, testDataloader, device=device)

