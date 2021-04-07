from train.train_gcn_model import Trainer as GcnTrainer
from train.train_gcn_lstm_model import Trainer as GcnLstmTrainer
from eval.evaluation import Eval
if __name__ == '__main__':
    # Trainer().train()
    GcnLstmTrainer().train()
    # Eval().mean_jaccard_index()