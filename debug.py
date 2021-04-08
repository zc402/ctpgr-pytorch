from train.train_gcn_model import GcnTrainer
from train.train_gcn_lstm_model import GcnLstmTrainer
from eval.evaluation import Eval
if __name__ == '__main__':
    GcnTrainer().train()
    # GcnLstmTrainer().train()
    # Eval().mean_jaccard_index_gcn()
    # Eval().mean_jaccard_index_gcn_lstm()