from train.train_gcn import GcnTrainer
from train.train_gcn_lstm import GcnLstmTrainer
from eval.evaluation import Eval
if __name__ == '__main__':
    # GcnTrainer().train()
    # GcnLstmTrainer().train()
    Eval("STGCN_FC").mean_jaccard_index()
    # Eval("BLA_LSTM").mean_jaccard_index()
    # Eval("STGCN_LSTM").mean_jaccard_index()