from train.train_gcn_model import Trainer
from eval.evaluation import Eval
if __name__ == '__main__':
    # Trainer().train()
    Eval().mean_jaccard_index()