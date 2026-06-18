import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.function import fasta_to_onehot, Spliceator, SpliceFinder, DeepSplicer, training_process, plot_all_results


def main():

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df_data = fasta_to_onehot(
        os.path.join(root,'data/acceptor_positive.fasta'), os.path.join(root,'data/acceptor_negative.fasta')
    )
    cnns = [Spliceator, SpliceFinder, DeepSplicer]
    results = training_process(df_data, n_folds=5, cnns=cnns)
    models = [cnn.__name__ for cnn in cnns] 
    plot_all_results(results, models, save_path=os.path.join(root, "results"))


if __name__ == "__main__":
    main()
