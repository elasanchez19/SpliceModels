import sys
sys.path.append("../")

from utils.function import fasta_to_onehot, Spliceator, SpliceFinder, DeepSplicer, training_process, plot_all_results


def main():

    df_data = fasta_to_onehot('../data/half_acceptor_test_positive.fasta', '../data/half_acceptor_test_negative.fasta')
    cnns = [Spliceator, SpliceFinder, DeepSplicer]
    results = training_process(df_data, 2, cnns)
    models = ['Spliceator', 'SpliceFinder'] # aquí conveniente utilizar cnns mejor
    plot_all_results(results, models, save_path="../results")


if __name__ == "__main__":
    main()
