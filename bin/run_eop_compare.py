import os

import pandas as pd
from sklearn.metrics import accuracy_score

from model.utils import calc_confusion_matrix, calc_measures


if __name__ == '__main__':
    eop_files = [f for f in os.listdir(os.path.join('..', 'output', 'eop')) if f.endswith('.txt')]

    for f in eop_files:
        print f
        opts = f.split('+')[1:]
        opts[-1] = opts[-1].split('_')[0]
        print 'options:', opts

        df_eop = pd.read_csv(os.path.join('..', 'output', 'eop', f), delimiter='\t', header=None)
        df_eop.columns = ['id', 'benchmark', 'predicted', 'confidence']

        df_eop.loc[df_eop.benchmark == 'ENTAILMENT', 'benchmark'] = 'for'
        df_eop.loc[df_eop.benchmark == 'CONTRADICTION', 'benchmark'] = 'against'
        df_eop.loc[df_eop.benchmark == 'UNKNOWN', 'benchmark'] = 'observing'

        df_eop.loc[df_eop.predicted == 'Entailment', 'predicted'] = 'for'
        df_eop.loc[df_eop.predicted == 'Contradiction', 'predicted'] = 'against'
        df_eop.loc[df_eop.predicted == 'Unknown', 'predicted'] = 'observing'

        y = df_eop.benchmark
        y_hat = df_eop.predicted
        cm = calc_confusion_matrix(y, y_hat)
        measures = calc_measures(cm)
        accuracy = accuracy_score(y, y_hat)

        print 'accuracy:', accuracy
        print 'per class measures:'
        print measures
        print ''


