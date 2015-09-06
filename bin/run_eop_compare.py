import os

import pandas as pd
from sklearn.metrics import accuracy_score

from model.utils import get_dataset, calc_confusion_matrix, calc_measures

_observing_threshold = 0.7

_stance_map = \
    {
        'NonEntailment': 'against',
        'Entailment': 'for'
    }

if __name__ == '__main__':
    df_clean_test = get_dataset('url-versions-2015-06-14-clean-test.csv')
    eop_files = [f for f in os.listdir(os.path.join('data', 'eop')) if f.endswith('.txt')]

    for f in eop_files:
        print f
        opts = f.split('+')[1:]
        opts[-1] = opts[-1].split('_')[0]
        print 'options:', opts

        df_eop = pd.read_csv(os.path.join('..', 'data', 'eop', f), delimiter='\t', header=None)
        df_eop.columns = ['id', 'na', 'stance', 'confidence']

        df_eop.loc[df_eop.confidence < _observing_threshold, 'stance'] = 'observing'
        for k, i in _stance_map.items():
            df_eop.loc[df_eop.stance == k, 'stance'] = i

        df_joined = df_eop.join(df_clean_test, on='id')

        y = df_joined.articleHeadlineStance
        y_hat = df_joined.stance

        cm = calc_confusion_matrix(y, y_hat)
        measures = calc_measures(cm)
        accuracy = accuracy_score(y, y_hat)
        print 'accuracy:', accuracy
        print 'per class measures:'
        print measures
        print ''


