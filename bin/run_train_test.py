import argparse
import sys
import os

sys.path.append(os.path.join('..', 'src'))

import numpy as np
import pandas as pd


from model.classifiers.lr_predictors import LogitPredictor, CompoundPredictor
from model.classifiers.rf_predictors import RandomForestPredictor
from model.utils import get_dataset, split_data, RunCV, run_test

from model.baseline.transforms import (
    RefutingWordsTransform,
    QuestionMarkTransform,
    HedgingWordsTransform,
    InteractionTransform,
    NegationOfRefutingWordsTransform,
    BoWTransform
)

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDistanceTransform,
    SVOTransform
)


_classifiers = {
    'COMPOUND': lambda (obs, fa): CompoundPredictor(obs, fa),
    'BLR': lambda t: LogitPredictor(t),
    'BRF': lambda t: RandomForestPredictor(t),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_baseline cmd-line arguments.')

    parser.add_argument('--classifier', default='BLR', type=str)
    args = parser.parse_args()

    classifier = args.classifier
    print('Using {0:s} classifier'.format(classifier))
    predictor = _classifiers.get(classifier)

    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)
    test_data = get_dataset('url-versions-2015-06-14-clean-test.csv')

    transforms = {
        'BoW': BoWTransform,
        'BoW-Ref': RefutingWordsTransform,
        'BoW-Hed': HedgingWordsTransform,
        'Q': QuestionMarkTransform,
        'I': InteractionTransform,
        'Sim-W2V': Word2VecSimilaritySemanticTransform,
        'Sim-Algn-PPDB': AlignedPPDBSemanticTransform,
        'BoW-Neg-Ref': NegationOfRefutingWordsTransform,
        'Neg-Algn': NegationAlignmentTransform,
        'Root-Dist': DependencyRootDistanceTransform,
        'SVO': SVOTransform
    }

    inc_transforms = [
        'BoW-Ref', 'BoW-Hed',
        'Q', 'I',
        'Sim-W2V',
        'Sim-Algn-PPDB', 'BoW-Neg-Ref',
        'Neg-Algn',
        'BoW',
        'Root-Dist',
        # 'SVO'
        ]

    df_out = pd.DataFrame(index=inc_transforms, columns=['accuracy-cv', 'accuracy-test'], data=np.nan)

    run_incremental = True
    if run_incremental:
        inc_transforms_cls = []
        for i, k in enumerate(inc_transforms):
            inc_transforms_cls.append(transforms[k])
            print(inc_transforms[:i+1])
            p = predictor(inc_transforms_cls)
            cv_score = RunCV(X, y, p, display=True).run_cv()
            test_score = run_test(X, y, test_data, p, display=True)
            df_out.ix[k, 'accuracy-cv'] = cv_score.accuracy
            df_out.ix[k, 'accuracy-test'] = test_score.accuracy
        print(df_out)
    else:
        p = predictor([transforms[t] for t in inc_transforms])
        cv_score = RunCV(X, y, p, display=True).run_cv()

        p = predictor([transforms[t] for t in inc_transforms])
        test_score = run_test(X, y, test_data, p, display=True)

