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
    BoWTransform,
    PolarityTransform,
    BrownClusterPairTransform
)

from model.ext.transforms import (
    AlignedPPDBSemanticTransform,
    NegationAlignmentTransform,
    Word2VecSimilaritySemanticTransform,
    DependencyRootDistanceTransform,
    SVOTransform
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_baseline cmd-line arguments.')

    parser.add_argument('-i', action='store_true', default=False)

    # inc_transforms = [
    #     'Q',                # 1             1
    #     'BoW-Hed',          # 2-36         35
    #     'BoW-Ref',          # 37-48         12
    #     'I',                # 49-1272       1,224
    #     'BoW',              # 1273-1772     500
    #     'Sim-Algn-W2V',     # 1773-1773     1
    #     'Sim-Algn-PPDB',    # 1774-1774     1
    #     'Root-Dist',        # 1775-1776     2
    #     'Neg-Algn',         # 1777-1779     3
    #     'SVO',              # 1780-1788     9
    #     ]

    parser.add_argument('-f',
                        default="Q,BoWHed,BoWRef,I,BoW,AlgnW2V,AlgnPPDB,RootDist,NegAlgn,SVO",
                        type=str)
    args = parser.parse_args()

    predictor = LogitPredictor

    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)

    test_data = get_dataset('url-versions-2015-06-14-clean-test.csv')

    transforms = {
        'BoW': lambda: BoWTransform(),
        'BoWRef': RefutingWordsTransform,
        'BoWHed': HedgingWordsTransform,
        'Q': QuestionMarkTransform,
        'I': InteractionTransform,
        'AlgnW2V': Word2VecSimilaritySemanticTransform,
        'AlgnPPDB': AlignedPPDBSemanticTransform,
        'NegAlgn': NegationAlignmentTransform,
        'RootDist': DependencyRootDistanceTransform,
        'SVO': SVOTransform,
    }

    inc_transforms = args.f.split(',')
    diff = set(inc_transforms).difference(transforms.keys())
    if diff:
        print 'Unrecognised features:', diff
        sys.exit(1)
    print 'Feature set:', inc_transforms
    if args.i:
        df_out = pd.DataFrame(index=inc_transforms,
                              columns=['accuracy-cv', 'accuracy-test'], data=np.nan)
        inc_transforms_cls = []
        for i, k in enumerate(inc_transforms):
            inc_transforms_cls.append(transforms[k])
            print 'Using features:', inc_transforms[:i+1]

            p = predictor(inc_transforms_cls)
            cv_score = RunCV(X, y, p, display=True).run_cv()
            test_score = run_test(X, y, test_data, p, display=True)

            df_out.ix[k, 'accuracy-cv'] = cv_score.accuracy
            df_out.ix[k, 'accuracy-test'] = test_score.accuracy
        print(df_out)
    else:
        p = predictor([transforms[t] for t in inc_transforms])
        test_score = run_test(X, y, test_data, p, display=True)

