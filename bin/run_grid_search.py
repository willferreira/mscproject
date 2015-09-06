from itertools import chain, combinations
import argparse

from sklearn import grid_search

from model.utils import get_dataset, split_data
from model.classifiers.lr_predictors import LogitPredictor
from model.cross_validation import ClaimKFold

from model.baseline.transforms import \
    SemanticRelationshipTransform1, \
    RefutingWordsTransform, \
    QuestionMarkTransform, \
    HedgingWordsTransform, \
    WordOverlapTransform, \
    BrownClusterPairTransform, \
    BrownClusterBigramTransform, \
    PolarityTransform, \
    AlignedWordsTransform, \
    AlignedSimilarityTransform, \
    Word2VecTransform


def do_grid_search(X, y, classifier, param_grid, cv):

    def scorer(estimator, XX, yy):
        return estimator.score(XX, yy)[3]


    clf = grid_search.GridSearchCV(classifier, param_grid, cv=cv, scoring=scorer, verbose=True)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_score_)
    return clf.best_estimator_


def powerset(iterable):
    s = list(iterable)
    return map(list, filter(lambda x: len(x) > 0, chain.from_iterable(combinations(s, r) for r in range(len(s)+1))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run_baseline cmd-line arguments.')

    parser.add_argument('--classifier', default='BLR', type=str)
    args = parser.parse_args()

    classifier = args.classifier
    print('Using {0:s} classifier'.format(classifier))

    baseline_transforms = [
        SemanticRelationshipTransform1,
        RefutingWordsTransform,
        QuestionMarkTransform,
        HedgingWordsTransform,
        WordOverlapTransform,
        PolarityTransform,
        # Word2VecTransform
    ]

    train_data = get_dataset('url-versions-2015-06-14-clean-train.csv')
    X, y = split_data(train_data)
    ckf = ClaimKFold(X)

    if classifier == 'BLR':
        classifier = LogitPredictor(baseline_transforms)

        param_grid = [
            {
                'transforms': powerset(baseline_transforms)
            }
        ]
    elif classifier == 'BRF':
        pass
    else:
        pass

    do_grid_search(X, y, classifier, param_grid, ckf)