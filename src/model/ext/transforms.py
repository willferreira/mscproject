import numpy as np

from model.base import StatelessTransform
from model.utils import get_stanparse_data, get_aligned_data, get_dep_graph, \
    get_cosine_similarity_data, get_hungarian_alignment_score_data, find_negated_word_idxs, \
    get_stanford_idx


class Word2VecSimilaritySemanticTransform(StatelessTransform):

    def transform(self, X):
        _, cos_mult = get_cosine_similarity_data()
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            if np.isnan(mat[i, 0]):
                print s.claimId, s.articleId,
            mat[i, 0] = cos_mult[(s.claimId, s.articleId)]
        return mat


_hungarian = get_hungarian_alignment_score_data()


class AlignedPPDBSemanticTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 1))
        for i, (_, s) in enumerate(X.iterrows()):
            mat[i, 0] = _hungarian[(s.claimId, s.articleId)][1]
        return mat


class NegationAlignmentTransform(StatelessTransform):

    def transform(self, X):
        mat = np.zeros((len(X), 3))
        for i, (_, s) in enumerate(X.iterrows()):
            claim_negated_idxs = find_negated_word_idxs(s.claimId)
            article_negated_idxs = find_negated_word_idxs(s.articleId)
            if not claim_negated_idxs and not article_negated_idxs:
                continue

            for a, b in _hungarian[(s.claimId, s.articleId)][0]:
                if a in claim_negated_idxs and b not in article_negated_idxs:
                    mat[i, 0] = 1
                if a not in claim_negated_idxs and b in article_negated_idxs:
                    mat[i, 1] = 1
                if a in claim_negated_idxs and b in article_negated_idxs:
                    mat[i, 2] = 1
        return mat

