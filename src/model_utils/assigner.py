import mindspore
import mindspore.numpy as np
from mindspore import nn, ops


class ElementsAssigner(nn.Cell):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_relations=False):
        super(ElementsAssigner, self).__init__()
        self.below_low = -1
        self.between = -2

        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_relations = allow_low_quality_relations

    def set_low_qual_relations(self, relations, all_relations, relation_quals):
        top_qual_gt, _ = relation_quals.max(axis=1)
        gt_pred_pairs_of_top_qual = ops.nonzero(relation_quals == top_qual_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_top_qual[:, 1]
        relations[pred_inds_to_update] = all_relations[pred_inds_to_update]
    
    def get_constant(self):
        return self.below_low, self.between

    def construct(self, relation_quals):
        if relation_quals.size() == 0:
            # handle empty case
            return np.empty((0,), dtype=mindspore.int64)

        relation_vals, relations = relation_quals.max(axis=0)
        if self.allow_low_quality_relations:
            all_matches = relations.copy()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = relation_vals < self.low_threshold
        between_thresholds = (relation_vals >= self.low_threshold) & (
            relation_vals < self.high_threshold
        )
        relations[below_low_threshold] = self.below_low
        relations[between_thresholds] = self.between

        if self.allow_low_quality_relations:
            self.set_low_qual_relations(relations, all_matches, relation_quals)

        return relations

if __name__ == '__main__':
    test = ElementsAssigner.get_constant()