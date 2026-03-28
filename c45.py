import math
from collections import Counter

import pandas as pd


class TreeNode:
    def __init__(self, node_id, depth=0):
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = False

        self.attribute = None
        self.threshold = None

        self.left = None
        self.right = None
        self.branches = {}

        self.prediction = None
        self.class_counts = {}
        self.samples_count = 0
        self.sample_indices = []
        self.probability = 0.0
        self.rule_text = "Racine"


class C45DecisionTree:
    def __init__(
        self,
        min_samples_split=5,
        max_depth=6,
        target_column="maladie",
        positive_class="Oui",
        excluded_features=None
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.target_column = target_column
        self.positive_class = positive_class
        self.excluded_features = excluded_features or []

        self.root = None
        self.node_counter = 0
        self.nodes_index = {}
        self.feature_columns = []

    def fit(self, df):
        self.feature_columns = [
            c for c in df.columns
            if c != self.target_column and c not in self.excluded_features
        ]
        self.node_counter = 0
        self.nodes_index = {}
        self.root = self._build_tree(df, df.index.tolist(), depth=0, rule_text="Racine")
        return self

    def predict(self, df):
        return [self.predict_one(row.to_dict()) for _, row in df.iterrows()]

    def predict_one(self, row_dict):
        node = self.root

        while node is not None and not node.is_leaf:
            if node.threshold is not None:
                value = row_dict.get(node.attribute)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return node.prediction

                if value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                value = row_dict.get(node.attribute)
                if value in node.branches:
                    node = node.branches[value]
                else:
                    return node.prediction

        return node.prediction if node is not None else None

    def predict_with_path(self, row_dict):
        node = self.root
        path = []

        while node is not None and not node.is_leaf:
            if node.threshold is not None:
                value = row_dict.get(node.attribute)
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    path.append({
                        "node_id": node.node_id,
                        "attribute": node.attribute,
                        "value": value,
                        "condition": "valeur invalide",
                        "next_node_id": None
                    })
                    return node.prediction, path, node

                if numeric_value <= node.threshold:
                    next_node = node.left
                    condition = f"{node.attribute} <= {node.threshold:.4f}"
                else:
                    next_node = node.right
                    condition = f"{node.attribute} > {node.threshold:.4f}"

                path.append({
                    "node_id": node.node_id,
                    "attribute": node.attribute,
                    "value": numeric_value,
                    "condition": condition,
                    "next_node_id": next_node.node_id if next_node else None
                })

                node = next_node

            else:
                value = row_dict.get(node.attribute)
                if value in node.branches:
                    next_node = node.branches[value]
                    condition = f"{node.attribute} = {value}"
                    path.append({
                        "node_id": node.node_id,
                        "attribute": node.attribute,
                        "value": value,
                        "condition": condition,
                        "next_node_id": next_node.node_id if next_node else None
                    })
                    node = next_node
                else:
                    path.append({
                        "node_id": node.node_id,
                        "attribute": node.attribute,
                        "value": value,
                        "condition": f"{node.attribute} = valeur inconnue",
                        "next_node_id": None
                    })
                    return node.prediction, path, node

        return (node.prediction if node else None), path, node

    def get_node(self, node_id):
        return self.nodes_index.get(node_id)

    def _next_node_id(self):
        self.node_counter += 1
        return self.node_counter

    def _entropy(self, labels):
        total = len(labels)
        if total == 0:
            return 0.0

        counts = Counter(labels)
        entropy = 0.0

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _split_info(self, subsets):
        total = sum(len(s) for s in subsets)
        if total == 0:
            return 0.0

        value = 0.0
        for subset in subsets:
            if len(subset) == 0:
                continue
            p = len(subset) / total
            value -= p * math.log2(p)

        return value

    def _possible_thresholds(self, series):
        values = sorted(series.dropna().unique())
        thresholds = []

        for i in range(len(values) - 1):
            thresholds.append((values[i] + values[i + 1]) / 2)

        return thresholds

    def _gain_ratio_categorical(self, df, feature):
        total_entropy = self._entropy(df[self.target_column])
        groups = [group for _, group in df.groupby(feature)]

        weighted_entropy = 0.0
        for group in groups:
            weighted_entropy += (len(group) / len(df)) * self._entropy(group[self.target_column])

        info_gain = total_entropy - weighted_entropy
        split_info = self._split_info(groups)

        if split_info == 0:
            return -1, None

        gain_ratio = info_gain / split_info
        return gain_ratio, None

    def _gain_ratio_numeric(self, df, feature):
        total_entropy = self._entropy(df[self.target_column])
        thresholds = self._possible_thresholds(df[feature])

        best_gain_ratio = -1
        best_threshold = None

        for threshold in thresholds:
            left = df[df[feature] <= threshold]
            right = df[df[feature] > threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            weighted_entropy = (
                (len(left) / len(df)) * self._entropy(left[self.target_column]) +
                (len(right) / len(df)) * self._entropy(right[self.target_column])
            )

            info_gain = total_entropy - weighted_entropy
            split_info = self._split_info([left, right])

            if split_info == 0:
                continue

            gain_ratio = info_gain / split_info

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

        return best_gain_ratio, best_threshold

    def _best_split(self, df, available_features):
        best_feature = None
        best_threshold = None
        best_score = -1

        for feature in available_features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                score, threshold = self._gain_ratio_numeric(df, feature)
            else:
                score, threshold = self._gain_ratio_categorical(df, feature)

            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold, best_score

    def _make_leaf(self, df, indices, depth, rule_text):
        node = TreeNode(self._next_node_id(), depth)
        node.is_leaf = True
        node.samples_count = len(df)
        node.sample_indices = indices
        node.class_counts = df[self.target_column].value_counts().to_dict()
        node.prediction = df[self.target_column].mode()[0]
        node.rule_text = rule_text

        positive_count = node.class_counts.get(self.positive_class, 0)
        node.probability = positive_count / len(df) if len(df) > 0 else 0.0

        self.nodes_index[node.node_id] = node
        return node

    def _build_tree(self, df, indices, depth, rule_text):
        if len(df) == 0:
            return None

        labels = df[self.target_column]

        if len(labels.unique()) == 1:
            return self._make_leaf(df, indices, depth, rule_text)

        if len(df) < self.min_samples_split or depth >= self.max_depth:
            return self._make_leaf(df, indices, depth, rule_text)

        available_features = self.feature_columns[:]
        if not available_features:
            return self._make_leaf(df, indices, depth, rule_text)

        best_feature, best_threshold, best_score = self._best_split(df, available_features)

        if best_feature is None or best_score <= 0:
            return self._make_leaf(df, indices, depth, rule_text)

        node = TreeNode(self._next_node_id(), depth)
        node.attribute = best_feature
        node.threshold = best_threshold
        node.samples_count = len(df)
        node.sample_indices = indices
        node.class_counts = labels.value_counts().to_dict()
        node.prediction = labels.mode()[0]
        node.rule_text = rule_text

        positive_count = node.class_counts.get(self.positive_class, 0)
        node.probability = positive_count / len(df) if len(df) > 0 else 0.0

        self.nodes_index[node.node_id] = node

        if best_threshold is not None:
            left_df = df[df[best_feature] <= best_threshold]
            right_df = df[df[best_feature] > best_threshold]

            node.left = self._build_tree(
                left_df,
                left_df.index.tolist(),
                depth + 1,
                f"{rule_text} -> {best_feature} <= {best_threshold:.4f}"
            )

            node.right = self._build_tree(
                right_df,
                right_df.index.tolist(),
                depth + 1,
                f"{rule_text} -> {best_feature} > {best_threshold:.4f}"
            )
        else:
            for value, subset in df.groupby(best_feature):
                node.branches[value] = self._build_tree(
                    subset,
                    subset.index.tolist(),
                    depth + 1,
                    f"{rule_text} -> {best_feature} = {value}"
                )

        return node