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
        self.feature_types = {}

    # -------------------------------------------------
    # PUBLIC
    # -------------------------------------------------
    def fit(self, df):
        df = df.copy()

        self.feature_columns = [
            c for c in df.columns
            if c != self.target_column and c not in self.excluded_features
        ]

        self.feature_types = {}

        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.feature_types[col] = "numeric"
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                self.feature_types[col] = "categorical"
                df[col] = df[col].astype(str)

        self.node_counter = 0
        self.nodes_index = {}

        self.root = self._build_tree(
            df=df,
            indices=df.index.tolist(),
            depth=0,
            rule_text="Racine",
            available_features=self.feature_columns[:]
        )
        return self

    def predict(self, df):
        df = df.copy()
        predictions = []

        for _, row in df.iterrows():
            predictions.append(self.predict_one(row.to_dict()))

        return predictions

    def predict_one(self, row_dict):
        prediction, _, _ = self.predict_with_path(row_dict)
        return prediction

    def predict_with_path(self, row_dict):
        node = self.root
        path = []

        while node is not None and not node.is_leaf:
            feature = node.attribute

            if feature is None:
                break

            if node.threshold is not None:
                value = row_dict.get(feature)

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    path.append({
                        "node_id": node.node_id,
                        "attribute": feature,
                        "value": value,
                        "condition": f"{feature} = valeur invalide",
                        "next_node_id": None
                    })
                    return node.prediction, path, node

                # Comparaison robuste
                if numeric_value <= node.threshold:
                    next_node = node.left
                    condition = f"{feature} <= {node.threshold:.4f}"
                else:
                    next_node = node.right
                    condition = f"{feature} > {node.threshold:.4f}"

                path.append({
                    "node_id": node.node_id,
                    "attribute": feature,
                    "value": numeric_value,
                    "condition": condition,
                    "next_node_id": next_node.node_id if next_node else None
                })

                if next_node is None:
                    return node.prediction, path, node

                node = next_node

            else:
                value = row_dict.get(feature)

                if value is None:
                    value = "ValeurManquante"
                else:
                    value = str(value)

                if value in node.branches:
                    next_node = node.branches[value]
                    condition = f"{feature} = {value}"
                    path.append({
                        "node_id": node.node_id,
                        "attribute": feature,
                        "value": value,
                        "condition": condition,
                        "next_node_id": next_node.node_id if next_node else None
                    })
                    node = next_node
                else:
                    path.append({
                        "node_id": node.node_id,
                        "attribute": feature,
                        "value": value,
                        "condition": f"{feature} = valeur inconnue",
                        "next_node_id": None
                    })
                    return node.prediction, path, node

        return (node.prediction if node else None), path, node

    def get_node(self, node_id):
        return self.nodes_index.get(node_id)

    # -------------------------------------------------
    # INTERNAL UTILS
    # -------------------------------------------------
    def _next_node_id(self):
        self.node_counter += 1
        return self.node_counter

    def _majority_class(self, labels):
        counts = Counter(labels)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

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
        total = sum(len(subset) for subset in subsets)
        if total == 0:
            return 0.0

        value = 0.0
        for subset in subsets:
            subset_len = len(subset)
            if subset_len == 0:
                continue
            p = subset_len / total
            value -= p * math.log2(p)

        return value

    def _possible_thresholds(self, series):
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        values = sorted(numeric_series.unique())

        thresholds = []
        for i in range(len(values) - 1):
            thresholds.append((values[i] + values[i + 1]) / 2)

        return thresholds

    def _gain_ratio_categorical(self, df, feature):
        total_entropy = self._entropy(df[self.target_column])

        working_df = df.copy()
        working_df[feature] = working_df[feature].fillna("ValeurManquante").astype(str)

        groups = [group for _, group in working_df.groupby(feature, sort=True)]

        if len(groups) <= 1:
            return -1, None

        weighted_entropy = 0.0
        for group in groups:
            weighted_entropy += (len(group) / len(working_df)) * self._entropy(group[self.target_column])

        info_gain = total_entropy - weighted_entropy
        split_info = self._split_info(groups)

        if split_info == 0:
            return -1, None

        gain_ratio = info_gain / split_info
        return gain_ratio, None

    def _gain_ratio_numeric(self, df, feature):
        total_entropy = self._entropy(df[self.target_column])

        working_df = df.copy()
        working_df[feature] = pd.to_numeric(working_df[feature], errors="coerce")
        working_df = working_df.dropna(subset=[feature])

        if len(working_df) < 2:
            return -1, None

        thresholds = self._possible_thresholds(working_df[feature])

        best_gain_ratio = -1
        best_threshold = None

        for threshold in thresholds:
            left = working_df[working_df[feature] <= threshold]
            right = working_df[working_df[feature] > threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            weighted_entropy = (
                (len(left) / len(working_df)) * self._entropy(left[self.target_column]) +
                (len(right) / len(working_df)) * self._entropy(right[self.target_column])
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
            feature_type = self.feature_types.get(feature, "categorical")

            if feature_type == "numeric":
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
        node.prediction = self._majority_class(df[self.target_column].tolist())
        node.rule_text = rule_text

        positive_count = node.class_counts.get(self.positive_class, 0)
        node.probability = positive_count / len(df) if len(df) > 0 else 0.0

        self.nodes_index[node.node_id] = node
        return node

    # -------------------------------------------------
    # TREE BUILDING
    # -------------------------------------------------
    def _build_tree(self, df, indices, depth, rule_text, available_features):
        if len(df) == 0:
            return None

        labels = df[self.target_column]

        # Tous les exemples de la même classe
        if len(labels.unique()) == 1:
            return self._make_leaf(df, indices, depth, rule_text)

        # Critères d'arrêt
        if len(df) < self.min_samples_split or depth >= self.max_depth:
            return self._make_leaf(df, indices, depth, rule_text)

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
        node.prediction = self._majority_class(labels.tolist())
        node.rule_text = rule_text

        positive_count = node.class_counts.get(self.positive_class, 0)
        node.probability = positive_count / len(df) if len(df) > 0 else 0.0

        self.nodes_index[node.node_id] = node

        feature_type = self.feature_types.get(best_feature, "categorical")

        # -------------------------
        # Split numérique
        # -------------------------
        if feature_type == "numeric" and best_threshold is not None:
            working_df = df.copy()
            working_df[best_feature] = pd.to_numeric(working_df[best_feature], errors="coerce")

            left_df = working_df[working_df[best_feature] <= best_threshold]
            right_df = working_df[working_df[best_feature] > best_threshold]

            if len(left_df) == 0 or len(right_df) == 0:
                return self._make_leaf(df, indices, depth, rule_text)

            node.left = self._build_tree(
                left_df,
                left_df.index.tolist(),
                depth + 1,
                f"{rule_text} -> {best_feature} <= {best_threshold:.4f}",
                available_features[:]
            )

            node.right = self._build_tree(
                right_df,
                right_df.index.tolist(),
                depth + 1,
                f"{rule_text} -> {best_feature} > {best_threshold:.4f}",
                available_features[:]
            )

        # -------------------------
        # Split catégoriel
        # -------------------------
        else:
            working_df = df.copy()
            working_df[best_feature] = working_df[best_feature].fillna("ValeurManquante").astype(str)

            remaining_features = [f for f in available_features if f != best_feature]

            for value, subset in working_df.groupby(best_feature, sort=True):
                child = self._build_tree(
                    subset,
                    subset.index.tolist(),
                    depth + 1,
                    f"{rule_text} -> {best_feature} = {value}",
                    remaining_features[:]
                )
                if child is not None:
                    node.branches[str(value)] = child

            if len(node.branches) == 0:
                return self._make_leaf(df, indices, depth, rule_text)

        return node