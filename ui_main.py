import pandas as pd
from PySide6.QtCore import Qt, QRectF, QPoint, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QBrush, QColor, QPen, QPainter
from PySide6.QtWidgets import (
    QFormLayout, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QLineEdit, QFrame,
    QGraphicsOpacityEffect, QScrollArea
)

from sklearn.model_selection import train_test_split

from c45 import C45DecisionTree
from dataset_loader import load_breast_cancer_dataset
from metrics_utils import evaluate_model


class LoadingOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: rgba(255, 255, 255, 210);
            border: 2px solid #cccccc;
            border-radius: 10px;
        """)
        layout = QVBoxLayout(self)
        self.label = QLabel("<b>⚙️ Entraînement de l'arbre en cours...</b><br>Veuillez patienter.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.hide()

    def update_position(self):
        if self.parent():
            self.setFixedSize(self.parent().size())


class GraphicsNodeItem(QGraphicsRectItem):
    def __init__(self, rect, node_id, text, main_win, base_color):
        super().__init__(rect)
        self.node_id = node_id
        self.main_win = main_win
        self.base_color = base_color
        self.is_collapsed = False

        self.child_items = []
        self.edge_items = []
        self.label_items = []

        self.setBrush(QBrush(base_color))
        self.setPen(QPen(Qt.black, 1.2))
        self.setAcceptHoverEvents(True)

        self.text_item = QGraphicsTextItem(text, self)
        self.text_item.setTextWidth(rect.width() - 10)
        self.text_item.setPos(5, 5)

    def mousePressEvent(self, event):
        self.main_win.on_graph_node_clicked(self.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.is_collapsed = not self.is_collapsed
        self.main_win.toggle_subtree(self.node_id, not self.is_collapsed)
        super().mouseDoubleClickEvent(event)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor(0, 120, 215), 3))
        self.main_win.show_hud(self.node_id, event.screenPos())
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(Qt.black, 1.2))
        self.main_win.hide_hud()
        super().hoverLeaveEvent(event)

    def reset_color(self):
        self.setBrush(QBrush(self.base_color))

    def set_highlight(self, color):
        self.setBrush(QBrush(color))


class TreeViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor(245, 247, 249)))

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def smooth_center(self, item):
        self.anim = QVariantAnimation()
        self.anim.setDuration(500)
        self.anim.setStartValue(self.mapToScene(self.viewport().rect().center()))
        self.anim.setEndValue(item.sceneBoundingRect().center())
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.valueChanged.connect(self.centerOn)
        self.anim.start()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logiciel Desktop - Arbre de décision C4.5 / Cancer du sein")
        self.resize(1800, 980)

        self.target_column = "maladie"
        self.id_column = "patient_id"
        self.name_column = "nom"

        self.df = None
        self.model = None
        self.node_items = {}
        self.scene = None
        self.current_table_df = pd.DataFrame()
        self.prediction_feature_order = []
        self._running_animations = []

        self._build_ui()

        self.overlay = LoadingOverlay(self.graphics_view)

        self.hud = QLabel(self)
        self.hud.setStyleSheet("""
            background: rgba(45, 45, 45, 220);
            color: white;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid white;
        """)
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud.hide()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_load_dataset = QPushButton("Charger le dataset cancer du sein")
        self.btn_train = QPushButton("Entraîner l'arbre")
        self.btn_predict = QPushButton("Prédire nouvelle patiente")
        self.btn_reset = QPushButton("🎯 Centrer Vue")
        self.btn_load_selected_patient = QPushButton("Charger la patiente sélectionnée")

        for b in [
            self.btn_load_dataset,
            self.btn_train,
            self.btn_predict,
            self.btn_reset,
            self.btn_load_selected_patient
        ]:
            b.setMinimumHeight(36)
            top.addWidget(b)

        top.addStretch()
        main_layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        self.graphics_view = TreeViewer()
        splitter.addWidget(self.graphics_view)

        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)

        self.dataset_info_text = QTextEdit()
        self.details_text = QTextEdit()
        self.metrics_text = QTextEdit()
        self.path_text = QTextEdit()

        for w in [self.dataset_info_text, self.details_text, self.metrics_text, self.path_text]:
            w.setReadOnly(True)

        middle_layout.addWidget(QLabel("Informations du dataset"))
        middle_layout.addWidget(self.dataset_info_text)

        middle_layout.addWidget(QLabel("Détails du nœud sélectionné"))
        middle_layout.addWidget(self.details_text)

        middle_layout.addWidget(QLabel("Métriques globales"))
        middle_layout.addWidget(self.metrics_text)

        middle_layout.addWidget(QLabel("Explication simple de la décision"))
        middle_layout.addWidget(self.path_text)

        splitter.addWidget(middle_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.patients_table = QTableWidget()
        right_layout.addWidget(QLabel("Patientes du nœud / feuille sélectionné(e)"))
        right_layout.addWidget(self.patients_table)

        prediction_group = QGroupBox("Tester une nouvelle patiente")
        prediction_group_layout = QVBoxLayout(prediction_group)

        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)

        self.form_container = QWidget()
        self.prediction_layout = QFormLayout(self.form_container)
        self.form_scroll.setWidget(self.form_container)

        prediction_group_layout.addWidget(self.form_scroll)

        self.predict_result_label = QLabel("Résultat : -")
        self.predict_result_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.simple_result_label = QLabel("")
        self.simple_result_label.setWordWrap(True)
        self.simple_result_label.setStyleSheet("color: #333;")

        prediction_group_layout.addWidget(self.predict_result_label)
        prediction_group_layout.addWidget(self.simple_result_label)

        right_layout.addWidget(prediction_group)

        splitter.addWidget(right_widget)
        splitter.setSizes([760, 520, 520])

        main_layout.addWidget(splitter)

        self.btn_load_dataset.clicked.connect(self.load_default_dataset)
        self.btn_train.clicked.connect(self.train_model)
        self.btn_predict.clicked.connect(self.predict_new_patient)
        self.btn_reset.clicked.connect(self.fit_view)
        self.btn_load_selected_patient.clicked.connect(self.load_selected_patient_into_form)

    def load_default_dataset(self):
        try:
            self.df = load_breast_cancer_dataset()
            self.clear_graph()
            self.populate_patients_table(self.df.head(30))
            self.show_dataset_info()
            self.build_prediction_form()
            self.details_text.clear()
            self.metrics_text.clear()
            self.path_text.clear()
            self.predict_result_label.setText("Résultat : -")
            self.simple_result_label.setText("")
            QMessageBox.information(self, "Succès", "Dataset chargé avec succès.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def clear_graph(self):
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.node_items = {}

    def show_dataset_info(self):
        if self.df is None:
            return

        class_counts = self.df[self.target_column].value_counts().to_dict()
        lines = [
            "Dataset : Breast Cancer Wisconsin",
            f"Nombre de patientes : {len(self.df)}",
            "",
            "Répartition des classes :",
            f"- Oui (malade) : {class_counts.get('Oui', 0)}",
            f"- Non (non malade) : {class_counts.get('Non', 0)}",
            "",
            "Remarque : les métriques sont calculées sur un jeu de test de 30%."
        ]
        self.dataset_info_text.setPlainText("\n".join(lines))

    def build_prediction_form(self):
        while self.prediction_layout.rowCount() > 0:
            self.prediction_layout.removeRow(0)

        self.inputs = {}
        self.prediction_feature_order = []

        if self.df is None:
            return

        feature_columns = [
            c for c in self.df.columns
            if c not in [self.target_column, self.id_column, self.name_column]
        ]

        self.prediction_feature_order = feature_columns

        for feature in feature_columns:
            line_edit = QLineEdit()
            if pd.api.types.is_numeric_dtype(self.df[feature]):
                mean_val = float(self.df[feature].mean())
                line_edit.setPlaceholderText(f"Exemple : {mean_val:.4f}")
            self.inputs[feature] = line_edit
            self.prediction_layout.addRow(feature, line_edit)

    def load_selected_patient_into_form(self):
        if self.current_table_df.empty:
            QMessageBox.warning(self, "Attention", "Aucune patiente n'est actuellement affichée.")
            return

        row = self.patients_table.currentRow()
        if row < 0 or row >= len(self.current_table_df):
            QMessageBox.warning(self, "Attention", "Sélectionnez d'abord une patiente dans le tableau.")
            return

        patient_row = self.current_table_df.iloc[row]

        for feature in self.prediction_feature_order:
            value = patient_row.get(feature, "")
            self.inputs[feature].setText(str(value))

        QMessageBox.information(
            self,
            "Succès",
            "Les données de la patiente sélectionnée ont été copiées dans le formulaire."
        )

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord le dataset.")
            return

        self.btn_train.setEnabled(False)
        self.overlay.show()
        self.overlay.update_position()

        try:
            train_df, test_df = train_test_split(
                self.df,
                test_size=0.30,
                random_state=42,
                stratify=self.df[self.target_column]
            )

            eval_model = C45DecisionTree(
                min_samples_split=5,
                max_depth=6,
                target_column=self.target_column,
                positive_class="Oui",
                excluded_features=[self.id_column, self.name_column]
            )
            eval_model.fit(train_df)

            X_test = test_df.drop(columns=[self.target_column])
            y_true = test_df[self.target_column].tolist()
            y_pred = eval_model.predict(X_test)

            metrics = evaluate_model(y_true, y_pred)

            final_model = C45DecisionTree(
                min_samples_split=5,
                max_depth=6,
                target_column=self.target_column,
                positive_class="Oui",
                excluded_features=[self.id_column, self.name_column]
            )
            final_model.fit(self.df)

            self.model = final_model

            cm = metrics["confusion_matrix"]
            metrics_text = (
                f"Accuracy : {metrics['accuracy']:.4f}\n"
                f"Precision : {metrics['precision']:.4f}\n"
                f"Recall : {metrics['recall']:.4f}\n"
                f"F1-score : {metrics['f1_score']:.4f}\n\n"
                f"Matrice de confusion (ordre : [Non, Oui]) :\n"
                f"TN = {metrics['tn']} | FP = {metrics['fp']}\n"
                f"FN = {metrics['fn']} | TP = {metrics['tp']}\n\n"
                f"Tableau :\n{cm}\n\n"
                f"Rapport de classification :\n{metrics['classification_report']}"
            )
            self.metrics_text.setPlainText(metrics_text)

            self.draw_graphical_tree()

            QMessageBox.information(
                self,
                "Info",
                f"Arbre entraîné avec succès.\nNœuds dessinés : {len(self.node_items)}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Erreur pendant l'entraînement", str(e))

        finally:
            self.btn_train.setEnabled(True)
            self.overlay.hide()

    def draw_graphical_tree(self):
        self.clear_graph()

        if not self.model or not self.model.root:
            QMessageBox.warning(self, "Attention", "Le modèle n'a pas été entraîné correctement.")
            return

        pos = {}
        leaf_count = [0]
        h_gap, v_gap = 240, 150
        w, h = 185, 85

        def calc_pos(node, depth=0):
            children = self.get_children_node(node)
            if not children:
                pos[node.node_id] = (leaf_count[0] * h_gap, depth * v_gap)
                leaf_count[0] += 1
            else:
                for _, child in children:
                    calc_pos(child, depth + 1)
                mean_x = sum(pos[child.node_id][0] for _, child in children) / len(children)
                pos[node.node_id] = (mean_x, depth * v_gap)

        calc_pos(self.model.root)

        for node_id, (px, py) in pos.items():
            node = self.model.get_node(node_id)

            if node.is_leaf:
                color = QColor(255, 210, 210) if node.prediction == "Oui" else QColor(210, 255, 210)
            else:
                color = QColor(230, 240, 255)

            attribute_text = node.attribute if node.attribute else "Feuille"
            txt = f"Nœud {node_id}\n{attribute_text}\nNb: {node.samples_count}"

            item = GraphicsNodeItem(QRectF(0, 0, w, h), node_id, txt, self, color)
            item.setPos(px, py)
            self.scene.addItem(item)
            self.node_items[node_id] = item

        for node_id, item in self.node_items.items():
            node = self.model.get_node(node_id)

            for label, child_node in self.get_children_node(node):
                child_item = self.node_items[child_node.node_id]

                parent_rect = item.sceneBoundingRect()
                child_rect = child_item.sceneBoundingRect()

                parent_center_x = parent_rect.center().x()
                child_center_x = child_rect.center().x()

                line = self.scene.addLine(
                    parent_center_x,
                    parent_rect.bottom(),
                    child_center_x,
                    child_rect.top(),
                    QPen(Qt.black, 1)
                )

                lbl = QGraphicsSimpleTextItem(label)
                lbl.setPos(
                    (parent_center_x + child_center_x) / 2,
                    (parent_rect.bottom() + child_rect.top()) / 2
                )
                self.scene.addItem(lbl)

                item.edge_items.append(line)
                item.child_items.append(child_item)
                item.label_items.append(lbl)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80))
        self.graphics_view.resetTransform()
        self.fit_view()

    def get_children_node(self, node):
        res = []

        if node is None:
            return res

        if node.threshold is not None:
            if node.left:
                res.append((f"<= {node.threshold:.4f}", node.left))
            if node.right:
                res.append((f"> {node.threshold:.4f}", node.right))
        else:
            for val, child in node.branches.items():
                res.append((str(val), child))

        return res

    def get_descendant_graphics_items(self, node_id):
        descendants = []

        start_item = self.node_items.get(node_id)
        if not start_item:
            return descendants

        visited = set()

        def collect(item):
            if id(item) in visited:
                return
            visited.add(id(item))

            for edge in item.edge_items:
                descendants.append(edge)
            for label in item.label_items:
                descendants.append(label)
            for child in item.child_items:
                descendants.append(child)
                collect(child)

        collect(start_item)
        return descendants

    def fade_item(self, item, visible):
        effect = QGraphicsOpacityEffect()
        item.setGraphicsEffect(effect)

        anim = QVariantAnimation(self)
        anim.setDuration(250)
        anim.setStartValue(0.0 if visible else 1.0)
        anim.setEndValue(1.0 if visible else 0.0)

        def on_value_changed(value):
            effect.setOpacity(float(value))

        def on_finished():
            if not visible:
                item.hide()
            else:
                item.show()
                effect.setOpacity(1.0)

        if visible:
            item.show()

        anim.valueChanged.connect(on_value_changed)
        anim.finished.connect(on_finished)
        anim.start()

        self._running_animations.append(anim)

        def cleanup():
            if anim in self._running_animations:
                self._running_animations.remove(anim)

        anim.finished.connect(cleanup)

    def toggle_subtree(self, node_id, visible):
        descendants = self.get_descendant_graphics_items(node_id)
        for item in descendants:
            self.fade_item(item, visible)

    def on_graph_node_clicked(self, node_id):
        if not self.model:
            return

        item = self.node_items[node_id]
        self.graphics_view.smooth_center(item)

        for graph_item in self.node_items.values():
            graph_item.reset_color()

        item.set_highlight(QColor(255, 255, 180))

        node = self.model.get_node(node_id)
        yes_count = node.class_counts.get("Oui", 0)
        no_count = node.class_counts.get("Non", 0)

        lines = [
            f"ID du nœud : {node.node_id}",
            f"Règle : {node.rule_text}",
            f"Nombre d'exemples : {node.samples_count}",
            f"Malades : {yes_count}",
            f"Non malades : {no_count}",
            f"Probabilité de maladie : {node.probability:.2%}",
            f"Prédiction majoritaire : {node.prediction}"
        ]
        self.details_text.setPlainText("\n".join(lines))

        subset = self.df.loc[node.sample_indices].copy() if self.df is not None else pd.DataFrame()
        self.populate_patients_table(subset)

    def populate_patients_table(self, subset):
        self.patients_table.clear()

        if subset is None or subset.empty:
            self.current_table_df = pd.DataFrame()
            self.patients_table.setRowCount(0)
            self.patients_table.setColumnCount(0)
            return

        self.current_table_df = subset.reset_index(drop=True).copy()

        cols = list(self.current_table_df.columns)
        self.patients_table.setRowCount(len(self.current_table_df))
        self.patients_table.setColumnCount(len(cols))
        self.patients_table.setHorizontalHeaderLabels(cols)

        for r, (_, row) in enumerate(self.current_table_df.iterrows()):
            for c, value in enumerate(row):
                self.patients_table.setItem(r, c, QTableWidgetItem(str(value)))

        self.patients_table.resizeColumnsToContents()

    def predict_new_patient(self):
        if not self.model:
            QMessageBox.warning(self, "Attention", "Entraînez d'abord l'arbre.")
            return

        try:
            patient = {}

            for feature in self.prediction_feature_order:
                raw_value = self.inputs[feature].text().strip()

                if raw_value == "":
                    raise ValueError(f"Le champ '{feature}' est vide. Veuillez remplir toutes les valeurs.")

                patient[feature] = float(raw_value)

            pred, path, final_node = self.model.predict_with_path(patient)

            self.predict_result_label.setText(f"Résultat : {pred}")

            for node_id, item in self.node_items.items():
                item.reset_color()

            path_ids = [step["node_id"] for step in path]

            for node_id in path_ids:
                if node_id in self.node_items:
                    self.node_items[node_id].set_highlight(QColor(255, 230, 100))

            if final_node and final_node.node_id in self.node_items:
                final_color = QColor(255, 140, 140) if pred == "Oui" else QColor(140, 255, 140)
                self.node_items[final_node.node_id].set_highlight(final_color)
                self.graphics_view.smooth_center(self.node_items[final_node.node_id])

            explanation_text = self.build_simple_explanation(pred, path, final_node)
            self.path_text.setPlainText(explanation_text)
            self.simple_result_label.setText(
                "Décision expliquée en langage simple dans la zone « Explication simple de la décision »."
            )

        except Exception as e:
            QMessageBox.warning(self, "Erreur", str(e))

        self.hud.hide()

    def build_simple_explanation(self, prediction, path, final_node):
        if not path:
            return (
                "Aucune explication n'a pu être générée.\n\n"
                "Le chemin de décision n'a pas été trouvé."
            )

        intro = (
            "Explication simple de la décision :\n\n"
            "L'arbre a analysé les valeurs médicales saisies une par une. "
            "À chaque étape, il a comparé une mesure de la patiente à une valeur seuil, "
            "puis il a choisi la branche correspondante. "
            "Voici le raisonnement suivi :\n"
        )

        explained_steps = []
        for i, step in enumerate(path, start=1):
            condition = step["condition"]
            value = step["value"]

            if "<=" in condition:
                feature, threshold = condition.split("<=")
                feature = feature.strip()
                threshold = threshold.strip()
                explained_steps.append(
                    f"{i}) La mesure « {feature} » vaut {value:.4f}, "
                    f"elle est inférieure ou égale à {threshold}. "
                    f"L'arbre a donc pris la branche de gauche."
                )
            elif ">" in condition:
                feature, threshold = condition.split(">")
                feature = feature.strip()
                threshold = threshold.strip()
                explained_steps.append(
                    f"{i}) La mesure « {feature} » vaut {value:.4f}, "
                    f"elle est supérieure à {threshold}. "
                    f"L'arbre a donc pris la branche de droite."
                )
            else:
                explained_steps.append(
                    f"{i}) L'arbre a utilisé la règle suivante : {condition}."
                )

        if final_node is not None:
            yes_count = final_node.class_counts.get("Oui", 0)
            no_count = final_node.class_counts.get("Non", 0)
            prob_yes = final_node.probability * 100
            prob_no = 100 - prob_yes
            total = final_node.samples_count
        else:
            yes_count = 0
            no_count = 0
            prob_yes = 0.0
            prob_no = 0.0
            total = 0

        if prediction == "Oui":
            conclusion = (
                "\nConclusion finale :\n\n"
                "Après toutes ces comparaisons, la patiente arrive dans un groupe final "
                f"de {total} cas similaires. Dans ce groupe, {yes_count} cas sont classés « Oui » "
                f"et {no_count} cas sont classés « Non ».\n\n"
                f"Autrement dit, environ {prob_yes:.2f}% des patientes de ce groupe sont considérées "
                "comme malades ou à risque. C'est pour cette raison que l'arbre prédit ici : « Oui »."
            )
        else:
            conclusion = (
                "\nConclusion finale :\n\n"
                "Après toutes ces comparaisons, la patiente arrive dans un groupe final "
                f"de {total} cas similaires. Dans ce groupe, {no_count} cas sont classés « Non » "
                f"et {yes_count} cas sont classés « Oui ».\n\n"
                f"Autrement dit, environ {prob_no:.2f}% des patientes de ce groupe sont considérées "
                "comme non malades ou à risque plus faible. C'est pour cette raison que l'arbre prédit ici : « Non »."
            )

        summary = (
            "\n\nRésumé très simple :\n"
            "La décision finale ne vient pas d'une seule mesure. "
            "Elle vient de l'ensemble des comparaisons effectuées par l'arbre. "
            "Les valeurs saisies ont progressivement orienté la patiente vers un groupe final "
            "dont la majorité des cas ressemble à la prédiction affichée."
        )

        return intro + "\n\n".join(explained_steps) + conclusion + summary

    def show_hud(self, node_id, screen_pos):
        if not self.model:
            return

        node = self.model.get_node(node_id)
        self.hud.setText(
            f"<b>Nœud {node_id}</b><br>"
            f"Nb : {node.samples_count}<br>"
            f"Prob. maladie : {node.probability:.1%}"
        )
        local_pos = self.mapFromGlobal(screen_pos)
        self.hud.move(local_pos + QPoint(15, 15))
        self.hud.show()
        self.hud.raise_()

    def hide_hud(self):
        self.hud.hide()

    def fit_view(self):
        if self.graphics_view.scene():
            self.graphics_view.fitInView(
                self.graphics_view.scene().sceneRect(),
                Qt.KeepAspectRatio
            )

    def resizeEvent(self, event):
        if hasattr(self, "overlay"):
            self.overlay.update_position()
        super().resizeEvent(event)