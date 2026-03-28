import pandas as pd
from PySide6.QtCore import (Qt, QRectF, QPointF, QThread, Signal, QPropertyAnimation, 
                            QEasingCurve, QVariantAnimation, QPoint)
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPainter
from PySide6.QtWidgets import (
    QFormLayout, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QLineEdit, QFrame, QGraphicsOpacityEffect
)

from c45 import C45DecisionTree
from dataset_loader import load_breast_cancer_dataset
from metrics_utils import evaluate_model

# --- BACKGROUND THREAD ---
class TrainingWorker(QThread):
    finished = Signal(object, dict)
    error = Signal(str)

    def __init__(self, df, target, pos_class, excluded):
        super().__init__()
        self.df, self.target, self.pos_class, self.excluded = df, target, pos_class, excluded

    def run(self):
        try:
            model = C45DecisionTree(min_samples_split=5, max_depth=6, 
                                   target_column=self.target, positive_class=self.pos_class, 
                                   excluded_features=self.excluded)
            model.fit(self.df)
            X = self.df.drop(columns=[self.target])
            y_true = self.df[self.target].tolist()
            y_pred = model.predict(X)
            metrics = evaluate_model(y_true, y_pred)
            self.finished.emit(model, metrics)
        except Exception as e: self.error.emit(str(e))

# --- LOADING OVERLAY ---
class LoadingOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 200); border: 2px solid #ccc; border-radius: 10px;")
        layout = QVBoxLayout(self)
        self.label = QLabel("<b>⚙️ Training Decision Tree...</b><br>Please wait.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.hide()

    def update_position(self):
        if self.parent(): self.setFixedSize(self.parent().size())

# --- INTERACTIVE NODE ITEM ---
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
        self.text_item.setPos(rect.x() + 5, rect.y() + 5)

    def mousePressEvent(self, event):
        self.main_win.on_graph_node_clicked(self.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.is_collapsed = not self.is_collapsed
        self.recursive_toggle(not self.is_collapsed)
        super().mouseDoubleClickEvent(event)

    def recursive_toggle(self, visible):
        for item in self.child_items + self.edge_items + self.label_items:
            eff = QGraphicsOpacityEffect()
            item.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity")
            anim.setDuration(350)
            anim.setStartValue(0.0 if visible else 1.0)
            anim.setEndValue(1.0 if visible else 0.0)
            if visible: item.show()
            else: anim.finished.connect(item.hide)
            anim.start()
            if isinstance(item, GraphicsNodeItem) and not visible:
                item.recursive_toggle(False)
            elif isinstance(item, GraphicsNodeItem) and not item.is_collapsed:
                item.recursive_toggle(True)

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

# --- VIEWER ---
class TreeViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor(245, 247, 249)))

    def wheelEvent(self, event):
        f = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def smooth_center(self, item):
        self.anim = QVariantAnimation()
        self.anim.setDuration(500)
        self.anim.setStartValue(self.mapToScene(self.viewport().rect().center()))
        self.anim.setEndValue(item.sceneBoundingRect().center())
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.valueChanged.connect(self.centerOn)
        self.anim.start()

# --- MAIN WINDOW ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logiciel Desktop - Arbre de décision C4.5 / Cancer du sein")
        self.resize(1700, 950)

        self.target_column = "maladie"
        self.id_column, self.name_column = "patient_id", "nom"
        self.df, self.model, self.node_items = None, None, {}

        self._build_ui()
        self.overlay = LoadingOverlay(self.graphics_view)
        self.hud = QLabel(self)
        self.hud.setStyleSheet("background: rgba(45, 45, 45, 220); color: white; padding: 8px; border-radius: 5px; border: 1px solid white;")
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud.hide()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_load_dataset = QPushButton("Charger le dataset cancer du sein")
        self.btn_train = QPushButton("Entraîner l'arbre (Threaded)")
        self.btn_predict = QPushButton("Prédire nouvelle patiente")
        self.btn_reset = QPushButton("🎯 Centrer Vue")
        for b in [self.btn_load_dataset, self.btn_train, self.btn_predict, self.btn_reset]: b.setMinimumHeight(35); top.addWidget(b)
        top.addStretch(); main_layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Interactive Map
        self.graphics_view = TreeViewer()
        splitter.addWidget(self.graphics_view)

        # Middle: Original Text Panels
        middle_widget = QWidget(); middle_layout = QVBoxLayout(middle_widget)
        self.dataset_info_text = QTextEdit(); self.details_text = QTextEdit()
        self.metrics_text = QTextEdit(); self.path_text = QTextEdit()
        for l in [self.dataset_info_text, self.details_text, self.metrics_text, self.path_text]: l.setReadOnly(True)
        
        middle_layout.addWidget(QLabel("Informations du dataset")); middle_layout.addWidget(self.dataset_info_text)
        middle_layout.addWidget(QLabel("Détails du nœud sélectionné")); middle_layout.addWidget(self.details_text)
        middle_layout.addWidget(QLabel("Métriques globales")); middle_layout.addWidget(self.metrics_text)
        middle_layout.addWidget(QLabel("Chemin de décision de la nouvelle patiente")); middle_layout.addWidget(self.path_text)
        splitter.addWidget(middle_widget)

        # Right: Patient Table & Original Predictor
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        self.patients_table = QTableWidget()
        right_layout.addWidget(QLabel("Patientes du nœud / feuille sélectionné(e)")); right_layout.addWidget(self.patients_table)
        
        prediction_group = QGroupBox("Tester une nouvelle patiente")
        prediction_layout = QFormLayout(prediction_group); self.inputs = {}
        prediction_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
        for feature in prediction_features:
            line_edit = QLineEdit(); self.inputs[feature] = line_edit
            prediction_layout.addRow(feature, line_edit)
        
        self.predict_result_label = QLabel("Résultat : -")
        prediction_layout.addRow("Prédiction", self.predict_result_label)
        right_layout.addWidget(prediction_group)

        splitter.addWidget(right_widget); splitter.setSizes([700, 500, 500]); main_layout.addWidget(splitter)

        # Connections
        self.btn_load_dataset.clicked.connect(self.load_default_dataset)
        self.btn_train.clicked.connect(self.train_model)
        self.btn_predict.clicked.connect(self.predict_new_patient)
        self.btn_reset.clicked.connect(self.fit_view)

    def load_default_dataset(self):
        try:
            self.df = load_breast_cancer_dataset()
            self.clear_graph()
            self.populate_patients_table(self.df.head(30))
            self.show_dataset_info()
        except Exception as e: QMessageBox.critical(self, "Erreur", str(e))

    def clear_graph(self):
        self.scene = QGraphicsScene(); self.graphics_view.setScene(self.scene); self.node_items = {}

    def show_dataset_info(self):
        if self.df is None: return
        class_counts = self.df[self.target_column].value_counts().to_dict()
        lines = [f"Dataset : Breast Cancer Wisconsin", f"Nombre de patientes : {len(self.df)}", "",
                 "Répartition des classes :", f"- Oui (malade) : {class_counts.get('Oui', 0)}", f"- Non (non malade) : {class_counts.get('Non', 0)}"]
        self.dataset_info_text.setPlainText("\n".join(lines))

    def train_model(self):
        if self.df is None: return
        self.btn_train.setEnabled(False); self.overlay.show(); self.overlay.update_position()
        self.worker = TrainingWorker(self.df, self.target_column, "Oui", [self.id_column, self.name_column])
        self.worker.finished.connect(self.on_training_done)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Erreur", e))
        self.worker.start()

    def on_training_done(self, model, metrics):
        self.model = model; self.btn_train.setEnabled(True); self.overlay.hide()
        self.metrics_text.setPlainText(f"Accuracy : {metrics['accuracy']:.4f}\n\n{metrics['classification_report']}")
        self.draw_graphical_tree()

    def draw_graphical_tree(self):
        self.clear_graph()
        if not self.model or not self.model.root: return
        pos = {}; leaf_count = [0]; h_gap, v_gap = 220, 140; w, h = 170, 80

        def calc_pos(node, depth=0):
            kids = self.get_children_node(node)
            if not kids: pos[node.node_id] = (leaf_count[0] * h_gap, depth * v_gap); leaf_count[0] += 1
            else:
                for _, k in kids: calc_pos(k, depth + 1)
                pos[node.node_id] = (sum(pos[k.node_id][0] for _, k in kids)/len(kids), depth * v_gap)
        
        calc_pos(self.model.root)

        # Create items
        for nid, (px, py) in pos.items():
            node = self.model.get_node(nid)
            color = QColor(255, 210, 210) if node.prediction == "Oui" else QColor(210, 255, 210)
            if not node.is_leaf: color = QColor(230, 240, 255)
            txt = f"Nœud {nid}\n{node.attribute[:15] if node.attribute else 'Feuille'}\nNb: {node.samples_count}"
            item = GraphicsNodeItem(QRectF(px, py, w, h), nid, txt, self, color)
            self.scene.addItem(item); self.node_items[nid] = item

        # Connect children & edges with original labels
        for nid, item in self.node_items.items():
            node = self.model.get_node(nid)
            for label, k_node in self.get_children_node(node):
                child_item = self.node_items[k_node.node_id]
                line = self.scene.addLine(item.rect().center().x(), item.rect().bottom(), 
                                          child_item.rect().center().x(), child_item.rect().top(), QPen(Qt.black, 1))
                lbl = QGraphicsSimpleTextItem(label); lbl.setPos((item.rect().center().x()+child_item.rect().center().x())/2, (item.rect().bottom()+child_item.rect().top())/2)
                self.scene.addItem(lbl)
                item.edge_items.append(line); item.child_items.append(child_item); item.label_items.append(lbl)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)); self.fit_view()

    def get_children_node(self, node):
        res = []
        if node.threshold is not None:
            if node.left: res.append((f"<= {node.threshold:.2f}", node.left))
            if node.right: res.append((f"> {node.threshold:.2f}", node.right))
        else:
            for val, child in node.branches.items(): res.append((str(val), child))
        return res

    def on_graph_node_clicked(self, nid):
        item = self.node_items[nid]
        self.graphics_view.smooth_center(item)
        for i in self.node_items.values(): i.reset_color()
        item.set_highlight(QColor(255, 255, 180))
        
        node = self.model.get_node(nid)
        yes_count = node.class_counts.get("Oui", 0); no_count = node.class_counts.get("Non", 0)
        lines = [f"ID du nœud : {node.node_id}", f"Règle : {node.rule_text}", f"Nb : {node.samples_count}",
                 f"Malades : {yes_count}", f"Non malades : {no_count}", f"Probabilité : {node.probability:.2%}"]
        self.details_text.setPlainText("\n".join(lines))
        self.populate_patients_table(self.df.loc[node.sample_indices].copy())

    def populate_patients_table(self, subset):
        self.patients_table.clear()
        if subset.empty: return
        cols = list(subset.columns[:12])
        self.patients_table.setRowCount(len(subset)); self.patients_table.setColumnCount(len(cols))
        self.patients_table.setHorizontalHeaderLabels(cols)
        for r, (_, row) in enumerate(subset[cols].iterrows()):
            for c, v in enumerate(row): self.patients_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.patients_table.resizeColumnsToContents()

    def predict_new_patient(self):
        if not self.model: return
        try:
            patient = {f: float(self.inputs[f].text()) for f in self.inputs}
            for col in self.df.columns:
                if col not in patient and col != self.target_column:
                    patient[col] = self.df[col].mean() if pd.api.types.is_numeric_dtype(self.df[col]) else self.df[col].mode()[0]
            
            pred, path, final = self.model.predict_with_path(patient)
            self.predict_result_label.setText(f"Résultat : {pred}")
            
            path_ids = [s["node_id"] for s in path]
            for nid, item in self.node_items.items():
                item.reset_color()
                if nid in path_ids: item.set_highlight(QColor(255, 230, 100))
            if final: 
                self.node_items[final.node_id].set_highlight(QColor(255, 140, 140) if pred == "Oui" else QColor(140, 255, 140))
                self.graphics_view.smooth_center(self.node_items[final.node_id])

            self.path_text.setPlainText("\n".join([f"{i+1}. {s['condition']} (Valeur: {s['value']})" for i, s in enumerate(path)]))
        except Exception as e: QMessageBox.warning(self, "Erreur", str(e))

    def show_hud(self, nid, screen_pos):
        n = self.model.get_node(nid)
        self.hud.setText(f"<b>Nœud {nid}</b><br>Nb: {n.samples_count}<br>Prob Maladie: {n.probability:.1%}")
        local_pos = self.mapFromGlobal(screen_pos); self.hud.move(local_pos + QPoint(15, 15)); self.hud.show(); self.hud.raise_()

    def hide_hud(self): self.hud.hide()
    
    def fit_view(self):
        if self.graphics_view.scene(): self.graphics_view.fitInView(self.graphics_view.scene().sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        if hasattr(self, 'overlay'): self.overlay.update_position()
        super().resizeEvent(event)