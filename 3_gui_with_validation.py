import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
from datetime import datetime
import os
import json
import pygad

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==================== 설정 ====================
RPMS = np.arange(1000, 18001, 1000)
N_OUT = len(RPMS)

X_SCALER_PATH = "machine/x_scaler_torque.gz"
Y_SCALER_PATH = "machine/y_scaler_torque.gz"
MODEL_PATH    = "machine/optimal_torque_model.pth"

CLASSIFIER_PATH   = "test/classifier_model.joblib"
CLASS_SCALER_PATH = "test/x_scaler_class.gz"

# ==================== 데이터 기반 파라미터 범위 ====================
RANGE_TABLE = {
    "L1_Magnet_Bar_Width_Ratio": (0.864024, 1.0),
    "Slot_Depth_Ratio": (0.477551, 0.583673),
    "Stator_Bore_Ratio": (0.6052, 0.739689),
    "Tooth_Width_Ratio": (0.605816, 0.740441),
    "Active_Length": (120.6, 147.4),
    "Airgap": (0.63, 0.77),
    "DC_Link_Voltage": (207.9, 254.1),
    "L1_Bridge_Thickness": (1.35, 1.65),
    "L1_Pole_V_Angle": (125.55, 153.45),
    "Max_Phase_Current": (1212.97, 1482.53),
}

GA_DEFAULT = dict(
    seed=42,
    num_generations=80,
    sol_per_pop=40,
    num_parents_mating=12,
    keep_parents=6,
    mutation_percent_genes=15
)

# Matplotlib (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ==================== 모델 ====================
class MultiOutputNN_Regulated(nn.Module):
    def __init__(self, in_dim=11, out_dim=18, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_torque_model():
    try:
        x_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        model = MultiOutputNN_Regulated(in_dim=11, out_dim=N_OUT)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, x_scaler, y_scaler
    except FileNotFoundError as e:
        messagebox.showerror("파일 오류", f"파일을 찾을 수 없습니다:\n{e.filename}")
        return None
    except RuntimeError as e:
        messagebox.showerror("모델 로드 오류", f"모델 weight 로드 실패(구조/키 불일치 가능)\n\n{e}")
        return None

def predict_torque_curve(input_values_11, bundle):
    model, x_scaler, y_scaler = bundle
    x = np.array(input_values_11, dtype=float).reshape(1, -1)
    x_scaled = x_scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy()
    return y_scaler.inverse_transform(pred_scaled)[0]

# ==================== 분류 로더/예측 ====================
def load_classifier_bundle():
    """RandomForest classifier + scaler 로드. 실패해도 GUI는 계속 동작."""
    try:
        clf = joblib.load(CLASSIFIER_PATH)
        sc = joblib.load(CLASS_SCALER_PATH)
        return clf, sc
    except FileNotFoundError as e:
        messagebox.showerror("파일 오류", f"분류 모델 파일을 찾을 수 없습니다:\n{e.filename}")
        return None
    except Exception as e:
        messagebox.showerror("분류 모델 로드 오류", f"분류 모델 로드 실패:\n{e}")
        return None

def predict_validity(input_values_11, param_names_all, clf_bundle):
    """
    input_values_11: 길이 11 (param_names_all 순서)
    return: (is_valid: bool|None, label_str: 'Valid(p=0.93)' / 'Invalid(p=0.03)' / 'N/A')
    """
    if clf_bundle is None:
        return None, "N/A"

    clf, sc = clf_bundle
    try:
        import pandas as pd
        df_in = pd.DataFrame([input_values_11], columns=param_names_all)
        x_s = sc.transform(df_in)
        proba = clf.predict_proba(x_s)[0]  # [p0, p1]
        p_valid = float(proba[1])
        pred = int(clf.predict(x_s)[0])    # 1=유효, 0=무효
        txt = ("Valid" if pred == 1 else "Invalid") + f"(p={p_valid:.2f})"
        return (pred == 1), txt
    except Exception:
        return None, "N/A"

# ==================== 스크롤 프레임 ====================
class ScrollFrame(ttk.Frame):
    def __init__(self, parent, bg, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=bg)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # mouse wheel (Windows)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

# ==================== GUI (Apple-like Light) ====================
class MotorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Torque Curve (Baseline vs GA) — Light UI")
        self.root.geometry("1380x900")

        # Apple-like palette
        self.C_BG = "#f5f5f7"
        self.C_CARD = "#ffffff"
        self.C_BORDER = "#e5e5ea"
        self.C_TEXT = "#1d1d1f"
        self.C_MUTED = "#6e6e73"
        self.C_ACCENT = "#0a84ff"
        self.C_ACCENT_DARK = "#0060df"
        self.C_GREEN = "#34c759"
        self.C_PURPLE = "#5856d6"
        self.C_SLIDER_TROUGH = "#d1d1d6"
        self.C_SLIDER_ACTIVE = "#b8b8bd"

        self.root.configure(bg=self.C_BG)

        self.bundle = load_torque_model()
        if not self.bundle:
            root.destroy()
            return

        # (추가) 분류기 로드 (없어도 앱은 동작)
        self.clf_bundle = load_classifier_bundle()
        self.valid_var = tk.StringVar(value="N/A")
        self._validity_after_id = None

        # Params (학습 순서)
        self.param_names_all = [
            "L1_Pole_V_Angle",
            "L1_Bridge_Thickness",
            "DC_Link_Voltage",
            "Airgap",
            "Turns",
            "Active_Length",
            "Tooth_Width_Ratio",
            "Stator_Bore_Ratio",
            "Slot_Depth_Ratio",
            "Max_Phase_Current",
            "L1_Magnet_Bar_Width_Ratio",
        ]
        self.base_values_all = [
            133.25, 1.40, 252.57, 0.67, 2.0, 122.91, 0.6509, 0.6967, 0.4961, 1407.68, 0.974
        ]

        self.turns_index = self.param_names_all.index("Turns")
        self.fixed_turns_value = float(self.base_values_all[self.turns_index])

        self.gui_param_names = [p for p in self.param_names_all if p != "Turns"]
        self.gui_base_values = [v for i, v in enumerate(self.base_values_all) if i != self.turns_index]
        self.ranges = [RANGE_TABLE.get(p, (v*0.9, v*1.1))
                       for p, v in zip(self.gui_param_names, self.gui_base_values)]

        # state
        self.baseline_inputs11 = None
        self.baseline_curve = None
        self.ga_inputs11 = None
        self.ga_curve = None
        self._click_marker = None

        # 폰트 설정
        self.PLOT_TITLE_FZ = 20
        self.PLOT_LABEL_FZ = 18
        self.PLOT_TICK_FZ  = 16
        self.PLOT_LEGEND_FZ = 14

        self.INFO_FONT = ("Segoe UI", 11)  # 아래 info 라벨 폰트

        # plot resize debounce
        self._pending_resize = None

        self._style()
        self._build_layout()
        self._attach_validity_traces()
        self.run_predict(set_baseline=True)

    # ----------------- styling -----------------
    def _style(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("App.TFrame", background=self.C_BG)
        self.style.configure("Card.TFrame", background=self.C_CARD)
        self.style.configure("Title.TLabel", background=self.C_BG, foreground=self.C_TEXT,
                             font=("Segoe UI", 16, "bold"))
        self.style.configure("Sub.TLabel", background=self.C_BG, foreground=self.C_MUTED,
                             font=("Segoe UI", 10))
        self.style.configure("CardTitle.TLabel", background=self.C_CARD, foreground=self.C_TEXT,
                             font=("Segoe UI", 12, "bold"))
        self.style.configure("Param.TLabel", background=self.C_CARD, foreground=self.C_TEXT,
                             font=("Segoe UI", 10, "bold"))
        self.style.configure("Hint.TLabel", background=self.C_CARD, foreground=self.C_MUTED,
                             font=("Segoe UI", 9))

        self.style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"),
                             padding=(14, 10), background=self.C_ACCENT, foreground="white", borderwidth=0)
        self.style.map("Accent.TButton",
                       background=[("active", self.C_ACCENT_DARK), ("disabled", "#c7c7cc")],
                       foreground=[("disabled", "#f5f5f7")])

        self.style.configure("Green.TButton", font=("Segoe UI", 11, "bold"),
                             padding=(14, 10), background=self.C_GREEN, foreground="white", borderwidth=0)
        self.style.map("Green.TButton",
                       background=[("active", "#248a3d"), ("disabled", "#c7c7cc")],
                       foreground=[("disabled", "#f5f5f7")])

        self.style.configure("Purple.TButton", font=("Segoe UI", 11, "bold"),
                             padding=(14, 10), background=self.C_PURPLE, foreground="white", borderwidth=0)
        self.style.map("Purple.TButton",
                       background=[("active", "#3c3ab6"), ("disabled", "#c7c7cc")],
                       foreground=[("disabled", "#f5f5f7")])

        self.style.configure("Mini.TButton", font=("Segoe UI", 9, "bold"),
                             padding=(8, 6), background="#f2f2f7", foreground=self.C_TEXT, borderwidth=0)
        self.style.map("Mini.TButton",
                       background=[("active", "#e5e5ea"), ("disabled", "#f2f2f7")])

        self.style.configure("TEntry", fieldbackground="white", foreground=self.C_TEXT)
        self.style.configure("TSpinbox", fieldbackground="white", foreground=self.C_TEXT)
        self.style.configure("Vertical.TScrollbar", background=self.C_BG, troughcolor=self.C_BG)

    # ----------------- plot resize helpers -----------------
    def _on_plot_area_resize(self, event):
        # 너무 잦아서 디바운스
        if self._pending_resize is not None:
            try:
                self.root.after_cancel(self._pending_resize)
            except Exception:
                pass
        self._pending_resize = self.root.after(80, self._apply_plot_resize)

    def _apply_plot_resize(self):
        self._pending_resize = None
        try:
            plot_wrap = self.canvas_plot.get_tk_widget().master  # = plot_wrap
            w_px = max(plot_wrap.winfo_width(), 100)
            h_px = max(plot_wrap.winfo_height(), 100)

            # 약간 margin
            pad_px = 10
            w_px = max(w_px - pad_px, 100)
            h_px = max(h_px - pad_px, 100)

            dpi = float(self.fig.get_dpi())
            w_in = w_px / dpi
            h_in = h_px / dpi

            # 너무 작아져도 최소 보장
            w_in = max(w_in, 3.0)
            h_in = max(h_in, 2.6)

            self.fig.set_size_inches(w_in, h_in, forward=True)

            # 레이아웃 재정렬(축/라벨 잘림 방지)
            self.fig.tight_layout(pad=2.0)
            self.canvas_plot.draw_idle()
        except Exception:
            pass

    # ----------------- layout -----------------
    def _build_layout(self):
        app = ttk.Frame(self.root, style="App.TFrame")
        app.pack(fill="both", expand=True)

        # top bar
        top = ttk.Frame(app, style="App.TFrame")
        top.pack(fill="x", padx=18, pady=(14, 10))

        left_top = ttk.Frame(top, style="App.TFrame")
        left_top.pack(side="left", fill="x", expand=True)
        ttk.Label(left_top, text="Torque Curve Optimizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left_top, text="Baseline(예측) vs GA Best(최적) — 두 커브만 비교", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        right_top = ttk.Frame(top, style="App.TFrame")
        right_top.pack(side="right")
        ttk.Label(
            right_top,
            text=f"GA seed={GA_DEFAULT['seed']} · gen={GA_DEFAULT['num_generations']} · pop={GA_DEFAULT['sol_per_pop']}",
            style="Sub.TLabel"
        ).pack(anchor="e")

        # main split
        main = ttk.Frame(app, style="App.TFrame")
        main.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # left card
        left = ttk.Frame(main, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        left.configure(width=430)
        left.grid_propagate(False)

        # right card
        right = ttk.Frame(main, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        # LEFT content
        pad = 16
        header = ttk.Frame(left, style="Card.TFrame")
        header.pack(fill="x", padx=pad, pady=(pad, 10))
        ttk.Label(header, text="Parameters", style="CardTitle.TLabel").pack(anchor="w")

        scroll = ScrollFrame(left, bg=self.C_CARD)
        scroll.pack(fill="both", expand=True, padx=pad, pady=(0, 10))
        scroll.inner.configure(style="Card.TFrame")

        self.param_rows = []
        for i, (name, base_val) in enumerate(zip(self.gui_param_names, self.gui_base_values)):
            low, high = self.ranges[i]

            row = ttk.Frame(scroll.inner, style="Card.TFrame")
            row.pack(fill="x", pady=8)

            top_line = ttk.Frame(row, style="Card.TFrame")
            top_line.pack(fill="x")

            ttk.Label(top_line, text=name, style="Param.TLabel").pack(side="left")

            init_val = float(base_val)
            if init_val < low: init_val = float(low)
            if init_val > high: init_val = float(high)
            var = tk.DoubleVar(value=init_val)

            reset_btn = ttk.Button(top_line, text="Reset", style="Mini.TButton",
                                   command=lambda v=var, bv=init_val: v.set(float(bv)))
            reset_btn.pack(side="right", padx=(6, 0))

            spin = ttk.Spinbox(
                top_line, textvariable=var, width=12,
                from_=low, to=high, increment=(high-low)/200 if (high-low) != 0 else 0.001,
                justify="right"
            )
            spin.pack(side="right")
            spin.bind("<FocusOut>", lambda e, v=var, lo=low, hi=high: self._clamp_var(v, lo, hi))
            spin.bind("<Return>", lambda e, v=var, lo=low, hi=high: self._clamp_var(v, lo, hi))

            ttk.Label(row, text=f"{low:.4g}  —  {high:.4g}", style="Hint.TLabel").pack(anchor="w", pady=(2, 4))

            scale = tk.Scale(
                row, from_=low, to=high, orient="horizontal",
                showvalue=False, resolution=(high-low)/200 if (high-low) != 0 else 0.001,
                sliderrelief="flat",
                troughcolor=self.C_SLIDER_TROUGH,
                activebackground=self.C_SLIDER_ACTIVE,
                highlightthickness=0, bd=0, relief="flat",
                length=340
            )
            scale.set(init_val)
            scale.pack(fill="x")

            def _on_scale(val, v=var):
                try:
                    v.set(float(val))
                except Exception:
                    pass
            scale.configure(command=_on_scale)

            def _sync_from_var(*args, s=scale, v=var, lo=low, hi=high):
                try:
                    x = float(v.get())
                    if x < lo: x = lo
                    if x > hi: x = hi
                    s.set(x)
                except Exception:
                    pass
            var.trace_add("write", _sync_from_var)

            self.param_rows.append(dict(name=name, var=var, spin=spin, scale=scale, low=low, high=high))

            sep = ttk.Separator(scroll.inner, orient="horizontal")
            sep.pack(fill="x", pady=(0, 0))

        # Turns fixed
        turns = ttk.Frame(left, style="Card.TFrame")
        turns.pack(fill="x", padx=pad, pady=(0, 12))
        turns_label = tk.Label(
            turns, text=f"Turns (Fixed): {self.fixed_turns_value:g}",
            bg=self.C_CARD, fg=self.C_MUTED, font=("Segoe UI", 10, "bold"),
            padx=12, pady=10
        )
        turns_label.pack(fill="x")

        # Buttons
        btns = ttk.Frame(left, style="Card.TFrame")
        btns.pack(fill="x", padx=pad, pady=(0, pad))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        self.btn_predict = ttk.Button(btns, text="Baseline 예측", style="Accent.TButton",
                                      command=lambda: self.run_predict(set_baseline=True))
        self.btn_predict.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.btn_save = ttk.Button(btns, text="저장 (그림+파라미터)", style="Green.TButton",
                                   command=self.on_save_plot)
        self.btn_save.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.btn_ga = ttk.Button(btns, text="GA 최적화", style="Purple.TButton",
                                 command=self.run_ga_idea)
        self.btn_ga.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Validity
        validity_row = ttk.Frame(left, style="Card.TFrame")
        validity_row.pack(fill="x", padx=pad, pady=(0, 6))

        tk.Label(
            validity_row,
            text="Classifier:",
            bg=self.C_CARD, fg=self.C_MUTED,
            font=("Segoe UI", 9)
        ).pack(side="left")

        self.validity_badge = tk.Label(
            validity_row,
            textvariable=self.valid_var,
            bg=self.C_CARD, fg=self.C_MUTED,
            font=("Segoe UI", 9, "bold"),
            padx=8, pady=2
        )
        self.validity_badge.pack(side="left", padx=(6, 0))

        self.status = tk.StringVar(value="준비 완료")
        status_lbl = tk.Label(
            left, textvariable=self.status,
            bg=self.C_CARD, fg=self.C_MUTED,
            font=("Segoe UI", 10),
            wraplength=390, justify="left", padx=16, pady=12
        )
        status_lbl.pack(fill="x", padx=0, pady=(6, 0))

        # RIGHT content (✅ 반응형 리사이즈)
        plot_wrap = ttk.Frame(right, style="Card.TFrame")
        plot_wrap.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        plot_wrap.rowconfigure(0, weight=1)
        plot_wrap.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(9.4, 6.4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_wrap)
        w = self.canvas_plot.get_tk_widget()
        w.grid(row=0, column=0, sticky="nsew")

        plot_wrap.bind("<Configure>", self._on_plot_area_resize)

        info = tk.Label(
            right,
            text="그래프를 클릭하면 해당 RPM의 Baseline/GA 토크를 표시합니다.",
            bg=self.C_CARD, fg=self.C_MUTED,
            font=self.INFO_FONT,
            padx=16, pady=12
        )
        info.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.info_label = info

        self.canvas_plot.mpl_connect("button_press_event", self.on_click)

    # ----------------- helpers -----------------
    def _clamp_var(self, var, lo, hi):
        try:
            x = float(var.get())
        except Exception:
            return
        if x < lo: x = lo
        if x > hi: x = hi
        var.set(float(x))

    def _update_validity_badge(self, inputs11):
        _, txt = predict_validity(inputs11, self.param_names_all, self.clf_bundle)
        self.valid_var.set(txt)
        if txt.startswith("Valid"):
            self.validity_badge.config(fg="#1f7a1f")
        elif txt.startswith("Invalid"):
            self.validity_badge.config(fg="#b00020")
        else:
            self.validity_badge.config(fg=self.C_MUTED)

    def _schedule_validity_update(self, *args):
        if self._validity_after_id is not None:
            try:
                self.root.after_cancel(self._validity_after_id)
            except Exception:
                pass
        self._validity_after_id = self.root.after(150, self._update_validity_only)

    def _update_validity_only(self):
        self._validity_after_id = None
        try:
            inputs11 = self._collect_input_11()
            self._update_validity_badge(inputs11)
        except Exception:
            pass

    def _attach_validity_traces(self):
        try:
            for r in self.param_rows:
                r["var"].trace_add("write", self._schedule_validity_update)
        except Exception:
            pass

    # ----------------- data helpers -----------------
    def _collect_input_11(self):
        vals10 = [float(r["var"].get()) for r in self.param_rows]
        vals11 = list(vals10)
        vals11.insert(self.turns_index, float(self.fixed_turns_value))
        return vals11

    def _apply_solution_10(self, sol10):
        for r, v in zip(self.param_rows, sol10):
            r["var"].set(float(v))

    def _extend_curve(self, curve):
        ext_r = np.concatenate(([0], RPMS))
        ext_t = np.concatenate(([curve[0]], curve))
        return ext_r, ext_t

    def _interp_torque(self, curve, rpm):
        ext_r, ext_t = self._extend_curve(curve)
        return float(np.interp(rpm, ext_r, ext_t))

    # ----------------- predict/draw -----------------
    def run_predict(self, set_baseline=True):
        try:
            inputs11 = self._collect_input_11()
            curve = predict_torque_curve(inputs11, self.bundle)
            self._update_validity_badge(inputs11)
        except Exception as e:
            self.status.set(f"❌ 예측 실패: {e}")
            return
        if set_baseline:
            self.baseline_inputs11 = inputs11
            self.baseline_curve = curve
            self.status.set(f"✅ Baseline 업데이트 | Avg Torque={float(np.mean(curve)):.3f}")
        self._redraw()

    def _redraw(self):
        self.ax.clear()

        self.ax.set_title("T-N Curve", fontsize=self.PLOT_TITLE_FZ, pad=15)
        self.ax.set_xlabel("RPM", fontsize=self.PLOT_LABEL_FZ)
        self.ax.set_ylabel("Torque (Nm)", fontsize=self.PLOT_LABEL_FZ)
        self.ax.tick_params(axis="both", labelsize=self.PLOT_TICK_FZ)
        self.ax.grid(True, linestyle="--", alpha=0.35)

        if self._click_marker is not None:
            try: self._click_marker.remove()
            except Exception: pass
            self._click_marker = None

        x = np.linspace(0, RPMS[-1], 600)

        if self.baseline_curve is not None:
            er, et = self._extend_curve(self.baseline_curve)
            y = np.interp(x, er, et)
            self.ax.plot(x, y, linewidth=3.2, label="Baseline")
            self.ax.fill_between(x, y, alpha=0.06)

        if self.ga_curve is not None:
            er, et = self._extend_curve(self.ga_curve)
            y = np.interp(x, er, et)
            self.ax.plot(x, y, "--", linewidth=3.2, label="GA Best")

        if (self.baseline_curve is not None) or (self.ga_curve is not None):
            self.ax.legend(loc="best", fontsize=self.PLOT_LEGEND_FZ)

        # ✅ 라벨 잘림 방지
        self.fig.tight_layout(pad=2.0)
        self.canvas_plot.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        rpm = float(np.clip(event.xdata, 0, RPMS[-1]))

        parts = [f"RPM {rpm:.0f}"]
        y_base = None
        y_ga = None
        if self.baseline_curve is not None:
            y_base = self._interp_torque(self.baseline_curve, rpm)
            parts.append(f"Baseline {y_base:.2f} Nm")
        if self.ga_curve is not None:
            y_ga = self._interp_torque(self.ga_curve, rpm)
            parts.append(f"GA {y_ga:.2f} Nm")

        self.info_label.config(text="  |  ".join(parts))

        y_marker = y_base if y_base is not None else y_ga
        if y_marker is None:
            return
        if self._click_marker is not None:
            try: self._click_marker.remove()
            except Exception: pass
            self._click_marker = None

        (m,) = self.ax.plot(rpm, y_marker, marker="o", markersize=8, zorder=10)
        self._click_marker = m
        self.canvas_plot.draw_idle()

    # ----------------- save -----------------
    def on_save_plot(self):
        return self._on_save_plot_impl()

    def _on_save_plot_impl(self):
        if self.baseline_curve is None and self.ga_curve is None:
            messagebox.showinfo("저장", "저장할 그래프가 없습니다.")
            return

        default_name = f"TN_baseline_vs_GA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png")]
        )
        if not file_path:
            return

        base = os.path.splitext(file_path)[0]
        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")

            records = []
            if self.baseline_inputs11 is not None:
                for n, v in zip(self.param_names_all, self.baseline_inputs11):
                    records.append({"set": "baseline", "param": n, "value": float(v)})
            if self.ga_inputs11 is not None:
                for n, v in zip(self.param_names_all, self.ga_inputs11):
                    records.append({"set": "ga_best", "param": n, "value": float(v)})

            import pandas as pd
            pd.DataFrame(records).to_csv(base + "_params.csv", index=False, encoding="utf-8-sig")
            with open(base + "_params.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            df = pd.DataFrame({"rpm": RPMS})
            if self.baseline_curve is not None:
                df["torque_baseline"] = self.baseline_curve
            if self.ga_curve is not None:
                df["torque_ga_best"] = self.ga_curve
            df.to_csv(base + "_curves.csv", index=False, encoding="utf-8-sig")

            messagebox.showinfo("저장 완료", f"저장 완료!\n\n{file_path}\n{base}_params.csv\n{base}_curves.csv")
        except Exception as e:
            messagebox.showerror("저장 실패", f"저장 중 오류:\n{e}")

    # ----------------- GA -----------------
    def run_ga_idea(self):
        if self.baseline_curve is None:
            self.run_predict(set_baseline=True)

        self.btn_ga.state(["disabled"])
        self.btn_predict.state(["disabled"])
        self.status.set("🧬 GA 실행 중...")
        self.root.update()

        gene_space = [{"low": float(lo), "high": float(hi)} for (lo, hi) in self.ranges]
        pygad.random.seed(GA_DEFAULT["seed"])
        np.random.seed(GA_DEFAULT["seed"])

        base_score = float(np.mean(self.baseline_curve)) if self.baseline_curve is not None else None

        def fitness_func(ga_instance, solution, solution_idx):
            sol10 = np.array(solution, dtype=float)
            inputs11 = list(sol10)
            inputs11.insert(self.turns_index, float(self.fixed_turns_value))
            try:
                curve = predict_torque_curve(inputs11, self.bundle)
                score = float(np.mean(curve))
                neg = curve[curve < 0]
                if neg.size > 0:
                    score -= 50.0 * float(np.mean(np.abs(neg)))
                return score
            except Exception:
                return -1e9

        try:
            ga = pygad.GA(
                num_generations=GA_DEFAULT["num_generations"],
                sol_per_pop=GA_DEFAULT["sol_per_pop"],
                num_parents_mating=GA_DEFAULT["num_parents_mating"],
                num_genes=len(self.gui_param_names),
                gene_space=gene_space,
                fitness_func=fitness_func,
                parent_selection_type="rank",
                keep_parents=GA_DEFAULT["keep_parents"],
                crossover_type="single_point",
                mutation_type="random",
                mutation_percent_genes=GA_DEFAULT["mutation_percent_genes"],
                suppress_warnings=True,
                stop_criteria=["saturate_25"],
            )
            ga.run()
# ... (기존 ga.run() 코드)

            # --- [추가 코드: 수렴 과정 시각화 및 저장] ---
            # 1. PyGAD 내장 함수로 즉시 그래프 확인 (별도 창 생성)
            # 논문용 Fig 6를 위한 기초 자료가 됩니다.
            # --- [수정] 에러 유발 코드 주석 처리 및 데이터 처리 ---
            # ga.plot_fitness(...)  <-- ❌ 이 코드가 Fatal Error의 원인입니다. 주석 처리하세요.
            
            # 1. 수렴 데이터 추출 (데이터만 가져오기)
            fitness_history = np.array(ga.best_solutions_fitness)
            np.save("machine/ga_fitness_history.npy", fitness_history) # 데이터 저장
            
            # 2. 결과 시각화 (기존 GUI 그래프 활용)
            # GA가 끝난 후 최적의 커브를 화면에 바로 그립니다.
            best_sol, best_fit, _ = ga.best_solution()
            best_sol = np.array(best_sol, dtype=float)
            
            self._apply_solution_10(best_sol.tolist())
            self.ga_inputs11 = self._collect_input_11()
            self.ga_curve = predict_torque_curve(self.ga_inputs11, self.bundle)
            
            # 3. GUI 업데이트 (메인 스레드에서 안전하게 호출)
            self._redraw() 
            
            # 4. 별도의 수렴 그래프가 필요하다면 파일로 저장 (팝업 없이)
            import matplotlib.pyplot as plt
            fig_conv, ax_conv = plt.subplots()
            ax_conv.plot(fitness_history)
            ax_conv.set_title("GA Convergence")
            ax_conv.set_xlabel("Generation")
            ax_conv.set_ylabel("Fitness")
            fig_conv.savefig("machine/ga_convergence.png")
            plt.close(fig_conv) # 메모리 해제
            # --------------------------------------------------

            msg = f"✅ GA 완료 | Best Avg={best_fit:.3f} ({len(fitness_history)}세대 수렴)"
            self.status.set(msg)
            # ------------------------------------------

            best_sol, best_fit, _ = ga.best_solution()
# ... (이후 기존 코드)
            best_sol = np.array(best_sol, dtype=float)

            self._apply_solution_10(best_sol.tolist())

            self.ga_inputs11 = self._collect_input_11()
            self.ga_curve = predict_torque_curve(self.ga_inputs11, self.bundle)
            self._update_validity_badge(self.ga_inputs11)

            self._redraw()

            msg = f"✅ GA 완료 | Best Avg={best_fit:.3f}"
            if base_score is not None:
                msg += f"  (Baseline {base_score:.3f}, Δ {best_fit-base_score:.3f})"
            self.status.set(msg)

        except Exception as e:
            messagebox.showerror("GA 오류", f"GA 실행 중 오류:\n{e}")
            self.status.set("❌ GA 실패")
        finally:
            self.btn_ga.state(["!disabled"])
            self.btn_predict.state(["!disabled"])

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorApp(root)
    root.mainloop()
