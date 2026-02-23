"""
regimes.py
----------
Classifies monthly observations into inflation regimes using two methods:
    1. Threshold-based  (fast, interpretable)
    2. Hidden Markov Model (HMM) (data-driven)

Regime Taxonomy
---------------
    LOW_STABLE   : CPI YoY < 2.5%  AND  momentum flat
    RISING       : CPI YoY trending up
    HIGH_STABLE  : CPI YoY > 4.0%  AND  momentum flat
    FALLING      : CPI YoY trending down from elevated levels
"""

import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "Low & Stable",
    1: "Rising",
    2: "High & Stable",
    3: "Falling",
}

REGIME_COLORS = {
    "Low & Stable" : "#2ecc71",
    "Rising"       : "#e67e22",
    "High & Stable": "#e74c3c",
    "Falling"      : "#3498db",
    "Unknown"      : "#95a5a6",
}


class RegimeClassifier:
    """
    Classifies monthly macro data into inflation regimes.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Output of DataFetcher.get_macro(). Must contain 'cpi_yoy' and 'cpi_3m_ann'.
    cpi_col  : str
        Column to use as primary inflation measure (default: 'cpi_yoy').
    """

    def __init__(
        self,
        macro_df: pd.DataFrame,
        cpi_col: str = "cpi_yoy",
    ):
        self.df      = macro_df.copy()
        self.cpi_col = cpi_col
        self._validate()

    def _validate(self):
        required = [self.cpi_col, "cpi_3m_ann"]
        missing  = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"macro_df missing columns: {missing}")

    def classify_threshold(
        self,
        low_threshold: float      = 2.5,
        high_threshold: float     = 4.0,
        momentum_window: int      = 6,
        momentum_threshold: float = 0.3,
    ) -> pd.DataFrame:
        """
        Rule-based regime classification using CPI level + momentum.
        """
        df  = self.df.copy()
        cpi = df[self.cpi_col].dropna()

        df["cpi_momentum"] = cpi.diff(momentum_window)

        conditions = []
        labels     = []

        conditions.append(
            (df[self.cpi_col] >= high_threshold) &
            (df["cpi_momentum"].abs() <= momentum_threshold)
        )
        labels.append("High & Stable")

        conditions.append(df["cpi_momentum"] > momentum_threshold)
        labels.append("Rising")

        conditions.append(
            (df["cpi_momentum"] < -momentum_threshold) &
            (df[self.cpi_col] > low_threshold)
        )
        labels.append("Falling")

        df["regime_label"] = "Low & Stable"
        for cond, label in zip(reversed(conditions), reversed(labels)):
            df.loc[cond, "regime_label"] = label

        label_to_id = {v: k for k, v in REGIME_LABELS.items()}
        label_to_id["High & Stable"] = 2
        df["regime_id"] = df["regime_label"].map(label_to_id).fillna(0).astype(int)

        counts = df["regime_label"].value_counts()
        log.info("Threshold regime distribution:")
        for label, n in counts.items():
            pct = n / len(df) * 100
            log.info(f"  {label:<18}: {n:4d} months ({pct:.1f}%)")

        return df

    def classify_hmm(
        self,
        n_states: int = 4,
        features: Optional[list] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        HMM-based regime classification using Gaussian emissions.
        States are labelled by sorting on mean CPI YoY.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("Install hmmlearn:  pip install hmmlearn")

        if features is None:
            features = ["cpi_yoy", "cpi_3m_ann", "real_fed_funds"]

        df      = self.df.copy()
        feat_df = df[features].dropna()
        X       = feat_df.values

        from sklearn.preprocessing import StandardScaler
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        log.info(f"Fitting GaussianHMM with {n_states} states on {len(X)} observations...")
        model = GaussianHMM(
            n_components    = n_states,
            covariance_type = "full",
            n_iter          = 200,
            random_state    = random_state,
        )
        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)

        feat_df = feat_df.copy()
        feat_df["hmm_state"] = hidden_states

        state_means = (
            feat_df.groupby("hmm_state")["cpi_yoy"]
            .mean()
            .sort_values()
        )
        state_order = state_means.index.tolist()
        rank_map    = {raw_state: rank for rank, raw_state in enumerate(state_order)}

        feat_df["hmm_regime_id"]    = feat_df["hmm_state"].map(rank_map)
        feat_df["hmm_regime_label"] = feat_df["hmm_regime_id"].map(REGIME_LABELS)

        df = df.join(feat_df[["hmm_state", "hmm_regime_id", "hmm_regime_label"]], how="left")
        return df

    @staticmethod
    def regime_stats(df: pd.DataFrame, regime_col: str = "regime_label") -> pd.DataFrame:
        """Summary statistics per regime."""
        cols = [
            "cpi_yoy", "core_cpi_yoy", "cpi_3m_ann",
            "fed_funds", "real_fed_funds", "yield_curve_slope",
            "breakeven_10y", "inflation_surprise",
        ]
        cols = [c for c in cols if c in df.columns]
        return df.groupby(regime_col)[cols].agg(["mean", "std", "count"]).round(2)

    @staticmethod
    def regime_spans(df: pd.DataFrame, regime_col: str = "regime_label") -> pd.DataFrame:
        """Returns contiguous regime spans with start/end dates for chart shading."""
        labels  = df[regime_col].dropna()
        spans   = []
        current = labels.iloc[0]
        start   = labels.index[0]

        for date, label in labels.items():
            if label != current:
                spans.append({"regime": current, "start": start, "end": date})
                current = label
                start   = date
        spans.append({"regime": current, "start": start, "end": labels.index[-1]})
        return pd.DataFrame(spans)

    def plot_regimes(
        self,
        df: Optional[pd.DataFrame] = None,
        regime_col: str = "regime_label",
        title: str = "Inflation Regimes (1960–Present)",
    ):
        """Plotly chart: CPI YoY line with regime background shading."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        if df is None:
            df = self.classify_threshold()

        spans = self.regime_spans(df, regime_col)
        cpi   = df["cpi_yoy"].dropna()

        fig = go.Figure()

        for _, row in spans.iterrows():
            color = REGIME_COLORS.get(row["regime"], "#cccccc")
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_vrect(
                x0         = row["start"],
                x1         = row["end"],
                fillcolor  = f"rgba({r},{g},{b},0.15)",
                line_width = 0,
            )

        fig.add_trace(go.Scatter(
            x    = cpi.index,
            y    = cpi.values,
            name = "CPI YoY %",
            line = dict(color="#2c3e50", width=1.5),
            hovertemplate = "%{x|%b %Y}: %{y:.2f}%<extra></extra>",
        ))

        for level, label, color in [
            (2.5, "Low threshold (2.5%)", "#2ecc71"),
            (4.0, "High threshold (4.0%)", "#e74c3c"),
        ]:
            fig.add_hline(
                y                   = level,
                line_dash           = "dash",
                line_color          = color,
                annotation_text     = label,
                annotation_position = "bottom right",
            )

        for label, color in REGIME_COLORS.items():
            if label == "Unknown":
                continue
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x      = [None],
                y      = [None],
                mode   = "markers",
                marker = dict(size=12, color=color, symbol="square"),
                name   = label,
            ))

        fig.update_layout(
            title     = dict(text=title, font=dict(size=18)),
            xaxis     = dict(title="Date", showgrid=False),
            yaxis     = dict(title="CPI Year-over-Year %", zeroline=True),
            hovermode = "x unified",
            template  = "plotly_white",
            legend    = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height    = 500,
        )
        return fig

    def plot_regime_heatmap(
        self,
        returns_df: pd.DataFrame,
        df: Optional[pd.DataFrame] = None,
        regime_col: str = "regime_label",
        title: str = "Median Monthly Return by Asset & Regime (%)",
    ):
        """Plotly heatmap: assets x regimes showing median monthly return."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        if df is None:
            df = self.classify_threshold()

        merged = returns_df.join(df[[regime_col]], how="inner").dropna(subset=[regime_col])
        pivot  = (
            merged.groupby(regime_col)[returns_df.columns.tolist()]
            .median()
            .T
        )

        regime_order = ["Low & Stable", "Rising", "High & Stable", "Falling"]
        pivot = pivot[[c for c in regime_order if c in pivot.columns]]

        fig = go.Figure(data=go.Heatmap(
            z             = pivot.values,
            x             = pivot.columns.tolist(),
            y             = pivot.index.tolist(),
            colorscale    = "RdYlGn",
            zmid          = 0,
            text          = np.round(pivot.values, 2),
            texttemplate  = "%{text:.2f}%",
            hovertemplate = "Asset: %{y}<br>Regime: %{x}<br>Median Return: %{z:.2f}%<extra></extra>",
            colorbar      = dict(title="Median<br>Monthly<br>Return %"),
        ))

        fig.update_layout(
            title    = title,
            xaxis    = dict(title="Inflation Regime"),
            yaxis    = dict(title="Asset", autorange="reversed"),
            template = "plotly_white",
            height   = max(400, len(pivot) * 35 + 100),
        )
        return fig
