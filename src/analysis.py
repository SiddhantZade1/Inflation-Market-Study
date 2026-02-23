"""
analysis.py
-----------
Core analytical functions for the inflation-market study.

Classes
-------
    AssetClassAnalysis   : regime-conditional returns, Sharpe, real returns
    SectorHedgeAnalysis  : rolling correlations, inflation-beta, hedge effectiveness
"""

import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ASSET_NAMES = {
    "SPY_ret"  : "US Equities",
    "IEF_ret"  : "Treasuries 7-10yr",
    "TIP_ret"  : "TIPS",
    "GLD_ret"  : "Gold",
    "VNQ_ret"  : "REITs",
    "DJP_ret"  : "Commodities",
    "BIL_ret"  : "T-Bills (Cash)",
    "SHY_ret"  : "Short Treasuries",
}

SECTOR_NAMES = {
    "XLE_sec_ret"  : "Energy",
    "XLF_sec_ret"  : "Financials",
    "XLU_sec_ret"  : "Utilities",
    "XLI_sec_ret"  : "Industrials",
    "XLB_sec_ret"  : "Materials",
    "XLP_sec_ret"  : "Consumer Staples",
    "XLY_sec_ret"  : "Consumer Discretionary",
    "XLK_sec_ret"  : "Technology",
    "XLV_sec_ret"  : "Health Care",
    "XLRE_sec_ret" : "Real Estate",
}

REGIME_ORDER = ["Low & Stable", "Rising", "High & Stable", "Falling"]


class AssetClassAnalysis:
    """
    Regime-conditional performance analysis for broad asset classes.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        regime_col: str = "regime_label",
        asset_cols: Optional[list] = None,
        rf_col: str = "BIL_ret",
    ):
        self.df         = master_df.dropna(subset=[regime_col]).copy()
        self.regime_col = regime_col
        self.rf_col     = rf_col
        self.asset_cols = asset_cols or [c for c in master_df.columns if c.endswith("_ret") and not c.endswith("_sec_ret")]
        self.names      = ASSET_NAMES

    def regime_returns_summary(self) -> pd.DataFrame:
        rows = []
        for regime in REGIME_ORDER:
            sub = self.df[self.df[self.regime_col] == regime]
            if sub.empty:
                continue
            rf = sub[self.rf_col].mean() / 12 if self.rf_col in sub.columns else 0

            for col in self.asset_cols:
                if col not in sub.columns:
                    continue
                r = sub[col].dropna()
                if r.empty:
                    continue
                rows.append({
                    "regime"      : regime,
                    "asset"       : self.names.get(col, col),
                    "n_months"    : len(r),
                    "mean_ret"    : r.mean(),
                    "median_ret"  : r.median(),
                    "std_ret"     : r.std(),
                    "sharpe"      : (r.mean() - rf) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan,
                    "pct_positive": (r > 0).mean() * 100,
                    "best_month"  : r.max(),
                    "worst_month" : r.min(),
                })
        return pd.DataFrame(rows)

    def real_returns_summary(self, cpi_mom_col: str = "cpi_mom") -> pd.DataFrame:
        df    = self.df.copy()
        r_inf = df[cpi_mom_col] / 100
        rows  = []

        for col in self.asset_cols:
            if col not in df.columns:
                continue
            r_nom = df[col] / 100
            real  = ((1 + r_nom) / (1 + r_inf) - 1) * 100
            real_df = pd.DataFrame({"real": real, self.regime_col: df[self.regime_col]}).dropna()

            for regime in REGIME_ORDER:
                sub = real_df[real_df[self.regime_col] == regime]["real"]
                if sub.empty:
                    continue
                rows.append({
                    "regime"       : regime,
                    "asset"        : self.names.get(col, col),
                    "mean_real_ret": sub.mean(),
                    "pct_positive" : (sub > 0).mean() * 100,
                    "annualised"   : sub.mean() * 12,
                })
        return pd.DataFrame(rows)

    def cumulative_returns(self, regime: Optional[str] = None) -> pd.DataFrame:
        df = self.df.copy()
        if regime:
            df = df[df[self.regime_col] == regime]
        cum = {}
        for col in self.asset_cols:
            if col not in df.columns:
                continue
            r      = df[col].dropna() / 100
            series = (1 + r).cumprod() * 100
            cum[self.names.get(col, col)] = series
        return pd.DataFrame(cum)

    def plot_regime_returns(self, metric: str = "mean_ret"):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        summary = self.regime_returns_summary()
        pivot   = summary.pivot(index="asset", columns="regime", values=metric)
        pivot   = pivot[[c for c in REGIME_ORDER if c in pivot.columns]]

        regime_colors = {
            "Low & Stable" : "#2ecc71",
            "Rising"       : "#e67e22",
            "High & Stable": "#e74c3c",
            "Falling"      : "#3498db",
        }

        fig = go.Figure()
        for regime in pivot.columns:
            fig.add_trace(go.Bar(
                name         = regime,
                x            = pivot.index.tolist(),
                y            = pivot[regime].tolist(),
                marker_color = regime_colors.get(regime, "#95a5a6"),
            ))

        labels = {
            "mean_ret"    : "Mean Monthly Return (%)",
            "median_ret"  : "Median Monthly Return (%)",
            "sharpe"      : "Annualised Sharpe Ratio",
            "pct_positive": "% Positive Months",
        }

        fig.update_layout(
            title    = f"{labels.get(metric, metric)} by Asset & Inflation Regime",
            barmode  = "group",
            xaxis    = dict(title="Asset Class", tickangle=-30),
            yaxis    = dict(title=labels.get(metric, metric), zeroline=True),
            template = "plotly_white",
            legend   = dict(title="Regime", orientation="h", yanchor="bottom", y=1.02),
            height   = 500,
        )
        return fig

    def plot_real_vs_nominal(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        nom_summary  = self.regime_returns_summary()
        real_summary = self.real_returns_summary()

        nom_avg  = nom_summary.groupby("asset")["mean_ret"].mean().reset_index()
        nom_avg["annualised"] = nom_avg["mean_ret"] * 12
        real_avg = real_summary.groupby("asset")["annualised"].mean().reset_index()

        merged = nom_avg.merge(real_avg, on="asset", suffixes=("_nom", "_real"))
        merged = merged.sort_values("annualised_nom", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name         = "Nominal (annualised)",
            x            = merged["asset"],
            y            = merged["annualised_nom"],
            marker_color = "#3498db",
        ))
        fig.add_trace(go.Bar(
            name         = "Real (inflation-adjusted, annualised)",
            x            = merged["asset"],
            y            = merged["annualised_real"],
            marker_color = "#e74c3c",
        ))

        fig.update_layout(
            title    = "Nominal vs. Real Annualised Returns (Full Period)",
            barmode  = "group",
            xaxis    = dict(tickangle=-30),
            yaxis    = dict(title="Annualised Return (%)", zeroline=True),
            template = "plotly_white",
            height   = 450,
        )
        return fig

    def plot_cumulative(self):
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError("pip install plotly")

        cum    = self.cumulative_returns()
        colors = px.colors.qualitative.Set2

        fig = go.Figure()
        for i, col in enumerate(cum.columns):
            fig.add_trace(go.Scatter(
                x    = cum.index,
                y    = cum[col],
                name = col,
                line = dict(color=colors[i % len(colors)], width=2),
            ))

        fig.update_layout(
            title     = "Cumulative Returns by Asset Class (Base = 100)",
            xaxis     = dict(title="Date"),
            yaxis     = dict(title="Cumulative Return Index", type="log"),
            template  = "plotly_white",
            hovermode = "x unified",
            height    = 500,
        )
        return fig


class SectorHedgeAnalysis:
    """
    Analyzes how equity sectors respond to inflation and inflation surprises.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        regime_col: str    = "regime_label",
        inflation_col: str = "cpi_yoy",
        surprise_col: str  = "inflation_surprise",
        sector_cols: Optional[list] = None,
    ):
        self.df            = master_df.copy()
        self.regime_col    = regime_col
        self.inflation_col = inflation_col
        self.surprise_col  = surprise_col
        self.sector_cols   = sector_cols or [c for c in master_df.columns if c.endswith("_sec_ret")]
        self.names         = SECTOR_NAMES

    def rolling_correlations(self, window: int = 24) -> pd.DataFrame:
        df    = self.df[[self.inflation_col] + self.sector_cols].dropna()
        corrs = {}
        for col in self.sector_cols:
            if col not in df.columns:
                continue
            corrs[self.names.get(col, col)] = df[col].rolling(window).corr(df[self.inflation_col])
        return pd.DataFrame(corrs)

    def inflation_betas(self, by_regime: bool = False) -> pd.DataFrame:
        from scipy import stats

        df    = self.df.copy()
        x_col = "cpi_mom" if "cpi_mom" in df.columns else self.inflation_col
        rows  = []

        def _run_ols(sub, col):
            pair = sub[[x_col, col]].dropna()
            if len(pair) < 24:
                return np.nan, np.nan, np.nan
            slope, intercept, r, p, se = stats.linregress(pair[x_col], pair[col])
            return slope, r**2, p

        if by_regime:
            for regime in REGIME_ORDER:
                sub = df[df[self.regime_col] == regime]
                for col in self.sector_cols:
                    if col not in df.columns:
                        continue
                    beta, r2, pval = _run_ols(sub, col)
                    rows.append({"regime": regime, "sector": self.names.get(col, col), "beta": beta, "r_squared": r2, "p_value": pval})
        else:
            for col in self.sector_cols:
                if col not in df.columns:
                    continue
                beta, r2, pval = _run_ols(df, col)
                rows.append({"sector": self.names.get(col, col), "beta": beta, "r_squared": r2, "p_value": pval})

        return pd.DataFrame(rows)

    def hedge_effectiveness(self) -> pd.DataFrame:
        df       = self.df.copy()
        high_inf = df[df[self.regime_col].isin(["Rising", "High & Stable"])]
        rf       = df.get("BIL_ret", pd.Series(0, index=df.index)).mean() / 100
        rows     = []

        for col in self.sector_cols:
            if col not in df.columns:
                continue
            name   = self.names.get(col, col)
            r_high = high_inf[col].dropna() / 100
            sharpe = (r_high.mean() - rf) / r_high.std() * np.sqrt(12) if r_high.std() > 0 else 0
            pair   = df[[self.inflation_col, col]].dropna()
            corr   = pair[self.inflation_col].corr(pair[col]) if len(pair) > 24 else 0
            pct_pos = (r_high > 0).mean() if len(r_high) > 0 else 0

            rows.append({
                "sector"          : name,
                "sharpe_high_inf" : round(sharpe, 3),
                "corr_with_cpi"   : round(corr, 3),
                "pct_pos_high_inf": round(pct_pos * 100, 1),
                "n_months"        : len(r_high),
            })

        result = pd.DataFrame(rows)
        for metric in ["sharpe_high_inf", "corr_with_cpi", "pct_pos_high_inf"]:
            col_min = result[metric].min()
            col_max = result[metric].max()
            rng     = col_max - col_min
            result[f"{metric}_norm"] = (result[metric] - col_min) / rng if rng > 0 else 0

        norm_cols = [c for c in result.columns if c.endswith("_norm")]
        result["hedge_score"] = result[norm_cols].mean(axis=1).round(3)
        return result.sort_values("hedge_score", ascending=False)

    def plot_rolling_correlations(self, window: int = 24):
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError("pip install plotly")

        corrs  = self.rolling_correlations(window=window)
        colors = px.colors.qualitative.Plotly

        fig = go.Figure()
        for i, col in enumerate(corrs.columns):
            fig.add_trace(go.Scatter(
                x    = corrs.index,
                y    = corrs[col],
                name = col,
                line = dict(color=colors[i % len(colors)], width=1.5),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="#aaa")
        fig.update_layout(
            title     = f"Rolling {window}-Month Correlation: Sector Returns vs CPI YoY",
            xaxis     = dict(title="Date"),
            yaxis     = dict(title="Pearson Correlation", range=[-1.1, 1.1]),
            template  = "plotly_white",
            hovermode = "x unified",
            height    = 500,
        )
        return fig

    def plot_inflation_betas(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        betas  = self.inflation_betas(by_regime=False).sort_values("beta")
        colors = ["#2ecc71" if b >= 0 else "#e74c3c" for b in betas["beta"]]

        fig = go.Figure(go.Bar(
            x            = betas["beta"],
            y            = betas["sector"],
            orientation  = "h",
            marker_color = colors,
            text         = betas["beta"].round(2),
            textposition = "outside",
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="#555")
        fig.add_vline(x=1, line_dash="dot", line_color="#3498db",
                      annotation_text="β = 1 (perfect pass-through)")

        fig.update_layout(
            title    = "Inflation Beta by Sector",
            xaxis    = dict(title="Inflation Beta (β)"),
            template = "plotly_white",
            height   = 450,
        )
        return fig

    def plot_hedge_scorecard(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        scores = self.hedge_effectiveness()
        colors = [
            "#2ecc71" if s >= 0.6 else "#f39c12" if s >= 0.4 else "#e74c3c"
            for s in scores["hedge_score"]
        ]

        fig = go.Figure(go.Bar(
            x            = scores["hedge_score"],
            y            = scores["sector"],
            orientation  = "h",
            marker_color = colors,
            text         = scores["hedge_score"].round(2),
            textposition = "outside",
        ))

        fig.add_vline(x=0.5, line_dash="dash", line_color="#aaa")
        fig.update_layout(
            title    = "Sector Inflation Hedge Effectiveness Score",
            xaxis    = dict(title="Composite Hedge Score (0-1)", range=[0, 1.1]),
            template = "plotly_white",
            height   = 450,
        )
        return fig

    def plot_sector_regime_heatmap(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("pip install plotly")

        df   = self.df.copy()
        cols = [c for c in self.sector_cols if c in df.columns]
        pivot = (
            df.groupby(self.regime_col)[cols]
            .median()
            .rename(columns=self.names)
        )
        pivot = pivot.loc[[r for r in REGIME_ORDER if r in pivot.index]]

        fig = go.Figure(data=go.Heatmap(
            z             = pivot.values,
            y             = pivot.index.tolist(),
            x             = pivot.columns.tolist(),
            colorscale    = "RdYlGn",
            zmid          = 0,
            text          = pivot.values.round(2),
            texttemplate  = "%{text:.2f}%",
            hovertemplate = "Regime: %{y}<br>Sector: %{x}<br>Median Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title    = "Median Monthly Sector Return by Inflation Regime (%)",
            xaxis    = dict(title="Sector", tickangle=-30),
            yaxis    = dict(title="Inflation Regime"),
            template = "plotly_white",
            height   = 400,
        )
        return fig
