"""Deterministic data-profiling, visualization, and cleaning tools for the EDA crew.

These tools do the real pandas/matplotlib work so the (small/cheap) LLM only has
to *reason* over the results, never generate fragile code. Each tool returns a
compact Markdown string the agents can read, plus writes artifacts to ./output.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from crewai.tools import tool

# crewai/pydantic replaces warnings.warn with a wrapper that rejects the
# `skip_file_prefixes` kwarg modern matplotlib passes when emitting deprecation
# warnings. Wrap it so a stray warning can't crash plotting.
import warnings as _warnings  # noqa: E402

_patched_warn = _warnings.warn


def _tolerant_warn(*args, **kwargs):  # pragma: no cover - defensive shim
    kwargs.pop("skip_file_prefixes", None)
    try:
        return _patched_warn(*args, **kwargs)
    except TypeError:
        return _patched_warn(args[0] if args else "")


_warnings.warn = _tolerant_warn

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_FILE = os.getenv(
    "DATA_FILE", r"C:\Users\ryhan\Downloads\AAZ_LifeScience_Data 1.xlsx"
)

# Sheets that get full numeric EDA (profile + plots + cleaning).
NUMERIC_SHEETS = ["Animal Master", "Lab Master", "Necropsy", "Medical Case"]
# Free-text-heavy sheet: profiled only, no plots.
TEXT_SHEETS = ["Case Journal"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
for _d in (FIG_DIR, CLEAN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
_SHEET_CACHE: dict[str, pd.DataFrame] = {}


def set_data_source(source: str) -> None:
    """Point the tools at a different workbook (local path or http(s) URL) and
    drop any sheets cached from a previous source.

    Used by the crew's before_kickoff hook so a per-run "data_file" input
    (e.g. a URL to a dataset a user uploaded elsewhere) overrides the DATA_FILE
    env default without requiring the deployed crew to have local filesystem
    access to it.
    """
    global DATA_FILE
    if source:
        DATA_FILE = source
        _SHEET_CACHE.clear()


def _all_sheets() -> dict[str, pd.DataFrame]:
    """Load and cache every sheet once (the workbook is read a single time)."""
    if not _SHEET_CACHE:
        frames = pd.read_excel(DATA_FILE, sheet_name=None, engine="openpyxl")
        _SHEET_CACHE.update(frames)
    return _SHEET_CACHE


def _safe_name(sheet: str) -> str:
    return sheet.replace(" ", "_").replace("/", "_")


def _is_yyyymmdd(series: pd.Series) -> bool:
    """True if an integer column looks like a YYYYMMDD date key (e.g. 20190506)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    s = s[(s > 19000101) & (s < 21001231)]
    if len(s) < 0.5 * series.notna().sum():
        return False
    months = (s.astype("int64") // 100) % 100
    days = s.astype("int64") % 100
    return bool(((months >= 1) & (months <= 12)).mean() > 0.95
                and ((days >= 1) & (days <= 31)).mean() > 0.95)


def _analytic_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Numeric columns worth plotting: exclude IDs, surrogate keys and date keys."""
    cols = []
    for c in df.select_dtypes(include=np.number).columns:
        name = c.lower()
        if name.endswith("id") or name.endswith("sk") or "datesk" in name:
            continue
        if _is_yyyymmdd(df[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


# --------------------------------------------------------------------------- #
# Tool 1 — Profiler
# --------------------------------------------------------------------------- #
@tool("Profile dataset")
def profile_dataset() -> str:
    """Profile every analysis sheet in the configured Excel workbook.

    Returns a Markdown report covering, per sheet: shape, per-column data types,
    missing-value counts/percentages, unique-value counts, numeric summary
    statistics, IQR-based outlier counts, duplicate-row counts, and date-like
    columns. Takes no arguments. Call this first to understand the raw data.
    """
    sheets = _all_sheets()
    targets = NUMERIC_SHEETS + TEXT_SHEETS
    out = [f"# Data Profile\n\n**Workbook:** `{Path(DATA_FILE).name}`  \n"
           f"**Sheets analyzed:** {len(targets)}\n"]

    for sheet in targets:
        if sheet not in sheets:
            out.append(f"\n## {sheet}\n_Sheet not found in workbook._\n")
            continue
        df = sheets[sheet]
        n_rows, n_cols = df.shape
        dup = int(df.duplicated().sum())
        is_text = sheet in TEXT_SHEETS
        out.append(f"\n## Sheet: {sheet}\n")
        out.append(f"- Rows: **{n_rows:,}**, Columns: **{n_cols}**")
        out.append(f"- Duplicate rows: **{dup:,}** "
                   f"({dup / n_rows * 100:.1f}%)" if n_rows else "- Duplicate rows: 0")
        if is_text:
            out.append("- _Free-text-heavy sheet: profiled only (no plots)._")

        out.append("\n| Column | Dtype | Missing | Missing % | Unique | Notes |")
        out.append("|---|---|---|---|---|---|")
        for col in df.columns:
            s = df[col]
            miss = int(s.isna().sum())
            miss_pct = miss / n_rows * 100 if n_rows else 0
            nuniq = int(s.nunique(dropna=True))
            notes = []
            if pd.api.types.is_numeric_dtype(s):
                if _is_yyyymmdd(s):
                    notes.append("date-key YYYYMMDD")
                elif col.lower().endswith(("id", "sk")):
                    notes.append("identifier")
            elif s.dtype == object:
                avg_len = s.dropna().astype(str).str.len().mean()
                if pd.notna(avg_len):
                    notes.append(f"avg len {avg_len:.0f}")
            if miss_pct >= 50:
                notes.append("HIGH missing")
            if nuniq == 1:
                notes.append("constant")
            out.append(f"| {col} | {s.dtype} | {miss:,} | {miss_pct:.1f}% | "
                       f"{nuniq:,} | {', '.join(notes)} |")

        # Numeric summary + outliers for analytic columns
        num_cols = _analytic_numeric_cols(df)
        if num_cols:
            out.append("\n**Numeric summary (analytic columns, IQR outliers):**\n")
            out.append("| Column | Mean | Std | Min | Median | Max | Outliers |")
            out.append("|---|---|---|---|---|---|---|")
            for col in num_cols:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if s.empty:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((s < lo) | (s > hi)).sum())
                out.append(f"| {col} | {s.mean():.2f} | {s.std():.2f} | "
                           f"{s.min():.2f} | {s.median():.2f} | {s.max():.2f} | "
                           f"{outliers:,} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Tool 2 — Visualizer
# --------------------------------------------------------------------------- #
@tool("Visualize dataset")
def visualize_dataset() -> str:
    """Generate and save EDA plots for the numeric analysis sheets.

    For each numeric sheet this writes PNGs to ./output/figures: a missing-value
    bar chart, a histogram grid of analytic numeric columns, and (when >= 2
    numeric columns) a correlation heatmap. Takes no arguments. Returns a Markdown
    report listing every saved figure path PLUS a text description of what each
    plot shows (distribution skew, strongest correlations) so insights can be
    written without viewing the images.
    """
    sheets = _all_sheets()
    out = ["# Visualizations\n"]
    saved: list[str] = []

    for sheet in NUMERIC_SHEETS:
        if sheet not in sheets:
            continue
        df = sheets[sheet]
        slug = _safe_name(sheet)
        out.append(f"\n## {sheet}\n")

        # --- Missingness bar chart ---
        miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        miss_pct = miss_pct[miss_pct > 0]
        if not miss_pct.empty:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(miss_pct))))
            miss_pct.plot.barh(ax=ax, color="#c0392b")
            ax.invert_yaxis()
            ax.set_xlabel("% missing")
            ax.set_title(f"{sheet} — missing values by column")
            fig.tight_layout()
            p = FIG_DIR / f"{slug}_missingness.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            saved.append(str(p))
            top_miss = ", ".join(f"{c} ({v:.0f}%)" for c, v in miss_pct.head(3).items())
            out.append(f"- ![missingness]({p.name}) Columns with most missing: {top_miss}.")
        else:
            out.append("- No missing values to chart.")

        num_cols = _analytic_numeric_cols(df)
        if not num_cols:
            out.append("- No analytic numeric columns to plot.")
            continue
        num_cols = num_cols[:12]  # keep grids readable

        # --- Histogram grid ---
        ncols = min(3, len(num_cols))
        nrows = int(np.ceil(len(num_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        skew_notes = []
        for i, col in enumerate(num_cols):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            axes[i].hist(s, bins=30, color="#2980b9", edgecolor="white")
            axes[i].set_title(col, fontsize=9)
            sk = float(s.skew()) if len(s) > 2 else 0.0
            shape = ("right-skewed" if sk > 1 else "left-skewed" if sk < -1
                     else "roughly symmetric")
            skew_notes.append(f"{col}: {shape} (skew {sk:.2f})")
        for j in range(len(num_cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{sheet} — numeric distributions")
        fig.tight_layout()
        p = FIG_DIR / f"{slug}_histograms.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        saved.append(str(p))
        out.append(f"- ![histograms]({p.name}) Distributions — " + "; ".join(skew_notes) + ".")

        # --- Correlation heatmap ---
        if len(num_cols) >= 2:
            corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
            fig, ax = plt.subplots(figsize=(1.1 * len(num_cols) + 2,
                                            1.0 * len(num_cols) + 2))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, vmin=-1, vmax=1, ax=ax, square=True,
                        cbar_kws={"shrink": 0.7})
            ax.set_title(f"{sheet} — correlation matrix")
            fig.tight_layout()
            p = FIG_DIR / f"{slug}_correlation.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            saved.append(str(p))
            # describe strongest off-diagonal pairs
            cm = corr.where(~np.eye(len(corr), dtype=bool))
            pairs = (cm.abs().unstack().dropna().sort_values(ascending=False))
            seen, top = set(), []
            for (a, b), v in pairs.items():
                key = frozenset((a, b))
                if key in seen:
                    continue
                seen.add(key)
                top.append(f"{a}~{b} = {cm.loc[a, b]:.2f}")
                if len(top) >= 3:
                    break
            out.append(f"- ![correlation]({p.name}) Strongest correlations: "
                       + "; ".join(top) + ".")

    out.append(f"\n**{len(saved)} figures saved to** `output/figures/`.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Tool 3 — Cleaner
# --------------------------------------------------------------------------- #
@tool("Clean dataset")
def clean_dataset() -> str:
    """Apply safe, reversible cleaning to each numeric sheet and save the result.

    Per sheet it: drops fully-empty columns, removes exact duplicate rows, trims
    whitespace from text columns, and converts YYYYMMDD integer date-keys to real
    dates. Cleaned data is written to ./output/cleaned/<sheet>.csv. Takes no
    arguments. Returns a Markdown changelog of every transformation applied (it
    does NOT impute missing values — it reports them for a human to decide).
    """
    sheets = _all_sheets()
    out = ["# Cleaning Report\n"]

    for sheet in NUMERIC_SHEETS:
        if sheet not in sheets:
            continue
        df = sheets[sheet].copy()
        slug = _safe_name(sheet)
        actions = []
        before_rows, before_cols = df.shape

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            actions.append(f"Dropped {len(empty_cols)} fully-empty column(s): "
                           f"{', '.join(map(str, empty_cols))}")

        dups = int(df.duplicated().sum())
        if dups:
            df = df.drop_duplicates()
            actions.append(f"Removed {dups:,} duplicate row(s)")

        obj_cols = df.select_dtypes(include="object").columns
        for c in obj_cols:
            df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        if len(obj_cols):
            actions.append(f"Trimmed whitespace on {len(obj_cols)} text column(s)")

        date_cols = [c for c in df.columns
                     if pd.api.types.is_numeric_dtype(df[c]) and _is_yyyymmdd(df[c])]
        for c in date_cols:
            df[c] = pd.to_datetime(df[c].astype("Int64").astype("string"),
                                   format="%Y%m%d", errors="coerce")
        if date_cols:
            actions.append(f"Converted {len(date_cols)} YYYYMMDD date-key(s) to dates: "
                           f"{', '.join(map(str, date_cols))}")

        still_missing = df.columns[df.isna().any()].tolist()
        path = CLEAN_DIR / f"{slug}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")

        out.append(f"\n## {sheet}")
        out.append(f"- Shape: {before_rows:,}x{before_cols} -> {df.shape[0]:,}x{df.shape[1]}")
        for a in actions:
            out.append(f"- {a}")
        if still_missing:
            out.append(f"- Columns still containing missing values "
                       f"(left for human review): {', '.join(map(str, still_missing[:10]))}"
                       + (" ..." if len(still_missing) > 10 else ""))
        out.append(f"- Saved cleaned CSV -> `output/cleaned/{slug}.csv`")

    return "\n".join(out)
