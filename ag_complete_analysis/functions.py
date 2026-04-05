"""
functions.py – Utility module for dental-implant survival analysis
=================================================================
Andersen–Gill Cox models for overall, early (≤365 d), and late (>365 d) failure.

Journal target : Journal of Clinical Periodontology (JCP)
Figure specs   : 600 dpi, Arial 9 pt, colorblind-safe Wong 2011 palette,
                 no top/right spines.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import transforms
from matplotlib.ticker import PercentFormatter
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2_contingency

# Optional – rpy2 for R-based Cox modelling
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

# Optional – lifelines for KM / Python-side Cox
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import multivariate_logrank_test

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


# =====================================================================
#  CONSTANTS
# =====================================================================
CENSOR_DATE_DEFAULT = "2025-08-24"
COHORT_CUTOFF_DATE = "2023-07-01"
EARLY_THRESHOLD_DAYS = 365          # ≤365 d = early failure
GENDER_MAP = {1: "Male", 0: "Female", "1": "Male", "0": "Female"}

# Categorisation bins
DIAMETER_BINS = [-np.inf, 3.7, 4.2, np.inf]
DIAMETER_LABELS = ["Narrow (<3.7)", "Medium (3.7–4.2)", "Wide (>4.2)"]
LENGTH_BINS = [-np.inf, 10, 11.5, np.inf]
LENGTH_LABELS = ["Short (<10)", "Medium (10–11.5)", "Long (>11.5)"]
AGE_BINS = [-np.inf, 40, 60, np.inf]
AGE_LABELS = ["<40", "40–60", "60+"]

# JCP-style figure defaults (Wong 2011 colorblind-safe palette)
WONG = {
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "black":   "#000000",
}
FIG_DEFAULTS = dict(
    dpi=600,
    font_family="Arial",
    font_size=9,
    sig_color=WONG["vermil"],
    ns_color=WONG["blue"],
    null_color="#999999",
    bg_alt="#F0F4FA",
    title_color="#214E72",
    text_color="#243746",
    grid_color="#D7E0EA",
    border_color="#D5DEE8",
    header_bg="#2C5F8A",
    header_fg="#FFFFFF",
    row_alt="#F8FBFD",
)

# Friendly label map for R coefficient names → publication names
LABEL_MAP: Dict[str, str] = {
    "implant_index_cat2":          "Implant sequence: 2nd",
    "implant_index_cat3+":         "Implant sequence: 3rd or later",
    "gender1":                     "Sex: Male",
    "gender1.0":                   "Sex: Male",
    "gendermale":                  "Sex: Male",
    "smoker":                      "Smoking (yes)",
    "has_diabetes":                "Diabetes (yes)",
    "has_hypertension":            "Hypertension (yes)",
    "takes_biphos":                "Anti-resorptive drug use (yes)",
    "Penicillin_Allergy":          "Penicillin allergy (yes)",
    "length_catShort (<10)":       "Implant length: Short (<10 mm)",
    "length_catLong (>11.5)":      "Implant length: Long (>11.5 mm)",
    "diameter_catNarrow (<3.7)":   "Implant diameter: Narrow (<3.7 mm)",
    "diameter_catWide (>4.2)":     "Implant diameter: Wide (>4.2 mm)",
    "jawmandible":                 "Jaw: Mandible",
    "jawmaxilla":                  "Jaw: Maxilla",
    "regionmolar":                 "Region: Molar",
    "regionpremolar":              "Region: Premolar",
    "regionanterior":              "Region: Anterior",
    "age_bin<40":                  "Age group: <40 yr",
    "age_bin60+":                  "Age group: ≥60 yr",
}

VARIABLE_LABELS: Dict[str, str] = {
    "jaw": "Jaw",
    "region": "Region",
    "diameter_cat": "Implant diameter",
    "length_cat": "Implant length",
    "age_bin": "Age group",
    "gender_bin": "Sex",
    "implant_num_surv": "Implant sequence",
    "implant_num_surv_lbl": "Implant sequence",
    "smoker": "Smoking",
    "has_diabetes": "Diabetes",
    "has_hypertension": "Hypertension",
    "takes_biphos": "Anti-resorptive drug use",
    "Penicillin_Allergy": "Penicillin allergy",
    "has_bonegraft_beforeimplant": "Bone graft before implant",
    "has_rama_onimplantday": "Rama on implant day",
    "has_mahash_onimplantday": "Mahash on implant day",
    "has_resm_onimplantday": "Resm on implant day",
    "has_resp_onorbeforeimplant": "Resp on or before implant",
}

VARIABLE_GROUP_LABELS: Dict[str, str] = {
    "implant_num_surv": "Implant Sequence",
    "implant_num_surv_lbl": "Implant Sequence",
    "implant_index_cat": "Implant Sequence",
    "gender": "Demographics",
    "gender_bin": "Demographics",
    "age_bin": "Demographics",
    "smoker": "Medical History",
    "has_diabetes": "Medical History",
    "has_hypertension": "Medical History",
    "takes_biphos": "Medical History",
    "Penicillin_Allergy": "Medical History",
    "has_bonegraft_beforeimplant": "Medical History",
    "has_rama_onimplantday": "Medical History",
    "has_mahash_onimplantday": "Medical History",
    "has_resm_onimplantday": "Medical History",
    "has_resp_onorbeforeimplant": "Medical History",
    "length_cat": "Implant Geometry",
    "diameter_cat": "Implant Geometry",
    "jaw": "Site Anatomy",
    "region": "Site Anatomy",
}

FOREST_GROUP_ORDER = [
    "Implant Sequence",
    "Demographics",
    "Medical History",
    "Implant Geometry",
    "Site Anatomy",
    "Other",
]

FOREST_TERM_ORDER = {
    "Implant sequence: 2nd": 1,
    "Implant sequence: 3rd or later": 2,
    "Sex: Male": 10,
    "Age group: <40 yr": 11,
    "Age group: ≥60 yr": 12,
    "Smoking (yes)": 20,
    "Diabetes (yes)": 21,
    "Hypertension (yes)": 22,
    "Anti-resorptive drug use (yes)": 23,
    "Penicillin allergy (yes)": 24,
    "Implant length: Short (<10 mm)": 30,
    "Implant length: Long (>11.5 mm)": 31,
    "Implant diameter: Narrow (<3.7 mm)": 32,
    "Implant diameter: Wide (>4.2 mm)": 33,
    "Jaw: Mandible": 40,
    "Jaw: Maxilla": 41,
    "Region: Anterior": 42,
    "Region: Premolar": 43,
    "Region: Molar": 44,
}


# =====================================================================
#  I. HELPER FUNCTIONS
# =====================================================================
def to_binary(x):
    """Robustly convert arbitrary values to 0/1 integer."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer)):
        try:
            return int(float(x) != 0)
        except Exception:
            return np.nan
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y", "t", "כן"}:
        return 1
    if s in {"0", "0.0", "false", "no", "n", "f", "לא"}:
        return 0
    return np.nan


def parse_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[col], errors="coerce", dayfirst=True)


def normalize_text(x) -> str:
    """Replace en-dash / em-dash with ASCII hyphen (avoids R encoding issues)."""
    if pd.isna(x):
        return pd.NA
    return str(x).replace("–", "-").replace("—", "-").strip()


def fmt_p(p: float) -> str:
    """Format p-value for publication."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "P<0.001"
    return f"{p:.3f}"


def _is_p_value_column(col: object) -> bool:
    low = str(col).strip().lower()
    return "p-value" in low or low.endswith(" p") or low.startswith("p ") or low == "p"


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_hr_ci(hr: float, ci_lower: float, ci_upper: float) -> str:
    return f"{hr:.2f} ({ci_lower:.2f}–{ci_upper:.2f})"


def _format_count(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def _publication_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    replacements = {
        "3+": "3rd or later",
        "60+": "≥60",
        "ARD use": "Anti-resorptive drug use",
        "ARD use (yes)": "Anti-resorptive drug use (yes)",
        "Implant sequence: 3rd or 4th+": "Implant sequence: 3rd or later",
        "Reference": "Ref.",
    }
    return replacements.get(text, text)


def _is_univariable_header_row(row: pd.Series) -> bool:
    return (
        str(row.get("Variable", "")).strip() != ""
        and str(row.get("Level", "")).strip() == ""
        and pd.isna(row.get("N"))
    )


def _clean_label(raw: str) -> str:
    """Map R coefficient name → friendly label."""
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    low = {k.lower(): v for k, v in LABEL_MAP.items()}
    if raw.lower() in low:
        return low[raw.lower()]
    # Fallback cosmetic cleanup
    return (
        raw.replace("implant_index_cat", "Implant sequence: ")
        .replace("_", " ")
        .strip()
    )


def _display_variable_name(name: str) -> str:
    return VARIABLE_LABELS.get(name, name.replace("_", " ").strip())


def _variable_group_name(name: str) -> str:
    return VARIABLE_GROUP_LABELS.get(name, "Other")


def _ordered_unique(values: pd.Series) -> List[object]:
    seen: List[object] = []
    for value in values.tolist():
        if pd.isna(value):
            continue
        if value not in seen:
            seen.append(value)
    return seen


def _reference_order_for_variable(col: str, values: pd.Series) -> List[object]:
    preferred = {
        "age_bin": ["40–60", "<40", "60+"],
        "length_cat": ["Medium (10–11.5)", "Short (<10)", "Long (>11.5)"],
        "diameter_cat": ["Medium (3.7–4.2)", "Narrow (<3.7)", "Wide (>4.2)"],
        "jaw": ["Mandible", "Maxilla"],
        "region": ["Molar", "Premolar", "Anterior"],
        "gender_bin": ["Female", "Male"],
        "implant_num_surv": ["1", "2", "3+"],
        "implant_num_surv_lbl": ["1", "2", "3+"],
        "smoker": ["No", "Yes"],
        "has_diabetes": ["No", "Yes"],
        "has_hypertension": ["No", "Yes"],
        "takes_biphos": ["No", "Yes"],
        "Penicillin_Allergy": ["No", "Yes"],
        "has_bonegraft_beforeimplant": ["No", "Yes"],
        "has_rama_onimplantday": ["No", "Yes"],
        "has_mahash_onimplantday": ["No", "Yes"],
        "has_resm_onimplantday": ["No", "Yes"],
        "has_resp_onorbeforeimplant": ["No", "Yes"],
    }
    observed = _ordered_unique(values)
    preferred_order = preferred.get(col)
    if preferred_order is None:
        try:
            return sorted(observed)
        except TypeError:
            return observed
    ordered = [level for level in preferred_order if level in observed]
    ordered.extend(level for level in observed if level not in ordered)
    return ordered


def _coerce_univariable_levels(series: pd.Series, col: str) -> pd.Series:
    out = series.copy()
    if col in {
        "gender_bin", "implant_num_surv", "smoker", "has_diabetes", "has_hypertension",
        "takes_biphos", "Penicillin_Allergy", "has_bonegraft_beforeimplant",
        "has_rama_onimplantday", "has_mahash_onimplantday", "has_resm_onimplantday",
        "has_resp_onorbeforeimplant",
    }:
        out = pd.to_numeric(out, errors="coerce")
    if col == "gender_bin":
        out = out.map({0: "Female", 1: "Male", 0.0: "Female", 1.0: "Male"})
    elif col == "implant_num_surv":
        out = out.map({1: "1", 2: "2", 3: "3+", 1.0: "1", 2.0: "2", 3.0: "3+"})
    elif col in {
        "smoker", "has_diabetes", "has_hypertension", "takes_biphos", "Penicillin_Allergy",
        "has_bonegraft_beforeimplant", "has_rama_onimplantday", "has_mahash_onimplantday",
        "has_resm_onimplantday", "has_resp_onorbeforeimplant",
    }:
        out = out.map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"})
    else:
        out = out.astype("string")
    return out.astype("string").str.strip()


def _forest_group(label: str) -> str:
    if label.startswith("Implant sequence"):
        return "Implant Sequence"
    if label.startswith("Sex") or label.startswith("Age group"):
        return "Demographics"
    if label.startswith(("Smoking", "Diabetes", "Hypertension", "ARD use", "Anti-resorptive drug use", "Penicillin allergy")):
        return "Medical History"
    if label.startswith(("Implant length", "Implant diameter")):
        return "Implant Geometry"
    if label.startswith(("Jaw", "Region")):
        return "Site Anatomy"
    return "Other"


def _forest_order(label: str) -> Tuple[int, int, str]:
    group = _forest_group(label)
    return (
        FOREST_GROUP_ORDER.index(group),
        FOREST_TERM_ORDER.get(label, 999),
        label,
    )


def model_diagnostics_table(
    model_label: str,
    n_obs: int,
    n_events: int,
    concordance: float,
    ph_global_p: float,
) -> pd.DataFrame:
    """Compact model-level diagnostics table for notebook display/export."""
    return pd.DataFrame([
        {
            "Model": model_label,
            "N": n_obs,
            "Events": n_events,
            "C-index": concordance,
            "PH test p-value": fmt_p(ph_global_p),
        }
    ])


def _default_numeric_formats(df: pd.DataFrame) -> Dict[str, object]:
    """Infer readable numeric formats for notebook display tables."""
    formats: Dict[str, object] = {}
    for col in df.columns:
        series = df[col]
        if not is_numeric_dtype(series):
            continue
        low = str(col).strip().lower()
        if low in {"n", "events"}:
            formats[col] = "{:,.0f}"
        elif _is_p_value_column(col):
            formats[col] = fmt_p
        elif "c-index" in low or "concordance" in low:
            formats[col] = "{:.3f}"
        elif pd.api.types.is_integer_dtype(series):
            formats[col] = "{:,.0f}"
        else:
            formats[col] = "{:.2f}"
    return formats


def _resolve_table_formats(
    df: pd.DataFrame,
    formats: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    resolved: Dict[str, object] = dict(formats or _default_numeric_formats(df))
    for col in df.columns:
        if is_numeric_dtype(df[col]) and _is_p_value_column(col):
            resolved[col] = fmt_p
    return resolved


def style_table(
    df: pd.DataFrame,
    caption: Optional[str] = None,
    formats: Optional[Dict[str, object]] = None,
    hide_index: bool = True,
) -> pd.io.formats.style.Styler:
    """Return a consistent notebook-friendly styled table."""
    styler = df.style.format(_resolve_table_formats(df, formats), na_rep="")
    styler = styler.set_table_styles([
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("text-align", "left"),
                ("font-weight", "bold"),
                ("font-size", "13px"),
                ("color", FIG_DEFAULTS["title_color"]),
                ("padding", "0 0 8px 0"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("background-color", FIG_DEFAULTS["header_bg"]),
                ("color", FIG_DEFAULTS["header_fg"]),
                ("font-weight", "bold"),
                ("text-align", "left"),
                ("border", f"1px solid {FIG_DEFAULTS['border_color']}"),
                ("padding", "6px 10px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("border", f"1px solid {FIG_DEFAULTS['border_color']}"),
                ("padding", "5px 10px"),
                ("color", FIG_DEFAULTS["text_color"]),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even)",
            "props": [("background-color", FIG_DEFAULTS["row_alt"])],
        },
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
                ("font-family", f"{FIG_DEFAULTS['font_family']}, sans-serif"),
                ("font-size", "12px"),
            ],
        },
    ])
    styler = styler.set_properties(**{"text-align": "left"})
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        styler = styler.set_properties(subset=numeric_cols, **{"text-align": "right"})
    if caption:
        styler = styler.set_caption(caption)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler


def style_result_table(tbl: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styled display for model result tables."""
    styler = style_table(tbl)
    hr_cols = [col for col in ["HR (95% CI)", "Adjusted HR (95% CI)"] if col in tbl.columns]
    if hr_cols:
        styler = styler.set_properties(subset=hr_cols, **{"text-align": "right"})
    if "p-value" in tbl.columns:
        styler = styler.set_properties(subset=["p-value"], **{"text-align": "right"})
    if "" in tbl.columns:
        styler = styler.apply(
            lambda s: [
                "color: #C0392B; font-weight: bold; text-align: center" if str(v).strip() else "text-align: center"
                for v in s
            ],
            subset=[""],
        )
    return styler


def style_logrank_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styled display for log-rank summary tables."""
    formats = {
        "N": "{:,.0f}",
    }
    styler = style_table(df, formats=formats)
    if "Significant" in df.columns:
        styler = styler.apply(
            lambda s: [
                "color: #C0392B; font-weight: bold; text-align: center" if bool(v) else "text-align: center"
                for v in s
            ],
            subset=["Significant"],
        )
    return styler


def style_univariable_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styled display for consolidated univariable log-rank/Cox tables."""
    styler = style_table(df)
    if not df.empty:
        header_mask = df.apply(_is_univariable_header_row, axis=1)
        styler = styler.apply(
            lambda row: [
                (
                    f"background-color: {FIG_DEFAULTS['bg_alt']}; "
                    f"font-weight: bold; color: {FIG_DEFAULTS['title_color']}; "
                    f"border-top: 2px solid {FIG_DEFAULTS['border_color']}; "
                    "text-align: left"
                )
                if header_mask.loc[row.name]
                else ""
                for _ in row
            ],
            axis=1,
        )
    numeric_like_cols = [
        col for col in [
            "N", "Events", "Success rate (%)",
            "Cox HR (95% CI)", "Log-rank p-value", "Cox p-value",
        ] if col in df.columns
    ]
    if numeric_like_cols:
        styler = styler.set_properties(subset=numeric_like_cols, **{"text-align": "right"})
    if "Cox HR (95% CI)" in df.columns:
        styler = styler.apply(
            lambda s: [
                "font-style: italic; color: #556270" if str(v) in {"Reference", "Ref."} else ""
                for v in s
            ],
            subset=["Cox HR (95% CI)"],
        )
    return styler


def make_univariable_publication_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return an export-friendly manuscript version of the univariable summary table."""
    if df.empty:
        return df.copy()

    out = df.copy()
    header_mask = out.apply(_is_univariable_header_row, axis=1)

    out["Variable"] = out["Variable"].apply(_publication_text)
    out["Level"] = out["Level"].apply(_publication_text)
    out["Cox HR (95% CI)"] = out["Cox HR (95% CI)"].apply(_publication_text)

    for col in ["Log-rank p-value", "Cox p-value"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda value: fmt_p(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value).strip())
            out.loc[header_mask, col] = ""

    out["N"] = out["N"].apply(_format_count)
    out["Events"] = out["Events"].apply(_format_count)
    out["Success rate (%)"] = out["Success rate (%)"].apply(
        lambda value: "" if pd.isna(value) else f"{float(value):.1f}"
    )

    out.loc[header_mask, ["N", "Events", "Success rate (%)", "Cox HR (95% CI)"]] = ""
    return out


def style_univariable_publication_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styled display for the manuscript-ready univariable table."""
    return style_univariable_table(df)


def style_comparison_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styled display for comparison/summary tables."""
    return style_table(df)


# =====================================================================
#  II. DATA LOADING & PREPROCESSING
# =====================================================================
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, sep="," if ext == ".csv" else "\t")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl" if ext == ".xlsx" else None)
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported extension: {ext}")


def fdi_to_jaw_region(fdi: pd.Series):
    """Decode FDI tooth numbering → jaw (Maxilla/Mandible) and region."""
    fdi_int = pd.to_numeric(fdi, errors="coerce")
    tens = (fdi_int // 10).astype("Int64")
    ones = (fdi_int % 10).astype("Int64")

    jaw = pd.Series(pd.NA, index=fdi.index, dtype="object")
    jaw[(tens == 1) | (tens == 2)] = "Maxilla"
    jaw[(tens == 3) | (tens == 4)] = "Mandible"

    region = pd.Series(pd.NA, index=fdi.index, dtype="object")
    region[ones.isin([1, 2, 3])] = "Anterior"
    region[ones.isin([4, 5])] = "Premolar"
    region[ones.isin([6, 7, 8])] = "Molar"
    return jaw, region


def categorize_diameter(x):
    return pd.cut(
        pd.to_numeric(x, errors="coerce"),
        bins=DIAMETER_BINS,
        labels=DIAMETER_LABELS,
        include_lowest=True,
        right=True,
    )


def categorize_length(x):
    return pd.cut(
        pd.to_numeric(x, errors="coerce"),
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
        include_lowest=True,
        right=True,
    )


def categorize_age(x):
    return pd.cut(
        pd.to_numeric(x, errors="coerce"),
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=False,
    )


def min_date_from_csv(series: pd.Series) -> pd.Series:
    """Extract the earliest date from a comma-separated date string."""
    s = series.astype("string").fillna(pd.NA)

    def _min_dt(x):
        if pd.isna(x):
            return pd.NaT
        parts = re.split(r"[,;|]", str(x))
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return pd.NaT
        dts = pd.to_datetime(parts, errors="coerce")
        dts = pd.Series(dts).dropna().drop_duplicates()
        return dts.min() if not dts.empty else pd.NaT

    return s.apply(_min_dt)


def add_rehabilitation_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Classify implants as fixed / denture-inferred / unknown rehabilitation."""
    out = df.copy()
    out["duedate"] = pd.to_datetime(out["duedate"], errors="coerce", dayfirst=True)
    out["failure_date"] = pd.to_datetime(out["failure_date"], errors="coerce", dayfirst=True)

    out["has_fixed_rehab"] = (
        out["has_rest"].apply(to_binary).fillna(0).astype(int)
        if "has_rest" in out.columns
        else 0
    )

    out["fixed_rehab_date"] = pd.NaT
    if "rest_dates_csv" in out.columns:
        out["fixed_rehab_date"] = min_date_from_csv(out["rest_dates_csv"])
    if "selected_rest_date" in out.columns:
        sel = pd.to_datetime(out["selected_rest_date"], errors="coerce", dayfirst=True)
        out["fixed_rehab_date"] = out["fixed_rehab_date"].fillna(sel)

    delta_rehab = (out["fixed_rehab_date"] - out["duedate"]).dt.days
    delta_fail = (out["failure_date"] - out["duedate"]).dt.days
    has_fixed = out["has_fixed_rehab"] == 1
    failed = out["failure_date"].notna()

    denture_failure_mask = (~has_fixed) & failed & (delta_fail >= 120)
    denture_success_mask = (~has_fixed) & (~failed)

    out["rehabilitation_type"] = np.select(
        [has_fixed, denture_success_mask, denture_failure_mask],
        ["fixed", "denture_inferred", "denture_inferred"],
        default="unknown",
    )

    out["fixed_loading_type"] = np.select(
        [
            has_fixed & pd.to_numeric(delta_rehab, errors="coerce").between(-60, 7, inclusive="both"),
            has_fixed & (pd.to_numeric(delta_rehab, errors="coerce") > 7),
        ],
        ["immediate", "delayed"],
        default=np.where(has_fixed, "unknown", pd.NA),
    )

    out["days_to_fixed_rehab"] = delta_rehab
    return out


def preprocess(
    df: pd.DataFrame,
    censor_date: str = CENSOR_DATE_DEFAULT,
    cohort_cutoff: str = COHORT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Full preprocessing pipeline: filtering, feature engineering, time-to-event."""
    # Filter implants only
    if "is_implant" in df.columns:
        df = df[df["is_implant"].apply(to_binary) == 1].copy()

    df["duedate"] = parse_date_col(df, "duedate")
    df["failure_date"] = parse_date_col(df, "failure_date")

    # Site ID
    pid = df["patient_id"].astype("string") if "patient_id" in df.columns else pd.Series("NA", index=df.index)
    tno = df["tooth_number"].astype("string") if "tooth_number" in df.columns else pd.Series("NA", index=df.index)
    df["site_id"] = pid.fillna("NA") + "_" + tno.fillna("NA")

    # Jaw / region from FDI
    tooth_col = df["tooth_number"] if "tooth_number" in df.columns else pd.Series(pd.NA, index=df.index)
    jaw, region = fdi_to_jaw_region(tooth_col)
    df["jaw"] = jaw
    df["region"] = region

    # Categories
    df["diameter_cat"] = categorize_diameter(df.get("diameter"))
    df["length_cat"] = categorize_length(df.get("length"))

    if "ageatduedate_years_float" in df.columns:
        df["age_years"] = pd.to_numeric(df["ageatduedate_years_float"], errors="coerce")
    else:
        bd = parse_date_col(df, "birth_date")
        df["age_years"] = (df["duedate"] - bd).dt.days / 365.25
    df["age_bin"] = categorize_age(df["age_years"])

    # Gender
    if "gender" in df.columns:
        def norm_gender(g):
            if pd.isna(g):
                return np.nan
            s = str(g).strip().lower()
            if s in {"f", "female", "נקבה", "נ", "b"}:
                return 0
            if s in {"m", "male", "זכר", "ז"}:
                return 1
            return np.nan
        df["gender_bin"] = df["gender"].apply(norm_gender)

    # Binary columns
    for col in [
        "is_failure", "smoker", "has_diabetes", "has_hypertension",
        "takes_biphos", "Penicillin_Allergy", "has_allergy",
        "has_heart_condition", "Has_Kidney_Disease", "Has_Osteoporosis",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(to_binary)

    # Implant index bucketing (≤4)
    idx = pd.to_numeric(df.get("implant_index"), errors="coerce")
    df = df.loc[idx.fillna(0) <= 4].copy()
    df["implant_num_eda"] = idx.clip(upper=4).astype("Int64")
    df["implant_num_surv"] = idx.mask(idx >= 3, 3).astype("Int64")
    df["implant_num_surv_lbl"] = df["implant_num_surv"].map({1: "1", 2: "2", 3: "3+"})

    # Event and time-to-event
    df["is_failure"] = df["failure_date"].notna().astype(int)
    censor_ts = pd.to_datetime(censor_date)
    df["days_to_failure"] = np.where(
        df["is_failure"] == 1,
        (df["failure_date"] - df["duedate"]).dt.days,
        (censor_ts - df["duedate"]).dt.days,
    )
    df.loc[df["days_to_failure"] < 0, "days_to_failure"] = np.nan

    # Rehabilitation
    df = add_rehabilitation_classification(df)

    # Calendar period
    df["year"] = df["duedate"].dt.year

    # Cohort cutoff
    df = df[df["duedate"] <= pd.to_datetime(cohort_cutoff)].copy()

    # Drop rows missing key columns
    required = ["site_id", "duedate", "age_years", "is_failure", "days_to_failure",
                 "implant_num_surv", "length", "jaw", "gender_bin"]
    existing = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing).copy()

    # Convenience aliases
    df["gender"] = df["gender_bin"]

    return df


# =====================================================================
#  III. COX TIME VARIABLES  (used by the AG + early/late sub-models)
# =====================================================================
def prepare_cox_time(
    df: pd.DataFrame,
    study_end: str = CENSOR_DATE_DEFAULT,
) -> pd.DataFrame:
    """Add failure_event, time (days), and cohort_last_date columns."""
    out = df.copy()
    out["duedate"] = pd.to_datetime(out["duedate"], errors="coerce")
    out["failure_date"] = pd.to_datetime(out["failure_date"], errors="coerce")
    out["failure_event"] = out["is_failure"].astype(int)

    se = pd.to_datetime(study_end)
    out["time"] = np.where(
        out["failure_event"].eq(1) & out["failure_date"].notna(),
        (out["failure_date"] - out["duedate"]).dt.days,
        (se - out["duedate"]).dt.days,
    ).astype(float)
    out.loc[out["time"] <= 0, "time"] = np.nan

    # Follow-up end date (for AG tstart/tstop)
    cohort_last = pd.to_datetime("2023-12-31")
    out["end_date"] = out["failure_date"].fillna(cohort_last)

    return out


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean columns for modelling; bucket implant_index."""
    cols = [
        "time", "failure_event", "patient_id", "implant_index", "gender",
        "smoker", "has_diabetes", "has_hypertension", "takes_biphos",
        "Penicillin_Allergy", "length_cat", "diameter_cat",
        "tooth_number", "duedate", "failure_date", "end_date",
    ]
    cols = [c for c in cols if c in df.columns]
    mf = df[cols].copy()

    # Age bin
    age_levels = ["40–60", "<40", "60+"]
    mf["age_bin"] = pd.Categorical(df["age_bin"], categories=age_levels, ordered=True)

    # Implant index bucket
    def _bucket(x):
        try:
            xi = int(x)
            return str(xi) if xi <= 2 else "3+"
        except Exception:
            return pd.NA
    mf["implant_index_cat"] = mf["implant_index"].apply(_bucket)

    # Coerce categoricals
    for c in ["implant_index_cat", "gender"]:
        mf[c] = mf[c].astype("string").fillna("unknown")
    for c in ["smoker", "has_diabetes", "has_hypertension", "takes_biphos",
              "Penicillin_Allergy", "failure_event"]:
        if c in mf.columns:
            mf[c] = pd.to_numeric(mf[c], errors="coerce").fillna(0).astype(int)

    mf = mf.dropna(subset=["time", "failure_event", "patient_id"])

    # FDI -> jaw
    def _parse_fdi(x):
        try:
            s = str(int(x))
        except Exception:
            s = str(x).strip()
        if not s.isdigit() or len(s) != 2:
            return (np.nan, np.nan)
        return int(s[0]), int(s[1])

    def _jaw(q):
        return "maxilla" if q in (1, 2, 5, 6) else ("mandible" if q in (3, 4, 7, 8) else "unknown")

    qp = mf["tooth_number"].apply(_parse_fdi)
    mf["jaw"] = [_jaw(q) for q, p in qp]

    # Categorical reference levels
    mf["length_cat"] = mf["length_cat"].astype("string").str.strip()
    mf["diameter_cat"] = mf["diameter_cat"].astype("string").str.strip()
    length_levels = ["Medium (10–11.5)", "Short (<10)", "Long (>11.5)"]
    diam_levels = ["Medium (3.7–4.2)", "Narrow (<3.7)", "Wide (>4.2)"]
    mf["length_cat"] = pd.Categorical(mf["length_cat"], categories=length_levels, ordered=True)
    mf["diameter_cat"] = pd.Categorical(mf["diameter_cat"], categories=diam_levels, ordered=True)

    return mf


# =====================================================================
#  IV. EARLY / LATE FAILURE CLASSIFICATION
# =====================================================================
def classify_failure_type(
    df: pd.DataFrame,
    threshold_days: int = EARLY_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Label each implant as Early (≤threshold), Late (>threshold), or Censored.
    Uses `failure_time_days` or computes from duedate/failure_date.
    """
    out = df.copy()
    event_col = "is_failure" if "is_failure" in out.columns else "failure_event"
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)

    if "failure_time_days" not in out.columns:
        if "days_to_failure" in out.columns:
            out["failure_time_days"] = pd.to_numeric(out["days_to_failure"], errors="coerce")
        elif {"duedate", "failure_date"}.issubset(out.columns):
            out["duedate"] = pd.to_datetime(out["duedate"], errors="coerce")
            out["failure_date"] = pd.to_datetime(out["failure_date"], errors="coerce")
            out["failure_time_days"] = np.where(
                out[event_col] == 1,
                (out["failure_date"] - out["duedate"]).dt.days,
                np.nan,
            )
    out["failure_time_days"] = pd.to_numeric(out.get("failure_time_days"), errors="coerce").clip(lower=0)

    is_fail = out[event_col] == 1
    out["failure_type"] = "Censored"
    out.loc[is_fail & out["failure_time_days"].notna() & (out["failure_time_days"] <= threshold_days), "failure_type"] = "Early"
    out.loc[is_fail & out["failure_time_days"].notna() & (out["failure_time_days"] > threshold_days), "failure_type"] = "Late"
    return out


# =====================================================================
#  V. AG DATA PREPARATION (Python → R)
# =====================================================================
def prepare_ag_data(mf: pd.DataFrame) -> pd.DataFrame:
    """
    Build Andersen–Gill counting-process dataset:
    each implant = one row with (tstart, tstop, event) relative to patient t0.
    """
    dat = mf.copy()
    dat["implant_date"] = pd.to_datetime(dat["duedate"], errors="coerce")
    dat["failure_date"] = pd.to_datetime(dat["failure_date"], errors="coerce")
    dat["last_followup_date"] = pd.to_datetime(dat["end_date"], errors="coerce")

    # t0 = patient's first implant date
    t0 = dat.groupby("patient_id")["implant_date"].min()
    dat = dat.join(t0.rename("t0"), on="patient_id")

    # stop_date
    dat["stop_date"] = np.where(
        pd.to_numeric(dat["failure_event"], errors="coerce").fillna(0).astype(int) == 1,
        dat["failure_date"],
        dat["last_followup_date"],
    )
    dat["stop_date"] = pd.to_datetime(dat["stop_date"], errors="coerce")

    dat["tstart"] = (dat["implant_date"] - dat["t0"]).dt.days.astype(float)
    dat["tstop"] = (dat["stop_date"] - dat["t0"]).dt.days.astype(float)
    dat["event"] = pd.to_numeric(dat["failure_event"], errors="coerce").fillna(0).astype(int)

    # Clean
    dat = dat.dropna(subset=["tstart", "tstop", "event", "patient_id"]).copy()
    dat = dat[(dat["tstop"] > dat["tstart"]) & (dat["tstart"] >= 0)].copy()
    assert set(dat["event"].unique()) <= {0, 1}, "event must be 0/1"
    return dat


def _prepare_dat_r(dat: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for transfer to R: normalize strings, coerce types."""
    dat_r = dat.copy()
    cat_cols = ["implant_index_cat", "gender", "length_cat", "diameter_cat", "jaw", "age_bin"]
    for c in cat_cols:
        if c in dat_r.columns:
            dat_r[c] = dat_r[c].astype("string").apply(normalize_text)
    bin_cols = ["smoker", "has_diabetes", "has_hypertension", "takes_biphos", "Penicillin_Allergy"]
    for c in bin_cols:
        if c in dat_r.columns:
            dat_r[c] = pd.to_numeric(dat_r[c], errors="coerce").fillna(0).astype(int)
    dat_r["patient_id"] = dat_r["patient_id"].astype("string")
    return dat_r


# =====================================================================
#  VI. R-BASED AG COX MODEL
# =====================================================================
_R_AG_TEMPLATE = r"""
library(survival)

cat("N rows (raw):", nrow(dat), "\n")

# --- factors + reference levels ---
dat$implant_index_cat <- factor(dat$implant_index_cat)
if ("1" %in% levels(dat$implant_index_cat))
    dat$implant_index_cat <- relevel(dat$implant_index_cat, ref = "1")

dat$gender       <- factor(dat$gender)
dat$jaw          <- factor(dat$jaw)
dat$age_bin      <- factor(dat$age_bin)
if ("40-60" %in% levels(dat$age_bin))
    dat$age_bin <- relevel(dat$age_bin, ref = "40-60")

dat$length_cat   <- factor(dat$length_cat)
if ("Medium (10-11.5)" %in% levels(dat$length_cat))
    dat$length_cat <- relevel(dat$length_cat, ref = "Medium (10-11.5)")

dat$diameter_cat <- factor(dat$diameter_cat)
if ("Medium (3.7-4.2)" %in% levels(dat$diameter_cat))
    dat$diameter_cat <- relevel(dat$diameter_cat, ref = "Medium (3.7-4.2)")

# --- identify & drop single-level factors after NA removal ---
base_terms <- c(
    "implant_index_cat","gender","smoker","has_diabetes","has_hypertension",
    "takes_biphos","Penicillin_Allergy","length_cat","diameter_cat",
    "jaw","age_bin"
)

test_formula <- as.formula(paste(
    "Surv(tstart, tstop, event) ~",
    paste(base_terms, collapse=" + "),
    "+ cluster(patient_id)"
))

mf2 <- model.frame(test_formula, data=dat, na.action=na.omit)
cat("Rows after na.omit:", nrow(mf2), "\n")

fac_vars <- names(mf2)[sapply(mf2, is.factor)]
bad <- fac_vars[sapply(mf2[fac_vars], function(x) nlevels(droplevels(x)) < 2)]
if (length(bad) > 0) cat("Dropping single-level factors:", paste(bad, collapse=", "), "\n")

keep_terms <- base_terms[!base_terms %in% bad]

final_formula <- as.formula(paste(
    "Surv(tstart, tstop, event) ~",
    paste(keep_terms, collapse=" + "),
    "+ cluster(patient_id)"
))

fit_ag <- coxph(final_formula, data=dat, ties="efron", na.action=na.omit, x=TRUE)
s <- summary(fit_ag)
zph <- tryCatch(cox.zph(fit_ag), error = function(e) NULL)
ph_global_p <- if (!is.null(zph) && "GLOBAL" %in% rownames(zph$table)) as.numeric(zph$table["GLOBAL", "p"]) else NA_real_

cat("\nModel summary:\n")
cat("  N observations used:", s$n, "\n")
cat("  N events:           ", s$nevent, "\n")
cat("  Concordance:        ", round(s$concordance["C"], 4), "\n")
cat(
    "  PH global p-value:  ",
    ifelse(
        is.na(ph_global_p),
        "NA",
        ifelse(ph_global_p < 0.001, "P<0.001", format(round(ph_global_p, 3), nsmall = 3))
    ),
    "\n"
)

res <- data.frame(
    term      = rownames(s$coefficients),
    HR        = s$coefficients[, "exp(coef)"],
    CI_lower  = s$conf.int[, "lower .95"],
    CI_upper  = s$conf.int[, "upper .95"],
    robust_se = s$coefficients[, "robust se"],
    z         = s$coefficients[, "z"],
    p_value   = s$coefficients[, "Pr(>|z|)"],
    stringsAsFactors = FALSE,
    row.names = NULL
)

# clean encoding
for (j in seq_along(res$term)) {
    res$term[j] <- gsub("\u2013", "-", res$term[j], fixed = TRUE)
    res$term[j] <- gsub("\u2014", "-", res$term[j], fixed = TRUE)
}

n_obs       <- s$n
n_events    <- s$nevent
concordance <- round(s$concordance["C"], 4)

list(res = res, n_obs = n_obs, n_events = n_events, concordance = concordance, ph_global_p = ph_global_p)
"""


def run_ag_model_r(
    dat: pd.DataFrame,
    model_label: str = "Andersen–Gill Cox Model",
) -> Tuple[pd.DataFrame, int, int, float, float]:
    """
    Run an Andersen–Gill Cox model via R's survival::coxph with cluster(patient_id).

    Returns
    -------
    res_df : DataFrame with HR, CI, p-value per term
    n_obs, n_events, concordance, ph_global_p : model summaries
    """
    if not HAS_RPY2:
        raise ImportError("rpy2 is required to run R-based AG models.")

    dat_r = _prepare_dat_r(dat)
    importr("survival")

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["dat"] = ro.conversion.py2rpy(dat_r)

    obj = ro.r(_R_AG_TEMPLATE)

    with localconverter(ro.default_converter + pandas2ri.converter):
        res_df = ro.conversion.rpy2py(obj.rx2("res"))

    n_obs = int(np.array(obj.rx2("n_obs"))[0])
    n_events = int(np.array(obj.rx2("n_events"))[0])
    concordance = float(np.array(obj.rx2("concordance"))[0])
    ph_global_p = float(np.array(obj.rx2("ph_global_p"))[0])

    # Friendly labels & formatted columns
    res_df["label"] = res_df["term"].apply(_clean_label)
    res_df["HR_str"] = res_df.apply(
        lambda r: f"{r['HR']:.2f} ({r['CI_lower']:.2f}–{r['CI_upper']:.2f})", axis=1
    )
    res_df["p_fmt"] = res_df["p_value"].apply(fmt_p)
    res_df["sig"] = res_df["p_value"].apply(sig_stars)

    print(f"\n{'='*65}")
    print(f"  {model_label}")
    print(f"  N = {n_obs:,}  |  Events = {n_events:,}  |  C-index = {concordance}  |  PH p = {fmt_p(ph_global_p)}")
    print(f"{'='*65}")
    return res_df, n_obs, n_events, concordance, ph_global_p


# =====================================================================
#  VII. RESULT TABLES
# =====================================================================
def result_table(
    res_df: pd.DataFrame,
    n_obs: int,
    n_events: int,
    concordance: float,
    ph_global_p: float,
    model_label: str = "Andersen–Gill Cox Model",
) -> pd.DataFrame:
    """Return a publication-ready results DataFrame."""
    tbl = res_df[["label", "HR_str", "p_fmt", "sig"]].copy()
    tbl.columns = ["Variable", "HR (95% CI)", "p-value", ""]
    tbl.attrs["model_label"] = model_label
    tbl.attrs["n_obs"] = n_obs
    tbl.attrs["n_events"] = n_events
    tbl.attrs["concordance"] = concordance
    tbl.attrs["ph_global_p"] = ph_global_p
    tbl.attrs["ph_global_p_fmt"] = fmt_p(ph_global_p)
    return tbl


def result_table_publication(
    res_df: pd.DataFrame,
    n_obs: int,
    n_events: int,
    concordance: float,
    ph_global_p: float,
    model_label: str = "Andersen–Gill Cox Model",
) -> pd.DataFrame:
    """Return a manuscript-ready multivariable Cox table."""
    base = result_table(
        res_df,
        n_obs,
        n_events,
        concordance,
        ph_global_p,
        model_label=model_label,
    )
    out = base[["Variable", "HR (95% CI)", "p-value"]].copy()
    out.columns = ["Variable", "Adjusted HR (95% CI)", "p-value"]
    out["Variable"] = out["Variable"].apply(_publication_text)
    out["Adjusted HR (95% CI)"] = out["Adjusted HR (95% CI)"].apply(_publication_text)
    out.attrs.update(base.attrs)
    return out


def result_table_html(
    tbl: pd.DataFrame,
    model_label: str = "Andersen–Gill Cox Model",
    n_obs: int = 0,
    n_events: int = 0,
    concordance: float = 0.0,
    ph_global_p: float = np.nan,
) -> str:
    """Generate a styled HTML table for display in Jupyter."""
    html = f"""
    <style>
            .ag-wrap {{ font-family:Arial,sans-serif; color:{FIG_DEFAULTS['text_color']}; }}
            .ag-meta {{ margin:0 0 10px 0; font-size:12px; color:#4b6072; }}
            .ag-tbl {{ border-collapse:collapse; font-family:Arial,sans-serif; font-size:12px; width:100%; border:1px solid {FIG_DEFAULTS['border_color']}; }}
            .ag-tbl th {{ background:{FIG_DEFAULTS['header_bg']}; color:{FIG_DEFAULTS['header_fg']}; padding:7px 12px; text-align:left; border:1px solid {FIG_DEFAULTS['border_color']}; }}
            .ag-tbl td {{ padding:6px 12px; border:1px solid {FIG_DEFAULTS['border_color']}; }}
            .ag-tbl tbody tr:nth-child(even) {{ background:{FIG_DEFAULTS['row_alt']}; }}
            .ag-tbl td.num {{ text-align:right; white-space:nowrap; }}
            .ag-tbl td.sig {{ color:#C0392B; font-weight:bold; text-align:center; width:44px; }}
            .ag-note {{ margin-top:8px; font-size:11px; color:#607285; }}
    </style>
        <div class="ag-wrap">
        <p class="ag-meta"><b>{model_label}</b> &nbsp;|&nbsp; N = {n_obs:,} &nbsp;|&nbsp;
               Events = {n_events:,} &nbsp;|&nbsp; Concordance = {concordance:.3f} &nbsp;|&nbsp; PH test <i>p</i> = {fmt_p(ph_global_p)}</p>
        <table class="ag-tbl">
      <thead><tr><th>Variable</th><th>HR (95% CI)</th><th><i>p</i>-value</th><th></th></tr></thead>
      <tbody>
    """
    for _, row in tbl.iterrows():
        html += (
            f"<tr><td>{row['Variable']}</td>"
            f"<td class='num'>{row['HR (95% CI)']}</td>"
            f"<td class='num'>{row['p-value']}</td>"
            f"<td class='sig'>{row['']}</td></tr>\n"
        )
    html += """</tbody></table>
    <p class="ag-note">
    * p&lt;0.05 &nbsp; ** p&lt;0.01 &nbsp; *** p&lt;0.001 &nbsp;|&nbsp;
    Robust SE via cluster(patient_id). Reference: implant index = 1, age 40–60 yr,
    length Medium (10–11.5 mm), diameter Medium (3.7–4.2 mm).</p></div>"""
    return html


# =====================================================================
#  VIII. FOREST PLOT (JCP-style)
# =====================================================================
def _apply_jcp_style():
    """Apply JCP-compatible matplotlib defaults."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FIG_DEFAULTS["font_family"], "Helvetica", "DejaVu Sans"],
        "font.size": FIG_DEFAULTS["font_size"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titleweight": "bold",
        "axes.edgecolor": FIG_DEFAULTS["border_color"],
        "figure.dpi": 300,  # screen; save at 600
        "savefig.dpi": FIG_DEFAULTS["dpi"],
    })


def forest_plot(
    res_df: pd.DataFrame,
    n_obs: int,
    n_events: int,
    concordance: float,
    model_label: str = "Andersen–Gill Cox Model",
    save_path: Optional[str] = None,
    figsize_width: float = 10,
) -> matplotlib.figure.Figure:
    """
    Publication-quality forest plot (log-scale HR with 95% CI).

    Parameters
    ----------
    res_df : DataFrame from run_ag_model_r (must contain HR, CI_lower, CI_upper, p_value, label)
    save_path : if given, saves .png and .pdf at 600 dpi
    """
    _apply_jcp_style()

    fp_plot = _prepare_forest_plot_data(res_df)

    n = len(fp_plot)
    fig_height = max(5.5, n * 0.42 + 1.6)
    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))

    sig_c = FIG_DEFAULTS["sig_color"]
    ns_c = FIG_DEFAULTS["ns_color"]

    # Alternating row background
    for i in range(n):
        row = fp_plot.iloc[i]
        if row["row_type"] == "header":
            ax.axhspan(i - 0.5, i + 0.5, color="#E7EEF5", zorder=0)
        elif i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=FIG_DEFAULTS["bg_alt"], zorder=0)

    data_rows = fp_plot[fp_plot["row_type"] == "data"]
    x_min = min(data_rows["log_lower"].min() - 0.3, np.log(0.2))
    x_max = max(data_rows["log_upper"].max() + 0.3, np.log(10))

    # Null line
    ax.axvline(0, color=FIG_DEFAULTS["null_color"], linewidth=1.0, linestyle="--", zorder=1)
    ax.axvspan(x_min, 0, color=WONG["blue"], alpha=0.03, zorder=0)
    ax.axvspan(0, x_max, color=WONG["vermil"], alpha=0.03, zorder=0)

    # CI + point estimate
    for i, row in fp_plot.iterrows():
        if row["row_type"] != "data":
            continue
        c = sig_c if row["is_sig"] else ns_c
        ax.plot([row["log_lower"], row["log_upper"]], [i, i], color=c, linewidth=1.4, zorder=2)
        ax.plot(row["log_HR"], i, marker="D", markersize=6, color=c, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)

    # X-axis (log scale ticks)
    hr_ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(np.log(hr_ticks))
    ax.set_xticklabels([str(h) for h in hr_ticks])
    ax.set_xlim(x_min, x_max)

    ax.set_yticks(range(n))
    ax.set_yticklabels(fp_plot["label"])
    ax.set_ylim(n - 0.2, -0.8)

    # HR + p text annotations
    x_text = ax.get_xlim()[1] + 0.08
    for i, row in fp_plot.iterrows():
        if row["row_type"] != "data":
            continue
        ax.text(x_text, i, f"{row['HR']:.2f}  {row['p_fmt']}",
                va="center", ha="left", fontsize=7.5, color="#333")
    ax.text(x_text, -0.55, "HR     p", va="center", ha="left",
            fontsize=8, fontweight="bold", color="#222")

    for tick, row_type in zip(ax.get_yticklabels(), fp_plot["row_type"]):
        if row_type == "header":
            tick.set_fontweight("bold")
            tick.set_color(FIG_DEFAULTS["title_color"])

    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title(
        f"{model_label}\n"
        f"N = {n_obs:,} implants · {n_events:,} events · C-index = {concordance}",
        fontweight="bold", color=FIG_DEFAULTS["title_color"], pad=10,
    )
    ax.text(0.03, 1.01, "Lower risk", transform=ax.transAxes, color=WONG["blue"], fontsize=8, fontstyle="italic")
    ax.text(0.97, 1.01, "Higher risk", transform=ax.transAxes, color=WONG["vermil"], fontsize=8, fontstyle="italic", ha="right")

    # Legend
    sig_patch = mpatches.Patch(color=sig_c, label="p < 0.05")
    ns_patch = mpatches.Patch(color=ns_c, label="p ≥ 0.05")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower right", framealpha=0.95,
              facecolor="white", edgecolor=FIG_DEFAULTS["border_color"])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    if save_path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
        print(f"  Saved: {save_path}.png / .pdf")

    return fig


def _prepare_forest_plot_data(res_df: pd.DataFrame) -> pd.DataFrame:
    fp = res_df.copy()
    fp["label"] = fp["label"].apply(_publication_text)
    fp["HR_str"] = fp.apply(
        lambda row: row["HR_str"] if "HR_str" in fp.columns and pd.notna(row.get("HR_str")) else _format_hr_ci(row["HR"], row["CI_lower"], row["CI_upper"]),
        axis=1,
    )
    fp["p_fmt"] = fp["p_value"].apply(fmt_p)
    fp["is_sig"] = fp["p_value"] < 0.05
    fp["log_HR"] = np.log(fp["HR"])
    fp["log_lower"] = np.log(fp["CI_lower"])
    fp["log_upper"] = np.log(fp["CI_upper"])
    fp["group"] = fp["label"].apply(_forest_group)
    fp["sort_key"] = fp["label"].apply(_forest_order)
    fp = fp.sort_values("sort_key").reset_index(drop=True)

    grouped_rows = []
    for group in FOREST_GROUP_ORDER:
        grp = fp[fp["group"] == group].copy()
        if grp.empty:
            continue
        grouped_rows.append(pd.DataFrame([{
            "row_type": "header",
            "label": group,
            "HR": np.nan,
            "HR_str": "",
            "log_HR": np.nan,
            "log_lower": np.nan,
            "log_upper": np.nan,
            "p_fmt": "",
            "is_sig": False,
        }]))
        grp["row_type"] = "data"
        grouped_rows.append(grp)

    return pd.concat(grouped_rows, ignore_index=True)


def forest_plot_journal(
    res_df: pd.DataFrame,
    n_obs: int,
    n_events: int,
    concordance: float,
    model_label: str = "Andersen–Gill Cox Model",
    save_path: Optional[str] = None,
    figsize_width: float = 8.5,
) -> matplotlib.figure.Figure:
    """Manuscript-ready monochrome forest plot for journal submission."""
    _apply_jcp_style()

    fp_plot = _prepare_forest_plot_data(res_df)
    n = len(fp_plot)
    fig_height = max(5.2, n * 0.34 + 1.4)
    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))
    fig.subplots_adjust(right=0.78)

    data_rows = fp_plot[fp_plot["row_type"] == "data"]
    x_min = min(data_rows["log_lower"].min() - 0.2, np.log(0.25))
    x_max = max(data_rows["log_upper"].max() + 0.2, np.log(8))

    ax.axvline(0, color="#404040", linewidth=0.9, linestyle="-", zorder=1)

    for i, row in fp_plot.iterrows():
        if row["row_type"] != "data":
            continue
        ax.plot([row["log_lower"], row["log_upper"]], [i, i], color="#404040", linewidth=1.1, zorder=2)
        ax.plot(row["log_HR"], i, marker="s", markersize=5, color="#202020", zorder=3)

    hr_ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(np.log(hr_ticks))
    ax.set_xticklabels([str(h) for h in hr_ticks])
    ax.set_xlim(x_min, x_max)

    ax.set_yticks(range(n))
    ax.set_yticklabels(fp_plot["label"])
    ax.set_ylim(n - 0.2, -0.8)
    ax.tick_params(axis="y", length=0)

    text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, row in fp_plot.iterrows():
        if row["row_type"] != "data":
            continue
        ax.text(
            1.02,
            i,
            f"{row['HR_str']}  {row['p_fmt']}",
            transform=text_transform,
            va="center",
            ha="left",
            fontsize=7.3,
            color="#202020",
            clip_on=False,
        )
    ax.text(
        1.02,
        -0.55,
        "HR (95% CI)     p-value",
        transform=text_transform,
        va="center",
        ha="left",
        fontsize=7.6,
        fontweight="bold",
        color="#202020",
        clip_on=False,
    )

    for tick, row_type in zip(ax.get_yticklabels(), fp_plot["row_type"]):
        if row_type == "header":
            tick.set_fontweight("bold")
            tick.set_color("#202020")

    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(
        f"{model_label}\n"
        f"N = {n_obs:,} implants · {n_events:,} events · C-index = {concordance:.3f}",
        fontweight="bold",
        color="#202020",
        pad=8,
    )
    ax.grid(axis="x", color="#D5D5D5", linewidth=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    if save_path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
        print(f"  Saved: {save_path}.png / .pdf")

    return fig


# =====================================================================
#  IX. KAPLAN–MEIER PLOT
# =====================================================================
def km_plot_by_group(
    df: pd.DataFrame,
    duration_col: str = "years_to_failure",
    event_col: str = "is_failure",
    group_col: str = "implant_group",
    label_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
    title: str = "Kaplan–Meier Survival Estimate",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """KM survival curves by group with log-rank test."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for KM plots.")

    _apply_jcp_style()

    valid = df[[duration_col, event_col, group_col]].copy()
    valid[event_col] = valid[event_col].fillna(0).astype(int)
    valid = valid.dropna(subset=[duration_col, group_col])

    if label_map is None:
        label_map = {g: str(g) for g in sorted(valid[group_col].unique())}
    if color_map is None:
        palette = list(WONG.values())
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(sorted(valid[group_col].unique()))}

    fig, ax = plt.subplots(figsize=(8, 7.0))
    ax.set_facecolor("white")
    km_fitters = []

    for grp in sorted(valid[group_col].unique()):
        gdf = valid[valid[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(gdf[duration_col], gdf[event_col], label=label_map.get(grp, str(grp)))
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.15,
            color=color_map.get(grp, "#333"), linewidth=1.8,
        )
        km_fitters.append(kmf)

    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, color=FIG_DEFAULTS["title_color"], pad=10)
    ax.grid(True, color=FIG_DEFAULTS["grid_color"], alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95, facecolor="white", edgecolor=FIG_DEFAULTS["border_color"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    x_cap = min(10.0, float(valid[duration_col].max())) if not valid.empty else 10.0
    if x_cap > 0:
        ax.set_xlim(0, x_cap)
        xticks = np.arange(0, np.floor(x_cap) + 1, 1)
        if len(xticks) > 1:
            ax.set_xticks(xticks)
        add_at_risk_counts(*km_fitters, ax=ax, xticks=ax.get_xticks(), rows_to_show=["At risk"])

    fig.subplots_adjust(bottom=0.34)

    # Log-rank test
    if valid[group_col].nunique() >= 2:
        lr = multivariate_logrank_test(valid[duration_col], valid[group_col], valid[event_col])
        ax.text(0.98, 0.03, f"Log-rank p = {fmt_p(lr.p_value)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=FIG_DEFAULTS["border_color"], alpha=0.9))

    fig.tight_layout(rect=[0, 0.16, 1, 1])

    if save_path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
        print(f"  Saved: {save_path}.png / .pdf")

    return fig


def compute_followup_time(
    df: pd.DataFrame,
    study_end: str = "2023-12-31",
    start_col: str = "duedate",
    event_date_col: str = "failure_date",
    event_col: str = "is_failure",
) -> pd.DataFrame:
    """Return a copy with internally consistent follow-up time columns."""
    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[event_date_col] = pd.to_datetime(out[event_date_col], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)

    study_end_ts = pd.to_datetime(study_end)
    followup_end = out[event_date_col].where(out[event_col].eq(1), study_end_ts)
    out["followup_end_date"] = pd.to_datetime(followup_end, errors="coerce")
    out["followup_days"] = (out["followup_end_date"] - out[start_col]).dt.days
    out.loc[out["followup_days"] < 0, "followup_days"] = np.nan
    out["followup_years"] = out["followup_days"] / 365.25
    out["reverse_km_event"] = 1 - out[event_col]
    return out


def _build_survival_plot_data(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
) -> pd.DataFrame:
    valid = df[[duration_col, event_col, group_col]].copy()
    valid[duration_col] = pd.to_numeric(valid[duration_col], errors="coerce")
    valid[event_col] = pd.to_numeric(valid[event_col], errors="coerce").fillna(0).astype(int)
    valid = valid.dropna(subset=[duration_col, group_col])
    valid = valid[valid[duration_col] >= 0].copy()
    return valid


def _resolve_group_maps(
    groups: List[object],
    label_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
) -> Tuple[dict, dict]:
    sorted_groups = sorted(groups)
    if label_map is None:
        label_map = {g: str(g) for g in sorted_groups}
    if color_map is None:
        palette = list(WONG.values())
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(sorted_groups)}
    return label_map, color_map


def _plot_survival_curves_on_axis(
    df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
    duration_col: str,
    event_col: str,
    group_col: str,
    label_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
    title: str = "Kaplan-Meier Survival Estimate",
    ylabel: str = "Survival probability",
    reverse: bool = False,
    x_cap: Optional[float] = None,
) -> Tuple[List[KaplanMeierFitter], pd.DataFrame, np.ndarray, float]:
    """Plot grouped KM-style curves onto an existing axis."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for KM plots.")

    valid = _build_survival_plot_data(df, duration_col, event_col, group_col)
    label_map, color_map = _resolve_group_maps(valid[group_col].unique().tolist(), label_map, color_map)

    ax.set_facecolor("white")
    fitters = []
    sorted_groups = sorted(valid[group_col].unique())
    for grp in sorted_groups:
        gdf = valid[valid[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(gdf[duration_col], gdf[event_col], label=label_map.get(grp, str(grp)))
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            ci_alpha=0.15,
            color=color_map.get(grp, "#333333"),
            linewidth=1.8,
        )
        fitters.append(kmf)

    if x_cap is None:
        x_cap = min(10.0, float(valid[duration_col].max())) if not valid.empty else 10.0
    xticks = np.arange(0, np.floor(x_cap) + 1, 1) if x_cap > 0 else np.array([0.0])

    ax.set_xlabel("Years")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, color=FIG_DEFAULTS["title_color"], pad=10)
    ax.grid(True, color=FIG_DEFAULTS["grid_color"], alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95, facecolor="white", edgecolor=FIG_DEFAULTS["border_color"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if x_cap > 0:
        ax.set_xlim(0, x_cap)
        if len(xticks) > 1:
            ax.set_xticks(xticks)

    if valid[group_col].nunique() >= 2:
        lr = multivariate_logrank_test(valid[duration_col], valid[group_col], valid[event_col])
        label = "Reverse log-rank" if reverse else "Log-rank"
        ax.text(
            0.98,
            0.03,
            f"{label} p = {fmt_p(lr.p_value)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=FIG_DEFAULTS["border_color"], alpha=0.9),
        )
        p_value = float(lr.p_value)
    else:
        p_value = np.nan

    return fitters, valid, xticks, p_value


def _at_risk_count(kmf: KaplanMeierFitter, time_point: float) -> int:
    event_table = kmf.event_table[["at_risk"]].reset_index()
    timeline = event_table.iloc[:, 0].to_numpy(dtype=float)
    at_risk = event_table["at_risk"].to_numpy(dtype=float)
    idx = np.searchsorted(timeline, float(time_point), side="right") - 1
    if idx < 0:
        return int(at_risk[0]) if len(at_risk) else 0
    return int(at_risk[idx])


def draw_km_risk_table(
    ax: matplotlib.axes.Axes,
    fitters: List[KaplanMeierFitter],
    xticks: np.ndarray,
    color_map: Optional[dict] = None,
) -> None:
    """Draw a compact at-risk table on a dedicated axis."""
    ax.axis("off")
    if not fitters:
        return

    labels = [fitter._label for fitter in fitters]
    if color_map is None:
        color_map = {label: "#333333" for label in labels}

    n_rows = len(fitters)
    ax.set_xlim(-1.4, max(len(xticks) - 0.2, 1))
    ax.set_ylim(-0.8, n_rows + 0.8)

    ax.text(-1.35, n_rows + 0.2, "At risk", ha="left", va="center", fontsize=8, fontweight="bold")
    for col_idx, tick in enumerate(xticks):
        ax.text(col_idx, n_rows + 0.2, f"{int(tick)}", ha="center", va="center", fontsize=8, fontweight="bold")

    for row_idx, fitter in enumerate(fitters):
        y = n_rows - 1 - row_idx
        label = fitter._label
        ax.text(-1.35, y, label, ha="left", va="center", fontsize=8, color=color_map.get(label, "#333333"))
        for col_idx, tick in enumerate(xticks):
            ax.text(col_idx, y, f"{_at_risk_count(fitter, tick):,}", ha="center", va="center", fontsize=8)


def km_time_at_risk_table(
    df: pd.DataFrame,
    duration_col: str = "years_to_failure",
    event_col: str = "is_failure",
    group_col: str = "implant_group",
    label_map: Optional[dict] = None,
    x_cap: Optional[float] = None,
) -> pd.DataFrame:
    """Return a standalone time-at-risk table for grouped KM curves."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for KM plots.")

    valid = _build_survival_plot_data(df, duration_col, event_col, group_col)
    label_map, _ = _resolve_group_maps(valid[group_col].unique().tolist(), label_map, None)

    if x_cap is None:
        x_cap = min(10.0, float(valid[duration_col].max())) if not valid.empty else 10.0
    xticks = np.arange(0, np.floor(x_cap) + 1, 1) if x_cap > 0 else np.array([0.0])

    rows = []
    for grp in sorted(valid[group_col].unique()):
        gdf = valid[valid[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(gdf[duration_col], gdf[event_col], label=label_map.get(grp, str(grp)))

        row = {"Group": label_map.get(grp, str(grp))}
        for tick in xticks:
            row[f"{int(tick)} years"] = _at_risk_count(kmf, tick)
        rows.append(row)

    return pd.DataFrame(rows)


def _km_time_at_survival_probability(kmf: KaplanMeierFitter, survival_probability: float) -> float:
    """Return the earliest time where KM survival drops to or below a target level."""
    sf = kmf.survival_function_.reset_index()
    time_col, surv_col = sf.columns[:2]
    crossed = sf[sf[surv_col] <= float(survival_probability)]
    if crossed.empty:
        return np.nan
    return float(crossed.iloc[0][time_col])


def reverse_km_followup_summary(
    df: pd.DataFrame,
    study_end: str = "2023-12-31",
    group_col: str = "implant_group",
    event_col: str = "is_failure",
    label_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Return potential follow-up summaries by group using reverse Kaplan-Meier."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for KM plots.")

    combined = compute_followup_time(df, study_end=study_end, event_col=event_col)
    valid = _build_survival_plot_data(combined, "followup_years", "reverse_km_event", group_col)
    label_map, _ = _resolve_group_maps(valid[group_col].unique().tolist(), label_map, None)

    rows = []
    for grp in sorted(valid[group_col].unique()):
        gdf = valid[valid[group_col] == grp].copy()
        kmf = KaplanMeierFitter()
        kmf.fit(gdf["followup_years"], gdf["reverse_km_event"], label=label_map.get(grp, str(grp)))

        q25 = _km_time_at_survival_probability(kmf, 0.75)
        q50 = _km_time_at_survival_probability(kmf, 0.50)
        q75 = _km_time_at_survival_probability(kmf, 0.25)

        rows.append({
            "Group": label_map.get(grp, str(grp)),
            "N": len(gdf),
            "Observed failures": int(pd.to_numeric(combined.loc[gdf.index, event_col], errors="coerce").fillna(0).sum()),
            "Administratively censored": int((1 - pd.to_numeric(combined.loc[gdf.index, event_col], errors="coerce").fillna(0)).sum()),
            "25th percentile FU (yr)": np.nan if pd.isna(q25) else round(q25, 2),
            "Median potential FU (yr)": np.nan if pd.isna(q50) else round(q50, 2),
            "75th percentile FU (yr)": np.nan if pd.isna(q75) else round(q75, 2),
            "Max observed FU (yr)": round(float(gdf["followup_years"].max()), 2),
        })

    return pd.DataFrame(rows)


def _annotate_reverse_km_medians(
    ax: matplotlib.axes.Axes,
    summary_df: pd.DataFrame,
    label_map: dict,
    color_map: dict,
) -> None:
    """Draw vertical median potential follow-up lines for each group on the plot."""
    if summary_df.empty or "Median potential FU (yr)" not in summary_df.columns:
        return

    display_order = [label_map.get(key, str(key)) for key in sorted(label_map)]
    annotation_levels = [0.24, 0.16, 0.08, 0.32]
    level_idx = 0
    last_x = None
    for group_label in display_order:
        row = summary_df[summary_df["Group"] == group_label]
        if row.empty:
            continue
        median_value = row["Median potential FU (yr)"].iloc[0]
        color_key = next((key for key, value in label_map.items() if value == group_label), group_label)
        color = color_map.get(color_key, "#333333")
        if pd.isna(median_value):
            continue
        if last_x is not None and abs(float(median_value) - last_x) < 0.6:
            level_idx += 1
        else:
            level_idx = 0
        annotation_y = annotation_levels[level_idx % len(annotation_levels)]
        ax.axvline(float(median_value), color=color, linestyle="--", linewidth=1.1, alpha=0.9, zorder=1)
        ax.text(
            float(median_value),
            annotation_y,
            f"{group_label}\n{median_value:.2f} y",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )
        last_x = float(median_value)


def summarize_followup_bins(
    df: pd.DataFrame,
    group_col: str = "implant_group",
    followup_col: str = "followup_years",
    allowed_groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Count implants by follow-up duration bins within each implant group."""
    bins = [-np.inf, 1, 3, 5, np.inf]
    labels = ["<=1 year", ">1-3 years", ">3-5 years", ">5 years"]

    tmp = df[[group_col, followup_col]].copy()
    tmp[followup_col] = pd.to_numeric(tmp[followup_col], errors="coerce")
    tmp = tmp.dropna(subset=[group_col, followup_col])
    if allowed_groups is not None:
        tmp = tmp[tmp[group_col].astype(str).isin(allowed_groups)].copy()

    tmp["followup_bin"] = pd.cut(tmp[followup_col], bins=bins, labels=labels, right=True)
    summary = (
        tmp.groupby([group_col, "followup_bin"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = summary.groupby(group_col)["count"].transform("sum")
    summary["percent"] = np.where(totals.gt(0), summary["count"] / totals * 100, np.nan)
    return summary


def plot_followup_bin_summary(
    summary_df: pd.DataFrame,
    ax: matplotlib.axes.Axes,
    label_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
    title: str = "Follow-up duration categories",
) -> None:
    """Plot grouped bars summarizing recurrent-implant follow-up bins."""
    if summary_df.empty:
        ax.axis("off")
        return

    groups = sorted(summary_df.iloc[:, 0].astype(str).unique())
    label_map, color_map = _resolve_group_maps(groups, label_map, color_map)
    bin_order = ["<=1 year", ">1-3 years", ">3-5 years", ">5 years"]
    pivot = (
        summary_df.assign(_group=summary_df.iloc[:, 0].astype(str))
        .pivot(index="followup_bin", columns="_group", values="count")
        .reindex(bin_order)
        .fillna(0)
    )

    x = np.arange(len(pivot.index))
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, num=len(groups)) if len(groups) > 1 else np.array([0.0])

    for offset, grp in zip(offsets, groups):
        heights = pivot[grp].to_numpy(dtype=float) if grp in pivot.columns else np.zeros(len(pivot.index))
        bars = ax.bar(
            x + offset,
            heights,
            width=width / max(len(groups), 1) * 1.8,
            label=label_map.get(grp, str(grp)),
            color=color_map.get(grp, "#333333"),
            alpha=0.9,
        )
        for bar, value in zip(bars, heights):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{int(value)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.set_ylabel("Number of implants")
    ax.set_title(title, color=FIG_DEFAULTS["title_color"], pad=8)
    ax.grid(axis="y", color=FIG_DEFAULTS["grid_color"], alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="white", edgecolor=FIG_DEFAULTS["border_color"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_implant_sequence_survival_followup_figure(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    study_end: str = "2023-12-31",
    duration_col: str = "years_to_failure",
    event_col: str = "is_failure",
    group_col: str = "implant_group",
    label_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
) -> matplotlib.figure.Figure:
    """Create a combined figure with panel A reverse KM and panel B KM."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for KM plots.")

    _apply_jcp_style()
    combined = compute_followup_time(df, study_end=study_end, event_col=event_col)
    label_map, color_map = _resolve_group_maps(
        combined[group_col].dropna().astype(str).unique().tolist(),
        label_map,
        color_map,
    )

    fig = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.1, 1.0],
        wspace=0.18,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b_curve = fig.add_subplot(gs[0, 1])

    reverse_summary = reverse_km_followup_summary(
        df,
        study_end=study_end,
        group_col=group_col,
        event_col=event_col,
        label_map=label_map,
    )

    _plot_survival_curves_on_axis(
        combined,
        ax=ax_a,
        duration_col="followup_years",
        event_col="reverse_km_event",
        group_col=group_col,
        label_map=label_map,
        color_map=color_map,
        title="Reverse Kaplan-Meier follow-up by implant sequence",
        ylabel="Probability of remaining under follow-up",
        reverse=True,
        x_cap=10.0,
    )
    _annotate_reverse_km_medians(ax_a, reverse_summary, label_map, color_map)
    ax_a.text(
        -0.12,
        1.04,
        "A",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=FIG_DEFAULTS["title_color"],
        clip_on=False,
    )

    _plot_survival_curves_on_axis(
        combined,
        ax=ax_b_curve,
        duration_col=duration_col,
        event_col=event_col,
        group_col=group_col,
        label_map=label_map,
        color_map=color_map,
        title="Kaplan-Meier survival by implant sequence",
        ylabel="Survival probability",
        reverse=False,
        x_cap=10.0,
    )
    ax_b_curve.text(
        -0.12,
        1.04,
        "B",
        transform=ax_b_curve.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=FIG_DEFAULTS["title_color"],
        clip_on=False,
    )

    if save_path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")
        print(f"  Saved: {save_path}.png / .pdf")

    return fig


# =====================================================================
#  X. DESCRIPTIVE / EDA TABLES
# =====================================================================
def continuous_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Summary statistics for continuous variables."""
    rows = []
    for col in [c for c in cols if c in df.columns]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Variable": col,
            "N": len(s),
            "Mean ± SD": f"{s.mean():.2f} ± {s.std():.2f}",
            "Median": f"{s.median():.2f}",
            "Min": f"{s.min():.2f}",
            "Max": f"{s.max():.2f}",
        })
    return pd.DataFrame(rows)


def frequency_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Consolidated frequency counts for categorical/binary columns."""
    rows = []
    for col in [c for c in cols if c in df.columns]:
        vc = df[col].value_counts(dropna=False).sort_index()
        total = vc.sum()
        for val, cnt in vc.items():
            val_str = str(val) if not pd.isna(val) else "Missing"
            pct = cnt / total * 100
            if val_str in ("Missing", "<NA>", "nan") and pct < 0.1:
                continue
            rows.append({
                "Variable": _display_variable_name(col),
                "Level": val_str,
                "N": f"{cnt:,}",
                "Percent": f"{pct:.1f}%",
            })
    return pd.DataFrame(rows)


def follow_up_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute follow-up time summary."""
    cohort_last = pd.to_datetime("2023-12-31")
    end = df["failure_date"].fillna(cohort_last)
    fu = (end - pd.to_datetime(df["duedate"], errors="coerce")).dt.days
    fu_years = fu / 365.25
    return pd.DataFrame([{
        "Metric": "Follow-up (years)",
        "Mean ± SD": f"{fu_years.mean():.2f} ± {fu_years.std():.2f}",
        "Median": f"{fu_years.median():.2f}",
        "Min": f"{fu_years.min():.2f}",
        "Max": f"{fu_years.max():.2f}",
    }])


# =====================================================================
#  XI. LOG-RANK TESTS
# =====================================================================
def logrank_all_variables(
    df: pd.DataFrame,
    test_cols: List[str],
    duration_col: str = "days_to_failure",
    event_col: str = "is_failure",
) -> pd.DataFrame:
    """Run multivariate log-rank test for each variable; return sorted results."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for log-rank tests.")

    base = df[[duration_col, event_col]].copy()
    base[event_col] = base[event_col].fillna(0).astype(int)

    results = []
    for col in test_cols:
        if col not in df.columns:
            continue
        tmp = pd.concat([base, df[[col]]], axis=1).dropna(subset=[duration_col, col])
        if tmp[col].nunique() < 2:
            continue
        res = multivariate_logrank_test(tmp[duration_col], tmp[col], tmp[event_col])
        results.append({"Variable": _display_variable_name(col), "p-value": res.p_value, "N": len(tmp)})

    out = pd.DataFrame(results).sort_values("p-value")
    out["p-value (fmt)"] = out["p-value"].apply(fmt_p)
    out["Significant"] = out["p-value"] < 0.05
    return out


def univariable_survival_summary(
    df: pd.DataFrame,
    test_cols: List[str],
    duration_col: str = "days_to_failure",
    event_col: str = "is_failure",
) -> pd.DataFrame:
    """Return one consolidated table with success rates, log-rank p-values, and univariable Cox results."""
    if not HAS_LIFELINES:
        raise ImportError("lifelines is required for univariable Cox summaries.")

    base = df[[duration_col, event_col]].copy()
    base[duration_col] = pd.to_numeric(base[duration_col], errors="coerce")
    base[event_col] = pd.to_numeric(base[event_col], errors="coerce").fillna(0).astype(int)

    rows = []
    for col in test_cols:
        if col not in df.columns:
            continue

        tmp = pd.concat([base, df[[col]]], axis=1)
        tmp[col] = _coerce_univariable_levels(tmp[col], col)
        tmp = tmp.dropna(subset=[duration_col, col]).copy()
        tmp = tmp[tmp[col].astype(str).str.lower() != "<na>"]
        if tmp.empty or tmp[col].nunique() < 2:
            continue

        levels = _reference_order_for_variable(col, tmp[col])
        if len(levels) < 2:
            continue
        tmp = tmp[tmp[col].isin(levels)].copy()
        tmp[col] = pd.Categorical(tmp[col], categories=levels, ordered=True)

        try:
            lr = multivariate_logrank_test(tmp[duration_col], tmp[col], tmp[event_col])
            logrank_p = lr.p_value
        except Exception:
            logrank_p = np.nan

        level_stats = (
            tmp.groupby(col, observed=True)[event_col]
            .agg(N="size", Events="sum")
            .reindex(levels)
            .fillna(0)
        )
        level_stats["Success rate (%)"] = np.where(
            level_stats["N"] > 0,
            (1 - level_stats["Events"] / level_stats["N"]) * 100,
            np.nan,
        )

        cox_df = tmp[[duration_col, event_col]].copy()
        dummies = pd.get_dummies(tmp[col], prefix=col, prefix_sep="=", drop_first=True, dtype=float)
        cox_df = pd.concat([cox_df, dummies], axis=1)

        hr_by_level: Dict[str, Dict[str, object]] = {}
        if not dummies.empty:
            try:
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
                summary = cph.summary.reset_index().rename(columns={"covariate": "term"})
                intervals = cph.confidence_intervals_.reset_index().rename(columns={"covariate": "term"})
                summary = summary.merge(intervals, on="term", how="left")
                lower_col = "95% lower-bound"
                upper_col = "95% upper-bound"
                if lower_col not in summary.columns or upper_col not in summary.columns:
                    alt_lower = [c for c in summary.columns if c.startswith("95% lower")]
                    alt_upper = [c for c in summary.columns if c.startswith("95% upper")]
                    lower_col = alt_lower[0]
                    upper_col = alt_upper[0]
                for _, row in summary.iterrows():
                    raw_term = str(row["term"])
                    level = raw_term.split("=", 1)[1] if "=" in raw_term else raw_term.replace(f"{col}_", "", 1)
                    hr = float(np.exp(row["coef"]))
                    ci_lower = float(np.exp(row[lower_col]))
                    ci_upper = float(np.exp(row[upper_col]))
                    hr_by_level[level] = {
                        "HR": hr,
                        "CI_lower": ci_lower,
                        "CI_upper": ci_upper,
                        "HR_str": f"{hr:.2f} ({ci_lower:.2f}–{ci_upper:.2f})",
                        "p_value": float(row["p"]),
                        "p_fmt": fmt_p(float(row["p"])),
                    }
            except Exception:
                hr_by_level = {}

        display_name = _display_variable_name(col)
        reference_level = str(levels[0])
        for level in levels:
            stats = level_stats.loc[level]
            level_str = str(level)
            cox_row = hr_by_level.get(level_str, {})
            is_reference = level_str == reference_level
            rows.append({
                "_group": _variable_group_name(col),
                "_source_col": col,
                "Variable": display_name,
                "Level": level_str,
                "N": int(stats["N"]),
                "Events": int(stats["Events"]),
                "Success rate (%)": round(float(stats["Success rate (%)"]), 1) if pd.notna(stats["Success rate (%)"]) else np.nan,
                "Log-rank p-value": fmt_p(logrank_p),
                "Cox HR (95% CI)": "Reference" if is_reference else cox_row.get("HR_str", ""),
                "Cox p-value": "" if is_reference else cox_row.get("p_fmt", ""),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    display_cols = [
        "Variable", "Level", "N", "Events", "Success rate (%)",
        "Log-rank p-value", "Cox HR (95% CI)", "Cox p-value",
    ]
    out["_group_order"] = out["_group"].map({group: i for i, group in enumerate(FOREST_GROUP_ORDER)}).fillna(len(FOREST_GROUP_ORDER))
    out["_var_order"] = out["_source_col"].map({c: i for i, c in enumerate(test_cols)}).fillna(len(test_cols))
    out["_level_order"] = out.groupby("_source_col").cumcount()
    out = out.sort_values(["_group_order", "_var_order", "_level_order", "Variable", "Level"]).reset_index(drop=True)

    grouped_rows = []
    for group in FOREST_GROUP_ORDER:
        group_df = out[out["_group"] == group]
        if group_df.empty:
            continue
        grouped_rows.append({
            "Variable": group,
            "Level": "",
            "N": np.nan,
            "Events": np.nan,
            "Success rate (%)": np.nan,
            "Log-rank p-value": "",
            "Cox HR (95% CI)": "",
            "Cox p-value": "",
        })
        grouped_rows.extend(group_df[display_cols].to_dict("records"))

    other_df = out[~out["_group"].isin(FOREST_GROUP_ORDER)]
    if not other_df.empty:
        grouped_rows.append({
            "Variable": "Other",
            "Level": "",
            "N": np.nan,
            "Events": np.nan,
            "Success rate (%)": np.nan,
            "Log-rank p-value": "",
            "Cox HR (95% CI)": "",
            "Cox p-value": "",
        })
        grouped_rows.extend(other_df[display_cols].to_dict("records"))

    return pd.DataFrame(grouped_rows, columns=display_cols)


# =====================================================================
#  XII. STACKED BAR CHART – SURVIVAL BY PROCEDURE SEQUENCE
# =====================================================================
def plot_survival_by_sequence(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Stacked 100% bar chart of survival vs. failure by implant sequence (1–4)."""
    _apply_jcp_style()

    idx_col = "implant_num_eda" if "implant_num_eda" in df.columns else "implant_index"
    data = df.copy()
    data["idx"] = pd.to_numeric(data[idx_col], errors="coerce").clip(1, 4)
    data["fail"] = pd.to_numeric(data["is_failure"], errors="coerce").fillna(0).astype(int)
    data = data[data["idx"].isin([1, 2, 3, 4])]

    agg = data.groupby("idx")["fail"].agg(n="size", fails="sum").reindex([1, 2, 3, 4]).fillna(0)
    agg["fail_pct"] = np.where(agg["n"] > 0, agg["fails"] / agg["n"] * 100, 0)
    agg["surv_pct"] = 100 - agg["fail_pct"]

    labels = [f"{ord_}\nn={int(agg.loc[i,'n']):,}" for i, ord_ in zip([1,2,3,4], ["1st","2nd","3rd","4th"])]
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("white")
    ax.bar(x, agg["surv_pct"], 0.6, label="Survival (%)", color=WONG["green"], edgecolor="black", lw=0.5)
    ax.bar(x, agg["fail_pct"], 0.6, bottom=agg["surv_pct"], label="Failure (%)", color=WONG["vermil"], edgecolor="black", lw=0.5)

    for i in range(4):
        s, f = agg.iloc[i]["surv_pct"], agg.iloc[i]["fail_pct"]
        if s > 0:
            ax.text(x[i], s / 2, f"{s:.1f}%", ha="center", va="center", color="white", fontsize=8, weight="bold")
        if f > 1:
            ax.text(x[i], 100 - f / 2, f"{f:.1f}%", ha="center", va="center", color="black", fontsize=8, weight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0, 100)
    ax.set_title("Implant Survival by Procedure Sequence", color=FIG_DEFAULTS["title_color"], pad=10)
    ax.yaxis.grid(True, linestyle="--", color=FIG_DEFAULTS["grid_color"], alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(bottom=0.22)

    if save_path:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=FIG_DEFAULTS["dpi"], bbox_inches="tight")

    return fig


# =====================================================================
#  XIII. VERSION REPORTING
# =====================================================================
def print_versions():
    """Print package versions for reproducibility."""
    import sys
    pkgs = {"numpy": np, "pandas": pd, "matplotlib": matplotlib}
    print(f"Python {sys.version}")
    for name, mod in pkgs.items():
        print(f"  {name}: {mod.__version__}")
    if HAS_LIFELINES:
        import lifelines
        print(f"  lifelines: {lifelines.__version__}")
    if HAS_RPY2:
        print(f"  rpy2: {ro.r('R.version.string')[0]}")
    print()
