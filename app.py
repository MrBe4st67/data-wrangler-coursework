import hashlib
import io
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Wrangler", layout="wide")


# state
def init_state() -> None:
    defaults = {
        "raw_df": None,
        "working_df": None,
        "history_stack": [],
        "log_steps": [],
        "upload_signature": None,
        "upload_name": None,
        "last_message": None,
        "last_validation": None,
        "last_validation_name": None,
        "last_scaling_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    for key in [
        "raw_df",
        "working_df",
        "history_stack",
        "log_steps",
        "upload_signature",
        "upload_name",
        "last_message",
        "last_validation",
        "last_validation_name",
        "last_scaling_stats",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()


def reset_to_raw() -> None:
    # bring the working file back to the original upload
    if st.session_state.raw_df is not None:
        st.session_state.working_df = st.session_state.raw_df.copy(deep=True)
        st.session_state.history_stack = []
        st.session_state.log_steps = []
        st.session_state.last_validation = None
        st.session_state.last_validation_name = None
        st.session_state.last_message = "All transformations were reset to the original uploaded dataset."


def undo_last_step() -> None:
    # go back one step if we can
    if st.session_state.history_stack:
        st.session_state.working_df = st.session_state.history_stack.pop()
        if st.session_state.log_steps:
            st.session_state.log_steps.pop()
        st.session_state.last_message = "The last transformation was undone."
    else:
        st.session_state.last_message = "There is no previous step to undo."


# loading and profiling
def get_file_signature(file_bytes: bytes, file_name: str) -> str:
    return hashlib.sha1(file_name.encode("utf-8") + file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    file_name_lower = file_name.lower()
    buffer = io.BytesIO(file_bytes)

    if file_name_lower.endswith(".csv"):
        return pd.read_csv(buffer)

    if file_name_lower.endswith(".xlsx"):
        return pd.read_excel(buffer)

    if file_name_lower.endswith(".json"):
        try:
            obj = json.loads(file_bytes.decode("utf-8"))
            if isinstance(obj, dict):
                return pd.json_normalize(obj)
            return pd.DataFrame(obj)
        except Exception:
            buffer.seek(0)
            try:
                return pd.read_json(buffer)
            except ValueError:
                buffer.seek(0)
                return pd.read_json(buffer, lines=True)

    raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON.")


@st.cache_data(show_spinner=False)
def build_profile_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(include=["object", "category", "bool", "string"])

    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().T.reset_index().rename(columns={"index": "Column"})
    else:
        numeric_summary = pd.DataFrame(columns=["Column"])

    if not categorical_df.empty:
        rows = []
        for col in categorical_df.columns:
            mode_series = categorical_df[col].mode(dropna=True)
            rows.append(
                {
                    "Column": col,
                    "Non-null count": int(categorical_df[col].notna().sum()),
                    "Unique values": int(categorical_df[col].nunique(dropna=True)),
                    "Most common value": mode_series.iloc[0] if not mode_series.empty else np.nan,
                }
            )
        categorical_summary = pd.DataFrame(rows)
    else:
        categorical_summary = pd.DataFrame(columns=["Column"])

    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2) if len(df) else pd.Series([0] * len(df.columns), index=df.columns)
    missing_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing count": missing_count.values,
            "Missing %": missing_percent.values,
        }
    )

    duplicate_count = int(df.duplicated().sum())

    outlier_rows = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_rows.append(
            {
                "Column": col,
                "IQR lower bound": lower,
                "IQR upper bound": upper,
                "Outlier count": outlier_count,
                "Outlier %": round(outlier_count / len(df) * 100, 2) if len(df) else 0,
            }
        )
    outlier_summary = pd.DataFrame(outlier_rows)

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "missing_table": missing_table,
        "duplicate_count": duplicate_count,
        "outlier_summary": outlier_summary,
    }


# small helpers
def add_log_step(action: str, params: dict[str, Any], columns: list[str], before_shape: tuple[int, int], after_shape: tuple[int, int], note: str = "") -> None:
    step = {
        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "columns": columns,
        "params": params,
        "before_shape": {"rows": int(before_shape[0]), "columns": int(before_shape[1])},
        "after_shape": {"rows": int(after_shape[0]), "columns": int(after_shape[1])},
        "note": note,
    }
    st.session_state.log_steps.append(step)



def commit_change(new_df: pd.DataFrame, action: str, params: dict[str, Any], columns: list[str], note: str) -> bool:
    # save the old version before changing anything
    old_df = st.session_state.working_df
    if old_df is None:
        return False

    if list(new_df.columns) != list(old_df.columns) or not new_df.equals(old_df):
        st.session_state.history_stack.append(old_df.copy(deep=True))
        add_log_step(action, params, columns, old_df.shape, new_df.shape, note)
        st.session_state.working_df = new_df.copy(deep=True)
        st.session_state.last_message = note
        return True

    st.session_state.last_message = "No rows or values changed, so nothing was added to the log."
    return False



def show_feedback() -> None:
    if st.session_state.last_message:
        st.success(st.session_state.last_message)



def compare_shape_table(before_df: pd.DataFrame, after_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or []
    return pd.DataFrame(
        {
            "Stage": ["Before", "After"],
            "Rows": [int(before_df.shape[0]), int(after_df.shape[0])],
            "Columns": [int(before_df.shape[1]), int(after_df.shape[1])],
            "Affected columns": [", ".join(columns), ", ".join(columns)],
        }
    )



def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2) if len(df) else pd.Series([0] * len(df.columns), index=df.columns)
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Missing count": missing_count.values,
            "Missing %": missing_percent.values,
        }
    )



def clean_dirty_numeric(series: pd.Series) -> pd.Series:
    # helps with values like $1,200 or 35%
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[$€£¥₽₹]", "", regex=True)
    cleaned = cleaned.str.replace(r"%", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")



def get_iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr



def get_zscore_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    z = (series - series.mean()) / std
    return z.abs() > threshold



def get_before_after_stats(before_series: pd.Series, after_series: pd.Series, column_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": [column_name, column_name],
            "Stage": ["Before", "After"],
            "Mean": [before_series.mean(), after_series.mean()],
            "Std": [before_series.std(), after_series.std()],
            "Min": [before_series.min(), after_series.min()],
            "Max": [before_series.max(), after_series.max()],
        }
    )



def prepare_validation_output(df: pd.DataFrame, bad_mask: pd.Series, rule_type: str, checked_columns: str, rule_details: str) -> pd.DataFrame:
    violations_df = df.loc[bad_mask].copy()
    if violations_df.empty:
        return violations_df
    violations_df.insert(0, "Row index", violations_df.index)
    violations_df.insert(1, "Rule type", rule_type)
    violations_df.insert(2, "Checked column(s)", checked_columns)
    violations_df.insert(3, "Rule details", rule_details)
    return violations_df



def as_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")



def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()



def build_mapping_seed(series: pd.Series, max_rows: int = 12) -> pd.DataFrame:
    unique_values = sorted(series.dropna().astype("string").unique().tolist())
    rows = [{"old_value": val, "new_value": ""} for val in unique_values[:max_rows]]
    if not rows:
        rows = [{"old_value": "", "new_value": ""}]
    return pd.DataFrame(rows)



def build_replay_snippet(log_steps: list[dict[str, Any]]) -> str:
    lines = [
        "# replay notes from the app",
        "import pandas as pd",
        "import numpy as np",
        "",
        "df = pd.read_csv('your_input_file.csv')  # change this if your file is not csv",
        "",
    ]

    if not log_steps:
        lines.extend([
            "# no transformation steps were logged yet",
            "# go to page b and apply at least one change",
            "# then come back here and export the recipe again",
        ])
        return "\n".join(lines)

    for i, step in enumerate(log_steps, start=1):
        lines.append(f"# step {i}: {step['action']}")
        lines.append(f"# columns: {step['columns']}")
        lines.append(f"# params: {json.dumps(step['params'], default=str)}")
        if step.get("note"):
            lines.append(f"# note: {step['note']}")
        lines.append("")

    lines.append("# you can turn these notes into full pandas code later if you want")
    return "\n".join(lines)


def show_overview(df: pd.DataFrame) -> None:
    profile = build_profile_tables(df)

    metric_cols = st.columns(4)
    metric_cols[0].metric("rows", int(df.shape[0]))
    metric_cols[1].metric("columns", int(df.shape[1]))
    metric_cols[2].metric("number of columns", int(len(df.columns)))
    metric_cols[3].metric("duplicate rows", int(profile["duplicate_count"]))

    st.subheader("column names")
    st.dataframe(pd.DataFrame({"Column": df.columns}), use_container_width=True)

    st.subheader("data types")
    dtype_df = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("first 10 rows")
    st.dataframe(df.head(10), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["numeric summary", "categorical summary", "missing values", "outliers"])

    with tab1:
        if profile["numeric_summary"].empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(profile["numeric_summary"], use_container_width=True)

    with tab2:
        if profile["categorical_summary"].empty:
            st.info("No categorical/text columns found.")
        else:
            st.dataframe(profile["categorical_summary"], use_container_width=True)

    with tab3:
        st.dataframe(profile["missing_table"], use_container_width=True)

    with tab4:
        if profile["outlier_summary"].empty:
            st.info("No numeric columns found for outlier profiling.")
        else:
            st.dataframe(profile["outlier_summary"], use_container_width=True)


# Page A
def render_upload_page() -> None:
    st.header("Page A — Upload & Overview")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("Reset session", type="secondary"):
            reset_session()
            st.rerun()
    with top_right:
        st.info("Supported file types: CSV, XLSX, JSON")

    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            signature = get_file_signature(file_bytes, uploaded_file.name)
            df = load_uploaded_file(file_bytes, uploaded_file.name)

            if st.session_state.upload_signature != signature:
                st.session_state.raw_df = df.copy(deep=True)
                st.session_state.working_df = df.copy(deep=True)
                st.session_state.history_stack = []
                st.session_state.log_steps = []
                st.session_state.last_validation = None
                st.session_state.last_validation_name = None
                st.session_state.last_scaling_stats = None
                st.session_state.upload_signature = signature
                st.session_state.upload_name = uploaded_file.name
                st.session_state.last_message = f"Loaded '{uploaded_file.name}' and started a fresh workflow."

                st.session_state.log_steps.append(
                    {
                        "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "file upload",
                        "columns": df.columns.tolist(),
                        "params": {
                            "file_name": uploaded_file.name,
                            "rows": int(df.shape[0]),
                            "columns_count": int(df.shape[1]),
                        },
                        "before_shape": {"rows": 0, "columns": 0},
                        "after_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                        "note": "Started a new workflow from uploaded file.",
                    }
                )
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            return

    if st.session_state.working_df is None:
        st.write("Upload a file to start profiling, cleaning, and visualizing your data.")
        return

    show_feedback()

    if st.session_state.upload_name:
        st.caption(f"current file: {st.session_state.upload_name}")
        st.caption("The overview below stays here even if you move to other pages and come back.")

    df = st.session_state.working_df.copy(deep=True)
    show_overview(df)


# Page B
def render_workflow_controls() -> None:
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("Undo last step"):
            undo_last_step()
            st.rerun()
    with c2:
        if st.button("Reset all transformations"):
            reset_to_raw()
            st.rerun()
    with c3:
        if st.session_state.working_df is not None:
            st.info(f"Current working shape: {st.session_state.working_df.shape[0]} rows × {st.session_state.working_df.shape[1]} columns")



def render_missing_values_section(df: pd.DataFrame) -> None:
    st.subheader("4.1 missing values")
    missing_table = get_missing_summary(df)
    st.dataframe(missing_table, use_container_width=True)

    action = st.selectbox(
        "Choose a missing-values action",
        [
            "Drop rows with missing values",
            "Drop columns above a missing threshold",
            "Fill missing values in one column",
        ],
        key="missing_action",
    )

    if action == "Drop rows with missing values":
        selected_cols = st.multiselect("Columns to check", df.columns.tolist(), key="dropna_cols")
        cols_to_use = selected_cols if selected_cols else df.columns.tolist()
        preview_mask = df[cols_to_use].isna().any(axis=1)
        preview_df = df.loc[preview_mask].head(10)
        after_df = df.dropna(subset=cols_to_use)
        st.write("Before / after preview")
        st.dataframe(compare_shape_table(df, after_df, cols_to_use), use_container_width=True)
        if not preview_df.empty:
            st.write("Rows that would be removed")
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Apply row drop", key="apply_drop_rows"):
            changed = commit_change(
                after_df,
                "Missing values",
                {"mode": "drop_rows", "columns_checked": cols_to_use, "rows_removed": int(len(df) - len(after_df))},
                cols_to_use,
                f"Dropped rows with missing values in the selected columns. Removed {len(df) - len(after_df)} rows.",
            )
            if changed:
                st.rerun()

    elif action == "Drop columns above a missing threshold":
        threshold = st.slider("Drop columns when missing % is above", 0, 100, 50, key="missing_threshold")
        cols_to_drop = missing_table.loc[missing_table["Missing %"] > threshold, "Column"].tolist()
        after_df = df.drop(columns=cols_to_drop) if cols_to_drop else df.copy()
        st.write("Before / after preview")
        st.dataframe(compare_shape_table(df, after_df, cols_to_drop), use_container_width=True)
        st.write("Columns that would be dropped", cols_to_drop if cols_to_drop else "None")

        if st.button("Apply column drop", key="apply_drop_cols"):
            if not cols_to_drop:
                st.info("No columns are above the selected threshold.")
            else:
                changed = commit_change(
                    after_df,
                    "Missing values",
                    {"mode": "drop_columns_above_threshold", "threshold_percent": threshold, "columns_dropped": cols_to_drop},
                    cols_to_drop,
                    f"Dropped {len(cols_to_drop)} columns above {threshold}% missing.",
                )
                if changed:
                    st.rerun()

    elif action == "Fill missing values in one column":
        target_col = st.selectbox("Choose a column", df.columns.tolist(), key="fill_target_col")
        is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        methods = ["Constant value", "Forward fill", "Backward fill"]
        if is_numeric:
            methods = ["Mean", "Median", "Mode"] + methods
        else:
            methods = ["Most frequent"] + methods

        method = st.selectbox("Fill method", methods, key="fill_method")
        constant_value = None
        if method == "Constant value":
            constant_value = st.text_input("Constant value", key="fill_constant")

        preview_df = df.loc[df[target_col].isna(), [target_col]].head(10)
        st.write(f"Missing values in '{target_col}': {int(df[target_col].isna().sum())}")
        if not preview_df.empty:
            st.write("Rows with missing values before filling")
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Apply fill", key="apply_fill_missing"):
            if df[target_col].isna().sum() == 0:
                st.info("This column has no missing values.")
                return

            new_df = df.copy(deep=True)
            try:
                if method == "Mean":
                    fill_value = new_df[target_col].mean()
                    new_df[target_col] = new_df[target_col].fillna(fill_value)
                elif method == "Median":
                    fill_value = new_df[target_col].median()
                    new_df[target_col] = new_df[target_col].fillna(fill_value)
                elif method == "Mode":
                    fill_value = new_df[target_col].mode(dropna=True).iloc[0]
                    new_df[target_col] = new_df[target_col].fillna(fill_value)
                elif method == "Most frequent":
                    fill_value = new_df[target_col].mode(dropna=True).iloc[0]
                    new_df[target_col] = new_df[target_col].fillna(fill_value)
                elif method == "Constant value":
                    fill_value = pd.to_numeric(constant_value) if is_numeric else constant_value
                    new_df[target_col] = new_df[target_col].fillna(fill_value)
                elif method == "Forward fill":
                    new_df[target_col] = new_df[target_col].ffill()
                elif method == "Backward fill":
                    new_df[target_col] = new_df[target_col].bfill()
                else:
                    raise ValueError("Unknown fill method.")

                filled_count = int(df[target_col].isna().sum() - new_df[target_col].isna().sum())
                note = f"Filled {filled_count} missing cells in '{target_col}' using {method}."
                changed = commit_change(
                    new_df,
                    "Missing values",
                    {"mode": "fill", "column": target_col, "method": method, "filled_cells": filled_count, "constant_value": constant_value},
                    [target_col],
                    note,
                )
                if changed:
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not fill missing values: {exc}")



def build_duplicate_preview(df: pd.DataFrame, subset_cols: list[str]) -> tuple[pd.DataFrame, int]:
    subset = subset_cols if subset_cols else None
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_df = df.loc[dup_mask].copy()
    count = int(df.duplicated(subset=subset).sum())
    if dup_df.empty:
        return dup_df, count

    grouping_cols = subset_cols if subset_cols else df.columns.tolist()
    dup_df.insert(0, "Duplicate group", dup_df.groupby(grouping_cols, dropna=False).ngroup() + 1)
    return dup_df, count



def render_duplicates_section(df: pd.DataFrame) -> None:
    st.subheader("4.2 duplicates")
    subset_cols = st.multiselect("Check duplicates by these key columns (leave empty for full-row duplicates)", df.columns.tolist(), key="dup_subset_cols")
    duplicate_preview, duplicate_count = build_duplicate_preview(df, subset_cols)
    st.metric("Duplicate rows found", duplicate_count)

    if duplicate_preview.empty:
        st.info("No duplicates found for this selection.")
    else:
        st.write("Duplicate groups preview")
        st.dataframe(duplicate_preview.head(25), use_container_width=True)

    keep_option = st.radio("When removing duplicates, keep", ["first", "last"], horizontal=True, key="dup_keep")

    if st.button("Remove duplicates", key="remove_duplicates"):
        subset = subset_cols if subset_cols else None
        new_df = df.drop_duplicates(subset=subset, keep=keep_option).copy(deep=True)
        removed_rows = int(len(df) - len(new_df))
        changed = commit_change(
            new_df,
            "Duplicates",
            {"subset_columns": subset_cols, "keep": keep_option, "rows_removed": removed_rows},
            subset_cols if subset_cols else df.columns.tolist(),
            f"Removed {removed_rows} duplicate rows and kept the {keep_option} occurrence.",
        )
        if changed:
            st.rerun()



def render_type_parsing_section(df: pd.DataFrame) -> None:
    st.subheader("4.3 data types and parsing")
    target_col = st.selectbox("Column to convert", df.columns.tolist(), key="type_target_col")
    st.write(f"Current dtype: {df[target_col].dtype}")

    target_type = st.selectbox("Convert to", ["numeric", "category", "datetime"], key="type_target_dtype")

    if target_type == "numeric":
        mode = st.selectbox("Numeric conversion mode", ["Simple numeric conversion", "Clean dirty numeric strings first"], key="numeric_mode")
        if st.button("Apply numeric conversion", key="apply_numeric_conversion"):
            new_df = df.copy(deep=True)
            missing_before = int(new_df[target_col].isna().sum())
            if mode == "Simple numeric conversion":
                new_df[target_col] = pd.to_numeric(new_df[target_col], errors="coerce")
            else:
                new_df[target_col] = clean_dirty_numeric(new_df[target_col])
            missing_after = int(new_df[target_col].isna().sum())
            new_nulls = missing_after - missing_before
            changed = commit_change(
                new_df,
                "Data type conversion",
                {"column": target_col, "target_type": "numeric", "mode": mode, "new_nulls": new_nulls},
                [target_col],
                f"Converted '{target_col}' to numeric. New nulls created during conversion: {new_nulls}.",
            )
            if changed:
                st.rerun()

    elif target_type == "category":
        if st.button("Apply category conversion", key="apply_category_conversion"):
            new_df = df.copy(deep=True)
            new_df[target_col] = new_df[target_col].astype("category")
            changed = commit_change(
                new_df,
                "Data type conversion",
                {"column": target_col, "target_type": "category"},
                [target_col],
                f"Converted '{target_col}' to category.",
            )
            if changed:
                st.rerun()

    elif target_type == "datetime":
        mode = st.selectbox("Datetime parsing mode", ["Auto parse", "Use a format"], key="datetime_mode")
        date_format = ""
        if mode == "Use a format":
            date_format = st.text_input("Datetime format", placeholder="%Y-%m-%d", key="datetime_format")
            st.caption("Examples: %Y-%m-%d, %d/%m/%Y, %Y-%m-%d %H:%M:%S")

        if st.button("Apply datetime conversion", key="apply_datetime_conversion"):
            new_df = df.copy(deep=True)
            missing_before = int(new_df[target_col].isna().sum())
            if mode == "Auto parse":
                new_df[target_col] = pd.to_datetime(new_df[target_col], errors="coerce")
            else:
                new_df[target_col] = pd.to_datetime(new_df[target_col], errors="coerce", format=date_format)
            missing_after = int(new_df[target_col].isna().sum())
            new_nulls = missing_after - missing_before
            changed = commit_change(
                new_df,
                "Data type conversion",
                {"column": target_col, "target_type": "datetime", "mode": mode, "format": date_format, "new_nulls": new_nulls},
                [target_col],
                f"Converted '{target_col}' to datetime. New nulls created during parsing: {new_nulls}.",
            )
            if changed:
                st.rerun()



def render_categorical_section(df: pd.DataFrame) -> None:
    st.subheader("4.4 categorical data tools")
    available_cols = df.columns.tolist()
    if not available_cols:
        st.info("No columns available.")
        return

    cat_col = st.selectbox("Text or categorical column", available_cols, key="cat_col")
    action = st.selectbox(
        "Categorical action",
        [
            "Trim whitespace",
            "Convert to lower case",
            "Convert to title case",
            "Replace values using a mapping table",
            "Group rare categories into Other",
            "One-hot encode column",
        ],
        key="cat_action",
    )

    if action in {"Trim whitespace", "Convert to lower case", "Convert to title case"}:
        preview_before = df[cat_col].head(10).tolist()
        st.write("Before preview", preview_before)

        if st.button("Apply text cleanup", key="apply_text_cleanup"):
            new_df = df.copy(deep=True)
            new_series = new_df[cat_col].astype("string")
            if action == "Trim whitespace":
                new_series = new_series.str.strip()
            elif action == "Convert to lower case":
                new_series = new_series.str.lower()
            else:
                new_series = new_series.str.title()
            new_df[cat_col] = new_series
            changed = commit_change(
                new_df,
                "Categorical cleaning",
                {"column": cat_col, "action": action},
                [cat_col],
                f"Applied '{action}' to '{cat_col}'.",
            )
            if changed:
                st.rerun()

    elif action == "Replace values using a mapping table":
        st.caption("Edit the table below. Add old and new values. Rows with blank new values are ignored.")
        mapping_seed = build_mapping_seed(df[cat_col])
        mapping_editor = st.data_editor(mapping_seed, num_rows="dynamic", use_container_width=True, key=f"mapping_editor_{cat_col}")
        set_other = st.checkbox("Set unmatched non-missing values to 'Other'", key="mapping_set_other")

        unique_preview = df[cat_col].dropna().astype("string").value_counts().reset_index().rename(columns={"index": "Value", cat_col: "Count"})
        st.write("Current values preview")
        st.dataframe(unique_preview.head(20), use_container_width=True)

        if st.button("Apply mapping", key="apply_mapping"):
            valid_rows = mapping_editor.dropna(how="all")
            mapping_dict = {}
            for _, row in valid_rows.iterrows():
                old_value = str(row.get("old_value", "")).strip()
                new_value = str(row.get("new_value", "")).strip()
                if old_value and new_value:
                    mapping_dict[old_value] = new_value

            if not mapping_dict:
                st.warning("Please enter at least one valid mapping pair.")
                return

            new_df = df.copy(deep=True)
            original = new_df[cat_col].astype("string")
            missing_mask = new_df[cat_col].isna()
            mapped = original.replace(mapping_dict)
            if set_other:
                matched_keys = list(mapping_dict.keys())
                unmatched_mask = (~original.isin(matched_keys)) & (~missing_mask)
                mapped = mapped.mask(unmatched_mask, "Other")
            mapped = mapped.mask(missing_mask, pd.NA)
            new_df[cat_col] = mapped

            changed = commit_change(
                new_df,
                "Categorical cleaning",
                {"column": cat_col, "action": "mapping", "mapping_rules": mapping_dict, "set_unmatched_to_other": set_other},
                [cat_col],
                f"Applied {len(mapping_dict)} mapping rules to '{cat_col}'.",
            )
            if changed:
                st.rerun()

    elif action == "Group rare categories into Other":
        rare_threshold = st.number_input("Group categories with frequency below", min_value=1, value=5, step=1, key="rare_threshold")
        value_counts = df[cat_col].value_counts(dropna=False)
        rare_values = value_counts[value_counts < rare_threshold].index.tolist()
        st.write("Values that would be grouped", [str(v) for v in rare_values[:20]])

        if st.button("Apply rare category grouping", key="apply_rare_grouping"):
            new_df = df.copy(deep=True)
            missing_mask = new_df[cat_col].isna()
            new_df[cat_col] = new_df[cat_col].where(~new_df[cat_col].isin(rare_values), "Other")
            new_df.loc[missing_mask, cat_col] = pd.NA
            changed = commit_change(
                new_df,
                "Categorical cleaning",
                {"column": cat_col, "action": "rare_grouping", "threshold": int(rare_threshold), "rare_values": [str(v) for v in rare_values]},
                [cat_col],
                f"Grouped rare values in '{cat_col}' into 'Other' using threshold {rare_threshold}.",
            )
            if changed:
                st.rerun()

    elif action == "One-hot encode column":
        drop_first = st.checkbox("Drop first dummy column", key="onehot_drop_first")
        prefix = st.text_input("Prefix for dummy columns", value=cat_col, key="onehot_prefix")

        if st.button("Apply one-hot encoding", key="apply_onehot"):
            new_df = df.copy(deep=True)
            dummies = pd.get_dummies(new_df[cat_col], prefix=prefix, dummy_na=False, drop_first=drop_first)
            new_df = pd.concat([new_df.drop(columns=[cat_col]), dummies], axis=1)
            changed = commit_change(
                new_df,
                "Categorical cleaning",
                {"column": cat_col, "action": "one_hot_encoding", "drop_first": drop_first, "new_columns": dummies.columns.tolist()},
                [cat_col] + dummies.columns.tolist(),
                f"One-hot encoded '{cat_col}' into {dummies.shape[1]} new columns.",
            )
            if changed:
                st.rerun()



def render_numeric_outlier_section(df: pd.DataFrame) -> None:
    st.subheader("4.5 numeric cleaning")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found.")
        return

    num_col = st.selectbox("Numeric column", numeric_cols, key="outlier_col")
    method = st.selectbox("Outlier detection method", ["IQR", "Z-score"], key="outlier_method")

    if method == "IQR":
        series = df[num_col].dropna()
        lower_bound, upper_bound = get_iqr_bounds(series) if not series.empty else (np.nan, np.nan)
        outlier_mask = (df[num_col] < lower_bound) | (df[num_col] > upper_bound)
        st.write(f"Lower bound: {lower_bound:.4f}" if pd.notna(lower_bound) else "Lower bound: n/a")
        st.write(f"Upper bound: {upper_bound:.4f}" if pd.notna(upper_bound) else "Upper bound: n/a")
    else:
        outlier_mask = get_zscore_mask(df[num_col])
        st.write("Using |z-score| > 3.0")

    outlier_count = int(outlier_mask.sum())
    st.metric("Outliers found", outlier_count)
    if outlier_count:
        st.dataframe(df.loc[outlier_mask, [num_col]].head(20), use_container_width=True)

    action = st.selectbox("Action", ["Do nothing", "Cap / winsorize at quantiles", "Remove outlier rows"], key="outlier_action")

    if action == "Cap / winsorize at quantiles":
        lower_q, upper_q = st.slider("Quantile range for capping", min_value=0.0, max_value=1.0, value=(0.01, 0.99), key="winsor_quantiles")
        st.caption("This matches the coursework wording better than clipping directly to IQR bounds.")

        if st.button("Apply outlier capping", key="apply_outlier_capping"):
            if lower_q >= upper_q:
                st.warning("The lower quantile must be smaller than the upper quantile.")
                return
            new_df = df.copy(deep=True)
            lower_cap = new_df[num_col].quantile(lower_q)
            upper_cap = new_df[num_col].quantile(upper_q)
            before_series = new_df[num_col].copy()
            new_df[num_col] = new_df[num_col].clip(lower=lower_cap, upper=upper_cap)
            changed_count = int((before_series.fillna(-999999) != new_df[num_col].fillna(-999999)).sum())
            changed = commit_change(
                new_df,
                "Numeric cleaning",
                {"column": num_col, "action": "winsorize", "detection_method": method, "lower_quantile": lower_q, "upper_quantile": upper_q, "values_changed": changed_count},
                [num_col],
                f"Capped {changed_count} values in '{num_col}' using quantiles {lower_q} to {upper_q}.",
            )
            if changed:
                st.rerun()

    elif action == "Remove outlier rows":
        after_df = df.loc[~outlier_mask].copy(deep=True)
        st.dataframe(compare_shape_table(df, after_df, [num_col]), use_container_width=True)

        if st.button("Remove outlier rows", key="apply_remove_outliers"):
            removed_rows = int(len(df) - len(after_df))
            changed = commit_change(
                after_df,
                "Numeric cleaning",
                {"column": num_col, "action": "remove_outlier_rows", "detection_method": method, "rows_removed": removed_rows},
                [num_col],
                f"Removed {removed_rows} rows flagged as outliers in '{num_col}'.",
            )
            if changed:
                st.rerun()



def render_scaling_section(df: pd.DataFrame) -> None:
    st.subheader("4.6 normalization and scaling")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for scaling.")
        return

    selected_cols = st.multiselect("Numeric columns to scale", numeric_cols, key="scale_columns")
    method = st.selectbox("Scaling method", ["Min-max scaling", "Z-score standardization"], key="scale_method")

    if st.button("Apply scaling", key="apply_scaling"):
        if not selected_cols:
            st.warning("Choose at least one numeric column.")
            return

        new_df = df.copy(deep=True)
        stats_list = []
        changed_cols = []

        for col in selected_cols:
            before_series = new_df[col].copy()
            if method == "Min-max scaling":
                col_min = before_series.min()
                col_max = before_series.max()
                if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                    continue
                new_df[col] = (before_series - col_min) / (col_max - col_min)
            else:
                mean_val = before_series.mean()
                std_val = before_series.std()
                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                    continue
                new_df[col] = (before_series - mean_val) / std_val

            stats_list.append(get_before_after_stats(before_series, new_df[col], col))
            changed_cols.append(col)

        if not changed_cols:
            st.info("No selected columns could be scaled. They may all have constant values or all-missing values.")
            return

        st.session_state.last_scaling_stats = pd.concat(stats_list, ignore_index=True)
        changed = commit_change(
            new_df,
            "Scaling",
            {"method": method, "columns": changed_cols},
            changed_cols,
            f"Applied {method} to {len(changed_cols)} column(s).",
        )
        if changed:
            st.rerun()

    if st.session_state.last_scaling_stats is not None:
        st.write("Before / after stats")
        st.dataframe(st.session_state.last_scaling_stats, use_container_width=True)



def render_column_operations_section(df: pd.DataFrame) -> None:
    st.subheader("4.7 column operations")
    action = st.selectbox(
        "Column operation",
        ["Rename one column", "Drop columns", "Create a new column", "Bin a numeric column"],
        key="column_operation",
    )

    if action == "Rename one column":
        old_name = st.selectbox("Column to rename", df.columns.tolist(), key="rename_old")
        new_name = st.text_input("New column name", key="rename_new")
        if st.button("Apply rename", key="apply_rename"):
            if not new_name.strip():
                st.warning("Enter a new column name.")
            elif new_name in df.columns:
                st.warning("That column name already exists.")
            else:
                new_df = df.rename(columns={old_name: new_name}).copy(deep=True)
                changed = commit_change(
                    new_df,
                    "Column operation",
                    {"action": "rename", "old_name": old_name, "new_name": new_name},
                    [old_name, new_name],
                    f"Renamed '{old_name}' to '{new_name}'.",
                )
                if changed:
                    st.rerun()

    elif action == "Drop columns":
        cols_to_drop = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_columns")
        if st.button("Apply column drop", key="apply_drop_columns"):
            if not cols_to_drop:
                st.warning("Choose at least one column.")
            else:
                new_df = df.drop(columns=cols_to_drop).copy(deep=True)
                changed = commit_change(
                    new_df,
                    "Column operation",
                    {"action": "drop_columns", "columns_dropped": cols_to_drop},
                    cols_to_drop,
                    f"Dropped {len(cols_to_drop)} column(s).",
                )
                if changed:
                    st.rerun()

    elif action == "Create a new column":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns are available for the built-in formulas.")
            return

        new_col = st.text_input("New column name", key="formula_new_col")
        formula = st.selectbox(
            "Formula",
            ["Column A / Column B", "log(Column A)", "Column A - mean(Column A)"],
            key="formula_kind",
        )
        col_a = st.selectbox("Column A", numeric_cols, key="formula_col_a")
        col_b = None
        if formula == "Column A / Column B":
            col_b = st.selectbox("Column B", numeric_cols, key="formula_col_b")

        if st.button("Create new column", key="apply_new_column"):
            if not new_col.strip():
                st.warning("Enter a name for the new column.")
                return
            if new_col in df.columns:
                st.warning("That column name already exists.")
                return

            new_df = df.copy(deep=True)
            if formula == "Column A / Column B":
                denominator = new_df[col_b].replace(0, np.nan)
                new_df[new_col] = new_df[col_a] / denominator
                params = {"action": "create_column", "formula": "A_divided_by_B", "col_a": col_a, "col_b": col_b, "new_column": new_col}
            elif formula == "log(Column A)":
                new_df[new_col] = np.where(new_df[col_a] > 0, np.log(new_df[col_a]), np.nan)
                params = {"action": "create_column", "formula": "log_A", "col_a": col_a, "new_column": new_col}
            else:
                new_df[new_col] = new_df[col_a] - new_df[col_a].mean()
                params = {"action": "create_column", "formula": "A_minus_mean_A", "col_a": col_a, "new_column": new_col}

            changed = commit_change(
                new_df,
                "Column operation",
                params,
                [col_a] + ([col_b] if col_b else []) + [new_col],
                f"Created '{new_col}' using {formula}.",
            )
            if changed:
                st.rerun()

    elif action == "Bin a numeric column":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns are available for binning.")
            return

        source_col = st.selectbox("Numeric column to bin", numeric_cols, key="bin_source_col")
        method = st.selectbox("Binning method", ["Equal-width bins", "Quantile bins"], key="bin_method")
        bin_count = st.number_input("Number of bins", min_value=2, max_value=20, value=4, step=1, key="bin_count")
        new_name = st.text_input("New binned column name", value=f"{source_col}_binned", key="bin_new_name")

        if st.button("Apply binning", key="apply_binning"):
            if not new_name.strip():
                st.warning("Enter a new name for the binned column.")
                return
            if new_name in df.columns:
                st.warning("That column name already exists.")
                return

            new_df = df.copy(deep=True)
            try:
                if method == "Equal-width bins":
                    new_df[new_name] = pd.cut(new_df[source_col], bins=int(bin_count), include_lowest=True)
                else:
                    new_df[new_name] = pd.qcut(new_df[source_col], q=int(bin_count), duplicates="drop")

                changed = commit_change(
                    new_df,
                    "Column operation",
                    {"action": "bin_numeric_column", "source_column": source_col, "method": method, "bins": int(bin_count), "new_column": new_name},
                    [source_col, new_name],
                    f"Created '{new_name}' from '{source_col}' using {method}.",
                )
                if changed:
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not bin this column: {exc}")



def render_validation_section(df: pd.DataFrame) -> None:
    st.subheader("4.8 data validation rules")
    rule = st.selectbox("Validation rule", ["Numeric range check", "Allowed categories list", "Non-null constraint"], key="validation_rule")

    if rule == "Numeric range check":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns found.")
            return
        col = st.selectbox("Numeric column", numeric_cols, key="range_col")
        min_val = st.number_input("Minimum allowed value", value=float(df[col].min()) if df[col].notna().any() else 0.0, key="range_min")
        max_val = st.number_input("Maximum allowed value", value=float(df[col].max()) if df[col].notna().any() else 0.0, key="range_max")

        if st.button("Run numeric range check", key="run_range_check"):
            if min_val > max_val:
                st.warning("Minimum value cannot be larger than maximum value.")
                return
            bad_mask = df[col].notna() & ((df[col] < min_val) | (df[col] > max_val))
            violations = prepare_validation_output(df, bad_mask, rule, col, f"Allowed range: {min_val} to {max_val}")
            st.session_state.last_validation = violations
            st.session_state.last_validation_name = "numeric_range_violations.csv"
            st.metric("Violations found", int(len(violations)))
            if violations.empty:
                st.success("No violations found.")
            else:
                st.dataframe(violations, use_container_width=True)
                st.download_button("Download violations as CSV", as_download_csv(violations), "numeric_range_violations.csv", "text/csv")

    elif rule == "Allowed categories list":
        cat_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
        if not cat_cols:
            st.info("No categorical or text columns found.")
            return
        col = st.selectbox("Categorical column", cat_cols, key="allowed_cat_col")
        allowed_text = st.text_area("Enter one allowed category per line", placeholder="Male\nFemale\nOther", key="allowed_text")

        if st.button("Run allowed categories check", key="run_allowed_check"):
            allowed_values = [line.strip() for line in allowed_text.splitlines() if line.strip()]
            if not allowed_values:
                st.warning("Enter at least one allowed value.")
                return
            bad_mask = df[col].notna() & (~df[col].astype("string").isin(allowed_values))
            violations = prepare_validation_output(df, bad_mask, rule, col, f"Allowed values: {allowed_values}")
            st.session_state.last_validation = violations
            st.session_state.last_validation_name = "allowed_categories_violations.csv"
            st.metric("Violations found", int(len(violations)))
            if violations.empty:
                st.success("No violations found.")
            else:
                st.dataframe(violations, use_container_width=True)
                st.download_button("Download violations as CSV", as_download_csv(violations), "allowed_categories_violations.csv", "text/csv")

    elif rule == "Non-null constraint":
        cols = st.multiselect("Columns that must not be null", df.columns.tolist(), key="nonnull_cols")
        if st.button("Run non-null check", key="run_nonnull_check"):
            if not cols:
                st.warning("Choose at least one column.")
                return
            bad_mask = df[cols].isna().any(axis=1)
            violations = prepare_validation_output(df, bad_mask, rule, ", ".join(cols), "Rows must not contain null values in the selected columns")
            st.session_state.last_validation = violations
            st.session_state.last_validation_name = "nonnull_violations.csv"
            st.metric("Violations found", int(len(violations)))
            if violations.empty:
                st.success("No violations found.")
            else:
                st.dataframe(violations, use_container_width=True)
                st.download_button("Download violations as CSV", as_download_csv(violations), "nonnull_violations.csv", "text/csv")



def render_log_section() -> None:
    st.subheader("6.1 transformation log")
    if st.session_state.log_steps:
        log_df = pd.DataFrame(
            [
                {
                    "time": step["time"],
                    "action": step["action"],
                    "columns": ", ".join(step["columns"]),
                    "params": json.dumps(step["params"], default=str),
                    "before_rows": step["before_shape"]["rows"],
                    "after_rows": step["after_shape"]["rows"],
                    "before_columns": step["before_shape"]["columns"],
                    "after_columns": step["after_shape"]["columns"],
                    "note": step["note"],
                }
                for step in st.session_state.log_steps
            ]
        )
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No transformations have been applied yet.")



def render_cleaning_page() -> None:
    st.header("Page B — Cleaning & Preparation Studio")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        return

    show_feedback()
    render_workflow_controls()

    df = st.session_state.working_df.copy(deep=True)
    st.subheader("current data preview")
    st.dataframe(df.head(15), use_container_width=True)

    with st.expander("Missing values", expanded=True):
        render_missing_values_section(df)
    with st.expander("Duplicates", expanded=False):
        render_duplicates_section(df)
    with st.expander("Data types and parsing", expanded=False):
        render_type_parsing_section(df)
    with st.expander("Categorical tools", expanded=False):
        render_categorical_section(df)
    with st.expander("Numeric cleaning and outliers", expanded=False):
        render_numeric_outlier_section(df)
    with st.expander("Normalization / scaling", expanded=False):
        render_scaling_section(df)
    with st.expander("Column operations", expanded=False):
        render_column_operations_section(df)
    with st.expander("Validation rules", expanded=False):
        render_validation_section(df)
    with st.expander("Transformation log", expanded=False):
        render_log_section()


# Page C
def render_visualization_page() -> None:
    st.header("Page C — Visualization Builder")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        return

    df = st.session_state.working_df.copy(deep=True)
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    st.write(f"Working dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("filters")
    viz_df = df.copy(deep=True)

    f1, f2 = st.columns(2)
    with f1:
        category_filter_col = st.selectbox("Categorical filter column (optional)", ["None"] + categorical_cols, key="viz_cat_filter_col")
        if category_filter_col != "None":
            choices = sorted(viz_df[category_filter_col].dropna().astype("string").unique().tolist())
            selected_choices = st.multiselect("Category values", choices, default=choices, key="viz_cat_filter_values")
            viz_df = viz_df[viz_df[category_filter_col].astype("string").isin(selected_choices)]

    with f2:
        numeric_filter_col = st.selectbox("Numeric range filter column (optional)", ["None"] + numeric_cols, key="viz_num_filter_col")
        if numeric_filter_col != "None":
            series = viz_df[numeric_filter_col].dropna()
            if not series.empty:
                min_val = float(series.min())
                max_val = float(series.max())
                if min_val == max_val:
                    st.info("Selected numeric filter column has the same value in every row.")
                else:
                    selected_range = st.slider("Numeric range", min_val, max_val, (min_val, max_val), key="viz_num_range")
                    viz_df = viz_df[viz_df[numeric_filter_col].between(selected_range[0], selected_range[1])]

    st.write(f"Rows after filters: {len(viz_df)}")
    if viz_df.empty:
        st.warning("No rows remain after filtering.")
        return

    st.subheader("choose your chart")
    plot_type = st.selectbox(
        "Chart type",
        ["Histogram", "Box plot", "Scatter plot", "Line chart", "Bar chart", "Heatmap / Correlation matrix"],
        key="plot_type",
    )

    fig = None

    if plot_type == "Histogram":
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            x_col = st.selectbox("Numeric column", numeric_cols, key="hist_x")
            group_col = st.selectbox("Optional group column", ["None"] + categorical_cols, key="hist_group")
            bins = st.number_input("Number of bins", min_value=5, max_value=100, value=20, step=1, key="hist_bins")
            fig, ax = plt.subplots(figsize=(10, 6))
            if group_col == "None":
                ax.hist(viz_df[x_col].dropna(), bins=int(bins))
            else:
                top_groups = viz_df[group_col].astype("string").value_counts().head(5).index.tolist()
                for group_name in top_groups:
                    values = viz_df.loc[viz_df[group_col].astype("string") == group_name, x_col].dropna()
                    if not values.empty:
                        ax.hist(values, bins=int(bins), alpha=0.6, label=str(group_name))
                ax.legend(title=group_col)
            ax.set_title(f"Histogram of {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Frequency")

    elif plot_type == "Box plot":
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            y_col = st.selectbox("Numeric Y column", numeric_cols, key="box_y")
            x_col = st.selectbox("Optional grouping column", ["None"] + categorical_cols, key="box_x")
            fig, ax = plt.subplots(figsize=(10, 6))
            if x_col == "None":
                ax.boxplot(viz_df[y_col].dropna())
                ax.set_xticks([1])
                ax.set_xticklabels([y_col])
            else:
                top_categories = viz_df[x_col].astype("string").value_counts().head(15).index.tolist()
                data = []
                labels = []
                for category in top_categories:
                    group_series = viz_df.loc[viz_df[x_col].astype("string") == category, y_col].dropna()
                    if not group_series.empty:
                        data.append(group_series)
                        labels.append(category)
                if not data:
                    st.info("No valid grouped data found for the box plot.")
                    fig = None
                else:
                    ax.boxplot(data, labels=labels)
                    plt.xticks(rotation=45, ha="right")
            if fig is not None:
                ax.set_title(f"Box plot of {y_col}")
                ax.set_ylabel(y_col)

    elif plot_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.info("You need at least two numeric columns for a scatter plot.")
        else:
            x_col = st.selectbox("X column", numeric_cols, key="scatter_x")
            y_choices = [col for col in numeric_cols if col != x_col]
            y_col = st.selectbox("Y column", y_choices, key="scatter_y")
            group_col = st.selectbox("Optional group column", ["None"] + categorical_cols, key="scatter_group")
            fig, ax = plt.subplots(figsize=(10, 6))
            if group_col == "None":
                ax.scatter(viz_df[x_col], viz_df[y_col])
            else:
                top_groups = viz_df[group_col].astype("string").value_counts().head(10).index.tolist()
                for group_name in top_groups:
                    group_df = viz_df[viz_df[group_col].astype("string") == group_name]
                    ax.scatter(group_df[x_col], group_df[y_col], label=str(group_name))
                ax.legend(title=group_col)
            ax.set_title(f"{y_col} vs {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

    elif plot_type == "Line chart":
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            x_col = st.selectbox("X column", all_cols, key="line_x")
            aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Median"], key="line_agg")
            group_col = st.selectbox("Optional group column", ["None"] + categorical_cols, key="line_group")
            y_col = None
            if aggregation != "Count":
                y_col = st.selectbox("Y column", numeric_cols, key="line_y")
            line_df = viz_df.copy(deep=True)
            line_df[x_col] = pd.to_datetime(line_df[x_col], errors="coerce")
            if line_df[x_col].isna().all():
                st.warning("The selected X column could not be parsed as datetime, so a time-series line chart is not possible.")
            else:
                line_df = line_df.dropna(subset=[x_col])
                agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count", "Median": "median"}
                fig, ax = plt.subplots(figsize=(10, 6))
                try:
                    if group_col == "None":
                        if aggregation == "Count":
                            plot_data = line_df.groupby(x_col).size().sort_index()
                        else:
                            plot_data = line_df.groupby(x_col)[y_col].agg(agg_map[aggregation]).sort_index()
                        plot_data.plot(ax=ax, marker="o")
                    else:
                        if aggregation == "Count":
                            plot_data = line_df.groupby([x_col, group_col]).size().unstack(fill_value=0).sort_index()
                        else:
                            plot_data = line_df.groupby([x_col, group_col])[y_col].agg(agg_map[aggregation]).unstack().sort_index()
                        plot_data.plot(ax=ax, marker="o")
                    ax.set_title("Line chart")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Count" if aggregation == "Count" else y_col)
                    plt.xticks(rotation=45, ha="right")
                except Exception as exc:
                    st.error(f"Could not build the line chart: {exc}")
                    fig = None

    elif plot_type == "Bar chart":
        x_col = st.selectbox("X column", all_cols, key="bar_x")
        aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Median"], key="bar_agg")
        group_col = st.selectbox("Optional group column", ["None"] + categorical_cols, key="bar_group")
        top_n = st.number_input("Show top N categories", min_value=1, max_value=50, value=10, step=1, key="bar_top_n")
        y_col = None
        if aggregation != "Count":
            if not numeric_cols:
                st.info("No numeric columns found for this aggregation.")
            else:
                y_col = st.selectbox("Y column", numeric_cols, key="bar_y")
        if aggregation == "Count" or y_col is not None:
            agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count", "Median": "median"}
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                if group_col == "None":
                    if aggregation == "Count":
                        plot_data = viz_df.groupby(x_col).size().sort_values(ascending=False).head(int(top_n))
                    else:
                        plot_data = viz_df.groupby(x_col)[y_col].agg(agg_map[aggregation]).sort_values(ascending=False).head(int(top_n))
                    plot_data.plot(kind="bar", ax=ax)
                else:
                    if aggregation == "Count":
                        grouped = viz_df.groupby([x_col, group_col]).size().unstack(fill_value=0)
                    else:
                        grouped = viz_df.groupby([x_col, group_col])[y_col].agg(agg_map[aggregation]).unstack(fill_value=0)
                    top_categories = grouped.sum(axis=1).sort_values(ascending=False).head(int(top_n)).index
                    grouped.loc[top_categories].plot(kind="bar", ax=ax)
                ax.set_title("Bar chart")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Count" if aggregation == "Count" else y_col)
                plt.xticks(rotation=45, ha="right")
            except Exception as exc:
                st.error(f"Could not build the bar chart: {exc}")
                fig = None

    elif plot_type == "Heatmap / Correlation matrix":
        if len(numeric_cols) < 2:
            st.info("Choose a dataset with at least two numeric columns.")
        else:
            selected_cols = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols[: min(6, len(numeric_cols))], key="heatmap_cols")
            if len(selected_cols) < 2:
                st.warning("Choose at least two numeric columns.")
            else:
                corr_df = viz_df[selected_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(10, 8))
                image = ax.imshow(corr_df)
                ax.set_xticks(range(len(corr_df.columns)))
                ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr_df.index)))
                ax.set_yticklabels(corr_df.index)
                for i in range(len(corr_df.index)):
                    for j in range(len(corr_df.columns)):
                        ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center")
                plt.colorbar(image)
                ax.set_title("Correlation matrix")

    if fig is not None:
        st.pyplot(fig)

    st.subheader("filtered data preview")
    st.dataframe(viz_df.head(20), use_container_width=True)


# Page D
def render_export_page() -> None:
    st.header("Page D — Export & Report")
    if st.session_state.working_df is None:
        st.warning("Upload a dataset first.")
        return

    df = st.session_state.working_df.copy(deep=True)
    log_steps = st.session_state.log_steps.copy()

    st.write(f"Final working shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("export cleaned dataset")
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button("Download cleaned dataset as CSV", as_download_csv(df), "cleaned_dataset.csv", "text/csv")
    with export_right:
        st.download_button(
            "Download cleaned dataset as Excel",
            to_excel_bytes(df, sheet_name="cleaned_data"),
            "cleaned_dataset.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("transformation report")
    report_payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": st.session_state.upload_name,
        "starting_dataset_shape": {
            "rows": int(st.session_state.raw_df.shape[0]) if st.session_state.raw_df is not None else None,
            "columns": int(st.session_state.raw_df.shape[1]) if st.session_state.raw_df is not None else None,
        },
        "final_dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "total_steps": len(log_steps),
        "steps": log_steps,
    }

    if log_steps:
        flat_report = pd.DataFrame(
            [
                {
                    "time": step["time"],
                    "action": step["action"],
                    "columns": ", ".join(step["columns"]),
                    "params": json.dumps(step["params"], default=str),
                    "before_shape": json.dumps(step["before_shape"]),
                    "after_shape": json.dumps(step["after_shape"]),
                    "note": step["note"],
                }
                for step in log_steps
            ]
        )
        st.dataframe(flat_report, use_container_width=True)
    else:
        st.info("No transformation steps have been logged yet.")
        flat_report = pd.DataFrame(columns=["time", "action", "columns", "params", "before_shape", "after_shape", "note"])

    report_json = json.dumps(report_payload, indent=4, default=str)
    report_csv = flat_report.to_csv(index=False).encode("utf-8")

    r1, r2 = st.columns(2)
    with r1:
        st.download_button("Download transformation report as JSON", report_json, "transformation_report.json", "application/json")
    with r2:
        st.download_button("Download transformation report as CSV", report_csv, "transformation_report.csv", "text/csv")

    st.subheader("json recipe")
    recipe_payload = {
        "recipe_name": "data_preparation_recipe",
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": log_steps,
    }

    if not log_steps:
        st.warning("No transformation steps have been logged yet. Go to page b and apply at least one cleaning step.")
    else:
        st.success(f"{len(log_steps)} step(s) are ready to export.")

    st.json(recipe_payload)
    st.download_button(
        "Download JSON recipe",
        json.dumps(recipe_payload, indent=4, default=str),
        "transformation_recipe.json",
        "application/json",
    )

    st.subheader("replay snippet")
    replay_code = build_replay_snippet(log_steps)
    st.code(replay_code, language="python")
    st.download_button(
        "Download replay snippet",
        replay_code,
        "replay_recipe.py",
        "text/x-python",
    )

# main app
def main() -> None:
    init_state()

    st.title("Data Wrangler")
    st.write("Upload a dataset, clean it step by step, make charts, and save the final data with the workflow report.")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page",
            [
                "Upload & overview",
                "Cleaning & preparation",
                "Visualization builder",
                "Export & report",
            ],
        )
        if st.session_state.upload_name:
            st.caption(f"Current file: {st.session_state.upload_name}")
        if st.session_state.working_df is not None:
            st.caption(f"Current shape: {st.session_state.working_df.shape[0]} × {st.session_state.working_df.shape[1]}")
            st.caption(f"Logged steps: {len(st.session_state.log_steps)}")

    if page == "Upload & overview":
        render_upload_page()
    elif page == "Cleaning & preparation":
        render_cleaning_page()
    elif page == "Visualization builder":
        render_visualization_page()
    else:
        render_export_page()


if __name__ == "__main__":
    main()
# file_name