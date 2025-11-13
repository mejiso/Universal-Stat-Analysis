import math
import re
from pathlib import Path
import pandas as pd


def _canon(name: str) -> str:  # takes a string
    # lowercase and remove spaces/underscores/hyphens for robust matching
    return re.sub(r"[\s_\-]+", "", str(name or "")).lower()


def _is_participant_id(col: str) -> bool:
    c = _canon(col)
    # match "participantid", "participant", "participant_id", "Participant ID", etc.
    return c == "participantid" or c == "participant"


def normalize_any(
    df: pd.DataFrame,
    id_cols=("Cohort", "ParticipantID"),
    families=(
        # trials regex pattern
        r"(?i)^(?:(?P<prefix>[A-Za-z0-9]{2,})_)?(?:(?P<side>[RLB])_)?Trial[ _]*(?P<trial_id>\d+)$",
        r"(?i)^(?:(?P<side>[RLB])_)?(?:(?P<prefix>[A-Za-z0-9]{2,})_)?Trial[ _]*(?P<trial_id>\d+)$",

        # spgq regex pattern (ex. spgq_any digit)
        r"(?i)^(?P<instrument>spgq)_(?P<subscale>[enb])[ _]*(?P<item>\d+)?$",

        # guess regex pattern (ex. guess_15)
        r"(?i)^(?P<instrument>guess)_(?P<item>\d+)$",
        r"(?i)^(?P<instrument>guess)_(?P<item>15)[ _-]*rev$",

        # --- NEW: generic instrument pattern ---
        # Handles things like: epds_1, xyz_total_3, etc.
        # Requirement: the name must contain at least one digit (so "Age" won't match).
        r"(?i)(?=.*\d)^(?P<instrument>[a-z0-9]+)(?:_(?P<subscale>[a-z]+))?[ _-]*(?P<item>\d+)?$",
    ),
    carry_extra_as_covariates=True,  # carry extra covariates
    drop_ids=True
) -> pd.DataFrame:  # copy into dataframe

    df = df.copy()  # create a new copy of the og df file
    # <-- normalize header whitespace
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)  # gather all columns

    # keep provided id_cols that exist
    id_cols = [c for c in id_cols if c in cols]
    # also auto-add any column that looks like a participant id
    auto_pid_cols = [c for c in cols if _is_participant_id(c)]
    # de-duplicate while preserving order
    for c in auto_pid_cols:
        if c not in id_cols:
            id_cols.append(c)

    frames = []  # create list for frames
    matched_cols = set()  # track matched columns

    for pat in families:  # search for patterns in families and store in variable rx
        rx = re.compile(pat)

        # exclude ID columns from regex matching
        remain = [
            c for c in cols
            if c not in matched_cols and c not in id_cols and not _is_participant_id(c)
        ]

        matches = [(c, rx.match(str(c))) for c in remain]
        matches = [(c, m) for (c, m) in matches if m]
        if not matches:
            continue

        value_cols = [c for (c, _) in matches]
        matched_cols |= set(value_cols)

        long = df[id_cols + value_cols].melt(
            id_vars=id_cols, value_vars=value_cols,
            var_name="_col", value_name="value"
        )

        groups = long["_col"].apply(lambda s: rx.match(str(s)).groupdict())
        grp_df = pd.DataFrame(groups.tolist(), index=long.index)
        long = pd.concat([long.drop(columns=["_col"]), grp_df], axis=1)

        # Cast dtypes
        for k in ["trial_id", "item", "subscale", "side", "prefix", "instrument"]:
            if k in long.columns:
                long[k] = long[k].astype("string")

        frames.append(long)

    if not frames:
        raise ValueError(
            "No columns matched any family. Adjust regex or headers."
        )

    tidy = pd.concat(frames, ignore_index=True)

    # ---- Cleanup ----
    for c in id_cols:
        tidy[c] = tidy[c].astype("category")

    if "side" in tidy.columns:
        tidy["side"] = tidy["side"].str.strip().str.upper()
        tidy.loc[~tidy["side"].isin(["R", "L", "B"]), "side"] = pd.NA
        tidy["side"] = tidy["side"].astype("string")

    if "instrument" in tidy.columns:
        tidy["instrument"] = tidy["instrument"].str.lower()

    if "instrument" in tidy.columns and "subscale" in tidy.columns:
        mask_spgq = tidy["instrument"].eq("spgq")
        sub_map = {"e": "empathy", "n": "negative_feelings", "b": "behavioral"}
        tidy.loc[mask_spgq, "subscale"] = (
            tidy.loc[mask_spgq, "subscale"].str.lower().map(
                sub_map).astype("string")
        )

    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])

    # Carry through covariates
    extra_cols = [
        c for c in cols
        if c not in set(id_cols) | matched_cols and not _is_participant_id(c)
    ]
    if carry_extra_as_covariates and extra_cols:
        tidy = tidy.merge(df[id_cols + extra_cols],
                          on=list(id_cols), how="left")

    # ---- Label builder ----
    def _safe_get(row, key):
        if key not in row.index:
            return ""
        v = row[key]
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    def _label(row):
        # Trials first
        trial_id = _safe_get(row, "trial_id").strip()
        if trial_id:
            pr = _safe_get(row, "prefix").strip()
            sd = _safe_get(row, "side").strip()
            base = "Trial_" + trial_id
            parts = [p for p in (pr, sd, base) if p]
            return "_".join(parts)

        inst = _safe_get(row, "instrument").strip().lower()
        sub = _safe_get(row, "subscale").strip()
        item = _safe_get(row, "item").strip()

        # SPGQ: use nice subscale names already mapped above
        if inst == "spgq":
            if sub and item:
                return f"{inst}_{sub}_{item}"
            elif sub:
                return f"{inst}_{sub}"
            return inst or "measure"

        # GUESS: keep the existing behavior
        if inst == "guess":
            return f"{inst}_{item}" if item else inst or "measure"

        # --- NEW: generic instrument labeling ---
        # For anything else (e.g., epds_1, xyz_total_3)
        if inst:
            parts = [inst]
            if sub:
                parts.append(sub)
            if item:
                parts.append(item)
            return "_".join(parts)

        return "measure"

    tidy["measure_label"] = tidy.apply(_label, axis=1).astype("string")

    drop_these = []
    if drop_ids:
        drop_these = [c for c in tidy.columns if _is_participant_id(c)]
    if drop_these:
        tidy = tidy.drop(columns=drop_these, errors="ignore")

    return tidy


def _read_excel_any(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python normalize.py <path_to_excel>")
        sys.exit(1)

    input_path = sys.argv[1]
    df = _read_excel_any(input_path)

    norm = normalize_any(df)

    print("Preview of tidy data:")
    with pd.option_context(
        "display.max_rows", 30,
        "display.max_columns", 0,
        "display.width", 200
    ):
        print(norm.head(30).to_string(index=False))

    p = Path(input_path)
    out_path = str(p.with_name(p.stem + "_tidy.csv"))
    norm.to_csv(out_path, index=False)
    print(f"\nNormalized data saved to: {out_path}")
