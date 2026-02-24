"""01_table1_agg.py

Aggregate per-site table1.json files into:
  - table1_by_site.csv / .json  — wide comparison across sites
  - table1_aggregated.csv / .json — pooled across all sites

Usage (from repo root):
    python code/agg/01_table1_agg.py
"""
import json
import math
import statistics
from glob import glob
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

# ── Constants ────────────────────────────────────────────────────────────────

GROUPS = [
    "hfno_after_extubation",
    "low_flow_after_extubation",
    "overall",
    "difference_95ci",
]

PCT_TOL = 0.15  # max allowed deviation between stored pct and n/N*100

FOOTNOTE_ROWS = [
    "---",
    "* Continuous (mean/SD): pooled weighted combined variance formula",
    "† Categorical (n/%): counts summed; proportions recalculated from pooled group N",
    "‡ Median [IQR]: median-of-medians across sites",
    "§ Difference 95% CI: Welch's t-test (continuous); Wald CI (proportions); not calculated (median)",
]


# ── Discovery ────────────────────────────────────────────────────────────────


def discover_sites(clif_dir: Path) -> dict:
    """Glob all table1.json files and return {site_name: data}."""
    pattern = str(clif_dir / "*/output_to_share/table1.json")
    paths = sorted(glob(pattern))
    sites = {}
    for p in paths:
        site_name = Path(p).parent.parent.name
        with open(p) as f:
            data = json.load(f)
        sites[site_name] = data
        n_h = data["N"]["hfno_after_extubation"]
        n_lf = data["N"]["low_flow_after_extubation"]
        print(f"  {site_name}: N_hfno={n_h}, N_lf={n_lf}")
    return sites


# ── Type detection ────────────────────────────────────────────────────────────


def detect_type(val) -> str:
    """Return 'count' | 'mean_sd' | 'n_pct' | 'median_iqr' | 'null'."""
    if val is None:
        return "null"
    if isinstance(val, (int, float)):
        return "count"
    if isinstance(val, dict):
        if "mean" in val and "sd" in val:
            return "mean_sd"
        if "n" in val and "pct" in val:
            return "n_pct"
        if "median" in val:
            return "median_iqr"
    return "null"


# ── Formatting ────────────────────────────────────────────────────────────────


def format_group_val(val) -> str:
    """Format a non-diff group value as a display string."""
    t = detect_type(val)
    if t == "null":
        return "N/A"
    if t == "count":
        return str(int(val))
    if t == "mean_sd":
        return f"{val['mean']:.1f} ({val['sd']:.1f})"
    if t == "n_pct":
        return f"{val['n']} ({val['pct']:.1f}%)"
    if t == "median_iqr":
        return f"{val['median']:.1f} [{val['q1']:.1f}, {val['q3']:.1f}]"
    return "N/A"


def format_diff_val(val) -> str:
    """Format a difference_95ci value as a display string."""
    if val is None:
        return "N/A"
    if not isinstance(val, dict):
        return "N/A"

    def _sign(v):
        return f"{'+' if v >= 0 else ''}{v:.1f}"

    if "diff_pct" in val:
        d = val["diff_pct"]
        lo = val.get("ci_low_pct")
        hi = val.get("ci_high_pct")
        pct_str = f"{'+' if d >= 0 else ''}{d:.1f}%"
        if lo is not None and hi is not None:
            return f"{pct_str} ({_sign(lo)} to {_sign(hi)})"
        return pct_str

    if "diff" in val:
        d = val["diff"]
        lo = val.get("ci_low")
        hi = val.get("ci_high")
        if lo is not None and hi is not None:
            return f"{_sign(d)} ({_sign(lo)} to {_sign(hi)})"
        return _sign(d)

    return "N/A"


_STAT_MARKERS = {"mean_sd": "*", "n_pct": "†", "median_iqr": "‡"}


def _row_markers(val_type: str, grp_results: dict) -> str:
    """Return footnote superscripts to append to a variable label."""
    stat = _STAT_MARKERS.get(val_type, "")
    diff = "§" if grp_results.get("difference_95ci") is not None else ""
    return stat + diff


# ── Pooling helpers ──────────────────────────────────────────────────────────


def pool_mean_sd(stats_list: list, ns: list) -> dict | None:
    """Weighted pooled mean and SD using combined-groups variance formula."""
    valid = [(s, n) for s, n in zip(stats_list, ns) if s is not None and n and n > 0]
    if not valid:
        return None
    total_n = sum(n for _, n in valid)
    if total_n == 0:
        return None
    pooled_mean = sum(s["mean"] * n for s, n in valid) / total_n
    numerator = sum(
        (n - 1) * s["sd"] ** 2 + n * (s["mean"] - pooled_mean) ** 2
        for s, n in valid
    )
    denom = total_n - 1
    pooled_sd = math.sqrt(numerator / denom) if denom > 0 else 0.0
    return {"mean": round(pooled_mean, 1), "sd": round(pooled_sd, 1)}


def pool_n_pct(stats_list: list, total_n: int) -> dict | None:
    """Sum n across sites and recalculate pct from total_n denominator."""
    valid = [s for s in stats_list if s is not None]
    if not valid or total_n == 0:
        return None
    pooled_n = sum(s["n"] for s in valid)
    return {"n": pooled_n, "pct": round(pooled_n / total_n * 100, 1)}


def compute_diff_ci_mean(hfno_s, hfno_n, lf_s, lf_n) -> dict | None:
    """Welch's t-interval for mean difference (HFNO - LF) from pooled stats."""
    if hfno_s is None or lf_s is None:
        return None
    n_a, n_b = hfno_n, lf_n
    if n_a < 2 or n_b < 2:
        return None
    mean_a, sd_a = hfno_s["mean"], hfno_s["sd"]
    mean_b, sd_b = lf_s["mean"], lf_s["sd"]
    diff = mean_a - mean_b
    var_a, var_b = sd_a ** 2, sd_b ** 2
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return {"diff": round(diff, 1)}
    df_ws = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = scipy_stats.t.ppf(0.975, df_ws)
    return {
        "diff": round(diff, 1),
        "ci_low": round(diff - t_crit * se, 1),
        "ci_high": round(diff + t_crit * se, 1),
    }


def compute_diff_ci_pct(hfno_s, hfno_n, lf_s, lf_n) -> dict | None:
    """Wald CI for proportion difference (HFNO - LF) from pooled n/pct stats."""
    if hfno_s is None or lf_s is None:
        return None
    if hfno_n == 0 or lf_n == 0:
        return None
    p_a = hfno_s["n"] / hfno_n
    p_b = lf_s["n"] / lf_n
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / hfno_n + p_b * (1 - p_b) / lf_n)
    return {
        "diff_pct": round(diff * 100, 1),
        "ci_low_pct": round((diff - 1.96 * se) * 100, 1),
        "ci_high_pct": round((diff + 1.96 * se) * 100, 1),
    }


# ── Validation ───────────────────────────────────────────────────────────────


def validate_sites(sites: dict) -> list[str]:
    """Run per-site integrity checks; return list of warning strings (empty = pass)."""
    warns = []
    data_groups = ("hfno_after_extubation", "low_flow_after_extubation", "overall")

    for site, data in sites.items():
        n_vals = data.get("N", {})
        n_hfno = n_vals.get("hfno_after_extubation")
        n_lf = n_vals.get("low_flow_after_extubation")
        n_overall = n_vals.get("overall")

        # Check 1: N_hfno + N_lf == N_overall
        if all(v is not None for v in (n_hfno, n_lf, n_overall)):
            if n_hfno + n_lf != n_overall:
                warns.append(
                    f"{site}: N_hfno({n_hfno}) + N_lf({n_lf}) = "
                    f"{n_hfno + n_lf} ≠ N_overall({n_overall})"
                )

        # Checks 2 & 3: per n_pct variable
        for var, var_data in data.items():
            if var in ("site", "N") or not isinstance(var_data, dict):
                continue

            for grp in data_groups:
                grp_val = var_data.get(grp)
                if detect_type(grp_val) != "n_pct":
                    continue
                grp_n = n_vals.get(grp)
                if not grp_n:
                    continue
                # Check 2: |n / group_N * 100 - pct| < PCT_TOL
                computed_pct = grp_val["n"] / grp_n * 100
                dev = abs(computed_pct - grp_val["pct"])
                if dev >= PCT_TOL:
                    warns.append(
                        f"{site}: {var!r} [{grp}]: "
                        f"n/N*100={computed_pct:.3f} stored pct={grp_val['pct']:.1f} "
                        f"(diff={dev:.3f})"
                    )

            # Check 3: n_hfno + n_lf == n_overall when all three groups present
            hv = var_data.get("hfno_after_extubation")
            lv = var_data.get("low_flow_after_extubation")
            ov = var_data.get("overall")
            if all(detect_type(v) == "n_pct" for v in (hv, lv, ov)):
                n_sum = hv["n"] + lv["n"]
                if n_sum != ov["n"]:
                    warns.append(
                        f"{site}: {var!r}: n_hfno({hv['n']}) + "
                        f"n_lf({lv['n']}) = {n_sum} ≠ n_overall({ov['n']})"
                    )

        # Check 4: _icu counts sum to group_N (only when all _icu keys are present)
        icu_keys = [k for k in data if "_icu" in k]
        if icu_keys:
            for grp in data_groups:
                grp_n = n_vals.get(grp)
                if grp_n is None:
                    continue
                icu_ns = []
                all_present = True
                for k in icu_keys:
                    kv = data[k]
                    if not isinstance(kv, dict):
                        all_present = False
                        break
                    gv = kv.get(grp)
                    if detect_type(gv) != "n_pct":
                        all_present = False
                        break
                    icu_ns.append(gv["n"])
                if all_present and sum(icu_ns) != grp_n:
                    warns.append(
                        f"{site}: ICU types [{grp}]: sum={sum(icu_ns)} ≠ N={grp_n}"
                    )

    return warns


def validate_aggregated(agg_numeric: dict, pooled_n: dict) -> list[str]:
    """Run integrity checks on pooled aggregated data; return list of warning strings."""
    warns = []
    data_groups = ("hfno_after_extubation", "low_flow_after_extubation", "overall")

    # Check 1: pooled_N_hfno + pooled_N_lf == pooled_N_overall
    nh = pooled_n.get("hfno_after_extubation")
    nl = pooled_n.get("low_flow_after_extubation")
    no = pooled_n.get("overall")
    if all(v is not None for v in (nh, nl, no)):
        if nh + nl != no:
            warns.append(
                f"pooled_N_hfno({nh}) + pooled_N_lf({nl}) = "
                f"{nh + nl} ≠ pooled_N_overall({no})"
            )

    for var, grp_results in agg_numeric.items():
        if var == "N":
            continue

        for grp in data_groups:
            grp_val = grp_results.get(grp)
            if detect_type(grp_val) != "n_pct":
                continue
            grp_n = pooled_n.get(grp)
            if not grp_n:
                continue
            # Check 2: |n / pooled_N * 100 - pct| < PCT_TOL
            computed_pct = grp_val["n"] / grp_n * 100
            dev = abs(computed_pct - grp_val["pct"])
            if dev >= PCT_TOL:
                warns.append(
                    f"{var!r} [{grp}]: n/N*100={computed_pct:.3f} "
                    f"stored pct={grp_val['pct']:.1f} (diff={dev:.3f})"
                )

        # Check 3: n_hfno + n_lf == n_overall when all three groups present
        hv = grp_results.get("hfno_after_extubation")
        lv = grp_results.get("low_flow_after_extubation")
        ov = grp_results.get("overall")
        if all(detect_type(v) == "n_pct" for v in (hv, lv, ov)):
            n_sum = hv["n"] + lv["n"]
            if n_sum != ov["n"]:
                warns.append(
                    f"{var!r}: n_hfno({hv['n']}) + n_lf({lv['n']}) = "
                    f"{n_sum} ≠ n_overall({ov['n']})"
                )

    return warns


# ── By-site table ────────────────────────────────────────────────────────────


def build_by_site(sites: dict) -> tuple:
    """Build wide DataFrame (one row per variable, three columns per site)
    and a nested dict for JSON output. Columns are grouped by stat type:
    all-sites hfno → all-sites lf → all-sites overall (difference_95ci excluded).
    """
    # Union of all variable keys, preserving first-seen order
    all_vars = []
    seen = set()
    for data in sites.values():
        for k in data:
            if k != "site" and k not in seen:
                all_vars.append(k)
                seen.add(k)

    sorted_sites = sorted(sites.keys())
    by_site_groups = [g for g in GROUPS if g != "difference_95ci"]

    rows = []
    nested = {}

    for var in all_vars:
        row = {"variable": var}
        nested[var] = {}
        for site in sorted_sites:
            var_data = sites[site].get(var)
            nested[var][site] = {}
            for grp in by_site_groups:
                val = var_data.get(grp) if var_data is not None else None
                formatted = format_group_val(val)
                row[f"{site}_{grp}"] = formatted
                nested[var][site][grp] = formatted
        rows.append(row)

    # Footnote rows
    for fn_text in FOOTNOTE_ROWS:
        row = {"variable": fn_text}
        for site in sorted_sites:
            for grp in by_site_groups:
                row[f"{site}_{grp}"] = ""
        rows.append(row)

    # Group-first column order: all hfno cols → all lf cols → all overall cols
    cols = ["variable"] + [
        f"{site}_{grp}" for grp in by_site_groups for site in sorted_sites
    ]
    return pd.DataFrame(rows, columns=cols), nested


# ── Aggregated table ─────────────────────────────────────────────────────────


def _infer_var_type(var: str, sites: dict, site_list: list) -> str:
    """Infer stat type for a variable by examining non-null group values."""
    for site in site_list:
        vd = sites[site].get(var)
        if vd is None:
            continue
        for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
            t = detect_type(vd.get(grp))
            if t != "null":
                return t
    return "null"


def build_aggregated(sites: dict) -> tuple:
    """Build standard pooled DataFrame, excluding ICU-type variables."""
    # Variable order: union across sites, skip 'site' key and ICU-type rows
    all_vars = []
    seen = set()
    for data in sites.values():
        for k in data:
            if k == "site" or "_icu" in k:
                continue
            if k not in seen:
                all_vars.append(k)
                seen.add(k)

    site_list = list(sites.keys())

    # Per-group N denominators from each site's "N" entry
    group_ns = {
        grp: {site: sites[site]["N"][grp] for site in site_list}
        for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall")
    }
    pooled_n = {grp: sum(group_ns[grp].values()) for grp in group_ns}

    rows = []
    agg_numeric = {}  # structured results for JSON
    var_labels: dict[str, str] = {}  # original var -> annotated label

    for var in all_vars:
        val_type = _infer_var_type(var, sites, site_list)
        grp_results = {}

        if var == "N":
            for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
                grp_results[grp] = sum(
                    sites[s]["N"][grp]
                    for s in site_list
                    if isinstance(sites[s]["N"].get(grp), (int, float))
                )
            grp_results["difference_95ci"] = None

        elif val_type == "mean_sd":
            for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
                stat_list, ns = [], []
                for site in site_list:
                    vd = sites[site].get(var)
                    if vd is None:
                        continue
                    s_val = vd.get(grp)
                    if detect_type(s_val) == "mean_sd":
                        stat_list.append(s_val)
                        ns.append(group_ns[grp][site])
                grp_results[grp] = pool_mean_sd(stat_list, ns)
            grp_results["difference_95ci"] = compute_diff_ci_mean(
                grp_results["hfno_after_extubation"],
                pooled_n["hfno_after_extubation"],
                grp_results["low_flow_after_extubation"],
                pooled_n["low_flow_after_extubation"],
            )

        elif val_type == "n_pct":
            for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
                stat_list = []
                for site in site_list:
                    vd = sites[site].get(var)
                    if vd is None:
                        continue
                    s_val = vd.get(grp)
                    if detect_type(s_val) == "n_pct":
                        stat_list.append(s_val)
                grp_results[grp] = pool_n_pct(stat_list, pooled_n[grp])
            grp_results["difference_95ci"] = compute_diff_ci_pct(
                grp_results["hfno_after_extubation"],
                pooled_n["hfno_after_extubation"],
                grp_results["low_flow_after_extubation"],
                pooled_n["low_flow_after_extubation"],
            )

        elif val_type == "median_iqr":
            for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
                medians, q1s, q3s = [], [], []
                for site in site_list:
                    vd = sites[site].get(var)
                    if vd is None:
                        continue
                    s_val = vd.get(grp)
                    if detect_type(s_val) == "median_iqr":
                        medians.append(s_val["median"])
                        q1s.append(s_val["q1"])
                        q3s.append(s_val["q3"])
                if medians:
                    grp_results[grp] = {
                        "median": round(statistics.median(medians), 1),
                        "q1": round(statistics.median(q1s), 1),
                        "q3": round(statistics.median(q3s), 1),
                    }
                else:
                    grp_results[grp] = None
            # Median-of-medians diff; CI not calculated from summaries
            h = grp_results.get("hfno_after_extubation")
            lf = grp_results.get("low_flow_after_extubation")
            if h is not None and lf is not None:
                grp_results["difference_95ci"] = {
                    "diff": round(h["median"] - lf["median"], 1)
                }
            else:
                grp_results["difference_95ci"] = None

        else:
            # null type: HFNO-only variables with no poolable low_flow data
            for grp in GROUPS:
                grp_results[grp] = None

        agg_numeric[var] = grp_results

        markers = _row_markers(val_type, grp_results)
        label = f"{var} {markers}" if markers else var
        var_labels[var] = label

        row = {"variable": label}
        for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall"):
            row[grp] = format_group_val(grp_results.get(grp))
        row["difference_95ci"] = format_diff_val(grp_results.get("difference_95ci"))
        rows.append(row)

    # Footnote rows
    for fn_text in FOOTNOTE_ROWS:
        rows.append({
            "variable": fn_text,
            "hfno_after_extubation": "",
            "low_flow_after_extubation": "",
            "overall": "",
            "difference_95ci": "",
        })

    cols = [
        "variable",
        "hfno_after_extubation",
        "low_flow_after_extubation",
        "overall",
        "difference_95ci",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Build formatted JSON (same structure as CSV but as nested dict)
    agg_json = {
        var_labels[var]: {
            grp: (
                format_diff_val(agg_numeric[var].get("difference_95ci"))
                if grp == "difference_95ci"
                else format_group_val(agg_numeric[var].get(grp))
            )
            for grp in GROUPS
        }
        for var in all_vars
    }

    return df, agg_json, agg_numeric


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    repo_root = Path(__file__).parent.parent.parent
    clif_dir = repo_root / "CLIF-HFNO-Extubation-Risk"
    output_dir = repo_root / "output_of_agg"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering sites...")
    sites = discover_sites(clif_dir)
    print(f"Found {len(sites)} sites: {sorted(sites.keys())}\n")

    print("Validating site data...")
    site_warns = validate_sites(sites)
    if site_warns:
        for w in site_warns:
            print(f"  WARNING: {w}")
    else:
        print("  All site checks passed.\n")

    print("Building by-site table...")
    by_site_df, by_site_json = build_by_site(sites)
    by_site_df.to_csv(output_dir / "table1_by_site.csv", index=False)
    with open(output_dir / "table1_by_site.json", "w") as f:
        json.dump(by_site_json, f, indent=2)
    print(
        f"  table1_by_site.csv   — {len(by_site_df)} rows, "
        f"{len(by_site_df.columns)} columns"
    )
    print(f"  table1_by_site.json\n")

    print("Building aggregated table...")
    agg_df, agg_json, agg_numeric = build_aggregated(sites)
    agg_df.to_csv(output_dir / "table1_aggregated.csv", index=False)
    with open(output_dir / "table1_aggregated.json", "w") as f:
        json.dump(agg_json, f, indent=2)
    print(
        f"  table1_aggregated.csv  — {len(agg_df)} rows "
        f"(excl. ICU-type variables)"
    )
    print(f"  table1_aggregated.json\n")

    pooled_n = {
        grp: sum(sites[s]["N"][grp] for s in sites)
        for grp in ("hfno_after_extubation", "low_flow_after_extubation", "overall")
    }
    print("Validating aggregated data...")
    agg_warns = validate_aggregated(agg_numeric, pooled_n)
    if agg_warns:
        for w in agg_warns:
            print(f"  WARNING: {w}")
    else:
        print("  All aggregated checks passed.\n")

    # Verification
    pooled_hfno_n = sum(d["N"]["hfno_after_extubation"] for d in sites.values())
    pooled_lf_n = sum(d["N"]["low_flow_after_extubation"] for d in sites.values())
    print("Verification:")
    print(f"  Pooled HFNO N = {pooled_hfno_n}")
    print(f"  Pooled LF N   = {pooled_lf_n}")
    print(f"  Pooled Total  = {pooled_hfno_n + pooled_lf_n}")

    n_row = agg_df[agg_df["variable"] == "N"]
    if not n_row.empty:
        r = n_row.iloc[0]
        print(
            f"  Aggregated N row: "
            f"hfno={r['hfno_after_extubation']}, "
            f"lf={r['low_flow_after_extubation']}, "
            f"overall={r['overall']}"
        )

    print(f"\nOutput written to: {output_dir}")


if __name__ == "__main__":
    main()
