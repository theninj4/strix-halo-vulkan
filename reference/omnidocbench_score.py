"""The leaderboard row from OmniDocBench result folders, for OCR.md stage O10.

Reads each <dir>/*_metric_result.json that reference/omnidocbench_eval.sh
wrote and prints the README's columns, taking the fields the benchmark's own
skills/scripts/parse_results.py takes (page averages; CDM and TEDS per page),
and Overall = ((1 - text edit) * 100 + TEDS + CDM) / 3.

    python3 reference/omnidocbench_score.py EVAL_DIR [EVAL_DIR ...]
"""

import glob
import json
import os
import sys


def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    cols = ["Overall", "Text Edit", "Formula CDM", "Table TEDS", "TEDS-S", "Read Order", "Formula Edit", "Table Edit"]
    print("| arm | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for d in sys.argv[1:]:
        m = json.load(open(glob.glob(os.path.join(d, "*_metric_result.json"))[0]))
        text = get(m, ["text_block", "all", "Edit_dist", "ALL_page_avg"])
        cdm = get(m, ["display_formula", "page", "CDM", "ALL"])
        teds = get(m, ["table", "page", "TEDS", "ALL"])
        tedss = get(m, ["table", "page", "TEDS_structure_only", "ALL"])
        ro = get(m, ["reading_order", "all", "Edit_dist", "ALL_page_avg"])
        fe = get(m, ["display_formula", "all", "Edit_dist", "ALL_page_avg"])
        te = get(m, ["table", "all", "Edit_dist", "ALL_page_avg"])
        overall = None
        if None not in (text, cdm, teds):
            overall = ((1 - text) * 100 + cdm * 100 + teds * 100) / 3
        f = lambda v, s=1, n=4: "-" if not isinstance(v, (int, float)) else f"{v * s:.{n}f}"
        row = [f(overall, 1, 2), f(text), f(cdm, 100, 2), f(teds, 100, 2), f(tedss, 100, 2), f(ro), f(fe), f(te)]
        print(f"| {os.path.basename(os.path.normpath(d))} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
