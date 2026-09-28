"""An OmniDocBench subset for OCR.md stage O10: every Nth page per data source.

Sorts OmniDocBench.json's pages by (data_source, image_path) and takes every
--every'th, so each of the ten sources keeps its share. Writes the image list
(one a line, for cmd/ocr -list and omnidocbench_paddlex.py --list) and the
ground truth cut to those pages (the evaluator scores every GT page).

    python3 reference/omnidocbench_subset.py --gt ~/repos/omnidocbench-data/ds/OmniDocBench.json \\
        --every 5 --out ~/repos/omnidocbench-data/subset5
"""

import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pages = json.load(open(args.gt))
    key = lambda p: (p["page_info"]["page_attribute"]["data_source"], p["page_info"]["image_path"])
    pages.sort(key=key)
    sub = pages[:: args.every]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "images.txt"), "w") as f:
        for p in sub:
            f.write(p["page_info"]["image_path"] + "\n")
    json.dump(sub, open(os.path.join(args.out, "gt.json"), "w"), ensure_ascii=False)
    print(len(sub), "pages", dict(collections.Counter(key(p)[0] for p in sub)))


if __name__ == "__main__":
    main()
