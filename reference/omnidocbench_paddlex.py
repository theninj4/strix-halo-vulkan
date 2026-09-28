"""PaddleOCR's own page pipeline over OmniDocBench pages, for OCR.md stage O10.

Runs in .venv-paddle (paddleocr 3.7.0, paddlex 3.7.2): PaddleOCRVL at
pipeline v1.6, its layout model on the CPU (Paddle), every recognition sent
to our `serve -ocr` over the vllm-server backend, exactly as O6's gate runs
`doc_parser`. Each page's markdown is saved with pretty=False, as
OmniDocBench's tools/model_infer/PaddleOCR_img2md.py saves it, to
<out>/<stem>.md. Pages already written are skipped, so a run resumes.

    .venv-paddle/bin/python reference/omnidocbench_paddlex.py \\
        --images ~/repos/omnidocbench-data/ds/images --list LIST \\
        --shape rect --out OUT --server http://127.0.0.1:18080/v1

--shape is layout_shape_mode: rect (what the Go pipeline ports, O8a) or auto
(PaddleX's default, polygons: O8b).
"""

import argparse
import os
import time
import traceback

from paddleocr import PaddleOCRVL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--list", help="file naming the images to take, one a line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", default="rect", choices=["rect", "auto"])
    ap.add_argument("--server", default="http://127.0.0.1:18080/v1")
    args = ap.parse_args()

    if args.list:
        names = [l.strip() for l in open(args.list) if l.strip()]
    else:
        names = sorted(os.listdir(args.images))
    os.makedirs(args.out, exist_ok=True)
    pipe = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                       vl_rec_server_url=args.server)
    tsv = open(os.path.join(args.out, "timings.tsv"), "a")
    failed = open(os.path.join(args.out, "failed.tsv"), "a")
    start = time.time()
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        md = os.path.join(args.out, stem + ".md")
        if os.path.exists(md):
            continue
        t0 = time.time()
        try:
            for res in pipe.predict(os.path.join(args.images, name), layout_shape_mode=args.shape,
                                    use_doc_orientation_classify=False, use_doc_unwarping=False):
                res.save_to_markdown(md, pretty=False)
            tsv.write(f"{name}\t{time.time() - t0:.3f}\n")
            tsv.flush()
        except Exception as e:
            failed.write(f"{name}\t{e!r}\n")
            failed.flush()
            traceback.print_exc()
        print(f"{i + 1}/{len(names)} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
