"""OCR.md O9's HTTP gate: Mistral's own SDK against /v1/ocr on
`ai -addr 127.0.0.1:18080 -token= -ocr`. Runs in .venv-paddle (mistralai 2.10.1).

    .venv-paddle/bin/python reference/gate_ocr_http.py
"""
import base64, time
from mistralai.client import Mistral

BASE = "http://127.0.0.1:18080"
pdf = open("testdata/ocr/two_pages.pdf", "rb").read()

# Mistral's own SDK, pointed at this server.
client = Mistral(api_key="x", server_url=BASE)
t = time.time()
resp = client.ocr.process(model="mistral-ocr-latest",
                          document={"type": "document_url", "document_url": "data:application/pdf;base64," + base64.b64encode(pdf).decode()},
                          include_image_base64=True, table_format="html")
print("mistral sdk:", f"{time.time()-t:.1f}s", resp.model, len(resp.pages), "pages,", resp.usage_info)
p0, p1 = resp.pages
print("  page 0:", p0.dimensions, [(i.id, i.top_left_x, i.bottom_right_y, (i.image_base64 or "")[:30]) for i in p0.images])
print("  page 0 markdown:", repr(p0.markdown[:160]))
print("  page 1 markdown:", repr(p1.markdown), [(tb.id, tb.content[:60]) for tb in p1.tables])
