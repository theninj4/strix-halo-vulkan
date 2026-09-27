"""OCR.md O9's HTTP gate: PaddleX-style /layout-parsing requests and Mistral's own SDK
against `ai -addr 127.0.0.1:18080 -token= -ocr`. Runs in .venv-paddle (mistralai 2.10.1).

    .venv-paddle/bin/python reference/gate_ocr_http.py
"""
import base64, json, time, requests
from mistralai.client import Mistral

BASE = "http://127.0.0.1:18080"
orc = json.load(open("reference/out/ocr_page/page/record.json"))
png = open("testdata/ocr/paddleocr_vl_demo.png", "rb").read()
pdf = open("testdata/ocr/two_pages.pdf", "rb").read()

# PaddleX's documented client: POST /layout-parsing with the file base64.
t = time.time()
r = requests.post(BASE + "/layout-parsing", json={"file": base64.b64encode(png).decode(), "fileType": 1}).json()
res = r["result"]["layoutParsingResults"][0]
print("layout-parsing image:", r["errorCode"], r["errorMsg"], f"{time.time()-t:.1f}s",
      "markdown == PaddleX:", res["markdown"]["text"] == orc["markdown"],
      "images:", sorted(res["markdown"]["images"]) == orc["markdown_images"],
      "blocks == PaddleX:", res["prunedResult"]["parsing_res_list"] == orc["pruned"]["parsing_res_list"],
      "dataInfo:", r["result"]["dataInfo"])
t = time.time()
r = requests.post(BASE + "/layout-parsing", json={"file": base64.b64encode(pdf).decode(), "fileType": 0}).json()
print("layout-parsing pdf:", r["errorCode"], f"{time.time()-t:.1f}s", r["result"]["dataInfo"],
      [x["prunedResult"]["page_count"] for x in r["result"]["layoutParsingResults"]])
r = requests.post(BASE + "/layout-parsing", json={"file": base64.b64encode(png).decode(), "layoutShapeMode": "poly"})
print("refusal:", r.status_code, r.json())

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
