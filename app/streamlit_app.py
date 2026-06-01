"""Read-only results dashboard. Launch: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HalluBench", layout="wide")
st.title("HalluBench | Evidence-grounded factuality benchmark")
st.caption("This dashboard reads persisted outputs only; absent results are not replaced by samples.")
base = Path(st.sidebar.text_input("Results directory", "results/run"))
if not (base/"summary.json").exists():
    st.warning("No completed run available at this location. Run the CLI benchmark first.")
    st.stop()
summary = json.loads((base/"summary.json").read_text())
manifest = json.loads((base/"experiment.json").read_text())
quality = json.loads((base/"dataset_quality.json").read_text())
st.subheader("Overview")
a, b, c, d = st.columns(4)
a.metric("Held-out examples", manifest["n_test"])
b.metric("Detectors", len(summary["detectors"]))
c.metric("Run status", manifest["status"])
d.metric("Missing evidence", quality["missing_evidence"])
st.write("Datasets / splits", quality["datasets"], quality["splits"])
st.subheader("Detector comparison")
rows = []
for name, block in summary["detectors"].items():
    metric = block["aggregate"]["metrics"] or {}
    rows.append({"detector": name, "coverage": block["aggregate"]["coverage"],
                 "evaluated": block["aggregate"]["evaluated"],
                 "precision": metric.get("precision"), "recall": metric.get("recall"),
                 "f1": metric.get("f1"), "accuracy": metric.get("accuracy"),
                 "latency_ms": block["mean_latency_ms"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
selected = st.selectbox("Detector", list(summary["detectors"]))
block = summary["detectors"][selected]
st.json(block["aggregate"])
st.subheader("Dataset explorer")
st.json({k: quality[k] for k in ("labels", "tasks", "splits", "datasets", "with_annotations", "missing_evidence")})
st.subheader("Errors and disagreements")
errors = base/f"errors_{selected}.jsonl"
if errors.exists():
    df = pd.read_json(errors, lines=True) if errors.stat().st_size else pd.DataFrame()
    st.dataframe(df, use_container_width=True)
else:
    st.info("No error file saved")
st.subheader("Claim-level explorer")
claim_files = list(base.glob("claims*.jsonl"))
if claim_files:
    for line in claim_files[0].read_text(encoding="utf-8").splitlines()[:20]:
        with st.expander("Claim annotations / predicted spans"):
            st.json(json.loads(line))
else:
    st.info("No claim evaluation available. Run `hallubench claims` on RAGTruth test records.")
st.subheader("Saved plots and exports")
for p in sorted(base.glob("*.png")):
    st.image(str(p), caption=p.name)
for p in sorted(base.glob("*.jsonl")) + sorted(base.glob("*.json")) + sorted(base.glob("findings.*")):
    st.download_button(p.name, data=p.read_bytes(), file_name=p.name, key=str(p))
