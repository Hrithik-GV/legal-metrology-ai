"""
Streamlit Demonstration Interface for Legal Metrology AI.
Hackathon-ready presentation UI for packaged commodity compliance analysis.

Run command:
streamlit run demo_app.py --server.port 8501
"""

import sys
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows console UTF-8 safety
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.pipeline.pipeline import run_pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Metrology AI — Product Compliance Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for Hackathon Presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-pass {
        color: #16A34A;
        font-weight: 600;
    }
    .status-warn {
        color: #D97706;
        font-weight: 600;
    }
    .status-violation {
        color: #DC2626;
        font-weight: 600;
    }
    .footer-disclaimer {
        margin-top: 3rem;
        padding: 14px;
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        color: #92400E;
        font-weight: 600;
        text-align: center;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚖️ Legal Metrology AI — Product Compliance Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated AI/Vision engine assessing statutory packaging declarations under Legal Metrology (Packaged Commodities) Rules, 2011.</div>', unsafe_allow_html=True)

# Test images directory
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
available_samples = {}
if TEST_IMAGES_DIR.exists():
    for f in sorted(TEST_IMAGES_DIR.glob("*.*")):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            clean_name = f.stem.replace("_", " ").title()
            available_samples[clean_name] = f

# Input Selection: Upload or Preset Benchmark
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a packaged commodity front/back label photo."
    )

with col_input2:
    selected_sample_name = None
    if available_samples:
        st.write("**Or select benchmark test sample:**")
        sample_choice = st.selectbox(
            "Benchmark Samples",
            ["None"] + list(available_samples.keys()),
            index=0,
            label_visibility="collapsed"
        )
        if sample_choice != "None":
            selected_sample_name = sample_choice

# Prepare active image
target_image_path = None

if uploaded_file is not None:
    # Save uploaded file to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    tfile.write(uploaded_file.read())
    tfile.flush()
    target_image_path = Path(tfile.name)
elif selected_sample_name:
    target_image_path = available_samples[selected_sample_name]

# Analyze Button
st.write("")
analyze_clicked = st.button("🔍 Analyze Product", type="primary", use_container_width=False)

if analyze_clicked:
    if target_image_path is None:
        st.warning("Please upload a product image or choose a benchmark sample first.")
    else:
        with st.spinner("Running Vision Pipeline: OpenCV → YOLOv8 → PaddleOCR → Declaration Extractor → Rule Engine..."):
            try:
                pipeline_result = run_pipeline(image_path=target_image_path)
            except Exception as e:
                st.error(f"Pipeline execution error: {e}")
                st.stop()

        st.success("Analysis Complete!")

        # ----------------------------------------------------
        # TOP SUMMARY METRICS
        # ----------------------------------------------------
        rule_eval = pipeline_result.get("rule_assessment", {})
        score = rule_eval.get("compliance_score", 0.0)
        summary = rule_eval.get("summary", {})
        violations_count = summary.get("potential_violations", 0)
        warnings_count = summary.get("warning_count", 0)

        if violations_count > 0:
            overall_badge = "POTENTIAL VIOLATIONS DETECTED"
            badge_color = "#DC2626"
        elif warnings_count > 0:
            overall_badge = "REVIEW REQUIRED"
            badge_color = "#D97706"
        else:
            overall_badge = "PRELIMINARY PASS"
            badge_color = "#16A34A"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compliance Score", f"{score:.0f}%")
        m2.metric("Overall Status", overall_badge)
        m3.metric("Declarations Found", f"{pipeline_result['summary']['declarations_found']} / 7")
        m4.metric("Potential Violations", f"{violations_count}")

        st.markdown("---")

        # ----------------------------------------------------
        # 1 & 2: ORIGINAL IMAGE vs AI ANNOTATED IMAGE
        # ----------------------------------------------------
        st.subheader("🖼️ Visual Inspection & Region Detection")
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.write("**Original Product Image**")
            orig_img = Image.open(target_image_path)
            st.image(orig_img, use_container_width=True)

        with img_col2:
            st.write("**AI Annotated Image (YOLO Regions + OCR Polygons)**")
            annotated_path = pipeline_result.get("annotated_image")
            if annotated_path and Path(annotated_path).exists():
                ann_img = Image.open(annotated_path)
                st.image(ann_img, use_container_width=True)
            else:
                st.info("Annotated image unavailable.")

        st.markdown("---")

        # ----------------------------------------------------
        # 3, 5 & 6: PRODUCT INFORMATION & COMPLIANCE CHECKS
        # ----------------------------------------------------
        info_col, comp_col = st.columns([1, 1])

        # Convert declarations list to dict for display
        dec_list = pipeline_result.get("declarations", [])
        dec_dict = {d["field"]: d for d in dec_list}

        with info_col:
            st.subheader("📦 PRODUCT INFORMATION")
            st.markdown("""
            Structured statutory entities extracted from packaged commodity text:
            """)

            prod_name = dec_dict.get("product_name", {}).get("value", "Not Detected")
            net_qty = dec_dict.get("net_quantity", {}).get("value", "Not Detected")
            mrp_val = dec_dict.get("mrp", {}).get("value", "Not Detected")
            mfg_name = dec_dict.get("manufacturer", {}).get("value", "Not Detected")
            date_val = (
                dec_dict.get("manufactured_date", {}).get("value")
                or dec_dict.get("packed_date", {}).get("value")
                or "Not Detected"
            )
            care_val = dec_dict.get("consumer_care", {}).get("value", "Not Detected")
            origin_val = dec_dict.get("country_of_origin", {}).get("value", "Not Detected")

            st.write(f"**Product Name:** `{prod_name}`")
            st.write(f"**Net Quantity:** `{net_qty}`")
            st.write(f"**MRP:** `{mrp_val}`")
            st.write(f"**Manufacturer / Packer:** `{mfg_name}`")
            st.write(f"**Date of Pkg / Mfg:** `{date_val}`")
            st.write(f"**Consumer Care Helpline:** `{care_val}`")
            st.write(f"**Country of Origin:** `{origin_val}`")

        with comp_col:
            st.subheader("📋 COMPLIANCE & RULE VERIFICATION")
            st.markdown("Automated statutory check against representative Legal Metrology rules:")

            rule_results = rule_eval.get("rule_results", [])
            for rule in rule_results:
                status = rule["status"]
                name = rule["field"].replace("_", " ").title()
                reason = rule["reason"]
                conf = rule.get("confidence", 0.0)

                if status == "PASS":
                    st.markdown(f"✅ **{name}** — Present <span style='color:#16A34A; font-size:0.85rem;'>(Conf: {conf:.2f})</span>", unsafe_allow_html=True)
                elif status == "WARNING":
                    st.markdown(f"⚠️ **{name}** — Manual Review Recommended <span style='color:#D97706; font-size:0.85rem;'>({reason})</span>", unsafe_allow_html=True)
                elif status == "POTENTIAL_VIOLATION":
                    st.markdown(f"❌ **{name}** — **Not Detected** <span style='color:#DC2626; font-size:0.85rem;'>(Potential Violation)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"❓ **{name}** — Not Verifiable Automatically", unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------------------------------
        # 4: OCR DETECTIONS & CONFIDENCES (EXPENDABLE)
        # ----------------------------------------------------
        with st.expander("🔍 View Raw OCR Text Tokens & Confidence Scores"):
            ocr_tokens = pipeline_result.get("ocr_results", [])
            st.write(f"Total Text Regions Detected: **{len(ocr_tokens)}**")

            token_data = []
            for t in ocr_tokens:
                token_data.append({
                    "Detected Text": t.get("text", ""),
                    "Confidence": f"{t.get('confidence', 0.0):.2%}",
                    "Bounding Box": str(t.get("bounding_box", [])[:2]) + "..."
                })
            st.table(token_data)

        # ----------------------------------------------------
        # STATUTORY MANDATORY DISCLAIMER
        # ----------------------------------------------------
        st.markdown(
            '<div class="footer-disclaimer">⚠️ AI-Assisted Preliminary Assessment — Manual Verification Required</div>',
            unsafe_allow_html=True
        )

else:
    # Initial state helper
    st.info("Upload an image above or select a benchmark sample, then click **Analyze Product**.")
