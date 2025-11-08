import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- STAGE 0: INITIAL CONFIGURATION AND IMPORTS ---

# Streamlit requires set_page_config to be called once at the start
st.set_page_config(
    page_title="Policy Decoder Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Load environment variables from .env file immediately
load_dotenv()

# 2. Check for API Key early BEFORE importing other modules
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Please ensure the **.env** file is in the root project directory and contains **GEMINI_API_KEY=\"Your_Key_Here\"**.")
    st.stop()
# --- END CONFIGURATION ---

# Import our custom modules (these now run *after* the key is confirmed loaded)
# NOTE: Assuming these modules are available in the execution environment
from extractor import extract_text_from_pdf, get_page_count
from clause_detector import segment_text_into_clauses, count_clauses
from chroma_helper import get_chroma_client, get_rag_collection
from rag_store import query_rag_store
from summarizer import generate_full_summary, analyze_page_content

# Initialize session state 
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'clauses' not in st.session_state:
    st.session_state.clauses = []
if 'page_texts' not in st.session_state:
    st.session_state.page_texts = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'


# --- THEMEING AND STYLING FUNCTIONS (Hardcoded Dark Theme) ---

def set_styles():
    """Injects custom CSS for a permanent dark theme."""
    # Hardcoded Dark Theme Colors
    theme_accent_color = "#FF6F91"  # Vibrant Coral/Pink for dark mode
    text_color = "white"
    bg_color = "#121212" # Very dark background

    st.markdown(f"""
        <style>
        /* Set background for the main container to handle theme switch better */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* General Aesthetics & Transitions */
        .stButton>button {{
            border-radius: 12px;
            transition: all 0.2s ease-in-out;
            border: 1px solid {theme_accent_color};
            color: {text_color};
            font-weight: 500;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* Stronger shadow for dark mode */
            background-color: {theme_accent_color}33; /* Light tint of accent color */
        }}
        
        /* Metric Boxes (Professional data display) */
        [data-testid="stMetric"] {{
            padding: 15px;
            border-radius: 15px;
            border: 2px solid {theme_accent_color}55;
            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.3);
            background-color: #1e1e1e; /* Slightly lighter dark background for contrast */
        }}

        /* Custom Header (Cute and Professional) */
        h1 {{
            color: {theme_accent_color};
            font-size: 2.5em;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }}
        
        /* Custom coloring for cute/safe alerts */
        .stAlert.stAlert--success {{ border-left: 5px solid #66bb6a; }} /* Lighter green for dark theme */
        .stAlert.stAlert--error {{ border-left: 5px solid #ef5350; }} /* Lighter red for dark theme */
        .stAlert.stAlert--warning {{ border-left: 5px solid #ffeb3b; }} /* Yellow for dark theme */
        
        /* Resizable text area style fix for better look */
        .stTextArea > label {{ display: none; }}
        .stTextArea {{ margin-top: 10px; }}
        </style>
    """, unsafe_allow_html=True)


# --- UI COMPONENTS ---

def display_stats(page_count, clause_count):
    """Displays key policy statistics using a clean layout."""
    st.markdown("### 📊 Policy Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Total Pages 📄", page_count)
    with col2:
        st.metric("Total Clauses (Segments) ✂️", clause_count)
    st.markdown("---")


def display_full_report():
    """Compiles and displays the full report."""
    if not st.session_state.analysis_results:
        st.warning("Please run the 'Analyze with Gemini' step first.")
        return

    report = ""
    analysis = st.session_state.analysis_results
    
    # ... Report content generation
    report += "========================================================================\n"
    report += f"INSURANCE POLICY DECODED REPORT: {st.session_state.file_name or 'Policy'}\n"
    report += "========================================================================\n\n"
    report += "--- 1. HIGH-LEVEL POLICY SUMMARY ---\n"
    summary = analysis.get('full_summary', 'Summary generation failed due to API or connection error.')
    report += summary + "\n\n"
    
    report += "--- 2. DETAILED PAGE-BY-PAGE ANALYSIS (Classification & Summary) ---\n"
    page_analysis = analysis.get('page_analysis', [])
    for item in page_analysis:
        page_num = item.get('pageNumber', 'N/A')
        classification = item.get('classification', 'N/A')
        summary = item.get('summary', 'N/A')
        report += f"\n[PAGE {page_num} | CLASSIFICATION: {classification.upper()}]\n"
        report += f"Simplified Summary: {summary}\n"
    
    # 3. Clauses (Optional, for detailed reference)
    report += "\n\n--- 3. RAW CLAUSE EXTRACT (For Reference) ---\n"
    report += f"Total Clauses: {len(st.session_state.clauses)}\n"
    for clause in st.session_state.clauses:
        # Assuming 'page_num' is set during processing
        page_num = clause.get('page_num', 'N/A')
        
        # Extract just the clause segment number (e.g., '5' from 'p1_c5')
        clause_id_parts = clause['clause_id'].split('_')
        clause_id_display = clause_id_parts[-1].replace('c', '') if len(clause_id_parts) > 1 else clause['clause_id']

        report += f"[P{page_num}-C{clause_id_display}]\n{clause['text'][:200]}...\n\n"
        
    st.text_area("Generated Decoded Report", report, height=500, label_visibility="collapsed")
    
    st.download_button(
        label="⬇️ Download Full Report (TXT)",
        data=report.encode('utf-8'),
        file_name=f"decoded_policy_report_{st.session_state.file_name or 'policy'}.txt",
        mime="text/plain",
        use_container_width=True
    )

# --- MAIN APPLICATION LOGIC ---

# 1. Apply Styles (Hardcoded Dark)
set_styles()

# 2. Header
st.title("🛡️ Insurance Policy Decoder Agent")
st.markdown("Upload your policy PDF to **simplify complex clauses** and **decode coverage** using the Gemini API. ")

st.markdown("---")

# --- SIDEBAR (Upload Control) ---
with st.sidebar:
    st.header("Upload Policy PDF ⬆️")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader",
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Powered by Google Gemini 2.5 Flash")


if uploaded_file is not None:
    main_container = st.container(border=True)
    with main_container:
        try:
            # File Processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
            
            st.session_state.file_name = uploaded_file.name

            with st.spinner("🚀 Extracting text and segmenting clauses..."):
                page_texts = extract_text_from_pdf(pdf_path)
                st.session_state.page_texts = page_texts
                
                page_count = get_page_count(pdf_path)
                clauses = segment_text_into_clauses(page_texts)
                st.session_state.clauses = clauses
                clause_count = count_clauses(clauses)

            st.success(f"✅ Policy '{st.session_state.file_name}' processed successfully! ({page_count} pages)")
            display_stats(page_count, clause_count)
            
            # NOTE: We assume the helper functions (like get_chroma_client) are correctly implemented
            chroma_client = get_chroma_client()
            rag_collection = get_rag_collection(chroma_client, clauses)

            
            # --- ACTION BUTTONS ---
            st.markdown("### ⚙️ Choose Action")
            col_buttons = st.columns(3)

            with col_buttons[0]:
                # The button to trigger the display of raw clauses
                if st.button("🔍 Show Raw Clauses", use_container_width=True):
                    # Expander text updated as requested
                    with st.expander("📖 View all clauses and definitions", expanded=True):
                        
                        with st.spinner("Querying RAG for key definitions..."):
                            # This query gets GENERAL definitions for the whole policy
                            key_terms_context = query_rag_store(rag_collection, "Definitions of key terms and words in the policy.", k=3)
                        
                        # --- RESIZABLE CLAUSES FEATURE IMPLEMENTATION (Using Verbose Format) ---
                        all_clauses_text = ""
                        for clause in st.session_state.clauses:
                            try:
                                # Extract just the clause segment number (e.g., '5' from 'p1_c5')
                                clause_segment_num = clause['clause_id'].split('_')[-1].replace('c', '')
                                page_num_match = clause['clause_id'].split('_')[0].replace('p', '')
                            except Exception:
                                clause_segment_num = 'N/A'
                                page_num_match = clause.get('page_num', 'N/A')
                            
                            # Applying the user-requested verbose format, using clause text for definition
                            formatted_clause = f"""pg no-> {page_num_match}
line no-> {clause_segment_num}
clause-> {clause['text']}
definition-> {clause['text']}
----------
"""
                            all_clauses_text += formatted_clause
                            
                        st.markdown("#### All Clauses from the document:")
                        # The st.text_area component is naturally resizable by the user via the bottom-right corner drag handle
                        st.text_area(
                            "Raw Policy Segments",
                            all_clauses_text,
                            height=400, # Initial height
                            label_visibility="collapsed",
                            key="raw_clauses_area"
                        )
                        # --- END RESIZABLE CLAUSES FEATURE ---
            
            with col_buttons[1]:
                if st.button("✨ Analyze with Gemini", use_container_width=True):
                    st.session_state.analysis_results = None
                    
                    full_policy_text = "\n".join(page_texts)
                    
                    with st.container(border=True):
                        st.markdown("### 🧠 Running AI Analysis...")
                        
                        with st.spinner("1/2: Generating full policy summary (up to 50 lines)..."):
                            full_summary = generate_full_summary(full_policy_text)
                            if full_summary.startswith("Client Initialization Error") or full_summary.startswith("Gemini API Error"):
                                st.error(full_summary)
                                st.stop()

                        with st.spinner("2/2: Classifying and simplifying content page-by-page..."):
                            page_analysis = analyze_page_content(page_texts)

                        st.session_state.analysis_results = {
                            'full_summary': full_summary,
                            'page_analysis': page_analysis
                        }
                        st.success("🎉 Analysis Complete! Scroll down for the full report.")
            
            with col_buttons[2]:
                if st.session_state.analysis_results:
                    # Report is ready, we show a visual indicator
                    st.markdown("<div style='text-align: center;'>✨ Report Ready!</div>", unsafe_allow_html=True)
                else:
                    # Report not ready, button is disabled
                    st.button("📄 Download Report", disabled=True, use_container_width=True)


            # --- DISPLAY ANALYSIS OUTPUT ---
            st.markdown("---")
            if st.session_state.analysis_results:
                analysis = st.session_state.analysis_results
                
                st.subheader("💡 Decoded Policy Report")

                summary_col, download_col = st.columns([2, 1])

                with summary_col:
                    st.markdown("#### 1. High-Level Policy Summary (The 'Gist')")
                    st.info(analysis.get('full_summary', 'Summary generation failed or returned no data.'))
                
                with download_col:
                    st.markdown("#### Download Full Report")
                    display_full_report()

                st.markdown("#### 2. Page-by-Page Classification & Plain-Language Summary")
                st.markdown("*(Quickly identify Coverages and Exclusions)*")

                for item in analysis.get('page_analysis', []):
                    classification = item.get('classification', 'N/A')
                    summary = item.get('summary', 'N/A')
                    page_num = item.get('pageNumber', 'N/A')
                    
                    if classification.lower() == 'exclusions':
                        st.error(f"**🚫 PAGE {page_num} | EXCLUSION:** {summary}")
                    elif classification.lower() == 'coverage':
                        st.success(f"**✅ PAGE {page_num} | COVERAGE:** {summary}")
                    else:
                        st.warning(f"**🟡 PAGE {page_num} | {classification.upper()}:** {summary}")
        
        except Exception as e:
            st.error(f"An unexpected error occurred during file processing: {e}")

else:
    st.info("⬆️ Please upload a PDF policy document in the sidebar to begin the analysis. We'll simplify the jargon!")