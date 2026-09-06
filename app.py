import streamlit as st
import requests
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
BACKEND_URL = "http://127.0.0.1:8000"

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="SwifTBaSkeT AI",
    page_icon="🛒",
    layout="wide"
)

st.title("SwifTBaSkeT AI")
st.caption("A hybrid AI assistant that routes analytical questions to SQL and specific/contextual questions to grounded RAG retrieval, with abstention when evidence is insufficient.")

# ==========================================
# SESSION STATE & CALLBACKS
# ==========================================
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

def set_question(q: str):
    """Callback to populate the text input from example buttons."""
    st.session_state.question_input = q

# ==========================================
# EXAMPLE QUESTIONS UI
# ==========================================
st.markdown("### Example Questions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Analytical (SQL)**")
    st.button("What is the average order value?", 
              on_click=set_question, args=("What is the average order value?",))
    st.button("What are the top 5 areas by average order value?", 
              on_click=set_question, args=("What are the top 5 areas by average order value?",))

with col2:
    st.markdown("**Contextual (RAG)**")
    st.button("Show me the details and status of order ORD0300000", 
              on_click=set_question, args=("Show me the details and status of order ORD0300000",))
    st.button("Who won the 2023 Cricket World Cup?", 
              on_click=set_question, args=("Who won the 2023 Cricket World Cup?",))

st.markdown("---")

# ==========================================
# MAIN INPUT & PROCESSING
# ==========================================
question = st.text_input("Ask a question:", key="question_input")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Analyzing and routing your query..."):
        try:
            # Call the FastAPI backend
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question.strip()},
                timeout=120  # Timeout to prevent hanging
            )
            
            if response.status_code == 200:
                data = response.json()
                
                route = data.get("route", "UNKNOWN")
                status = data.get("status")
                retrieval_status = data.get("retrieval_status")
                
                st.markdown("---")
                
                # 1. Visual Route Badge
                if route == "SQL":
                    st.markdown(":blue-badge[ROUTE: SQL]")
                elif route == "RAG":
                    st.markdown(":green-badge[ROUTE: RAG]")
                else:
                    st.markdown(":gray-badge[ROUTE: UNKNOWN]")
                
                # 2. ABSTAIN Warning Treatment
                if route == "RAG" and retrieval_status == "ABSTAIN":
                    st.warning("⚠️ **ABSTAIN:** The system abstained from generating an answer because sufficient evidence was not found in the retrieved context.")
                
                # 3. Prominent Answer Display
                st.markdown("### Answer")
                if route == "SQL":
                    df_data = data.get("data")
                    if isinstance(df_data, list) and len(df_data) > 0:
                        # Render records safely as a Pandas DataFrame
                        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
                    else:
                        st.info("The query executed successfully but returned 0 rows.")
                else:
                    st.write(data.get("answer"))
                
                # 4. Technical Details Expander
                with st.expander("Technical Details"):
                    st.write(f"**Route:** {route}")
                    st.write(f"**Status:** {status}")
                    st.write(f"**Route Reason:** {data.get('route_reason')}")
                    st.write(f"**Matched Signals:** {data.get('matched_signals')}")
                    st.write(f"**Exact Identifier:** {data.get('exact_identifier')}")
                    
                    if route == "SQL":
                        st.write("**Generated SQL:**")
                        st.code(data.get("generated_sql"), language="sql")
                        
                    elif route == "RAG":
                        st.write(f"**Retrieval Status:** {retrieval_status}")
                        st.write(f"**Retrieval Method:** {data.get('retrieval_method')}")
                    
                    # Log any non-fatal errors returned in payload
                    if data.get("error"):
                        st.error(f"Internal Error/Warning: {data.get('error')}")

            else:
                # Handle structured backend HTTP errors (e.g., 503, 500)
                err_detail = "An unknown error occurred on the server."
                try:
                    err_detail = response.json().get("detail", err_detail)
                except Exception:
                    pass
                st.error(f"Backend Error ({response.status_code}): {err_detail}")
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend took too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Ensure the FastAPI server is running.")
        except Exception:
            # Generic catch-all to prevent raw Python tracebacks
            st.error("An unexpected error occurred while communicating with the backend.")

elif ask_clicked:
    st.warning("Please enter a question before asking.")
