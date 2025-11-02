import streamlit as st
import requests

# ==================== CONFIG ====================
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="🎓 Student Assistant Bot", page_icon="🎓", layout="wide")
st.title("🎓 Student Assistant Bot")

# ==================== SIDEBAR: HISTORY ====================
st.sidebar.header("🕘 Chat History")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# Sidebar previous queries
for i, item in enumerate(reversed(st.session_state.history)):
    if st.sidebar.button(item["query"], key=f"hist_{i}"):
        st.session_state.current_query = item["query"]
        st.session_state.current_response = item["response"]

# Clear sidebar history
if st.sidebar.button("🧹 Clear History"):
    st.session_state.history = []
    st.session_state.current_query = ""
    st.session_state.current_response = ""

# ==================== MAIN AREA ====================

# Create a "search bar" layout with text input + button side by side
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask your question:",
        value=st.session_state.get("current_query", ""),
        placeholder="Type your question here and press Enter or click Ask...",
        label_visibility="collapsed",  # hides the label
    )
with col2:
    ask_clicked = st.button("Ask")

# Pressing Enter in the input triggers the same event
submitted = ask_clicked or (query and st.session_state.get("last_query") != query)

# ==================== BACKEND REQUEST ====================
if submitted and query.strip():
    st.session_state["last_query"] = query
    st.info("⏳ Sending request to backend...")
    try:
        r = requests.post(API_URL, json={"query": query})
        if r.status_code == 200:
            data = r.json()
            summary = data.get("summary", "(no summary)")
            sources = data.get("sources", [])

            # Update session state
            st.session_state.current_query = query
            st.session_state.current_response = summary
            st.session_state.history.append({"query": query, "response": summary})

            # Display result
            st.subheader("🧠 Final Answer")
            st.write(summary)

            if sources:
                st.markdown("### 🔗 Sources used:")
                for s in sources:
                    st.markdown(f"- {s}")

        else:
            st.error(f"Backend error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"⚠️ Could not reach backend: {e}")
elif submitted and not query.strip():
    st.warning("Please enter a question before clicking Ask.")

# Display last response when page reloads
if st.session_state.get("current_response"):
    st.subheader("🧠 Final Answer")
    st.write(st.session_state.current_response)
