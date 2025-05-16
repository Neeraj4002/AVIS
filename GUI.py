# ui.py
import os
import re
import streamlit as st
import time
from langchain.schema import HumanMessage
from code_agent import get_compiled_graph

# ——————————————————————————————————————————
# 1. Sidebar: Configuration
# ——————————————————————————————————————————
st.set_page_config(page_title="LangGraph Code Assistant", page_icon="🧑‍💻", layout="wide")
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE"),
        type="password",
        help="Enter your Google API Key or set the GOOGLE_API_KEY env variable.",
    )
    st.subheader("Retrieval Settings")
    k_docs_slider = st.slider("Number of documents to retrieve (k)", 1, 15, 7)
    rerank_flag_checkbox = st.checkbox("Enable LLM Reranking", value=True)
    st.subheader("Streaming Settings")
    typing_speed_slider = st.slider("Typing speed (words/min)", 50, 800, 800)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{int(time.time())}"


st.title("🧑‍💻 LangGraph RAG Code Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=False)

# User input
user_input = st.chat_input("Ask about LangGraph...")
if user_input:
    # Immediately display the user's message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Build and compile graph
    graph = get_compiled_graph()
    if graph is None:
        st.error("Agent unavailable.")
    else:
        # Stream response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_resp = ""
            delay = 60.0 / (typing_speed_slider * 5)
            # Prepare config for agent
            agent_config = {
                "api_key": api_key_input,
                "k_docs": k_docs_slider,
                "rerank": rerank_flag_checkbox,
                "thread_id": st.session_state.thread_id
            }
            docs = None
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": agent_config}
            ):
                if "rag_qa" in chunk:
                    node_out = chunk['rag_qa']
                    msgs = node_out["messages"]
                    if "docs" in node_out:
                        docs = node_out["docs"]
                    else: 
                        print("No docs found in node output.")
                    if msgs:
                        new = msgs[-1].content
                        # Simulate typing
                        for i in range(len(full_resp), len(new)):
                            full_resp = new[: i + 1]
                            placeholder.markdown(full_resp + "▌")
                            time.sleep(delay)
                        placeholder.markdown(full_resp)
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
            
            # Show retrieved source chunks with detailed metadata
            with st.expander("Sources / Chunks used"):
                if not docs:
                    st.write("_No sources available_")
                for d in docs:
                    # Pull out metadata fields
                    headers = d.metadata.get("headers", {})
                    h1 = headers.get("h1", "No H1")
                    h2 = headers.get("h2", "No H2")
                    topic = d.metadata.get("topic", "No Topic")
                    tags = ", ".join(d.metadata.get("tags", []))
                    num_code_blocks = d.metadata.get("num_code_blocks", 0)
                    
                    # Render each chunk summary
                    st.markdown(f"""\
- **Topic:** `{topic}`
- **Header 1:** {h1}
- **Header 2:** {h2}
- **Tags:** {tags}
- **Code Blocks:** {num_code_blocks}

<details>
  <summary>📄 Preview chunk content</summary>

  ```markdown
  {d.page_content[:]}{'...' if len(d.page_content) > 500 else ''}
  ```
</details>
""")


