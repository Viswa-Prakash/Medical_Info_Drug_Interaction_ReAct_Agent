import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import medical_agent

st.set_page_config(page_title="Medical Information and Drug Interaction Advisor", page_icon=":pill:")

st.title("Medical Information and Drug Interaction Advisor")

st.markdown("""
Ask questions for safe drug usage, side effects, or possible interactions.
*Examples:*
- `"Can I take ibuprofen with amlodipine?"`
- `"Safe non-prescription painkillers for hypertension?"`
- `"What are the common side effects for Zyrtec?"`
""")

with st.form("query_form"):
    user_query = st.text_area("Your medical question:", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = medical_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
