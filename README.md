#  Medical Information and Drug Interaction Advisor

This is a ReAct-style AI agent designed to provide **reliable medical information** about drugs, symptoms, potential interactions, and general usage guidance. Built with **LangGraph**, **LangChain**, and **Streamlit**, it intelligently selects the best search or knowledge tool to answer your medical questions in real-time.

---


## Features

-  Identifies medicines related to symptoms
-  Detects potential drug interactions
-  Suggests usage guidance, dosage, and side effects
-  Uses multiple web tools (Google, Serper, Wikipedia, Tavily) for up-to-date info
-  ReAct-style loop with tool reasoning and controlled flow
-  Streamlit-based interactive UI

---

##  How It Works

The agent follows a **Reasoning + Acting (ReAct)** paradigm with memory of conversation. Here's the step-by-step flow:

1. **LLM (OpenAI GPT-4.1)** reasons based on user input.
2. Selects the most relevant tool:
   - `Serper` for general medical queries
   - `Google` for updated web searches
   - `Wikipedia` for encyclopedic knowledge
   - `Tavily` for summarized recent articles
3. Executes the tool call if needed.
4. Loops until a **Final Answer** is generated.
5. Ends the session after max steps or when a final summary is ready.

Built using:

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI GPT-4.1](https://openai.com/)
- [Google Custom Search](https://programmablesearchengine.google.com/)
- [Serper API](https://serper.dev/)
- [Tavily API](https://www.tavily.com/)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)

---

---

##  Example Queries

- “Can I take ibuprofen with amlodipine?”
- “Safe non-prescription painkillers for hypertension?”
- “What are the common side effects of Zyrtec?”

---

##  Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Viswa-Prakash/Medical_Info_Drug_Interaction_ReAct_Agent.git
cd Medical_Info_Drug_Interaction_ReAct_Agent


### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

### 3. Install Dependencies

```bash
pip install -r requirements.txt

### 4. Set Environment Variables

Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_custom_search_id
TAVILY_API_KEY=your_tavily_key


### 5. Run the App

```bash
streamlit run app.py

🧩 Agent Logic (LangGraph State Diagram)

- Check the Agent logic diagram in ipynb file. 