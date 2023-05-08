import pickle
import tempfile
import pandas as pd
import streamlit as st
import openai
from langchain import PromptTemplate
import os
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.agents import create_pandas_dataframe_agent , ZeroShotAgent,   initialize_agent
from langchain.memory import ConversationBufferMemory
global api_key
api_key="xxx"




template = """
you are an expert data scientist, who has a niche in analyzing tabular data of any kind. You are a master at 
using python and pandas for greater insights and analysis of CSV data when user ask you {question}.
"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

def generate_response_openai(api_key, prompt):
    openai.api_key = api_key
    model_engine = "text-davinci-002"

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def generate_response_huggingface(prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    output_tokens = model.generate(input_tokens, max_length=150, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output_tokens[:, input_tokens.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()


def generate_response_langchain(prompt, agent): 
    return agent.run(prompt)
def agent_prompt():
    prefix = """Have a conversation with a human, answering the following questions as best you can. considering you are an expert datascientist:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "chat_history", "agent_scratchpad"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    return prompt
        

    # llm = OpenAI(temperature=0.9)  # model_name="text-davinci-003"
    # chain = LLMChain(llm, tokenizer)
    # template = PromptTemplate(prompt)
    # response = chain.generate(template, max_length=150, temperature=0.7)
    # return response.strip()
def load_history():
    try:
        with open('history.pkl', 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        history = []
    return history

def save_history(history):
    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)

def create_agent(df):
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent =  create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True, return_intermediate_steps=False,
                                              memory=memory)
    return agent
def main():
    st.set_page_config(page_title="FrameGPT", page_icon=":speech_balloon:")

    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 700px;
            }}
            h1{{
                font-size: 36px;
            }}
            input[type="text"] {{
                padding: 12px 20px;
                margin: 8px 0;
                box-sizing: border-box;
                border: 2px solid #ccc;
                border-radius: 4px;
                font-size: 16px;
            }}
        </style>
    """,
        unsafe_allow_html=True,
    )

    api_option = st.sidebar.radio("Select the AI model:", ["LangChain"])

    if api_option == "OpenAI API" or api_option == "LangChain":
        api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
        os.environ["OPENAI_API_KEY"] = api_key
        if not api_key:
            st.sidebar.warning("Please enter a valid OpenAI API key.")
            return
    else:
        st.sidebar.info("Using Hugging Face Transformers with GPT-2.")

    st.title("DataGPT App")
    # Sidebar components
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        data = CSVLoader(temp_file_path).load()
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(temp_file_path, header=None)
        agent = create_agent(df)
        # Convert column 1 to string
        df[1] = df[1].apply(str)
        st.write("## CSV Data")
        st.write(df)
   
    # chat_history = st.sidebar.empty()
    chat_history_container = st.container()
    

    user_input = st.empty()
    input_text = user_input.text_input("Type your message here...", "")
    history = load_history()
    if st.button("Send"):
        if input_text:
            prompt = f"User: {input_text}\nAI:"
            #prompt = agent_prompt()
            if api_option == "LangChain":
                response = generate_response_langchain(prompt, agent)
                #response = "test response"
            history.append({"user": input_text, "ai": response})
            save_history(history)
            with chat_history_container:
                st.markdown("## Chat History")
                for h in history:
                    st.write(f"**User:** {h['user']}")
                    st.write(f"**AI:** {h['ai']}")
                    st.write("---")
        else:
            st.warning("Please enter a message before sending.")


if __name__ == "__main__":
    history = [{"user": "Hi", "ai": "Hello, how can I help you?", "ai": "Hello, how can I help you?"}]
    main()
