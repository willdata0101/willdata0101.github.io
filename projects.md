---
layout: default
title: LLM Project Portfolio
permalink: /projects/

# LLM Project portfolio

## Project 1: Historical Archive Assistant Chatbot
### Overview
A RAG-based chatbot that can answer any question in any language about a given database of historical records.

### Technologies Used
- Python
- AWS Sagemaker for code development
- AWS Bedrock for access to foundational models (LLMs) such as GPT, Claude, and Llama
- LangChain for RAG implementation
- Streamlit for deployment

### Full Project Code
```
# Importing libraries
import boto3
from botocore.exceptions import ClientError

from pydantic import BaseModel
from langchain_community.llms import Bedrock
from langchain_aws import ChatBedrock
from langchain.chains import LLMChain
from langchain_community.retrievers import WikipediaRetriever
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setting session and region variables
session = boto3.Session()
region = session.region_name
bedrock_client = boto3.client('bedrock-runtime', region_name = region)

class Assistant:
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client  # Amazon Bedrock client for LLM
        self.memory = ConversationBufferMemory()  # Memory to track conversation flow
        
    def handle_query(self, user_query):
        # Use Amazon Bedrock to generate a response
        llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
        
        # Creating a system prompt to instruct the model how to answer the query
        system_prompt = (
            """
            You are an historical archives assistant
            tasked with answering questions based on
            archival records. Use the following
            historical documents to answer the question.
            If you don't know the answer, say you don't
            know. Keep the answer between 500-1000 words.
            If you're using a bulleted or numbered list,
            use a new line for each list item.
            \n\n
            {context}
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]   
         
        )

        retriever = WikipediaRetriever()

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_query})

        return response

import streamlit as st

# Building streamlit app
st.title("Multilingual Historical Archive Assistant")

def run_assistant():
    with st.sidebar:
        with st.chat_message("assistant"):
            st.write("""Hello 👋! Ask me anything about your historical documents. 📜
                     I can take questions in any language.
                     Please make sure your question is clear, concise, and provides
                     sufficient context. This will result in higher
                     quality responses. ✨""")

    prompt = st.chat_input("Ask me anything")
    if prompt:
        message = st.chat_message("assistant")
        message.write("Researching your question...")
        assistant = Assistant(bedrock_client)
        # Handle the query
        response = assistant.handle_query(prompt)
        clean_response = "".join(response['answer'])
        st.write(clean_response)

run_assistant()
```

## Project 2: Syntax Shift
### Overview
A code translator that takes a Python file and converts it to C, C++, Rust, or Javascript.
It's like Google Translate, but for code.

### Technologies Used
- Python
- OpenAI GPT-4o
- The `subprocess` library to compile/run the translated code
- Streamlit and Gradio for deployment

### Code Example
```
  class TranslateCode:
    def __init__(self, openai_client, model):
        self.openai = openai_client
        self.model = model

    def user_prompt_for(self, python, lang_select):
        user_prompt = f"Rewrite this Python code in {lang_select} with the fastest possible implementation that produces identical output in the least time. "
        user_prompt += f"Respond only with {lang_select} code; do not explain your work; only return {lang_select} code. "
        user_prompt += "Pay attention to number types to ensure no int overflows. Remember to include all necessary dependencies and libraries.\n\n"
        user_prompt += "If translating to Rust, make sure to include the necessary packages and crates."
        user_prompt += python
        return user_prompt

    def messages_for(self, python, lang_select):
        # System message for OpenAI API
        system_message = "You are an assistant that reimplements Python code in high performance code for a Windows PC. "
        system_message += "Respond only with code; do not provide any explanations. "
        system_message += "The response needs to produce an identical output in the fastest possible time."

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": self.user_prompt_for(python, lang_select)}
        ]

    def translate_code(self, code_file, lang_select):
        stream = self.openai.chat.completions.create(model=self.model, messages=self.messages_for(code_file, lang_select), stream=True)
        code = ""
        for chunk in stream:
            fragment = chunk.choices[0].delta.content or ""
            code += fragment
        pattern = r"```(c|cpp|rust|javascript)\n"
        code = re.sub(pattern, "", code).replace("```", "")
        return code
  ```
