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

## Project 2: Syntax Shift
### Overview
A code translator that takes a Python file and converts it to C, C++, Rust, or Javascript.
It's like Google Translate, but for code.

### Technologies Used
- Python
- OpenAI GPT-4o
- The `subprocess` library to compile/run the translated code
- Streamlit and Gradio for deployment
- Code Example
- ```
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
