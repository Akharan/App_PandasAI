import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai import Agent
from pandasai_openai import AzureOpenAI
import json
from PIL import Image
import base64
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np

class QA_Agent:
    def __init__(self, dfs:list[pai.DataFrame]):
        self.dfs = dfs
        self.llm = AzureOpenAI(
                api_token=st.secrets["api"]["AZURE_API_KEY"],
                azure_endpoint=st.secrets["api"]["AZURE_ENDPOINT"],
                api_version=st.secrets["api"]["API_VERSION"],
                deployment_name=st.secrets["api"]["MODEL"],
                enable_memory=True)
        
        pai.config.set({"llm": self.llm, 
        "use_error_correction_framework":True,"enable_cache":True})
        self.agent = Agent(self.dfs)
        self.history = []
    
    
    def get_rules(self,question):
        # Loading index and rule list
        index = faiss.read_index("rules.index")
        with open("rules.pkl", "rb") as f:
            rules = pickle.load(f)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Embedding the user query
        query = question
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Search top-k similar rules
        D, I = index.search(query_embedding, k=4)
        top_rules = [rules[i] for i in I[0]]

        # Example: Construct context + question prompt
        rules_list = "\n".join(top_rules)
        rag_prompt = f"""You are a pharma analytics agent.

                    A. Current Question: {query}

                    B. General Rules (Takes first priority): 
                    1. Output should have text,table or plot (only 1).
                    2. Output type should be in ["string","dataframe","plot"]
                    3. If the answer is too short(just a number), please answer with natural language.
                    4. Plot only if the question specifically mentions plot, otherwise stick with dataframe or text as required.
                    5. If the question is vague like [do this, do that]: get context from previous questions

                    C. Use the following rules if needed before answering:
                    {rules_list}
                    """
        return rag_prompt

    
    def ask(self,question,max_turns=3):
        # Trimming History
        trimmed_history = self.history[-max_turns:]

        # Building Context
        context = ""
        for q, a in trimmed_history:
            context += f"User: {q}\nAssistant: {a}\n"
        
        # Building the user prompt
        full_prompt = self.get_rules(question)
        
        # Final context + current prompt
        full_context = context + f"User: {full_prompt}"
        
        # Calling the agent
        self.response = self.agent.chat(full_context)

        # Storing previous question and answer
        self.history.append((question, self.response))
        
        return self.response