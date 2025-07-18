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

from agent import QA_Agent
from utils import *

    
# --------- Preparing Datasets ---------
if "dataset" not in st.session_state:
    st.session_state.dataset = load_dataset("real/sales-synthetic-data")
smart_df_sales = st.session_state.dataset


# --------- Setting Up Agent -----------

if "qa_agent" not in st.session_state:
    st.session_state.qa_agent = QA_Agent([smart_df_sales])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.qa_agent


# --------- Configuration ---------


encoded_bg = get_base64_img("assets/bg_img.jpg")
encoded_left_logo = get_base64_img("assets/EYLEA_left logo.png")
encoded_right_logo = get_base64_img("assets/EYLEA_right logo.png")


business_highlights = [
    "First quarter 2025 EYLEA HD® U.S. net sales increased 54% to $307 million versus first quarter 2024",
    "FDA accepted for priority review EYLEA HD sBLA for both retinal vein occlusion (RVO) and for monthly dosing in approved indications",
    "EYLEA® HD (8 mg) approved in key markets with dosing up to 16 weeks, enhancing patient adherence.",
    "Rapid uptake of EYLEA® HD supports lifecycle extension amid biosimilar entry and evolving market dynamics."
]


# ---------- Custom CSS Styling ----------
# --- CSS Styling ---
st.markdown(f"""
    <style>
    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    .topbar-wrapper {{
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        background-image: url("data:image/jpeg;base64,{encoded_bg}");
        background-size: cover;
        background-position: center;
        padding: 4rem 3rem 2rem 3rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        color: white;
    }}

    .topbar-inner {{
        max-width: 1400px;
        margin: 0 auto;
    }}

    .topbar-logos {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 2rem;
    }}

    .topbar-logos img {{
        height: 50px;
        max-width: 200px;
    }}

   
    .highlight-title {{
        text-align: center;
        color: #002B55 !important;
        font-size: 1.6rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    
    .highlight-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1rem;
    }}

    .highlight-card {{
        background: linear-gradient(135deg, #2C70C9, #A5C9FF);
        color: #F5FAFF;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 500;
        text-align: center;
        height: 100%;
        border: none;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease-in-out;
    }}

    .highlight-card:hover {{
        background-color: rgba(255, 255, 255, 0.15);
        transform: scale(1.02);
    }}

    .user-bubble {{
        background-color: #D6EAF8;
        color: #003366;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        text-align: left;
    }}
    

    .assistant-bubble {{
        background-color: #F4F6F7;
        color: #222;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        max-width: 80%;
        margin: 0.5rem auto;
        text-align: left;
    }}
    

    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        padding-bottom: 1rem;
    }}

    .response-text {{
        font-family: 'Courier New', monospace; /* or any font you like */
        font-size: 18px;
        white-space: pre-wrap;      
        word-break: break-word;     
    }}

    @media (max-width: 768px) {{
        .topbar-logos {{
            flex-direction: column;
            gap: 1rem;
        }}
    }}

    </style>
""", unsafe_allow_html=True)


# --- Render Top Bar ---
st.markdown(f"""
    <div class="topbar-wrapper">
        <div class="topbar-inner">
            <div class="topbar-logos">
                <img src="data:image/png;base64,{encoded_left_logo}" alt="Left Logo">
                <img src="data:image/png;base64,{encoded_right_logo}" alt="Right Logo">
            </div>
            <div class="highlight-title">Business Highlights</div>
            <div class="highlight-cards">
                {''.join(f'<div class="highlight-card">{point}</div>' for point in business_highlights)}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)



if agent:
    # Display previous conversation
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{q}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            if hasattr(a, "type"):
                if a.type == "dataframe":
                    styled_df = beautify_table(a.value)
                    st.dataframe(styled_df, hide_index=True, use_container_width=False)
                elif a.type == "chart":
                    st.image(a.value,width=750)
                elif a.type == "string":
                    st.markdown(a.value)
                else:
                    st.markdown(a.value)
            else:
                st.markdown(a.value)
                
    # Chat input box
    user_query = st.chat_input("Type your question")
    if user_query:
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{user_query}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.ask(user_query)
                    if result.type == "dataframe":
                        styled_df = beautify_table(result.value)
                        st.dataframe(styled_df, hide_index=True, use_container_width=False)
                        st.session_state.chat_history.append((user_query, result))
                    elif result.type == "string":
                        st.markdown(result.value)
                        st.session_state.chat_history.append((user_query, result))
                    elif result.type == "chart":  
                        st.image(result.value,width=750)
                        st.session_state.chat_history.append((user_query, result))
                    else:
                        st.write(result.value)
                        st.session_state.chat_history.append((user_query, result))
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                