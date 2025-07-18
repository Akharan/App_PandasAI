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

def load_dataset(folder_path):
    df = pai.load(folder_path)
    return df

def get_base64_img(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def format_integer_like_columns(df):
    # Identify numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    int_like_cols = []
    for col in num_cols:
        # Check if all non-null values are whole numbers
        s = df[col].dropna()
        if np.all(np.isclose(s, s.astype(int))):  # all values are int-like
            # Cast to int for clean formatting
            df[col] = s.astype(int)
            int_like_cols.append(col)

    for col in int_like_cols:
        df[col] = df[col].apply(lambda x: f"{x:,}")
    
    return df

def format_float_like_columns(df):
    float_cols = df.select_dtypes(include=['float', 'float32', 'float64']).columns
    for col in float_cols:
        df[col] = df[col].apply(lambda x: f"{x:,.2f}")
    
    return df

def beautify_table(df):

    df = format_integer_like_columns(df)
    df = format_float_like_columns(df)
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
    # check if all non-null values are first of month and no time component
        if ((df[col].dropna().dt.day == 1).all() 
            and (df[col].dropna().dt.hour == 0).all()
            and (df[col].dropna().dt.minute == 0).all()
            and (df[col].dropna().dt.second == 0).all()):
            # Only then format to %b-%y
            df[col] = df[col].dt.strftime("%b-%y")
         
    df.columns = df.columns.str.replace('_', ' ', regex=False)
    df = df.reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df)+1))

    styled = (
        df.style.set_properties(**{"text-align": "center"})  # center alignment
    )

    return styled
