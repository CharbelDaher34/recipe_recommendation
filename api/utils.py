from fastapi import HTTPException
from PIL import Image
from transformers import AutoModel
from deep_translator import GoogleTranslator
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import numpy as np
import ast
import json
import csv
import os
import base64
from io import BytesIO
from globals import model, df

# Configuration constants
MODEL_PATH = "./jina_clip_v1_model"
MODEL_WEIGHTS = "./jina_clip_v1_model/jina.pt"
USER_EMBEDDINGS_PATH = "./user_embeddings.json"
RECIPE_EMBEDDINGS_PATH = "./embeddings.json"
DATA_PATH = "./data.csv"
FEEDBACK_PATH = "./recipe_feedback.csv"
RECIPES_ADD_PATH = "./recipes_add.csv"
REVIEWS_PATH = "./user_reviews.csv"
device="cpu"
def initialize_model():
    global model
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = torch.load(MODEL_WEIGHTS,map_location=torch.device('cpu'))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    return model

def load_user_embeddings():
    if os.path.exists(USER_EMBEDDINGS_PATH):
        with open(USER_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_user_embeddings(user_embeddings):
    with open(USER_EMBEDDINGS_PATH, "w") as f:
        json.dump(user_embeddings, f)

def get_user_embeddings(user_id, user_embeddings):
    user_id = str(user_id)
    return list(user_embeddings[user_id]) if user_id in user_embeddings else None

def update_user_embeddings(user_id, user_embeddings, new_embedding, alpha=0.8):
    user_id = str(user_id)
    if user_id in user_embeddings:
        previous_embedding = torch.tensor(user_embeddings[user_id])
        new_embedding = torch.tensor(new_embedding)
        updated_embedding = (1 - alpha) * new_embedding + alpha * previous_embedding
        user_embeddings[user_id] = [float(x) for x in updated_embedding]
    else:
        user_embeddings[user_id] = [float(x) for x in new_embedding]

    if user_embeddings:
        save_user_embeddings(user_embeddings)

def filter_df(df, **kwargs):
    filtered_df = df.copy()
    for key, value in kwargs.items():
        if key == "Title" or not value:
            continue
        if key not in df.columns:
            raise ValueError(f"Column '{key}' is not in the DataFrame.")
        
        if pd.api.types.is_numeric_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key] <= value]
        elif pd.api.types.is_string_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        elif key == "Cleaned_Ingredients":
            filtered_df = filtered_df[
                filtered_df[key].apply(lambda ingredients: any(
                    ingredient in ingredients for ingredient in value
                ))
            ]
    return filtered_df

# ... (other utility functions)

def compute_average_embedding(title_text=None, image=None):
    global model
    embeddings = []
    if title_text:
        title_embedding = torch.tensor(model.encode_text(title_text)).to(device)
        embeddings.append(title_embedding)
    if image:
        # Check if `image` is a file path or a file-like object
        if isinstance(image, str) or hasattr(image, 'read'):
            # Open the image file
            image = Image.open(image)
        # image = Image.open(image)
        image_embedding = torch.tensor(model.encode_image(image)).to(device)
        embeddings.append(image_embedding)
    if len(embeddings) == 0:
        return list(torch.zeros(768).cpu().numpy())
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return list(avg_embedding.cpu().numpy())


# Function to find the most similar recipes
def find_most_similar_recipe(avg_embedding, embeddings_json_path, df, top_n=5):
    # Load embeddings from JSON file
    with open(embeddings_json_path, "r") as f:
        recipe_embeddings = json.load(f)

    # Filter the embeddings based on IDs in the DataFrame
    df_ids = set(df["ID"].astype(str))  # Ensure IDs are strings
    filtered_embeddings = {k: v for k, v in recipe_embeddings.items() if k in df_ids}

    # Convert the filtered dictionary to list of IDs and embeddings
    recipe_ids = list(filtered_embeddings.keys())
    embeddings = [torch.tensor(embed) for embed in filtered_embeddings.values()]

    # Calculate cosine similarity between the average embedding and all recipe embeddings
    similarities = cosine_similarity([avg_embedding], embeddings)[0]

    # Get top_n most similar recipes
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_ids = [int(recipe_ids[i]) for i in top_indices]

    return top_ids







### Feedback part
import csv
from deep_translator import GoogleTranslator
from langdetect import detect


def is_hindi(text):
    return detect(text) == "hi"


def translate_text(text, target_lang, source_lang="auto"):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)


def update_embedding_from_feedback(user_id, title_text, image, rating):
    user_embeddings = load_user_embeddings()
    if image is not None:
        # read the image
        try:
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            try:
                image = Image.fromarray(np.array(image))
            except Exception as e:
                pass
    # Compute average embedding from title and/or image
    input_is_hindi = is_hindi(title_text) if title_text else False
    if input_is_hindi:
        title_text = translate_text(title_text, "en", "hi")
    avg_embedding = compute_average_embedding(title_text, image)
    update_user_embeddings(
        user_id, user_embeddings, new_embedding=list(avg_embedding), alpha=rating / 5
    )


def save_feedback(user_id, recipe_titles, rating, title_text, image):
    # Save feedback as before
    feedback_file = "./recipe_feedback.csv"  # Path to save feedback
    feedback_data = {
        "user_id": user_id,
        "recipe_titles": recipe_titles,
        "rating": rating,
    }
    
    with open(feedback_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=feedback_data.keys())
        if file.tell() == 0:  # Add header if the file is new
            writer.writeheader()
        writer.writerow(feedback_data)
    # Update embeddings based on rating
    update_embedding_from_feedback(user_id, title_text, image, rating / 5)
# Define FastAPI endpoint