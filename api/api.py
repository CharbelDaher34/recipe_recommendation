# prediction_api.py

from fastapi import FastAPI, HTTPException, Request
import torch
import pandas as pd
import ast
import json
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
from transformers import AutoModel
import numpy as np

model = AutoModel.from_pretrained("./jina_clip_v1_model", trust_remote_code=True)
model = torch.load("./jina.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()


# Functions used
def load_user_embeddings():
    if os.path.exists("./user_embeddings.json"):
        with open("./user_embeddings.json", "r") as f:
            user_embeddings = json.load(f)
    else:
        user_embeddings = {}
    return user_embeddings


def save_user_embeddings(user_embeddings):

    with open("./user_embeddings.json", "w") as f:
        json.dump(user_embeddings, f)
        print("saved")


def get_user_embeddings(user_id, user_embeddings):
    user_id=str(user_id)
    if user_id in user_embeddings.keys():
        return list(user_embeddings[user_id])
    else:
        return None


def update_user_embeddings(user_id, user_embeddings, new_embedding, alpha=0.8):
    user_id=str(user_id)
    # Update the user's embeddings
    if user_id in user_embeddings.keys():
        previous_embedding = torch.tensor(user_embeddings[user_id])
        new_embedding = torch.tensor(new_embedding)
        # Blend the old and new embeddings
        updated_embedding = (1 - alpha) * new_embedding + alpha * previous_embedding
        # Convert to list of floats for JSON serialization
        user_embeddings[user_id] = [float(x) for x in updated_embedding]
    else:
        # Convert to list if adding a new embedding
        user_embeddings[user_id] = [float(x) for x in new_embedding]

    if user_embeddings:
        save_user_embeddings(user_embeddings)


def filter_df(df, **kwargs):
    filtered_df = df.copy()
    for key, value in kwargs.items():
        if key == "Title":
            continue
        if value == [] or value == "" or value is None:
            continue
        if key not in df.columns:
            raise ValueError(f"Column '{key}' is not in the DataFrame.")
        if pd.api.types.is_numeric_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key] <= value]
        elif pd.api.types.is_string_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        elif key == "Cleaned_Ingredients":
            filtered_df = filtered_df[
                filtered_df[key].apply(
                    lambda ingredients: any(
                        ingredient in ingredients for ingredient in value
                    )
                )
            ]
    return filtered_df


def compute_average_embedding(title_text=None, image=None):
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


# Load the DataFrame
df = pd.read_csv("./data.csv")
df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)


# Compute distinct ingredients, cuisines, courses, and diets
distinct_ingredients = set()
for row in df["Cleaned_Ingredients"]:
    for ingredient in row:
        distinct_ingredients.add(ingredient)
distinct_ingredients = sorted(list(distinct_ingredients))

cuisines = df["Cuisine"].dropna().unique().tolist()
cuisines = [cuisine for cuisine in cuisines if cuisine.lower() != "unknown"]

courses = df["Course"].dropna().unique().tolist()
courses = [course for course in courses if course.lower() != "unknown"]

diets = df["Diet"].dropna().unique().tolist()
diets = [diet for diet in diets if diet.lower() != "unknown"]


# Endpoint to handle predictions
@app.post("/predict/")
async def predict(request: Request):
    # try:
    data = await request.json()

    # Access data directly from the request body (as a dictionary)
    user_id = data.get("user_id")
    title_text = data.get("title_text")
    prep_time = data.get("prep_time")
    cook_time = data.get("cook_time")
    selected_cuisines = data.get("selected_cuisines", [])
    selected_courses = data.get("selected_courses", [])
    selected_diets = data.get("selected_diets", [])
    selected_ingredients = data.get("selected_ingredients", [])
    image = data.get("image", None)
    # Filter the DataFrame
    filtered_df = filter_df(
            df,
            Prep_Time=prep_time,
            Cook_Time=cook_time,
            Cuisine=selected_cuisines,
            Course=selected_courses,
            Diet=selected_diets,
            Cleaned_Ingredients=selected_ingredients,
        )
    if filtered_df.empty:
        return "No matching recipes found. Please adjust your inputs.", filtered_df[
                [
                    "Title",
                    "Cuisine",
                    "Course",
                    "Diet",
                    "Prep_Time",
                    "Cook_Time",
                    "Cleaned_Ingredients",
                    "Instructions",
                ]
            ].to_markdown(index=False)
    if image is not None:
        # read the image
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))

    # Compute the average embedding
    avg_embedding = compute_average_embedding(title_text, image)
    # Load user embeddings
    user_embeddings = load_user_embeddings()
    user_embedding = get_user_embeddings(user_id, user_embeddings)
    avg_embedding = np.array(avg_embedding)
    user_embedding = np.array(user_embedding)
    if user_embedding is not None:
        avg_embedding = 0.8 * avg_embedding + 0.2 * user_embedding

    if avg_embedding is None:
        final_df = filtered_df.head(5)

    else:
        top_ids = find_most_similar_recipe(
            avg_embedding, "./embeddings.json", filtered_df, top_n=5
        )
        final_df = filtered_df[filtered_df["ID"].apply(lambda x: x in top_ids)]

    recipe_titles = final_df["Title"].tolist()
    details = (
            final_df[
                [
                    "Title",
                    "Cuisine",
                    "Course",
                    "Diet",
                    "Prep_Time",
                    "Cook_Time",
                    "Cleaned_Ingredients",
                    "Instructions",
                ]
            ].to_markdown(index=False)
            # .to_dict(orient="records")
        )

    return {"titles": recipe_titles, "details": details}

# except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))


@app.get("/dropdown-data/")
async def get_dropdown_data():
    return {
        "cuisines": cuisines,
        "courses": courses,
        "diets": diets,
        "ingredients": distinct_ingredients,
    }


@app.get("/")
async def health_check():
    return {"Api is up"}


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
@app.post("/submit-feedback/")
async def submit_feedback(request: Request):
    # try:
    # Read the JSON data from the request body
    data = await request.json()

    # Extract the relevant fields from the JSON data
    user_id = data.get("user_id")
    recipe_titles = data.get("recipe_titles", [])
    rating = data.get("rating")
    title_text = data.get("title_text")
    
    image = data.get("image")
    try:
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))
    except:
        image=None
    # Save feedback
    save_feedback(user_id, recipe_titles, rating, title_text, image)
    return {"message": "Feedback received"}
# except Exception as e:
#     raise HTTPException(status_code=500, detail="Feedback not received")


import csv
recipes_add_path = "./recipes_add.csv"
@app.post("/add-recipe/")
async def add_recipe(request: Request):
    try:
        data = await request.json()

        # Access data directly from the request body (as a dictionary)
        recipe_name = data.get("recipe_name")
        prep_time = data.get("prep_time")
        cook_time = data.get("cook_time")
        selected_cuisines = data.get("selected_cuisines", [])
        selected_courses = data.get("selected_courses", [])
        selected_diets = data.get("selected_diets", [])
        selected_ingredients = data.get("selected_ingredients", [])
        image_input = data.get("image_input")  # Assuming base64 string

        # Convert lists to strings
        selected_cuisines = selected_cuisines
        selected_courses = selected_courses
        selected_diets =selected_diets
        selected_ingredients = selected_ingredients

        # Define the recipe as a dictionary
        recipe = {
            "Title": recipe_name,
            "Prep_Time": prep_time,
            "Cook_Time": cook_time,
            "Cuisine": selected_cuisines,
            "Course": selected_courses,
            "Diet": selected_diets,
            "Cleaned_Ingredients": selected_ingredients,
            "Image": image_input,  # Already in base64 format
        }

        # Open the CSV file and append the new recipe
        with open(recipes_add_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=recipe.keys())

            # Check if the file is empty to write the header
            if file.tell() == 0:
                writer.writeheader()

            writer.writerow(recipe)

        return {"status": "Recipe added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Feedback not added")


@app.post("/save-review/")
async def save_review(request: Request):
    try:
        data = await request.json()

        # Access data directly from the request body (as a dictionary)
        review_text = data.get("review_text")

        review_file = "./user_reviews.csv"  # Path to save reviews

        # Prepare the review data
        review_data = {"review_text": review_text}

        # Append the review to the CSV file
        with open(review_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=review_data.keys())
            if file.tell() == 0:  # Add header if the file is new
                writer.writeheader()
            writer.writerow(review_data)

        return {"status": "Review saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="review not added")
