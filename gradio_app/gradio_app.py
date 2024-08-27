# gradio_app.py
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time

api_url = "http://api:8000/predict/"
dropdown_data_url = "http://api:8000/dropdown-data/"
feedback_url = "http://api:8000/submit-feedback/"
add_recipe_url = "http://api:8000/add-recipe/"
save_review_url = "http://api:8000/save-review/"
# Fetch dropdown data from the API
while True:
    try:
        response = requests.get(dropdown_data_url)
        if response.status_code == 200:
            break
        else:
            time.sleep(2)
    except:
        pass
dropdown_data = response.json()

cuisines = dropdown_data.get("cuisines", [])
courses = dropdown_data.get("courses", [])
diets = dropdown_data.get("diets", [])
ingredients = dropdown_data.get("ingredients", [])

# Custom CSS for styling and dynamic effects
custom_css = """
body {background-color: #f3f4f6; font-family: 'Helvetica', sans-serif; color: #1a1a1a; margin: 0; padding: 0;}
#header {background: linear-gradient(to right, #003366, #336699, #6699cc, #99ccff, #cce6ff); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px;}
#header h1 {margin: 0; font-size: 2.5em; font-weight: bold; letter-spacing: 1.5px;}
#nav {display: flex; justify-content: center; padding: 10px 0; margin-bottom: 20px;}
#nav a {color: #1a1a1a; text-decoration: none; font-weight: bold; margin: 0 15px; font-size: 1.1em; cursor: pointer; background-color: transparent; border: none; outline: none; padding: 0;}
#nav a:hover {color: #336699; text-decoration: underline;}
"""


def image_to_base64(image):
    """Encodes a Pillow Image object into a base64 string.

    Args:
      image: A Pillow Image object.

    Returns:
      A base64-encoded string representing the image.
    """
    try:
        image = Image.fromarray(np.array(image))
    except Exception as e:
        pass

    # Convert the image to JPEG format for better compatibility
    image = image.convert("RGB")

    # Save the image to a BytesIO object in JPEG format
    img_bytes = BytesIO()
    image.save(img_bytes, format="JPEG")

    # Encode the image data as base64
    base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    return base64_str


def recommend_recipes(
    user_id,
    title_text,
    prep_time,
    cook_time,
    selected_cuisines=[],
    selected_courses=[],
    selected_diets=[],
    selected_ingredients=[],
    image=None,
):
    try:
        # Convert image to base64 if provided
        image_base64 = None
        if image is not None:
            image_base64 = image_to_base64(
                image
            )  # Implement this function to convert image to base64
        if prep_time is None:
            prep_time = 90
        if cook_time is None:
            cook_time = 90
        if selected_cuisines is None:
            selected_cuisines = []
        if selected_courses is None:
            selected_courses = []
        if selected_diets is None:
            selected_diets = []
        if selected_ingredients is None:
            selected_ingredients = []
        # Prepare the data for the API request
        payload = {
            "user_id": user_id,
            "title_text": title_text,
            "prep_time": prep_time,
            "cook_time": cook_time,
            "selected_cuisines": selected_cuisines,
            "selected_courses": selected_courses,
            "selected_diets": selected_diets,
            "selected_ingredients": selected_ingredients,
            "image": image_base64,
        }

        # Send the request to the API
        response = requests.post(api_url, json=payload)
        response_data = response.json()

        if "titles" not in response_data:
            return response_data.get("message", "An error occurred"), ""

        recipe_titles = "\n".join(response_data["titles"])
        details = response_data["details"]

        return (
            recipe_titles,
            details,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    except Exception as e:
        return (
            str(e),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def handle_feedback(user_id, recipe_titles, feedback, title_text, image_input):
    try:
        # Prepare the JSON data to send in the request
        image_base64 = None
        if image_input is not None:
            image_base64 = image_to_base64(image_input)
        data = {
            "user_id": str(user_id),
            "recipe_titles": recipe_titles,
            "rating": feedback,
            "title_text": title_text,
            "image": image_base64,
        }

        # Send a POST request to the endpoint
        response = requests.post(feedback_url, json=data)

        # Check the response
        if response.status_code == 200:
            print(response.json())  # Should print: {"message": "Feedback received"}
            return gr.update(value="Thank you for your feedback!", visible=True)

        else:
            print(f"Error: {response.status_code}, {response.text}")
            return gr.update(value="Feedback not added", visible=True)

    except Exception as e:
        return gr.update(value=f"Error saving feedback: {str(e)}", visible=True)


def imageTo64(image):
    """Encodes a Pillow Image object into a base64 string.

    Args:
      image: A Pillow Image object.

    Returns:
      A base64-encoded string representing the image.
    """
    try:
        # If the input is not already a Pillow Image, convert it
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert the image to JPEG format for better compatibility
        image = image.convert("RGB")

        # Save the image to a BytesIO object in JPEG format
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")

        # Encode the image data as base64
        base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        return base64_str

    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


# Custom CSS for styling and dynamic effects
custom_css = """
body {background-color: #f3f4f6; font-family: 'Helvetica', sans-serif; color: #1a1a1a; margin: 0; padding: 0;}
#header {background: linear-gradient(to right, #003366, #336699, #6699cc, #99ccff); color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 20px;}
#header h1 {margin: 0; font-size: 2.5em; font-weight: bold; letter-spacing: 1.5px;}
#nav {display: flex; justify-content: center; padding: 10px 0; margin-bottom: 20px;}
#nav a {color: #1a1a1a; text-decoration: none; font-weight: bold; margin: 0 15px; font-size: 1.1em; cursor: pointer; background-color: transparent; border: none; outline: none; padding: 0;}
#nav a:hover {color: #336699; text-decoration: underline;}
#feedback_message {background-color: #336699; color: #5580A3; padding: 15px; border-radius: 5px; margin-top: 20px; text-align: center; font-weight: bold;}
#review_message {background-color: #336699; color: #5580A3; padding: 15px; border-radius: 5px; margin-top: 20px; text-align: center; font-weight: bold;}
#submit_message {background-color: #336699; color: #5580A3; padding: 15px; border-radius: 5px; margin-top: 20px; text-align: center; font-weight: bold;}
"""


# Functions to switch pages
def show_overview():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),  # Clear review textbox
        gr.update(value="", visible=False),  # Hide review message
        gr.update(value=""),  # Clear recipe name
        gr.update(value=30),  # Reset prep time slider
        gr.update(value=30),  # Reset cook time slider
        gr.update(value=[]),  # Clear selected cuisines
        gr.update(value=[]),  # Clear selected courses
        gr.update(value=[]),  # Clear selected diets
        gr.update(value=[]),  # Clear selected ingredients
        gr.update(value=None),  # Clear image input
        gr.update(value="", visible=False),  # Hide submit message
    )


def show_terms():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),  # Clear review textbox
        gr.update(value="", visible=False),  # Hide review message
        gr.update(value=""),  # Clear recipe name
        gr.update(value=30),  # Reset prep time slider
        gr.update(value=30),  # Reset cook time slider
        gr.update(value=[]),  # Clear selected cuisines
        gr.update(value=[]),  # Clear selected courses
        gr.update(value=[]),  # Clear selected diets
        gr.update(value=[]),  # Clear selected ingredients
        gr.update(value=None),  # Clear image input
        gr.update(value="", visible=False),  # Hide submit message
    )


def show_recipes():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),  # Clear review textbox
        gr.update(value="", visible=False),  # Hide review message
        gr.update(value=""),  # Clear recipe name
        gr.update(value=30),  # Reset prep time slider
        gr.update(value=30),  # Reset cook time slider
        gr.update(value=[]),  # Clear selected cuisines
        gr.update(value=[]),  # Clear selected courses
        gr.update(value=[]),  # Clear selected diets
        gr.update(value=[]),  # Clear selected ingredients
        gr.update(value=None),  # Clear image input
        gr.update(value="", visible=False),  # Hide submit message
    )


def show_review_page():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),  # Clear recipe name
        gr.update(value=30),  # Reset prep time slider
        gr.update(value=30),  # Reset cook time slider
        gr.update(value=[]),  # Clear selected cuisines
        gr.update(value=[]),  # Clear selected courses
        gr.update(value=[]),  # Clear selected diets
        gr.update(value=[]),  # Clear selected ingredients
        gr.update(value=None),  # Clear image input
        gr.update(value="", visible=False),  # Hide submit message
    )


def show_submit_recipe_page():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),  # Clear review textbox
        gr.update(value="", visible=False),  # Hide review message
    )


def handle_review_submission(review_text):

    # Data to be sent in the JSON request body
    review_data = {"review_text": review_text}

    # Send POST request
    response = requests.post(save_review_url, json=review_data)

    # Check response
    if response.status_code == 200:
        gr.update(value="Thank you for your review!", visible=True),
        gr.update(value="")

    else:
        return (
            gr.update(value="Error saving review", visible=True),
            gr.update(value=""),
        )


def handle_recipe_submission(
    recipe_name,
    prep_time,
    cook_time,
    selected_cuisines,
    selected_courses,
    selected_diets,
    selected_ingredients,
    image_input,
):
    # Check if all required fields are filled out
    if (
        not recipe_name
        or not selected_cuisines
        or not selected_courses
        or not selected_diets
        or not selected_ingredients
    ):
        return gr.update(
            value="Error: Please fill out all required fields.", visible=True
        )

    # Data to be sent in the JSON request body
    recipe_data = {
        "recipe_name": recipe_name,
        "prep_time": prep_time,
        "cook_time": cook_time,
        "selected_cuisines": selected_cuisines,
        "selected_courses": selected_courses,
        "selected_diets": selected_diets,
        "selected_ingredients": selected_ingredients,
        "image_input": imageTo64(
            image_input
        ),  # Base64 string of the image can be added here
    }

    # Send POST request
    response = requests.post(add_recipe_url, json=recipe_data)

    # Check response
    if response.status_code == 200:
        return gr.update(
            value="Thank you for submitting a recipe of your choice! We'll get back to you whether it was approved or denied!",
            visible=True,
        )
    else:
        return gr.update(value="Recipe not submitted", visible=False)


# Define the Gradio Blocks interface with navigation
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<div id='header'><h1>🍲 MAGIC CHEF </h1></div>")

    with gr.Row(elem_id="nav"):
        overview_btn = gr.Button("Overview", elem_classes="nav-link")
        terms_btn = gr.Button("Terms", elem_classes="nav-link")
        recipes_btn = gr.Button("Recipes", elem_classes="nav-link")
        review_btn = gr.Button("Leave a Review", elem_classes="nav-link")
        submit_recipe_nav_btn = gr.Button(
            "Submit Your Recipe", elem_classes="nav-link"
        )  # Renamed for clarity

    with gr.Column(visible=True) as overview_page:
        gr.Markdown(
            """
        ## The Ultimate Recipe API for Your Culinary Needs

        Our expert team has meticulously built an extensive food database designed to understand the intricate connections between ingredients, recipes, cooking methods, and dietary preferences.

        With our API, you can easily discover recipes that match your specific requirements, whether they're based on preparation time, cuisine, or dietary needs.

        ### Why Choose Magic Chef?

        - **Intelligent Filtering:** Our system intelligently filters and suggests recipes, ensuring that you find exactly what you're looking for.
        - **Customizable Results:** Whether you're searching for a quick weeknight dinner, a vegetarian delight, or something that aligns with your unique dietary restrictions, our API has got you covered.
        - **Comprehensive Data:** Access a database that's as diverse as your culinary imagination.

        Magic Chef transforms your kitchen experience by providing personalized recipe recommendations at your fingertips.
        """
        )

    with gr.Column(visible=False) as terms_page:
        gr.Markdown("This is the terms page.")

    with gr.Column(visible=False) as recipe_page:
        with gr.Row():
            with gr.Column():
                user_id = gr.Number(
                    label="User ID", value=0, precision=0, interactive=True
                )
                title_text = gr.Textbox(
                    label="Recipe Name (optional)",
                    placeholder="e.g., Chicken Curry",
                    interactive=True,
                )
                prep_time = gr.Slider(
                    0,
                    120,
                    step=1,
                    label="Max Preparation Time (in minutes)",
                    value=30,
                    interactive=True,
                )
                cook_time = gr.Slider(
                    0,
                    120,
                    step=1,
                    label="Max Cooking Time (in minutes)",
                    value=30,
                    interactive=True,
                )
                selected_cuisines = gr.Dropdown(
                    cuisines, multiselect=True, label="Select Cuisine(s)"
                )
                selected_courses = gr.Dropdown(
                    courses, multiselect=True, label="Select Course(s)"
                )
                selected_diets = gr.Dropdown(
                    diets, multiselect=True, label="Select Diet(s)"
                )
                selected_ingredients = gr.Dropdown(
                    ingredients, multiselect=True, label="Search and Select Ingredients"
                )
                image_input = gr.Image(label="Upload an Image (optional)")

        recipe_titles = gr.Textbox(label="Top 5 Matching Recipes")
        details = gr.Markdown(label="Recipe Details", elem_id="recipe_details")
        find_recipes_btn = gr.Button("Find Recipes")
        feedback_slider = gr.Slider(
            1, 5, step=1, label="Rate the Recommendations", value=3, visible=False
        )
        submit_feedback_btn = gr.Button("Submit Your Feedback", visible=False)
        feedback_message = gr.Markdown(
            value="", elem_id="feedback_message", visible=False
        )

        find_recipes_btn.click(
            fn=recommend_recipes,
            inputs=[
                user_id,
                title_text,
                prep_time,
                cook_time,
                selected_cuisines,
                selected_courses,
                selected_diets,
                selected_ingredients,
                image_input,
            ],
            outputs=[
                recipe_titles,
                details,
                feedback_slider,
                submit_feedback_btn,
                feedback_message,
            ],
        )
        #             inputs=[
        #                 user_id,
        #                 title_text,
        #                 prep_time,
        #                 cook_time,
        #                 selected_cuisines,
        #                 selected_courses,
        #                 selected_diets,
        #                 selected_ingredients,
        #                 image_input,
        #             ],
        #             outputs=[
        #                 recipe_titles,
        #                 details,
        #                 feedback_slider,
        #                 submit_feedback_btn,
        #                 feedback_message,
        #             ],
        #         )
        submit_feedback_btn.click(
            fn=handle_feedback,
            inputs=[user_id, recipe_titles, feedback_slider, title_text, image_input],
            outputs=feedback_message,
        )

    with gr.Column(visible=False) as review_page:
        review_textbox = gr.Textbox(
            label="Write your review here", placeholder="Write your review...", lines=5
        )
        submit_review_btn = gr.Button("Submit Your Review")
        review_message = gr.Markdown(value="", elem_id="review_message", visible=False)

        submit_review_btn.click(
            fn=handle_review_submission,
            inputs=review_textbox,
            outputs=[
                review_message,
                review_textbox,
            ],  # Reset the textbox after submission
        )

    with gr.Column(visible=False) as submit_recipe_page:
        with gr.Row():
            with gr.Column():
                recipe_name = gr.Textbox(
                    label="Recipe Name",
                    placeholder="e.g., My Special Pasta",
                    interactive=True,
                )
                prep_time = gr.Slider(
                    0,
                    120,
                    step=1,
                    label="Max Preparation Time (in minutes)",
                    value=30,
                    interactive=True,
                )
                cook_time = gr.Slider(
                    0,
                    120,
                    step=1,
                    label="Max Cooking Time (in minutes)",
                    value=30,
                    interactive=True,
                )
                selected_cuisines = gr.Dropdown(
                    cuisines, multiselect=False, label="Select Cuisine(s)"
                )
                selected_courses = gr.Dropdown(
                    courses, multiselect=False, label="Select Course(s)"
                )
                selected_diets = gr.Dropdown(
                    diets, multiselect=False, label="Select Diet(s)"
                )
                selected_ingredients = gr.Dropdown(
                    ingredients, multiselect=True, label="Select Ingredients"
                )
                image_input = gr.Image(
                    label="Upload an Image of Your Recipe (optional)"
                )

        submit_recipe_btn = gr.Button("Submit Your Recipe Here!")
        submit_message = gr.Markdown(value="", elem_id="submit_message", visible=False)

        submit_recipe_btn.click(
            fn=handle_recipe_submission,
            inputs=[
                recipe_name,
                prep_time,
                cook_time,
                selected_cuisines,
                selected_courses,
                selected_diets,
                selected_ingredients,
                image_input,
            ],
            outputs=submit_message,
        )

    # Navigation button clicks
    overview_btn.click(
        show_overview,
        outputs=[
            overview_page,
            terms_page,
            recipe_page,
            review_page,
            submit_recipe_page,
            review_textbox,
            review_message,
        ],
    )
    terms_btn.click(
        show_terms,
        outputs=[
            overview_page,
            terms_page,
            recipe_page,
            review_page,
            submit_recipe_page,
            review_textbox,
            review_message,
        ],
    )
    recipes_btn.click(
        show_recipes,
        outputs=[
            overview_page,
            terms_page,
            recipe_page,
            review_page,
            submit_recipe_page,
            review_textbox,
            review_message,
        ],
    )
    review_btn.click(
        show_review_page,
        outputs=[
            overview_page,
            terms_page,
            recipe_page,
            review_page,
            submit_recipe_page,
        ],
    )
    submit_recipe_nav_btn.click(
        show_submit_recipe_page,
        outputs=[
            overview_page,
            terms_page,
            recipe_page,
            review_page,
            submit_recipe_page,
        ],
    )

demo.launch(server_name="0.0.0.0", server_port=8010, debug=True)


# # Functions to switch pages
# def show_overview():
#     return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


# def show_terms():
#     return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


# def show_recipes():
#     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

# # Define the Gradio Blocks interface with navigation
# with gr.Blocks(css=custom_css) as demo:
#     gr.Markdown("<div id='header'><h1>🍲 MAGIC CHEF </h1></div>")

#     with gr.Row(elem_id="nav"):
#         overview_btn = gr.Button("Overview", elem_classes="nav-link")
#         terms_btn = gr.Button("Terms", elem_classes="nav-link")
#         recipes_btn = gr.Button("Recipes", elem_classes="nav-link")

#     with gr.Column(visible=True) as overview_page:
#         gr.Markdown(
#             """
#         ## The Ultimate Recipe API for Your Culinary Needs

#         Our expert team has meticulously built an extensive food database designed to understand the intricate connections between ingredients, recipes, cooking methods, and dietary preferences.

#         With our API, you can easily discover recipes that match your specific requirements, whether they're based on preparation time, cuisine, or dietary needs.

#         ### Why Choose Magic Chef?

#         - **Intelligent Filtering:** Our system intelligently filters and suggests recipes, ensuring that you find exactly what you're looking for.
#         - **Customizable Results:** Whether you're searching for a quick weeknight dinner, a vegetarian delight, or something that aligns with your unique dietary restrictions, our API has got you covered.
#         - **Comprehensive Data:** Access a database that's as diverse as your culinary imagination.

#         Magic Chef transforms your kitchen experience by providing personalized recipe recommendations at your fingertips.
#         """
#         )

#     with gr.Column(visible=False) as terms_page:
#         gr.Markdown("This is the terms page.")

#     with gr.Column(visible=False) as recipe_page:
#         with gr.Row():
#             with gr.Column():
#                 user_id = gr.Number(
#                     label="User ID", value=0, precision=0, interactive=True
#                 )
#                 title_text = gr.Textbox(
#                     label="Recipe Name (optional)",
#                     placeholder="e.g., Chicken Curry",
#                     interactive=True,
#                 )
#                 prep_time = gr.Slider(
#                     0,
#                     120,
#                     step=1,
#                     label="Max Preparation Time (in minutes)",
#                     value=30,
#                     interactive=True,
#                 )
#                 cook_time = gr.Slider(
#                     0,
#                     120,
#                     step=1,
#                     label="Max Cooking Time (in minutes)",
#                     value=30,
#                     interactive=True,
#                 )
#                 selected_cuisines = gr.Dropdown(
#                     cuisines, multiselect=True, label="Select Cuisine(s)"
#                 )
#                 selected_courses = gr.Dropdown(
#                     courses, multiselect=True, label="Select Course(s)"
#                 )
#                 selected_diets = gr.Dropdown(
#                     diets, multiselect=True, label="Select Diet(s)"
#                 )
#                 selected_ingredients = gr.Dropdown(
#                     ingredients, multiselect=True, label="Search and Select Ingredients"
#                 )
#                 image_input = gr.Image(label="Upload an Image (optional)")

#         recipe_titles = gr.Textbox(label="Top 5 Matching Recipes")
#         details = gr.Markdown(label="Recipe Details", elem_id="recipe_details")
#         find_recipes_btn = gr.Button("Find Recipes")
#         feedback_slider = gr.Slider(
#             1, 5, step=1, label="Rate the Recommendations", value=3, visible=False
#         )
#         submit_feedback_btn = gr.Button("Submit Your Feedback", visible=False)
#         feedback_message = gr.Markdown(
#             value="", elem_id="feedback_message", visible=False
#         )

#         find_recipes_btn.click(
#             fn=recommend_recipes,
#             inputs=[
#                 user_id,
#                 title_text,
#                 prep_time,
#                 cook_time,
#                 selected_cuisines,
#                 selected_courses,
#                 selected_diets,
#                 selected_ingredients,
#                 image_input,
#             ],
#             outputs=[
#                 recipe_titles,
#                 details,
#                 feedback_slider,
#                 submit_feedback_btn,
#                 feedback_message,
#             ],
#         )

#         submit_feedback_btn.click(
#             fn=handle_feedback,
#             inputs=[user_id, recipe_titles, feedback_slider, title_text, image_input],
#             outputs=feedback_message,
#         )

#     overview_btn.click(show_overview, outputs=[overview_page, terms_page, recipe_page])
#     terms_btn.click(show_terms, outputs=[overview_page, terms_page, recipe_page])
#     recipes_btn.click(show_recipes, outputs=[overview_page, terms_page, recipe_page])

# demo.launch(server_name="0.0.0.0", server_port=8010, debug=True)
