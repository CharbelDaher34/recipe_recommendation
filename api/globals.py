from transformers import AutoModel
import torch
import pandas as pd

# Global variables
model = None
df = None
distinct_ingredients = None
cuisines = None
courses = None
diets = None