# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8008 to the outside world
EXPOSE 8010

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run Gradio app
CMD ["python", "gradio_app.py"]
