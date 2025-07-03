
FROM python:3.10-slim

# Set the working directory in the container

WORKDIR /app

# Copy the dependency file into the container at /app
COPY app_requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r app_requirements.txt

# Copy the application code and the model into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Run the application using Gunicorn
# 0.0.0.0 makes it accessible from outside the container
# app:app refers to 'app.py' file and 'app' Flask instance
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "app:app"]