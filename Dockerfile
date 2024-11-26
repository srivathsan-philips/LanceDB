# Use the official Python image from the Docker Hub
FROM python:3.12-slim-bullseye


RUN apt-get update && \
    pip install --no-cache-dir --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]