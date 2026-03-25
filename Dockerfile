# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV and RAM management
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# This is the default, but Render's "Docker Command" setting will override this
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
