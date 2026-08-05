# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system dependencies for audio processing (libsndfile/ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirement files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Configure Streamlit environment variables to optimize container execution
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application (assumes main app script is named app.py)
CMD ["streamlit", "run", "app.py"]
