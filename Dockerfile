# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the Python script and data files into the container
COPY train_linear_regression.py /app/
COPY data.txt /app/
COPY data.db /app/

# Install required Python packages
RUN pip install --no-cache-dir pandas scikit-learn joblib

# Command to run the script when the container starts
CMD ["python", "train_linear_regression.py"]
