# Use official Python image as base
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the script
CMD ["python", "stock_prediction.py"]
