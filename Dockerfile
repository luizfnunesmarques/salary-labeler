# Use the official Python image with tag 3.8 as the base image
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG USER_ID=9000
ARG GROUP_ID=9000

RUN useradd -m -u $USER_ID modelapi

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first
COPY requirements_prod.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application files
COPY setup.py train_model.py /app/
COPY model/ /app/model/
COPY config/ /app/config/
COPY ml/ /app/ml/
COPY data/ /app/data/
COPY api/ /app/api/

COPY artifacts/encoder.pkl /app/artifacts/
COPY artifacts/regressor_model.pkl /app/artifacts/


# Change ownership of files to the created user
RUN chown -R modelapi:modelapi /app

# Switch to the created user
USER modelapi

# Expose the port and specify the command to run the server
EXPOSE 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "api.main:app"]
