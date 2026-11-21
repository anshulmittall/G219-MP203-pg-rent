# Small official Python image
FROM python:3.12-slim

WORKDIR /app

# Install libraries first: this layer is cached and only rebuilt when
# requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy only what the service needs: the API code and the champion model.
COPY app/ app/
COPY models/ models/

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
