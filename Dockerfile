FROM python:3.10-slim

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Use a virtualenv-style isolation via pip install --user
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Use gunicorn for production-style serving; 2 workers is reasonable for small app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
