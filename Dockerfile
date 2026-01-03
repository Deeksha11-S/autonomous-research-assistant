FROM python:3.11-slim

# Install only absolutely essential packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    # Minimal dependencies for headless browsing
    libnss3 \
    libxss1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright with minimal browser
RUN python -m playwright install chromium

# Copy app
COPY . .

EXPOSE ${PORT:-8000}

CMD ["python", "app.py"]
