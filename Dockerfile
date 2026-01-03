FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Playwright (new method without apt-key)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub > /etc/apt/trusted.gpg.d/google.asc
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium  # Install system dependencies for Playwright

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "app.py"]