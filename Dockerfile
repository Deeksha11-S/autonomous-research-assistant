FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add Google Chrome repo (new method)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub > /etc/apt/trusted.gpg.d/google.asc
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium

COPY . .

EXPOSE 8000

CMD ["python", "app.py"] 
