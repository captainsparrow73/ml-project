FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY main.py .

#  Copy models last (so if only models change, only last layer changes)
COPY grid_search_svc.pkl .
COPY vectorizer.pkl .

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
