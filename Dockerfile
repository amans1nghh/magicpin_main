FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expand dataset at build time (deterministic, no API needed)
RUN python main.py --expand-dataset

EXPOSE 8080

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
