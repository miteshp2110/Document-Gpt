FROM python:3-alpine3.12

WORKDIR /app
COPY . .

# Install build dependencies
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["python", "main.py"]
