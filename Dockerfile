FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt evo

COPY degrade_imu.py .
COPY src ./src
COPY tools ./tools
COPY README.md .

# One-command bench (synthetic + validate)
CMD ["python", "tools/run_synthetic_benchmark.py"]
