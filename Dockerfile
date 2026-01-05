FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better cache)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy the full project (INCLUDING src/)
COPY . .

# Install the project itself (editable)
RUN uv pip install -e .

# Default command
CMD ["uv", "run", "train"]