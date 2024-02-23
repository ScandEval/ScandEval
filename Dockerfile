FROM python:3.11-slim-bookworm

# Install dependencies
RUN pip install scandeval[generative]

# Move the existing evaluation results into the container, to avoid re-running the
# evaluation
WORKDIR /project
COPY scandeval_benchmark_results* /project

# Set the environment variable with the evaluation arguments. These can be overridden
# when running the container
ENV MODEL "missing-model"

# Run the script
CMD ["scandeval", "-m", "${MODEL}"]
