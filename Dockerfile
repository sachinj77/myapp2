FROM python:3.11
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN ls /app
RUN pip install -r requirements.txt
EXPOSE 8506
ENTRYPOINT ["streamlit", "run", "inventory_streamlit.py", "--server.port=8506", "--server.address=0.0.0.0"]
