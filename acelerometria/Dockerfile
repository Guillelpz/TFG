# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Debian slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libpcre2-dev \
    libboost-all-dev \
    libfftw3-dev \
    libtiff5-dev \
    libjpeg-dev \
    libsndfile1-dev \
    libsndfile1 \
    libgfortran5 \
    libfreetype6-dev \
    libfontconfig1-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libcairo2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libglib2.0-dev \
    libpng-dev \
    libgit2-dev \
    # herramientas necesarias para compilar httpuv/devtools
    autoconf \
    automake \
    libtool \
    pkg-config \
    m4 \
    perl \
    file \
    curl \
    git \
    r-base \
    r-base-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Copy improved R package installer
COPY install_packages.R /app/install_packages.R

# Install R packages with full dependencies
RUN R -e "install.packages('devtools', repos='https://cloud.r-project.org/')"

RUN Rscript /app/install_packages.R

# Add entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port Django will run on
EXPOSE 8000


# Default command to run Gunicorn
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "acelerometria.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "300", "--graceful-timeout", "30", "--access-logfile", "-", "--error-logfile", "-"]
