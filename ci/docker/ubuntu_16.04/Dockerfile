FROM ubuntu:18.04

# Set environment variables - keep these alpha-sorted

# Set up core apt requirements.
# Do not add build requirements here.
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
    software-properties-common \
    jq \
    curl \
    apt-transport-https

# Install build dependency packages.
# Keep these alpha-sorted.
RUN apt-get -qq update && \
    apt-get -qq install -y  --no-install-recommends \
    bison \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    g++ \
    git \
    openjdk-8-jdk \
    shellcheck \
    ssh \
    unzip \
    zip \
    zlib1g-dev \
    python3-pip \
    python3-setuptools

RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN pip3 install pipenv
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pipenv --version

# Download and install gosu, so we can drop privs after the container starts.
RUN curl -LSs -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && chmod +x /usr/local/bin/gosu

# Add github public SSH RSA to prevent fingerprint confirmation
RUN mkdir -p /etc/ssh && echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" > /etc/ssh/ssh_known_hosts

# Copy the entrypoint into the image and set it as the target.
COPY bin/ /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
