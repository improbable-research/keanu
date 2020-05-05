FROM ubuntu:16.04

# Set environment variables - keep these alpha-sorted

# Set up core apt requirements.
# Do not add build requirements here.
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
    software-properties-common \
    jq \
    curl \
    apt-transport-https

# Add new repositories and their signing keys here.
RUN add-apt-repository ppa:git-core/ppa \
    && add-apt-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main" \
    && curl -LSs "https://dl-ssl.google.com/linux/linux_signing_key.pub" | apt-key add - \
    && curl -LSs "https://apt.llvm.org/llvm-snapshot.gpg.key" | apt-key add -

# Install build dependency packages.
# Keep these alpha-sorted.
RUN apt-get -qq update && \
    apt-get -qq install -y  --no-install-recommends \
    bison=2:3.0.4.dfsg-1 \
    bash-completion=1:2.1-4.2ubuntu1.1 \
    build-essential=12.1ubuntu2 \
    ca-certificates=20170717~16.04.2 \
    clang=1:3.8-33ubuntu3.1 \
    clang-format-3.8=1:3.8-2ubuntu4 \
    clang-format-7=1:7.1.0~svn353565-1~exp1~20190408084827.60 \
    curl=7.47.0-1ubuntu2.14 \
    g++=4:5.3.1-1ubuntu1 \
    git \
    make=4.1-6 \
    openjdk-8-jdk \
    shellcheck=0.3.7-5 \
    ssh=1:7.2p2-4ubuntu2.8 \
    unzip=6.0-20ubuntu1 \
    zip=3.0-11 \
    zlib1g-dev \
    python3-pip=8.1.1-2ubuntu0.4 \
    python3-setuptools=20.7.0-1

# Be on latest python package manager && install prometheus client
RUN pip3 install --upgrade pip
RUN pip3 install pipenv
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Download and install gosu, so we can drop privs after the container starts.
RUN curl -LSs -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && chmod +x /usr/local/bin/gosu

# Add github public SSH RSA to prevent fingerprint confirmation
RUN mkdir -p /etc/ssh && echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==" > /etc/ssh/ssh_known_hosts

# Copy the entrypoint into the image and set it as the target.
COPY bin/ /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
