FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
ARG UID
ARG GID

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    sudo build-essential git make nano xterm \
    libc6-dbg gdb valgrind \
    openmpi-bin libopenmpi-dev \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for OpenMPI, as it does not like to run
# applications as root. This is a very hacky workaround for the issue.
# The code below creates a user with the same UID and GID as the
# user running the Docker container and name it 'steps'. While some
# systems would allow the user to be created with the same UID and GID as
# the host user, some systems would not. This is why an explicit deletion
# of the user with the same UID is performed if it exists. This happens
# ONLY INSIDE the Docker image, and does not affect the host system.
RUN \
    # 1. Handle the group with our desired GID
    if getent group $GID > /dev/null 2>&1; then \
        # If a group with our GID exists but isn't 'steps', rename it
        if [ "$(getent group $GID | cut -d: -f1)" != "steps" ]; then \
            groupmod -n steps $(getent group $GID | cut -d: -f1); \
        fi; \
    else \
        # If no group with our GID exists, create it
        groupadd --gid $GID steps; \
    fi && \
    # 2, Handle the user with our desired UID
    if getent passwd steps > /dev/null 2>&1; then \
        # If user exists, update its UID and GID
        usermod -u $UID -g $GID steps; \
    else \
        # If UID is taken by another user, remove that user
        if getent passwd $UID > /dev/null 2>&1; then \
            userdel $(getent passwd $UID | cut -d: -f1); \
        fi && \
        # Create new user with our specifications
        useradd -l -m --uid $UID --gid $GID steps; \
    fi && \
    # Set up sudo access and fix home directory permissions
    echo 'steps ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    chown -R steps:steps /home/steps
# Set the non-root user as the default user to use OpenMPI with
USER steps

# Clone the software repository from GitHub
#RUN git clone https://github.com/eltevo/StePS.git /home/steps/StePS

# Set the working directory
WORKDIR /home/steps/StePS
