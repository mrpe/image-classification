FROM ambolt/emily:1.0.0-torch-cv

WORKDIR /workspace

# Static project file that allows us to COPY, RUN and cache setup scripts conditionally
ARG EMILY_IDREQ=.emily/.pid

# Build arguments
# Provided by Emily when opening projects in PyCharm
ARG CONFIGURE_PYCHARM_SSH_SCRIPT

# Run and cache the SSH configuration required for Pycharm if the script is present
COPY $EMILY_IDREQ $CONFIGURE_PYCHARM_SSH_SCRIPT ./
RUN if [ ! -z "$CONFIGURE_PYCHARM_SSH_SCRIPT" ]; then \
  SCRIPT=$(basename "$CONFIGURE_PYCHARM_SSH_SCRIPT"); \
  PID=$(basename "$EMILY_IDREQ"); \
  sudo /bin/bash "$SCRIPT"; \
  sudo rm $SCRIPT; \
  sudo rm $PID; \
fi;

# Add your changes here e.g. apt package installations
# RUN apt-get --allow-releaseinfo-change -y update && apt-get install -y #     package1 #     package2 ... 

# Create and set the project user
RUN adduser --disabled-password --gecos "" --home /home/image-classific --force-badname image-classific
RUN adduser image-classific sudo
RUN chown image-classific /workspace
RUN chown image-classific /home/image-classific
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER image-classific

ENV PATH="/home/image-classific/.local/bin:$PATH"

# Install Pip requirements
COPY requirements.txt requirements.txt
RUN pip install --disable-pip-version-check -r requirements.txt

# Copy all directory contents (sans .dockerignore) into container
COPY . .

CMD ["/bin/bash", "run.sh"]
