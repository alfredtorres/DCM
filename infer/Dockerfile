# Use an appropriate base image with GPU support
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Copy inference script and requirements
COPY ./ /workspace/

ENV PATH="/opt/conda/bin:$PATH"

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Chmod the prediction script
RUN chmod +x /workspace/predict.sh

# Set default command
CMD ["/bin/bash"]
