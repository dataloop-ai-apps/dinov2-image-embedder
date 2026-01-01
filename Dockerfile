FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_pytorch2

USER 1000
WORKDIR /tmp
ENV HOME=/tmp

RUN pip install --upgrade pip && pip install --user transformers

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dinov2-adapter:0.1.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dinov2-adapter:0.1.1
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dinov2-adapter:0.1.1 bash