ARG BASE_IMG=pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

FROM ${BASE_IMG} 

ENV PATH="/opt/ml/code:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    jq

RUN ldconfig -v

RUN apt-get install -y git
RUN cd /opt && git clone https://github.com/aosokin/os2d

RUN apt-get install ffmpeg libsm6 libxext6 -y

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN pip install flask gevent gunicorn boto3

ENV PATH="/opt/os2d:${PATH}"
WORKDIR /opt/ml/code
COPY serve.py /opt/ml/code
COPY wsgi.py /opt/ml/code
COPY predictor.py /opt/ml/code
COPY nginx.conf /opt/ml/code
COPY bestv1.pt /opt/ml/code
ENTRYPOINT ["python", "serve.py"]
