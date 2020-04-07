FROM python:3.6
MAINTAINER Fernando Trevino - fernando@yaneztrevino.com

ADD https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 mask_rcnn_coco.h5
ADD mrcnn mrcnn
ADD usage usage
ADD samples samples

COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install -r requirements.txt && \
    python setup.py install && \
    pip install pycocotools

WORKDIR usage

ENTRYPOINT ["python", "run_inference.py"]

