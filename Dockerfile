FROM python:3.9

WORKDIR /app

COPY requirements.txt ./

COPY models ./models/
COPY utils ./utils/
COPY widgets ./widgets/
COPY main.py ./
COPY setup.py ./

RUN pip install -r requirements.txt
RUN python setup.py

RUN git clone https://github.com/AIM-Harvard/pyradiomics.git
RUN cd pyradiomics
RUN python setup.py install
RUN python setup.py build_ext --inplace

RUN apt-get update \
 && apt-get install -y libx11-6 libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libxcursor-dev libxtst-dev tk-dev ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py"]

