FROM python:3.8


RUN apt-get update
RUN apt-get -y install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev


RUN git clone https://github.com/moves-rwth/storm.git
WORKDIR /storm
RUN mkdir build
WORKDIR /storm/build
RUN cmake ..
RUN make -j 1
WORKDIR /
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    maven \
    uuid-dev \
    python3 \
    virtualenv

RUN git clone https://github.com/moves-rwth/pycarl.git
WORKDIR /pycarl
RUN python3 setup.py build_ext --jobs 1 develop

WORKDIR /

RUN git clone https://github.com/moves-rwth/stormpy.git
WORKDIR /stormpy
RUN python3.8 setup.py build_ext --storm-dir /storm/build/ --jobs 1 develop

WORKDIR /
COPY requirements.txt .
RUN pip3.8 install -r requirements.txt


COPY common .
COPY custom_openai_gyms .
COPY openai_gym_training .
COPY safe_gym_training .
COPY unit_testing .
COPY verify_rl_agent .
COPY control_ui.py .
COPY cool_mc.py .
COPY start_ui.sh .


ENTRYPOINT /bin/bash