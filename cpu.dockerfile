FROM python:3.11-slim

RUN apt-get update -qq \
    && apt-get install -yqq \
      gcc \
      graphviz \
      graphviz-dev

RUN pip install -U pip

WORKDIR /assignment

COPY ./requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
RUN rm ./requirements-cpu.txt

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--notebook-dir", "/assignment"]
