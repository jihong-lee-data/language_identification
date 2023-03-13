FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3-pip

WORKDIR /langid

COPY ./requirements.txt /langid/requirements.txt

COPY ./model /langid/model

COPY ./resource /langid/resource

COPY ./app /langid/app

COPY ./run_server.sh /langid/run_server.sh

RUN pip install --no-cache-dir --upgrade -r /langid/requirements.txt

EXPOSE 11000

CMD ["uvicorn", "app.main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "11000"]
