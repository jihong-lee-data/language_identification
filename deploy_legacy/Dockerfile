FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3-pip
COPY ./requirements.txt requirements.txt
COPY ./module/ module
COPY ./model/ model
COPY ./app.py app.py

RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python3" ] 
CMD [ "app.py" ] 
