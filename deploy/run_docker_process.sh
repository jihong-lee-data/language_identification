#!/bin/bash

## dockerize 
echo dockerizing...
sudo docker build -t langid .
echo done

## remove running container forcefully
echo removing existing container...
sudo docker ps -a | grep -i language_identification | awk '{ print $1 }' | xargs sudo docker rm -f 
echo done

## running new docker container
echo running new docker container
sudo docker run -p 11000:11000 -d --name language_identification langid:latest 
echo done