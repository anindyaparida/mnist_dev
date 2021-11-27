# mnist
Scikit-learn mnist dataset
Question 1:

<img width="831" alt="model_hypertune" src="https://user-images.githubusercontent.com/78500544/143684151-2bb17cc7-be2d-4143-a69c-06d6d53cb15c.PNG">

Question 2:
Docker Commands:

FROM ubuntu:18.04

COPY mnist_dev /exp/mnist_dev

COPY requirements.txt /exp/requirements.txt
\
RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install --no-cache-dir -r /exp/requirements.txt

RUN mkdir /exp/models

WORKDIR /exp

CMD ["python3", "./mnist_dev/mnist_hypertune.py"]
