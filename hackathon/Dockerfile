FROM centos:7
MAINTAINER Kimice <kimiceqc@gmail.com>

RUN yum update -y
RUN yum install -y libxslt-devel libxml2-devel python-devel postgresql-devel gcc-c++ git python-virtualenv vim

#RUN mkdir /data
RUN mkdir -p /hackathon

COPY requirements.txt /hackathon
COPY run_v2.py /hackathon
COPY predict.py /hackathon
COPY prepare.py /hackathon
COPY hackathonCaseB.sh /

RUN chmod 777 *.sh

#COPY files /data

RUN chmod -R 777 /hackathon
#RUN chmod -R 777 /data

RUN virtualenv .venv
RUN .venv/bin/pip install pip==8.1.2 -i http://pypi.douban.com/simple
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r /hackathon/requirements.txt

EXPOSE 80

