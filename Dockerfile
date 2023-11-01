FROM almalinux:9


RUN dnf install -y cargo python3 python3-pip bluez glib2-devel


COPY requirements.txt .
RUN pip3 install -r requirements.txt