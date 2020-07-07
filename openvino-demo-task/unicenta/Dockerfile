FROM ubuntu:18.04
LABEL maintainer "Bryan J Rodriguez <bryan.j.rodrigez@intel.com>"

# ARG JAVA_URL
ARG UNICENTA_VER=4.5

RUN apt update && apt-get install -y \
    bash \
    wget \
	unzip \
    openjdk-8-jdk \
    dos2unix \
    # gcc \
    # make \
    # bzip2
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# RUN mkdir -p /opt && \
#     wget ${JAVA_URL} && \
#     tar -zxf $(basename ${JAVA_URL}) && \
#     rm $(basename ${JAVA_URL})

# RUN update-alternatives --install /usr/bin/java java /opt/$(ls /opt/ | grep jre | head -1)/bin/java 110 && \
#     update-java-alternatives -s /opt/$(ls /opt/ | grep jre | head -1)/java

RUN update-java-alternatives -s /usr/lib/jvm/java-1.8.0-openjdk-amd64

RUN mkdir -p /opt && \
    cd /opt && \
    wget -O unicenta-installer.run https://sourceforge.net/projects/unicentaopos/files/releases/linux/unicentaopos-${UNICENTA_VER}-linux-x64-installer.run/download && \
    chmod a+x unicenta-installer.run && \
    ./unicenta-installer.run --unattendedmodeui minimal --mode unattended --installer-language en >/dev/null 2>&1 && \
    rm ./unicenta-installer.run && \
    mv /opt/unicentaopos-* /opt/unicentaopos && \
    chmod a+x /opt/unicentaopos/start.sh && \
    dos2unix /opt/unicentaopos/start.sh


RUN addgroup --gid 1100 pos && \
    adduser --uid 1100 --gid 1100 --home /home/pos --shell /bin/bash --disabled-password --gecos GECOS pos

COPY bin/init /usr/local/bin/init
RUN dos2unix /usr/local/bin/init
COPY conf/unicentaopos.properties /tmp/unicentaopos.properties
RUN dos2unix /tmp/unicentaopos.properties
RUN apt update && apt-get install x11-xserver-utils -y
ENV DISPLAY=:1
#RUN xhost +
WORKDIR /home/pos
USER pos

ENTRYPOINT ["/usr/local/bin/init"]