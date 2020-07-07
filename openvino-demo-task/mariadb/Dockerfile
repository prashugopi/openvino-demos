FROM alpine:3.10
LABEL maintainer "Bryan J Rodriguez <bryan.j.rodrigez@intel.com>"

RUN apk update && apk add --no-cache \
    dumb-init \
    mariadb \
    mariadb-client \
    mariadb-mytop \
    mariadb-server-utils \
    rsync \
    libaio \
    pwgen

RUN mkdir -p /usr/local/bin && \
    mkdir -p /usr/local/lib && \
    mkdir -p /run/mysqld

COPY bin/init /usr/local/bin/init
RUN dos2unix /usr/local/bin/init
COPY conf/TEMPLATE.my.cnf /root/TEMPLATE.my.cnf
RUN dos2unix /root/TEMPLATE.my.cnf
# USER mysql

ENTRYPOINT [ "dumb-init", "/usr/local/bin/init" ]
CMD []