FROM ubuntu:18.04

WORKDIR /

# Prerequisites settings
RUN apt-get update && \
    apt-get install -y git

# Install Python
RUN apt-get install -y python3.6 && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Install Java 8 (JDK)
RUN apt-get install -y openjdk-8-jdk
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Install Ray with TensorFlow 2.x
RUN pip3 install gym lxml numpy pillow && \
    pip3 install tensorflow==2.4.1 ray[default]==1.6.0 ray[rllib]==1.6.0 ray[tune]==1.6.0 attrs==19.1.0 pandas

# Install Desktop Components for Headless
RUN apt-get install -y xvfb && \
    echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get install -y lxde

# Install Malmo
RUN pip3 install --index-url https://test.pypi.org/simple/ malmo==0.36.0
ENV MALMO_PATH=/malmo_package
WORKDIR $MALMO_PATH
RUN python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"
ENV MALMO_XSD_PATH=$MALMO_PATH/MalmoPlatform/Schemas

WORKDIR /

# Install custom Gym env
RUN git clone https://github.com/tsmatz/minecraft-rl-on-ray-cluster
RUN cd minecraft-rl-on-ray-cluster && \
    pip3 install Malmo_Maze_Sample/

EXPOSE 6379 8265
