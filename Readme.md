# Minecraft Reinforcement Learning on Ray Cluster (Distributed RL)

This tutorial shows you how to configure and run distributed reinforcement learning with Minecraft RL framework, Project Malmo.<br>
This example is the cluster version of [Malmo maze sample](https://github.com/tsmatz/minecraft-rl-example), in which the agent will learn to solve the maze in Minecraft using frame pixels.

<ins>Table of Contents</ins>

- Prerequisites
- Run Training on Single Machine (Test)
- Run Training on Manually Configured Cluster (Multiple Machines)
- Run Training on Ray Autoscaler for Azure

> Note : See [my post](https://tsmatz.wordpress.com/2021/10/08/rllib-reinforcement-learning-multiple-machines-ray-cluster/) for the background architecture.

## Prerequisites

In this example, I assume Ubuntu Server 20.04 LTS in Microsoft Azure.<br>
This example will require much resources for running workloads and I then recommend that you should use powerful machines. (Here I used Standard D3 v2 (4 vcpus, 14 GB memory) VM on Microsoft Azure.)

> Note : This example is for tutorials, but it's better to run on GPU in practical training. (Use ```tensorflow-gpu``` and change Dokerfile for running on GPU.)

Before running this example, please install and configure the required software as follows. (See [here](https://github.com/tsmatz/minecraft-rl-example) for details.)<br>
Malmo source code is compiled and the file ```~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so``` is then generated.

```
# install python 3.6
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6

# configure to make python3 command to use python3.6
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --config python3

# install required components
sudo apt-get install \
  build-essential \
  libpython3.6-dev \
  openjdk-8-jdk \
  swig \
  doxygen \
  xsltproc \
  ffmpeg \
  python-tk \
  python-imaging-tk \
  zlib1g-dev

# set environment for Java
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

# update certificates
sudo update-ca-certificates -f

# download and build cmake
mkdir ~/cmake
cd ~/cmake
wget https://cmake.org/files/v3.11/cmake-3.11.0.tar.gz
tar xvf cmake-3.11.0.tar.gz
cd cmake-3.11.0
./bootstrap
make -j4
sudo make install
cd

# download and build boost
mkdir ~/boost
cd ~/boost
wget http://sourceforge.net/projects/boost/files/boost/1.66.0/boost_1_66_0.tar.gz
tar xvf boost_1_66_0.tar.gz
cd boost_1_66_0
./bootstrap.sh --with-python=/usr/bin/python3.6 --prefix=.
./b2 link=static cxxflags=-fPIC install
cd

# download and install Malmo
git clone https://github.com/Microsoft/malmo.git ~/MalmoPlatform
wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P ~/MalmoPlatform/Schemas
echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
source ~/.bashrc
cd ~/MalmoPlatform
mkdir build
cd build
cmake -DBoost_INCLUDE_DIR=/home/$USER/boost/boost_1_66_0/include -DBOOST_PYTHON_NAME=python3 -DCMAKE_BUILD_TYPE=Release ..
make install
cd

# after installation, configure to make python3 command to use python3.8 (default in Ubuntu 20.04)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3

# set up pip
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
sudo apt-get install python3.6-distutils

# install required packages
python3.6 -m pip install \
  gym==0.21.0 \
  lxml \
  numpy \
  pillow \
  tensorflow==2.4.1 \
  gpustat==0.6.0 \
  aiohttp==3.7.4 \
  prometheus-client==0.8.0 \
  redis==3.5.3 \
  ray[default]==1.6.0 \
  dm-tree==0.1.7 \
  attrs==19.1.0 \
  pandas

python3.6 -m pip install \
  ray[rllib]==1.6.0 \
  ray[tune]==1.6.0
```

In this example, we don't use a real monitor, but use a virtual monitor for showing Minecraft UI.<br>
To do this, install virtual monitor (```xvfb```) and desktop components.

```
sudo apt-get install -y xvfb
sudo apt-get install -y lxde
```

Clone this repository.

```
git clone https://github.com/tsmatz/minecraft-rl-on-ray-cluster
cd minecraft-rl-on-ray-cluster
```

Copy the file ```~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so``` in current folder (```minecraft-rl-on-ray-cluster```).

```
cp ~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so .
```

Install custom Gym environment.

```
python3.6 -m pip install Malmo_Maze_Sample/
```

> Note : To uninstall custom Gym env (custom package), run as follows.<br>
> ```python3.6 -m pip uninstall malmo-maze-env```

## Run Training on Single Machine (Test)

In this section, we will test script on a single machine.

All workers on cluster (multiple machines) will run the training without real monitors. Minecraft UI should then be redirected to virtual monitor, and ```xvfb``` (X Virtual Frame Buffer) can be used for launching headless Minecraft.<br>
With this installed custom Gym environment, the agent will run on Minecraft in this hidden virtual monitor (```xvfb```).

```
python3.6 train_single.py
```

> Note : When you run on GPU, please specify ```--num_gpus``` option.<br>
> ```python3 train_single.py --num_gpus 1```

When this custom Gym environment is initialized in this script (```train_single.py```), headless Minecraft instance will automatically start. (See ```Malmo_Maze_Sample/custom_malmo_env/env/maze_env.py```.)

> Note : For the first time to run, it will take a long time, because Malmo will build (compile) the modded Minecraft.

The output (statistics in each training iterations) will be shown in the console, and the results (checkpoint files with trained parameters) will be logged in ```./logs``` folder.

## Run Training on Manually Configured Cluster (Multiple Machines)

In this section, we will configure Ray cluster (multiple machines) manually and run script on this cluster.

First, please prepare multiple machines - 1 head node and multiple worker nodes. All machines should be connected (by port 6379) each other on the same network.<br>
In my case, I have created multiple virtual machines on the same resource group in Microsoft Azure. (Then all machines can be connected each other on the same virtual network with internal addresses.)

Next, please setup the required software on these multiple machines using above "Prerequisites" script. (The custom Gym env should also be installed in all machines.)

Since we use IMPALA (which is optimized for distributed RL) in RLlib built-in algorithms in cluster, please install the following additional package in all machines.

```
python3.6 -m pip install ale-py==0.7
```

Now login to head node, and run the following command to start Ray head role (driver process) on head node.<br>
When you run this command, ray command to connect to this runtime (which will be needed in the following steps) is shown in console and please copy this command's text.

```
ray start --head --port=6379
```

On each worker nodes (multiple machines), run the copied command (see above) to connect to your head node.<br>
Please change the following value of ```--address``` option to meet your environment.

```
# Change values to meet your environment (See above)
ray start --address='10.6.0.5:6379' --redis-password='5241590000000000'
```

See Ray dashboard (http://127.0.0.1:8265) running on head node, and ensure that all worker machines have correctly connected to this cluster.

Login to head node and run ```train_cluster.py``` on head node.<br>
The following command will run 2 training workers. (Please change this value to meet the number of training workers. See the following note for settings.)

```
cd minecraft-rl-on-ray-cluster
python3.6 train_cluster.py --num_workers 2 --num_cpus_per_worker 3
```

> Note (Important) : Because of Minecraft port's confliction, mutiple worker process on a single machine cannot run in this training. Set ```num_cpus_per_worker``` to run exact 1 worker on a single worker node. (In this example, I have used virtual machines with 4 cores.)<br>
> When you run the training again, please restart your VMs to clean up Minecraft clients.

The output (statistics in each training iterations) will be shown in the console, and the results (checkpoint files with trained parameters) will be logged in ```./logs``` folder.

When you have finished training, please run the following command on each machines (both head node and workers) to stop Ray runtime process.

````
ray stop
````

See [here](https://github.com/tsmatz/azureml-examples/blob/master/azureml_minecraft_rl_ray_cluster/azureml_minecraft_rl_ray_cluster.ipynb) for running this example (RL on Ray cluster) in Azure Machine Learning.

## Run Training on Ray Autoscaler for Azure

In this section, we will run Ray cluster with Azure provider of Ray autoscaler.

In this example, we use pre-configured docker image for Minecraft training (which runs on Ubuntu 18.04), and operate Ray cluster on Ubuntu 20.04 client.

First, please prepare a client machine (assuming Ubuntu 20.04) and setup the prerequisite's components in this client as follows.

```
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip

#
# Install Ray
#
pip3 install gym==0.21.0 lxml numpy pillow
pip3 install ray[default]==1.6.0 attrs==19.1.0

#
# Install specific version of Azure CLI
# (see https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux)
# 

# get packages needed for the installation process
sudo apt-get install ca-certificates curl apt-transport-https lsb-release gnupg
# download and install the Microsoft signing key
sudo mkdir -p /etc/apt/keyrings
curl -sLS https://packages.microsoft.com/keys/microsoft.asc |
    gpg --dearmor |
    sudo tee /etc/apt/keyrings/microsoft.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/microsoft.gpg
# add the Azure CLI software repository
AZ_DIST=$(lsb_release -cs)
echo "deb [arch=`dpkg --print-architecture` signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ $AZ_DIST main" |
    sudo tee /etc/apt/sources.list.d/azure-cli.list
# install Azure CLI
sudo apt-get update
sudo apt-get install azure-cli=2.27.2-1~focal

#
# Install Azure libraries for Python
#
pip3 install protobuf==3.20.3
pip3 install azure-cli-core==2.10.0
pip3 install azure==4.0.0
pip3 install azure-mgmt-resource==10.1.0
pip3 install knack
```

> Note : See [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux) for details about Azure CLI installation. In this example, we need a specific version of Azure CLI.

All tasks will be operated in this client by connecting to Ray nodes in Microsoft Azure.<br>
First, generate a new ssh key pair. This certificate is used to connect to Ray nodes. (Please remember the passphrase.)

```
ssh-keygen -t rsa -b 4096
```

Next, login to Azure subscription with Azure CLI.

```
az login
az account set -s {your_subscription_id}
```

Now create Ray cluster with YAML configuration (```azure_ray_config.yaml```) in this repository. (See below.)<br>
All Azure resources in this cluster are generated in resource group ```ray-cluster-test01```.

As you can see in this configuration (```azure_ray_config.yaml```), custom docker image ```tsmatz/malmo-maze:0.36.0``` is used in both head and worker containers. All the required components (including custom Gym env) are already installed in this docker image. (Dockerfile for this image is ```docker/Dockerfile``` in this repository. I note that this image is not for running on GPUs, but CPUs.)

```
cd ~/minecraft-rl-on-ray-cluster
ray up ./azure_ray_config.yaml
```

> Note : Ray config and its result will be cached in the client. When you want to repeatedly run this command ignoring this cache, please run with ```--no-config-cache``` option as follows.<br>
> ```ray up ./azure_ray_config.yaml --no-config-cache```

By running the following Ray CLI command, you can submit distributed training on this cluster.

```
ray submit ./azure_ray_config.yaml train_cluster.py --num_cpus_per_worker 3
```

You can also connect to the driver (container) on head node and run commands manually as follows.

```
ray attach ./azure_ray_config.yaml

$ python3 -c 'import ray; ray.init(address="auto")'
$ exit
```

When you have finished training, you can tear down the cluster as follows. (All Ray nodes will be deleted.)

```
ray down ./azure_ray_config.yaml
```
