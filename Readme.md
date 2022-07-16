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

In this example, I assume Ubuntu Server 18.04 in Microsoft Azure.<br>
This example will require much resources for running workloads and I then recommend that you should use powerful machines. (Here I used Standard D3 v2 (4 vcpus, 14 GB memory) VM on Microsoft Azure.)

> Note : This example is for getting started, but it's better to run on GPU in practical training. (Use ```tensorflow-gpu``` and change Dokerfile for running on GPU.)

Before running this example, please install and configure the required software as follows.

```
# Upgrade PIP
sudo apt-get update
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

# Install Java 8 for building and running Minecraft client
sudo apt-get install -y openjdk-8-jdk
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

# Install Ray with Tensorflow 2.x
# (Install tensorflow-gpu when you run on GPU)
pip3 install gym==0.21.0 lxml numpy pillow
pip3 install tensorflow==2.4.1 ray[default]==1.6.0 ray[rllib]==1.6.0 ray[tune]==1.6.0 attrs==19.1.0 pandas

# Install virtual monitor (xvfb) and desktop components
sudo apt-get install -y xvfb
sudo apt-get install -y lxde
### Following is only needed for using real monitor
# sudo apt-get install -y xrdp
# /etc/init.d/xrdp start

# Install and setup Malmo (Minecraft RL framework)
pip3 install --index-url https://test.pypi.org/simple/ malmo==0.36.0
python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download();"
echo -e "export MALMO_XSD_PATH=$HOME/MalmoPlatform/Schemas" >> ~/.bashrc
source ~/.bashrc
```

Please install custom Gym environment in this repository.

```
# Clone this repo and install custom Gym env
git clone https://github.com/tsmatz/minecraft-rl-on-ray-cluster
cd minecraft-rl-on-ray-cluster
pip3 install Malmo_Maze_Sample/
```

> Note : To uninstall custom Gym env (custom package), run as follows.<br>
> ```pip3 uninstall malmo-maze-env```

## Run Training on Single Machine (Test)

In this section, we will test script on a single machine.

All workers on cluster (multiple machines) will run the training without real monitors. Minecraft UI should then be redirected to virtual monitor, and ```xvfb``` (X Virtual Frame Buffer) can be used for launching headless Minecraft.<br>
With this installed custom Gym environment, the agent will run on Minecraft in this hidden virtual monitor (```xvfb```).

```
# Run training on single machine
python3 train_single.py
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

Next, please setup the required software on these multiple machines using above "Prerequisites" script. (The custom Gym env should also be installed.)

Since we use IMPALA (which is optimized for distributed RL) in RLlib built-in algorithms in cluster, please install the following additional package in all machines.

```
pip3 install ale-py==0.7
```

Now login to head node, and run the following command to start Ray head role (driver process) on head node.<br>
When you run this command, ray command for other workers (which will be needed in the following step) is shown in console and please copy this text.

```
ray start --head --port=6379
```

On each worker nodes (multiple machines), run the copied command (see above) to connect into your head node.<br>
Please change the following value of ```--address``` option to meet your copied text.

```
# Change values to meet your environment (See above)
ray start --address='10.6.0.5:6379' --redis-password='5241590000000000'
```

See Ray dashboard (http://127.0.0.1:8265) running on head node, and ensure that all worker machines have correctly connected to this cluster.

Login to head node and run ```train_cluster.py``` on head node.<br>
The following command will run 2 training workers. (Please change this value to meet the number of training workers. See the following note for settings.)

```
cd minecraft-rl-on-ray-cluster
python3 train_cluster.py --num_workers 2 --num_cpus_per_worker 3
```

> Note (Important) : Because of Minecraft port conflicts, mutiple worker process cannot run on a single machine. Set ```num_cpus_per_worker``` to run exact 1 worker on a single worker node. (In this example, I have used virtual machines with 4 cores.)

The output (statistics in each training iterations) will be shown in the console, and the results (checkpoint files with trained parameters) will be logged in ```./logs``` folder.

When you have finished training, please run the following command on each machines (both head node and workers) to stop Ray runtime process.

````
ray stop
````

See [here](https://github.com/tsmatz/azureml-examples/blob/master/azureml_minecraft_rl_ray_cluster.ipynb) for running this example (RL on Ray cluster) in Azure Machine Learning.

## Run Training on Ray Autoscaler for Azure

In this section, we will run Ray cluster with Azure provider of Ray autoscaler.

First, please prepare a client machine (also assuming Ubuntu 18.04) and setup the required components in this client as follows.

```
sudo apt-get update
sudo apt-get install -y  python3-pip
sudo -H pip3 install --upgrade pip

# Install Ray
pip3 install gym lxml numpy pillow
pip3 install ray[default]==1.6.0 attrs==19.1.0

# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Azure libraries for Python
pip3 install azure-cli-core==2.10.0
pip3 install azure==4.0.0
pip3 install knack
```

All the following tasks will be operated in this working client.<br>
First, generate a new ssh key pair, which will be used to authenticate Ray nodes.

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

As you can see in this configuration (```azure_ray_config.yaml```), custom docker image ```tsmatz/malmo-maze:0.36.0``` is used in both head and worker containers. All the required components (including custom Gym env) are already installed in this docker image. (Dockerfile for this image is ```docker/Dockerfile``` in this repository. I note that this image is not for running on GPU.)

```
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
