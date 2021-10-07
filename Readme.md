# Minecraft Reinforcement Learning on Ray Cluster (Distributed RL)

This tutorial shows you how to configure and run distributed reinforcement learning with Minecraft RL framework, Project Malmo.

- Prerequisites
- Run Training on Single Machine (Test)
- Run Training on Manually Configured Cluster (Multiple Machines)

## Prerequisites

This example is the cluster version of [this repo](https://github.com/tsmatz/malmo-maze-sample), in which the agent will learn to solve the maze using frame pixels.

In this example, I assume Ubuntu 18.04. This example requires much resources for running workloads and you should then use powerful machines. (Here I used Standard D3 v2 (4 vcpus, 14 GB memory) VM on Microsoft Azure.)

Before running this example, please install and configure the required software as follows.

```
sudo apt-get update
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

# Install Java 8 for running Minecraft client
sudo apt-get install -y openjdk-8-jdk
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

# Install Ray with Tensorflow 2.x
pip3 install gym lxml numpy pillow
pip3 install tensorflow==2.4.1 ray[default]==1.6.0 ray[rllib]==1.6.0 ray[tune]==1.6.0 attrs==19.1.0 pandas

# Install for virtual monitor
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

# Clone this repo and install custom Gym env
git clone https://github.com/tsmatz/minecraft-rl-on-ray-cluster
cd minecraft-rl-on-ray-cluster
pip3 install Malmo_Maze_Sample/
```

## Run Training on Single Machine (Test)

In this section, we will test script on a single machine.

All workers on cluster (multiple machines) will run the training without real monitors. Minecraft UI should then be redirected to virtual monitor, and I have used ```xvfb``` (X Virtual Frame Buffer) in this example for headless Minecraft.

When you initialize custom Gym environment, this headless Minecraft will automatically start. (See ```Malmo_Maze_Sample/custom_malmo_env/env/maze_env.py```.)

```
# Run training on single machine
python3 train_single.py
```

> Note : When you run on GPU, please specify ```--num_gpus``` option.<br>
> ```python3 train_single.py --num_gpus 1```

> Note : For the first time to run, it will take a long time, because Malmo will build (compile) modded Minecraft.

## Run Training on Manually Configured Cluster (Multiple Machines)

In this section, we will configure Ray cluster (multiple machines) manually, and run script on this cluster.

First, please prepare multiple machines - 1 head node and multiple worker nodes. All machines should be connected (by port 6379) each other on network.<br>
In my case, I have provisioned multiple virtual machines on the same resource group in Microsoft Azure. (Then all machines can be connected each other with internal addresses.)

Next, please setup (configure) multiple machines using above "Prerequisites" script.

Since we use IMPALA (which is optimized for distributed RL) for algorithms on training in cluster, please install the following additional package in all machines.

```
pip3 install ale-py==0.7
```

Now let's start Ray runtime (master) on head node.<br>
When you run this command, ray commands on workers will be shown in console and please copy this text.

```
ray start --head --port=6379
```

On each worker nodes (multiple machines), run the copied command and connect to the head node.<br>
Please change the following values of ```--address``` and ```--redis-password``` to meet your copied text.

```
ray start --address='10.6.0.5:6379' --redis-password='5241590000000000'
```

See Ray dashboard (http://127.0.0.1:8265) on head node, and ensure that all worker machines are correctly connected.

Login to head node and run ```train_cluster.py``` on head node.<br>
The following command will run 2 workers.

```
cd minecraft-rl-on-ray-cluster
python3 train_cluster.py --num_workers 2 --num_cpus_per_worker 3
```

> Note : This script cannot run mutiple worker process on a single machine (because of port conflicts). Set ```num_cpus_per_worker``` to run exact 1 worker on a single node. (In this example, I have used virtual machines with each 4 cores.)

The output (statistics in each training iterations) will be shown in the console, and the results will be logged in ```./logs``` folder.

When you have finished training, please run the following command on each machines (head node and workers) to stop Ray runtime.

````
ray stop
````
