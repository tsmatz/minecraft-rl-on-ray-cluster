import gym
import malmo.MalmoPython as MalmoPython
import random
import time
import numpy as np
from enum import Enum
import pathlib
import os
import subprocess
import socket
import signal

TIME_WAIT = 0.05                    # time to wait for retreiving world state (when MsPerTick=20)
MAX_LOOP = 50                       # wait till TIME_WAIT * MAX_LOOP seconds for each action

class AgentActionSpace(gym.spaces.Discrete):
    def __init__(self):
        actions = []
        actions.append("move")
        actions.append("right")
        actions.append("left")
        self.actions = actions
        gym.spaces.Discrete.__init__(self, len(self.actions))

    def sample(self):
        return random.randint(1, len(self.actions)) - 1

    def __getitem__(self, action):
        return self.actions[action]

    def __len__(self):
        return len(self.actions)

class MalmoMazeEnv(gym.Env):
    """
    A class implementing OpenAI gym environment to
    run Project Malmo 0.36.0 Python API for solving
    maze.
    """
    def __init__(self):
        # Set up gym.Env
        super(MalmoMazeEnv, self).__init__()
        # Initialize variables
        mission_file = str(pathlib.Path(__file__).parent.parent.absolute()) + "/mission_files/lava_maze_malmo.xml"
        self.mission_xml = pathlib.Path(mission_file).read_text()
        self.height = 84
        self.width = 84
        self.shape = (self.height, self.width, 3)
        self.millisec_per_tick = 20
        self.maze_seed = "random"
        self.malmo_port = 9000
        # Action space definition : none-0, move-1, right-2, left-3
        self.action_space = AgentActionSpace()
        # Observation space definition
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.float32)
        # Launch headless Minecraft
        # (see _start_instance function)
        self._start_instance()
        # Create AgentHost
        self.agent_host = MalmoPython.AgentHost()
        # Create MissionRecordSpec
        self.my_mission_record = MalmoPython.MissionRecordSpec()
        self.my_mission_record.recordRewards()
        self.my_mission_record.recordObservations()
        # Create ClientPool
        self.pool = MalmoPython.ClientPool()
        client_info = MalmoPython.ClientInfo('127.0.0.1', self.malmo_port)
        self.pool.add(client_info)

    def __del__(self):
        self._kill_instance()

    """
    Public methods
    """

    def reset(self):
        # Create MissionSpec
        xml = self.mission_xml
        xml = xml.format(
            PLACEHOLDER_MSPERTICK=self.millisec_per_tick,
            PLACEHOLDER_WIDTH=self.width,
            PLACEHOLDER_HEIGHT=self.height,
            PLACEHOLDER_MAZESEED=self.maze_seed)
        my_mission = MalmoPython.MissionSpec(xml,True)
        # Start mission
        self.agent_host.startMission(my_mission,
            self.pool,
            self.my_mission_record,
            0,
            'test1')
        ### # For releasing instance when error occurs
        ### try:
        ###     self.agent_host.startMission(my_mission,
        ###         self.pool,
        ###         self.my_mission_record,
        ###         0,
        ###         'test1')
        ### except:
        ###     self._kill_instance()
        ###     raise
        # Wait till mission begins
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
        # Get reward, done, and frame
        frame, _, _ = self._process_state(False)
        if frame is None:
            self.last_obs = np.zeros(self.shape, dtype=np.float32)
        else:
            self.last_obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        return self.last_obs

    def step(self, action):
        # Take corresponding actions
        """ none:0, move:1, right:2, left:3 """
        if self.action_space[action] == "move":
            self.agent_host.sendCommand("move 1")
        elif self.action_space[action] == "right":
            self.agent_host.sendCommand("turn 1")
        elif self.action_space[action] == "left":
            self.agent_host.sendCommand("turn -1")

        # Get reward, done, and frame
        frame, reward, done = self._process_state()
        if reward is None:
            reward = 0
        # Clean up
        if done:
            frame2, reward2 = self._comsume_state()
            if frame2 is not None:
                frame = frame2
            reward = reward + reward2
        # Return observations
        if frame is None:
            self.last_obs = np.zeros(self.shape, dtype=np.uint8)
        else:
            self.last_obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
        return self.last_obs, reward, done, {}

    """
    Internal methods
    """

    # Extract frames, rewards, done_flag
    def _process_state(self, get_reward=True):
        reward_flag = False
        reward = 0
        frame_flag = False
        frame = None
        done = False
        loop = 0
        while True:
            # get world state
            time.sleep(TIME_WAIT * self.millisec_per_tick / 20)
            world_state = self.agent_host.getWorldState()
            # reward (loop till command's rewards are all retrieved)
            if (not reward_flag) and (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = True;
                reward = reward + world_state.rewards[-1].getValue()
            # frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
                frame_flag = True
            # done flag
            done = not world_state.is_mission_running
            # (Do not quit before comsuming)
            # if done:
            #     break;
            # exit loop when extraction is completed
            if get_reward and reward_flag and frame_flag:
                break;
            elif (not get_reward) and frame_flag:
                break;
            # exit when MAX_LOOP exceeds
            loop = loop + 1
            if loop > MAX_LOOP:
                reward = None
                break;
        return frame, reward, done

    def _comsume_state(self):
        reward_flag = True
        reward = 0
        frame = None
        loop = 0
        while True:
            # get next world state
            time.sleep(TIME_WAIT * self.millisec_per_tick / 5)
            world_state = self.agent_host.getWorldState()
            # reward (loop till command's rewards are all retrieved)
            if reward_flag and not (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = False;
            if reward_flag:
                reward = reward + world_state.rewards[-1].getValue()
            # frame
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
            if not reward_flag:
                break;
        return frame, reward

    def _start_instance(self):
        #
        # Launch headless Minecraft
        #
        # The following command is written in {package folder}/shell_files/launchClient_headless.sh
        # xvfb-run -a -e /dev/stdout -s '-screen 0 640x480x16' ./launchClient.sh -port $1
        #

        # Check whether the port is already in use
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect(("127.0.0.1", self.malmo_port))
                s.close()
            print("Malmo port {} is already in use. Try to connect existing Minecraft instance.".format(self.malmo_port))
            return
        except (ConnectionError, socket.timeout):
            print("Start Minecraft instance")

        # Launch Minecraft
        launch_shell_file = str(pathlib.Path(__file__).parent.parent.absolute()) + "/shell_files/launchClient_headless.sh"
        dev_null = open(os.devnull, "w")
        self.proc = subprocess.Popen(
            ["bash", launch_shell_file, str(self.malmo_port)],
            stdout=dev_null,
            preexec_fn=os.setsid)
        ### # For Debug : Verbose Output
        ### self.proc = subprocess.Popen(
        ###     ["xvfb-run", "-a", "-e", "/dev/stdout", "-s", "-screen 0 640x480x16", "./launchClient.sh", "-port", str(self.malmo_port)])
        ### # For Debug : Screen Output (Need Monitor)
        ### self.proc = subprocess.Popen(
        ###     ["./launchClient.sh", "-port", str(self.malmo_port)])

        # Wait till instance runs
        print("Waiting Minecraft instance to start ...")
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10)
                    s.connect(("127.0.0.1", self.malmo_port))
                    s.close()
                print("Finished waiting for instance")
                break
            except (ConnectionError, socket.timeout):
                time.sleep(5)

    def _kill_instance(self):
        if self.proc is not None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc = None
            print("Terminated Minecraft instance")
