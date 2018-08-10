import numpy as np
import math
import copy
import time
import random
import pickle
import mujoco_py
import torch
from mujoco_py import functions
from torch.autograd import Variable
import time
import torch.nn.functional as F

########## problem2: when contact movement can't hold 
class Env():

    def __init__(self, render, current, limit, depth):
        self.model = mujoco_py.load_model_from_path("/home/newuhe/YCBRender/xmls/YCB.xml")
        #self.model = mujoco_py.load_model_from_path("/home/newuhe/YCBRender/xmls/sensor.xml")
        print("Model Load Successfully!")
        self.sim = mujoco_py.MjSim(self.model)
        self.render = render
        if self.render:
            print("render:",self.render)
            self.viewer = mujoco_py.MjViewer(self.sim)

        #print(self.model.body_pos)
        # print(self.sim.data.get_joint_qpos("joint1").shape)
        # print(type(self.sim.data.get_joint_qpos("joint1")))
  
        self.sim.data.set_joint_qpos("joint1_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint1_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint1_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint1_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint1_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint2_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint2_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint2_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint2_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint2_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint3_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint3_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint3_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint3_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint3_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint4_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint4_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint4_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint4_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint4_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint5_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint5_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint5_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint5_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint5_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint6_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint6_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint6_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint6_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint6_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint7_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint7_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint7_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint7_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint7_6",random.uniform(-math.pi, math.pi))

        self.sim.forward()

    def reset(self):
        self.sim.reset()
        self.sim.data.set_joint_qpos("joint1_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint1_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint1_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint1_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint1_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint2_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint2_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint2_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint2_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint2_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint3_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint3_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint3_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint3_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint3_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint4_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint4_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint4_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint4_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint4_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint5_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint5_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint5_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint5_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint5_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint6_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint6_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint6_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint6_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint6_6",random.uniform(-math.pi, math.pi))

        self.sim.data.set_joint_qpos("joint7_1",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint7_2",random.uniform(-0.3, 0.3))
        self.sim.data.set_joint_qpos("joint7_4",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint7_5",random.uniform(-math.pi, math.pi))
        self.sim.data.set_joint_qpos("joint7_6",random.uniform(-math.pi, math.pi))
        
        self.sim.forward()

    def Sample(self, r, alpha, beta):
        # camera in sphere coordinate (r, alpha, beta)
        x = r*math.sin(beta)*math.cos(alpha)
        y = r*math.sin(beta)*math.sin(alpha)
        z = r*math.cos(beta)

        self.sim.data.set_joint_qpos("camera_joint1", x)
        self.sim.data.set_joint_qpos("camera_joint2", y)
        self.sim.data.set_joint_qpos("camera_joint3", z)

        self.sim.data.set_joint_qpos("camera_joint5", math.atan(np.sqrt(x*x+y*y)/z))
        if y>=0:
            self.sim.data.set_joint_qpos("camera_joint6", math.acos(x/np.sqrt(x*x+y*y)))
        else:
            self.sim.data.set_joint_qpos("camera_joint6", 2*math.pi-math.acos(x/np.sqrt(x*x+y*y)))
        
        # self.sim.data.set_mocap_pos("cameramover",[x-0.6,y+0.6,z])
        # cos1 = np.sqrt(( z/np.sqrt(x*x+y*y+z*z)+1 )/2)
        # sin1 = np.sqrt(x*x+y*y)/np.sqrt(x*x+y*y+z*z)/2/cos1
        # ux1 = -y/np.sqrt(x*x+y*y)
        # uy1 = x/np.sqrt(x*x+y*y)
        # uz1 = 0
        # self.sim.data.set_mocap_quat("cameramover",[cos2, sin2*ux2, sin2*uy2, sin2*uz2])
        # cos2 = np.sqrt( ( x/np.sqrt(x*x+y*y) +1)/2 )
        # sin2 = y/np.sqrt(x*x+y*y)/2/cos2
        # ux2 = 0
        # uy2 = 0
        # uz2 = 1
        # self.sim.data.set_mocap_quat("cameramover",[cos, sin*ux, sin*uy, sin*uz])
        # self.sim.data.set_mocap_quat("cameramover",[cos1*cos2, ux2*cos1*sin2/np.sqrt(sin1*sin1*cos2*cos2+cos1*cos1*sin2*sin2), uy2*cos1*sin2/np.sqrt(sin1*sin1*cos2*cos2+cos1*cos1*sin2*sin2), sin1*cos2/np.sqrt(sin1*sin1*cos2*cos2+cos1*cos1*sin2*sin2)])

        self.sim.forward()     
        self.getPointCloud()  
        self.viewer.render()

    def getPointCloud(self):
        scalingFactor = 1000
        fovy = 45 
        idd = 0.068
        render_buffer = self.sim.render(width=64, height=32, camera_name="mycamera", depth=True)
        color , depth = render_buffer
        depth = np.array(depth.tolist())
        print(depth.shape)
        width = depth.shape[1]
        height = depth.shape[0]
        
        # for v in range(height):
        #     for u in range(width):
        #         Z = depth[v][u]/ scalingFactor
        #         print(Z)
        #         if Z==0: continue
                # X = (u - centerX) * Z / focalLength
                # Y = (v - centerY) * Z / focalLength
                # points.append("%f %f %f %d %d %d 0\n")

if __name__ == '__main__':
    e = Env(True, 0.083, 0.15, 0.08)

    # while(1):
    #     e.reset()
    #     if abs(e.sim.data.sensordata[0])>2 or abs(e.sim.data.sensordata[1])>2 or abs(e.sim.data.sensordata[2])>2:
    #         break

    # for i in range(50):
    #     e.sim.data.ctrl[2] = -0.3
    #     e.sim.step()
    # e.sim.data.ctrl[2]=0
    
    e.reset()
    for i in range(500):
        e.sim.step()
    while(1): 
        alpha = random.uniform(0,math.pi*2)
        beta = random.uniform(0, math.pi/2)
        e.Sample(1, alpha, beta)

    # while(1):
    #     e.reset()
    #     for i in range(100):
    #         e.randomSample()

