import os
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import cv2
import numpy as np
import PIL.Image as Image
from grasp_detector import Grasper
from scipy.spatial.transform import Rotation as R

robotUrdfPath = "franka_panda/simple_gripper.urdf"

cameraPos=[0,0,0.5]
viewMatrix=None
rot=None

class Gripper():
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(True)
        p.setGravity(0, 0, -10) # NOTE
        self.planeID = p.loadURDF("plane.urdf")
        #######################################
        ###    define and setup robot       ###
        #######################################
        self.robot=self.load_robot()
        p.changeDynamics(self.robot, -1, lateralFriction=10)
        self.controlJoints = ["panda_finger_joint1",
                         "panda_finger_joint2"]
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.robot)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])

        joints = AttrDict()
        self.dummy_center_indicator_link_index = 0

        # get jointInfo and index of dummy_center_indicator_link
        for i in range(numJoints):
            info = p.getJointInfo(self.robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity)
            joints[singleInfo.name] = singleInfo
            # register index of dummy center link
            if jointName == "gripper_roll":
                self.dummy_center_indicator_link_index = i
        self.joints=joints
        self.position_control_joint_name = ["center_x",
                                       "center_y",
                                       "center_z",
                                       "gripper_yaw",
                                       "gripper_pitch",
                                        "gripper_roll",]
        self.position_control_group=self.addUserDebuger()
    def load_robot(self):
        robotStartPos = [0, 0, 1.2]
        robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
        robotID = p.loadURDF(robotUrdfPath, robotStartPos, robotStartOrn,
                             flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        # trayUid = p.loadURDF("tray/traybox.urdf", basePosition=[0, 0, 0])  # 加载一个箱子，设置初始位置为（0，0，0）
        return robotID
    def load_obj(self):
        # load Object
        objid=np.random.randint(0,999)
        objid=str(objid).zfill(3)
        obj_path=os.path.join(pybullet_data.getDataPath(), "random_urdfs/"+objid+'/'+objid+'.urdf')
        ObjectID = p.loadURDF(obj_path, [0, 0, 0.20], globalScaling=0.7)
        # ObjectID = p.loadURDF("./urdf/object_demo.urdf", [0, 0, 0.10], globalScaling=0.0030)
        p.changeDynamics(ObjectID,-1,lateralFriction=10)


    def pos(self,position=None):
        if position==None:
            parameter = []
            position_control_group=self.position_control_group
            for i in range(6):
                parameter.append(p.readUserDebugParameter(position_control_group[i]))

            parameter_orientation = p.getQuaternionFromEuler([parameter[3], parameter[4], parameter[5]])
            print((parameter_orientation))
        else:
            parameter=np.array(position[0])
            parameter_orientation=position[1]

        jointPose = p.calculateInverseKinematics(self.robot,
                                                 self.dummy_center_indicator_link_index,
                                                 [parameter[0], parameter[1], parameter[2]],
                                                 parameter_orientation)

        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]

                p.setJointMotorControl2(self.robot, joint.id, p.POSITION_CONTROL,
                                        targetPosition=jointPose[joint.id], force=1000,
                                        maxVelocity=1)
                    # p.stepSimulation()

    def grasp(self,position=None):

        position_control_group=self.position_control_group
        # gripper control
        if position == None:
            gripper_opening_length = p.readUserDebugParameter(position_control_group[-1])
        else:
            gripper_opening_length=position
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        for i in range(2):
            p.setJointMotorControl2(self.robot,
                                    self.joints[self.controlJoints[i]].id,
                                    p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle,
                                    force=self.joints[self.controlJoints[i]].maxForce,
                                    maxVelocity=self.joints[self.controlJoints[i]].maxVelocity)


    def addUserDebuger(self):
        # id of gripper control user debug parameter
        # angle calculation
        # openning_length = 0.010 + 0.1143 * math.sin(0.7180367310119331 - theta)
        # theta = 0.715 - math.asin((openning_length - 0.010) / 0.1143)
        # gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",
        #                                                 0,
        #                                                 0.085,
        #                                                 0.085)
        # position control
        position_control_group = []
        position_control_group.append(p.addUserDebugParameter('x', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('y', -0.5, 0.5, 0))
        position_control_group.append(p.addUserDebugParameter('z', -0.25, 1.5, 1.2))
        position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, 0.1))
        position_control_group.append(p.addUserDebugParameter('pitch', 0, 3.14, 1.7))
        position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 3.14, 0.1))
        position_control_group.append(p.addUserDebugParameter("gripper_opening_length",
                                                        0.077,
                                                        0.085,
                                                        0.080))

        return position_control_group

def mask_pic(seg,num):
    sp=np.array(seg).shape
    mask=np.array([False for i in range(sp[0]*sp[1])]).reshape(sp)#np.full(sp, False, dtype=bool)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if seg[i][j]==num:
                mask[i][j]=True
                try:
                    mask[i-1][j-1]=True
                    mask[i-1][j+1]=True
                    mask[i][j-1]=True
                    mask[i][j+1]=True
                except:
                    pass


    return mask

def take_pic():
    global viewMatrix,rot
    width = 1280  # 图像宽度
    height = 720   # 图像高度

    fov = 50  # 相机视角
    aspect = width / height  # 宽高比
    near = 0.01  # 最近拍摄距离
    far = 20  # 最远拍摄距离
    qq = np.array([0,0,0,1])
    rot = R.from_quat(qq).as_matrix()

    targetPos = np.matmul(rot, np.array([0, 0, -1])) + cameraPos

    cameraupPos = np.matmul(rot, np.array([-1, 0, 0]))
    # cameraPos = [0,0,0.3]  # 相机位置
    # targetPos = [0,0,0]  # 目标位置，与相机位置之间的向量构成相机朝向
    # cameraupPos = [0,1,0]  # 相机顶端朝向

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos,
        physicsClientId=0
    )  # 计算视角矩阵
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵
    print(projection_matrix)
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    segmask=mask_pic(seg,3)

    i = 1
    # 开始渲染

    images = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGRA2BGR)
    depImg = far * near / (far - (far - near) * images[3])
    depImg = np.asanyarray(depImg).astype(np.float32) * 1000.
    depImg = (depImg.astype(np.uint16))
    print(type(depImg[0, 0]))
    t=depImg
    depImg = Image.fromarray(depImg)

    # cv2.imwrite('image/seg' + str(i) + '.jpg', images[4])
    # cv2.imshow('1',rgbImg)
    # cv2.waitKey(0)
    viewMatrix = np.array(viewMatrix)
    viewMatrix = np.array(viewMatrix).reshape(4, 4).T
    return rgbImg,t,segmask

def tstgripper():
    robotiq = Gripper()
    robotiq.load_obj()

    while p.isConnected():
        # p.stepSimulation()
        robotiq.pos()
        robotiq.grasp()
        keys = p.getKeyboardEvents()

        if ord("c") in keys and keys[ord("c")] & p.KEY_WAS_RELEASED:
            robotiq.load_obj()
def drawline(frame_start_postition,R_Mat):
    p.removeAllUserDebugItems()
    # x 轴


    x_axis = R_Mat[:, 0]
    x_end_p = (np.array(frame_start_postition) + np.array(x_axis * 3)).tolist()
    x_line_id = p.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0])

    # y 轴
    y_axis = R_Mat[:, 1]
    y_end_p = (np.array(frame_start_postition) + np.array(y_axis * 3)).tolist()
    y_line_id = p.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0])

    # z轴
    z_axis = R_Mat[:, 2]
    z_end_p = (np.array(frame_start_postition) + np.array(z_axis * 3)).tolist()
    z_line_id = p.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1])

if __name__ == '__main__':
    # tstgripper()

    robotiq=Gripper()
    robotiq.load_obj()

    grasper=Grasper()
    robotiq.pos()
    robotiq.grasp()

    while p.isConnected():

        keys = p.getKeyboardEvents()

        if ord("z") in keys and keys[ord("z")] & p.KEY_WAS_RELEASED:
            rgbimg,depthimg,segmask=take_pic()

            gg, cloud = grasper.demo(inputpic=True,color=rgbimg,depth=depthimg,seg=None,show=True)
            gg.nms()
            gg.sort_by_score()
            gg = gg[:50]
            try:
                # parameter = []
                # position_control_group = robotiq.position_control_group
                # for i in range(6):
                #     parameter.append(p.readUserDebugParameter(position_control_group[i]))
                # euler=parameter[:3]
                # r = R.from_euler('xyz', euler, degrees=True)
                # rotation_matrix = r.as_matrix()
                R1 = rot
                gg1 = gg[0]
                ori = np.array(gg1.rotation_matrix)
                tran=np.array(gg1.translation)
                R1=np.matmul( R1,np.array([[0, 1, 0],  [1, 0, 0],[0, 0, -1]]))
                ori=np.matmul(R1,ori)
                # orien = np.matmul(ori,)
                camera_tran=np.dot(R1, tran.T).T
                # camera_tran[2]=-camera_tran[2]
                world_tran =camera_tran + cameraPos
                # ori = np.array(gg1.rotation_matrix)
                grasp_vector=np.matmul(ori, np.array([0.1, -0., -0.]).T).T
                tsts=np.matmul(grasp_vector,np.array([0,0,-1]))
                ready_pos = world_tran + np.matmul(ori, np.array([-0.1, -0., -0.]).T).T
                print('oooo',ori)
                # RR=np.matmul(orien,rotation_matrix)
                # orien=R.from_matrix(np.matmul(orien,np.array(R1))).as_quat()
                orien=R.from_matrix(ori).as_quat()
                matrix=R.from_quat(orien).as_matrix()
                drawline(world_tran, matrix)

                robotiq.pos(position=[ready_pos,orien])
                time.sleep(3)
                robotiq.pos(position=[world_tran,orien])
                time.sleep(3)
                robotiq.grasp(0.085)
                print(gg1)
            except:
                print('no grasp')
                pass
        if ord("a") in keys and keys[ord("a")] & p.KEY_WAS_RELEASED:
            parameter = []
            position_control_group = robotiq.position_control_group
            for i in range(6):
                parameter.append(p.readUserDebugParameter(position_control_group[i]))
            euler=parameter[3:]
            r = R.from_euler('xyz', euler, degrees=False)
            rotation_matrix = r.as_matrix()

            drawline(parameter[:3], rotation_matrix)

            robotiq.pos()
            # p.stepSimulation()
            time.sleep(3)
            robotiq.pos([[0.,0.5,0.5],[0,0,0.,1]])
            time.sleep(3)
            robotiq.grasp()
        if ord("c") in keys and keys[ord("c")] & p.KEY_WAS_RELEASED:
            robotiq.load_obj()
