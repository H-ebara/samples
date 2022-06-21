#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''CRANE側の行動関数

CRANE側の行動関数をまとめたモジュール

'''
from __future__ import print_function
import rospy
import tf
import time
import math
import numpy as np
import pandas as pd
import actionlib
import moveit_msgs
import moveit_commander
import dynamic_reconfigure.client
from queue import Queue
from std_msgs.msg import String
from std_msgs.msg import UInt8
import sys, tty, termios, select
from std_msgs.msg import String,UInt8
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionIKRequest, GetPositionIK
from moveit_msgs.msg import JointConstraint, Constraints
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandActionGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import random
import os
from . import singleton

try:
    import torch
    from torch import nn
except ImportError:
    torch=None

import csv
from csv import reader

pi = math.pi

#たわみ推定のモデル
class DeflectionEstimator(nn.Module):
    '''たわみ推定
    
    アームに生じるたわみ量を推定する関数をまとめたクラス
    
    '''
    #ネットワークの定義、学習済みモデルを読み込む
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2,10)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(10,2)      
        self.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)), "model_0511.pth")))

    #入力をネットワークに順伝搬する
    def forward(self, data):
        '''入力をネットワークに順伝搬

        入力をネットワークに順伝搬する関数

        Args:
            *data(torch.float): 入力値
        
        Returns:
            torch.float: 入力値をネットワークに代入した結果値
        '''
        data = self.linear1(data)
        data = self.sigmoid(data)
        data = self.linear2(data)
        return data


    #入力値からたわみ量を推定する
    def predict(self,length,height):
        '''入力値からたわみ量を推定

        入力値からたわみ量を推定する関数

        Args:
            *length(float): 目標値までの距離

            *height(float): 目標値の高さ

        Returns:
            numpy.float64: 推定されたたわみ量
        '''
        inputs = torch.Tensor([length*100,height*100])
        pred = self(inputs).detach().numpy()
        deflection = np.round((height - pred[1]/100),3)
        print("推定たわみ値:",deflection)
        return deflection


#Craneの行動関数
class Crane(singleton.Singleton):
    '''CRANEの行動関数クラス

    CRANEの行動関数をまとめたクラス
    
    '''
    
    #インスタンス変数定義
    def __init__(self):
        super(Crane, self).__init__()
        self.robot = moveit_commander.RobotCommander()
        self.arm = moveit_commander.MoveGroupCommander("arm")               
        self.gripper = moveit_commander.MoveGroupCommander("gripper")
        self.pub_preset = rospy.Publisher("preset_gain_no", UInt8, queue_size=1)

        self.joint_names = [
            "/crane_x7/crane_x7_control/crane_x7_shoulder_fixed_part_pan_joint", 
            "/crane_x7/crane_x7_control/crane_x7_shoulder_revolute_part_tilt_joint", 
            "/crane_x7/crane_x7_control/crane_x7_upper_arm_revolute_part_twist_joint", 
            "/crane_x7/crane_x7_control/crane_x7_upper_arm_revolute_part_rotate_joint", 
            "/crane_x7/crane_x7_control/crane_x7_lower_arm_fixed_part_joint", 
            "/crane_x7/crane_x7_control/crane_x7_lower_arm_revolute_part_joint", 
            "/crane_x7/crane_x7_control/crane_x7_wrist_joint", 
            "/crane_x7/crane_x7_control/crane_x7_gripper_finger_a_joint"
        ]

        self.joint_names1 = [
            "crane_x7_shoulder_fixed_part_pan_joint", 
            "crane_x7_shoulder_revolute_part_tilt_joint", 
            "crane_x7_upper_arm_revolute_part_twist_joint", 
            "crane_x7_upper_arm_revolute_part_rotate_joint", 
            "crane_x7_lower_arm_fixed_part_joint", 
            "crane_x7_lower_arm_revolute_part_joint", 
            "crane_x7_wrist_joint"
        ]


        #シミュレーション環境のときはコメントアウト
        #self.param_clients = [ dynamic_reconfigure.client.Client(name, timeout=10 ) for name in self.joint_names ]

        self.joint_names_for_ik = [
        "crane_x7_shoulder_fixed_part_pan_joint", 
        "crane_x7_shoulder_revolute_part_tilt_joint", 
        "crane_x7_upper_arm_revolute_part_twist_joint", 
        "crane_x7_upper_arm_revolute_part_rotate_joint", 
        "crane_x7_lower_arm_fixed_part_joint", 
        "crane_x7_lower_arm_revolute_part_joint", 
        "crane_x7_wrist_joint"      
        ]   

        #IK計算に用いるインスタンス変数
        self.get_ik = rospy.ServiceProxy("compute_ik", GetPositionIK)
        rospy.wait_for_service("/compute_ik")

        #たわみ推定のインスタンス変数
        if torch is not None:
            self.defle = DeflectionEstimator()

        self.gripper_pub = rospy.Publisher('/crane_x7/gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)


    #移動速度を変更
    def set_scalefactor(self, vel=0.5, acc=1.0):
        '''アームの移動速度を変更する

        アームの移動速度を変更する関数、引数に何も入れなければ、デフォルトの速度で動く

        Args:
            *vel(float): 速度を決定するパラメータ
            *acc(float): 加速度を決定するパラメータ
            
        '''       
        self.arm.set_max_velocity_scaling_factor(vel)
        self.arm.set_max_acceleration_scaling_factor(acc) 


    #初期姿勢に戻る
    def go_home(self):
        '''アームをhome姿勢に戻す

        アームをhome姿勢に戻す関数

        Returns:
            bool: home 姿勢に到達したらTrueを返す
        '''        
        self.gripper.set_joint_value_target([0.7, 0.7])
        self.gripper.go()
        self.arm.set_named_target("home")
        ret = self.arm.go()
        return ret


    #直立させる
    def stand_up(self):
        '''アームを直立姿勢にする

        アームを直立姿勢にする関数

        Returns:
            bool: 直立姿勢に到達したらTrueを返す 
        '''             
        self.gripper.set_joint_value_target([0.7, 0.7])
        self.gripper.go()
        self.arm.set_named_target("vertical")
        ret = self.arm.go()
        return ret


    #グリッパを開く
    def open_gripper(self, _open=True ):
        '''グリッパ開閉

        グリッパを開閉する関数

        Args:
            *_open(bool): bool値でグリッパの開閉を選択する(True→開く、False→閉じる)

        Returns:
            bool: グリッパの開閉が完了したらTrueを返す
        '''
        self.gripper.set_start_state_to_current_state()
        if _open:
            self.gripper.set_joint_value_target([0.7, 0.7])
        else:
            self.gripper.set_joint_value_target([0.1, 0.1])
        ret = self.gripper.go()
        return ret


    #座標・姿勢を指定して移動(引数でたわみ補正のON/OFFを切替可能)
    def set_pose(self, x, y, z, rx, ry, rz, correct=True):
        '''座標・姿勢を指定して移動

        座標・姿勢を指定して移動する関数

        Args:
            *x(numpy.float64): 指定位置のx座標

            *y(numpy.float64): 指定位置のy座標

            *z(numpy.float64): 指定位置のz座標

            *rx(numpy.float64): 指定姿勢のオイラー角のx軸周りの回転

            *ry(numpy.float64): 指定姿勢のオイラー角のy軸周りの回転

            *rz(numpy.float64): 指定姿勢のオイラー角のz軸周りの回転

            *correct(bool): bool値でたわみ補正をするかを選択する(True→補正する、False→しない)

        Returns:
            bool:アームが目標位置に到達したらTrueを返す
        '''

        #補正する場合は、物体までの距離から補正値を推定
        if correct:
            if torch is not None:
                print("補正あり")
                #物体までの距離を計算
                length = math.sqrt(x**2+y**2)

                #補正値を計算
                correction = self.defle.predict(length, z)
                z+=correction 

        else:
            print("補正なし")

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        rx,ry,rz,rw = tf.transformations.quaternion_from_euler(rz, ry, rx)
        pose.orientation.x = rx
        pose.orientation.y = ry
        pose.orientation.z = rz
        pose.orientation.w = rw

        #補正して指定位置へ移動
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose)
        ret = self.arm.go()
        return  ret


    #アームのトルクをON/OFFにする
    def enable_torque(self, enable=True ):
        '''アームのトルクをON/OFFにする

        アームのトルクをON/OFFにする関数

        Args:
            *enable(bool): bool値でトルクをON/OFFを選択する(True→ON、False→OFF)
        '''
        def send_gain( pgain ):
            for client in self.param_clients:
                client.update_configuration( {"position_p_gain":pgain, "position_i_gain":0,"position_d_gain":0} )

        if enable:
            #いきなり動きだすので，今の姿勢を目標姿勢にする
            self.arm.set_joint_value_target( self.arm.get_current_joint_values() )
            self.arm.go()
            self.gripper.set_joint_value_target(self.gripper.get_current_joint_values())
            self.gripper.go()

            #徐々にトルクをもとに戻す
            send_gain(100)
            time.sleep(1)
            send_gain(200)
            time.sleep(1)
            send_gain(400)
            time.sleep(1)
            send_gain(800)
        else:
            send_gain(1)
        time.sleep(1)


    #1つのタイムステップについてIK計算する
    def compute_ik(self, x, y, z, rz, ry, rx):
        '''IK計算

        1つのタイムステップについてIK計算する関数

        Args:
            *x(numpy.float64): 指定位置のx座標

            *y(numpy.float64): 指定位置のy座標

            *z(numpy.float64): 指定位置のz座標

            *rx(numpy.float64): 指定姿勢のオイラー角のx軸周りの回転

            *ry(numpy.float64): 指定姿勢のオイラー角のy軸周りの回転

            *rz(numpy.float64): 指定姿勢のオイラー角のz軸周りの回転

        Returns:
            list: 目標位置・姿勢を実現する7つの関節角を返す
        '''        

        #関節に制約をつける
        def def_const(name, position, above, below, weight):
            '''関節に制約をつける

            関節に制約をつける関数

            Args:
                *position(int): 基準とする角度

                *above(float): 関節角の上限値を設定(position+aboveが最大角)

                *below(float): 関節角の下限値を設定(position-belowが最小角)     

                *weight(float): 他の関節角に対する重み(0〜1の範囲)

            Returns:
                moveit_msgs.msg._JointConstraint.JointConstraint: 各関節角の可動範囲
            '''
            joint_constraint = moveit_msgs.msg.JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = position
            #上限
            joint_constraint.tolerance_above = above
            #下限
            joint_constraint.tolerance_below = below
            #他の関節角に対する重み
            joint_constraint.weight = weight

            return joint_constraint


        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        x,y,z,w = tf.transformations.quaternion_from_euler(rz, ry, rx)
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        
        req = GetPositionIKRequest()
        req.ik_request.group_name = "arm"
        req.ik_request.robot_state = self.robot.get_current_state()
        req.ik_request.pose_stamped = pose

        #制約を入れる関節
        joint_constraint0 = def_const('crane_x7_shoulder_fixed_part_pan_joint', 0, 1.5, 1.5, 1.0)
        joint_constraint1 = def_const('crane_x7_shoulder_revolute_part_tilt_joint', 0, 0.5, 2.5, 1.0)
        joint_constraint2 = def_const('crane_x7_upper_arm_revolute_part_twist_joint', 0, 1.0, 1.0, 1.0)
        joint_constraint3 = def_const('crane_x7_upper_arm_revolute_part_rotate_joint', 0, 0, 2.7, 1.0)
        joint_constraint4 = def_const('crane_x7_lower_arm_fixed_part_joint', 0, 1.5, 1.5, 1.0)
        joint_constraint5 = def_const('crane_x7_lower_arm_revolute_part_joint', 0.5, 1.5, 2.0, 1.0)


        constraints = Constraints()
        constraints.joint_constraints = [joint_constraint0,joint_constraint1, joint_constraint2,joint_constraint3,joint_constraint4,joint_constraint5]

        req.ik_request.constraints = constraints

        ik = self.get_ik(req)

        if ik.error_code.val==1:
            #print("IK成功")
            solution = list(ik.solution.joint_state.position[:7])
            return solution
        else:
            print("IK失敗")
            return None


    #すべてのタイムステップについて複数回IK計算する
    def compute_ik_multiple(self, num_ik_loop, poses):
        '''IK計算(複数回)

        全てのタイムステップについて複数回IK計算する関数

        Args:
            *num_ik_loop(int): IK計算を繰り返す回数

            *poses(list): タイムステップ順に保存された位置・姿勢のリスト
        
        Returns:
            list: IK計算でタイムステップ順に算出された7つの関節角のリスト
        '''        
        #IKをk回繰り返し解いて、求めた関節角を配列に追加
        joint = {}
        time_step = len(poses)

        for k in range(num_ik_loop):
            joint[k] = []
            print("IK計算回数:",k+1)

            for i in range(time_step):
                joints = self.compute_ik(poses[i][0], poses[i][1], poses[i][2], poses[i][3], poses[i][4], poses[i][5])

                if joints==None:
                    joints=[]                
                joint[k].append(joints)


        print("IK計算終了")

        solutions = []
        loss_sum = []
        loss = {}
        sums = {}
        lmin = {}

        #1番目の行をk個の解の中からランダムに抽出
        rand = random.randint(0, num_ik_loop-1)
        extract = np.array(joint[rand][0][:])
        solutions.append(extract)

        for k in range(time_step-1):          
            for i in range(num_ik_loop):

                if len(joint[i][k+1][:])==7:
                    nexts = np.array(joint[i][k+1][:])

                    #7つの関節角に対して、タイムステップ間の値の差を計算
                    loss[i] = [(x-y)**2 for (x,y) in zip( extract, nexts )]

                    #二乗誤差の和を計算
                    sums[i] = np.sqrt(np.sum(loss[i]))
                    loss_sum.append(sums[i])

                else:
                    pass

            #loopした回数の中で誤差が最小のIDを取得
            lmin = np.argmin(loss_sum)

            #最小のIDの行を抽出
            extract = np.array(joint[lmin][k+1][:])

            solutions.append(extract)         
            loss_sum.clear()

        return solutions


    #動作軌道を生成する
    def generate_trajectory(self,joint_angles, duration=0.1):
        '''軌道を生成

        IK計算で得られた関節角をタイムステップ順に読み出して、軌道を生成する関数

        Args:
            *joint_angles(list): タイムステップ順に並べられた、7つの関節角のリスト

            *duration(float): タイムステップの軌道を補間する間隔、小さいほど動作が速くなる(デフォルト値は0.1）
        
        Returns:
        '''
        #動作生成に必要なタイムステップ順の関節角とグリッパ角を読み込む
        final = pd.DataFrame(joint_angles, columns=self.joint_names1)
        df_gripper = pd.read_csv(filepath_or_buffer="/home/nakalab/デスクトップ/learned_trajectory/1951_joints.csv")
        gripper_val = df_gripper.iloc[:,[2, 5, 6, 8, 7, 3, 4, 9]].values[:].tolist()

        # JointStateをFollowJointTrajectoryGoalに変換して送信
        action_client = actionlib.SimpleActionClient( "/crane_x7/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction )
        action_client.wait_for_server()
    

        #グリッパ同期
        def time_gripper(gripper_val, tempdata, idx, vel=2):
            '''グリッパを同期

            生成された軌道と、グリッパの開閉のタイミングを同期させる関数

            Args:
                *joint_angles(list): タイムステップ順に並べられた、7つの関節角のリスト

                *duration(float): タイムステップの軌道を補間する間隔、小さいほど動作が速くなる(デフォルト値は0.1）
            
            Returns:
            '''
            time = tempdata.header.seq -idx
            if time % vel == 0 and time < len(gripper_val*vel):
                temptime = int(time/vel)
                goal_gripper = GripperCommandActionGoal()
                goal_gripper.goal.command.position = gripper_val[temptime][0]
                self.gripper_pub.publish( goal_gripper )

        # JointStateをFollowJointTrajectoryGoalに変換して送信
        all_joint_values = final.iloc[:]
        joint_goal = FollowJointTrajectoryGoal()
        joint_goal.trajectory.joint_names = self.joint_names1

        if all_joint_values.shape[0]> 0:
            for i in range(all_joint_values.shape[0]):
                joint_values = tuple(all_joint_values.iloc[i,:].values.tolist())
                jp = JointTrajectoryPoint( positions=joint_values, time_from_start=rospy.Duration((i+1)*duration))
                joint_goal.trajectory.points.append( jp )

        # gripper以外の角度情報をsend
        action_client.send_goal( joint_goal )
        q = Queue()
        sub = rospy.Subscriber('/crane_x7/arm_controller/follow_joint_trajectory/feedback', String ,lambda msg: q.put(msg))

        n = False
        while 1:
            # 動作が終了したらhomeに戻る
            if action_client.get_result() != None:
                self.go_home()
                break
            tempdata = q.get()

            #グリッパを同期させる
            if n == False:
                idx = tempdata.header.seq
                n = True
            time_gripper(gripper_val, tempdata, idx)

