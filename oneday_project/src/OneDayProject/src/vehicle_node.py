#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import pickle

import numpy as np
import random
import rospy
import math
import copy
import rospkg
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
import tf

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray
from hmmlearn.hmm import GMMHMM, GaussianHMM

from msgs.msg import dataset_array_msg, dataset_msg, map_array_msg, map_msg, point_msg

from scipy.stats import multivariate_normal

class CTRA():
    def __init__(self, dt=0.1):
        """
        Initializes the CTRA model with a fixed time step.
        :param dt: Time step for simulation in seconds.
        """
        self.dt = dt

    def step(self, x):
        """
        Performs a single step of motion update based on the CTRA model.
        :param x: State vector [x, y, v, a, theta, theta_rate] representing
                  position (x, y), velocity (v), acceleration (a),
                  heading (theta), and rate of turn (theta_rate).
        :return: Updated state vector after applying the motion model.
        """
        px, py, v, a, yaw, r = x
        dt = self.dt

        # Check the turn rate to prevent division by zero
        if np.abs(r)>0.1:
            self.x = [px+v/r*(np.sin(yaw+r*dt)-np.sin(yaw))+v/(r**2)*(np.cos(yaw+r*dt)+dt*r*np.sin(yaw+r*dt)-np.cos(yaw)),
                      py+v/r*(-np.cos(yaw+r*dt)+np.cos(yaw))+v/(r**2)*(np.sin(yaw+r*dt)-dt*r*np.cos(yaw+r*dt)-np.sin(yaw)),
                      v+a*dt,
                      a,
                      yaw+r*dt,
                      r]
        else:
            self.x = [px+v*np.cos(yaw)*dt,
                      py+v*np.sin(yaw)*dt,
                      v+a*dt,
                      a,
                      yaw,
                      r]

        return self.x

    def H(self, x):
        """
        Measurement function that maps the true state space into the observed space.
        :param x: State vector.
        :return: Observed state vector [x, y, v, theta].
        """
        return np.array([x[0],x[1],x[2],x[4]])

    def JA(self, x, dt = 0.1):
        """
        Computes the Jacobian matrix of the state transition function with respect to the state.
        :param x: State vector.
        :return: Jacobian matrix of the state transition function.
        """
        px, py , v , a, yaw, r = x


        # upper
        if np.abs(r)>0.1:
            JA_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]
        else:
            JA_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,dt],
                    [0,0,0,0,0,1]]

        return np.array(JA_)

    def JH(self, x, dt = 0.1):
        px, py, v, a, yaw, r = x

        # upper
        if np.abs(r)>0.1:
            JH_ = [[1,0,(np.sin(yaw+r*dt)-np.sin(yaw))/r,(-np.cos(yaw)+np.cos(yaw+r*dt)+r*dt*np.sin(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.cos(yaw+r*dt)-a*np.sin(yaw+r*dt)-v*r*np.cos(yaw)+a*np.sin(yaw))/r**2,
                    -2/r**3*((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))+
                    ((v+a*dt)*np.sin(yaw+r*dt)+dt*(r*v+a*r*dt)*np.cos(yaw+r*dt)-dt*a*np.sin(yaw+r*dt)-v*np.sin(yaw))/r**2],
                    [0,1,(-np.cos(yaw+r*dt)+np.cos(yaw))/r,(-np.sin(yaw)+np.sin(yaw+r*dt)-r*dt*np.cos(yaw+r*dt))/r**2,
                    ((r*v+a*r*dt)*np.sin(yaw+r*dt)+a*np.cos(yaw+r*dt)-v*r*np.sin(yaw)-a*np.cos(yaw))/r**2,
                    -2/r**3*((-r*v-a*r*dt)*np.cos(yaw+r*dt)+a*np.sin(yaw+r*dt)+v*r*np.cos(yaw)-a*np.sin(yaw))+
                    ((-v-a*dt)*np.cos(yaw+r*dt)+dt*(r*v+a*r*dt)*np.sin(yaw+r*dt)+a*dt*np.cos(yaw+r*dt)+v*np.cos(yaw))/r**2],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        else:
            JH_ = [[1, 0 , np.cos(yaw)*dt, 1/2*np.cos(yaw)*dt**2,-(v+1/2*a*dt)*np.sin(yaw)*dt ,0],
                    [0, 1 , np.sin(yaw)*dt, 1/2*np.sin(yaw)*dt**2, (v+1/2*a*dt)*np.cos(yaw)*dt,0],
                    [0,0,1,dt,0,0],
                    [0,0,0,0,1,dt]]

        return np.array(JH_)

    def pred(self, x, future_step=10):
        self.x = x

        x_list = [self.x]
        for t in range(future_step):
            x_list.append(self.step(self.x))

        return np.array(x_list)

class Extended_KalmanFilter:
    def __init__(self, x_dim, z_dim):
        """
        Initializes the Extended Kalman Filter.
        :param x_dim: Dimension of the state vector.
        :param z_dim: Dimension of the measurement vector.
        """
        self.x = np.zeros((x_dim, 1))  # State vector
        self.P = np.eye(x_dim)         # State covariance matrix
        self.Q = np.eye(x_dim)         # Process noise covariance
        self.R = np.eye(z_dim)         # Measurement noise covariance
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.JA = None
        self.JH = None
        self.F = (lambda x:x)
        self.H = (lambda x:np.zeros(z_dim,1))
        self._I = np.eye(x_dim)  # Identity matrix
        self.likelihood = 1.0

    def predict(self, u=None, JA=None, F=None, Q=None):
        """
        Prediction step of the EKF.
        :param JA: Jacobian matrix of the state transition function, if available.
        :param F: State transition function, if available.
        :param Q: Process noise covariance matrix, if available.
        """
        if F is None:
            F = self.F  # Default to identity function
        if Q is None:
            Q = self.Q
        if JA is None:
            if self.JA is None:
                JA_ = np.eye(self.x_dim)
            else:
                JA_ = self.JA(self.x)
        else:
            JA_ = JA(self.x)

        self.x = F(self.x)
        self.P = np.dot(np.dot(JA_, self.P), JA_.T) + Q

    def correction(self, z, JH=None, H=None, R=None):
        """
        Correction step of the EKF using the new measurement.
        :param z: Measurement vector.
        :param JH: Jacobian matrix of the measurement function, if available.
        :param H: Measurement function, if available.
        :param R: Measurement noise covariance matrix, if available.
        """
        if H is None:
            H = self.H  # Default to zero measurement function
        if R is None:
            R = self.R
        if JH is None:
            if self.JH is None:
                JH_ = np.zeros((self.x_dim,self.z_dim))
            else:
                JH_ = self.JH(self.x)
        else:
            JH_ = JH(self.x)

        z_pred = H(self.x)
        self.y = z - z_pred  # Innovation

        PHT = np.dot(self.P, JH_.T)
        S = np.dot(JH_, PHT) + R
        SI = np.linalg.inv(S)
        self.K = np.dot(PHT, SI)  # Kalman gain
        self.x += np.dot(self.K, self.y)
        I_KH = self._I - np.dot(self.K, JH_)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        self.likelihood = multivariate_normal.pdf(self.y, np.zeros_like(self.y), S)


class Predictor:
    def __init__(self, dt, T, w):
        self.dt = dt  # 시간 간격
        self.T = T  # 예측 시간
        self.w = w  # 조작 기반 예측에 사용되는 폭 변수

    def handle_vehicle_data(self, veh_data):
        """ 차량 데이터를 처리하여 필요한 변수를 계산하고 반환합니다. """
        x = veh_data[:, 4]
        y = veh_data[:, 5]
        v = veh_data[:, 7]
        v_diff = np.diff(veh_data[:, 7])
        a = np.concatenate([[0], v_diff / self.dt])
        yaw = veh_data[:, 6]
        yaw_diff = np.diff(veh_data[:, 6])
        yaw_rate = np.concatenate([[0], yaw_diff / self.dt])
        return x, y, v, a, yaw, yaw_rate

    def physics_based_prediction(self, state_history):
        """ 물리 기반 예측을 수행합니다. """
        filters = Extended_KalmanFilter(6, 4)
        model = CTRA(dt=self.dt)
        Q = [0.1, 0.1, 0.1, 0.1, 0.001, 0.01]
        filters.F = model.step
        filters.H = model.H
        filters.JA = model.JA
        filters.JH = model.JH
        filters.Q = np.diag(Q)
        filters.R = np.diag([0.1, 0.1, 0.1, 0.01])
        filters.x = state_history[:,0]

        z_history = state_history[:4,1:]  # state, x는 [x, y, v, yaw] 형식

        for z in z_history.T:
            filters.predict()             # 예측 단계
            filters.correction(z)         # 보정 단계

        future_step = int(self.T/self.dt)
        pred = model.pred(filters.x, future_step=future_step)
        return pred.T

    def maneuver_based_prediction(self, d_history, state_history):
        """ 조작 기반 예측을 수행합니다. """
        future_step = int(self.T/self.dt)
        sigma_dela = 0.02 # 불확실성을 모델링하는 데 사용되는 변동성 파라미터

        # 조작 기반 이동 가능성 계산
        # 도로 따라 주행 확률: 차선 중앙에서 가까울 수록 증가
        p_lk = lambda d: multivariate_normal.pdf(d, 0, 0.3)
        # 차선 변경 확률: 차선 중앙에서 벗어날수록 확률 증가
        p_lc = lambda d: multivariate_normal.pdf(d, 1, 0.6) + multivariate_normal.pdf(d, -1, 0.6)

        P_LK = []
        P_LC = []

        # 각 시점에서의 확률 계산
        for i, d in enumerate(d_history):
            plk = p_lk(d)
            plc = p_lc(d)
            total_p = plk + plc

            P_LK.append(plk/total_p)
            P_LC.append(plc/total_p)

        p_lk_f = 0
        p_lc_f = 0
        alpha = 0.3
        for plc, plk in zip(P_LC, P_LK):
            p_lk_f = alpha * plk + (1 - alpha) * p_lk_f
            p_lc_f = alpha * plc + (1 - alpha) * p_lc_f


        # 물리 기반 예측을 위한 최종 상태
        x, y, v, a, yaw, yaw_rate = state_history[:, -1]
        d = d_history[-1]

        # 초기 위치
        Xm = [x]
        Ym = [y]

        # 차선 변경의 확률이 차선 유지의 확률보다 클 때
        if p_lc_f > p_lk_f:
            return "LC"
        else:
            return "LK"

    def model_mixing_prediction(self, physics_based_trajectory, maneuver_based_trajectory):
        """ 물리 기반 예측과 조작 기반 예측을 혼합하여 예측을 수행합니다. """
        ft = lambda t : (1-1/(1+np.exp(-3*(t-1.2))))

        future_step = int(self.T/self.dt)
        Xp = physics_based_trajectory[0,]
        Yp = physics_based_trajectory[1,]
        Xm = maneuver_based_trajectory[0]
        Ym = maneuver_based_trajectory[1]

        X = []
        Y = []

        for t in range(future_step):
            wp = ft(t*self.dt)
            wm = 1-wp

            X.append(wp*Xp[t]+wm*Xm[t])
            Y.append(wp*Yp[t]+wm*Ym[t])

        return np.array([X,Y])

class Environments(object):
    def __init__(self):
        rospy.init_node('Environments')

        self.init_variable()
        self.set_subscriber()
        self.set_publisher()
        self.load_map()

        r = rospy.Rate(20)
        while not rospy.is_shutdown():

            self.loop()
            r.sleep()


    def load_map(self):
        self.map_file = []
        map_path = rospy.get_param("map_path")
        matfiles = ["waypoints_0_rev.mat",
                    "waypoints_1_rev.mat",
                    "waypoints_2_rev.mat",
                    "waypoints_3_rev.mat"
                    ]

        # 124.20403395607009 738.5587856439931 1356.4053875886939
        station_offset = [0, 0, 0, 0]

        for i, matfile in enumerate(matfiles):
            mat = sio.loadmat(map_path+matfile)

            easts = mat["east"][0]
            norths = mat["north"][0]
            stations = mat["station"][0]+station_offset[i]

            # if i==2:
            self.map_file.append(np.stack([easts, norths, stations],axis=-1))


        self.D_list = [ 0, -3.85535188 ,-7.52523438, -7.37178602]
        self.D_list = np.array(self.D_list)


    def init_variable(self):
        self.pause = False
        self.time = 11
        self.br = tf.TransformBroadcaster()

        SamplePath = rospy.get_param("SamplePath")
        SampleList = sorted(glob.glob(SamplePath+"/*.pickle"))
        SampleId = (int)(rospy.get_param("SampleId"))

        ######################### Load Vehicle #######################

        with open(SampleList[SampleId], 'rb') as f:
            self.vehicles = pickle.load(f)

        # self.Logging[veh.track_id].append([veh.lane_id, veh.target_lane_id, veh.s,
        #         #                          veh.d, veh.pose[0], veh.pose[1], veh.pose[2], veh.v, veh.yawrate, MODE[veh.mode], veh.ax, veh.steer, veh.length, veh.width])

        ########################## HMM ###############################
        # with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LC.pickle", 'rb') as f:
        #     self.hmm_lc = pickle.load(f)

        # with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LK.pickle", 'rb') as f:
        #     self.hmm_lk = pickle.load(f)


    def loop(self):
        if self.pause:
            pass
        else:
            self.publish()
            self.pub_map()
            self.time+=1

        if self.time>=(len(self.vehicles[0])-1):
            rospy.signal_shutdown("End of the logging Time")
            # asdf


    def callback_plot(self, data):
        if data.linear.x>0 and data.angular.z>0: #u
            self.pause = True
        else:
            self.pause = False


    def publish(self, is_delete = False):
        ObjectsData = dataset_array_msg()

        for i in range(len(self.vehicles)):
            ObjectData = dataset_msg()
            ObjectData.id = i
            ObjectData.lane_id = self.vehicles[i][self.time][0]
            ObjectData.length = self.vehicles[i][self.time][12]
            ObjectData.width = self.vehicles[i][self.time][13]

            for t in range(self.time-10, self.time+1):
                ObjectData.x.append(self.vehicles[i][t][4])
                ObjectData.y.append(self.vehicles[i][t][5])
                ObjectData.yaw.append(self.vehicles[i][t][6])
                ObjectData.vx.append(self.vehicles[i][t][7])
                ObjectData.s.append(self.vehicles[i][t][2])
                ObjectData.d.append(self.vehicles[i][t][3])

            ObjectsData.data.append(ObjectData)

        self.history_pub.publish(ObjectsData)


    def callback_result(self, data):
        Objects = MarkerArray()
        Texts = MarkerArray()

        predictors = [Predictor(dt=0.05, T=0.4, w=3.833) for _ in range(len(self.vehicles))]

        for i, predictor in enumerate(predictors):
            veh_data = np.array(self.vehicles[i][self.time-10:self.time+1])
            """
            To Do
            i번째 veh history data인 veh_data를 활용하여 LC intention에 대한 pred 수행

            veh_data의 row
            [-1:] : 현재 관측 데이터
            [:-1] : 과거 관측 데이터

            veh_data의 column
            0 : lane_id
            1 : target_lane_id ( unobservable )
            2 : s (1차선 시작 지점 기준)
            3 : d (1차선 기준 감소)
            4 : GX (Global X 좌표 (GPS))
            5 : GY (Global Y 좌표 (GPS))
            6 : Gyaw (Global heading (GPS))
            7 : v (차량 속도, m/s)
            8 : yawrate ( 차량 yawrate, rad/s ) ( unobservable )
            9 : Mode ( LK, LC mode ‒ GT ) ( pred target )
            10 : ax ( 차량 가속도, m^2/s ) ( unobservable )
            11 : steer ( 차량 steering angle, rad ) ( unobservable )
            12 : length ( 차량 length )
            13 : width ( 차량 width )
            """

            x, y, v, a, yaw, yaw_rate = predictor.handle_vehicle_data(veh_data)

            # rows : x, y, v, a, yaw, yaw_rate, columns : time
            state_history = np.array([x, y, v, a, yaw, yaw_rate])

            trajectory_physics = predictor.physics_based_prediction(state_history)

            d_history = veh_data[:,3] - self.D_list[veh_data[:,0].astype(int)]

            pred = predictor.maneuver_based_prediction(d_history, state_history)

            gt = "LC" if self.vehicles[i][self.time][9] == 1 else "LK"

            q = tf.transformations.quaternion_from_euler(0, 0, self.vehicles[i][self.time][6])  # yaw

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CUBE

            marker.pose.position.x = self.vehicles[i][self.time][4]   # x
            marker.pose.position.y = self.vehicles[i][self.time][5]   # y
            marker.pose.position.z = 0.5

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = self.vehicles[i][self.time][12]    # length
            marker.scale.y = self.vehicles[i][self.time][13]    # width
            marker.scale.z = 1
            marker.color.a = 1.0

            Objects.markers.append(marker)

            text = Marker()
            text.header.frame_id = "world"
            text.ns = "text"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING

            text.action = Marker.ADD

            text.color = ColorRGBA(1, 1, 1, 1)
            text.scale.z = 5
            text.text = str(i)+" / True : " + gt+" / Pred : "  + pred
            text.pose.position = Point(self.vehicles[i][self.time][4], self.vehicles[i][self.time][5], 3)

            Texts.markers.append(text)

        self.sur_pose_plot.publish(Objects)
        self.text_plot.publish(Texts)

        self.br.sendTransform((self.vehicles[0][self.time][4], self.vehicles[0][self.time][5], 0),
                                tf.transformations.quaternion_from_euler(0, 0,self.vehicles[0][self.time][6]),
                                rospy.Time.now(),
                                "base_link",
                                "world")


    def pub_map(self, is_delete = False):
        MapData = map_array_msg()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i

            temp = self.map_file[i]
            for j in range(len(temp)):
                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            MapData.data.append(MapSeg)
        self.map_pub.publish(MapData)

        Maps = MarkerArray()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i

            line_strip = Marker()
            line_strip.type = Marker.LINE_STRIP
            line_strip.id = i
            line_strip.scale.x = 2
            line_strip.scale.y = 0.1
            line_strip.scale.z = 0.1

            line_strip.color = ColorRGBA(1.0,1.0,1.0,0.5)
            line_strip.header = Header(frame_id='world')

            temp = self.map_file[i]
            for j in range(len(temp)):
                point = Point()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.z = 0

                line_strip.points.append(point)

                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            Maps.markers.append(line_strip)

        self.map_plot.publish(Maps)


    def set_subscriber(self):
        rospy.Subscriber('/cmd_vel',Twist, self.callback_plot,queue_size=1)
        rospy.Subscriber('/result', dataset_array_msg, self.callback_result, queue_size=1)


    def set_publisher(self):
        self.sur_pose_plot = rospy.Publisher('/rviz/sur_obj_pose', MarkerArray, queue_size=1)
        self.map_plot = rospy.Publisher('/rviz/maps', MarkerArray, queue_size=1)
        self.text_plot = rospy.Publisher('/rviz/text', MarkerArray, queue_size=1)
        self.map_pub = rospy.Publisher('/map_data', map_array_msg, queue_size=1)
        self.history_pub = rospy.Publisher('/history', dataset_array_msg, queue_size=1)



if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')

