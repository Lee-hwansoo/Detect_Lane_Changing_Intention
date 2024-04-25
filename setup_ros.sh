#!/bin/bash
echo 'ROS Noetic 환경 설치를 시작합니다...'

export DEBIAN_FRONTEND=noninteractive

# ROS Noetic 설치
apt-get update && apt-get install -y gnupg curl lsb-release
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
apt-get update && apt-get install -y ros-noetic-desktop-full ros-noetic-teleop-twist-keyboard

# ROS 환경 설정
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc

# 빌드 도구 및 C++ 개발 도구 설치
apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-pip

# 의존성 관리를 위해 rosdep 초기화
rosdep init && rosdep update

echo 'ROS Noetic 환경 설치가 완료되었습니다.'
