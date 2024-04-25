#!/bin/bash

# 도커가 설치되어 있는지 확인
if ! command -v docker &> /dev/null
then
    echo "도커가 설치되어 있지 않습니다. 도커를 설치해 주세요."
    exit 1
fi

# 컨테이너 생성 및 실행 변수 설정
CONTAINER_NAME="one"
IMAGE_NAME="ubuntu:20.04"

# UI permissions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker

# 이미지가 없으면 다운로드
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "$IMAGE_NAME 이미지를 다운로드합니다."
  docker pull $IMAGE_NAME
fi

# 기존 컨테이너 제거
docker rm -f $CONTAINER_NAME &>/dev/null

# 새 컨테이너 생성 및 실행
echo "새 우분투 20.04 컨테이너 생성 및 실행"
docker run -td --privileged --net=host --ipc=host \
    --cpus=8 --memory=5g --memory-swap=5g \
    --name="$CONTAINER_NAME" \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "$XSOCK:$XSOCK:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e "ROS_IP=127.0.0.1" \
    --cap-add=SYS_PTRACE \
    -v `pwd`/oneday_project:/root/oneday_project \
    -w /root/oneday_project \
    $IMAGE_NAME bash

echo "$CONTAINER_NAME 컨테이너가 성공적으로 생성되었습니다."

# ubuntu 기본 설정 설치
docker cp ./setup_basic.sh $CONTAINER_NAME:/root/setup_basic.sh
docker exec -it $CONTAINER_NAME bash -c "/root/setup_basic.sh"

# CPP 개발 환경 설정
docker cp ./setup_cpp.sh $CONTAINER_NAME:/root/setup_cpp.sh
docker exec -it $CONTAINER_NAME bash -c "/root/setup_cpp.sh"

# ROS Noetic 개발 환경 설치
docker cp ./setup_ros.sh $CONTAINER_NAME:/root/setup_ros.sh
docker exec -it $CONTAINER_NAME bash -c "/root/setup_ros.sh"

# 현재 프로젝트 설정
docker cp ./setup_project.sh $CONTAINER_NAME:/root/setup_project.sh
docker exec -it $CONTAINER_NAME bash -c "/root/setup_project.sh"
