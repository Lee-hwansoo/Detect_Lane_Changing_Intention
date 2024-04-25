#!/bin/bash
echo 'C++ 개발 환경 설정을 시작합니다...'

export DEBIAN_FRONTEND=noninteractive

# gcc, g++, cmake 등 필요한 패키지 업데이트 및 설치
apt-get update && apt-get install -y build-essential gcc g++ cmake

echo 'C++ 개발 환경 설정이 완료되었습니다.'
