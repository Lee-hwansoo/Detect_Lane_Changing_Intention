#!/bin/bash
echo '기본 설정 및 패키지 설치를 시작합니다...'

# 비대화형 설정
export DEBIAN_FRONTEND=noninteractive

# 필요한 패키지 업데이트 및 설치
apt-get update && apt-get install -y wget git curl locales zip gpg gedit

# 지역 설정
locale-gen ko_KR.UTF-8 && update-locale LANG=ko_KR.UTF-8

# vscode 설치
# wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
# install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
# sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
# rm -f packages.microsoft.gpg
# apt-get install apt-transport-https && apt-get update && apt-get install -y code

echo '기본 설치가 완료되었습니다.'
