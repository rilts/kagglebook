## 샘플코드

「Kaggle에서 우승하는 데이터분석의 기술」([amazon](https://www.amazon.co.jp/dp/4297108437)) 의 샘플코드입니다.

<img src="misc/cover_small.jpg" width="200">

### 각 폴더의 내용

|폴더| 내용 |
|:----|:-------|
| input | 입력파일 |
| ch01 | 제1장의 샘플코드 |
| ch02 | 제2장의 샘플코드 |
| ch03 | 제3장의 샘플코드 |
| ch04 | 제4장의 샘플코드 |
| ch05 | 제5장의 샘플코드 |
| ch06 | 제6장의 샘플코드 |
| ch07 | 제7장의 샘플코드 |
| ch04-model-interface | 제4장의「분석대회용 클래스와 폴더의 구성」의 코드 |

* 각장의 디렉토리를 현재디렉토리로 해서 코드를 실행해 주세요.
* 제1장의 타이타닉데이터는 [input/readme.md](input/readme.md)에 적힌대로 다운로드해 주세요.
* 제4장의 「분석대회용클래스와 폴더의 구성」의 코드에 대해서는[ch04-model-interface/readme.md](ch04-model-interface)을 참고해 주세요.


### Requirements

샘플코드의 동작은 Google Cloud Platform(GCP)에서 확인했습니다.  

환경은 아래와 같습니다.

* Ubuntu 18.04 LTS  
* Anaconda 2019.03 Python 3.7
* 필요한 Python 패키지(하기 스크립트 참조)

이하의 스크립트대로 GCP의 환경구축을 진행합니다.
```
# utils -----

# 개발에 필요한 툴을 인스톨
cd ~/
sudo apt-get update
sudo apt-get install -y git build-essential libatlas-base-dev
sudo apt-get install -y python3-dev

# anaconda -----

# Anaconda를 다운로드하고 인스톨
mkdir lib
wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O lib/anaconda.sh
/bin/bash lib/anaconda.sh -b

# PATH설정
echo export PATH=~/anaconda3/bin:$PATH >> ~/.bashrc
source ~/.bashrc

# python packages -----

# Python패키지 인스톨
# numpy, scipy, pandas는 Anaconda 2019.03 버전 그대로
# pip install numpy==1.16.2 
# pip install scipy==1.2.1 
# pip install pandas==0.24.2
pip install scikit-learn==0.21.2

pip install xgboost==0.81
pip install lightgbm==2.2.2
pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install hyperopt==0.1.1
pip install bhtsne==0.1.9
pip install rgf_python==3.4.0
pip install umap-learn==0.3.9

# set backend for matplotlib to Agg -----

# GCP상에서 실행하기 때문에 matplotlib의 backend를 다시 지정합니다.
matplotlibrc_path=$(python -c "import site, os, fileinput; packages_dir = site.getsitepackages()[0]; print(os.path.join(packages_dir, 'matplotlib', 'mpl-data', 'matplotlibrc'))") && \
sed -i 's/^backend      : qt5agg/backend      : agg/' $matplotlibrc_path
```
