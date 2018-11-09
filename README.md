# text-index
## Prerequisite
- Python 3.7
- pip18.0
- pipenv 2018.7.1

## install & start
```bash
git clone https://github.com/Angeliz/text-index.git
cd text-index
pipenv install --dev
cp config.py config_local.py
# config your path :)
python setup.py
```

## data
```bash
➜  text-index-data pwd
/Users/user/LocalProjects/text-index-data
```
```bash
.
├── experiment.dat
├── experiment_tfidfspace.dat
├── experimentdataset
│   ├── file01.txt
│   ├── ......
├── seg_experimentdataset
│   ├── file01.txt
│   ├── ......
├── seg_trainingdataset
│   ├── class1
│   │    ├── file01.txt
│   │    └── ......
│   ├── class2
│   ├── class3
│   ├── class4
│   ├── class5
│   ├── class6
│   ├── class7
│   └── class8
├── tfidfspace.dat
├── train.dat
└── trainingdataset
    ├── class1
    │    ├── file01.txt
    │    └── ......
    ├── class2
    ├── class3
    ├── class4
    ├── class5
    ├── class6
    ├── class7
    └── class8
```
