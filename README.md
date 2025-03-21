Only for Bro(师弟)

Project struct as follow

```
├── base_aff.py    // for AffectNet dataset train and eval
├── base_fer.py    // for RAF-DB dataset train and eval
├── data          // struct for dataset folder
├── experiment    //folder for results
├── models        // named as struct
│   ├── convnext.py
│   ├── __init__.py
│   └── __pycache__
│       ├── convnext.cpython-38.pyc
│       └── __init__.cpython-38.pyc
├── utils        // func utils, as metric, cost func
│   ├── loss.py
│   ├── metric.py
│   ├── __pycache__
│   │   ├── loss.cpython-38.pyc
│   │   ├── metric.cpython-38.pyc
│   │   └── utli.cpython-38.pyc
│   └── utli.py
└── weight      // folder for pretrained weight and saved weight
    ├── model
    └── pretrain
```
