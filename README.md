Training ML
===========

Datasets are cached in the `~/.dataset/` directory.

Instructions
------------

```shell
virtualenv . -p python3 --system-site-packages && source bin/activate
pip3 install -r requirements.txt
python3 -m training.convnet.py
```

Scripts
-------

| Name | Description | Result |
| ---- | ----------- | ------ |
| `convnet` | A standard convolutional network trained on MINST | 0.73% |
| `lstm` | An LSTM network trained on rows of MINST images | 1.42% |
