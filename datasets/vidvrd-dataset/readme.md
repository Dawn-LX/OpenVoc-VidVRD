download dataset from https://xdshang.github.io/docs/imagenet-vidvrd.html

refer to https://github.com/xdshang/VidVRD-helper

The dir structure is like:

(**NOTE**: actually, the raw video data (.mp4 files) is not required to run this repo)
```
datasets/
|   vidvrd-dataset/
|   |   train/
|   |   |   ILSVRC2015_train_00005003.json
|   |   |   ...
|   |   test/
|   |   |   ILSVRC2015_train_00005004.json
|   |   |   ...
|   |   videos/
|   |   |   ILSVRC2015_train_00005003.mp4
|   |   |   ILSVRC2015_train_00005004.mp4
|   |   |   ...
|   vidor-dataset/
|   |   annotation/
|   |   |   training/
|   |   |   |   0000/
|   |   |   |   |   2401075277.json
|   |   |   |   |   ...
|   |   |   |   0001/
|   |   |   |   |   ...
|   |   |   validation/
|   |   |   |   0001/
|   |   |   |   |   2793806282.json
|   |   |   |   |   ...
|   |   |   |   0004/
|   |   |   |   |   ...
|   |   videos/
|   |   |   0000/
|   |   |   |   2401075277.mp4
|   |   |   |   ....
```