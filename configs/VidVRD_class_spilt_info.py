# class_split_info = {
#     "cls2id":{"person": 1, "dog": 2, ...},
#     "id2cls":{1: "person", 2: "dog", ...},
#     "cls2split":{"person":"base",...,"watercraft":"novel",...}
# }

class_split_info = {"cls2id": {"__background__": 0, "airplane": 1, "bicycle": 2, "bird": 3, "bus": 4, "car": 5, "dog": 6, "domestic_cat": 7, "elephant": 8, "hamster": 9, "lion": 10, "monkey": 11, "rabbit": 12, "sheep": 13, "snake": 14, "squirrel": 15, "tiger": 16, "train": 17, "turtle": 18, "whale": 19, "zebra": 20, "ball": 21, "frisbee": 22, "sofa": 23, "skateboard": 24, "person": 25, "horse": 26, "watercraft": 27, "giant_panda": 28, "fox": 29, "red_panda": 30, "cattle": 31, "motorcycle": 32, "bear": 33, "antelope": 34, "lizard": 35}, "id2cls": {"0": "__background__", "1": "airplane", "2": "bicycle", "3": "bird", "4": "bus", "5": "car", "6": "dog", "7": "domestic_cat", "8": "elephant", "9": "hamster", "10": "lion", "11": "monkey", "12": "rabbit", "13": "sheep", "14": "snake", "15": "squirrel", "16": "tiger", "17": "train", "18": "turtle", "19": "whale", "20": "zebra", "21": "ball", "22": "frisbee", "23": "sofa", "24": "skateboard", "25": "person", "26": "horse", "27": "watercraft", "28": "giant_panda", "29": "fox", "30": "red_panda", "31": "cattle", "32": "motorcycle", "33": "bear", "34": "antelope", "35": "lizard"}, "cls2split": {"__background__": "base", "airplane": "base", "bicycle": "base", "bird": "base", "bus": "base", "car": "base", "dog": "base", "domestic_cat": "base", "elephant": "base", "hamster": "base", "lion": "base", "monkey": "base", "rabbit": "base", "sheep": "base", "snake": "base", "squirrel": "base", "tiger": "base", "train": "base", "turtle": "base", "whale": "base", "zebra": "base", "ball": "base", "frisbee": "base", "sofa": "base", "skateboard": "base", "person": "base", "horse": "novel", "watercraft": "novel", "giant_panda": "novel", "fox": "novel", "red_panda": "novel", "cattle": "novel", "motorcycle": "novel", "bear": "novel", "antelope": "novel", "lizard": "novel"}, "cls2count": {"__background__": {"total": 0, "train": 0, "test": 0}, "airplane": {"total": 123, "train": 103, "test": 20}, "bicycle": {"total": 168, "train": 114, "test": 54}, "bird": {"total": 126, "train": 105, "test": 21}, "bus": {"total": 31, "train": 30, "test": 1}, "car": {"total": 270, "train": 248, "test": 22}, "dog": {"total": 196, "train": 154, "test": 42}, "domestic_cat": {"total": 46, "train": 43, "test": 3}, "elephant": {"total": 117, "train": 89, "test": 28}, "hamster": {"total": 11, "train": 9, "test": 2}, "lion": {"total": 63, "train": 56, "test": 7}, "monkey": {"total": 128, "train": 101, "test": 27}, "rabbit": {"total": 39, "train": 37, "test": 2}, "sheep": {"total": 81, "train": 79, "test": 2}, "snake": {"total": 23, "train": 21, "test": 2}, "squirrel": {"total": 33, "train": 31, "test": 2}, "tiger": {"total": 16, "train": 14, "test": 2}, "train": {"total": 28, "train": 26, "test": 2}, "turtle": {"total": 28, "train": 27, "test": 1}, "whale": {"total": 69, "train": 45, "test": 24}, "zebra": {"total": 140, "train": 90, "test": 50}, "ball": {"total": 22, "train": 20, "test": 2}, "frisbee": {"total": 5, "train": 3, "test": 2}, "sofa": {"total": 13, "train": 12, "test": 1}, "skateboard": {"total": 7, "train": 6, "test": 1}, "person": {"total": 526, "train": 386, "test": 140}, "horse": {"total": 84, "train": 66, "test": 18}, "watercraft": {"total": 85, "train": 67, "test": 18}, "giant_panda": {"total": 64, "train": 48, "test": 16}, "fox": {"total": 64, "train": 50, "test": 14}, "red_panda": {"total": 53, "train": 40, "test": 13}, "cattle": {"total": 94, "train": 81, "test": 13}, "motorcycle": {"total": 66, "train": 56, "test": 10}, "bear": {"total": 53, "train": 44, "test": 9}, "antelope": {"total": 122, "train": 114, "test": 8}, "lizard": {"total": 23, "train": 15, "test": 8}}}

'''
novel class:

horse                total:84 train:66 test:18
watercraft           total:85 train:67 test:18
giant_panda          total:64 train:48 test:16
fox                  total:64 train:50 test:14
red_panda            total:53 train:40 test:13
cattle               total:94 train:81 test:13
motorcycle           total:66 train:56 test:10
bear                 total:53 train:44 test:9
antelope             total:122 train:114 test:8
lizard               total:23 train:15 test:8


base class:

person               total:526 train:386 test:140
bicycle              total:168 train:114 test:54
zebra                total:140 train:90 test:50
dog                  total:196 train:154 test:42
elephant             total:117 train:89 test:28
monkey               total:128 train:101 test:27
whale                total:69 train:45 test:24
car                  total:270 train:248 test:22
bird                 total:126 train:105 test:21
airplane             total:123 train:103 test:20
lion                 total:63 train:56 test:7
domestic_cat         total:46 train:43 test:3
frisbee              total:5 train:3 test:2
ball                 total:22 train:20 test:2
hamster              total:11 train:9 test:2
squirrel             total:33 train:31 test:2
snake                total:23 train:21 test:2
sheep                total:81 train:79 test:2
tiger                total:16 train:14 test:2
rabbit               total:39 train:37 test:2
train                total:28 train:26 test:2
sofa                 total:13 train:12 test:1
turtle               total:28 train:27 test:1
bus                  total:31 train:30 test:1
skateboard           total:7 train:6 test:1

'''