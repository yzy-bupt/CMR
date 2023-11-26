# import os
# import os.path as osp
# from pycocotools.coco import COCO
# import pprint

# """process class order
# Record the mapping between tightened/discretized 0-base class ID,
# original class ID and class name in `class-name.COCO.txt`,
# with format `<original ID> <class name>`.

# The class order is consistent to the ascending order of the original IDs.
# """

# COCO_P = "./"
# ANNO_P = osp.join(COCO_P, "annotations")
# SPLIT = ["val", "train"]

# for _split in SPLIT:
#     print("---", _split, "---")
#     anno_file = osp.join(ANNO_P, "instances_{}2014.json".format(_split))
#     coco = COCO(anno_file)
#     cats = coco.loadCats(coco.getCatIds())
#     # print(cats[0])
#     cat_list = sorted([(c["id"], c["name"]) for c in cats],
#         key=lambda t: t[0])  # 保证升序
#     # pprint.pprint(cat_list)
#     with open(osp.join(COCO_P, "class-name.COCO.txt"), "w") as f:
#         for old_id, c in cat_list:
#             cn = c.replace(" ", "_")
#             # format: <original ID> <class name>
#             f.write("{} {}\n".format(old_id, cn))

#     break  # 只用 val set




# import os
# import os.path as osp

# """discretization of the original file ID
# Map the file ID to sequential {0, 1, ..., n},
# and record this mapping in `id-map.txt`,
# with format `<new id> <original id> <image file name>`.

# Note that the new ids are 0-base.
# """

# COCO_P = "datapath_to_coco"
# TRAIN_P = osp.join(COCO_P, "train2014")
# VAL_P = osp.join(COCO_P, "val2014")

# file_list = [f for f in os.listdir(TRAIN_P) if (".jpg" in f)]
# file_list.extend([f for f in os.listdir(VAL_P) if (".jpg" in f)])
# print("#data:", len(file_list))  # 12,3287

# id_key = lambda x: int(x.split(".jpg")[0].split("_")[-1])
# file_list = sorted(file_list, key=id_key)  # ascending of image ID
# # print(file_list[:15])

# with open(osp.join(COCO_P, "id-map.COCO.txt"), "w") as f:
#     # format: <original id> <image file name>
#     for f_name in file_list:
#         _original_id = id_key(f_name)
#         f.write("{} {}\n".format(_original_id, f_name))
# print("DONE")


import os
import os.path as osp
import numpy as np
import scipy.io as sio
from pycocotools.coco import COCO


"""process labels
Data in both train & val set will be all put together,
with data order determined by `id-map.COCO.txt`
and catetory order by `class-name.COCO.txt`.
"""


COCO_P = "./"
ANNO_P = osp.join(COCO_P, "annotations")
SPLIT = ["val", "train"]


id_map_cls = {}
with open(osp.join(COCO_P, "class-name.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        _old_id, _ = line.strip().split()
        id_map_cls[int(_old_id)] = _new_id
N_CLASS = len(id_map_cls)
print("#class:", N_CLASS)  # 80

id_map_data = {}
img_name = []
oid = []
with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
    for _new_id, line in enumerate(f):
        line = line.strip()
        _old_id, _img_name = line.strip().split()
        img_name.append(_img_name)
        oid.append(int(_old_id))
        id_map_data[int(_old_id)] = _new_id
N_DATA = len(id_map_data)
print("#data:", N_DATA)  # 123,287


# labels = np.zeros([N_DATA, N_CLASS], dtype=np.uint8)
# delete = []
# for _split in SPLIT:
#     print("---", _split, "---")
#     anno_file = osp.join(ANNO_P, "instances_{}2014.json".format(_split))
#     coco = COCO(anno_file)
#     id_list = coco.getImgIds()
#     for _old_id in id_list:
#         _new_id = id_map_data[_old_id]
#         _annIds = coco.getAnnIds(imgIds=_old_id)
#         _anns = coco.loadAnns(_annIds)
#         if len(_anns) == 0:
#             delete.append(_new_id)
#         for _a in _anns:
#             _cid = id_map_cls[_a["category_id"]]
#             labels[_new_id][_cid] = 1

# labels = np.delete(labels, delete, 0)
# img_name = np.delete(np.array(img_name), delete, 0)
# oid = np.delete(np.array(oid), delete, 0)


# print("labels:", labels.shape, labels.sum())  # (123287, 80) 357627
# sio.savemat(osp.join(COCO_P, "labels.COCO.mat"), {"labels": labels, "img_name": img_name, 'id':oid}, do_compression=True)

from scipy.io import loadmat, savemat

c_id = loadmat("labels.COCO.mat")['id']
c_img = loadmat("labels.COCO.mat")['img_name']

for _split in SPLIT:
    print("---", _split, "---")
    anno_file = osp.join(ANNO_P, "instances_{}2014.json".format(_split))
    caps_file = osp.join(ANNO_P, "captions_{}2014.json".format(_split))
    coco = COCO(anno_file)
    coco_caps = COCO(caps_file)
    id_list = coco.getImgIds()
    skip = 0
    for _old_id in id_list:
        if _old_id not in c_id:
            skip += 1
            continue
        _new_id = id_map_data[_old_id]
        _annIds = coco_caps.getAnnIds(imgIds=_old_id)
        _anns = coco_caps.loadAnns(_annIds)
        # print(len(anns))
        # pprint.pprint(anns)
        sentences = [_a["caption"] for _a in _anns]





# import codecs
# import multiprocessing
# import os
# import os.path as osp
# import pprint
# import time
# import threading
# from pycocotools.coco import COCO
# import numpy as np
# import scipy.io as sio


# """Multi-Threading version of make.text.py"""


# # COCO
# COCO_P = "./"
# ANNO_P = osp.join(COCO_P, "annotations")
# SPLIT = ["val", "train"]


# id_map_data = {}
# with open(osp.join(COCO_P, "id-map.COCO.txt"), "r") as f:
#     for _new_id, line in enumerate(f):
#         _old_id, _ = line.strip().split()
#         id_map_data[int(_old_id)] = _new_id
# N_DATA = len(id_map_data)
# print("#data:", N_DATA)  # 123,287


# # multi-threading vars
# N_THREAD = max(4, multiprocessing.cpu_count() - 2)
# results, mutex_res = [], threading.Lock()
# meta_index, mutex_mid = 0, threading.Lock()
# mutex_d2v = threading.Lock()


# def run(tid, id_list):
#     global results, meta_index, id_map_data, model
#     n = len(id_list)
#     while True:
#         mutex_mid.acquire()
#         meta_idx = meta_index
#         meta_index += 1
#         mutex_mid.release()
#         if meta_idx >= n:
#             break

#         _old_id = id_list[meta_idx]
#         _new_id = id_map_data[_old_id]
#         _annIds = coco_caps.getAnnIds(imgIds=_old_id)
#         _anns = coco_caps.loadAnns(_annIds)
#         # print(len(anns))
#         # pprint.pprint(anns)
#         sentences = [_a["caption"] for _a in _anns]
#         # pprint.pprint(sentences)
#         doc = prep_text(tid, sentences)
#         # pprint.pprint(doc)
#         mutex_d2v.acquire()
#         model.random.seed(D2V_SEED)  # to keep it consistent
#         vec = model.infer_vector(doc)
#         mutex_d2v.release()
#         # print(vec.shape)
#         mutex_res.acquire()
#         results.append((_new_id, vec[np.newaxis, :]))
#         mutex_res.release()
#         if meta_idx % 1000 == 0:
#             print(meta_idx, ',', time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))

#     # remove the intermedia output files (when using Stanford CoreNLP)
#     for f in ["input.{}.txt".format(tid), "input.{}.txt.conll".format(tid)]:
#         if osp.exists(f):
#             os.remove(f)


# for _split in SPLIT:
#     print("---", _split, "---")
#     tic = time.time()
#     anno_file = osp.join(ANNO_P, "instances_{}2014.json".format(_split))
#     caps_file = osp.join(ANNO_P, "captions_{}2014.json".format(_split))
#     coco = COCO(anno_file)
#     coco_caps = COCO(caps_file)
#     id_list = coco.getImgIds()

#     meta_index = 0  # reset for each split
#     t_list = []
#     for tid in xrange(N_THREAD):
#         t = threading.Thread(target=run, args=(tid, id_list))
#         t_list.append(t)
#         t.start()

#     for t in t_list:
#         t.join()

#     del t_list


# assert len(results) == N_DATA
# texts = sorted(results, key=lambda t: t[0])  # ascending by new ID
# for i in xrange(100):#N_DATA):
#     assert texts[i][0] == i, "* order error"
# texts = [t[1] for t in texts]
# texts = np.vstack(texts).astype(np.float32)
# assert texts.shape[0] == N_DATA
# print("texts:", texts.shape, texts.dtype)  # (123287, 300) dtype('<f4')
# print(texts.mean(), texts.min(), texts.max())  # -0.0047004167, -0.7569326, 0.804541
# sio.savemat(osp.join(COCO_P, "texts.COCO.d2v-{}d.mat".format(texts.shape[1])), {"texts": texts})
