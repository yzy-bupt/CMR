import numpy as np
import scipy
import scipy.spatial
from scipy.io import loadmat, savemat


def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    sim = (np.dot(label, label.T) > 0).astype(float)
    tindex = np.arange(numcases, dtype=float) + 1
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = min(sim[i].sum(), k)
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i][:k] / tindex[:k]).sum() / num]

    return np.mean(res)


def fx_calc_recall_label(image, text, label, k=10, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    sim = (np.eye(numcases)).astype(float)
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        res += [sim[i][:k].sum()]

    return np.mean(res)

def cal_val_list(text_test, img_test, label_test):
    # txt_to_img
    txt_to_img1 = fx_calc_recall_label(text_test, img_test, label_test, k=1, dist_method='COS')
    print('...Text to Image R@1 = {}'.format(txt_to_img1))
    txt_to_img5 = fx_calc_recall_label(text_test, img_test, label_test, k=5, dist_method='COS')
    print('...Text to Image R@5 = {}'.format(txt_to_img5))
    txt_to_img10 = fx_calc_recall_label(text_test, img_test, label_test, k=10, dist_method='COS')
    print('...Text to Image R@10 = {}'.format(txt_to_img10))

    # img_to_txt
    img_to_txt1 = fx_calc_recall_label(img_test, text_test, label_test, k=1, dist_method='COS')
    print('...Image to Text R@1 = {}'.format(img_to_txt1))
    img_to_txt5 = fx_calc_recall_label(img_test, text_test, label_test, k=5, dist_method='COS')
    print('...Image to Text R@5 = {}'.format(img_to_txt5))
    img_to_txt10 = fx_calc_recall_label(img_test, text_test, label_test, k=10, dist_method='COS')
    print('...Image to Text R@10 = {}'.format(img_to_txt10))

    # average
    print('...Average R@1 = {}'.format(((img_to_txt1 + txt_to_img1) / 2.)))
    print('...Average R@5 = {}'.format(((img_to_txt5 + txt_to_img5) / 2.)))    
    print('...Average R@10 = {}'.format(((img_to_txt10 + txt_to_img10) / 2.)))


if __name__ == '__main__':
    mat_train = loadmat("model/train.mat")
    mat_test = loadmat("model/test.mat")

    img_train = mat_train['img_train']
    text_train = mat_train['txt_train']
    label_train = mat_train['lab_train']

    img_test = mat_test['img_test']
    text_test = mat_test['txt_test']
    label_test = mat_test['lab_test']

    cal_val_list(text_test, img_test, label_test)
    
    img_to_txt = fx_calc_map_label(img_test, text_test, label_test)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(text_test, img_test, label_test)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
    # # Recall@1
    # img_to_txt = fx_calc_recall_label(img_test, text_test, label_test, k=1, dist_method='COS')
    # print('...Image to Text R@1 = {}'.format(img_to_txt))
    # txt_to_img = fx_calc_recall_label(text_test, img_test, label_test, k=1, dist_method='COS')
    # print('...Text to Image R@1 = {}'.format(txt_to_img))
    # print('...Average R@1 = {}'.format(((img_to_txt + txt_to_img) / 2.)))

    # # Recall@5
    # img_to_txt = fx_calc_recall_label(img_test, text_test, label_test, k=5, dist_method='COS')
    # print('...Image to Text R@5 = {}'.format(img_to_txt))
    # txt_to_img = fx_calc_recall_label(text_test, img_test, label_test, k=5, dist_method='COS')
    # print('...Text to Image R@5 = {}'.format(txt_to_img))
    # print('...Average R@5 = {}'.format(((img_to_txt + txt_to_img) / 2.)))

    # # Recall@10
    # img_to_txt = fx_calc_recall_label(img_test, text_test, label_test, k=10, dist_method='COS')
    # print('...Image to Text R@10 = {}'.format(img_to_txt))
    # txt_to_img = fx_calc_recall_label(text_test, img_test, label_test, k=10, dist_method='COS')
    # print('...Text to Image R@10 = {}'.format(txt_to_img))
    # print('...Average R@10 = {}'.format(((img_to_txt + txt_to_img) / 2.)))


