import numpy as np
import cv2
import torch
cimport numpy as np
cimport cython
cimport libcpp
cimport libcpp.pair
cimport libcpp.queue
from libcpp.pair cimport *
from libcpp.queue cimport *
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=2] _pa(np.ndarray[np.uint8_t, ndim=3] kernels,
                                        np.ndarray[np.float32_t, ndim=3] emb,
                                        np.ndarray[np.int32_t, ndim=2] label,
                                        np.ndarray[np.int32_t, ndim=2] cc,
                                        int kernel_num,
                                        int label_num,
                                        float min_area=0):
    cdef int H, W
    H = label.shape[0]
    W = label.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] pred = np.zeros((H, W), dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] area = np.full((label_num,), -1, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] flag = np.zeros((label_num,), dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=3] inds = np.zeros((label_num, H, W), dtype=np.bool)
    cdef np.ndarray[np.int32_t, ndim=2] p = np.zeros((label_num, 2), dtype=np.int32)
    cdef np.float32_t max_rate = 1024
    cdef int i, j, tmp
    
#     print(H*W) -> 376832
    for i in prange(H, nogil=True):
        for j in range(W):
            tmp = label[i, j]
            if tmp == 0:
                continue
            else:
                inds[tmp, i, j] = True
                if area[tmp] == -1:
                    area[tmp] = 1
                else:
                    area[tmp] += 1
                if p[tmp, 0] == 0 and p[tmp, 1] == 0:
                    p[tmp, 0] = i
                    p[tmp, 1] = j
    
    for i in range(1, label_num):
        if area[i] < min_area:
            label[inds[i]] = 0
            continue
        for j in range(1, i):
            if area[j] < min_area:
                continue
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:
                continue
            rate = area[i] / area[j]
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, inds[i]], axis=1)
                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)
            
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] nxt_que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef np.int16_t*dx = [-1, 1, 0, 0]
    cdef np.int16_t*dy = [0, 0, -1, 1]
    cdef np.int16_t tmpx, tmpy
    
    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]
    
    cdef libcpp.pair.pair[np.int16_t, np.int16_t] cur
    cdef int cur_label
    for kernel_idx in range(kernel_num - 2, -1, -1):
        while not que.empty():
            cur = que.front()
            que.pop()
            cur_label = pred[cur.first, cur.second]

            is_edge = True
            for j in range(4):
                tmpx = cur.first + dx[j]
                tmpy = cur.second + dy[j]
                if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                    continue
                if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                if flag[cur_label] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[cur_label]) > 3:
                    continue

                que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
                is_edge = False
            if is_edge:
                nxt_que.push(cur)

        que, nxt_que = nxt_que, que
    return pred

def pa(kernels, emb, min_area=0):
    kernel_num = kernels.shape[0]
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4)
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4)
    return _pa(kernels[:-1], emb, label, cc, kernel_num, label_num, min_area)