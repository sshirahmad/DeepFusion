import os
import cv2
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import glob
import random
import copy
import pickle
import threading
import queue
import ctypes
import inspect

import utils.train_tools as train_tools
import utils.test_tools as test_tools
import collections

import logging

from skimage import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Inria(object):
    __data_queue = None
    __batch_queue = None
    __read_threads = None
    __batch_threads = None
    __threads_name = None

    def __init__(self, config, for_what, batch_size=1, whether_aug=False):
        """init
        Args:
            batch_size: the size of a batch
            for_what: indicate train or test, must be "train" or "test"
            whether_aug: whether to augument the img
        """
        ##

        assert batch_size > 0
        assert type(whether_aug) == bool

        self.__whether_aug = whether_aug
        self.config = config

        if for_what not in ["train", "predict", "evaluate"]:
            raise ValueError('pls ensure for_what must be "train", "predict" or "evaluate"')
        else:
            self.__for_what = for_what
            if for_what == "train":
                self._inria_infos = self.load_infos(config.train_dir)
            else:
                self._inria_infos = self.load_infos(config.val_dir)

        if for_what in ["train", "predict"]:
            # set batch size and start queue
            self.__batch_size = batch_size
            self.__threads_name = []
            self.__start_read_data(batch_size=self.__batch_size)
            self.__start_batch_data(batch_size=self.__batch_size)
            logger.info('Start loading queue for %s' % (for_what))
        else:
            # for evaluation
            assert whether_aug == False  # when evaluate, aug must be false
            self.__val_infos_que = collections.deque(self._inria_infos)
            logger.info('For evaluation')

    def load_infos(self, path):
        with open(path, "rb") as f:
            inria_infos_all = pickle.load(f)

        return inria_infos_all

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.__threads_name is not None:
                if len(self.__threads_name) > 0:
                    exist_threads = threading.enumerate()
                    exist_threads_name = [exist_thread.name for exist_thread in exist_threads]
                    for thread_name in self.__threads_name:
                        if thread_name not in exist_threads_name:
                            names = str.split(thread_name, "_")
                            if names[0] == "read":
                                restart_thread = threading.Thread(target=self.__send_data)
                                restart_thread.setName(thread_name)
                                restart_thread.setDaemon(True)
                                restart_thread.start()
                                print("restart a down thread")
                                return True
                            elif names[0] == "batch":
                                restart_thread = threading.Thread(target=self.__batch_data,
                                                                  args=(self.__batch_size,))
                                restart_thread.setName(thread_name)
                                restart_thread.setDaemon(True)
                                restart_thread.start()
                                print("restart a down thread")
                                return True

            print(exc_type)
            print(exc_val)
            print(exc_tb)
            exit(1)

    def load_data_eval(self):
        """
        Traversing the test set in sequence
        Return:
            img: one img with shape (h, w, c), if end, None
            bboxes: shape is (n, 4), if end, None
        """

        try:
            data_info = self.__val_infos_que.popleft()

            img, bboxes, img_path = self.__read_one_sample(data_info)
            img, bboxes = train_tools.normalize_data(img, bboxes, self.config.img_size)
            return img, bboxes, img_path
        except IndexError:
            return None, None, None

    def load_batch(self):
        """
        get the batch data
        Return:
            if dataset is for training, return imgs, labels, t_bboxes:
                imgs: a list of img, with the shape (h, w, c)
                labels: a list of labels, with the shape (grid_h, grid_w, pboxes_num, 1)
                        0 is background, 1 is object
                t_bboxes: a list of t_bboxes with the shape (grid_h, grid_w, pboxes_num, 4)
            if dataset is for test, return imgs, corner_bboxes
                imgs: a list of img, with the shape (h, w, c)
                corner_bboxes: a list of bboxes, with the shape (?, 4), encoded by [ymin,
                xin, ymax, xmax]
        """
        batch_data = self.__batch_queue.get()
        if self.__for_what == "train":
            # return imgs, tboxes1, tboxes2, tboxes3, tcls1, tcls2, tcls3, tgrid1, tgrid2, tgrid3
            return batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5], batch_data[
                6], batch_data[7], batch_data[8], batch_data[9]
        else:
            # return imgs , corner_bboxes
            return batch_data[0], batch_data[1]

    def stop_loading(self):
        """
        to kill all threads
        """
        threads = self.__read_threads + self.__batch_threads
        for thread in threads:
            self.__async_raise(thread.ident, SystemExit)
        pass

    def __start_read_data(self, batch_size, thread_num=4, capacity_scalar=2):
        """
        start use multi thread to read data to the queue
        Args:
            thread_num: the number of threads used to read data
            batch_size: the buffer size which used to store the data
        """
        self.__read_threads = []
        maxsize = np.maximum(batch_size * capacity_scalar, 5)
        self.__data_queue = queue.Queue(maxsize=maxsize)

        # start threads
        for i in range(thread_num):
            thread = threading.Thread(target=self.__send_data)
            thread.setDaemon(True)
            thread.setName("read_thread_id%d" % (i))
            self.__threads_name.append("read_thread_id%d" % (i))
            thread.start()
            self.__read_threads.append(thread)

    def __start_batch_data(self, batch_size, thread_num=4, queue_size=5):
        """
        start the threads to batch data into the batch_queue
        Args:
            batch_size: the batch size.
            thread_num: the number of threads
            queue_size: the max batch queue length
        """
        assert queue_size > 0

        self.__batch_threads = []
        self.__batch_queue = queue.Queue(queue_size)

        for i in range(thread_num):
            thread = threading.Thread(target=self.__batch_data, args=(batch_size,))
            thread.setDaemon(True)
            thread.setName("batch_thread_id%d" % (i))
            self.__threads_name.append("batch_thread_id%d" % (i))
            thread.start()
            self.__batch_threads.append(thread)

    def __batch_data(self, batch_size):
        """
        dequeue the data_queue and batch the data into a batch_queue
        """
        first = True
        batch_container_list = []
        while True:
            for i in range(batch_size):
                data_list = self.__data_queue.get()
                if first:
                    # init the batch_list
                    for i in range(len(data_list)):
                        batch_container_list.append([])
                    first = False

                for batch_container, data_item in zip(batch_container_list, data_list):
                    batch_container.append(data_item)

            # put the batch data into batch_queue
            self.__batch_queue.put(copy.deepcopy(batch_container_list))

            for batch_container in batch_container_list:
                batch_container.clear()

    def __send_data(self):
        """
         a single thread which send a data to the data queue
        """
        anchors = self.config.anchors / self.config.img_size
        while True:
            data_info = random.sample(self._inria_infos, 1)[0]
            img, bboxes, _ = self.__read_one_sample(data_info)

            # resize img and normalize img and bboxes
            if self.__for_what == "train":
                if self.__whether_aug:
                    img, bboxes = train_tools.img_aug(img, bboxes)
                    if len(bboxes) == 0:
                        # sometimes aug func will corp no person
                        # logger.warning("No person img, abandoned...")
                        continue

                # for box in bboxes:
                #     cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 2)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)

                img, bboxes = train_tools.normalize_data(img, bboxes, self.config.img_size)
                tbox, tcls, tgrid = \
                    train_tools.ground_truth_one_img(gt_boxes=bboxes,
                                                     anchors=anchors,
                                                     grid_size=self.config.grid_size,
                                                     top_k=self.config.top_k,
                                                     obj_threshold=self.config.obj_threshold,
                                                     surounding_size=self.config.surounding_size,
                                                     )
                tbox1 = tbox[0]
                tbox2 = tbox[1]
                tbox3 = tbox[2]
                tcls1 = tcls[0]
                tcls2 = tcls[1]
                tcls3 = tcls[2]
                tgrid1 = tgrid[0]
                tgrid2 = tgrid[1]
                tgrid3 = tgrid[2]

                # put data into data queue
                self.__data_queue.put([img, tbox1, tbox2, tbox3, tcls1, tcls2, tcls3, tgrid1, tgrid2, tgrid3])
            else:
                if self.__whether_aug:
                    img, bboxes = test_tools.img_aug(img, bboxes)
                    if len(bboxes) == 0:
                        # sometimes aug func will corp no person
                        # logger.warning("No person img, abandoned...")
                        continue
                img, bboxes = train_tools.normalize_data(img, bboxes, self.config.img_size)
                self.__data_queue.put([img, bboxes])

    def __read_one_sample(self, data_info):
        """
        read one sample
        Args:
            img_name: img name, like "/usr/img/image001.jpg"
            label_name: the label file responding the img_name, like "/usr/label/image001.xml"
        Return:
            An ndarray with the shape [img_h, img_w, img_c], bgr format
            An ndarray with the shape [?,4], which means [ymin, xmin, ymax, xmax]
        """

        img_name = data_info['image_path']
        img = cv2.imread(img_name)
        bboxes = data_info['gt_boxes']

        return img, bboxes, img_name

    def __async_raise(self, tid, exctype):
        """
        raises the exception, performs cleanup if needed
        """
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")


if __name__ == '__main__':
    dt = provider(batch_size=1, for_what="train", whether_aug=True)

    for step in range(100):
        imgs, labels, t_bboxes = dt.load_batch()

        pass
        # do sth ##

    # imgs_name = glob.glob(os.path.join('F:\my_project\pedestrian-detection-in-hazy-weather\dataset\inria_person\PICTURES_LABELS_TEST\PICTURES', '*png'))
    #
    # for img_name in imgs_name:
    #     name = os.path.basename(img_name).split('.')[0]
    #     a = cv2.imread(img_name)
    #     cv2.imwrite(os.path.join('F:\my_project\pedestrian-detection-in-hazy-weather\dataset\inria_person\PICTURES_LABELS_TEST\\temp', name+'.jpg'), a)
    #     print(img_name)
