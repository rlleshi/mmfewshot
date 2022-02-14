# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from mmfewshot.utils import local_numpy_seed
from .base import BaseFewShotDataset


@DATASETS.register_module()
class ZIMDataset(BaseFewShotDataset):
    """ZIM dataset for few shot classification.

    Args:
        classes_id_seed (int | None): A random seed to shuffle order
            of classes. If seed is None, the classes will be arranged in
            alphabetical order. Default: None.
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
    """

    ALL_CLASSES = [
        # * train
        'english valse chasse bg',
        'english valse chasse fw',
        'english valse natural turn bw 1 - 3',
        'english valse natural turn fw 1 - 3',
        'english valse natural turn starting bw 1 - 6',
        'english valse natural turn starting fw 1 - 6',
        # 'tango 2 walking steps bw',
        # 'tango 2 walking steps fw',
        # 'tango rock turn starting bw',
        # 'tango rock turn starting fw',
        #
        # * not groupped
        #
        # 'chasse fw bouncing',
        # 'tango fw accelerating',
        # 'tango fw curved',
        # 'chasse fw not bouncing',
        # 'tango fw not accelerating',
        # 'tango fw not curved',
        #
        # * groupped

        # * valse
        # 'chasse_bw_bounc',
        # 'chasse_bw_bounc_not',
        # 'chasse_fw_bounc',
        # 'chasse_fw_bounc_not',

        'nt_bw_1-3_rising',
        'nt_bw_1-3_rising_not',
        'nt_fw_1-3_rising',
        'nt_fw_1-3_rising_not'

        # * tango
        # 'tango fw accelerating curved',
        # 'tango fw not accelerating curved',
        # 'tango fw accelerating not curved',
        # 'tango fw not accelerating not curved'
        ]


    def __init__(self,
                 classes_id_seed: int = None,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 *args,
                 **kwargs) -> None:
        self.classes_id_seed = classes_id_seed
        self.num_all_classes = len(self.ALL_CLASSES)

        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        super().__init__(*args, **kwargs)


    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.
           6 train classes / 3 val classes / 3 test classes.

        Args:
            classes (Sequence[str] | str | None): Three types of input
                will correspond to different processing logics:

                - If `classes` is a tuple or list, it will override the
                  CLASSES predefined in the dataset.
                - If `classes` is None, we directly use pre-defined CLASSES
                  will be used by the dataset.
                - If `classes` is a string, it is the path of a classes file
                  that contains the name of all classes. Each line of the file
                  contains a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            classes_ids = list(range(self.num_all_classes))
            if self.classes_id_seed is not None:
                with local_numpy_seed(self.classes_id_seed):
                    np.random.shuffle(classes_ids)
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if i <= 5
                    ]
                elif subset_ == 'val':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if (i > 7) and (i <= 9)
                    ]
                elif subset_ == 'test':
                    class_names += [
                        self.ALL_CLASSES[i] for i in classes_ids if (i > 5) and (i <=7)
                    ]
                else:
                    raise ValueError(f'invalid subset {subset_} only support '
                                     f'train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names


    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []

        print(self.ann_file)
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
