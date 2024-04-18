import argparse
import torch
import os


class ArgumentsBase(object):
    def __init__(self):
        self.name = argparse.Namespace()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.current_path = os.path.join(os.getcwd(), 'perspective_transform')
        self.base_path = os.getcwd()
        self.isTrain = False

    def main_args_initialization(self):
        self.parser.add_argument("--ad", action="store_true",
                                 help="Show advertisement of venue")
        self.parser.add_argument("--detector", type=str,
                                 default="yolo", help="Path to the model")
        self.parser.add_argument("--model_path", type=str,
                                 default="model/yolov8x.pt", help="Path to the model")
        self.parser.add_argument("--video", type=str,
                                 default="videos/soccer_possession.mp4", help="Path to the input video", )
        self.parser.add_argument("--bev", action="store_true",
                                 help="Show bird eye view")
        self.parser.add_argument("--nospeed", action="store_false",
                                 help="Show speed")
        self.parser.add_argument('--output', type=str, default='inference/output.mp4',
                                 help='output folder')
        self.parser.add_argument('--output_fps', type=int, default=25,
                                 help='Output Video FPS')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    def _parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt


class Arguments(ArgumentsBase):
    def __init__(self):
        super().__init__()
        self.main_args_initialization()

    def parse(self):
        opt = self._parse()

        return opt
