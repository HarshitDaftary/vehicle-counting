from random import shuffle
from utils.getter import *
import argparse
import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import draw_boxes_v2, write_to_video
from utils.postprocess import postprocessing
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation
from augmentations.transforms import MEAN, STD
from models.deepsort.deep_sort import DeepSort
from .count import check_bbox_intersect_polygon, counting_moi, load_zone_anno

parser = argparse.ArgumentParser(description='Perfom Objet Detection')
parser.add_argument('--weight', type=str, default = None,help='version of EfficentDet')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--min_conf', type=float, default= 0.1, help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
parser.add_argument('--tta', action='store_true', help='whether to use test time augmentation')
parser.add_argument('--tta_ensemble_mode', type=str, default='wbf', help='tta ensemble mode')
parser.add_argument('--tta_conf_threshold', type=float, default=0.01, help='tta confidence score threshold')
parser.add_argument('--tta_iou_threshold', type=float, default=0.9, help='tta iou threshold')
parser.add_argument('--debug', action='store_true', help='save detection at')


class VideoSet:
    def __init__(self, config, input_path):
        self.input_path = input_path # path to video file
        self.image_size = config.image_size
        self.transforms = A.Compose([
            get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0)
        ])

        self.get_video_info()

    def get_video_info(self):
        self.stream = cv2.VideoCapture(self.input_path)
        self.current_frame_id = 0
        self.video_info = {}

        if self.stream.isOpened(): 
            # get self.stream property 
            self.WIDTH  = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            self.HEIGHT = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
            self.FPS = int(self.stream.get(cv2.CAP_PROP_FPS))
            self.NUM_FRAMES = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info = {
                'name': os.path.basename(self.input_path),
                'width': self.WIDTH,
                'height': self.HEIGHT,
                'fps': self.FPS,
                'num_frames': self.NUM_FRAMES
            }
        else:
            raise f"Cannot read video {os.path.basename(self.input_path)}"

    def __getitem__(self, idx):
        success, ori_frame = self.stream.read()
        if not success:
            print(f"Cannot read frame {self.current_frame_id} from {self.video_info['name']}")
            return None
        else:
            self.current_frame_id = idx+1
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame /= 255.0
        if self.transforms is not None:
            inputs = self.transforms(image=frame)['image']

        image_w, image_h = self.image_size
        ori_height, ori_width, _ = ori_frame.shape

        return {
            'img': inputs,
            'frame': self.current_frame_id,
            'ori_img': ori_frame,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        imgs = torch.stack([s['img'] for s in batch])   
        ori_imgs = [s['ori_img'] for s in batch]
        frames = [s['frame'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)   

        return {
            'imgs': imgs,
            'frames': frames,
            'ori_imgs': ori_imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
        return self.NUM_FRAMES

    def __str__(self):
        s2 = f"Number of frames: {self.NUM_FRAMES}"
        return s2

class VideoLoader(DataLoader):
    def __init__(self, config, video_path):
        self.video_path = video_path
        dataset = VideoSet(config, video_path)
        self.video_info = dataset.video_info
       
        super(VideoLoader, self).__init__(
            dataset,
            batch_size= 1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            collate_fn= dataset.collate_fn)
        

class VideoWriter:
    def __init__(self, video_info, saved_path, obj_list):
        self.video_info = video_info
        self.saved_path = saved_path
        self.obj_list = obj_list

        video_name = self.video_info['name']
        outpath =os.path.join(self.saved_path, video_name)
        FPS = self.video_info['fps']
        WIDTH = self.video_info['width']
        HEIGHT = self.video_info['height']
        NUM_FRAMES = self.video_info['num_frames']
        self.outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            FPS, 
            (WIDTH, HEIGHT))

    def write(self, img, boxes, labels, scores=None, tracks=None):
        write_to_video(
            img, boxes, labels, 
            scores = scores,
            tracks=tracks, 
            imshow=False, 
            outvid = self.outvid, 
            obj_list=self.obj_list)
        

class VideoDetect:
    def __init__(self, args, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   
        self.debug = args.debug
        self.config = config
        self.min_iou = args.min_iou
        self.min_conf = args.min_conf
        self.max_dets=config.max_post_nms
        self.keep_ratio=config.keep_ratio
        self.fusion_mode=config.fusion_mode

        if args.tta:
            self.tta = TTA(
                min_conf=args.tta_conf_threshold, 
                min_iou=args.tta_iou_threshold, 
                postprocess_mode=args.tta_ensemble_mode)
        else:
            self.tta = None

        if args.weight is not None:
            self.class_names, num_classes = get_class_names(args.weight)
        self.class_names.insert(0, 'Background')

        net = get_model(
            args, config,
            num_classes=num_classes)

        self.num_classes = num_classes
        self.model = Detector(model = net, device = self.device)
        self.model.eval()

        if args.weight is not None:                
            load_checkpoint(self.model, args.weight)

    def run(self, batch):
        with torch.no_grad():
            boxes_result = []
            labels_result = []
            scores_result = []
                
            if self.tta is not None:
                preds = self.tta.make_tta_predictions(self.model, batch)
            else:
                preds = self.model.inference_step(batch)

            for idx, outputs in enumerate(preds):
                img_w = batch['image_ws'][idx]
                img_h = batch['image_hs'][idx]
                img_ori_ws = batch['image_ori_ws'][idx]
                img_ori_hs = batch['image_ori_hs'][idx]
                
                outputs = postprocessing(
                    outputs, 
                    current_img_size=[img_w, img_h],
                    ori_img_size=[img_ori_ws, img_ori_hs],
                    min_iou=self.min_iou,
                    min_conf=self.min_conf,
                    max_dets=self.max_dets,
                    keep_ratio=self.keep_ratio,
                    output_format='xywh',
                    mode=self.fusion_mode)

                boxes = outputs['bboxes'] 
                labels = outputs['classes']  
                scores = outputs['scores']

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        return {
            "boxes": boxes_result, 
            "labels": labels_result,
            "scores": scores_result }

class VideoTracker:
    def __init__(self, num_classes, cam_config, video_info, deepsort_chepoint):
        tracking_config = cam_config["tracking_config"]
        self.num_classes = num_classes 
        self.video_info = video_info
        self.NUM_CLASSES = video_info['num_frames']

        ## Build up a tracker for each class
        self.deepsort = [self.build_tracker(deepsort_chepoint, tracking_config) for i in range(num_classes)]

    def build_tracker(self, checkpoint, cam_cfg):
        return DeepSort(
                checkpoint, 
                max_dist=cam_cfg['MAX_DIST'],
                min_confidence=cam_cfg['MIN_CONFIDENCE'], 
                nms_max_overlap=cam_cfg['NMS_MAX_OVERLAP'],
                max_iou_distance=cam_cfg['MAX_IOU_DISTANCE'], 
                max_age=cam_cfg['MAX_AGE'],
                n_init=cam_cfg['N_INIT'],
                nn_budget=cam_cfg['NN_BUDGET'],
                use_cuda=1)

    def run(self, image, boxes, labels, scores, frame_id):
        # Dict to save object's tracks per class
        # boxes: xywh
        self.obj_track = [{} for i in range(self.num_classes)]

        ## Draw polygons to frame
         
        # cv2.putText(im_moi,str(frame_id), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)

        bbox_xyxy = boxes.copy()
        bbox_xyxy[:, 2] += bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] += bbox_xyxy[:, 1]

        # class index starts from 1
        labels = labels - 1

        result_dict = {
            'tracks': [],
            'boxes': [],
            'labels': [],
            'scores': []
        }

        for i in range(self.num_classes):
            mask = (labels == i)     
            bbox_xyxy_ = bbox_xyxy[mask]
            scores_ = scores[mask]
            labels_ = labels[mask]

            if len(labels_) > 0:

                # output: x1,y1,x2,y2,track_id, track_feat, score
                outputs = self.deepsort[i].update(bbox_xyxy_, scores_, image)
                
                for obj in outputs:
                    box = obj[:4]
                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]
                    result_dict['tracks'].append(obj[4])
                    result_dict['boxes'].append(box)
                    result_dict['labels'].append(i)
                    # result_dict['scores'].append(obj[6])

        result_dict['boxes'] = np.array(result_dict['boxes'])
        
        return result_dict
                
class VideoCounting:
    def __init__(self, class_names, zone_path, minimum_length=4) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.track_dict_ls = [{} * self.num_classes]
        self.minimum_length = minimum_length
        self.polygons_first, self.polygons_last, self.paths, self.polygons = load_zone_anno(zone_path)
    

    def run(self, frames, tracks, labels, boxes):
        for (frame_id, track_id, label_id, box) in zip(frames, tracks, labels, boxes):
            for _, polygon in self.polygons.items():
                if check_bbox_intersect_polygon(polygon, box):
                    if track_id not in self.track_dict_ls[label_id].keys():
                        # find obj id which intersect with polygons
                        self.track_dict_ls[label_id][track_id] = []
                    self.track_dict_ls[label_id][track_id].append((box, frame_id))

        # Remove tracks that have short length
        for label_id in range(self.num_classes):
            for track_id in self.track_dict_ls[label_id].keys():
                if len(self.track_dict_ls[label_id][track_id]) < self.minimum_length:
                    del self.track_dict_ls[label_id][track_id]
        
        vehicle_tracks = [[]*self.num_classes]
        for label_id in range(self.num_classes):
            track_dict = self.track_dict_ls[label_id]
            for tracker_id, tracker_list in track_dict.items():
                if len(tracker_list) > 1:
                    first = tracker_list[0]
                    last = tracker_list[-1]
                    # Get center point of first box and last box
                    first_point = ((first[2] + first[0])/2,
                                (first[3] + first[1])/2)
                    last_point = ((last[2] + last[0])/2, (last[3] + last[1])/2)
                    vehicle_tracks[label_id].append(
                        (first_point, last_point, last[4], tracker_id, first[:4], last[:4]))

        vehicles_moi_detections_ls = [[] * self.num_classes]
        vehicles_moi_detections_dict = [{} * self.num_classes]
        for label_id in range(self.num_classes):
            vehicles_moi_detections_ls[label_id], vehicles_moi_detections_dict[label_id] = \
                counting_moi((self.polygons_first, self.polygons_last),
                            self.paths, vehicle_tracks[label_id], label_id)
        
        print(vehicles_moi_detections_ls)
        print(vehicles_moi_detections_dict)

    


class Pipeline:
    def __init__(self, args, config, cam_config):
        self.detector = VideoDetect(args, config)
        self.class_names = self.detector.class_names
        self.video_path = args.input_path
        self.saved_path = args.output_path
        self.cam_config = cam_config
        self.zone_path = cam_config.zone_path

        if os.path.isdir(self.video_path):
            video_names = sorted(os.listdir(self.video_path))
            self.all_video_paths = [os.path.join(self.video_path, i) for i in video_names]
        else:
            self.all_video_paths = [self.video_path]

    def get_cam_name(self, path):
        filename = os.path.basename(path)
        cam_name = filename[:-4]
        return cam_name

    def run(self):
        for video_path in self.all_video_paths:
            cam_name = self.get_cam_name(video_path)
            videoloader = VideoLoader(config, video_path)
            self.tracker = VideoTracker(
                len(self.class_names),
                self.cam_config.cam[cam_name],
                videoloader.dataset.video_info,
                deepsort_chepoint=self.cam_config.checkpoint)

            videowriter = VideoWriter(
                videoloader.dataset.video_info,
                saved_path=self.saved_path,
                obj_list=self.class_names)
            
            videocounter = VideoCounting(
                class_names = self.class_names,
                zone_path = os.path.join(self.zone_path, cam_name+".json"))

            obj_dict = {
                'frames': [],
                'tracks': [],
                'labels': [],
                'boxes': []
            }
            for batch in tqdm(videoloader):
                preds = self.detector.run(batch)
                ori_imgs = batch['ori_imgs']

                for i in range(len(ori_imgs)):
                    boxes = preds['boxes'][i]
                    labels = preds['labels'][i]
                    scores = preds['scores'][i]
                    frame_id = preds['frames'][i]

                    ori_img = ori_imgs[i]
                    
                    track_result = self.tracker.run(ori_img, boxes, labels, scores, 0)

                    videowriter.write(
                        ori_img,
                        boxes = track_result['boxes'],
                        labels = track_result['labels'],
                        tracks = track_result['tracks'])

                    for j in range(len(track_result['boxes'])):
                        obj_dict['frames'].append(frame_id)
                        obj_dict['tracks'].append(track_result['tracks'][j])
                        obj_dict['labels'].append(track_result['labels'][j])
                        obj_dict['boxes'].append(track_result['boxes'][j])

            videocounter.run(
                frames = obj_dict['frames'],
                tracks = obj_dict['tracks'], 
                labels = obj_dict['labels'],
                boxes = obj_dict['boxes'])
            


def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices_info = get_devices_info(args.gpus)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)


    ## Print info
    print(config)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    cam_config = Config(os.path.join('configs', 'cam_configs.yaml'))             
    pipeline = Pipeline(args, config, cam_config)
    pipeline.run()

if __name__ == '__main__':
    args = parser.parse_args() 

    ignore_keys = [
        'min_iou_val',
        'min_conf_val',
        'tta',
        'gpu_devices',
        'tta_ensemble_mode',
        'tta_conf_threshold',
        'tta_iou_threshold',
    ]

    config = get_config(args.weight, ignore_keys)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")    

    main(args, config)
    