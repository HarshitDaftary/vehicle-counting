from modules.track import VideoTracker, VideoCounting
from modules.datasets import VideoWriter, VideoLoader
from modules.detect import ImageDetect

from utilities.getter import *
import argparse
import os
import numpy as np
import pika
import json

parser = argparse.ArgumentParser(description='Perform Counting vehicles')
parser.add_argument('--weight', type=str, default = None,help='checkpoint of yolo')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--debug', action='store_true', help='save detection at')
parser.add_argument('--mapping', default=None, help='Specify a class mapping if using pretrained')

class DeepshortWorker:
    def __init__(self, args, config, cam_config):
        detector = ImageDetect(args, config)
        self.class_names = detector.class_names
        
        video_path = args.input_path
        
        self.cam_config = cam_config
        cam_name = self.get_cam_name(video_path)
        videoloader = VideoLoader(config, video_path)

        self.tracker = VideoTracker(
                len(self.class_names),
                self.cam_config.cam[cam_name],
                videoloader.dataset.video_info,
                deepsort_chepoint=self.cam_config.checkpoint)

    def get_cam_name(self, path):
        filename = os.path.basename(path)
        cam_name = filename[:-4]
        return cam_name

    def start_worker(self):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()

        channel.queue_declare(queue='task_queue', durable=True)
        print(' [*] Waiting for messages. To exit press CTRL+C')

        def callback(ch, method, properties, body):
            import pickle

            file_name = body.decode()
            print(f"file name {file_name}")
            
            try:
                with open(file_name, 'rb') as f:
                    params = pickle.load(f)

                # print(f"Data type is :: {type(params)}")
                ori_imgs = params["ori_imgs"]
                preds = params["preds"] 
                batch = params["batch"]
                self.track_in_bg(ori_imgs, preds, batch)
            except Exception as e:
                print(e)

            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue='task_queue', on_message_callback=callback)
        channel.start_consuming()


    def track_in_bg(self, ori_imgs, preds, batch):
        import datetime
        d1 = datetime.datetime.now()

        print("started tracking")
        obj_dict = {
            'frames': [],
            'tracks': [],
            'labels': [],
            'boxes': []
        }
        for i in range(len(ori_imgs)):
            boxes = preds['boxes'][i]
            labels = preds['labels'][i]
            scores = preds['scores'][i]
            frame_id = batch['frames'][i]

            ori_img = ori_imgs[i]

            if len(boxes) == 0:
                continue
            track_result = self.tracker.run(ori_img, boxes, labels, scores)
            

            # box_xywh = change_box_order(track_result['boxes'],'xyxy2xywh');
            # videowriter.write(
            #     ori_img,
            #     boxes = track_result['boxes'],
            #     labels = box_xywh,
            #     tracks = track_result['tracks'])
            
            for j in range(len(track_result['boxes'])):
                obj_dict['frames'].append(frame_id)
                obj_dict['tracks'].append(track_result['tracks'][j])
                obj_dict['labels'].append(track_result['labels'][j])
                obj_dict['boxes'].append(track_result['boxes'][j])

        
        d2 = datetime.datetime.now()
        diff = d2 - d1
        print(f"Time taken is :: {diff.total_seconds()}")


def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices_info = {} #get_devices_info(args.gpus)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    ## Print info
    print(config)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    cam_config = Config(os.path.join('configs', 'cam_configs.yaml'))             
    # pipeline = CountingPipeline(args, config, cam_config)
    worker = DeepshortWorker(args, config, cam_config)
    worker.start_worker()
    # pipeline.run()

if __name__ == '__main__':
    args = parser.parse_args() 
    config = Config(os.path.join('configs','configs.yaml'))

    # If you not use any weight and want to use pretrained on COCO, uncomment these lines
    MAPPING_DICT = {
        0: "bike",
        1: "bike",
        2: "car",
        3: "bike",
        5: "truck",
        7: 3
    }
    args.mapping_dict = MAPPING_DICT

    main(args, config)
