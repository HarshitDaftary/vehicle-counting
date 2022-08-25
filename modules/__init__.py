import os
from tqdm import tqdm
from .detect import ImageDetect
from .track import VideoTracker, VideoCounting
from .datasets import VideoWriter, VideoLoader
import numpy as np
from threading import Thread
import pika
import sys
import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CountingPipeline:
    def __init__(self, args, config, cam_config):
        self.detector = ImageDetect(args, config)
        self.class_names = self.detector.class_names
        self.video_path = args.input_path
        self.saved_path = args.output_path
        self.cam_config = cam_config
        self.zone_path = cam_config.zone_path
        self.config = config

        if os.path.isdir(self.video_path):
            video_names = sorted(os.listdir(self.video_path))
            self.all_video_paths = [os.path.join(self.video_path, i) for i in video_names]
        else:
            self.all_video_paths = [self.video_path]

        self.connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='task_queue', durable=True)

    def get_cam_name(self, path):
        filename = os.path.basename(path)
        cam_name = filename[:-4]
        return cam_name

    def send_to_remote_worker(self,ori_imgs,preds, batch):
        
        import pickle
        import string
        import random

        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + ".pkl"

        params = {"ori_imgs":ori_imgs, "preds": preds, "batch": batch}

        with open(filename, 'wb') as f:
            pickle.dump(params,f)

        

        print(f"orig_imgs {type(ori_imgs)}")
        # print(f"parmas {params}")

        # message = ' '.join(sys.argv[1:]) or "Hello World!"
        message = filename
        self.channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
            ))
        # self.connection.close()


    def dump_to_file(self, data, file_name=None):
        import pickle
        import string
        import random

        
        if file_name == None:
            file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + ".pkl"

        # params = {"ori_imgs":ori_imgs, "preds": preds, "batch": batch}

        with open(file_name, 'wb') as f:
            pickle.dump(data,f)
                    

    def run(self):
        for video_path in self.all_video_paths:
            cam_name = self.get_cam_name(video_path)
            videoloader = VideoLoader(self.config, video_path)
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

            for idx, batch in enumerate(tqdm(videoloader)):
                # if idx == 0:
                    # imgs = batch["imgs"]
                    # print(f"Batch is = {numpy.shape(imgs)}")


                if idx >= 0:
                    if batch is None:
                        continue
                    preds = self.detector.run(batch)
                    ori_imgs = batch['imgs']

                    # self.send_to_remote_worker(ori_imgs, preds, batch)
                    
                    
                    for i in range(len(ori_imgs)):
                        boxes = preds['boxes'][i]
                        labels = preds['labels'][i]
                        scores = preds['scores'][i]
                        frame_id = batch['frames'][i]

                        ori_img = ori_imgs[i]

                        if len(boxes) == 0:
                            continue

                        
                        # self.dump_to_file(ori_img,file_name="ori_img.pkl")
                        # self.dump_to_file(boxes,file_name="boxes.pkl")
                        # self.dump_to_file(labels,file_name="labels.pkl")
                        # self.dump_to_file(scores,file_name="scores.pkl")
                        # print("========================")
                        # print(f"{boxes}")
                        # print("========================")
                        # break

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

            

                    result_dict = videocounter.run(
                            frames = obj_dict['frames'],
                            tracks = obj_dict['tracks'], 
                            labels = obj_dict['labels'],
                            boxes = obj_dict['boxes'],
                            output_path=os.path.join(self.saved_path, cam_name+'.csv'))

            videoloader.reinitialize_stream()
            videowriter.write_full_to_video(
                videoloader,
                num_classes = len(self.class_names),
                csv_path=os.path.join(self.saved_path, cam_name+'.csv'),
                paths=videocounter.directions,
                polygons=videocounter.polygons)