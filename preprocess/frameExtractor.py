import cv2 
import os 
import logging
from multiprocessing import Queue, Process

logger = logging.getLogger()

def get_images(label_folder, video_file, video_name):
    vid = cv2.VideoCapture(video_file)

    index = 0
    while True:
        _, frame = vid.read()
        
        if frame is None:
            break 

        img_name = f'{video_name}_{index}.jpg'
        img_name = os.path.join(label_folder, img_name)
        cv2.imwrite(img_name, frame)

        index += 1

class Extract_images_multiprocess:
    def __init__(self, args, protocol_info, subset_folder, label_to_process):
        '''
            Carrega o video e extrai as imagens dele
        '''
        if args['subset'] == 'Train':
            self.videos_folder = args['train_folder']
        elif args['subset'] == 'Test':
            self.videos_folder = args['test_folder']
        elif args['subset'] == 'Dev':
            self.videos_folder = args['dev_folder']

        self.protocol_info = protocol_info

        self.label_folder = os.path.join(subset_folder, label_to_process)
        if not os.path.isdir(self.label_folder):
            os.mkdir(self.label_folder)

        self.name = label_to_process

        self.finished = False

        self.Q = Queue(maxsize=1)

    def start(self):
        # start the thread 
        self.P = Process(target=self.loop, name=self.name)
        self.P.daemon = True
        self.P.start()
        
        return self

    def loop(self):
        i = 1
        for video_name in self.protocol_info:

            video_file = os.path.join(self.videos_folder, f'{video_name}.avi')

            if not os.path.isfile(video_file):
                logger.error(f'[Thread {self.name}] The file {video_file} does not exist')

            logger.info(f'[Thread {self.name}] Extracting images from video {video_name} ({i}/{len(self.protocol_info)})')
            get_images(self.label_folder, video_file, video_name)
            i += 1

        self.Q.put(True)
        while not self.Q.empty():
            logger.info(f"[Thread {self.name}] Waiting for principal process to get the queue info..")

        self.finish()
        logger.info(f"[Thread {self.name}] Finished..")

    def finish(self):
        self.finished = True

    def __del__(self):
        logger.info(f'[Thread {self.name}] Bye Bye..')
