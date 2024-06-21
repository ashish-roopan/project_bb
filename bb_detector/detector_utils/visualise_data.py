import os
import cv2
import argparse
import numpy as np

class Visualiser:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.train_img_dir = f'{self.data_dir}/train/images'
        self.val_img_dir = f'{self.data_dir}/val/images'

        self.train_images = [os.path.join(self.train_img_dir, img) for img in os.listdir(self.train_img_dir)]
        self.val_images = [os.path.join(self.val_img_dir, img) for img in os.listdir(self.val_img_dir)]

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            label = line.strip().split(' ')
            labels.append(label)
        return labels
    
    def visualise(self, filename):
        img = cv2.imread(filename)
        label_file = filename.replace('.jpg', '.txt').replace('images', 'labels')
        labels = self.read_labels(label_file)
        
        for label in labels:
            cls, cx, cy, w, h = map(float, label)
            x = int(cx * img.shape[1] - w * img.shape[1]/2)
            y = int(cy * img.shape[0] - h * img.shape[0]/2)
            w = int(w * img.shape[1])
            h = int(h * img.shape[0])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('image', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()        
    
    def run(self):
        while True:
            idx = np.random.randint(0, len(self.train_images))
            self.visualise(self.train_images[idx])
    
            idx = np.random.randint(0, len(self.val_images))
            self.visualise(self.val_images[idx])

def argparser():
    parser = argparse.ArgumentParser(description='Visualise the dataset')
    parser.add_argument("--data_dir", type=str, default='/home/ashish/datasets/bb_dataset', help="Path to the image or video")
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    visualiser = Visualiser(args)
    visualiser.run()