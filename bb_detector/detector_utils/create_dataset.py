import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


class CreateDataset:
    def __init__(self, args):
        self.num_images = args.num_images
        self.data_dir = args.data_dir
        self.ball_dir = args.ball_dir
        self.player_dir = args.player_dir
        self.fg = cv2.imread(args.fg, cv2.IMREAD_UNCHANGED)
        self.bg = cv2.imread(args.bg, cv2.IMREAD_UNCHANGED)
        self.ball_images = [cv2.imread(f'{self.ball_dir}/{img}',  cv2.IMREAD_UNCHANGED) for img in os.listdir(self.ball_dir) if img.endswith('.png')]
        self.player_images = [cv2.imread(f'{self.player_dir}/{img}',  cv2.IMREAD_UNCHANGED) for img in os.listdir(self.player_dir) if img.endswith('.png')]

    def add_images(self, bg, obj, pos):
        #. Find the object mask
        obj_mask = obj[:, :, 3]
        ret, obj_mask = cv2.threshold(obj_mask, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)

        #. Create the foreground with the object
        obj_fg = cv2.bitwise_and(obj, obj, mask=obj_mask)
        fg = np.zeros_like(bg)
        fg[pos[1]:pos[1]+obj.shape[0], pos[0]:pos[0]+obj.shape[1]] = obj_fg

        #. Create the background with the object
        inv_obj_mask = cv2.bitwise_not(obj_mask)
        bg_mask = np.ones_like(bg)[:,:,0] * 255
        bg_mask[pos[1]:pos[1]+obj.shape[0], pos[0]:pos[0]+obj.shape[1]] = inv_obj_mask
        bg = cv2.bitwise_and(bg, bg, mask=bg_mask)

        #. Add the object to the background
        img = cv2.add(bg, fg)
        return img
        
    def generate_image(self, ball, player1, player2):
        ball_img = ball['img']
        ball_x, ball_y = ball['pos']
        player1_img = player1['img']
        player1_x, player1_y = player1['pos']
        player2_img = player2['img']
        player2_x, player2_y = player2['pos']

        img = np.copy(self.bg)
        
        #. Draw the players 
        img = self.add_images(img, player1_img, (player1_x, player1_y))
        img = self.add_images(img, player2_img, (player2_x, player2_y))
        
        #. Draw the ball
        img = self.add_images(img, ball_img, (ball_x, ball_y))

        #. Add the foreground
        img = self.add_images(img, self.fg, (0, 0))
        
        return img
    
    def get_bbox(self, obj, pos):
        #. Find Bounding box using countours
        mask = obj[:, :, 3]
        ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        
        #. Convert the bounding box to absolute coordinates
        x = x + pos[0]
        y = y + pos[1]
        cx = (x + w/2)/self.bg.shape[1]
        cy = (y + h/2)/self.bg.shape[0]
        w = w/self.bg.shape[1]
        h = h/self.bg.shape[0]
        return [cx, cy, w, h]
    
    def get_random_ball(self):
        #. Get random ball image
        ball_img = self.ball_images[np.random.randint(0, len(self.ball_images))]
        angle = np.random.randint(0, 360)
        cv2.rotate(ball_img, angle)
        
        #. Get random ball position
        ball_x = np.random.randint(0, self.bg.shape[1]-ball_img.shape[1])
        ball_y = np.random.randint(0, self.bg.shape[0]*0.7)
        
        #. Get bounding box of the ball
        cx, cy, w, h = self.get_bbox(ball_img, (ball_x, ball_y))
        
        ball = {'img': ball_img, 'pos': (ball_x, ball_y), 'bbox': (cx, cy, w, h)}
        return ball
    
    def get_random_player(self, side):
        player_img = self.player_images[np.random.randint(0, len(self.player_images))]
        #. get the player position
        if side == 'left':
            player_x = np.random.randint(0, self.bg.shape[1]/2-player_img.shape[1])
        else:
            player_x = np.random.randint(self.bg.shape[1]/2+player_img.shape[1], self.bg.shape[1]-player_img.shape[1])
        player_y = np.random.randint(2.5 * player_img.shape[0], self.bg.shape[0]-player_img.shape[0]*2)
        
        #. Get the player bounding box
        cx, cy, w, h = self.get_bbox(player_img, (player_x, player_y))

        player = {'img': player_img, 'pos': (player_x, player_y), 'bbox': (cx, cy, w, h)}
        return player

    def draw_bbox(self, img, bbox, color):
        cx, cy, w, h = bbox
        x = int(cx * img.shape[1] - w * img.shape[1]/2)
        y = int(cy * img.shape[0] - h * img.shape[0]/2)
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        return img
    
    def save_labels(self, file, ball_bbox, player1_bbox, player2_bbox):
        with open(file, 'w') as f:
            f.write(f'0 {" ".join(map(str, ball_bbox))}\n')
            f.write(f'1 {" ".join(map(str, player1_bbox))}\n')
            f.write(f'1 {" ".join(map(str, player2_bbox))}\n')

    def create_dataset(self, save_dir, num_images):
        for i in tqdm(range(num_images)):
            #. Get the random ball and players
            ball = self.get_random_ball()
            player1 = self.get_random_player("left")
            player2 = self.get_random_player("right")

            #. Generate the image
            img = self.generate_image(ball, player1, player2)
            
            #. Save the image
            out_img = cv2.resize(img, (640, 360))
            cv2.imwrite(f'{save_dir}/images/image_{i}.jpg', out_img)

            #. Save the labels
            self.save_labels(f'{save_dir}/labels/image_{i}.txt', ball['bbox'], player1['bbox'], player2['bbox'])

            #. Display the image
            if args.display:
                #. draw the bounding boxes
                img = self.draw_bbox(img, ball['bbox'], color=(255, 255, 255))
                img = self.draw_bbox(img, player1['bbox'], color=(255, 0, 0))
                img = self.draw_bbox(img, player2['bbox'], color=(255, 255, 0))
                img = cv2.resize(img, (640, 480))
                cv2.imshow('img', img)  
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break  

    def run(self):
        #. Create the dataset directory
        train_dir = f'{self.data_dir}/train'
        val_dir = f'{self.data_dir}/val'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        #. Create the dataset
        num_train_images = int(self.num_images * 0.8)
        self.create_dataset(train_dir, num_train_images)
        self.create_dataset(val_dir, self.num_images - num_train_images)
        

def arg_parse():
    parser = argparse.ArgumentParser(description="Create a dataset of images")
    parser.add_argument("--data_dir", type=str, default='/home/ashish/datasets/bb_dataset', help="Path to the image or video")
    parser.add_argument("--ball_dir", type=str, default='bb_detector/data/objects/balls', help="Path to the image or video")
    parser.add_argument("--player_dir", type=str, default='bb_detector/data/objects/players', help="Path to the image or video")
    parser.add_argument("--fg", type=str, default='bb_detector/data/objects/foreground.png', help="Path to the image or video")
    parser.add_argument("--bg", type=str, default='bb_detector/data/objects/background.png', help="Path to the image or video")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--display", action='store_true', help="Display the image")
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()
    dataset = CreateDataset(args)
    dataset.run()