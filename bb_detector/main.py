import argparse
from ultralytics import YOLO


class BbTrainer:
    def __init__(self, args):
        self.model = YOLO(args.weights)
        self.yaml = args.yaml
        self.epochs = args.epochs

    def train(self):
        self.model.train(data=self.yaml, epochs=self.epochs)

    def val(self):
        return self.model.val()
    
    def run(self):
        self.train()
        self.val()


def argparser():
    parser = argparse.ArgumentParser(description='Train a custom YOLO model')
    parser.add_argument("--weights", type=str, default='checkpoints/yolov8n.pt', help="Path to the weights file")
    parser.add_argument("--yaml", type=str, default='configs/bbTrain.yaml', help="Path to the yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    trainer = BbTrainer(args)
    trainer.run()
    