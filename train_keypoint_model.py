import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import torchvision.transforms as transforms

from pafs_network import PAFsNetwork
from aic_dataset import AICDataset

class Trainer:
    def __init__(self, batch_size, debug_mode):
        # Debug mode:
        #    Set num_worker to 0. Otherwise pycharm debug won't work due to multithreading.
        #    Set batch_size to 1, ignore the argument given.
        self.debug_mode = debug_mode
        self.epochs = 100
        self.val_step = 500

        if torch.cuda.is_available():
            print("GPU available.")
        else:
            print("GPU not available! running with CPU.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_pose = PAFsNetwork(b1_classes=14, b2_classes=11 * 2)
        self.model_pose.to(self.device)

        self.model_optimizer = optim.Adam(self.model_pose.parameters(), lr=1e-4)
        self.model_path = Path("pose_model.pt")
        self.l2 = torch.nn.MSELoss()

        if self.debug_mode:
            self.batch_size = 1
            workers = 0
        else:
            self.batch_size = batch_size
            workers = 4

        # TODO: Use train dataset (much bigger)

        train_dataset = AICDataset(Path.home() / "AI_challenger_keypoint", input_img_size=(512, 512), heat_size=(64, 64), is_train=True)
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=workers,
                                       pin_memory=True, drop_last=True)
        test_dataset = AICDataset(Path.home() / "AI_challenger_keypoint", input_img_size=(512, 512), heat_size=(64, 64), is_train=False)
        self.val_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=workers, pin_memory=True,
                                     drop_last=True)
        self.val_iter = iter(self.val_loader)

    def set_train(self):
        """Convert models to training mode
        """
        self.model_pose.train()

    def set_eval(self):
        """Convert models to testing/evaluation mode
        """
        self.model_pose.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.load_model()
        for self.epoch in range(self.epochs):
            print("Epoch:{}".format(self.epoch))
            self.run_epoch()
            self.save_model()

    def run_epoch(self):

        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            inputs["resized"] = inputs["resized"].to(self.device)
            inputs["heatmap"]["vis_or_not"] = inputs["heatmap"]["vis_or_not"].to(self.device, dtype=torch.float)
            inputs["heatmap"]["visible"] = inputs["heatmap"]["visible"].to(self.device, dtype=torch.float)

            predicts, loss = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            if self.step % self.val_step == 0:
                # self.log("train", inputs, predicts, loss)
                self.val()

            if self.step % 10 == 0:
                print("step {}; Loss {}".format(self.step, loss.item()))

            self.step += 1

    def val(self, ):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = next(self.val_iter)
            inputs = self.val_iter.next()
        inputs["resized"] = inputs["resized"].to(self.device)
        inputs["heatmap"]["vis_or_not"] = inputs["heatmap"]["vis_or_not"].to(self.device)
        inputs["heatmap"]["visible"] = inputs["heatmap"]["visible"].to(self.device)
        with torch.no_grad():
            predicts, loss = self.process_batch(inputs)

        img = inputs["resized"][0].cpu().numpy()
        img = img.transpose((1, 2, 0))
        predicts = predicts[0].cpu().numpy()
        pred_vis = predicts[:, :]
        pred_vis = np.amax(pred_vis, axis=0)
        gt_vis = inputs["heatmap"]["visible"][0].cpu().numpy()
        gt_vis = np.amax(gt_vis, axis=0)

        plt.close('all')
        fig, ax = plt.subplots(2, 3, figsize=(15, 15))
        img = np.asarray(img)
        ax[0, 0].imshow(img)
        ax[0, 2].imshow(gt_vis)
        ax[1, 2].imshow(pred_vis)
        plt.show()
        plt.pause(1.0)

        self.set_train()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        b1_stages, b2_stages, b1, b2 = self.model_pose(inputs["resized"])
        gt_heat = inputs["heatmap"]  # ["heatmap"]: {"vis_or_not": NJHW, "visible": NJHW}
        gt_heat_NCHW = gt_heat["visible"] # 以前是两种都保留：torch.cat((gt_heat["vis_or_not"], gt_heat["visible"]), dim=1)
        b1_all = torch.stack(b1_stages, dim=0)
        heat_all = torch.stack(3*[gt_heat_NCHW])

        loss = self.l2(b1_all, heat_all)

        return b1, loss

    def save_model(self):

        torch.save(self.model_pose, self.model_path)
        print("Model saved.")

    def load_model(self):
        if Path.is_file(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model_pose.load_state_dict(checkpoint)
        else:
            print("No model file found.")


def main():
    plt.ion()
    Trainer(batch_size=5, debug_mode=True).train()


if __name__ == "__main__":
    main()
