import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import visdom
from constants import settings
from constants.enum_keys import HK
from models.paf import PAF
from keypoint_network.pafs_network import PAFsLoss
from aichallenger import AicNorm


class Trainer:
    def __init__(self, batch_size, is_unittest=False):
        # self.debug_mode: Set num_worker to 0. Otherwise pycharm debug won't work due to multithreading.
        self.is_unittest = is_unittest
        self.epochs = 5
        self.val_step = 500
        self.batch_size = batch_size
        self.vis = visdom.Visdom()
        self.img_key = HK.NORM_IMAGE
        self.pcm_key = HK.PCM_ALL
        self.paf_key = HK.PAF_ALL

        if torch.cuda.is_available():
            print("GPU available.")
        else:
            print("GPU not available. running with CPU.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_pose = PAF()

        self.model_optimizer = optim.Adam(self.model_pose.parameters(), lr=1e-3)

        self.loss = PAFsLoss()

        train_dataset = AicNorm(Path.home() / "AI_challenger_keypoint", is_train=True,
                                resize_img_size=(512, 512), heat_size=(64, 64))
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=settings.num_workers,
                                       pin_memory=True, drop_last=True)

        # test_dataset = AicNorm(Path.home() / "AI_challenger_keypoint", is_train=False,
        #                        resize_img_size=(512, 512), heat_size=(64, 64))
        # self.val_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=workers, pin_memory=True,
        #                              drop_last=True)
        # self.val_iter = iter(self.val_loader)

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
        self.step = 1  # Step 1 not 0 prevents saving immediately at beginning, so as to support unit test.
        self.model_pose.load_ckpt()
        for self.epoch in range(self.epochs):
            print("Epoch:{}".format(self.epoch))
            self.run_epoch()

            if self.is_unittest:
                break

    def run_epoch(self):

        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            inputs[self.img_key] = inputs[self.img_key].to(self.device, dtype=torch.float32)
            inputs[self.pcm_key] = inputs[self.pcm_key].to(self.device, dtype=torch.float32)
            inputs[self.paf_key] = inputs[self.paf_key].to(self.device, dtype=torch.float32)
            loss, b1_out, b2_out = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            # Clear visdom environment
            if self.step == 0:
                self.vis.close()
            # Validate
            if self.step % self.val_step == 0:
                # self.val()
                self.model_pose.save_ckpt()

            if self.step % 50 == 0:
                print("step {}; Loss {}".format(self.step, loss.item()))

            # Show training materials
            if self.step % self.val_step == 0:
                pcm_CHW = b1_out[0].cpu().detach().numpy()
                paf_CHW = b2_out[0].cpu().detach().numpy()
                img_CHW = inputs[self.img_key][0].cpu().detach().numpy()[::-1, ...]
                pred_pcm_amax = np.amax(pcm_CHW, axis=0)  # HW
                gt_pcm_amax = np.amax(inputs[self.pcm_key][0].cpu().detach().numpy(), axis=0)
                pred_paf_amax = np.amax(paf_CHW, axis=0)
                gt_paf_amax = np.amax(inputs[self.paf_key][0].cpu().detach().numpy(), axis=0)
                self.vis.image(img_CHW, win="Input", opts={'title': "Input"})
                self.vis.heatmap(np.flipud(pred_pcm_amax), win="Pred-PCM", opts={'title': "Pred-PCM"})
                self.vis.heatmap(np.flipud(gt_pcm_amax), win="GT-PCM", opts={'title': "GT-PCM"})
                self.vis.heatmap(np.flipud(pred_paf_amax), win="Pred-PAF", opts={'title': "Pred-PAF"})
                self.vis.heatmap(np.flipud(gt_paf_amax), win="GT-PAF", opts={'title': "GT-PAF"})
                self.vis.line(X=np.array([self.step]), Y=loss.cpu().detach().numpy()[np.newaxis], win='Loss', update='append')
            if self.is_unittest:
                break
            self.step += 1


    # def val(self, ):
    #     """Validate the model on a single minibatch
    #     """
    #     self.set_eval()
    #
    #     try:
    #         inputs = next(self.val_iter)
    #     except StopIteration:
    #         self.val_iter = iter(self.val_loader)
    #         inputs = next(self.val_iter)
    #
    #     inputs_gpu = {self.img_key: inputs[self.img_key].to(self.device, dtype=torch.float32),
    #                   self.pcm_key: inputs[self.pcm_key].to(self.device, dtype=torch.float32),
    #                   self.paf_key: inputs[self.paf_key].to(self.device, dtype=torch.float32)}
    #     with torch.no_grad():
    #         loss, b1_out, b2_out = self.process_batch(inputs_gpu)
    #     pred_pcm_amax = np.amax(b1_out[0].cpu().numpy(), axis=0)  # HW
    #     gt_pcm_amax = np.amax(inputs[self.pcm_key][0].cpu().numpy(), axis=0)
    #     pred_paf_amax = np.amax(b2_out[0].cpu().numpy(), axis=0)
    #     gt_paf_amax = np.amax(inputs[self.paf_key][0].cpu().numpy(), axis=0)
    #     # Image augmentation disabled due to pred phase
    #     self.vis.image(inputs[self.img_key][0].cpu().numpy()[::-1, ...], win="Input", opts={'title': "Input"})
    #     self.vis.heatmap(np.flipud(pred_pcm_amax), win="Pred-PCM", opts={'title': "Pred-PCM"})
    #     self.vis.heatmap(np.flipud(gt_pcm_amax), win="GT-PCM", opts={'title': "GT-PCM"})
    #     self.vis.heatmap(np.flipud(pred_paf_amax), win="Pred-PAF", opts={'title': "Pred-PAF"})
    #     self.vis.heatmap(np.flipud(gt_paf_amax), win="GT-PAF", opts={'title': "GT-PAF"})
    #     self.vis.line(X=np.array([self.step]), Y=loss.cpu().numpy()[np.newaxis], win='Loss', update='append')
    #     self.set_train()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        res = self.model_pose(inputs[self.img_key])
        gt_pcm = inputs[self.pcm_key]  # ["heatmap"]: {"vis_or_not": NJHW, "visible": NJHW}
        gt_pcm = gt_pcm.unsqueeze(1)
        gt_paf = inputs[self.paf_key]
        gt_paf = gt_paf.unsqueeze(1)
        b1_stack = torch.stack(res[HK.B1_SUPERVISION], dim=1)  # Shape (N, Stage, C, H, W)
        b2_stack = torch.stack(res[HK.B2_SUPERVISION], dim=1)

        loss = self.loss(b1_stack, b2_stack, gt_pcm, gt_paf)
        return loss, res[HK.B1_OUT], res[HK.B2_OUT]

