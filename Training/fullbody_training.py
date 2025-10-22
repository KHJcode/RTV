import math
import os
import sys
import time

import numpy as np
import torch
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Datasets.fullbody_garment.fullbody_garment import FullBodyGarment
from model.pix2pixHD.models import create_model
from options.train_options import TrainOptions
import util.util as util
from util.visualizer import Visualizer


def lcm(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def main():
    opt = TrainOptions().parse()
    if opt.dataset_path is None:
        print("Please specify a dataset for training!")
        sys.exit(0)

    path_list = opt.dataset_path.split(",")
    dataset = FullBodyGarment(path_list[0], img_size=opt.img_size)
    if len(path_list) > 1:
        for extra_path in path_list[1:]:
            dataset = dataset + FullBodyGarment(extra_path, img_size=opt.img_size)

    dataset_size = len(dataset)
    use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads),
        pin_memory=use_cuda,
        drop_last=False,
    )
    model = create_model(opt)
    visualizer = Visualizer(opt)
    start_epoch, epoch_iter = 1, 0
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataloader):
            garment_img, vm_img, dp_img, garment_mask = [
                tensor.to(device, non_blocking=True) for tensor in data
            ]

            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            save_fake = total_steps % opt.display_freq == display_delta

            input_img = torch.cat([vm_img, dp_img], 1)
            gt_image = torch.cat([garment_img, garment_mask], 1)
            losses, generated = model(input_img, gt_image, infer=save_fake)

            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            loss_D = (loss_dict["D_fake"] + loss_dict["D_real"]) * 0.5
            loss_G = loss_dict["G_GAN"] + loss_dict.get("G_GAN_Feat", 0) + loss_dict.get("G_VGG", 0)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            if save_fake:
                real_list = [
                    ("garment_img0", util.tensor2im((garment_img / 2.0 + 0.5)[0], rgb=True)),
                ]
                fake_list = [
                    ("fake_img0", util.tensor2im((generated.data[:, [0, 1, 2], :, :] / 2.0 + 0.5)[0], rgb=True)),
                ]
                fake2_list = [
                    ("fake_mask0", util.tensor2im((generated.data[:, [3, 3, 3], :, :])[0], rgb=True)),
                ]
                input_list = [
                    ("vm_image0", util.tensor2im((vm_img / 2.0 + 0.5)[0], rgb=True)),
                ]
                dp_list = [
                    ("dp_image0", util.tensor2im((dp_img / 2.0 + 0.5)[0], rgb=True)),
                ]
                visuals = OrderedDict(real_list + input_list + fake_list + dp_list + fake2_list)
                visualizer.display_current_results(visuals, epoch, total_steps)

            if total_steps % opt.save_latest_freq == save_delta:
                print(f"saving the latest model (epoch {epoch}, total_steps {total_steps})")
                model.module.save("latest")
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=",", fmt="%d")

            if epoch_iter >= dataset_size:
                break

        print(f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec")

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_steps}")
            model.module.save("latest")
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=",", fmt="%d")

        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        if epoch > opt.niter:
            model.module.update_learning_rate()


if __name__ == "__main__":
    main()
