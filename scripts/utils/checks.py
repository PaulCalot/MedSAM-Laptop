import torch
import matplotlib.pyplot as plt
import random
import pathlib
import logging

# local lib
import medsamlaptop
import medsamtools

def perform_dataset_sanity_check(data_root):
    assert(isinstance(data_root, pathlib.Path))
    try:
        tr_dataset = medsamlaptop.data.NpyDataset(data_root, data_aug=True)
    except Exception as e:
        logging.error("Could not perform sanity check. Are you sure the data"
                      f"directory path corresponds to a NpyDataset ? Trace: {e}")
        return
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        medsamlaptop.plot.images.show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        medsamlaptop.plot.images.show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        medsamlaptop.plot.images.show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        medsamlaptop.plot.images.show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            medsamtools.user.get_path_to_results() / f'{data_root.stem}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break