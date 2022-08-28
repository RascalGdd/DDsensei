import os
import pandas
from pathlib import Path
from crop import *
from torch.utils.data import DataLoader
import torch
import numpy as np
from vgg16 import *
from tqdm import tqdm
from cfg import *

def seed_worker(id):
    np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    pass

if __name__ == '__main__':


    device = torch.device('cuda')

    network = VGG16(False, padding='none').to(device)
    extract   = lambda img: network.fw_relu(img, 13)[-1]
    crop_size = 196 # VGG-16 receptive field at relu 5-3
    dim       = 512 # channel width of VGG-16 at relu 5-3
    num_crops = 15

    dataset = ImageDataset(file_list_fake)
    loader  = DataLoader(dataset,
            batch_size=1, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    print('Computing mean/std...')

    m, s = [], []
    for i, batch in tqdm(zip(range(1000), loader)):
        m.append(batch.img.mean(dim=(2, 3)))
        s.append(batch.img.std(dim=(2, 3)))
        pass

    m = torch.cat(m, 0).mean(dim=0)
    s = torch.cat(s, 0).mean(dim=0)

    network.set_mean_std(m[0], m[1], m[2], s[0], s[1], s[2])

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=1, pin_memory=True, drop_last=False,
                                         worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    features = np.zeros((len(dataset) * num_crops, dim), np.float16)

    print('Sampling crops...')

    ip = 0
    with open(out_dir / f'crop_fake.csv', 'w') as log:
        log.write('id,path,r0,r1,c0,c1\n')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):

                n, _, h, w = batch.img.shape
                assert n == 1

                if i == 0:
                    print(f'Image size is {h}x{w} - sampling {num_crops} crops per image.')
                    pass

                c0s = torch.randint(w - crop_size + 1, (num_crops, 1))
                r0s = torch.randint(h - crop_size + 1, (num_crops, 1))

                samples = []
                for j in range(num_crops):
                    r0 = r0s[j].item()
                    c0 = c0s[j].item()
                    r1 = r0 + crop_size
                    c1 = c0 + crop_size
                    samples.append(batch.img[0, :, r0:r1, c0:c1].reshape(1, 3, crop_size, crop_size))
                    log.write(f'{ip},{batch.path[0]},{r0},{r1},{c0},{c1}\n')
                    ip += 1
                    pass

                samples = torch.cat(samples, 0)
                samples = samples.to(device, non_blocking=True)
                f = extract(samples)

                features[ip - num_crops:ip, :] = f.cpu().numpy().astype(np.float16).reshape(num_crops, dim)
                pass
            pass
        pass

    print('Saving features.')
    np.savez_compressed(out_dir / f'crop_fake', crops=features)
    pass



    dataset = ImageDataset(file_list_real)
    loader  = DataLoader(dataset,
            batch_size=1, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=False, worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    print('Computing mean/std...')

    m, s = [], []
    for i, batch in tqdm(zip(range(1000), loader)):
        m.append(batch.img.mean(dim=(2, 3)))
        s.append(batch.img.std(dim=(2, 3)))
        pass

    m = torch.cat(m, 0).mean(dim=0)
    s = torch.cat(s, 0).mean(dim=0)

    network.set_mean_std(m[0], m[1], m[2], s[0], s[1], s[2])

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=1, pin_memory=True, drop_last=False,
                                         worker_init_fn=seed_worker, collate_fn=ImageBatch.collate_fn)

    features = np.zeros((len(dataset) * num_crops, dim), np.float16)

    print('Sampling crops...')

    ip = 0
    with open(out_dir / f'crop_real.csv', 'w') as log:
        log.write('id,path,r0,r1,c0,c1\n')
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):

                n, _, h, w = batch.img.shape
                assert n == 1

                if i == 0:
                    print(f'Image size is {h}x{w} - sampling {num_crops} crops per image.')
                    pass

                c0s = torch.randint(w - crop_size + 1, (num_crops, 1))
                r0s = torch.randint(h - crop_size + 1, (num_crops, 1))

                samples = []
                for j in range(num_crops):
                    r0 = r0s[j].item()
                    c0 = c0s[j].item()
                    r1 = r0 + crop_size
                    c1 = c0 + crop_size
                    samples.append(batch.img[0, :, r0:r1, c0:c1].reshape(1, 3, crop_size, crop_size))
                    log.write(f'{ip},{batch.path[0]},{r0},{r1},{c0},{c1}\n')
                    ip += 1
                    pass

                samples = torch.cat(samples, 0)
                samples = samples.to(device, non_blocking=True)
                f = extract(samples)

                features[ip - num_crops:ip, :] = f.cpu().numpy().astype(np.float16).reshape(num_crops, dim)
                pass
            pass
        pass

    print('Saving features.')
    np.savez_compressed(out_dir / f'crop_real', crops=features)
    pass