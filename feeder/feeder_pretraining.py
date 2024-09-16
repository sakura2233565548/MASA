# sys
import pickle

# torch
import numpy.random
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
from feeder.SLR_Dataset import datasets

try:
    from feeder import augmentations
except:
    import augmentations


def my_collate_fn(data_list, keys):
    vid_len = [x[keys[0]]['kp2d'].shape[0] for x in data_list]
    max_len = max(vid_len)
    new_batch = {}
    for key in keys:
        new_batch[key] = {}
        for k,v in data_list[0][key].items():
            new_batch[key][k] = torch.zeros((len(data_list), max_len, *v.shape[1:]), dtype=v.dtype)
    new_batch['right']['vid_len'] = torch.zeros((len(data_list), max_len), dtype=torch.float32)
    for i in range(len(data_list)):
        data = data_list[i]
        for key in keys:
            for k, v in data[key].items():
                new_batch[key][k][i, :vid_len[i]] = v
        new_batch['right']['vid_len'][i, :vid_len[i]] = 1
    return new_batch

def my_collate_fn_second(data_list, keys):
    vid_len = [x[keys[0]].shape[0] for x in data_list]
    max_len = max(vid_len)
    new_batch = {}
    for key in keys:
        new_batch[key] = torch.zeros((len(data_list), *data_list[0][key].shape[0:]), dtype=torch.float32)
    for i in range(len(data_list)):
        data = data_list[i]
        for key in keys:
            new_batch[key][i, :data[key].shape[0]] = torch.tensor(data[key])
    return new_batch


def collate_fn(batch):
    data_v1 = [k[0] for k in batch]
    data_v2 = [k[1] for k in batch]
    data_v1 = my_collate_fn(data_v1, keys = ['right', 'left', 'body'])
    data_v2 = my_collate_fn(data_v2, keys=['right', 'left', 'body'])
    return data_v1, data_v2


class Feeder_SLR(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mask_ratio=0.9,
                 mmap=True):
        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.input_representation = input_representation
        self.crop_resize = True
        self.l_ratio = l_ratio
        self.mask_ratio = mask_ratio

        self.ds = datasets.TotalDataset(subset_name=['NMFs_CSL', 'SLR500', 'WLASL', 'MS_ASL'])
        self.temporal_crop = augmentations.TemporalRandomCrop(size=input_size, interval=1)
        print('sample length is ', self.__len__())
        print("l_ratio", self.l_ratio)


    def __len__(self):
        return self.ds.len()

    def __iter__(self):
        return self

    def collect_data(self, data_numpy, data_raw, indices):
        results = {}
        rh = data_numpy[:, :, :21, 0].transpose(1, 2, 0).astype('float32')
        lh = data_numpy[:, :, 21:42, 0].transpose(1, 2, 0).astype('float32')
        body = data_numpy[:, :, 42:, 0].transpose(1, 2, 0).astype('float32')
        seed = np.random.randint(low=0, high=10)
        if seed % 2 == 0:
            results['rh'] = rh
            results['lh'] = lh
            results['body'] = body
            rh_mask = data_raw['right']['mask'][indices, :, :].numpy()
            lh_mask = data_raw['left']['mask'][indices, :, :].numpy()
            results['mask'] = np.concatenate([rh_mask, lh_mask], axis=0)
        else:
            results['lh'] = rh
            results['rh'] = lh
            results['body'] = body
            rh_mask = data_raw['right']['mask'][indices, :, :].numpy()
            lh_mask = data_raw['left']['mask'][indices, :, :].numpy()
            results['mask'] = np.concatenate([lh_mask, rh_mask], axis=0)
        return results

    def collect_data_mask(self, data_raw, indices, interval=3):
        results = {}
        parts = ['right', 'left', 'body']
        frames_id = [i for i in range(len(indices))]
        random.shuffle(frames_id)
        for part in parts:
            results[part] = {}
            part_data = data_raw[part]
            for k,v in part_data.items():
                if type(v) is list:
                    continue
                if v.dim() == 3:
                    results[part][k] = v[indices, :, :]
                if 'gt' in k:
                    temp_tensor = torch.zeros_like(v[indices, :, :])
                    temp_tensor[:-interval, :, :] = v[indices, :, :][interval:, :, :] - v[indices, :, :][:-interval, :, :]
                    results[part][k] = temp_tensor
                if k in ['flag_2d', 'body_pose_conf']:
                    temp_tensor = torch.zeros_like(v[indices, :, :])
                    temp_tensor[:-interval, :, :] = v[indices, :, :][interval:, :, :] * v[indices, :, :][:-interval, :, :]
                    results[part][k] = temp_tensor

        results['right']['flag_2d'] = results['right']['flag_2d'] * (results['right']['gts'] > 3.0)
        results['left']['flag_2d'] = results['left']['flag_2d'] * (results['left']['gts'] > 3.0)
        results['body']['body_pose_conf'] = results['body']['body_pose_conf'] * (results['body']['body_pose_gt'] > 3.0)

        results['right']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)
        results['left']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)
        results['body']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)

        rh_masked_index = torch.topk(torch.sum(torch.sum(results['right']['gts'] > 3.0, dim=-1), dim=-1), k=(torch.sum(torch.sum(results['right']['gts'] > 3.0, dim=-1), dim=-1)>1).sum()).indices.numpy().tolist()
        lh_masked_index = torch.topk(torch.sum(torch.sum(results['left']['gts'] > 3.0, dim=-1), dim=-1), k=(torch.sum(torch.sum(results['left']['gts'] > 3.0, dim=-1), dim=-1)>1).sum()).indices.numpy().tolist()
        body_masked_index = torch.topk(torch.sum(torch.sum(results['body']['body_pose_gt'] > 3.0, dim=-1), dim=-1), k=(torch.sum(torch.sum(results['body']['body_pose_gt'] > 3.0, dim=-1), dim=-1)>1).sum()).indices.numpy().tolist()

        random.shuffle(rh_masked_index), random.shuffle(lh_masked_index), random.shuffle(body_masked_index)

        if len(rh_masked_index) != 0:
            results['right']['masked'][rh_masked_index[:int(len(rh_masked_index)*self.mask_ratio)], :] = 1.0
        if len(lh_masked_index) != 0:
            results['left']['masked'][lh_masked_index[:int(len(lh_masked_index)*self.mask_ratio)], :] = 1.0
        if len(body_masked_index) != 0:
            results['body']['masked'][body_masked_index[:int(len(body_masked_index)*self.mask_ratio)], :] = 1.0
        return results

    def collect_data_mask_v2(self, data_raw, indices, mask_ratio=0.5):
        results = {}
        parts = ['right', 'left', 'body']
        frames_id = [i for i in range(len(indices))]
        random.shuffle(frames_id)
        for part in parts:
            results[part] = {}
            part_data = data_raw[part]
            for k,v in part_data.items():
                if type(v) is list:
                    continue
                if v.dim() == 3:
                    results[part][k] = v[indices, :, :]
                if 'gt' in k:
                    temp_tensor = torch.zeros_like(v[indices, :, :])
                    temp_tensor[:-1, :, :] = v[indices, :, :][1:, :, :] - v[indices, :, :][:-1, :, :]
                    results[part][k] = temp_tensor
                if k in ['flag_2d', 'body_pose_conf']:
                    temp_tensor = torch.zeros_like(v[indices, :, :])
                    temp_tensor[:-1, :, :] = v[indices, :, :][1:, :, :] * v[indices, :, :][:-1, :, :]
                    results[part][k] = temp_tensor

        results['right']['flag_2d'] = results['right']['flag_2d'] * (results['right']['gts'] > 3.0)
        results['left']['flag_2d'] = results['left']['flag_2d'] * (results['left']['gts'] > 3.0)
        results['body']['body_pose_conf'] = results['body']['body_pose_conf'] * (results['body']['body_pose_gt'] > 3.0)

        results['right']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)
        results['left']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)
        results['body']['masked'] = torch.zeros((len(indices), 1), dtype=torch.float32)

        rh_masked_index = [i for i in range(len(indices))]
        lh_masked_index = [i for i in range(len(indices))]
        body_masked_index = [i for i in range(len(indices))]

        random.shuffle(rh_masked_index), random.shuffle(lh_masked_index), random.shuffle(body_masked_index)
        if len(rh_masked_index) != 0:
            results['right']['masked'][rh_masked_index[:int(len(rh_masked_index)*mask_ratio)], :] = 1.0
        if len(lh_masked_index) != 0:
            results['left']['masked'][lh_masked_index[:int(len(lh_masked_index)*mask_ratio)], :] = 1.0
        if len(body_masked_index) != 0:
            results['body']['masked'][body_masked_index[:int(len(body_masked_index)*mask_ratio)], :] = 1.0
        return results

    def temporal_shifted(self, indices, num_of_frames, overlap=0.6):
        if num_of_frames < len(indices):
            max_shifted = int(len(indices) * (1-overlap))
            selected_indices = [i for i in range(max(0, indices[0]-max_shifted), indices[0]-1)] + indices + [i for i in range(indices[-1]+1, min(indices[-1]+max_shifted, num_of_frames-1))]
            start_id = int(numpy.random.random() * (len(selected_indices) - len(indices)))
            output_indices = selected_indices[start_id:start_id+len(indices)]
        else:
            output_indices = self.temporal_crop([i for i in range(num_of_frames)])
        return output_indices

    def __getitem__(self, index):
        # get raw input
        data_raw = self.ds.get_sample(index)
        data_numpy = torch.cat([data_raw['right']['kp2d'], data_raw['left']['kp2d'], data_raw['body']['body_pose']], dim=1)
        data_numpy = data_numpy[:, :, :, None].permute(2, 0, 1, 3).numpy()

        # input: C, T, V, M
        number_of_frames = min(data_numpy.shape[1], 300) 

        indices_v1 =[i for i in range(number_of_frames)]
        data_v1 = self.collect_data_mask(data_raw, indices_v1)

        indices_v2 = [i for i in range(number_of_frames)]
        data_v2 = self.collect_data_mask_v2(data_raw, indices_v2)

        return data_v1, data_v2