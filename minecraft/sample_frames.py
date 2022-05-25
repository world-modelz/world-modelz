import argparse
import random
from pathlib import Path

from PIL import Image
import numpy as np

import minerl
from minerl.data import BufferedBatchIter
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/minerl/', type=str)
    parser.add_argument('--skip_frames', default=30, type=int)
    opt = parser.parse_args()
    return opt


environment_names = [
    'MineRLBasaltBuildVillageHouse-v0',
    'MineRLBasaltCreatePlainsAnimalPen-v0',
    'MineRLBasaltCreateVillageAnimalPen-v0',
    'MineRLBasaltFindCave-v0',
    'MineRLBasaltMakeWaterfall-v0',
    'MineRLNavigateDense-v0',
    #'MineRLNavigateDenseVectorObf-v0',
    'MineRLNavigateExtremeDense-v0',
    #'MineRLNavigateExtremeDenseVectorObf-v0',
    'MineRLNavigateExtreme-v0',
    #'MineRLNavigateExtremeVectorObf-v0',
    'MineRLNavigate-v0',
    #'MineRLNavigateVectorObf-v0',
    'MineRLObtainDiamondDense-v0',
    #'MineRLObtainDiamondDenseVectorObf-v0',
    #'MineRLObtainDiamondSurvivalVectorObf-v0',
    'MineRLObtainDiamond-v0',
    #'MineRLObtainDiamondVectorObf-v0',
    'MineRLObtainIronPickaxeDense-v0',
    #'MineRLObtainIronPickaxeDenseVectorObf-v0',
    'MineRLObtainIronPickaxe-v0',
    #'MineRLObtainIronPickaxeVectorObf-v0',
    'MineRLTreechop-v0',
    #'MineRLTreechopVectorObf-v0'
]


class BufferedTrajSampler:
    def __init__(self, environment_names, data_dir, buffer_size=100000, max_segment_length=1000, traj_len=16, skip_frames=2, frame_shape=(64,64,3)):
        self.environment_names = environment_names
        self.data_dir = data_dir
        self.frame_shape = frame_shape

        # build list of all trajectories
        traj_names = []
        for environment_name in environment_names:
            data = minerl.data.make(environment_name, data_dir=data_dir)
            traj_names.extend((environment_name, t) for t in data.get_trajectory_names())
        self.traj_names = traj_names

        self.buffer_size = buffer_size
        self.traj_len = traj_len
        self.skip_frames = skip_frames
        self.max_segment_length = max_segment_length
        self.example_offsets = []
        self.example_index = -1

    def fill_buffer(self):
        
        total_frames = 0
        segments = []
        example_offsets = []

        order = [] 
        i = 0
        
        while total_frames < self.buffer_size:
            if i >= len(order):
                order = torch.randperm(len(self.traj_names)).tolist()
                i = 0

            environment_name, trajectory_name = self.traj_names[i]
            i += 1
            data = minerl.data.make(environment_name, data_dir=self.data_dir)
            traj_data = data.load_data(trajectory_name)

            # read whole trajectory into buffer
            frames = []
            skip = 0
            for data_tuple in traj_data:
                if skip > 0:
                    skip -= 1
                else:
                    obs = data_tuple[0]
                    pov = obs['pov']
                    frames.append(pov)
                    skip = self.skip_frames

            if len(frames) <= self.traj_len:
                continue

            print('#frames:', len(frames))

            # select random segment of trajectory to keep
            max_offset = len(frames) - self.max_segment_length
            if max_offset > 0:
                begin = random.randint(0, max_offset)
                frames = frames[begin:begin+self.max_segment_length]

            segment_index = len(segments)
            segments.append(frames)
            total_frames += len(frames)

            # generate random offsets into segment as basis for training examples
            sample_divisor = 8
            for j in range((len(frames)-self.traj_len) // sample_divisor):   # sample depending on traj len
                offset = random.randint(0, len(frames)-self.traj_len)
                example_offsets.append((segment_index, offset))

            print(f'total_frames: {total_frames}; examples: {len(example_offsets)}')

        self.segments = segments
        p = np.random.permutation(len(example_offsets))
        self.example_offsets = [example_offsets[k] for k in p] 
        self.example_index = 0


    def sample_batch(self, batch_size):
        l = self.traj_len

        batch_shape = (batch_size, l) + self.frame_shape
        batch = np.ndarray(batch_shape, dtype=np.uint8)
        
        for i in range(batch_size):
            if self.example_index >= len(self.example_offsets):
                self.fill_buffer()
            
            segment, offset = self.example_offsets[self.example_index]
            self.example_index += 1
            batch[i] = self.segments[segment][offset:offset+l]
        
        return batch



def sample_batch():
    data_dir = '/mnt/minerl/'
    #environment_names = ['MineRLTreechop-v0']
    smp = BufferedTrajSampler(environment_names, data_dir, buffer_size=10000)
    print('total trajectories: ', len(smp.traj_names))

    for i in range(100):
        b = smp.sample_batch(20)

        x = torch.from_numpy(b)
        x = x.permute(0,1,4,2,3)

        import torchvision

        grid = torchvision.utils.make_grid(x.reshape(-1, 3,64,64), nrow=16).permute(1,2,0).numpy()
        print(grid.shape, grid.dtype)

        #im = Image.fromarray(grid)
        #im.save(f'test{i:03d}.png', format=None)

    #print(b.shape)
    
        
    #     data_loader = data.load_data(traj_names[0])
    #     for data_tuple in data_loader:
    #         obs = data_tuple[0]


    # print(len(data_loader[0]))


def main():
    print('Using pytorch version {}'.format(torch.__version__))

    opt = parse_args()
    print('Options:', opt)

    data_dir = opt.data_dir

    output_base_path = Path('/mnt/minerl/frames/')

    file_names = []
    total_frames = 0
    frame_index = 0

    # random frame skipping
    skip_frames = opt.skip_frames   # 30 is default
    max_skip_frames = 2 * skip_frames
    frames_to_skip = 0

    for env_index,name in enumerate(environment_names):
        data = minerl.data.make(name, data_dir=data_dir)
        traj_names = data.get_trajectory_names()
        traj_index = 0

        env_folder_path = folder_path = output_base_path / name
        env_folder_path.mkdir(parents=True, exist_ok=True)

        for traj_name in traj_names:

            data_loader = data.load_data(traj_name)

            folder_path = env_folder_path / f'{traj_index:06d}'
            folder_path.mkdir(exist_ok=True)

            try:
                frame_index = 0
                for data_tuple in data_loader:
                    obs = data_tuple[0]
                    
                    fn = f'{frame_index:06d}.png'
                    frame_index += 1
                    fn = folder_path / fn

                    if frames_to_skip <= 0:
                        img = obs['pov']    # ndarray, 64x64x3
                        img = Image.fromarray(img).save(fn)
                        total_frames += 1
                        #rel_path = fn.relative_to(output_base_path)
                        #file_names.append(str(rel_path))
                        file_names.append(fn)
                        frames_to_skip = random.randint(0, max_skip_frames-1)
                    else:
                        frames_to_skip -= 1
            except KeyboardInterrupt:
                raise
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                pass

            traj_index += 1
            print(f'env: {env_index}/{len(environment_names)}; traj: {traj_index}; total_frames: {total_frames}')

    torch.save(file_names, output_base_path / 'file_list.pth')
    

if __name__ == '__main__':
    #main()
    sample_batch()
