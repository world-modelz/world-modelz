
import threading
import random
import numpy as np
import torch
import minerl


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
        self.example_index = 0
        self.fill_thread = None

        self.start_fill_buffer()

    def fill_buffer_thread(self):
        
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

            #print('#frames:', len(frames))

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

            #print(f'total_frames: {total_frames}; examples: {len(example_offsets)}')

        self.next_segments = segments
        p = np.random.permutation(len(example_offsets))
        self.next_example_offsets = [example_offsets[k] for k in p]

    def start_fill_buffer(self):
        if self.fill_thread is None:
            self.fill_thread = threading.Thread(target=self.fill_buffer_thread, daemon=True)
            self.fill_thread.start()

    def wait_for_next_buffer(self):
        self.fill_thread.join()
        self.segments = self.next_segments
        self.example_offsets = self.next_example_offsets
        self.example_index = 0

        self.fill_thread = None
        self.start_fill_buffer()

    def sample_batch(self, batch_size):
        l = self.traj_len

        batch_shape = (batch_size, l) + self.frame_shape
        batch = np.ndarray(batch_shape, dtype=np.uint8)
        
        for i in range(batch_size):
            if self.example_index >= len(self.example_offsets):
                self.wait_for_next_buffer()
            
            segment, offset = self.example_offsets[self.example_index]
            self.example_index += 1
            batch[i] = self.segments[segment][offset:offset+l]
        
        return batch
