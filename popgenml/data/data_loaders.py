import h5py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator

class OnTheFlySimulationDataset(IterableDataset):
    """
    An IterableDataset that continuously yields simulated data.
    """
    def __init__(self, simulator, parse_fn, num_simulations=None):
        self.simulator = simulator
        self.parse_fn = parse_fn
        # If None, the dataset generates an infinite stream of simulations
        self.num_simulations = num_simulations 

    def __iter__(self):
        # 1. Handle Multiprocessing RNG safely
        worker_info = get_worker_info()
        if worker_info is not None:
            # We are in a worker process. Generate a unique seed for this worker
            # using PyTorch's initial seed (which PyTorch ensures is unique per worker).
            worker_seed = torch.initial_seed() % (2**32)
            
            # Note: You need to implement a way to pass this seed to your simulator.
            # Example: self.simulator.set_seed(worker_seed) 
            # Or if it uses numpy under the hood: np.random.seed(worker_seed)
            if hasattr(self.simulator, 'set_seed'):
                self.simulator.set_seed(worker_seed)

        sims_run = 0
        while True:
            # Stop if we hit the requested number of simulations per epoch
            if self.num_simulations is not None and sims_run >= self.num_simulations:
                break
                
            # 2. Run the simulation
            raw_result = self.simulator.simulate()
            
            # 3. Parse the result using your custom function
            parsed_data = self.parse_fn(raw_result)
            
            # Yield the dictionary (or tensor) to the DataLoader for batching
            yield parsed_data
            sims_run += 1


class SimulatorDataLoader(DataLoader):
    """
    A wrapper to directly accept your simulator and parser function.
    """
    def __init__(self, simulator, parse_fn, batch_size=32, num_simulations=None, **kwargs):
        dataset = OnTheFlySimulationDataset(
            simulator=simulator, 
            parse_fn=parse_fn,
            num_simulations=num_simulations
        )
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            **kwargs
        )

class HDF5ChunkDataset(Dataset):
    """
    Reads individual chunks from an HDF5 file. 
    Handles lazy-loading to ensure compatibility with DataLoader multiprocessing.
    """
    def __init__(self, file_path, split='train'):
        self.file_path = file_path
        self.split = split
        self.h5_file = None
        
        # Briefly open the file on the main process to get the list of chunk names
        with h5py.File(self.file_path, 'r') as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not found in HDF5 file.")
            
            # Assuming the structure is f['train']['chunk_0'], f['train']['chunk_1'], etc.
            self.chunk_keys = list(f[self.split].keys())

    def __len__(self):
        return len(self.chunk_keys)

    def __getitem__(self, idx):
        # Lazy initialization: open the HDF5 file only when the first item is requested.
        # This prevents PyTorch multiprocessing errors when num_workers > 0.
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
            
        chunk_name = self.chunk_keys[idx]
        chunk_group = self.h5_file[self.split][chunk_name]
        
        chunk_data = {}
        # Dynamically grab 'x', 'y', and any other keys present in the chunk
        for key in chunk_group.keys():
            # Read from HDF5 and convert to PyTorch tensor
            chunk_data[key] = torch.from_numpy(chunk_group[key][:])
            
        return chunk_data

def flatten_chunk_collate_fn(batch):
    """
    By default, DataLoader would stack n_chunks into shape (n_chunks, chunk_size, ...).
    This custom collate function concatenates them along the 0th dimension 
    so the final batch shape is (n_chunks * chunk_size, ...).
    """
    collated_batch = {}
    
    # Grab the keys from the first chunk in the batch ('x', 'y', etc.)
    keys = batch[0].keys()
    
    for key in keys:
        # Concatenate all chunks for this specific key
        collated_batch[key] = torch.cat([chunk[key] for chunk in batch], dim=0)
        
    return collated_batch

class HDF5ChunkLoader(DataLoader):
    """
    A DataLoader subclass that automatically binds the dataset and the collate function.
    """
    def __init__(self, file_path, split='train', n_chunks=4, **kwargs):
        dataset = HDF5ChunkDataset(file_path, split=split)
        
        # Pass the dataset, n_chunks (as batch_size), and our custom collate function
        # to the parent DataLoader class.
        super().__init__(
            dataset=dataset,
            batch_size=n_chunks,
            collate_fn=flatten_chunk_collate_fn,
            **kwargs
        )
        
        