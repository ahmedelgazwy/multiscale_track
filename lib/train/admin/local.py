import os

class EnvironmentSettings:
    def __init__(self):
        try:
            slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        except:
            slurm_tmpdir = os.environ.get('TMPDIR')
            # if interactive session, source slurm_vars.sh file
            # vars_file = os.path.expanduser('~/slurm_vars.sh')
            # if os.path.exists(vars_file):
            #     subprocess.run(['source', vars_file])
            #     slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
            # else:
            #     raise Exception('SLURM_TMPDIR not found in environment variables or in slurm_vars.sh file.')
            # raise Exception('SLURM_TMPDIR not found')
        
        self.workspace_dir = f'{slurm_tmpdir}/tracking/ODtrack_multiscale/ODTrack'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        self.lasot_dir = self.workspace_dir + '/data/lasot'
        self.got10k_dir = self.workspace_dir + '/data/got-10k/train'
        self.got10k_val_dir = self.workspace_dir + '/data/got-10k/val'
        self.lasot_lmdb_dir = self.workspace_dir + '/data/lasot_lmdb'
        self.got10k_lmdb_dir = self.workspace_dir + '/data/got10k_lmdb'
        self.trackingnet_dir = self.workspace_dir + '/data/trackingnet'
        self.trackingnet_lmdb_dir = self.workspace_dir + '/data/trackingnet_lmdb'
        self.coco_dir = self.workspace_dir + '/data/coco'
        self.coco_lmdb_dir = self.workspace_dir + '/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = self.workspace_dir + '/data/vid'
        self.imagenet_lmdb_dir = self.workspace_dir + '/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''