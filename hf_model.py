class LFS:

    def __init__(self, oid, size, pointerSize):
        self.oid = oid
        self.size = size
        self.pointerSize = pointerSize


class ModelFileInfo:

    def __init__(self, file_type, oid, size, path, lfs):
        self.type = file_type
        self.oid = oid
        self.size = size
        self.path = path
        self.lfs = lfs
        self.is_dir = False
        self.is_lfs = False
        self.append_path = None
        self.skip_download = False
        self.filter_skip = False
        self.download_link = None

    def __str__(self):
        return f'{self.type} {self.oid} {self.size} {self.path} {self.download_link}'
