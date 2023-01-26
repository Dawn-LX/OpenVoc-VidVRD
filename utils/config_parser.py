import os.path as osp
import tempfile
import shutil
import sys
from importlib import import_module

def parse_config_py(filename):
    """
    This function is borrowed from Open-MMLab, with some modification. Because we don't need the whole mmcv package.
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py#L121
    """
    filename = osp.abspath(osp.expanduser(filename))
    assert filename.endswith('.py')
    cfg_dict = {}
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix='.py')
        temp_config_name = osp.basename(temp_config_file.name)
        shutil.copyfile(filename,
                        osp.join(temp_config_dir, temp_config_name))
        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        # delete imported module
        del sys.modules[temp_module_name]
        # close temp file
        temp_config_file.close()
    return cfg_dict

# TODO add more format