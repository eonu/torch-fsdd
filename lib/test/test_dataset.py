import os, shutil, pytest, glob, torch
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose
from torchfsdd import TorchFSDD, TorchFSDDGenerator, TrimSilence

def fetch_rec_num(file):
    name = os.path.splitext(os.path.basename(file))[0]
    rec_num = int(name.split('_')[-1])
    return rec_num

def test_generator_local():
    path = 'lib/test/data/v1.0.10'
    fsdd = TorchFSDDGenerator(version='local', path=path)
    with open('lib/test/filenames/v1.0.10') as f:
        true_files = [l.rstrip() for l in f.readlines()]
    read_files = [os.path.basename(file) for file in fsdd.all_files]
    assert len(true_files) == len(read_files) and sorted(true_files) == sorted(read_files)
    assert fsdd.path == path

def test_generator_repo_master():
    path = 'lib/test/data/recordings'
    try:
        fsdd = TorchFSDDGenerator(version='master', path='lib/test/data')
        assert len(fsdd.all_files) > 0
        assert fsdd.path == path
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path)

def test_generator_repo_tag1():
    path = 'lib/test/data/recordings'
    try:
        fsdd = TorchFSDDGenerator(version='v1.0.3', path='lib/test/data')
        with open('lib/test/filenames/v1.0.3') as f:
            true_files = [l.rstrip() for l in f.readlines()]
        read_files = [os.path.basename(file) for file in fsdd.all_files]
        assert len(true_files) == len(read_files) and sorted(true_files) == sorted(read_files)
        assert fsdd.path == path
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path)

def test_generator_repo_tag2():
    path = 'lib/test/data/recordings'
    try:
        fsdd = TorchFSDDGenerator(version='v1.0.10', path='lib/test/data')
        with open('lib/test/filenames/v1.0.10') as f:
            true_files = [l.rstrip() for l in f.readlines()]
        read_files = [os.path.basename(file) for file in fsdd.all_files]
        assert len(true_files) == len(read_files) and sorted(true_files) == sorted(read_files)
        assert fsdd.path == path
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path)

def test_generator_full():
    path = 'lib/test/data/v1.0.10'
    fsdd = TorchFSDDGenerator(version='local', path=path)
    full = fsdd.full()
    with open('lib/test/filenames/v1.0.10') as f:
        true_files = [l.rstrip() for l in f.readlines()]
    read_files = [os.path.basename(file) for file in full.files]
    assert isinstance(full, TorchFSDD)
    assert len(true_files) == len(read_files) and sorted(true_files) == sorted(read_files)
    assert fsdd.path == path

def test_generator_train_test_split_99_1():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.01)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(50))
    assert len(train) == 3000

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set()
    assert len(test) == 0

def test_generator_train_test_split_95_5():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.05)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(2, 50))
    assert len(train) == 2880

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(2))
    assert len(test) == 120

def test_generator_train_test_split_90_10():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.1)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(5, 50))
    assert len(train) == 2700

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(5))
    assert len(test) == 300

def test_generator_train_test_split_50_50():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.5)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(25, 50))
    assert len(train) == 1500

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(0, 25))
    assert len(test) == 1500

def test_generator_train_test_split_10_90():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.9)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(45, 50))
    assert len(train) == 300

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(45))
    assert len(test) == 2700

def test_generator_train_test_split_5_95():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.95)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(47, 50))
    assert len(train) == 180

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(47))
    assert len(test) == 2820

def test_generator_train_test_split_1_99():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, test = fsdd.train_test_split(test_size=0.99)
    assert isinstance(train, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(49, 50))
    assert len(train) == 60

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(49))
    assert len(test) == 2940

def test_generator_train_val_test_split_98_1_1():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, val, test = fsdd.train_val_test_split(test_size=0.01, val_size=0.01)
    assert isinstance(train, TorchFSDD)
    assert isinstance(val, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(50))
    assert len(train) == 3000

    val_nums = {fetch_rec_num(file) for file in val.files}
    assert val_nums == set()
    assert len(val) == 0

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set()
    assert len(test) == 0

def test_generator_train_val_test_split_95_2p5_2p5():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, val, test = fsdd.train_val_test_split(test_size=0.025, val_size=0.025)
    assert isinstance(train, TorchFSDD)
    assert isinstance(val, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(2, 50))
    assert len(train) == 2880

    val_nums = {fetch_rec_num(file) for file in val.files}
    assert val_nums == set(range(1, 2))
    assert len(val) == 60

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(1))
    assert len(test) == 60

def test_generator_train_val_test_split_50_33_17():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, val, test = fsdd.train_val_test_split(test_size=0.17, val_size=0.33)
    assert isinstance(train, TorchFSDD)
    assert isinstance(val, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(24, 50))
    assert len(train) == 1560

    val_nums = {fetch_rec_num(file) for file in val.files}
    assert val_nums == set(range(8, 24))
    assert len(val) == 960

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(8))
    assert len(test) == 480

def test_generator_train_val_test_split_17_50_33():
    fsdd = TorchFSDDGenerator(version='local', path='lib/test/data/v1.0.10')

    train, val, test = fsdd.train_val_test_split(test_size=0.33, val_size=0.50)
    assert isinstance(train, TorchFSDD)
    assert isinstance(val, TorchFSDD)
    assert isinstance(test, TorchFSDD)

    train_nums = {fetch_rec_num(file) for file in train.files}
    assert train_nums == set(range(41, 50))
    assert len(train) == 540

    val_nums = {fetch_rec_num(file) for file in val.files}
    assert val_nums == set(range(16, 41))
    assert len(val) == 1500

    test_nums = {fetch_rec_num(file) for file in test.files}
    assert test_nums == set(range(16))
    assert len(test) == 960

def test_dataset_no_transforms():
    """Note: This test may fail if not on Linux."""
    fsdd = TorchFSDD(glob.glob('lib/test/data/v1.0.10/*.wav'))
    x, y = fsdd[0]
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 1
    assert x.min() >= -1
    assert x.max() <= 1
    assert y == 7

def test_dataset_transforms_single():
    """Note: This test may fail if not on Linux."""
    files = glob.glob('lib/test/data/v1.0.10/*.wav')
    x_original, y = TorchFSDD(files)[0]
    x_trans, _ = TorchFSDD(files, transforms=TrimSilence(threshold=0.1))[0]
    assert isinstance(x_trans, torch.Tensor)
    assert x_trans.ndim == 1
    assert x_trans.min() >= -1
    assert x_trans.max() <= 1
    assert len(x_original) != len(x_trans)
    assert y == 7

def test_dataset_transforms_multiple():
    """Note: This test may fail if not on Linux."""
    n_mfcc = 13
    files = glob.glob('lib/test/data/v1.0.10/*.wav')
    x_original, y = TorchFSDD(files)[0]
    x_trans, _ = TorchFSDD(files, transforms=Compose([
        TrimSilence(threshold=0.05),
        MFCC(sample_rate=8e3, n_mfcc=n_mfcc)
    ]))[0]
    assert isinstance(x_trans, torch.Tensor)
    assert x_trans.ndim == 2
    assert x_trans.shape == (n_mfcc, 9)
    assert y == 7

def test_dataset_load_all_false():
    files = glob.glob('lib/test/data/v1.0.10/*.wav')
    fsdd = TorchFSDD(files, load_all=False)
    assert not hasattr(fsdd, 'recordings')
    assert not hasattr(fsdd, 'labels')

def test_dataset_load_all_true():
    files = glob.glob('lib/test/data/v1.0.10/*.wav')
    fsdd = TorchFSDD(files, load_all=True)
    assert hasattr(fsdd, 'recordings')
    assert isinstance(fsdd.recordings, list)
    assert isinstance(fsdd.recordings[0], torch.Tensor)
    assert len(fsdd.recordings) == len(files)
    assert hasattr(fsdd, 'labels')
    assert isinstance(fsdd.labels, list)
    assert isinstance(fsdd.labels[0], int)
    assert len(fsdd.labels) == len(files)