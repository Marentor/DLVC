from ..dataset import Sample, Subset, ClassificationDataset


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        # See the CIFAR-10 website on how to load the data files
        self.cifar_train_labels = []
        import pickle
        import numpy as np
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        if subset == 1:
            for i in range(1, 5):
                cifar_train_data_dict = unpickle(fdir + "/data_batch_{}".format(i))
                if i == 1:
                    cifar_train_data = cifar_train_data_dict[b'data']
                else:
                    cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
                self.cifar_train_labels += cifar_train_data_dict[b'labels']

        if subset == 2:
            cifar_train_data_dict = unpickle(fdir + "/data_batch_5")
            cifar_train_data = cifar_train_data_dict[b'data']
            self.cifar_train_labels += cifar_train_data_dict[b'labels']

        if subset == 3:
            cifar_train_data_dict = unpickle(fdir + "/test_batch")
            cifar_train_data = cifar_train_data_dict[b'data']
            self.cifar_train_labels += cifar_train_data_dict[b'labels']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
        permute = [2, 1, 0]
        cifar_train_data=cifar_train_data[:,:,:,permute]

        self.cifar_train_labels = np.array(self.cifar_train_labels)
        # Locate position of labels that equal to i
        pos_i = np.argwhere((self.cifar_train_labels == 3) | (self.cifar_train_labels == 5))
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        x_i = np.array([cifar_train_data[j] for j in pos_i])
        y_i = np.array([self.cifar_train_labels[j] for j in pos_i])

        y_i[y_i < 4] = 0
        y_i[y_i > 4] = 1
        self.cifar_train_data = x_i
        self.cifar_train_labels = y_i

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return len(self.cifar_train_labels)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''
        if (idx < 0 or idx > len(self.cifar_train_labels)): raise IndexError

        return Sample(idx, self.cifar_train_data[idx], self.cifar_train_labels[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        import numpy as np

        len(np.unique(self.cifar_train_labels))
