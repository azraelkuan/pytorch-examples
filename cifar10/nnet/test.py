import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from skimage import io
from skimage.transform import resize
from tqdm import tqdm



TEST_DATA_ROOT = "../data/testData/"
use_cuda = torch.cuda.is_available()


def load_data():
    file_paths = []
    file_names = os.listdir(TEST_DATA_ROOT)
    for file_name in file_names:
        file_path = os.path.join(TEST_DATA_ROOT, file_name)
        file_paths.append(file_path)
    return file_paths


class ImageData(Dataset):

    def __init__(self, file_paths, transform):
        self.transform = transform
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = io.imread(self.file_paths[idx])
        # image = resize(image, (32, 32), mode='reflect')
        file_name = self.file_paths[idx].split('/')[-1]
        if self.transform:
            image = self.transform(image)
        return image, file_name


def test():
    test_txt = open('test.txt', 'w', encoding='utf-8')

    file_paths = load_data()
    image_dataset = ImageData(file_paths, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    image_dataloader = DataLoader(image_dataset, num_workers=1, batch_size=1, shuffle=False)

    model = torch.load('pkls/resnet50.pkl')
    net = model['model']
    acc = model['acc']
    print("acc: {}".format(acc))
    if use_cuda:
        net = net.cuda()

    for data, file_name in tqdm(image_dataloader):
        data = Variable(data)
        if use_cuda:
            data = data.cuda()
        predict = net(data)
        _, index = torch.max(predict.cpu().data, 1)
        test_txt.write("{} {}\n".format(file_name[0], index[0]))


if __name__ == '__main__':
    test()





