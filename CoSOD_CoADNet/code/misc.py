from CoSOD_CoADNet.code.common_packages import *


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N - len(num_str)) * '0' + num_str


def min_max_normalize(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed


def visualize_image_tensor(image_tensor):
    # image_tensor: [3, H, W]
    # image_pil: PIL.Image object
    unloader = transforms.ToPILImage()
    image_tensor = image_tensor.cpu()
    image_pil = unloader(image_tensor)
    return image_pil


def visualize_label_tensor(label_tensor):
    # label_tensor: [1, H, W]
    # label_pil: PIL.Image object
    label_numpy = label_tensor.squeeze(0).cpu().data.numpy() # (H, W)
    label_numpy = min_max_normalize(label_numpy) * 255
    label_pil = Image.fromarray(label_numpy).convert('L')
    return label_pil


def binarize_label(label_tensor, threshold):
    assert label_tensor.min()>=0 and label_tensor.max()<=1
    assert threshold>=0 and threshold<=1
    label_tensor[label_tensor < threshold] = 0
    label_tensor[label_tensor >= threshold] = 1
    return label_tensor


def random_crop_tensor(x, cropped_h, cropped_w):
    # x: [in_channels, input_h, input_w]
    # y: [in_channels, cropped_h, cropped_w]
    in_channels, input_h, input_w = x.size()
    assert input_h>cropped_h and input_w>cropped_w
    h_start = np.random.randint(0, (input_h-cropped_h))
    w_start = np.random.randint(0, (input_w-cropped_w))
    y = x[:, h_start:(h_start+cropped_h), w_start:(w_start+cropped_w)] # [in_channels, cropped_h, cropped_w]
    return y


def random_crop_image_label(image, label, expansion_ratio):
    # image & label: PIL.Image
    # image_cropped, label_cropped: torch.tensor, [3, height, width] & [1, height, width]
    assert image.height==label.height and image.width==label.width
    assert expansion_ratio > 0
    height, width = image.height, image.width
    exp_height, exp_width = int(height*(1+expansion_ratio)), int(width*(1+expansion_ratio))
    to_tensor = transforms.ToTensor()
    image = to_tensor(image.resize((exp_height, exp_width))) # [3, exp_height, exp_width]
    label = to_tensor(label.resize((exp_height, exp_width))) # [1, exp_height, exp_width]
    image_concat_label = torch.cat((image, label), dim=0) # [4, exp_height, exp_width]
    image_concat_label_cropped = random_crop_tensor(image_concat_label, height, width) # [4, height, width]
    image_cropped = image_concat_label_cropped[0:3, :, :] # [3, height, width]
    label_cropped = image_concat_label_cropped[3:4, :, :] # [1, height, width]
    label_cropped = binarize_label(label_cropped, 0.50) # [1, height, width]
    return image_cropped, label_cropped


# def read_image(path_list_every, return_list):
#     for load_path in path_list_every:
#         return_list.append(Image.open(load_path))

        
# def image_loader_multi_processing(path_list, num_cores):
#     num_total = len(path_list)
#     num_every = int(num_total / num_cores) + 1
#     path_list_every = []
#     for index in range(num_cores):
#         index_start = index * num_every
#         index_end = (index + 1) * num_every
#         path_list_every.append(path_list[index_start:index_end])
#     manager = Manager()
#     return_list = manager.list()
#     jobs = []
#     for i in range(num_cores):
#         prcs = Process(target=read_image, args=(path_list_every[i], return_list))
#         jobs.append(prcs)
#         prcs.start()
#     for jb in jobs:
#         jb.join()
#     return return_list


def load_image_as_tensor(image_full_path):
    loader = transforms.ToTensor()
    image_tensor = loader(Image.open(image_full_path)) # [C, H, W], [0, 1]
    image_tensor_batch = image_tensor.unsqueeze(0) # add batch dimension, [1, C, H, W]
    return image_tensor_batch


def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y


def convert2img(x):
    return Image.fromarray(x*255).convert('L')


def save_smap(smap, path, negative_threshold=0.0):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalize(unload(smap)))
    smap.save(path)
    

def cache_model(model, path, multi_gpu):
    if multi_gpu:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    
    