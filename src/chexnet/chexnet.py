import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # embeddings
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


model = DenseNet121(N_CLASSES).cuda()

modelCheckpoint = torch.load(CKPT_PATH)
pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$' )
state_dict = modelCheckpoint['state_dict']
for key in list(state_dict.keys()):

    res = pattern.match(key)

    if res:
        new_key = (res.group(1) + res.group(2))[7:]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
    else:
        new_key = key[7:]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
model.load_state_dict(state_dict, strict=False)
