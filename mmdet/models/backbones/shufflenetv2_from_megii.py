from collections import OrderedDict
import torch
import torch.nn as nn

from ..builder import BACKBONES

## ******************************************************************************************************************************************
## ShuffleV2Block

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        # x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        # x = x.permute(1, 0, 2)
        # x = x.reshape(2, -1, num_channels // 2, height, width)
        # return x[0], x[1]

        x = x.reshape(-1, num_channels // 2, 2, height * width)
        x = x.permute(0,2,1,3)
        x0,x1 = torch.split(x,[1,1],dim=1)
        x0 = x0.reshape(-1, num_channels // 2, height, width)
        x1 = x1.reshape(-1, num_channels // 2, height, width)
        return x0,x1

## ShuffleV2Block
## ******************************************************************************************************************************************


## ******************************************************************************************************************************************
## Megii_ShuffleNetV2

@BACKBONES.register_module()
class Megii_ShuffleNetV2(nn.Module):
    def __init__(self,
        model_size='1.5x', 
        out_indexs=(3,11,15), 
        use_last_conv = False,
        ):
        super(Megii_ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
                
        self.features = nn.Sequential(*self.features)

        self.fea_indexs = out_indexs
        assert self.fea_indexs[-1] == len(self.features) -1,"return feas before network last, len feas: {} , curr return : {}".format(len(self.features),self.fea_indexs)
        
        self.use_last_conv = use_last_conv
        if  self.use_last_conv:
            self.conv_last = nn.Sequential(
                nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.stage_out_channels[-1]),
                nn.ReLU(inplace=True)
            )
            self.fea_indexs = self.fea_indexs[:-1]


        # self.globalpool = nn.AvgPool2d(7)
        # if self.model_size == '2.0x':
        #     self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        # self._initialize_weights()

    def forward(self, x):
        # x = self.first_conv(x)
        # x = self.maxpool(x)
        # x = self.features(x)
        # x = self.conv_last(x)
        # x = self.globalpool(x)
        # if self.model_size == '2.0x':
        #     x = self.dropout(x)
        # x = x.contiguous().view(-1, self.stage_out_channels[-1])
        # x = self.classifier(x)
        # return x

        feas = []
        
        x = self.first_conv(x)
        x = self.maxpool(x)

        if -1 in self.fea_indexs:
            feas.append(x)

        for findex, fea_module in enumerate(self.features):
            x = fea_module(x)
            # print(findex,x.shape)
            if findex in self.fea_indexs:
                # print('fee')
                feas.append(x)

        # last fea
        if self.use_last_conv:
            # print('using last conv')
            x = self.conv_last(x)
            # print('fee')
            feas.append(x)

        return feas



    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        
        if isinstance(pretrained, str):
            print('loading weights from {} ...'.format(pretrained))

            state_dict = torch.load(pretrained,map_location='cpu')
            state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict

            last_conv_keys = [
                "conv_last.0.weight", "conv_last.1.weight", "conv_last.1.bias",
                "conv_last.1.running_mean", "conv_last.1.running_var", "conv_last.1.num_batches_tracked",
            ]

            drop_keys = ["classifier.0.weight"]

            if not self.use_last_conv:
                drop_keys += last_conv_keys

            new_state_dict = OrderedDict()
            for k in state_dict:
                new_k = k.replace('module.','')
                new_state_dict[new_k] = state_dict[k]
            for k in drop_keys:
                new_state_dict.pop(k,None)

            self.load_state_dict(new_state_dict)
        else:
            print('do not load weight fro shufflenetv2')
            self._initialize_weights()
            pass

## Megii_ShuffleNetV2
## ******************************************************************************************************************************************


if __name__ == "__main__":
    model = Megii_ShuffleNetV2(
        model_size='1.5x', 
        out_indexs=(3,11,15),
        use_last_conv = False,
    )
    # model = Megii_ShuffleNetV2(
    #     model_size='1.0x', 
    #     out_indexs=(3,11,15),
    #     use_last_conv = False,
    # )
    # model = Megii_ShuffleNetV2(
    #     model_size='0.5x', 
    #     out_indexs=(3,11,15),
    #     use_last_conv = False,
    # )
    # print(model)


    model.init_weights('/mnt/lustre/fuzuoyi/weights/shufflenet_megii/ShuffleNetV2.1.5x.pth.tar')

    test_data = torch.rand(2, 3, 224, 224)
    test_outputs = model(test_data)
    # print(test_outputs.size())
    for o in test_outputs:
        print(o.shape)

