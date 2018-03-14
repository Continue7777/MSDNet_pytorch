import math
import torch
import torch.nn as nn
'''
question:
1.the convs dont't care about the pad,stride and so on.
2.no transition and lazy evalution
3.1*1 convs don't do batch
'''


class MSDFirstLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_nums):
        """
        Creates the first layer of the MSD network, which takes
        an input tensor (image) and generates a list of size num_scales
        with deeper features with smaller (spatial) dimensions.
        :param in_channels: number of input channels to the first layer
        :param out_channels: number of output channels in the first scale
        :param num_scales: number of output scales in the first layer
        :param args: other arguments
        :return: a list of Variable
        """
        super(MSDFirstLayer, self).__init__()

        # Init params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_nums = scale_nums
        self.first_layer = self.create_modules()

    def create_modules(self):
        in_channels = int(self.in_channels)
        out_channels = int(self.out_channels)
        # create first scale feature maps
        first_layer = nn.ModuleList()

        #scale1 conv3*3 -> bn -> ReLU
        first_scale = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        first_layer.append(first_scale)

        #use a strided convolution to create next scale features
        for s in range(1,self.scale_nums):
            in_channels = out_channels
            out_channels = out_channels *2
    
            sub_scale = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
                )

            first_layer.append(sub_scale)

        return first_layer

    def forward(self,x):
        output = [None] * self.scale_nums
        current_input = x
        for s in range(0,self.scale_nums):
            output[s] = self.first_layer[s](current_input)
            current_input = output[s]
        return output

class MSD_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels,stride=1):
        super(MSD_conv,self).__init__()
        self.add_module('conv1*1',nn.Conv2d(in_channels,out_channels,1,stride=stride))
        self.add_module('bn1',nn.BatchNorm2d(out_channels))
        self.add_module('relu1',nn.ReLU())
        
        self.add_module('conv3*3',nn.Conv2d(out_channels,out_channels,3,padding=1))
        self.add_module('bn2',nn.BatchNorm2d(out_channels))
        self.add_module('relu2',nn.ReLU())

class MSDLayer_inblock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_nums):
        """
        Creates a regular/transition MSDLayer. this layer uses DenseNet like concatenation on each scale,
        and performs spatial reduction between scales. if input and output scales are different, than this
        class creates a transition layer and the first layer (with the largest spatial size) is dropped.
        :param current_channels: number of input channels
        :param args: other arguments
        """
        super(MSDLayer_inblock, self).__init__()

        # Init vars
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_nums = scale_nums

        self.sub_layers = self.create_modules()

    def create_modules(self):
        """
        Builds the different scales of the MSD network layer.
        :return: A list of scale modules
        """
        in_channels = self.in_channels
        out_channels = self.out_channels
        sub_layers = nn.ModuleList()

        #scale 2-s conv1*1 -> bn -> ReLu -> conv3*3 -> bn -> ReLU
        first_scale = MSD_conv(in_channels,out_channels)
        sub_layers.append(first_scale)

        for s in range(1,self.scale_nums):
            in_channels = in_channels * 2
            out_channels = out_channels * 2
            horizontal_seq = MSD_conv(in_channels,int(out_channels/2),stride=1)
            diagonal_seq = MSD_conv(int(in_channels/2),int(out_channels/2),stride=2)
            sub_layers.append(horizontal_seq)
            sub_layers.append(diagonal_seq)


        return sub_layers
    
    def forward(self,x):
        #here need to attention that the output for classifier should be parse.its a little diff with densenet.
        out = [None] * self.scale_nums
        out[0] = self.sub_layers[0](x[0])
        

        for s in range(1,self.scale_nums):
            out_horizontal  = self.sub_layers[s*2-1](x[s])
            out_diagonal = self.sub_layers[s*2](x[s-1])
            out[s] = torch.cat([out_horizontal, out_diagonal], 1)

        for s in range(0,self.scale_nums):
            out[s] = torch.cat([x[s], out[s]], 1)
        return out


class MSDLayers_block(nn.Module):
    def __init__(self,in_channels,out_channels,layer_nums,scale_nums,grow_rate):
        super(MSDLayers_block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_nums = layer_nums
        self.scale_nums = scale_nums
        self.grow_rate = grow_rate

        self.msdlayers_block = nn.Sequential()
        for i in range(0,layer_nums):
            current_channels = in_channels + i*grow_rate
            self.msdlayers_block.add_module("msd_layer_"+str(i),
                MSDLayer_inblock(
                    current_channels,
                    out_channels,
                    scale_nums
                ))


class Transition(nn.Module):
    def __init__(self, input_feature_nums, output_feature_nums,input_scale_nums,output_scale_nums):
        super(Transition,self).__init__()

        self.input_scale_nums = input_scale_nums
        self.output_scale_nums = output_scale_nums
        self.input_feature_nums = input_feature_nums
        self.output_feature_nums = output_feature_nums

        # Define a parallel stream for the different scales
        self.scales_module = nn.ModuleList()

        start_scale = self.input_scale_nums - self.output_scale_nums
        for s in range(start_scale,self.output_scale_nums+1):
            in_channels = input_feature_nums * 2 ** (s)
            out_channels = output_feature_nums * 2 ** (s)
            self.scales_module.append(self.conv1x1(in_channels, out_channels))


    def conv1x1(self, in_channels, out_channels):
        """
        Inner function to define the basic operation
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :return: A Sequential module to perform 1x1 convolution
        """
        scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,stride=2)
        )

        return scale

    def forward(self,x):
        output = []
        start_scale = self.input_scale_nums - self.output_scale_nums
        for s in range(start_scale,self.output_scale_nums+1):
            output.append(self.scales_module[s-start_scale](x[s]))

        return output



class CifarClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.
        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """

        super(CifarClassifier, self).__init__()
        self.inner_channels = 128

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Linear(self.inner_channels, num_classes)

    def forward(self, x):
        """
        Drive features to classification.
        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), self.inner_channels)
        x = self.classifier(x)
        return x

class MSDNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=6,layer_nums_list=[4,3],grow_rate_list=[6,12],scale_nums_list=[3,2]):
        super(MSDNet,self).__init__()
        self.msdfirst_layer = MSDFirstLayer(in_channels,out_channels,scale_nums_list[0])

        current_channels = out_channels
        self.msd_block1 = MSDLayers_block(in_channels=current_channels,out_channels=current_channels,
                                            layer_nums=layer_nums_list[0],scale_nums=scale_nums_list[0],grow_rate=grow_rate_list[0]).msdlayers_block

        current_channels = current_channels + grow_rate_list[0] * layer_nums_list[0]
        self.msd_transition = Transition(input_feature_nums=current_channels,output_feature_nums=out_channels,input_scale_nums=scale_nums_list[0],output_scale_nums=scale_nums_list[1])

        current_channels = out_channels * 2
        self.msd_block2 = MSDLayers_block(in_channels=current_channels,out_channels=current_channels,
                                            layer_nums=layer_nums_list[1],scale_nums=scale_nums_list[1],grow_rate=grow_rate_list[1]).msdlayers_block

        current_channels = out_channels * 4 + grow_rate_list[1] * layer_nums_list[1] * 2 #the third scales's output channels

        self.msd_cls_final = CifarClassifier(current_channels,10)
        print self.msd_cls_final

    def forward(self,x):

        out_layer1 = self.msdfirst_layer(x)

        out_block1 = self.msd_block1(out_layer1)

        out_trans1 = self.msd_transition(out_block1)

        out_block2 = self.msd_block2(out_trans1)

        print out_block2
        out = self.msd_cls_final(out_block2[-1])

        return out

if __name__ == '__main__':
    input = autograd.Variable(torch.rand(1,3,64,64))
    module = MSDNet()
    module(input)   