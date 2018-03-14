from msdfirst_layer import *

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

class MSDNet_cifar10(nn.Module):
    def __init__(self,in_channels=3,out_channels=6,layer_nums_list=[4,3],grow_rate_list=[6,12],scale_nums_list=[3,2]):
        super(MSDNet_cifar10,self).__init__()
        self.msdfirst_layer = MSDFirstLayer(in_channels,out_channels,scale_nums_list[0])

        current_channels = out_channels
        self.msd_block1 = MSDLayers_block(in_channels=current_channels,out_channels=current_channels,
                                            layer_nums=layer_nums_list[0],scale_nums=scale_nums_list[0],grow_rate=grow_rate_list[0]).msdlayers_block

        current_channels = (current_channels + grow_rate_list[0] * layer_nums_list[0])*4

        self.msd_cls_final = CifarClassifier(current_channels,10)

    def forward(self,x):

        out_layer1 = self.msdfirst_layer(x)

        out_block1 = self.msd_block1(out_layer1)

    #     out_trans1 = self.msd_transition(out_block1)

    #     out_block2 = self.msd_block2(out_trans1)

        out = self.msd_cls_final(out_block1[-1])

        return out