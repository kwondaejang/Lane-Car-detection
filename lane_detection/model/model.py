
import torch.nn as nn
import torch
import torch.nn.functional as F

# seg net implementation
class SegNet(nn.Module):

	def __init__(self, BN_momentum=.1):
		super(SegNet, self).__init__()

		self.in_chn = 3
		self.out_chn = 1

		# Max Pooling layers
		self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)
		self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

		# Encoding COnvolution layers
		self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
		self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

		# Encoding batch norm layer
		self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)

		# decoding conv layer
		self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
		self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)

		# decoding batchnrom layers
		self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
		self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
		self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
		self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
		self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

	def forward(self, x):

		#Stage 1
		x = F.relu(self.BNEn11(self.ConvEn11(x))) 
		x = F.relu(self.BNEn12(self.ConvEn12(x))) 
		x, ind1 = self.MaxEn(x)
		size1 = x.size()

		#Stage 2
		x = F.relu(self.BNEn21(self.ConvEn21(x))) 
		x = F.relu(self.BNEn22(self.ConvEn22(x))) 
		x, ind2 = self.MaxEn(x)
		size2 = x.size()

		#Stage 3
		x = F.relu(self.BNEn31(self.ConvEn31(x))) 
		x = F.relu(self.BNEn32(self.ConvEn32(x))) 
		x = F.relu(self.BNEn33(self.ConvEn33(x))) 	
		x, ind3 = self.MaxEn(x)
		size3 = x.size()

		#Stage 4
		x = F.relu(self.BNEn41(self.ConvEn41(x))) 
		x = F.relu(self.BNEn42(self.ConvEn42(x))) 
		x = F.relu(self.BNEn43(self.ConvEn43(x))) 	
		x, ind4 = self.MaxEn(x)
		size4 = x.size()

		#Stage 5
		x = F.relu(self.BNEn51(self.ConvEn51(x))) 
		x = F.relu(self.BNEn52(self.ConvEn52(x))) 
		x = F.relu(self.BNEn53(self.ConvEn53(x))) 	
		x, ind5 = self.MaxEn(x)

		#Stage 5
		x = self.MaxDe(x, ind5, output_size=size4)
		x = F.relu(self.BNDe53(self.ConvDe53(x)))
		x = F.relu(self.BNDe52(self.ConvDe52(x)))
		x = F.relu(self.BNDe51(self.ConvDe51(x)))

		#Stage 4
		x = self.MaxDe(x, ind4, output_size=size3)
		x = F.relu(self.BNDe43(self.ConvDe43(x)))
		x = F.relu(self.BNDe42(self.ConvDe42(x)))
		x = F.relu(self.BNDe41(self.ConvDe41(x)))

		#Stage 3
		x = self.MaxDe(x, ind3, output_size=size2)
		x = F.relu(self.BNDe33(self.ConvDe33(x)))
		x = F.relu(self.BNDe32(self.ConvDe32(x)))
		x = F.relu(self.BNDe31(self.ConvDe31(x)))

		#Stage 2
		x = self.MaxDe(x, ind2, output_size=size1)
		x = F.relu(self.BNDe22(self.ConvDe22(x)))
		x = F.relu(self.BNDe21(self.ConvDe21(x)))

		#Stage 1
		x = self.MaxDe(x, ind1)
		x = F.relu(self.BNDe12(self.ConvDe12(x)))
		x = self.ConvDe11(x)

		return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features = [64, 128, 256, 512]):
        super(UNET,self).__init__()
        self.up_sample = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down Sample Build Arch
        for feature in features:
            self.down_sample.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up Sample build arch
        for feature in reversed(features):
            self.up_sample.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_sample.append(DoubleConv(feature*2, feature))

        # Bottom layer
        self.bottom = DoubleConv(features[-1], features[-1] * 2)

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
		# buiild forward pass
        skip_connections = []
        for down in self.down_sample:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.up_sample) ,2 ):
            x = self.up_sample[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_sample[idx + 1](concat_skip)

        return (self.final_conv(x))


def test_unet():
    x = torch.randn((3,3,256,512))
    model = UNET(in_channels=3, out_channels=2)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


