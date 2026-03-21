import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torchvision.models as models


class InceptionMixed6e(nn.Module):
    def __init__(self):
        super().__init__()
        #On charge le modèle avec les poids officiels
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval() #Mode évaluation (important pour Dropout/BatchNorm)
        
        #On regroupe les couches dans l'ordre exact jusqu'à Mixed_6e
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        )
        
        #On gèle les poids puisque le modèle utilise des poids préentrainés
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        #on passe l'image 256x256 qui sortira en 14x14 et 768 canaux
        return self.features(x)

class FiLMGenerator(nn.Module):
    def __init__(self):
        super(FiLMGenerator, self).__init__()
        self.inception = InceptionMixed6e()
        self.bottleneck = nn.Sequential(
            nn.Linear(768, 100),    #On compresse (Bottleneck)
            nn.ReLU(),
            nn.Linear(100, 2758)    #On déploie vers les gammas/betas
        )

    def forward(self, x):
        parms = self.inception(x) #(B, 768, 14, 14)
        parms = torch.mean(parms, dim=[2, 3], keepdim=True) #Global Average Pooling (B, 768, 1, 1)
        parms = self.bottleneck(parms.squeeze()) #On enlève les dimensions 1,1 pour le Linear (B, 768) -> (B, 2758)
        return parms

class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        self.FiLMGenerator = FiLMGenerator()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.res1conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res1conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res2conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res2conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res3conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res3conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res4conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res4conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.res5conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.res5conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convUpsample1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convUpsample2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same')
        
        self.convfinal = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding='same')

    def apply_film(self, x, gamma, beta):
            #Normalisation d'instance
            eps = 1e-5
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + eps)
            return gamma * x_norm + beta
    

    def forward(self, x, vecteur_style):
        params       = self.FiLMGenerator(vecteur_style).unsqueeze(2).unsqueeze(3)  #(B, 2C, 1, 1)
        #gamma, beta  = params.chunk(2, dim=1)
        
        #On crée une fonction qui "consomme" le vecteur au fur et à mesure
        pointer = 0
        
        def get_next_params(num_channels, current_pointer):
            #On prend 2 * num_channels (moitié gamma, moitié beta)
            start = current_pointer
            mid = start + num_channels
            end = mid + num_channels
            
            gamma = params[:,start:mid,:,:] #On prend les gammas
            beta = params[:,mid:end,:,:]   #On prend les betas
            return gamma, beta, end

        #Premieres convolutions
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        #ResBlocks

        #Block1
        res = x

        x = self.res1conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res1conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block2
        res = x

        x = self.res2conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res2conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block3
        res = x

        x = self.res3conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x =torch.relu(x)

        x = self.res3conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block4
        res = x

        x = self.res4conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res4conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Block5
        res = x

        x = self.res5conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.res5conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta)
        x = x + res

        #Upsampling
        x = self.upsample1(x)
        x = self.convUpsample1(x)
        gamma, beta, pointer = get_next_params(64, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.upsample2(x)
        x = self.convUpsample2(x)
        gamma, beta, pointer = get_next_params(32, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.relu(x)

        x = self.convfinal(x)
        gamma, beta, pointer = get_next_params(3, pointer)
        x = self.apply_film(x, gamma, beta)
        x = torch.sigmoid(x)

        return x


class VGGExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # On charge VGG16 préentraîné
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg[:23] #On s'arrête après relu4_2
        
        #On fige les poids puisque on veut pas l'entrainer
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
            
        #indices des couches auxquelles on veut acceder
        self.style_idx = ['3', '8', '15', '22']
        self.content_idx = ['22'] #relu4_2

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_idx or name in self.content_idx:
                features[name] = x
        return features
    
def get_gram_matrix(tensor):
    #Batch, Canaux, Hauteur, Largeur
    b, c, h, w = tensor.size()

    #On aplatit les dimensions spatiales (Hauteur et Largeur)
    #(b, c, h, w) à (b, c, h * w)
    features = tensor.view(b, c, h * w)

    #On multiplie la matrice par sa transposée
    #torch.bmm = Batch Matrix Multiplication
    gram = torch.bmm(features, features.transpose(1, 2))

    #On normalise
    return gram / (c * h * w)



def compute_loss(vgg, gen_img, content_img, style_img):
    #Extraire les features pour les 3 images
    gen_feats = vgg(gen_img)
    cont_feats = vgg(content_img)
    style_feats = vgg(style_img)
    
    mse = nn.MSELoss()
    
    #LOSS DE CONTENU
    #On compare relu4_2 de l'image générée et de l'image de contenu
    content_loss = mse(gen_feats['22'], cont_feats['22'])
    
    #LOSS DE STYLE
    style_loss = 0
    style_layers_idx = ['3', '8', '15']
    
    #On parcours la liste d'indice et on extrait les matrices de gram et on calcule la MSE
    for idx in style_layers_idx:
        gen_gram = get_gram_matrix(gen_feats[idx])
        style_gram = get_gram_matrix(style_feats[idx])
        
        style_loss += mse(gen_gram, style_gram)
        

    style_loss /= len(style_layers_idx)
    
    return content_loss, style_loss
