
#args = configs.configs_DFCN_Certificate_l1_plus_loc_NLayerGAN(args)   #mask 不能按顺序复制。在源数据中，位置不是对应的

"""resize,IMD2020"""
#args = configs.configs_DFCN_IMD2020_resize_GAN_plus_l1_plus_lossf_plus_loc128(args)
"""resize,DEFACTO"""
#args = configs.configs_DFCN_DEFACTO_resize_GAN_plus_l1_plus_lossf_plus_loc128(args)
"""resize,Certificate PS"""
#args = configs.configs_DFCN_Certificate_resize_GAN_plus_l1_plus_lossf_plus_loc128(args)
#args = configs.configs_SCSE_IMD2020_resize_GAN_plus_l1_plus_lossf_plus_loc128(args)
#args = configs.configs_SCSE_DEFACTO_resize_GAN_plus_l1_plus_lossf_plus_loc128(args)
#args = configs.configs_SCSE_DEFACTO_resize_GAN_plus_l1_plus_lossf_plus_loc_cosine(args)
args = configs.configs_DFCN_IMD2020_resize_GAN_plus_l1_plus_lossf_plus_loc_cosine(args)
#args = configs.configs_DFCN_IMD2020_resize_GAN_plus_l1_loc_cosine(args)
create_dir(args.save_model_path)
create_dir(os.path.join(args.save_model_path,"Loc_model"))
create_dir(os.path.join(args.save_model_path,"Restore_model"))
create_dir(os.path.join(args.save_model_path, "Dis_model"))
if (args.train_f==True):
    create_dir(os.path.join(args.save_model_path, "F_model"))
def random_crop(img, mask, crop_shape):
    if (img.shape[0]==crop_shape[0] and img.shape[1]==crop_shape[1]):
        return img,mask
    if img.shape[0] < crop_shape[0] or img.shape[1] < crop_shape[1]:
        img = cv2.resize(img, (crop_shape[1], crop_shape[0]))
        mask = cv2.resize(mask, (crop_shape[1], crop_shape[0]), interpolation=cv2.INTER_NEAREST)

    original_shape = mask.shape
    crop_mask = np.zeros((crop_shape[0],crop_shape[1],3))
    count = 0
    # print(mask.shape,crop_mask.shape)
    start_h = np.random.randint(0, original_shape[0] - crop_shape[0] + 1)
    start_w = np.random.randint(0, original_shape[1] - crop_shape[1] + 1)
    crop_img = img[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1], :]
    crop_mask = mask[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1],:]
    return crop_img, crop_mask
def aug(img, mask, degraded_img,deraded_label):
    H, W, _ = img.shape
    # Flip
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
        degraded_img = cv2.flip(degraded_img, 0)
        deraded_label = cv2.flip(deraded_label, 0)

    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        degraded_img = cv2.flip(degraded_img, 1)
        deraded_label = cv2.flip(deraded_label, 1)

    if random.random() < 0.5:
        tmp = random.random()
        if tmp < 0.33:

            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_90_CLOCKWISE)
            deraded_label = cv2.rotate(deraded_label, cv2.ROTATE_90_CLOCKWISE)
        elif tmp < 0.66:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            deraded_label = cv2.rotate(deraded_label, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            degraded_img = cv2.rotate(degraded_img, cv2.ROTATE_180)
            deraded_label = cv2.rotate(deraded_label, cv2.ROTATE_180)


    return img, mask,degraded_img,deraded_label

def create_file_list(image_path,mask_path,image_file):
    if(image_file!=''):
        with open(image_file,'rb') as f:
            images = pickle.load(f)
    else:
        images = os.listdir(image_path)
    files = []
    random.shuffle(images)
    for image in images:
        if(mask_path!=''):
            mask_name = image.replace('.jpg','.png')
            mask_name = mask_name.replace('rma_2','rma_3')
            mask_name = mask_name.replace('ps_','ms_')
            # if 'IMD' in image_path and 'test' in image_path:
            #     mask_name = image.replace('.png','_mask.png')
            files.append([os.path.join(image_path, image), os.path.join(mask_path, mask_name)])
        else:
            files.append(os.path.join(image_path,image))
    return files

"""" resize"""
class Tampering_Dataset_resize(Dataset):
    def __init__(self, file,choice='train',patch_size=768,with_aug = False,resize_ratio = None):
        self.patch_size = patch_size
        self.choice = choice
        self.filelist = file
        self.with_aug = with_aug
        self.resize_ratio = resize_ratio
    def __getitem__(self, idx):
        return self.load_item(idx)   #到125 load_item()

    def __len__(self):
        return len(self.filelist)


    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.filelist[idx], ''
        img = cv2.imread(fname1)
        # if self.choice == 'val':
        #     print(fname1)

        H, W, _ = img.shape

        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
            # if self.choice == 'val':
            #     print(fname2)
            if(np.max(mask)<=1):
                mask = mask*255
        if self.choice =='train':
            img,mask = random_crop(img, mask,(self.patch_size, self.patch_size)) #先分块再数据增强
            #随机裁剪128 128
            degraded_img,degraded_label = img, mask   #cv2.imdecode(encimg, 1)
            # 获取原始图像的宽度和高度
            h, w = degraded_img.shape[:2]
            resize_ratio =random.uniform(0.7,1.4)
            resize_ratio = round(resize_ratio,1)
            # 计算缩放后的宽度和高度
            new_w = int(w * resize_ratio)
            new_h = int(h * resize_ratio)
            # 缩放图像
            degraded_img = cv2.resize(img, (new_w, new_h))
            degraded_label = cv2.resize(mask, (new_w, new_h))
            # H,W,_ = degraded_img.shape
            # if(H%32!=0 or W%32!=0): # SCSEUnet处理的输入图像长宽如果不能被32整除，那么输出的结果尺寸会与输入图像不一致，因此这里直接对输入图像进行处理，舍弃部分像素
            #     # img = img[:H // 32 * 32, :W // 32 * 32, :]
            #     # mask = mask[:H // 32 * 32, :W // 32 * 32, :]
            #     degraded_img = degraded_img[:H // 32 * 32, :W // 32 * 32, :]
            # H_r,W_r,_ = img.shape
            # if(H%32!=0 or W%32!=0): # SCSEUnet处理的输入图像长宽如果不能被32整除，那么输出的结果尺寸会与输入图像不一致，因此这里直接对输入图像进行处理，舍弃部分像素
            #     img = img[:H_r // 32 * 32, :W_r // 32 * 32, :]
            #     mask = mask[:H_r // 32 * 32, :W_r // 32 * 32, :]
            #     #degraded_img = degraded_img[:H // 32 * 32, :W // 32 * 32, :]

            img, mask, degraded_img, degraded_label = aug(img,mask,degraded_img,degraded_label)
            h, w = degraded_img.shape[:2]

        if self.choice == 'val':
            


            degraded_img, degraded_label = img, mask   # cv2.imdecode(encimg, 1)
                            # 获取原始图像的宽度和高度
            h, w = degraded_img.shape[:2]

            # 获取原始图像的宽度和高度
            h, w = degraded_img.shape[:2]
            resize_ratio =random.uniform(0.7,1.4)
            resize_ratio = round(resize_ratio,1)
            # 计算缩放后的宽度和高度
            new_w = int(w * resize_ratio)
            new_h = int(h * resize_ratio)
            # 缩放图像
            degraded_img = cv2.resize(img, (new_w, new_h))
            degraded_label = cv2.resize(degraded_label,(new_w, new_h))


        img = img[:,:,::-1]
        degraded_img = degraded_img[:, :, ::-1]
        img = img.astype('float') / 255.
        degraded_img = degraded_img.astype('float') / 255.

        mask = mask.astype('float')
        mask[np.where(mask < 127.5)] = 0
        mask[np.where(mask >= 127.5)] = 1
        degraded_label = degraded_label.astype('float')
        degraded_label[np.where(degraded_label < 127.5)] = 0
        degraded_label[np.where(degraded_label >= 127.5)] = 1

        if(self.choice=='train' or self.choice=='val'):
            return self.tensor(img), self.tensor(mask[:, :, :1]), self.tensor(degraded_img),self.tensor(degraded_label[:,:,:1]),fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


def compute_metrics(image_batch,label_batch,outputs_batch,f1_all,iou_all,auc_all):
    # #计算每一个batch里面每一张图的F1和AUC
    image_batch = image_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()
    outputs_batch = outputs_batch.cpu().detach().numpy()
    # print(image_batch.shape,label_batch.shape,outputs_softmax.shape,outputs_threshold.shape)
    for image, label, predict_map in zip(image_batch, label_batch, outputs_batch):
        predict_map = predict_map[0,:,:]
        label = label[0,:,:]
        predict_threshold = np.copy(predict_map)
        predict_threshold[np.where(predict_map<0.5)] = 0
        predict_threshold[np.where(predict_map>=0.5)] = 1
        if(len(np.unique(label))<2):
            continue
        try:
            tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(predict_threshold, label)
            auc = roc_auc_score(label.reshape(-1, ), predict_map.reshape(-1, )) # 部分训练可能只有一类，此时计算auc的代码会报错，因此设置一个try语句来跳过计算这部分数据的几个指标
            f1_all.append(f1)
            auc_all.append(auc)
            iou_all.append(iou)
        except Exception as e:
            continue
    return f1_all,iou_all,auc_all
class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self,y_pred, y_true, eps=1e-8):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        return dice_loss

def get_logger(filename, verbosity=1, name=None):
    '''
    打印训练过程中的日志
    '''
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Discriminator(nn.Module):
    '''
    判别器网络
    '''
    def __init__(self,nc = 3,ndf = 64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ndf * 8, 1, 1, 1, 0, bias=False),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)   

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Multilevel_feature_cov_pyramid(nn.Module):
    def __init__(self,nc=3,ndf=64,out_w=64) -> None:
        super(Multilevel_feature_cov_pyramid,self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.out_w =out_w


        self.conv1 = nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=(3,3), stride=(1,1), bias=False)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.ndf, out_channels=ndf*2, kernel_size=(3,3), stride=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=self.ndf * 2, out_channels=ndf*4, kernel_size=(3,3), stride=(1,1), bias=False)
        self.pool1 = nn.AdaptiveAvgPool2d(self.out_w)
        self.conv4 = nn.Conv2d(self.ndf * 4, 3, 1, 1, 0, bias=False)
        #self.spp =
    def forward(self,input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pool1(x) #torch.Size([8, 256, 1, 1])
        x = self.conv4(x)

        return x
class Multilevel_feature_cov_pyramid_block(nn.Module):
    def __init__(self, nc=3, ndf=64, out_w=128) :
        super(Multilevel_feature_cov_pyramid_block,self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.out_w =out_w

        self.block1 = Multilevel_feature_cov_pyramid(nc=self.nc, ndf=self.ndf, out_w=self.out_w)

    def forward(self,img_A, img_B):
        input1 = self.block1(img_A)
        input2 = self.block1(img_B)

        return input1,input2

if args.mode=='train':
    train_file = create_file_list(args.train_image_path,args.train_mask_path,args.train_images_split)
    val_file = create_file_list(args.test_image_path,args.test_mask_path,args.test_images_split)
    
    # train_dataset_restore = Tampering_Dataset(train_file, choice='train', patch_size=args.restore_patch_size,with_aug=args.aug,compress_quality=args.quality)
    # train_dataset_loc = Tampering_Dataset(train_file, choice='train', patch_size=args.loc_patch_size,with_aug=args.aug,compress_quality=args.quality)
    # val_dataset = Tampering_Dataset(val_file, choice='val', patch_size=args.loc_patch_size,compress_quality=args.quality)
    
    train_dataset_restore = Tampering_Dataset_resize(train_file, choice='train', patch_size=args.restore_patch_size,with_aug=args.aug,resize_ratio=args.resize_ratio)
    train_dataset_loc = Tampering_Dataset_resize(train_file, choice='train', patch_size=args.loc_patch_size,with_aug=args.aug,resize_ratio=args.resize_ratio)
    val_dataset = Tampering_Dataset_resize(val_file, choice='val', patch_size=args.loc_patch_size,resize_ratio=args.resize_ratio)
    
    
    """本地PC测试"""
    # train_file,val_file = [],[]
    # for i in range(100):
    #     train_file.append(i)
    #     val_file.append(i)
    # train_dataset_restore = Tampering_Dataset_test(train_file, choice='train', patch_size=args.restore_patch_size,with_aug=args.aug,compress_quality=args.quality)
    # train_dataset_loc = Tampering_Dataset_test(train_file, choice='train', patch_size=args.loc_patch_size,with_aug=args.aug,compress_quality=args.quality)
    # val_dataset = Tampering_Dataset_test(val_file, choice='val', patch_size=args.loc_patch_size,compress_quality=args.quality) collate_fn=collate_wrapper, 

    train_dataloader_restore = DataLoader(dataset=train_dataset_restore, batch_size=args.restore_batch_size, shuffle=True,drop_last=True,pin_memory=False,num_workers=8)
    train_dataloader_loc = DataLoader(dataset=train_dataset_loc, batch_size=args.loc_batch_size, shuffle=True,
                                          drop_last=True, pin_memory=False, num_workers=8)

    #valid_dataloader = DataLoader(dataset=val_dataset, batch_size=args.loc_batch_size if 'OnlyOneRma2' in args.train_image_path else 1, shuffle=False,pin_memory=False,num_workers=8,drop_last=True)
    valid_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,pin_memory=False,num_workers=8,drop_last=True)
    restore_model = network_scunet.SCUNet(in_nc=3)
    if(args.localization_model == "SCSEUnet"):
        loc_model = SCSEUnet(backbone_arch='senet154')
        loc_model = nn.DataParallel(loc_model).cuda()
    elif args.localization_model =='denseFCN':
        # loc_model = denseFCN.normal_denseFCN(bn_in=args.bn_in)
        loc_model = denseFCN.normal_denseFCN(bn_in=args.bn_in).cuda()
        # loc_model = nn.DataParallel(loc_model)
    elif (args.localization_model == 'MVSS_net'):
        loc_model = get_mvss(backbone='resnet50',
                             pretrained_base=True,
                             nclass=1,
                             sobel=True,
                             constrain=True,
                             n_input=3)
                             
    """使用DCGAN"""
    dis_model = Discriminator(nc= 3,ndf = 64) 

    """使用PatchGAN"""
    #dis_model = Patch_Discriminator(in_channels=3) 

    #dis_model = NLayerDiscriminator(input_nc=3)

    dis_model.apply(weights_init)
    dis_model = nn.DataParallel(dis_model).cuda()
    restore_model = nn.DataParallel(restore_model).cuda()

    if(args.restore_path!=""):
        print("Restore the weights of restoration module from {} ".format(args.restore_path))
        restore_model.load_state_dict(torch.load(args.restore_path))

    if(args.loc_restore_path!=""):
        # 这里在加载定位模型参数的时候，可能会存在网络参数名不对应的情况。产生这种问题的原因是当时使用I_plain训练模型的时候，
        # 有些使用了nn.DataParallel进行处理，最后保存模型参数的时候直接使用torch.save(loc_model.state_dict()对其进行保存，
        # 而部分模型没有使用nn.DataParallel进行处理。因此在加载I_plain训练好的模型的时候，注意观察加载的模型参数名称是否一致，不一致的话要对其进行处理。
        # pretrain_dict = torch.load(args.loc_restore_path)
        # print("The variable name of the pre-trained localization module: ")
        # for k,v in pretrain_dict.items():
        #     print(k)
        # # loc_model = nn.DataParallel(loc_model).cuda()
        # print("The variable name of the present localization module:")
        # for name in loc_model.state_dict():
        #     print(name)
        # # 如果这二者的名称不一致，则可以使用nn.DataParallel对loc_model进行处理，让其一致。
        # loc_model.load_state_dict(torch.load(args.loc_restore_path))

        # pretrain_dict = torch.load(args.loc_restore_path)
        # model_dict = loc_model.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        # model_dict.update(pretrain_dict)
        # loc_model.load_state_dict(model_dict)
        pretrain_dict = torch.load(args.loc_restore_path)
        print("The variable name of the pre-trained localization module: ")
        for k,v in pretrain_dict.items():
            print(k)
        loc_model = nn.DataParallel(loc_model).cuda()
        print("The variable name of the present localization module:")
        for name in loc_model.state_dict():
            print(name)

        # 将模型的结构键值名字和权重的键值名字一致
        print("The variable name of the 处理 pre-trained localization module:")
        new_state_dict = {}
        for k, v in pretrain_dict.items():
            name =  k  #.replace('module.', '') # 删除'module.'前缀
            new_state_dict[name] = v
            
            print(name)

        #net.load_state_dict(new_state_dict)

        loc_model.load_state_dict(new_state_dict)
        #loc_model.load_state_dict(pretrain_dict)  #torch.load(args.loc_path)

    # loc_model = nn.DataParallel(loc_model).cuda()

    if(args.loss == 'bce'):
        criterion = nn.BCEWithLogitsLoss().to(device) # torch.nn.BCELoss()
    criterion_dice = SoftDiceLoss().to(device)
    criterion_l1 = torch.nn.L1Loss().to(device)
    # if(args.F_loss_function)
    # criterion_cosine = torch.nn.CosineSimilarity().to(device)

    optimizer_restore = torch.optim.Adam(restore_model.parameters(),lr = args.restore_learning_rate)
    optimizer_loc = torch.optim.Adam(loc_model.parameters(),lr=args.loc_learning_rate)

    if('GAN' in args.restore_loss_type):
        optimizer_dis = torch.optim.Adam(dis_model.parameters(),lr = args.dis_learning_rate,betas=(0.5,0.999))

    if(args.lr_schdular==''):
        lr_schdular_restore = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_restore,'min',factor=0.95,patience=args.patience)
        lr_schdular_loc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_loc,'min',factor=0.95,patience=args.patience)
    logger = get_logger(os.path.join(args.save_model_path,"log.log"))

    # 将args中所有变量进行打印 #
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    best_f1_plus_iou = 0
    best_f1 = 0
    best_auc = 0
    best_val_loss = 99999
    logger.info("Train Loc data:{} Train Restore data: {} Val data:{}".format(len(train_dataloader_loc),len(train_dataloader_restore),len(valid_dataloader)))
    total_iter_restore = 0
    total_iter_loc = 0
    total_iter_dis = 0
    if(args.train_f==True):
        f_model = Multilevel_feature_cov_pyramid_block(nc=3, ndf=64, out_w=args.feature_w)   
        f_model.to(device)
        optimizer_f = torch.optim.Adam(f_model.parameters(),lr=args.feature_learning_rate)
    # Net2 = my_block(nc=1, ndf=64, out_w=64) 
    # Net2.to(device)

    for epoch in range(args.epochs):
        lr_restore = optimizer_restore.param_groups[0]['lr']
        lr_loc = optimizer_loc.param_groups[0]['lr']
        if(args.train_f==True):
            lr_f = optimizer_f.param_groups[0]['lr']
        # restore_model.train()
        # loc_model.train()
        # Net1.train()
        #Net2.train()

        if((epoch%2==0 or args.only_train_restore) and args.only_train_loc==False):
            '''训练restore model'''
            for param in restore_model.parameters():
                param.requires_grad = True
            if(args.restore_loss_type=='only_l1'):
                for param in loc_model.parameters():
                    param.requires_grad = False
            else:
                for param in loc_model.parameters():
                    param.requires_grad = True
            loc_model.eval()
            restore_model.train()

            if (args.train_f==True):
                f_model.train()

            #
            #Net2.train()

            train_epoch_loss = []
            train_epoch_restore_loss, train_epoch_loc_loss = [], []
            train_f1, train_iou, train_auc = [], [], []
            train_epoch_D,train_epoch_G,train_epoch_Dx,train_epoch_D_G_z1,train_epoch_D_G_z2 = [],[],[],[],[]
            train_epoch_D_err_fake,train_epoch_D_err_real = [],[]
            train_epoch_feature_loss = []
            for idx, (data_x, data_y, data_degraded,label_degraded, file_name) in enumerate(train_dataloader_restore, 0):
                #print(file_name)
                data_x = data_x.to(device)
                data_y = data_y.to(device)
                data_degraded = data_degraded.to(device)
                label_degraded = label_degraded.to(device)
                t1 = time.time()
                restore_image = restore_model(data_degraded).to(device)
                
                t2 = time.time()
                if (args.localization_model != "MVSS_net"):
                    outputs = loc_model(restore_image).to(device)
                    
                else:
                    edge_outputs, outputs = loc_model(restore_image)
                    outputs = torch.sigmoid(outputs).to(device)
                    edge_outputs = torch.sigmoid(edge_outputs)

                total_iter_restore += 1
                if(args.restore_loss_type=='GAN_plus_l1_plus_loc' or args.restore_loss_type =='l1_plus_GAN' or args.restore_loss_type=='GAN_plus_l1_plus_lossf_plus_loc' ):
                    #restore_image,data_x = Net1(restore_image.cuda(), data_x.cuda())
                    for _ in range(args.dis_step_iters):   #args.dis_step_iters = 1
                        dis_model.zero_grad()
                        b_size = data_x.size(0)
                        real_label = 1
                        fake_label = 0
                        label = torch.full((b_size,), real_label, dtype=torch.float).cuda() # 有GPU得使用这个
                        # label = torch.full((b_size,), real_label, dtype=torch.float)

                        # Forward pass real batch through D
                        output_hr = dis_model(data_x).view(-1)
                        # Calculate loss on all-real batch
                        errD_real = criterion(output_hr, label)
                        # Calculate gradients for D in backward pass
                        errD_real.backward()
                        D_x = output_hr.mean().item()
                        label.fill_(fake_label)
                        # Classify all fake batch with D
                        output_restore = dis_model(restore_image.detach()).view(-1)
                        # Calculate D's loss on the all-fake batch
                        errD_fake = criterion(output_restore, label)
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                        errD_fake.backward()
                        D_G_z1 = output_restore.mean().item()
                        # Compute error of D as sum over the fake and the real batches
                        errD = errD_real + errD_fake
                        # errD.backward()
                        # Update D
                        optimizer_dis.step()
                    ##################
                    # 更新生成网络  最大化log{D{G(z)}}
                    optimizer_restore.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output_restore = dis_model(restore_image).view(-1)
                    # Calculate G's loss based on this output
                    errG = criterion(output_restore, label)
                    D_G_z2 = output_restore.mean().item()

                    if(args.restore_loss_type=='GAN_plus_l1_plus_lossf_plus_loc'):
                        h,w =data_x.shape[2:]
                        restore_image = F.interpolate(restore_image, size=(128, 128), mode='nearest')
                        loss_MAE = criterion_l1(restore_image.contiguous().view(restore_image.size(0), -1),data_x.contiguous().view(data_x.size(0),-1))
                        loss_MAE.to(device)
                        restore_image_f, data_x_f = f_model(restore_image.cuda(), data_x.cuda())
                        # outputs = F.interpolate(outputs, size=(128, 128), mode='nearest')#, align_corners=True
                        # data_y =  F.interpolate(data_y, size=(128, 128), mode='nearest')      #
                        #往往
                        if(args.F_loss_function=='cosine'):
                            loss_feature_map = compute_cosine_sum(restore_image_f.view(restore_image_f.size(0), -1),data_x_f.view(data_x_f.size(0),-1))
                        elif(args.F_loss_function=='euclidean'):
                            loss_feature_map = compute_euclidean_sum(restore_image_f.view(restore_image_f.size(0), -1),data_x_f.view(data_x_f.size(0),-1))
                        #loss_feature_map.to(device) 
                        #loss_feature_map =loss_feature_map.clone().detach()
                        #print(type(loss_feature_map))

                        loss_MAE = args.restore_weight * loss_MAE
                        loss_G = args.GAN_weight * errG
                        loss_Loc = args.loc_weight *((args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),label_degraded.view(label_degraded.size(0),-1))) +((1-args.cross_entropy_weight)*criterion_dice(outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))) )
                        loss_f = args.f_weight * loss_feature_map.mean() #
                        loss_restore =(loss_MAE + loss_G + loss_Loc+ loss_f ) .to(device) 
                        
                        loss_restore.backward()
                        optimizer_restore.step()
                        optimizer_f.step()
                    elif(args.restore_loss_type=='GAN_plus_l1_plus_loc'):
                        h,w =data_x.shape[2:]
                        restore_image = F.interpolate(restore_image, size=(128, 128), mode='nearest')
                        loss_MAE = criterion_l1(restore_image.contiguous().view(restore_image.size(0), -1),data_x.contiguous().view(data_x.size(0),-1))
                        loss_MAE.to(device)
                        loss_MAE = args.restore_weight * loss_MAE
                        loss_G = args.GAN_weight * errG
                        loss_Loc = args.loc_weight *((args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),label_degraded.view(label_degraded.size(0),-1))) +((1-args.cross_entropy_weight)*criterion_dice(outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))) )
                        
                        loss_restore =(loss_MAE + loss_G + loss_Loc ) .to(device) 

                        loss_restore.backward()
                        optimizer_restore.step()


                        train_epoch_restore_loss.append(loss_MAE.item())
                        train_epoch_loc_loss.append(loss_Loc.item())
                        train_epoch_D.append(errD.item())
                        train_epoch_G.append(errG.item())
                        train_epoch_Dx.append(D_x)
                        train_epoch_D_G_z1.append(D_G_z1)
                        train_epoch_D_G_z2.append(D_G_z2)
                        train_epoch_D_err_fake.append(errD_fake.item())
                        train_epoch_D_err_real.append(errD_real.item())
                        if total_iter_restore % args.display_step == 0:
                            logger.info(
                                "epoch={}/{},{}/{}of train {}, Learning rate={} Restore loss = {}  Loc loss = {} Loss_D: {} Loss D_fake: {} Loss D_real: {} Loss_G: {} D(x): {} D(G(z)):{}/{}".format(
                                    epoch, args.epochs, idx, len(train_dataloader_restore), total_iter_restore, lr_restore,
                                    np.mean(train_epoch_restore_loss),  np.mean(train_epoch_loc_loss),np.mean(train_epoch_D),
                                    np.mean(train_epoch_D_err_fake),np.mean(train_epoch_D_err_real),np.mean(train_epoch_G),
                                    np.mean(train_epoch_Dx), np.mean(train_epoch_D_G_z1),np.mean(train_epoch_D_G_z2)))




        else:
            '''训练localization module'''
            for param in loc_model.parameters():
                param.requires_grad = True
            for param in restore_model.parameters():
                param.requires_grad = False
            loc_model.train()
            restore_model.eval()
            #net1 net2 不训练
            if(args.train_f==True):
                f_model.eval()
            #Net2.eval()
            train_epoch_loss = []
            train_epoch_restore_loss, train_epoch_loc_loss = [], []
            train_f1, train_iou, train_auc = [], [], []
            for idx, (data_x, data_y, data_degraded, label_degraded,file_name) in enumerate(train_dataloader_loc, 0):
                data_x = data_x.cuda()
                data_y = data_y.cuda()
                data_degraded = data_degraded.cuda()
                label_degraded.to(device)
                if(args.only_train_loc==True):
                    restore_image = data_degraded.to(device)
                else:
                    restore_image = restore_model(data_degraded).to(device)

                if (args.localization_model != 'MVSS_net'):

                    outputs = loc_model(restore_image).to(device)
                    
                    # logger.info(data_x.shape)
                    # logger.info(data_y.shape)
                    # logger.info(outputs.shape)
                    # h,w = data_y.shape[2:]
                    # outputs = F.interpolate(outputs, size=(h,w ), mode='nearest')#, align_corners=True
                    # #data_y =  F.interpolate(data_y, size=(h,w ), mode='bilinear', align_corners=True)   
                    loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),
                                               label_degraded.view(label_degraded.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(
                        outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))
                else:
                    edge_outputs, outputs = loc_model(restore_image)
                    outputs = torch.sigmoid(outputs).to(device)
                    # h,w = data_y.shape[2:]
                    # outputs = F.interpolate(outputs, size=(h,w ), mode='nearest')#, align_corners=True
                    # edge_outputs = torch.sigmoid(edge_outputs)
                    loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),
                                               label_degraded.view(label_degraded.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(
                        outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))
                total_iter_loc += 1

                optimizer_loc.zero_grad()
                loss_loc.backward()
                optimizer_loc.step()

                train_epoch_loc_loss.append(loss_loc.item())
                if total_iter_loc % args.display_step == 0:
                    logger.info("epoch={}/{},{}/{}of train {}, Learning rate={} Loc loss = {} ".format(
                        epoch, args.epochs, idx, len(train_dataloader_loc), total_iter_loc, lr_loc,
                        np.mean(train_epoch_loc_loss)))

        if (True):
            '''验证模型性能'''
            restore_model.eval()
            loc_model.eval()
            if(args.train_f==True):
                f_model.eval()
            valid_epoch_loss = []
            val_epoch_loc_loss,val_epoch_restore_loss = [],[]
            val_f1, val_iou, val_auc = [], [], []
            with torch.no_grad():
                for val_index, (data_x, data_y,data_degraded,label_degraded,file_name) in enumerate(valid_dataloader):
                    data_x = data_x.cuda() # Plain
                    #logger.info("data_x{}".format(data_x.shape))
                    data_y = data_y.cuda()  # mask
                    #logger.info("data_y{}".format(data_y.shape))
                    data_degraded = data_degraded.cuda() # resize_degraded
                    label_degraded = label_degraded.cuda()

                    #logger.info("data_degraded{}".format(data_degraded.shape))

                    if(args.only_train_loc==True):
                        restore_image = data_degraded
                    else:
                        restore_image = restore_model(data_degraded)

                    if (args.localization_model != 'MVSS_net'):
                        outputs = loc_model(restore_image)
                        # h,w = data_y.shape[2:]
                        # outputs = F.interpolate(outputs, size=(h,w ), mode='nearest')
                        #outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)
                        #data_y =  F.interpolate(data_y, size=(128, 128), mode='bilinear', align_corners=True)

                        loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),
                                                   label_degraded.view(label_degraded.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(
                            outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))
                    else:
                        edge_outputs, outputs = loc_model(restore_image)
                        outputs = torch.sigmoid(outputs)
                        # h,w = data_y.shape[2:]
                        # outputs = F.interpolate(outputs, size=(h,w ), mode='nearest')
                        edge_outputs = torch.sigmoid(edge_outputs)
                        #outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)
                        #data_y =  F.interpolate(data_y, size=(128, 128), mode='bilinear', align_corners=True)
                        loss_loc = args.cross_entropy_weight * criterion(outputs.view(outputs.size(0), -1),
                                                   label_degraded.view(label_degraded.size(0), -1)) + (1-args.cross_entropy_weight) * criterion_dice(
                            outputs.view(outputs.size(0), -1), label_degraded.view(label_degraded.size(0), -1))
                    #restore_image, data_x = Net1(restore_image.cuda(), data_x.cuda())
                    #loss_restore = criterion_l1(restore_image.contiguous().view(restore_image.size(0),-1),data_x.contiguous().view(data_x.size(0),-1))
                    """师兄做的是压缩后处理，我做的是resize后处理，在这里我计算loss_restore是restore和degraded比较"""
                    h,w =data_x.shape[2:]
                    restore_image = F.interpolate(restore_image, size=(h, w), mode='nearest')
                    loss_MAE = criterion_l1(restore_image.contiguous().view(restore_image.size(0), -1),data_x.contiguous().view(data_x.size(0),-1))
                    #loss_restore = criterion_l1(restore_image.contiguous().view(restore_image.size(0),-1),data_degraded.contiguous().view(data_degraded.size(0),-1))
                    #loss_feature  需要Net1
                    loss = loss_loc + loss_MAE #+ loss_featue
                    valid_epoch_loss.append(loss.item())
                    val_epoch_loc_loss.append(loss_loc.item())
                    val_epoch_restore_loss.append(loss_MAE.item())
                    val_f1, val_iou, val_auc = compute_metrics(data_x, label_degraded, outputs, val_f1, val_iou, val_auc)

            if(np.mean(valid_epoch_loss)<best_val_loss or np.mean(val_f1)>best_f1 or np.mean(val_auc)>best_auc):
                if(np.mean(valid_epoch_loss)<best_val_loss):
                    best_val_loss = np.mean(valid_epoch_loss)
                if (np.mean(val_f1)>best_f1):
                    best_f1 = np.mean(val_f1)
                if (np.mean(val_auc)>best_auc):
                    best_auc = np.mean(val_auc)

                torch.save(dis_model.state_dict(), os.path.join(args.save_model_path, "Dis_model",
                                                                    "Epoch_{}_Loss_{}_F1_{}_IOU_{}_AUC_{}.pth".format(
                                                                        epoch,
                                                                        round(np.mean(valid_epoch_loss), 4),
                                                                        round(np.mean(val_f1), 4),
                                                                        round(np.mean(val_iou), 4),
                                                                        round(np.mean(val_auc), 4))))
                torch.save(restore_model.state_dict(), os.path.join(args.save_model_path, "Restore_model",
                                                                    "Epoch_{}_Loss_{}_F1_{}_IOU_{}_AUC_{}.pth".format(
                                                                        epoch,
                                                                        round(np.mean(valid_epoch_loss), 4),
                                                                        round(np.mean(val_f1), 4),
                                                                        round(np.mean(val_iou), 4),
                                                                        round(np.mean(val_auc), 4))))
                torch.save(loc_model.state_dict(), os.path.join(args.save_model_path, "Loc_model",
                                                                "Epoch_{}_Loss_{}_F1_{}_IOU_{}_AUC_{}.pth".format(
                                                                    epoch, round(np.mean(valid_epoch_loss), 4),
                                                                    round(np.mean(val_f1), 4),
                                                                    round(np.mean(val_iou), 4),
                                                                    round(np.mean(val_auc), 4))))
                if(args.train_f==True):
                    torch.save(f_model.state_dict(), os.path.join(args.save_model_path, "F_model", "Epoch_{}_Loss_{}_F1_{}_IOU_{}_AUC_{}.pth".format(
                                                                            epoch,
                                                                            round(np.mean(valid_epoch_loss), 4),
                                                                            round(np.mean(val_f1), 4),
                                                                            round(np.mean(val_iou), 4),
                                                                            round(np.mean(val_auc), 4))))



            logger.info("Validation {}, Restore Learning rate = {} Loc learning rate = {} Total Loss = {} Restore loss = {} Loc Loss = {} F1 = {} IOU = {} AUC = {} Best loss = {} Best f1 = {} Best auc = {}".format(epoch,lr_restore,lr_loc,np.mean(valid_epoch_loss),np.mean(val_epoch_restore_loss),np.mean(val_epoch_loc_loss),np.mean(val_f1),np.mean(val_iou),np.mean(val_auc),best_val_loss,best_f1,best_auc))

            val_mean_loss = np.mean(valid_epoch_loss)

        if(epoch%2==0):
            lr_schdular_restore.step(np.mean(valid_epoch_loss))
        else:
            lr_schdular_loc.step(np.mean(valid_epoch_loss))
