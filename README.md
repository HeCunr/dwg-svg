# 将svg通过模型得到潜在空间向量z,通过比较z得到相似度
## 目标：将svg通过模型得到潜在空间向量z,通过比较z得到相似度
## 做法：
### 文件目录结构为https://github.com/alexandre01/deepsvg这篇论文的结构
### 上述文件为我新增到上述论文代码中的py文件
### 1.在notebooks内添加mul_mytrain.py,change.py,read_pth.py,Withdraw.py,to_pkl.py
### 2.在\configs\deepsvg\内添加myconfig.py
### 3.mul_mytrain.py的作用为训练得到.pth文件，因为我们需要更调整nb_groups,max_len_group,total_len来满足自己数据集的需求
### 4.需要在下述几个地方调整相关配置
    '''
      configs/deepsvg/default_icons.py
      self.max_num_groups = 60
      self.max_total_len = 120
    
      configs/deepsvg/myconfig.py
      self.max_num_groups = 60
      self.max_total_len = 120
      def set_train_vars(self, train_vars, dataloader):
          train_vars.x_inputs_train = [dataloader.dataset.get(idx, [*self.model_args, "tensor_grouped"])
                                     for idx in random.sample(range(len(dataloader.dataset)), k=1)]#k=5->k=1，此处k表示每轮取出多少张图片训练，如果我们提供训练的pkl数量小于5张，可以将k更改为1 **
    deepsvg/model/config.py
    self.max_num_groups = 60           # Number of paths (N_P) 
    self.max_seq_len = 120             # Number of commands (N_C) 
    self.max_total_len = self.max_num_groups * self.max_seq_len  # Concatenated sequence length for baselines
    
    deepsvg/config.py
       self.data_dir = "/tmp/deepsvg-train/dataset/mytensor"              #更改为自己得到的tensor文件（一系列pkl文件）
       self.meta_filepath = "/tmp/deepsvg-train/dataset/svg_meta.csv"    #更改为自己得到的svg_meta.csv
       self.max_num_groups = 60                         
       self.max_seq_len = 120          
       
    deepsvg/mytrain.py
      final_model_path = r"C:\Users\15653\deepsvg\pretrained\my_pth\my.pth"更改为训练后得到的pth文件保存的路径
      在训练过程中如果报错，可考虑将下述代码注释掉
        #print("Dataset size:", len(dataset))
        #
        # if len(dataset) > 0:
        #     try:
        #         single_forward_dataloader = DataLoader(
        #             dataset,
          #             batch_size=1,#作者是cfg.batch_size // cfg.num_gpus，我调整为1
        #             shuffle=True,
        #             drop_last=True,
        #             num_workers=cfg.loader_num_workers,
        #             collate_fn=cfg.collate_fn
        #         )
        #         data = next(iter(single_forward_dataloader))
        #         print("Data loaded successfully:", data)
        #     except StopIteration:
        #         print("No data could be loaded. Check batch size and dataset size.")
        # else:
        #     print("Dataset is empty. Please check your dataset.")'''
### 5.在实验室GPU集群中选择最新的cuda10.1版本
      直接运行mul_mytrain.py,边报错边安装缺少的包
      有一个地方需要注意：使用“apt-get install libcairo2-dev”而不是“sudo apt-get install libcairo2-dev“
### 6.准备需要训练的svg图片集：利用游览器在线转化工具：将dwg直接转化为svg格式，老师给的dwg非常大，我们需要借助autoCAD将其拆分成较小的部分
### 7.在cmd中进入dataset/preprocess.py，运行下述代码，将我们准备的svg集转化为简化后的svg集（svg_simplified）以及得到svg_meta.csv文件
     ''' python -m dataset.preprocess --data_folder dataset/svgs/ --output_folder dataset/svgs_simplified/ --output_meta_file dataset/svg_meta.csv'''
### 8.利用notebooks/to_pkl.py将svg_simplified中svg转化为pkl格式，作为self.data_dir = "/tmp/deepsvg-train/dataset/mytensor" 
### 9.运行mul_mytrain.py开始训练，即可得到自己的pth文件
### 10.运行change.py将得到的pth文件转化为模型规定格式的pth文件
### 11.运行withdraw_z.py得到z
### 12.奇怪的地方
####  max_len_group需要设置成total_len，而total_len则需要设置成nb_groups*max_len_group
####  比如说nb_groups（组数）最大为60，max_len_group（每组中最大长度）为2，正常total_len应该为120
####  但是，我们需要如下设置，才不会报错nb_groups：60，max_len_group：120，正常total_len7200
### 13.一个说明
#### 由于每段路径有起始符和结束符，所以报错中经常-2就可以符合我们的认知
    ![image](https://github.com/user-attachments/assets/0d5d4e11-90f7-40fa-ac54-f569e4b35d4a)
    96-2=94为实际的total_len,4-2=2,为max_len_group
### 13.这篇论文的方法，目前能做到的就是得到小尺度dwg（nb_groups为几十的）的潜在空间向量z，而对于老师所给的dwg(nb_groups为万级别，会报显存错误
####   还剩一条路：借鉴这篇论文svg是怎么表示的，我们自己找其它的自编码模型套进去，再得到z(从0开始)
####   如果得到了z,那么相似度比较就差不多完成了




















        
