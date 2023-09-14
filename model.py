import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np


### HGCL
class MODEL(nn.Module):
    def __init__(self,args, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers):
        super(MODEL, self).__init__()
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        
        uimat   = self.uiMat[: self.userNum,  self.userNum:]  # 提取 self.uiMat 的一个子矩阵
        values  = torch.FloatTensor(uimat.tocoo().data)   # .data 获取稀疏矩阵中的非零值
        indices = np.vstack(( uimat.tocoo().row,  uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape =  uimat.tocoo().shape
        uimat1=torch.sparse.FloatTensor(i, v, torch.Size(shape))  # 建立一个稀疏张量 uimat1，该张量的值、行索引、列索引和形状分别为 v、i、shape
        self.uiadj = uimat1  # 得到用户到项目的邻接矩阵。
        self.iuadj = uimat1.transpose(0,1)  # 对稀疏张量 uimat1 进行转置操作，得到项目到用户的邻接矩阵，并将其赋值给 self.iuadj
        
        self.gating_weightub=nn.Parameter(
            torch.FloatTensor(1, hide_dim))  # 使用 nn.Parameter() 创建一个可训练的张量参数 self.gating_weightub。torch.FloatTensor创建一个形状为的浮点张量作为参数的初始值
        nn.init.xavier_normal_(self.gating_weightub.data)  # 对 self.gating_weightub.data 进行 Xavier 初始化，即使用均匀分布生成初始值。
        self.gating_weightu=nn.Parameter( 
            torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib=nn.Parameter( 
            torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti=nn.Parameter(
            torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
        self.k = args.rank 
        k = self.k
        self.mlp  = MLP(hide_dim,hide_dim*k, hide_dim//2, hide_dim*k)
        self.mlp1 = MLP(hide_dim,hide_dim*k, hide_dim//2, hide_dim*k)
        self.mlp2 = MLP(hide_dim,hide_dim*k, hide_dim//2, hide_dim*k)
        self.mlp3 = MLP(hide_dim,hide_dim*k, hide_dim//2, hide_dim*k)
        self.meta_netu = nn.Linear(hide_dim*4, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim*4, hide_dim, bias=True)

        self.embedding_dict = nn.ModuleDict({
        'uu_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
        'ii_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
        'user_emb': torch.nn.Embedding(userNum , hide_dim).cuda(),
        'item_emb': torch.nn.Embedding(itemNum , hide_dim).cuda(),
        })

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def metaregular(self,em0,em,adj):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:,torch.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[torch.randperm(embedding.shape[0])]
            return corrupted_embedding
        def score(x1,x2):
            x1=F.normalize(x1,p=2,dim=-1)
            x2=F.normalize(x2,p=2,dim=-1)
            return torch.sum(torch.multiply(x1,x2),1)
        user_embeddings = em
        Adj_Norm =t.from_numpy(np.sum(adj,axis=1)).float().cuda()
        adj=self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = torch.spmm(adj.cuda(),user_embeddings)/Adj_Norm
        user_embeddings=em0
        graph = torch.mean(edge_embeddings,0)
        pos   = score(user_embeddings,graph)
        neg1  = score(row_column_shuffle(user_embeddings),graph)
        global_loss = torch.mean(-torch.log(torch.sigmoid(pos-neg1)))
        return global_loss 

    def self_gatingu(self,em):
        return torch.multiply(em, torch.nn.functional.softmax(torch.matmul(em,self.gating_weightu) + self.gating_weightub, dim=1))  # gai   # nn.functional.softmax()
    def self_gatingi(self,em):
        return torch.multiply(em, torch.nn.functional.softmax(torch.matmul(em,self.gating_weighti) + self.gating_weightib, dim=1))  # gai   # nn.

    def metafortansform(self, auxiembedu,targetembedu,auxiembedi,targetembedi):
       
        # Neighbor information of the target node
        uneighbor=t.matmul( self.uiadj.cuda(),self.ui_itemEmbedding)
        ineighbor=t.matmul( self.iuadj.cuda(),self.ui_userEmbedding)

        # Meta-knowlege extraction
        tembedu=(self.meta_netu(t.cat((auxiembedu,targetembedu,uneighbor,uneighbor),dim=1).detach()))  # Muu   auxiembedu:Euu
        tembedi=(self.meta_neti(t.cat((auxiembedi,targetembedi,ineighbor,ineighbor),dim=1).detach()))  # Mii   auxiembedi:Eii
        
        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1=self.mlp( tembedu). reshape(-1,self.hide_dim,self.k)  # d*k
        metau2=self.mlp1(tembedu). reshape(-1,self.k,self.hide_dim)  # k*d
        metai1=self.mlp2(tembedi). reshape(-1,self.hide_dim,self.k)  # d*k
        metai2=self.mlp3(tembedi). reshape(-1,self.k,self.hide_dim)  # k*d
        meta_biasu =(torch.mean( metau1,dim=0))
        meta_biasu1=(torch.mean( metau2,dim=0))
        meta_biasi =(torch.mean( metai1,dim=0))
        meta_biasi1=(torch.mean( metai2,dim=0))
        low_weightu1=F.softmax( metau1 + meta_biasu, dim=1)  # Wuu m1
        low_weightu2=F.softmax( metau2 + meta_biasu1,dim=1)  # Wuu m2
        low_weighti1=F.softmax( metai1 + meta_biasi, dim=1)  # Wii m1
        low_weighti2=F.softmax( metai2 + meta_biasi1,dim=1)  # Wii m2

        # The learned matrix as the weights of the transformed network
        tembedus = t.multiply(auxiembedu, targetembedu)
        tembedus = (t.sum(t.multiply( (tembedus).unsqueeze(-1), low_weightu1), dim=1))  # Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembedus =  t.sum(t.multiply( (tembedus)  .unsqueeze(-1), low_weightu2), dim=1)
        tembedis = t.multiply(auxiembedi, targetembedi)
        tembedis = (t.sum(t.multiply( (tembedis).unsqueeze(-1), low_weighti1), dim=1))
        tembedis =  t.sum(t.multiply( (tembedis)  .unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed  # EuuM EiiM  缺个激活函数
    def forward(self, iftraining, uid, iid, norm = 1):
        
        item_index=np.arange(0,self.itemNum)
        user_index=np.arange(0,self.userNum)
        ui_index = np.array(user_index.tolist() + [ i + self.userNum for i in item_index])
        
        # Initialize Embeddings
        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight
        uu_embed0  = self.self_gatingu(userembed0)  # Euu0
        ii_embed0  = self.self_gatingi(itemembed0)  # Eii0
        self.ui_embeddings       = t.cat([ userembed0, itemembed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings   = [self.ui_embeddings]
        # Encoder
        for i in range(len(self.encoder)):  # i = 0, 1 。
            layer = self.encoder[i]
            if i == 0:  
                userEmbeddings0 = layer(uu_embed0, self.uuMat, user_index)      # 3.2.2节的图卷积
                itemEmbeddings0 = layer(ii_embed0, self.iiMat, item_index)
                uiEmbeddings0   = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.iiMat, item_index)
                uiEmbeddings0   = layer(uiEmbeddings,   self.uiMat, ui_index)   # 图卷积加注意力
            
            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [self.userNum, self.itemNum])
            if i % 2 == 0:  # i % 2 == 0
                userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2.0  # Euul+1 Eul+1
                itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2.0  # 这里是f()，为了降低模型的复杂性，使用元素级 均值池，作为融合函数。
                userEmbeddings = userEd
                itemEmbeddings = itemEd
                uiEmbeddings = torch.cat([userEd, itemEd], 0)
            else:
                userEmbeddings = userEmbeddings0
                itemEmbeddings = itemEmbeddings0
                uiEmbeddings = uiEmbeddings0

            if norm == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings   += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [norm_embeddings]
                self.all_ui_embeddings   += [norm_embeddings]
        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim=1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)  
        self.itemEmbedding = t.mean(self.itemEmbedding, dim=1)
        self.uiEmbedding   = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding   = t.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [self.userNum, self.itemNum])
        
        # Personalized Transformation of Auxiliary Domain Features
        metatsuembed,metatsiembed = self.metafortansform(self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)  # EuuM EiiM
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed
        
        # Regularization: the constraint of transformed reasonableness
        metaregloss = 0
        if iftraining == True :
            self.reg_lossu = self.metaregular((self.ui_userEmbedding[uid.cpu().numpy()]),(self.userEmbedding),self.uuMat[uid.cpu().numpy()])
            self.reg_lossi = self.metaregular((self.ui_itemEmbedding[iid.cpu().numpy()]),(self.itemEmbedding),self.iiMat[iid.cpu().numpy()])
            metaregloss =  (self.reg_lossu +  self.reg_lossi)/2.0
        return self.userEmbedding, self.itemEmbedding,(self.args.wu1*self.ui_userEmbedding + self.args.wu2*self.userEmbedding), (self.args.wi1*self.ui_itemEmbedding + self.args.wi2*self.itemEmbedding), self.ui_userEmbedding, self.ui_itemEmbedding ,  metaregloss
        

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()
        self.mlp = MLP2(32, [32 * 3], 32)#32 * 3)  #改
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()  # device:cpu
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def calculate_attention_weights(self, features):  #改
        # 使用一个MLP计算注意力权重
        weights = self.mlp(features)

        # 使用softmax进行归一化
        attention_weights = F.softmax(weights, dim=1)

        return attention_weights

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)

        # 计算注意力权重
        #attention_weights = self.calculate_attention_weights(subset_features)

        # 将注意力权重作用于邻接矩阵上
        #subset_Mat = subset_Mat * attention_weights.cpu().detach()
        #subset_Mat = coo_matrix(subset_Mat)

        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()  #subset_Mat实际上没有那末多，不是每个都有值，但维度确实有那末大。

        # out_features = subset_sparse_tensor * attention_weights  #.cpu().detach()

        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP2, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =    nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()
        x = prelu(x) 
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
















