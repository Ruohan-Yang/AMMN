
import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from sklearn import metrics

class PGD():  # Projected Gradient Descent 迭代多次，慢慢找到最优的扰动
    # PGD“小步走，多走几步”，如果走出了扰动半径为epsilon的空间，就映射回“球面”上，以保证扰动不要过大
    # 对wordEmbedding空间扰动是累加的。每次都在上一次加扰动的基础上再加扰动，取最后一次的梯度来更新网络参数
    def __init__(self, model, emb_name, epsilon, alpha):
        # emb_name模型中embedding参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}  # 权重参数备份
        self.grad_backup = {}  # 梯度备份

    def attack(self, is_first_attack=False):  # 对抗部分数据扰动
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()  # 权重参数备份
                norm = torch.norm(param.grad)  # 默认求2范数（平方和再开方）
                if norm != 0:  # 扰动
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)  # 添加对抗扰动
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():  # 对抗部分参数备份
            if param.requires_grad and self.emb_name in name:
                # assert的用途就是确定这个语句是正确的。
                # 如果正确，什么结果也不出。如果不正确，则输出：AssertionError
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):  # 模型全部梯度备份
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):  # 模型全部梯度恢复
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class GateFusion(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super(GateFusion, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2

class GCN_NET(nn.Module):
    def __init__(self, feature_dims, out_dims, hidden_dims=64):
        super(GCN_NET, self).__init__()
        self.conv1 = GCNConv(feature_dims, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dims, out_dims)
        # dropout是指在神经网络中丢弃掉一些隐藏或可见神经元。
        # 通常来说是在神经网络的训练阶段，每一次迭代时都会随机选择一批神经元让其被暂时忽略掉，
        # 所谓的忽略是不让这些神经元参与前向推理和后向传播。
        # BN和dropout一般不同时使用，如果一定要同时使用，可以将dropout放置于BN后面
        # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # 输出[所有节点数,hidden_dims]
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)  # 输出[所有节点数,out_dims]
        return x

class AMMN_Net(nn.Module):
    def __init__(self, origin_dims, embedding_dims, missing_case_count, num_user, num_item, gcn_data):
        super(AMMN_Net, self).__init__()
        # 初始化
        self.num_user = num_user
        self.num_item = num_item
        self.gcn_data = gcn_data
        self.edge_dim = embedding_dims * 2

        self.generalized_gcn = GCN_NET(origin_dims, embedding_dims)
        self.specific_gcn = GCN_NET(origin_dims, embedding_dims)
        self.fusion_gate = GateFusion(self.edge_dim, self.edge_dim, self.edge_dim)

        # 交叉熵损失函数会自动对输入模型的预测值进行softmax
        # 因此如果使用nn.CrossEntropyLoss()输出层无需添加softmax层
        # nn.CrossEntropyLoss()=nn.LogSoftmax()+nn.NLLLoss()
        self.trans_classifier = nn.Linear(self.edge_dim, missing_case_count)
        self.pred = nn.Linear(self.edge_dim, 2)

    def forward(self, users_tensor, items_tensor):

        nodes_generalized = self.generalized_gcn(self.gcn_data)
        users_generalized = nodes_generalized[users_tensor]
        items_generalized = nodes_generalized[items_tensor]
        generalized_features = torch.cat((torch.squeeze(users_generalized), torch.squeeze(items_generalized)), dim=1)

        nodes_specific = self.specific_gcn(self.gcn_data)
        users_specific = nodes_specific[users_tensor]
        items_specific = nodes_specific[items_tensor]
        specific_features = torch.cat((torch.squeeze(users_specific), torch.squeeze(items_specific)), dim=1)

        fused_features = self.fusion_gate(generalized_features, specific_features)

        generalized_output = self.trans_classifier(generalized_features)
        link_output = self.pred(fused_features)

        return link_output, generalized_output

    def metrics_eval(self, eval_data):
        scores = []
        labels = []
        preds = []
        for data in eval_data:
            users_tensor, items_tensor, link_labels = data
            if torch.cuda.is_available():
                with torch.no_grad():  # 不计算参数梯度
                    users_tensor = Variable(users_tensor).cuda()
                    items_tensor = Variable(items_tensor).cuda()
                    link_labels = Variable(link_labels).cuda()
            else:
                with torch.no_grad():
                    users_tensor = Variable(users_tensor)
                    items_tensor = Variable(items_tensor)
                    link_labels = Variable(link_labels)

            output, _ = self.forward(users_tensor, items_tensor)
            _, argmax = torch.max(output, 1)
            scores += list(output[:, 1].cpu().detach().numpy())
            labels += list(link_labels.cpu().detach().numpy())
            preds += list(argmax.cpu().detach().numpy())

        acc = metrics.accuracy_score(labels, preds)
        pre = metrics.precision_score(labels, preds, average='macro')
        auc = metrics.roc_auc_score(labels, scores, average='macro')
        return acc, pre, auc

