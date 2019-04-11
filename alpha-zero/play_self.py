
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
import random
​
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
# GOMOKU
SPACE = 0.
BLACK = 1.
WHITE = 2.
SIZE = 8

# MCTS
CPUCT = 5
MCTSSIMNUM = 40
HISTORY = 3
TEMPTRIG = 8

# Dirichlet
DLEPS = .25
DLALPHA = .03

# Net params
IND = HISTORY * 2 + 2
OUTD = SIZE**2
BLOCKS = 10
RES_BLOCK_FILLTERS = 128

# Train params
USECUDA = torch.cuda.is_available()
EPOCHS = 5
GAMETIMES = 3000
CHECKOUT = 50
EVALNUMS = 20
MINIBATCH = 512
WINRATE = .55
TRAINLEN = 10000

# Optim
LR = 0.03
L2 = 0.0001
# GOMOKU
SPACE = 0.
BLACK = 1.
WHITE = 2.
SIZE = 8
​
# MCTS
CPUCT = 5
MCTSSIMNUM = 40
HISTORY = 3
TEMPTRIG = 8
​
# Dirichlet
DLEPS = .25
DLALPHA = .03
​
# Net params
IND = HISTORY * 2 + 2
OUTD = SIZE**2
BLOCKS = 10
RES_BLOCK_FILLTERS = 128
​
# Train params
USECUDA = torch.cuda.is_available()
EPOCHS = 5
GAMETIMES = 3000
CHECKOUT = 50
EVALNUMS = 20
MINIBATCH = 512
WINRATE = .55
TRAINLEN = 10000
​
# Optim
LR = 0.03
L2 = 0.0001
网络
class ResBlockNet(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,#128
               block_filters=RES_BLOCK_FILLTERS,#128
               kr_size=3,
               stride=1,
               padding=1,
               bias=False):

    super().__init__()

    self.layers = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
        nn.ReLU(),
        nn.Conv2d(block_filters, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
    )

  def forward(self, x):
    #print("ResBlockNet forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    res = x#这里的res是拿来干什么的呢？
	#为什么特征数据要经过这样的处理呢？
    out = self.layers(x) + x#那这里传入的x又是什么呢？x应该是网络的输入数据的，难道这里把神经网络作为输入数据吗？
    #处理之后还给他一个relu函数
    return F.relu(out)#activation function
class ResBlockNet(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,#128
               block_filters=RES_BLOCK_FILLTERS,#128
               kr_size=3,
               stride=1,
               padding=1,
               bias=False):
​
    super().__init__()
​
    self.layers = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
        nn.ReLU(),
        nn.Conv2d(block_filters, block_filters, kr_size,
                  stride=stride,
                  padding=padding,
                  bias=bias),
        nn.BatchNorm2d(block_filters),
    )
​
  def forward(self, x):
    #print("ResBlockNet forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    res = x#这里的res是拿来干什么的呢？
    #为什么特征数据要经过这样的处理呢？
    out = self.layers(x) + x#那这里传入的x又是什么呢？x应该是网络的输入数据的，难道这里把神经网络作为输入数据吗？
    #处理之后还给他一个relu函数
    return F.relu(out)#activation function
class Feature(nn.Module):
  def __init__(self,
               ind=IND,#8
               outd=RES_BLOCK_FILLTERS):#RES_BLOCK_FILLTERS 128，

    super().__init__()#继承父类的属性,模型里面的一个环节
    #Sequential在里面搭建神经网络
    self.layer = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, outd,#torch Conv2d 卷积
                  stride=1,#步长
                  kernel_size=3,#kernel_size
                  padding=1,#每一维补0的数量
                  bias=False),#偏置
        nn.BatchNorm2d(outd),#没有affine为带有线性的参数
        nn.ReLU(),#激活函数
    )
    self.encodes = nn.ModuleList([ResBlockNet() for _ in range(BLOCKS)])#下面的total 9层，为什么要定义九个那么多层呢？

  def forward(self, x):
    #print("Feature forward x is {0}".format(x))#不会直接调用到这个函数吗？torch.Size([1, 8, 8, 8])
    x = self.layer(x)#传入输入参数
    #print("self.layer(x) is {0}".format(x))#torch.Size([1, 128, 8, 8]),为什么从里面出来的数据就是那个样子了呢？
    for enc in self.encodes:
      x = enc(x)#每层传入参数，大概是这个意思吧！
    return x
class Feature(nn.Module):
  def __init__(self,
               ind=IND,#8
               outd=RES_BLOCK_FILLTERS):#RES_BLOCK_FILLTERS 128，
​
    super().__init__()#继承父类的属性,模型里面的一个环节
    #Sequential在里面搭建神经网络
    self.layer = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, outd,#torch Conv2d 卷积
                  stride=1,#步长
                  kernel_size=3,#kernel_size
                  padding=1,#每一维补0的数量
                  bias=False),#偏置
        nn.BatchNorm2d(outd),#没有affine为带有线性的参数
        nn.ReLU(),#激活函数
    )
    self.encodes = nn.ModuleList([ResBlockNet() for _ in range(BLOCKS)])#下面的total 9层，为什么要定义九个那么多层呢？
​
  def forward(self, x):
    #print("Feature forward x is {0}".format(x))#不会直接调用到这个函数吗？torch.Size([1, 8, 8, 8])
    x = self.layer(x)#传入输入参数
    #print("self.layer(x) is {0}".format(x))#torch.Size([1, 128, 8, 8]),为什么从里面出来的数据就是那个样子了呢？
    for enc in self.encodes:
      x = enc(x)#每层传入参数，大概是这个意思吧！
    return x
class Value(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,
               outd=OUTD,#OUTD 为64
               hsz=256,
               kernels=1):
    super().__init__()

    self.outd = outd#这个是没有包含在网络里的，但是他在这个类里面，后续将他铺成一维数据

    self.conv = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )

    self.linear = nn.Sequential(
        nn.Linear(outd, hsz),#第一个输入,后面一个参数是输出
        nn.ReLU(),
        nn.Linear(hsz, 1),
        nn.Tanh(),#和sigmod类似的一个激活函数
    )

    self._reset_parameters()

  def forward(self, x):
    #print("Value forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    x = self.conv(x)#这里也是经过一层网络
    #print("self.conv(x) is {0}".format(x))#torch.Size([1, 1, 8, 8])
    x = x.view(-1, self.outd)#
    #print(" x.view(-1, self.outd) is {0}".format(x))#torch.Size([1, 64])
    return self.linear(x)#那返回的是什么呢？linear是上面那个定义的网络

  def _reset_parameters(self):#重置他的参数是什么意思呢？
    for layer in self.modules():
      if type(layer) == nn.Linear:
        layer.weight.data.uniform_(-.1, .1)
class Value(nn.Module):
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,
               outd=OUTD,#OUTD 为64
               hsz=256,
               kernels=1):
    super().__init__()
​
    self.outd = outd#这个是没有包含在网络里的，但是他在这个类里面，后续将他铺成一维数据
​
    self.conv = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )
​
    self.linear = nn.Sequential(
        nn.Linear(outd, hsz),#第一个输入,后面一个参数是输出
        nn.ReLU(),
        nn.Linear(hsz, 1),
        nn.Tanh(),#和sigmod类似的一个激活函数
    )
​
    self._reset_parameters()
​
  def forward(self, x):
    #print("Value forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    x = self.conv(x)#这里也是经过一层网络
    #print("self.conv(x) is {0}".format(x))#torch.Size([1, 1, 8, 8])
    x = x.view(-1, self.outd)#
    #print(" x.view(-1, self.outd) is {0}".format(x))#torch.Size([1, 64])
    return self.linear(x)#那返回的是什么呢？linear是上面那个定义的网络
​
  def _reset_parameters(self):#重置他的参数是什么意思呢？
    for layer in self.modules():
      if type(layer) == nn.Linear:
        layer.weight.data.uniform_(-.1, .1)
class Policy(nn.Module):#policy是在这里
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,#128
               outd=OUTD,#64
               kernels=2):

    super().__init__()

    self.out = outd * kernels#转换成一个线性的东西？why要这样做呢?

    self.conv = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )

    self.linear = nn.Linear(kernels * outd, outd)#转换成一个线性的东西？why要这样做呢?
    self.linear.weight.data.uniform_(-.1, .1)#这个应该是固定在一个范围内

  def forward(self, x):
    #print("Policy forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    x = self.conv(x)
    #print("self.conv(x) is {0}".format(x.shape))#torch.Size([1, 2, 8, 8])
    x = x.view(-1, self.out)#view只是将他铺成一维
    #print("x.view(-1, self.out) is {0}".format(x.shape))#torch.Size([1, 128])
    x = self.linear(x)
    #print("self.linear(x) is {0}".format(x.shape))#torch.Size([1, 64])
    return F.log_softmax(x, dim=-1)#softmax的最后多做一个log的操作
class Policy(nn.Module):#policy是在这里
  def __init__(self,
               ind=RES_BLOCK_FILLTERS,#128
               outd=OUTD,#64
               kernels=2):
​
    super().__init__()
​
    self.out = outd * kernels#转换成一个线性的东西？why要这样做呢?
​
    self.conv = nn.Sequential(#相当于keras的惯序模型，会自动调用forward方法
        nn.Conv2d(ind, kernels, kernel_size=1),
        nn.BatchNorm2d(kernels),
        nn.ReLU(),
    )
​
    self.linear = nn.Linear(kernels * outd, outd)#转换成一个线性的东西？why要这样做呢?
    self.linear.weight.data.uniform_(-.1, .1)#这个应该是固定在一个范围内
​
  def forward(self, x):
    #print("Policy forward x is {0}".format(x.shape))#torch.Size([1, 128, 8, 8])
    x = self.conv(x)
    #print("self.conv(x) is {0}".format(x.shape))#torch.Size([1, 2, 8, 8])
    x = x.view(-1, self.out)#view只是将他铺成一维
    #print("x.view(-1, self.out) is {0}".format(x.shape))#torch.Size([1, 128])
    x = self.linear(x)
    #print("self.linear(x) is {0}".format(x.shape))#torch.Size([1, 64])
    return F.log_softmax(x, dim=-1)#softmax的最后多做一个log的操作
#构建的网络在这里
class Net(nn.Module):#使用pytorch搭建网络必备的
  def __init__(self):
    super().__init__()
​
    self.feat = Feature()#特征是什么特征呢？
    #print(" Net self.feat is {0}".format(self.feat))
    self.value = Value()
    #print(" Net self.value is {0}".format(self.value))
    self.policy = Policy()
    #print(" Net self.policy is {0}".format(self.policy))
​
  def forward(self, x):#这个是前向传播函数
    #x包含6个历史和当前棋盘和最终落子
    #print("Net forward x is {0}".format(x))#这个的输入数据也是0,torch.Size([1, 8, 8, 8])
    feats = self.feat(x)#返回的应该是一个网络
    #print("Net forward feats is {0}".format(feats))#torch.Size([1, 128, 8, 8])
    winners = self.value(feats)#特征值经过feature网路已经被处理了，这个是不是就是获胜的概率了呢？确实是
    #print("Net forward winners is {0}".format(winners.shape))#([1, 1])
    #print("Net forward winners is {0}".format(winners))#tensor([[-0.7781]], grad_fn=<TanhBackward>)
    props = self.policy(feats)#传入的也是feats
​
    return winners, props
​
  def save_model(self, path="model.pt"):#默认的名称
    torch.save(self.state_dict(), path)#只是存储参数
​
  def load_model(self, path="model.pt", cuda=True):
    #print("Net load_model cuda is {0}".format(cuda))#cuda的值一直都是false
    if cuda:
      self.load_state_dict(torch.load(path))
      self.cuda()
    else:#go here
      #在gpu上训练的模型在cpu上运行,获取到的参数是直接被放在网络里面,如何查看他的参数呢？
      self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))#加载他的参数,
      self.cpu()#将所有参数和buffer移动到cpu
board
class Board(object):
    def __init__(self,
                 size=SIZE,
                 hist_num=HISTORY,
                 c_action=-1,
                 player=BLACK):
​
        self.size = size#size是8
        self.c_action = c_action#action是-1,为什么是-1呢?这个应该是整个棋盘上的位置索引
        self.hist_num = hist_num#self.hist_num值是3，但是为什么是3呢？
        self.valid_moves = list(range(size**2))#这个应该是8的平方
        self.invalid_moves = []
        self.board = np.zeros([size, size])#board是8×8,上面有黑子是1，白子是2，没有棋子是0
        self.c_player = player#棋手应该是黑子或者是白子
        #print("self.size is {0}\t self.c_action is {1}\t self.hist_num is {2}".format(self.size, self.c_action, self.hist_num))
        #print("self.valid_moves is {0}".format(self.valid_moves))
        self.players = {"black": BLACK, "white": WHITE}#这个字典拿来有什么作用呢？
​
        # BLACK -> 0 | WHITE -> 1
        #基本上就多存储了3个棋盘
        self.history = [np.zeros((hist_num, size, size)),
                        np.zeros((hist_num, size, size))]
        #print("self.history is {0}".format(self.history))
​
    # private method
    def _mask_pieces_by_player(self, player):
        '''binary feature planes'''
        #print("_mask_pieces_by_player self.board is {0} player is {1}".format(self.board, player))
        new_board = np.zeros([self.size, self.size])
        new_board[np.where(self.board == player)] = 1.
        return new_board
​
    def _get_piece(self, x, y):#这个应该是返回0或者是返回1
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.board[x, y]
        return SPACE
​
    def _is_space(self, x, y):
        assert 0 <= x < self.size and 0 <= y < self.size
        return self.board[x, y] == SPACE
​
    @property
    def last_player(self):
        if self.c_player == self.players["white"]:
            return self.players["black"]
        return self.players["white"]
​
    def clone(self):#为什么要clone一个棋盘呢?
        c_board = Board(size=self.size,
                        hist_num=self.hist_num,
                        player=self.c_player,
                        c_action=self.c_action)
​
        c_board.valid_moves = self.valid_moves.copy()
        c_board.invalid_moves = self.invalid_moves.copy()
        c_board.board = self.board.copy()
        c_board.history = [h.copy() for h in self.history]
​
        return c_board
​
    def move(self, action):
        x, y = action // self.size, action % self.size
        assert self._is_space(x, y)
​
        self.valid_moves.remove(action)#
        self.invalid_moves.append(action)#里面存储的是已经移动的或者是即将马上移动的action
        self.c_action = action
        self.board[x, y] = self.c_player
        #print("before self.history is {0}".format(self.history))
        p_index = int(self.c_player - BLACK)
        self.history[p_index] = np.roll(self.history[p_index], 1, axis=0)
        self.history[p_index][0] = self._mask_pieces_by_player(self.c_player)
        #print("after self.history is {0}".format(self.history))
​
    #以最后落子为中心判断横竖斜方向有没有同样颜色的五颗棋子
    def is_game_over(self, player=None):#如何去定义这个game is over 的规则呢？
        x, y = self.c_action // self.size, self.c_action % self.size#第一个是整除，第二个是求余数
        #print("is_game_over x is {0}, y is {1} self.c_action is {2}".format(x, y, self.c_action))
        if player is None:
            player = self.c_player#默认的是黑子
​
        #这个是指横向的有5个连续的全部为一个颜色
        for i in range(x - 4, x + 5):#range是只包含前面，不包含后面
            if self._get_piece(i, y) == self._get_piece(i + 1, y) == self._get_piece(i + 2, y) == self._get_piece(i + 3, y) == self._get_piece(i + 4, y) == player:
                return True
        
        #垂直方向上
        for j in range(y - 4, y + 5):
            if self._get_piece(x, j) == self._get_piece(x, j + 1) == self._get_piece(x, j + 2) == self._get_piece(x, j + 3) == self._get_piece(x, j + 4) == player:
                return True
​
        #左到右的方向上
        j = y - 4
        for i in range(x - 4, x + 5):
            if self._get_piece(i, j) == self._get_piece(i + 1, j + 1) == self._get_piece(i + 2, j + 2)== self._get_piece(i + 3, j + 3) == self._get_piece(i + 4, j + 4) == player:
                return True
            j += 1
​
        #右到左边的方向上
        i = x + 4
        for j in range(y - 4, y + 5):
            if self._get_piece(i, j) == self._get_piece(i - 1, j + 1) == self._get_piece(i - 2, j + 2) == self._get_piece(i - 3, j + 3) == self._get_piece(i - 4, j + 4) == player:
                return True
            i -= 1
​
        return False
​
    def is_draw(self):#棋盘是否已经全部被画满了
        #print("self.board is {0} SPACE is {1}".format(self.board, SPACE))
        index = np.where(self.board == SPACE)#这个应该是查找语句
        #print("is_draw is {0}\n".format(index))#为什么这里是两个数组？？？？
        return len(index[0]) == 0
​
    def gen_state(self):#获取当前棋局的特征，包括历史盘面，当前棋手和最后落子
        to_action = np.zeros((1, self.size, self.size))
        #print("before to_action is {0}".format(to_action))
        to_action[0][self.c_action // self.size,#为什么这里要赋值为1呢？
                     self.c_action % self.size] = 1.#最后落子的位置使用1来表示
        #print("after to_action is {0}".format(to_action))
        #to_play表示当前棋手吗？黑子表示0,白子使用1来表示
        to_play = np.full((1, self.size, self.size), self.c_player - BLACK)#np.full()函数可以生成初始化为指定值的数组
        #print("to_play is {0}".format(to_play))
        state = np.concatenate(self.history + [to_play, to_action], axis=0)#基本上是8个8×8的棋盘，拼接
        #print("gen_state state is {0}".format(state))
​
        return state
​
    def trigger(self):#trigger是触发，这里是触发什么呢？cplay吗？c_player之前不是已经被赋值了嘛！
        self.c_player = self.players["black"] if self.c_player == self.players["white"] else self.players["white"]
        #print("board trigger and self.c_player is {0}".format(self.c_player))#这个是触发下一个需要走的棋子
​
    def show(self):
        for x in range(self.size):
            print("{0:8}".format(x), end='')
        print('\r\n')
​
        for row in range(self.size):
            print("{:4d}".format(row), end='')
            for col in range(self.size):
                if self.board[row, col] == SPACE:
                    print("-".center(8), end='')
                elif self.board[row, col] == BLACK:
                    print("O".center(8), end='')
                else:
                    print("X".center(8), end='')
            print('\r\n\r\n')
TreeNode
class TreeNode(object):
    def __init__(self,
                 action=None,
                 props=None,
                 parent=None):
​
        self.parent = parent#估计这两个都是传入的节点
        self.action = action
        self.children = []
        self.P = props  # prior probability#先验概率
        
        self.N = 0  # visit count
        self.Q = .0  # mean action value#这里难道是指动作函数吗？
        self.W = .0  # total action value
​
​
    def is_leaf(self):
        return len(self.children) == 0#子树的数量是否为0，子树使用一个list来表示
​
    def select_child(self):#子节点选择q+u值最大的值
        index = np.argmax(np.asarray([c.uct() for c in self.children]))
        return self.children[index]
​
    def uct(self):#原来他是这里定义的函数， CPUCT是5
        return self.Q + self.P * CPUCT * (np.sqrt(self.parent.N) / (1 + self.N))
​
    def expand_node(self, props):#先验证概率里面包含了action和prop吗？
        #print("TreeNode props is {0}".format(props))
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        #是产生了64个treeNode
        self.children = [TreeNode(action=action, props=p, parent=self)
                         for action, p in enumerate(props) if p > 0.]#实际上就产生了64个对象
        for action, p in enumerate(props):
            pass
            #print("action is {0} \t p is {1}".format(action, p))
        #print("expand_node self.children is {0}".format(self.children))
​
    def backup(self, v):#backup是什么意思呢？
        self.N += 1
        self.W += v#v应该是传入的q值
        self.Q = self.W / self.N
​
import numpy as np
dirichlet01=np.random.dirichlet((1,1,1,1,1,1))
print(dirichlet01)
​
[0.14520761 0.40009315 0.06961225 0.3031438  0.07042522 0.01151797]
MonteCarloTreeSearch
#
def to_tensor(x, use_cuda=USECUDA, unsqueeze=False):
    x = torch.from_numpy(x).type(torch.Tensor)
    if use_cuda:#默认是不使用的
        x = x.cuda()
​
    if unsqueeze:
        x = x.unsqueeze(0)
    #print("to_tensor x is {0}".format(x))
    return x
​
​
def to_numpy(x, use_cuda=True):
    if use_cuda:
        return x.data.cpu().numpy().flatten()
    else:
        return x.data.numpy().flatten()#这个应该是转换成一维
class MonteCarloTreeSearch(object):
    def __init__(self, net,
                 ms_num=MCTSSIMNUM):
​
        self.net = net
        self.ms_num = ms_num#self.ms_num数值是400
        print("self.ms_num is {0}".format(self.ms_num))
​
    """
    1、从根节点开始往下搜索直到叶节点
    2、将当前棋面使用神经网络给出落子概率和价值评估
    3、然后从叶节点返回到根节点一路更新
    """
    def search(self, borad, node, temperature=.001):
        self.borad = borad
        self.root = node#节点
​
        for _ in range(self.ms_num):
            node = self.root
            borad = self.borad.clone()#为什么这里需要clone一个board?
            #print("node is {0} borad is {1}, num is {2}".format(node, borad, _))
            
            #print("node.is_leaf is {0}".format(node.is_leaf()))
            while not node.is_leaf():#node.is_leaf()返回true或者false
                #print("while node.is_leaf is {0}".format(node.is_leaf()))#先暂时不走这里
                node = node.select_child()
                borad.move(node.action)#移动到clone的棋盘上
                borad.trigger()#开始另外一个棋子开始移动,实际在borad上只更新了一步
            #print("search borad show and num is {0}".format(_))
            borad.show()
​
            # be carefull - opponent state
            #已经有一部分forward开始被调用了
            """
            Zero的net输入为历史盘面和当前盘面特征，二进制格式，即0和1，输出策略p和价值v，
            其中p为在棋盘上每个点落子的概率，v为评估当前盘面下当前玩家胜利的概率。
            """
            #net的先将他作为一个黑盒子
            value, props = self.net(#应该是在他的前向函数里面进行返回的
                to_tensor(borad.gen_state(), unsqueeze=True))#unsqueeze是处理成二维数据,不知道这里是不是
            #print("before MonteCarloTreeSearch value is {0}".format(value))#tensor([[0.0450]], grad_fn=<TanhBackward>)
            #print("before MonteCarloTreeSearch props is {0}".format(props))#torch.Size([1, 64])
​
            value = to_numpy(value, USECUDA)[0]#USECUDA faise,这个应该是转换成np数据
            #print("value is {0} USECUDA is {1}".format(value, USECUDA))
            props = np.exp(to_numpy(props, USECUDA))#np.exp计算e的多少次方
            #print("after MonteCarloTreeSearch value is {0}".format(value))
            #print("after MonteCarloTreeSearch props is {0}".format(props))#(64,)#原来的props都计算了e的props次方
            
​
            
            # add dirichlet noise for root node# dirichlet狄氏噪音，这是个什么鬼呢？
            #print("node.parent is {0}\t borad.invalid_moves is {1}, node.parent is {2}".format(node.parent, borad.invalid_moves, node.parent))
            if node.parent is None:#第一次这里返回的是none
                props = self.dirichlet_noise(props)
​
            # normalize，这里如何进行正则化呢？
            #print("before now prop is {0}, borad.invalid_moves is {1}".format(props, borad.invalid_moves))
            props[borad.invalid_moves] = 0.
            #print("after now prop is {0}, borad.invalid_moves is {1}".format(props, borad.invalid_moves))
            total_p = np.sum(props)#所有概率总和
            #print("total_p is {0}\t props is {1}".format(total_p, props))#props实际上是一个list
​
            if total_p > 0:
                props /= total_p#why？
    
            # winner, draw or continue
            if borad.is_draw():#如果棋盘已经全部被画完了，那实际上游戏终止了，不此时应该是平局
                #print("enter value = 0.")
                value = 0.#平均的话，value就是0
            else:
                #print("not enter value = 0.\t borad.last_player is {0}".format(borad.last_player))
                done = borad.is_game_over(player=borad.last_player)#这个是在何处被更新呢？我去，这个就是当前的c_player啊！
                #print("done is {0}".format(done))
                if done:#输了，就是-1
                    value = -1.#最后一个下棋的，难道不是当前的player吗？如果是当前的play导致他赢了，那他不应该是1吗？不最后一个应该是他的对手
                else:#下面应该有更新c_player的地方
                    node.expand_node(props)#需要扩展这个node吗？
​
            while node is not None:#这里应该是更新mcts
                value = -value#为什么这里是负数
                node.backup(value)#q值成负数了
                node = node.parent
                #print("search and node is {0}".format(node is not None))
​
​
        action_times = np.zeros(borad.size**2)#动作的次数
        for child in self.root.children:
            action_times[child.action] = child.N
        #print("action_times is {0}".format(action_times))#(64,)#更加子树的times难道不需要去统计吗？
​
        action, pi = self.decision(action_times, temperature)
        #print("search action is {0}, pi is {1}".format(action, pi))
        for child in self.root.children:
            if child.action == action:
                #print("search pi is {0}\n child is {1}".format(pi,child))
                return pi, child#返回应该是下一个节点
​
    @staticmethod
    def dirichlet_noise(props, eps=DLEPS, alpha=DLALPHA):#DLEPS 0.25，DLALPHA 0.03
        #np.random.dirichlet((1,1,1,1,1,1))产生dirichlet分布的数据
        return (1 - eps) * props + eps * np.random.dirichlet(np.full(len(props), alpha))
​
    @staticmethod
    def decision(pi, temperature):#根据pi产生以一定的概率去选择动作
        #temp -- temperature parameter in (0, 1] that controls the level of exploration
        pi = (1.0 / temperature) * np.log(pi + 1e-10)#temperature用来控制探索的水平
        #下面这两步就是softmax函数
        pi = np.exp(pi - np.max(pi))
        pi /= np.sum(pi)
        action = np.random.choice(len(pi), p=pi)# np.arange(5) 中产生一个size为3的随机采样:
        return action, pi
​
Play
load_model
class Play(object):
    def __init__(self):
        net = Net()
        if USECUDA:#这个为false
            net = net.cuda()
        net.load_model("model.pt", cuda=USECUDA)
        self.net = net
        self.net.eval()#这样会打印出网络的结构，如果是字典的话，就是求里面的数据
        #print("Play __init__ self.net.eval() is {0}".format(self.net.eval()))
​
    def go(self):
        print("One rule:\r\n Move piece form 'x,y' \r\n eg 1,3\r\n")
        print("-" * 60)
        print("Ready Go")
​
        mc = MonteCarloTreeSearch(self.net, 1000)
        node = TreeNode()
        board = Board()
​
        while True:
            print("Play board.c_player is {0}".format(board.c_player))#白子走
            if board.c_player == BLACK:
                action = input(f"Your piece is 'O' and move: ")
                action = [int(n, 10) for n in action.split(",")]
                action = action[0] * board.size + action[1]
                print("Play and action is {0}".format(action))#1-8,2-16这样依次下去
                next_node = TreeNode(action=action)#如果不传入参数的话，里面的值就是None，就是默认的
            else:#上一步有了trigger的动作，所以下一次循环就开始c_player = white
                _, next_node = mc.search(board, node)#白子需要在
​
            board.move(next_node.action)
            board.show()
​
            next_node.parent = None
            node = next_node
​
            if board.is_draw():
                print("board bas all been drawed\n")
                print("-" * 28 + "Draw" + "-" * 28)
                return
​
            if board.is_game_over():#游戏结束
                if board.c_player == BLACK:
                    print("-" * 28 + "Win" + "-" * 28)
                else:
                    print("-" * 28 + "Loss" + "-" * 28)
                return
​
            board.trigger()
main
if __name__ == "__main__":
    p = Play()
    p.go()
