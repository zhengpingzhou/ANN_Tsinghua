# 使用方法

- 测试环境：Windows 10 + Python 2.7

- 运行方式：首先将作业说明中提供的data/文件夹复制到src/根目录下，然后在src/目录下运行

  ```bash
  $ python run_mlp.py
  ```
- 特别说明：在run_mlp.py中提供了双隐层MLP，并在注释中提供了单隐层MLP。若需要测试单隐层MLP，只需将这两部分的注释/非注释模式交换即可。

# 修改说明

## layers.py

- Sigmoid：加入forward、backward部分代码，实现$\sigma (x) = \frac{1}{1+e^{-x}}$及$\sigma'(x) = \sigma(x) \cdot (1 - \sigma (x))$

  ```python
  def forward(self, input):
      '''Your codes here'''
      # f(x) = 1 / (1 + exp(-x))
      output = 1 / (1 + np.exp(-input))
      self._save_output_for_backward(output)
      return output

  def backward(self, grad_output):
      '''Your codes here'''
      # f'(x) = f(x)(1 - f(x))
      return grad_output * self._saved_output * (1 - self._saved_output)
  ```

- Relu：加入forward、backward部分代码，实现$relu(x) = max(x, 0)$及$relu'(x) = (x > 0)$

  ```python
  def forward(self, input):
      '''Your codes here'''
      # f(x) = max(0, x)
      self._save_input_for_backward(input)
      return np.maximum(0, input)

  def backward(self, grad_output):
      '''Your codes here'''
      # f'(x) = 0 if x <= 0 else 1
      return grad_output * np.array(self._saved_input > 0)
  ```

- Linear：加入forward、backward部分代码，实现$u = xW+b$及线性层的梯度计算（详见report）

  ```python
  def forward(self, input):
      '''Your codes here'''
      # u = x * W + b
      self._save_input_for_backward(input)
      return np.matmul(input, self.W) + self.b

  def backward(self, grad_output):
      '''Your codes here'''
      grad_input  = np.matmul(grad_output, np.transpose(self.W))
      self.grad_W = np.matmul(np.transpose(self._saved_input), grad_output)
      self.grad_b = grad_output
      return grad_input
  ```

- Layer：加入_saved_input, _saved_output, _save_input_for_backward, _save_output_for_backward，用于保存输入/输出向量，在backward求梯度时使用：

  ```python
  def __init__(self, name, trainable=False):
      self._saved_input = None
      self._saved_output = None
      
  def _save_input_for_backward(self, input):
      self._saved_input = input

  def _save_output_for_backward(self, output):
      self._saved_output = output
  ```

## loss.py

- EuclideanLoss：加入forward、backward部分代码，实现$E = \frac{1}{2N}\sum_n^N||T(n) - y(n)||^2_2$及$E' = \frac{1}{N}(T - y)$

  ```python
  def forward(self, input, target):
      '''Your codes here'''
      return 0.5 * np.mean(np.square(input - target), axis=1)

  def backward(self, input, target):
      '''Your codes here'''
      return (input - target) / len(input)
  ```

- SoftmaxCrossEntropyLoss：加入forward、backward部分代码，实现softmax交叉熵损失函数（详见report）：

  ```python
  def forward(self, input, target):
      self.softmax = np.transpose(np.transpose(np.exp(input)) / np.exp(input).sum(axis=1))
      cross_entropy = -np.mean(np.sum(target * np.log(self.softmax), axis=1))
      return cross_entropy

  def backward(self, input, target):
      return (self.softmax - target) / len(input)
  ```

## run_mlp.py

- config：加入隐层节点数超参h1\_dim和h2\_dim，方便参数调节：

  ```json
  config = {
      'h1_dim': 392,
      'h2_dim': 196,
      'learning_rate': 0.1,
      'weight_decay': 0.0002,
      'momentum': 0.0001,
      'batch_size': 100,
      'max_epoch': 1000,
      'disp_freq': 100,
      'test_epoch': 1
  }
  ```

- model：增加单、双隐层2种网络：

  ```python
  model = Network()

  ''' Network with 2 hiden layers'''
  model.add(Linear('fc1', 784, config['h1_dim'], 0.01))
  model.add(Relu('h1'))
  model.add(Linear('fc2', config['h1_dim'], config['h2_dim'], 0.01))
  model.add(Sigmoid('h2'))
  model.add(Linear('fc3', config['h2_dim'], 10, 0.01))

  ''' Network with 1 hiden layer'''
  # model.add(Linear('fc1', 784, config['h1_dim'], 0.01))
  # model.add(Relu('h1'))
  # model.add(Linear('fc2', config['h1_dim'], 10, 0.01))
  loss = EuclideanLoss(name='loss')
  ```

  ​