## 使用cnode上的jupyter编写代码
http://166.111.156.58:8000/ 输入自己的用户名和密码

### interact plot:
使用专为jupyter设计的ipywidgets，可以更好地展示、调整代码，尤其适合运行较快、有较多参数需要探索、选取的场合，比如绘图

以使用python的seaborn绘制boxplot为例，我们有多个参数待选，比如：

- context & style 可以设置不同的背景和字体风格
- width & height 设置图片的宽和高
- boxplot/violinplot 使用传统的box表示还是用可以同时表示点的分布密度的violinplot表示
- showdot & showbox  可以选择是否显示点，以及是否显示box，或者同时都显示、都不显示
- fondsize & dotsize，分别调整文字大小和点的大小
- boxwidth 调整box和violin的宽度
- ylim 调整纵坐标的最大与最小值
- compareheight*  调整几个注释文字距离box顶点的高度
- palettesind 选择第几套配色方案
- saturation 调整色彩饱和度

可以看到python的绘图函数一般都留有很大的调整空间，可选参数较多，可以组合出很多效果，但是也给调整带来了很多麻烦。如果把这些参数都传入绘图函数中，每次调整其中的一个或几个，依照经验来调整，效率就比较低。这个时候就可以考虑使用ipywidgets方便地调整这些参数。通过ipywidgets，可以把这些控制浮点型、整型或字符串控制参数用Float、Int slider、Dropdown，RadioButtion等控制，不需要每次更改参数再运行代码框，可以快速地调整出自己想要的图像。

效果如下：

![Markdown](http://i4.fuimg.com/640680/a5c74f6ebc23a233.png)

**一份示例代码plots_roc.ipynb放在了interact_plot下，将该文件和数据拷贝到同一个文件夹下，用jupyter打开该ipynb文件，逐代码框运行或者直接选择cell-->run all也可以一次性运行所有代码。**

### 其他jupyter使用技巧
**nbextension**,是一个神奇的插件，可以帮助提升一些效率。

目前cnode上已经安装了该插件，在登陆进去后的页面，可以看到标签栏的nbextension，点击进入后可以看到几十种插件供选择。下面列举几种比较好用的：

#### table of contents
jupyter支持markdown，只需要将某个代码框选为markdown格式，使用table of contents插件，就会自动在左边栏生成目录。对于写的很长的代码，可以帮助整理思路，快速定位代码。大家用jupyter一般是做前期的各种各样的实验，思路可能比较发散，所以用table of contents可以帮忙梳理思路，也方便以后再寻找、理解代码

**效果很棒**：
![Markdown](http://i4.fuimg.com/640680/5bfe6a9b5a48e822.png)

#### freeze
可以“冰冻”某个代码块，有个代码块儿暂时不再使用，就可以暂时冰冻，这样就无法运行，也不会被误删除。是个很有用的功能。
![Markdown](http://i4.fuimg.com/640680/a6412152652237a6.png)

#### highlighter
可以高亮注释
![Markdown](http://i4.fuimg.com/640680/5cf4c30a3f01295d.png)

#### Gist-it
Adds a button to publish the current notebook as a gist

#### Snippets menu
帮助插入一些经典库的经典方法的代码块，比去Stack Overflow搜要快一些，不过支持的还是比较少的。
![Markdown](http://i4.fuimg.com/640680/7fc861662f88de4a.png)

#### Snippet:
**推荐！**可以帮助节约一些写代码的时间

![Markdown](http://i4.fuimg.com/640680/5a2fe35422f87b5e.png)

This extension adds a drop-down menu to the IPython toolbar that allows easy insertion of code snippet cells into the current notebook. The code snippets are defined in a JSON file in nbextensions/snippets/snippets.json and an example snippet is included with this extension

Snippets are specified by adding a new JSON block to the list of existing snippets in $(jupyter --data-dir)/nbextensions/snippets/snippets.json. **(I put my customized json file in jupyter-notebooks directory in /Users/james/)** For example, to add a new snippet that imports numpy, matplotlib, and a print statement, the JSON file should be modified。

列一下自定义的各种情景下需要导入的库，总结起来就是下面这样，注意语法别错，东西一多看着还是很头疼的，配置好就可以用了。

```
{
    "snippets" : [
	{
		"name" : "science basic",
		"code" : [
			"import argparse, sys, os, errno",
			"%pylab inline",
			"import numpy as np",
			"import pandas as pd",
			"import matplotlib.pyplot as plt",
			"plt.style.use('ggplot')",
			"import seaborn as sns",
			"import h5py",
			"import os",
			"from tqdm import tqdm",
			"import scipy",
			"import sklearn",
			"from scipy.stats import pearsonr",
			"import warnings",
			"warnings.filterwarnings('ignore')"
		]
        },
	{
        	"name" : "high level plot",
        	"code" : [
                    "import matplotlib.animation as animation",
		    "from matplotlib import rc",
		    "from IPython.display import HTML, Image",
		    "rc('animation', html='html5')",
		    "import plotly",
		    "import plotly.offline as off",
		    "import plotly.plotly as py",
		    "import plotly.graph_objs as go"
                ]
        },
	{
		"name" : "deep learning",
		"code" : [
			"import keras",
			"from keras import backend as K",
			"from keras.callbacks import TensorBoard",
			"from keras.callbacks import EarlyStopping",
			"from keras.optimizers import Adam",
			"from keras.callbacks import ModelCheckpoint",
			"import tensorflow as tf",
			"from keras.models import Model",
			"from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Lambda, Dot,average,add, concatenate",
			"from keras.layers.normalization import BatchNormalization",
			"from keras.layers.core import Dropout, Activation,Reshape",
			"from keras.layers.merge import concatenate",
			"from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint",
			"from keras.initializers import RandomNormal",
			"import os",
			"os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'",
			"os.environ['CUDA_VISIBLE_DEVICES'] = '4'",
			"from keras.backend.tensorflow_backend import set_session",
			"config = tf.ConfigProto()",
			"config.gpu_options.per_process_gpu_memory_fraction = 0.99",
			"set_session(tf.Session(config=config))"
			]
		},
	{
        	"name" : "pytorch",
        	"code" : [
                    "import torch",
		    "import math",
		    "import torch.nn as nn",
		    "import torch.nn.functional as F"
                ]
        }
    ]
}
```
如果在本地和几个不同的服务器上都使用snippets，而且后续会不断加入新的希望导入的模块，可以找到snippets.json在各个服务器上的位置：

```
$(jupyter --data-dir)/nbextensions/snippets/snippets.json
```


然后写个同步的小脚本及时同步。


效果如下

![Markdown](http://i1.fuimg.com/640680/c493f72487089e34.png)


