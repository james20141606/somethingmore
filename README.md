

# Something More

- [Bioinfo machine learning tutorial](https://github.com/james20141606/somethingmore/blob/master/bioinfo.ipynb)
- [Data Mining - Deng Era](https://github.com/james20141606/somethingmore/tree/master/datamining_dxp)

# 统计画图部分
## 工具
使用python进行表格的处理和画图，需要额外安装package: plotly

```
pip install plotly
```
可使用jupyter notebook打开.ipynb文件，逐个代码框运行即可查看表格和绘图，直接在github上查看时无法看到animation和用plotly绘制的图片。

## 步骤
### 表格处理
首先将summary.txt读入，使用numpy和pandas进行数据的处理，以及计算平均值，重新写入.csv文件。注意平均值的计算要分别统计绝对数值再求比例进而求平均

### 绘图
分别绘制mapping ratio和length distribution的图
#### mapping ratio
这部分尝试了pie chart来展示整体的平均ratio，又尝试使用plotly、matplotlib的animation模块、以及seaborn的boxplot分别展示细节的样本的ratio分布。

#### length distribution
这部分尝试了使用matplotlib的3D模块、plotly以及折线图展示length distribution。

可以发现几种方法各有优劣，对数据进行了不同维度的展示，其中plotly的交互方式非常先进，可以用鼠标查看更多数据的细节。

<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2:before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3:before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4:before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5:before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6:before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
</style>
