# 对邓小平时代的分词与词频统计
# Use THULAC for Chinese words partition
## convert to UTF-8 format
```
file ip.txt
#use vim to change encoding format
:set fileencoding=utf-8
```
## Use Jieba分词
```
import os, codecs  
import jieba  
from collections import Counter  
with codecs.open('output.txt', 'r', 'utf8') as f:  
    txt = f.read() 
seg_list = jieba.cut(txt) 
c = Counter()  
for x in seg_list:  
    if len(x)>1 and x != '\r\n':  
        c[x] += 1
np.savetxt('count10000.txt',np.array(c.most_common(10000)),fmt='%s')
```

### 表格
绘制词频靠前的统计表格
```
import os, codecs  
#import jieba  
import numpy as np
from collections import Counter 
import matplotlib.pylab as plt
plt.style.use('ggplot')
import pandas as pd
import matplotlib
plt.rcParams['font.style'] = u'normal'
plt.rcParams['font.family'] = u'Microsoft YaHei'
with codecs.open('output.txt', 'r', 'utf8') as f:  
    txt = f.read() 
seg_list = jieba.cut(txt) 
c = Counter()  
for x in seg_list:  
    if len(x)>1 and x != '\r\n':  
        c[x] += 1
np.savetxt('count10000.txt',np.array(c.most_common(10000)),fmt='%s')
data = np.loadtxt('count10000.txt',dtype='str')
with codecs.open('output.txt', 'r', 'utf8') as f:  
    txt = f.read() 
wordlist = np.array(txt.split(' '))
#wordlist.shape
countlist = []
for i in range(10000):
    countlist.append(data[i,0]+': '+str(data[i,1]))
pd.DataFrame(np.array(countlist)[:200].reshape(20,10)).head()
```
[matplotlib中文显示解决方案](https://segmentfault.com/a/1190000005144275)

![Markdown](http://i4.bvimg.com/640680/f405c02d19042f6b.png)

### 画图
#### 画bar图
```
namelist = [u'邓小平',u'中国',u'毛泽东',u'工作',u'干部',u'问题',u'北京',u'美国',u'领导人',u'会议',u'经济',u'关系',u'香港',u'1975',u'领导',u'胡耀邦',u'苏联',u'政治',u'支持',
u'军队',u'陈云',u'政策',u'赵紫阳',u'周恩',u'讲话',u'学生',u'华国锋',u'改革',u'日本']
index_25 = [0,1,3,4,6,7,8,11,12,14,16,20,21,23,26,27,28,31,33,37,39,40,42,46,47,48,49]
count = 27
fig,ax=plt.subplots(1,figsize=(20,10))
ax.bar(range(count),data[index_25,1].astype('int'),color = 'b')
#ax.bar(range(count),data[:count,1].astype('int'))
ax.set_xticks(range(count))
ax.set_xticklabels(namelist)
#plt.savefig('tst.png')
ax.set_title(str(count)+' key words frequency in book')
```

![Markdown](http://i4.bvimg.com/640680/9190c33461d494bd.png)

#### 画重要词汇不同章节的变化图，通过行划分章节统计
**找到不同章节的起始位点**
- [x] 画heatmap图
```
chapterind = np.array([16590,  31267,  54053,69769,  90171, 104745,121010, 138136,  147048,  161724,170963,  193593, 206502, 214129,230193, 
                245828,  260400,285768, 303284, 324922, 337101, 349241, 362426, 377184])-1
def count_frequent(chap):
    freqlist =[]
    if chap <23:
        for i in range(27):
            freqlist.append(np.where(wordlist[chapterind[chap]:chapterind[chap+1]] ==namelist[i])[0].shape[0])
    else:
        for i in range(27):
            freqlist.append(np.where(wordlist[chapterind[chap]:] ==namelist[i])[0].shape[0])
    return np.array(freqlist)

freq_var = np.ndarray([24,27])
for i in range(24):
    freq_var[i] = count_frequent(i)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
transformed = scaler.fit_transform(freq_var)

fig,ax=plt.subplots(1,figsize =(10,10))
ax.matshow(transformed.T ,cmap ='jet')
ax.set_title('27 key words fluctuation in 24 chapters')
ax.set_xticks(range(24))
ax.set_yticks(range(27))
ax.set_yticklabels(namelist)
```
为了不受影响，heatmap逐行做归一化
![Markdown](http://i4.bvimg.com/640680/231ecc794e04d4d7.png)
- [ ] 分析  

- [x] 画折线图
```
fig,ax=plt.subplots(1,figsize =(20,10))
#ax.plot(freq_var[:,:10])
count =10
for x,y in zip(freq_var[:,:count].T,namelist[:count]):
    plt.plot(x,label =y)
plt.title(str(count)+' key words fluctuation in 24 chapters')
plt.legend()
plt.show()
```

![Markdown](http://i4.bvimg.com/640680/c79d691ce875618e.png)
### 高级花样  比如谁和谁容易一起出现
 已知所有词位置  任意两个词比较，用数量少的词，每个位置找离得最近的，计算距离  做差绝对值的最小值   返回   
**可以看距离分布  平均数   Boxplot**
用repeat 生成两个array做差然后绝对值最小值应该比循环快很多
**如何定义这种距离？**
```
def calculate_distance(ind1,ind2):
    pos1 = np.where(wordlist==namelist[ind1])[0]
    pos2 = np.where(wordlist==namelist[ind2])[0]
    num1 ,num2 = pos1.shape[0],pos2.shape[0]
    if num1>num2:
        small = num2
        large = num1
        lararr = pos1
        smarr = pos2
    else:
        small = num1
        large = num2
        lararr = pos2
        smarr = pos1
    disarr = np.ndarray([small,large])  #each line calculate the small set's ith word's and large set's every words distance
    arr1= np.repeat(smarr,large).reshape(-1,large)
    arr2= np.repeat(lararr,small).reshape(-1,small).T
    mindis = np.min(np.abs(arr2-arr1),axis=1)
    return mindis

def draw_dist_count(ind1,ind2):
    fig,ax=plt.subplots(1,figsize=(20,10))
    ax.bar(range(calculate_distance(0,1).shape[0]),calculate_distance(0,1),color='g')
    ax.set_title('Minimum Distance of '+namelist[ind1]+" and "+namelist[ind2])
draw_dist_count(0,1)
```
![Markdown](http://i4.bvimg.com/640680/a421383dc7da2619.png)
**两个人或事物之间的关联程度，随书籍的变化情况**
```
fig,ax=plt.subplots(4,2,figsize=(20,20))
for i in range(4):
    for j in range(2):
        ax[i,j].bar(range(calculate_distance(0,1+2*i+j).shape[0]),calculate_distance(0,1+2*i+j))
        ax[i,j].set_title('Minimum Distance of '+namelist[0]+" and "+namelist[1+2*i+j])
```
![Markdown](http://i4.bvimg.com/640680/4414d03a5b228e77.png)
- [ ] **Hist of distance**
```
fig,ax=plt.subplots(4,2,figsize=(20,20))
for i in range(4):
    for j in range(2):
        ax[i,j].hist(calculate_distance(0,1+2*i+j),bins =50,color='b',alpha=0.4)
        ax[i,j].set_title('Minimum Distance of '+namelist[0]+" and "+namelist[1+2*i+j])
```
![Markdown](http://i4.bvimg.com/640680/3c6048942654bc2c.png)
- [ ] **Boxplot**
```
dist_data = {}
for i in np.arange(1,20):
    dist_data[i] = calculate_distance(0,i)
dataframe_dxp = pd.concat((pd.DataFrame({namelist[i]:dist_data[i]}) for i in np.arange(1,20)),axis=1)
import seaborn as sns
fig, ax = plt.subplots(figsize=(100,20))
sns.boxplot(data =dataframe_dxp,ax=ax,boxprops=dict(alpha=.5),color='g')
ax.set_title(u'Dengxiaoping and others',fontsize=80)
ax.set_xticks(range(19))
ax.set_xticklabels(namelist[1:20],fontsize=80)
fig.savefig('boxplot.png')
```
![Markdown](http://i4.bvimg.com/640680/5ef0af735a848eca.png)



