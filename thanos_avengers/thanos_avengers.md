
**Question:**
六人组，每人带一宝石手套，去找灭霸决战，到了草屋，七人围圈而坐，六对一，灭霸人少，给予先打响指的权利，灭霸打后，超级英雄全化灰就灭霸胜，没全化灰就由一个活着的超级英雄打一个响指，看二分之一消失的概率能不能把灭霸打成灰。没成灰就继续第二回合，灭霸继续优先打响指 </br>
问灭霸死的概率是多少


```python
import numpy as np
%pylab inline
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import pandas as pd
np.random.seed(1111)
#setup figure template

figure_template_path = 'bin'
if figure_template_path not in sys.path:
    sys.path.append(figure_template_path)
from importlib import reload
import figure_template
#force reload of the module
reload(figure_template)
from figure_template import display_dataframe, embed_pdf_figure, embed_pdf_pages,std_plot
```

    Populating the interactive namespace from numpy and matplotlib



```python
np.random.randint(0,2,size=6)
```




    array([0, 1, 1, 1, 0, 0])




```python
def check_survive(input_num):
    '''
    input_num: how many people to be decided to survive by stones
    status: 0 means dead, 1 means alive
    '''
    random_output = np.random.randint(0,2,size=input_num)
    alive = random_output.sum()
    dead = random_output.shape[0] - alive
    return alive, dead
```


```python
avengers_alive_num = 6
round_num = 0
while True:
    round_num += 1
    avengers_alive_num,avengers_dead_num = check_survive(avengers_alive_num)
    print ('Round {}, avengers alive: {}, dead: {}'.format(round_num,avengers_alive_num,avengers_dead_num) )
    if avengers_alive_num ==0:
        print ('Avengers die first')
        break
    elif np.random.randint(2) ==0:
        print ('Thanos die first')
        break
```

    Round 1, avengers alive: 1, dead: 5
    Thanos die first



```python
def summarize(avengers_alive_num = 6):
    round_num = 0
    while True:
        round_num += 1
        avengers_alive_num,avengers_dead_num = check_survive(avengers_alive_num)
        #print ('Round {}, avengers alive: {}, dead: {}'.format(round_num,avengers_alive_num,avengers_dead_num) )
        if avengers_alive_num ==0:
            #print ('Avengers die first')
            return round_num, 1
        elif np.random.randint(2) ==0:
            #print ('Thanos die first')
            return round_num, 0
```


```python
test_time = 14000605
summarize_data = np.ndarray([test_time,2])
for i in tqdm(range(test_time)):
    summarize_data[i] = summarize()
```


    HBox(children=(IntProgress(value=0, max=14000605), HTML(value='')))


    



```python
np.savetxt('summarize_data_num_6.txt',summarize_data.astype('uint8'),fmt='%1d')
```


```python
fig,ax=plt.subplots(figsize=(8,4))
sns.distplot(summarize_data[:,1],kde=0,ax=ax)
embed_pdf_figure()
```


<object width="640" height="480" data="data:application/pdf;base64,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IC9Bc2NlbnQgOTI5IC9DYXBIZWlnaHQgMCAvRGVzY2VudCAtMjM2IC9GbGFncyAzMgovRm9udEJCb3ggWyAtMTAyMSAtNDYzIDE3OTQgMTIzMyBdIC9Gb250TmFtZSAvRGVqYVZ1U2FucyAvSXRhbGljQW5nbGUgMAovTWF4V2lkdGggMTM0MiAvU3RlbVYgMCAvVHlwZSAvRm9udERlc2NyaXB0b3IgL1hIZWlnaHQgMCA+PgplbmRvYmoKMTIgMCBvYmoKWyA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMAo2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDMxOCA0MDEgNDYwIDgzOCA2MzYKOTUwIDc4MCAyNzUgMzkwIDM5MCA1MDAgODM4IDMxOCAzNjEgMzE4IDMzNyA2MzYgNjM2IDYzNiA2MzYgNjM2IDYzNiA2MzYgNjM2CjYzNiA2MzYgMzM3IDMzNyA4MzggODM4IDgzOCA1MzEgMTAwMCA2ODQgNjg2IDY5OCA3NzAgNjMyIDU3NSA3NzUgNzUyIDI5NQoyOTUgNjU2IDU1NyA4NjMgNzQ4IDc4NyA2MDMgNzg3IDY5NSA2MzUgNjExIDczMiA2ODQgOTg5IDY4NSA2MTEgNjg1IDM5MCAzMzcKMzkwIDgzOCA1MDAgNTAwIDYxMyA2MzUgNTUwIDYzNSA2MTUgMzUyIDYzNSA2MzQgMjc4IDI3OCA1NzkgMjc4IDk3NCA2MzQgNjEyCjYzNSA2MzUgNDExIDUyMSAzOTIgNjM0IDU5MiA4MTggNTkyIDU5MiA1MjUgNjM2IDMzNyA2MzYgODM4IDYwMCA2MzYgNjAwIDMxOAozNTIgNTE4IDEwMDAgNTAwIDUwMCA1MDAgMTM0MiA2MzUgNDAwIDEwNzAgNjAwIDY4NSA2MDAgNjAwIDMxOCAzMTggNTE4IDUxOAo1OTAgNTAwIDEwMDAgNTAwIDEwMDAgNTIxIDQwMCAxMDIzIDYwMCA1MjUgNjExIDMxOCA0MDEgNjM2IDYzNiA2MzYgNjM2IDMzNwo1MDAgNTAwIDEwMDAgNDcxIDYxMiA4MzggMzYxIDEwMDAgNTAwIDUwMCA4MzggNDAxIDQwMSA1MDAgNjM2IDYzNiAzMTggNTAwCjQwMSA0NzEgNjEyIDk2OSA5NjkgOTY5IDUzMSA2ODQgNjg0IDY4NCA2ODQgNjg0IDY4NCA5NzQgNjk4IDYzMiA2MzIgNjMyIDYzMgoyOTUgMjk1IDI5NSAyOTUgNzc1IDc0OCA3ODcgNzg3IDc4NyA3ODcgNzg3IDgzOCA3ODcgNzMyIDczMiA3MzIgNzMyIDYxMSA2MDUKNjMwIDYxMyA2MTMgNjEzIDYxMyA2MTMgNjEzIDk4MiA1NTAgNjE1IDYxNSA2MTUgNjE1IDI3OCAyNzggMjc4IDI3OCA2MTIgNjM0CjYxMiA2MTIgNjEyIDYxMiA2MTIgODM4IDYxMiA2MzQgNjM0IDYzNCA2MzQgNTkyIDYzNSA1OTIgXQplbmRvYmoKMTUgMCBvYmoKPDwgL2UgMTYgMCBSIC9laWdodCAxNyAwIFIgL2ZvdXIgMTggMCBSIC9vbmUgMTkgMCBSIC9wZXJpb2QgMjAgMCBSCi9zZXZlbiAyMSAwIFIgL3NpeCAyMiAwIFIgL3R3byAyMyAwIFIgL3plcm8gMjQgMCBSID4+CmVuZG9iagozIDAgb2JqCjw8IC9GMSAxNCAwIFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0ExIDw8IC9DQSAwIC9UeXBlIC9FeHRHU3RhdGUgL2NhIDEgPj4KL0EyIDw8IC9DQSAwLjQgL1R5cGUgL0V4dEdTdGF0ZSAvY2EgMC40ID4+Ci9BMyA8PCAvQ0EgMSAvVHlwZSAvRXh0R1N0YXRlIC9jYSAxID4+ID4+CmVuZG9iago1IDAgb2JqCjw8ID4+CmVuZG9iago2IDAgb2JqCjw8ID4+CmVuZG9iago3IDAgb2JqCjw8ID4+CmVuZG9iagoyIDAgb2JqCjw8IC9Db3VudCAxIC9LaWRzIFsgMTAgMCBSIF0gL1R5cGUgL1BhZ2VzID4+CmVuZG9iagoyNSAwIG9iago8PCAvQ3JlYXRpb25EYXRlIChEOjIwMTkwNTA2MjExODMyKzA4JzAwJykKL0NyZWF0b3IgKG1hdHBsb3RsaWIgMy4wLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZykKL1Byb2R1Y2VyIChtYXRwbG90bGliIHBkZiBiYWNrZW5kIDMuMC4zKSAvVGl0bGUgKEltYWdlKSA+PgplbmRvYmoKeHJlZgowIDI2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAxNiAwMDAwMCBuIAowMDAwMDA2MTE2IDAwMDAwIG4gCjAwMDAwMDU4NzkgMDAwMDAgbiAKMDAwMDAwNTkxMSAwMDAwMCBuIAowMDAwMDA2MDUzIDAwMDAwIG4gCjAwMDAwMDYwNzQgMDAwMDAgbiAKMDAwMDAwNjA5NSAwMDAwMCBuIAowMDAwMDAwMDY1IDAwMDAwIG4gCjAwMDAwMDAzODUgMDAwMDAgbiAKMDAwMDAwMDIwOCAwMDAwMCBuIAowMDAwMDAxNzU1IDAwMDAwIG4gCjAwMDAwMDQ2ODkgMDAwMDAgbiAKMDAwMDAwNDQ4OSAwMDAwMCBuIAowMDAwMDA0MTI4IDAwMDAwIG4gCjAwMDAwMDU3NDIgMDAwMDAgbiAKMDAwMDAwMTc3NiAwMDAwMCBuIAowMDAwMDAyMDk0IDAwMDAwIG4gCjAwMDAwMDI1NTkgMDAwMDAgbiAKMDAwMDAwMjcyMSAwMDAwMCBuIAowMDAwMDAyODczIDAwMDAwIG4gCjAwMDAwMDI5OTQgMDAwMDAgbiAKMDAwMDAwMzEzNCAwMDAwMCBuIAowMDAwMDAzNTI0IDAwMDAwIG4gCjAwMDAwMDM4NDUgMDAwMDAgbiAKMDAwMDAwNjE3NiAwMDAwMCBuIAp0cmFpbGVyCjw8IC9JbmZvIDI1IDAgUiAvUm9vdCAxIDAgUiAvU2l6ZSAyNiA+PgpzdGFydHhyZWYKNjM0NQolJUVPRgo=" download="Image.pdf"></object>



```python
fig,ax=plt.subplots(figsize=(8,4))
sns.distplot(summarize_data[np.where(summarize_data[:,1]==0)][:,0],kde=0,ax=ax,color='r')
sns.distplot(summarize_data[np.where(summarize_data[:,1]==1)][:,0],kde=0,ax=ax,color='m')
embed_pdf_figure()
```


<object width="640" height="480" data="data:application/pdf;base64,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pZ9vvMf3D0AAU7QKZW5kc3RyZWFtCmVuZG9iagoxNCAwIG9iago8PCAvQmFzZUZvbnQgL0RlamFWdVNhbnMgL0NoYXJQcm9jcyAxNSAwIFIKL0VuY29kaW5nIDw8Ci9EaWZmZXJlbmNlcyBbIDQ4IC96ZXJvIC9vbmUgL3R3byAvdGhyZWUgL2ZvdXIgL2ZpdmUgL3NpeCAvc2V2ZW4gL2VpZ2h0IF0KL1R5cGUgL0VuY29kaW5nID4+Ci9GaXJzdENoYXIgMCAvRm9udEJCb3ggWyAtMTAyMSAtNDYzIDE3OTQgMTIzMyBdIC9Gb250RGVzY3JpcHRvciAxMyAwIFIKL0ZvbnRNYXRyaXggWyAwLjAwMSAwIDAgMC4wMDEgMCAwIF0gL0xhc3RDaGFyIDI1NSAvTmFtZSAvRGVqYVZ1U2FucwovU3VidHlwZSAvVHlwZTMgL1R5cGUgL0ZvbnQgL1dpZHRocyAxMiAwIFIgPj4KZW5kb2JqCjEzIDAgb2JqCjw8IC9Bc2NlbnQgOTI5IC9DYXBIZWlnaHQgMCAvRGVzY2VudCAtMjM2IC9GbGFncyAzMgovRm9udEJCb3ggWyAtMTAyMSAtNDYzIDE3OTQgMTIzMyBdIC9Gb250TmFtZSAvRGVqYVZ1U2FucyAvSXRhbGljQW5nbGUgMAovTWF4V2lkdGggMTM0MiAvU3RlbVYgMCAvVHlwZSAvRm9udERlc2NyaXB0b3IgL1hIZWlnaHQgMCA+PgplbmRvYmoKMTIgMCBvYmoKWyA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMAo2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDMxOCA0MDEgNDYwIDgzOCA2MzYKOTUwIDc4MCAyNzUgMzkwIDM5MCA1MDAgODM4IDMxOCAzNjEgMzE4IDMzNyA2MzYgNjM2IDYzNiA2MzYgNjM2IDYzNiA2MzYgNjM2CjYzNiA2MzYgMzM3IDMzNyA4MzggODM4IDgzOCA1MzEgMTAwMCA2ODQgNjg2IDY5OCA3NzAgNjMyIDU3NSA3NzUgNzUyIDI5NQoyOTUgNjU2IDU1NyA4NjMgNzQ4IDc4NyA2MDMgNzg3IDY5NSA2MzUgNjExIDczMiA2ODQgOTg5IDY4NSA2MTEgNjg1IDM5MCAzMzcKMzkwIDgzOCA1MDAgNTAwIDYxMyA2MzUgNTUwIDYzNSA2MTUgMzUyIDYzNSA2MzQgMjc4IDI3OCA1NzkgMjc4IDk3NCA2MzQgNjEyCjYzNSA2MzUgNDExIDUyMSAzOTIgNjM0IDU5MiA4MTggNTkyIDU5MiA1MjUgNjM2IDMzNyA2MzYgODM4IDYwMCA2MzYgNjAwIDMxOAozNTIgNTE4IDEwMDAgNTAwIDUwMCA1MDAgMTM0MiA2MzUgNDAwIDEwNzAgNjAwIDY4NSA2MDAgNjAwIDMxOCAzMTggNTE4IDUxOAo1OTAgNTAwIDEwMDAgNTAwIDEwMDAgNTIxIDQwMCAxMDIzIDYwMCA1MjUgNjExIDMxOCA0MDEgNjM2IDYzNiA2MzYgNjM2IDMzNwo1MDAgNTAwIDEwMDAgNDcxIDYxMiA4MzggMzYxIDEwMDAgNTAwIDUwMCA4MzggNDAxIDQwMSA1MDAgNjM2IDYzNiAzMTggNTAwCjQwMSA0NzEgNjEyIDk2OSA5NjkgOTY5IDUzMSA2ODQgNjg0IDY4NCA2ODQgNjg0IDY4NCA5NzQgNjk4IDYzMiA2MzIgNjMyIDYzMgoyOTUgMjk1IDI5NSAyOTUgNzc1IDc0OCA3ODcgNzg3IDc4NyA3ODcgNzg3IDgzOCA3ODcgNzMyIDczMiA3MzIgNzMyIDYxMSA2MDUKNjMwIDYxMyA2MTMgNjEzIDYxMyA2MTMgNjEzIDk4MiA1NTAgNjE1IDYxNSA2MTUgNjE1IDI3OCAyNzggMjc4IDI3OCA2MTIgNjM0CjYxMiA2MTIgNjEyIDYxMiA2MTIgODM4IDYxMiA2MzQgNjM0IDYzNCA2MzQgNTkyIDYzNSA1OTIgXQplbmRvYmoKMTUgMCBvYmoKPDwgL2VpZ2h0IDE2IDAgUiAvZml2ZSAxNyAwIFIgL2ZvdXIgMTggMCBSIC9vbmUgMTkgMCBSIC9zZXZlbiAyMCAwIFIKL3NpeCAyMSAwIFIgL3RocmVlIDIyIDAgUiAvdHdvIDIzIDAgUiAvemVybyAyNCAwIFIgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL0YxIDE0IDAgUiA+PgplbmRvYmoKNCAwIG9iago8PCAvQTEgPDwgL0NBIDAgL1R5cGUgL0V4dEdTdGF0ZSAvY2EgMSA+PgovQTIgPDwgL0NBIDAuNCAvVHlwZSAvRXh0R1N0YXRlIC9jYSAwLjQgPj4KL0EzIDw8IC9DQSAxIC9UeXBlIC9FeHRHU3RhdGUgL2NhIDEgPj4gPj4KZW5kb2JqCjUgMCBvYmoKPDwgPj4KZW5kb2JqCjYgMCBvYmoKPDwgPj4KZW5kb2JqCjcgMCBvYmoKPDwgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL0NvdW50IDEgL0tpZHMgWyAxMCAwIFIgXSAvVHlwZSAvUGFnZXMgPj4KZW5kb2JqCjI1IDAgb2JqCjw8IC9DcmVhdGlvbkRhdGUgKEQ6MjAxOTA1MDYyMTE4MzUrMDgnMDAnKQovQ3JlYXRvciAobWF0cGxvdGxpYiAzLjAuMywgaHR0cDovL21hdHBsb3RsaWIub3JnKQovUHJvZHVjZXIgKG1hdHBsb3RsaWIgcGRmIGJhY2tlbmQgMy4wLjMpIC9UaXRsZSAoSW1hZ2UpID4+CmVuZG9iagp4cmVmCjAgMjYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE2IDAwMDAwIG4gCjAwMDAwMDc0MDggMDAwMDAgbiAKMDAwMDAwNzE3MSAwMDAwMCBuIAowMDAwMDA3MjAzIDAwMDAwIG4gCjAwMDAwMDczNDUgMDAwMDAgbiAKMDAwMDAwNzM2NiAwMDAwMCBuIAowMDAwMDA3Mzg3IDAwMDAwIG4gCjAwMDAwMDAwNjUgMDAwMDAgbiAKMDAwMDAwMDM4NSAwMDAwMCBuIAowMDAwMDAwMjA4IDAwMDAwIG4gCjAwMDAwMDI3NjQgMDAwMDAgbiAKMDAwMDAwNTk3OSAwMDAwMCBuIAowMDAwMDA1Nzc5IDAwMDAwIG4gCjAwMDAwMDU0MjkgMDAwMDAgbiAKMDAwMDAwNzAzMiAwMDAwMCBuIAowMDAwMDAyNzg1IDAwMDAwIG4gCjAwMDAwMDMyNTAgMDAwMDAgbiAKMDAwMDAwMzU3MCAwMDAwMCBuIAowMDAwMDAzNzMyIDAwMDAwIG4gCjAwMDAwMDM4ODQgMDAwMDAgbiAKMDAwMDAwNDAyNCAwMDAwMCBuIAowMDAwMDA0NDE0IDAwMDAwIG4gCjAwMDAwMDQ4MjUgMDAwMDAgbiAKMDAwMDAwNTE0NiAwMDAwMCBuIAowMDAwMDA3NDY4IDAwMDAwIG4gCnRyYWlsZXIKPDwgL0luZm8gMjUgMCBSIC9Sb290IDEgMCBSIC9TaXplIDI2ID4+CnN0YXJ0eHJlZgo3NjM3CiUlRU9GCg==" download="Image.pdf"></object>



```python
print ('Avengers win {}, Thanos win {}'.format(1 - summarize_data[:,1].sum()/test_time,summarize_data[:,1].sum()/test_time) )
```

    Avengers win 0.7940833985388489, Thanos win 0.20591660146115115



```python
unique_sum = np.unique(summarize_data[np.where(summarize_data[:,1]==0)][:,0],return_counts=1)
df = pd.DataFrame(np.concatenate((unique_sum[1].reshape(-1,1),unique_sum[0].astype('int').reshape(-1,1),unique_sum[1].reshape(-1,1)/test_time),axis=1))
df.iloc[:,1] = np.array(df.iloc[:,1]).astype('int')
df = df.set_index(1)
df.columns = ['win time','win probability']
display_dataframe(df,filename='Avengers win times in {} times'.format(test_time))
    
```


<style  type="text/css" >
</style>  
<table id="T_754e7ea2_7001_11e9_9cd0_8c8590804826" ><caption>Avengers win times in 14000605 times</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >win time</th> 
        <th class="col_heading level0 col1" >win probability</th> 
    </tr>    <tr> 
        <th class="index_name level0" >1</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row0_col0" class="data row0 col0" >6.89032e+06</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row0_col1" class="data row0 col1" >0.492144</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row1_col0" class="data row1 col0" >2.8789e+06</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row1_col1" class="data row1 col1" >0.205627</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row2" class="row_heading level0 row2" >3</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row2_col0" class="data row2 col0" >964446</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row2_col1" class="data row2 col1" >0.068886</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row3" class="row_heading level0 row3" >4</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row3_col0" class="data row3 col0" >281804</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row3_col1" class="data row3 col1" >0.020128</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row4" class="row_heading level0 row4" >5</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row4_col0" class="data row4 col0" >75835</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row4_col1" class="data row4 col1" >0.00541655</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row5" class="row_heading level0 row5" >6</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row5_col0" class="data row5 col0" >19639</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row5_col1" class="data row5 col1" >0.00140273</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row6" class="row_heading level0 row6" >7</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row6_col0" class="data row6 col0" >4988</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row6_col1" class="data row6 col1" >0.00035627</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row7" class="row_heading level0 row7" >8</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row7_col0" class="data row7 col0" >1274</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row7_col1" class="data row7 col1" >9.09961e-05</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row8" class="row_heading level0 row8" >9</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row8_col0" class="data row8 col0" >331</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row8_col1" class="data row8 col1" >2.36418e-05</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row9" class="row_heading level0 row9" >10</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row9_col0" class="data row9 col0" >91</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row9_col1" class="data row9 col1" >6.49972e-06</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row10" class="row_heading level0 row10" >11</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row10_col0" class="data row10 col0" >17</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row10_col1" class="data row10 col1" >1.21423e-06</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row11" class="row_heading level0 row11" >12</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row11_col0" class="data row11 col0" >5</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row11_col1" class="data row11 col1" >3.57127e-07</td> 
    </tr>    <tr> 
        <th id="T_754e7ea2_7001_11e9_9cd0_8c8590804826level0_row12" class="row_heading level0 row12" >13</th> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row12_col0" class="data row12 col0" >1</td> 
        <td id="T_754e7ea2_7001_11e9_9cd0_8c8590804826row12_col1" class="data row12 col1" >7.14255e-08</td> 
    </tr></tbody> 
</table> 



<input type="button" id="button_585974664" value="Download">



<script>
    document.getElementById("button_585974664").addEventListener("click", function(event){
        var filename = "Avengers win times in 14000605 times.csv";
        var data = "data:text/csv;base64,MSx3aW4gdGltZSx3aW4gcHJvYmFiaWxpdHkKMSw2ODkwMzE4LjAsMC40OTIxNDQzMDM3NjQwMTU5CjIsMjg3ODg5OS4wLDAuMjA1NjI2NzU2ODQzNzIyMQozLDk2NDQ0Ni4wLDAuMDY4ODg2MDIzMTM5NzE0MzEKNCwyODE4MDQuMCwwLjAyMDEyNzk4NzMyNjI2MTk3Mgo1LDc1ODM1LjAsMC4wMDU0MTY1NTE2NDE4NzU0NzYKNiwxOTYzOS4wLDAuMDAxNDAyNzI1MDk2NTIyNjE0Ngo3LDQ5ODguMCwwLjAwMDM1NjI3MDMxODMxODM4Njk1CjgsMTI3NC4wLDkuMDk5NjA2NzY2OTkzMjgzZS0wNQo5LDMzMS4wLDIuMzY0MTgzNTQ3NzgyMzk5NWUtMDUKMTAsOTEuMCw2LjQ5OTcxOTExOTI4MDkxN2UtMDYKMTEsMTcuMCwxLjIxNDIzMzI0MjA2MzQ2OGUtMDYKMTIsNS4wLDMuNTcxMjc0MjQxMzYzMTQxZS0wNwoxMywxLjAsNy4xNDI1NDg0ODI3MjYyODJlLTA4Cg==";
        const element = document.createElement('a');
        element.setAttribute('href', data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    });
</script>



```python
unique_sum = np.unique(summarize_data[np.where(summarize_data[:,1]==1)][:,0],return_counts=1)
df = pd.DataFrame(np.concatenate((unique_sum[1].reshape(-1,1),unique_sum[0].astype('int').reshape(-1,1),unique_sum[1].reshape(-1,1)/test_time),axis=1))
df.iloc[:,1] = np.array(df.iloc[:,1]).astype('int')
df = df.set_index(1)
df.columns = ['win time','win probability']
display_dataframe(df,filename='Thanos win times in {} times'.format(test_time))
    
```


<style  type="text/css" >
</style>  
<table id="T_759d3e8e_7001_11e9_9cd0_8c8590804826" ><caption>Thanos win times in 14000605 times</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >win time</th> 
        <th class="col_heading level0 col1" >win probability</th> 
    </tr>    <tr> 
        <th class="index_name level0" >1</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row0_col0" class="data row0 col0" >218176</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row0_col1" class="data row0 col1" >0.0155833</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row1_col0" class="data row1 col0" >1.13563e+06</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row1_col1" class="data row1 col1" >0.0811129</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row2" class="row_heading level0 row2" >3</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row2_col0" class="data row2 col0" >948005</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row2_col1" class="data row2 col1" >0.0677117</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row3" class="row_heading level0 row3" >4</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row3_col0" class="data row3 col0" >402860</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row3_col1" class="data row3 col1" >0.0287745</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row4" class="row_heading level0 row4" >5</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row4_col0" class="data row4 col0" >128810</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row4_col1" class="data row4 col1" >0.00920032</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row5" class="row_heading level0 row5" >6</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row5_col0" class="data row5 col0" >36483</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row5_col1" class="data row5 col1" >0.00260582</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row6" class="row_heading level0 row6" >7</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row6_col0" class="data row6 col0" >9634</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row6_col1" class="data row6 col1" >0.000688113</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row7" class="row_heading level0 row7" >8</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row7_col0" class="data row7 col0" >2508</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row7_col1" class="data row7 col1" >0.000179135</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row8" class="row_heading level0 row8" >9</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row8_col0" class="data row8 col0" >660</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row8_col1" class="data row8 col1" >4.71408e-05</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row9" class="row_heading level0 row9" >10</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row9_col0" class="data row9 col0" >147</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row9_col1" class="data row9 col1" >1.04995e-05</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row10" class="row_heading level0 row10" >11</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row10_col0" class="data row10 col0" >27</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row10_col1" class="data row10 col1" >1.92849e-06</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row11" class="row_heading level0 row11" >12</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row11_col0" class="data row11 col0" >11</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row11_col1" class="data row11 col1" >7.8568e-07</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row12" class="row_heading level0 row12" >13</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row12_col0" class="data row12 col0" >5</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row12_col1" class="data row12 col1" >3.57127e-07</td> 
    </tr>    <tr> 
        <th id="T_759d3e8e_7001_11e9_9cd0_8c8590804826level0_row13" class="row_heading level0 row13" >14</th> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row13_col0" class="data row13 col0" >1</td> 
        <td id="T_759d3e8e_7001_11e9_9cd0_8c8590804826row13_col1" class="data row13 col1" >7.14255e-08</td> 
    </tr></tbody> 
</table> 



<input type="button" id="button_803906024" value="Download">



<script>
    document.getElementById("button_803906024").addEventListener("click", function(event){
        var filename = "Thanos win times in 14000605 times.csv";
        var data = "data:text/csv;base64,MSx3aW4gdGltZSx3aW4gcHJvYmFiaWxpdHkKMSwyMTgxNzYuMCwwLjAxNTU4MzMyNjU3NzY3Mjg5NAoyLDExMzU2MzAuMCwwLjA4MTExMjkyMzMzNDM4NDQ4CjMsOTQ4MDA1LjAsMC4wNjc3MTE3MTY3NDM2NjkzCjQsNDAyODYwLjAsMC4wMjg3NzQ0NzA4MTc1MTExCjUsMTI4ODEwLjAsMC4wMDkyMDAzMTY3MDA1OTk3MjQKNiwzNjQ4My4wLDAuMDAyNjA1ODE1OTYyOTUzMDI5NAo3LDk2MzQuMCwwLjAwMDY4ODExMzEyMDgyNTg1CjgsMjUwOC4wLDAuMDAwMTc5MTM1MTE1OTQ2Nzc1MTYKOSw2NjAuMCw0LjcxNDA4MTk5ODU5OTM0NjZlLTA1CjEwLDE0Ny4wLDEuMDQ5OTU0NjI2OTYwNzYzNWUtMDUKMTEsMjcuMCwxLjkyODQ4ODA5MDMzNjA5NjRlLTA2CjEyLDExLjAsNy44NTY4MDMzMzA5OTg5MWUtMDcKMTMsNS4wLDMuNTcxMjc0MjQxMzYzMTQxZS0wNwoxNCwxLjAsNy4xNDI1NDg0ODI3MjYyODJlLTA4Cg==";
        const element = document.createElement('a');
        element.setAttribute('href', data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    });
</script>


灭霸死的有点惨，其实复仇者初始只有两个人的时候概率稍微对等


```python
def summarize(avengers_alive_num = 6):
    round_num = 0
    while True:
        round_num += 1
        avengers_alive_num,avengers_dead_num = check_survive(avengers_alive_num)
        #print ('Round {}, avengers alive: {}, dead: {}'.format(round_num,avengers_alive_num,avengers_dead_num) )
        if avengers_alive_num ==0:
            #print ('Avengers die first')
            return round_num, 1
        elif np.random.randint(2) ==0:
            #print ('Thanos die first')
            return round_num, 0
```


```python
test_time = 14000605
summarize_data = np.ndarray([test_time,2])
for i in tqdm(range(test_time)):
    summarize_data[i] = summarize(avengers_alive_num =2)
```


    HBox(children=(IntProgress(value=0, max=14000605), HTML(value='')))


    



```python
fig,ax=plt.subplots(2,1,figsize=(8,8))
sns.distplot(summarize_data[:,1],kde=0,ax=ax[0])
sns.distplot(summarize_data[np.where(summarize_data[:,1]==0)][:,0],kde=0,ax=ax[1],color='r')
sns.distplot(summarize_data[np.where(summarize_data[:,1]==1)][:,0],kde=0,ax=ax[1],color='m')
embed_pdf_figure()
```


<object width="640" height="480" data="data:application/pdf;base64,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MTc5NCAxMjMzIF0gL0ZvbnREZXNjcmlwdG9yIDEzIDAgUgovRm9udE1hdHJpeCBbIDAuMDAxIDAgMCAwLjAwMSAwIDAgXSAvTGFzdENoYXIgMjU1IC9OYW1lIC9EZWphVnVTYW5zCi9TdWJ0eXBlIC9UeXBlMyAvVHlwZSAvRm9udCAvV2lkdGhzIDEyIDAgUiA+PgplbmRvYmoKMTMgMCBvYmoKPDwgL0FzY2VudCA5MjkgL0NhcEhlaWdodCAwIC9EZXNjZW50IC0yMzYgL0ZsYWdzIDMyCi9Gb250QkJveCBbIC0xMDIxIC00NjMgMTc5NCAxMjMzIF0gL0ZvbnROYW1lIC9EZWphVnVTYW5zIC9JdGFsaWNBbmdsZSAwCi9NYXhXaWR0aCAxMzQyIC9TdGVtViAwIC9UeXBlIC9Gb250RGVzY3JpcHRvciAvWEhlaWdodCAwID4+CmVuZG9iagoxMiAwIG9iagpbIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwCjYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgNjAwIDYwMCA2MDAgMzE4IDQwMSA0NjAgODM4IDYzNgo5NTAgNzgwIDI3NSAzOTAgMzkwIDUwMCA4MzggMzE4IDM2MSAzMTggMzM3IDYzNiA2MzYgNjM2IDYzNiA2MzYgNjM2IDYzNiA2MzYKNjM2IDYzNiAzMzcgMzM3IDgzOCA4MzggODM4IDUzMSAxMDAwIDY4NCA2ODYgNjk4IDc3MCA2MzIgNTc1IDc3NSA3NTIgMjk1CjI5NSA2NTYgNTU3IDg2MyA3NDggNzg3IDYwMyA3ODcgNjk1IDYzNSA2MTEgNzMyIDY4NCA5ODkgNjg1IDYxMSA2ODUgMzkwIDMzNwozOTAgODM4IDUwMCA1MDAgNjEzIDYzNSA1NTAgNjM1IDYxNSAzNTIgNjM1IDYzNCAyNzggMjc4IDU3OSAyNzggOTc0IDYzNCA2MTIKNjM1IDYzNSA0MTEgNTIxIDM5MiA2MzQgNTkyIDgxOCA1OTIgNTkyIDUyNSA2MzYgMzM3IDYzNiA4MzggNjAwIDYzNiA2MDAgMzE4CjM1MiA1MTggMTAwMCA1MDAgNTAwIDUwMCAxMzQyIDYzNSA0MDAgMTA3MCA2MDAgNjg1IDYwMCA2MDAgMzE4IDMxOCA1MTggNTE4CjU5MCA1MDAgMTAwMCA1MDAgMTAwMCA1MjEgNDAwIDEwMjMgNjAwIDUyNSA2MTEgMzE4IDQwMSA2MzYgNjM2IDYzNiA2MzYgMzM3CjUwMCA1MDAgMTAwMCA0NzEgNjEyIDgzOCAzNjEgMTAwMCA1MDAgNTAwIDgzOCA0MDEgNDAxIDUwMCA2MzYgNjM2IDMxOCA1MDAKNDAxIDQ3MSA2MTIgOTY5IDk2OSA5NjkgNTMxIDY4NCA2ODQgNjg0IDY4NCA2ODQgNjg0IDk3NCA2OTggNjMyIDYzMiA2MzIgNjMyCjI5NSAyOTUgMjk1IDI5NSA3NzUgNzQ4IDc4NyA3ODcgNzg3IDc4NyA3ODcgODM4IDc4NyA3MzIgNzMyIDczMiA3MzIgNjExIDYwNQo2MzAgNjEzIDYxMyA2MTMgNjEzIDYxMyA2MTMgOTgyIDU1MCA2MTUgNjE1IDYxNSA2MTUgMjc4IDI3OCAyNzggMjc4IDYxMiA2MzQKNjEyIDYxMiA2MTIgNjEyIDYxMiA4MzggNjEyIDYzNCA2MzQgNjM0IDYzNCA1OTIgNjM1IDU5MiBdCmVuZG9iagoxNSAwIG9iago8PCAvZWlnaHQgMTYgMCBSIC9maXZlIDE3IDAgUiAvZm91ciAxOCAwIFIgL29uZSAxOSAwIFIgL3BlcmlvZCAyMCAwIFIKL3NldmVuIDIxIDAgUiAvc2l4IDIyIDAgUiAvdGhyZWUgMjMgMCBSIC90d28gMjQgMCBSIC96ZXJvIDI1IDAgUiA+PgplbmRvYmoKMyAwIG9iago8PCAvRjEgMTQgMCBSID4+CmVuZG9iago0IDAgb2JqCjw8IC9BMSA8PCAvQ0EgMCAvVHlwZSAvRXh0R1N0YXRlIC9jYSAxID4+Ci9BMiA8PCAvQ0EgMC40IC9UeXBlIC9FeHRHU3RhdGUgL2NhIDAuNCA+PgovQTMgPDwgL0NBIDEgL1R5cGUgL0V4dEdTdGF0ZSAvY2EgMSA+PiA+PgplbmRvYmoKNSAwIG9iago8PCA+PgplbmRvYmoKNiAwIG9iago8PCA+PgplbmRvYmoKNyAwIG9iago8PCA+PgplbmRvYmoKMiAwIG9iago8PCAvQ291bnQgMSAvS2lkcyBbIDEwIDAgUiBdIC9UeXBlIC9QYWdlcyA+PgplbmRvYmoKMjYgMCBvYmoKPDwgL0NyZWF0aW9uRGF0ZSAoRDoyMDE5MDUwNjIxMjIzMCswOCcwMCcpCi9DcmVhdG9yIChtYXRwbG90bGliIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcpCi9Qcm9kdWNlciAobWF0cGxvdGxpYiBwZGYgYmFja2VuZCAzLjAuMykgL1RpdGxlIChJbWFnZSkgPj4KZW5kb2JqCnhyZWYKMCAyNwowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMTYgMDAwMDAgbiAKMDAwMDAwODEwMCAwMDAwMCBuIAowMDAwMDA3ODYzIDAwMDAwIG4gCjAwMDAwMDc4OTUgMDAwMDAgbiAKMDAwMDAwODAzNyAwMDAwMCBuIAowMDAwMDA4MDU4IDAwMDAwIG4gCjAwMDAwMDgwNzkgMDAwMDAgbiAKMDAwMDAwMDA2NSAwMDAwMCBuIAowMDAwMDAwMzg1IDAwMDAwIG4gCjAwMDAwMDAyMDggMDAwMDAgbiAKMDAwMDAwMzMwOSAwMDAwMCBuIAowMDAwMDA2NjU2IDAwMDAwIG4gCjAwMDAwMDY0NTYgMDAwMDAgbiAKMDAwMDAwNjA5NSAwMDAwMCBuIAowMDAwMDA3NzA5IDAwMDAwIG4gCjAwMDAwMDMzMzAgMDAwMDAgbiAKMDAwMDAwMzc5NSAwMDAwMCBuIAowMDAwMDA0MTE1IDAwMDAwIG4gCjAwMDAwMDQyNzcgMDAwMDAgbiAKMDAwMDAwNDQyOSAwMDAwMCBuIAowMDAwMDA0NTUwIDAwMDAwIG4gCjAwMDAwMDQ2OTAgMDAwMDAgbiAKMDAwMDAwNTA4MCAwMDAwMCBuIAowMDAwMDA1NDkxIDAwMDAwIG4gCjAwMDAwMDU4MTIgMDAwMDAgbiAKMDAwMDAwODE2MCAwMDAwMCBuIAp0cmFpbGVyCjw8IC9JbmZvIDI2IDAgUiAvUm9vdCAxIDAgUiAvU2l6ZSAyNyA+PgpzdGFydHhyZWYKODMyOQolJUVPRgo=" download="Image.pdf"></object>



```python
print ('Avengers win {}, Thanos win {}'.format(1 - summarize_data[:,1].sum()/test_time,summarize_data[:,1].sum()/test_time) )
```

    Avengers win 0.523922359069483, Thanos win 0.4760776409305169



```python
unique_sum = np.unique(summarize_data[np.where(summarize_data[:,1]==0)][:,0],return_counts=1)
df = pd.DataFrame(np.concatenate((unique_sum[1].reshape(-1,1),unique_sum[0].astype('int').reshape(-1,1),unique_sum[1].reshape(-1,1)/test_time),axis=1))
df.iloc[:,1] = np.array(df.iloc[:,1]).astype('int')
df = df.set_index(1)
df.columns = ['win time','win probability']
display_dataframe(df,filename='Avengers win times in {} times'.format(test_time))
    
unique_sum = np.unique(summarize_data[np.where(summarize_data[:,1]==1)][:,0],return_counts=1)
df = pd.DataFrame(np.concatenate((unique_sum[1].reshape(-1,1),unique_sum[0].astype('int').reshape(-1,1),unique_sum[1].reshape(-1,1)/test_time),axis=1))
df.iloc[:,1] = np.array(df.iloc[:,1]).astype('int')
df = df.set_index(1)
df.columns = ['win time','win probability']
display_dataframe(df,filename='Thanos win times in {} times'.format(test_time))
```


<style  type="text/css" >
</style>  
<table id="T_00bf9b60_7002_11e9_9cd0_8c8590804826" ><caption>Avengers win times in 14000605 times</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >win time</th> 
        <th class="col_heading level0 col1" >win probability</th> 
    </tr>    <tr> 
        <th class="index_name level0" >1</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row0_col0" class="data row0 col0" >5.25233e+06</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row0_col1" class="data row0 col1" >0.37515</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row1_col0" class="data row1 col0" >1.53058e+06</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row1_col1" class="data row1 col1" >0.109322</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row2" class="row_heading level0 row2" >3</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row2_col0" class="data row2 col0" >409989</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row2_col1" class="data row2 col1" >0.0292837</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row3" class="row_heading level0 row3" >4</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row3_col0" class="data row3 col0" >106393</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row3_col1" class="data row3 col1" >0.00759917</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row4" class="row_heading level0 row4" >5</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row4_col0" class="data row4 col0" >26925</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row4_col1" class="data row4 col1" >0.00192313</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row5" class="row_heading level0 row5" >6</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row5_col0" class="data row5 col0" >6764</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row5_col1" class="data row5 col1" >0.000483122</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row6" class="row_heading level0 row6" >7</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row6_col0" class="data row6 col0" >1723</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row6_col1" class="data row6 col1" >0.000123066</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row7" class="row_heading level0 row7" >8</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row7_col0" class="data row7 col0" >377</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row7_col1" class="data row7 col1" >2.69274e-05</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row8" class="row_heading level0 row8" >9</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row8_col0" class="data row8 col0" >102</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row8_col1" class="data row8 col1" >7.2854e-06</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row9" class="row_heading level0 row9" >10</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row9_col0" class="data row9 col0" >35</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row9_col1" class="data row9 col1" >2.49989e-06</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row10" class="row_heading level0 row10" >11</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row10_col0" class="data row10 col0" >11</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row10_col1" class="data row10 col1" >7.8568e-07</td> 
    </tr>    <tr> 
        <th id="T_00bf9b60_7002_11e9_9cd0_8c8590804826level0_row11" class="row_heading level0 row11" >13</th> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row11_col0" class="data row11 col0" >1</td> 
        <td id="T_00bf9b60_7002_11e9_9cd0_8c8590804826row11_col1" class="data row11 col1" >7.14255e-08</td> 
    </tr></tbody> 
</table> 



<input type="button" id="button_754759131" value="Download">



<script>
    document.getElementById("button_754759131").addEventListener("click", function(event){
        var filename = "Avengers win times in 14000605 times.csv";
        var data = "data:text/csv;base64,MSx3aW4gdGltZSx3aW4gcHJvYmFiaWxpdHkKMSw1MjUyMzI5LjAsMC4zNzUxNTAxNDUyOTcyOTI1CjIsMTUzMDU4MS4wLDAuMTA5MzIyNDg5OTkyMzk2NzYKMyw0MDk5ODkuMCwwLjAyOTI4MzY2MzA5ODg0NDY1Ngo0LDEwNjM5My4wLDAuMDA3NTk5MTcxNjA3MjI2OTc0CjUsMjY5MjUuMCwwLjAwMTkyMzEzMTE3ODk3NDA1MTUKNiw2NzY0LjAsMC4wMDA0ODMxMjE5NzkzNzE2MDU3CjcsMTcyMy4wLDAuMDAwMTIzMDY2MTEwMzU3MzczODQKOCwzNzcuMCwyLjY5Mjc0MDc3Nzk4NzgwODNlLTA1CjksMTAyLjAsNy4yODUzOTk0NTIzODA4MDhlLTA2CjEwLDM1LjAsMi40OTk4OTE5Njg5NTQxOTg4ZS0wNgoxMSwxMS4wLDcuODU2ODAzMzMwOTk4OTFlLTA3CjEzLDEuMCw3LjE0MjU0ODQ4MjcyNjI4MmUtMDgK";
        const element = document.createElement('a');
        element.setAttribute('href', data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    });
</script>



<style  type="text/css" >
</style>  
<table id="T_011a1c7a_7002_11e9_9cd0_8c8590804826" ><caption>Thanos win times in 14000605 times</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >win time</th> 
        <th class="col_heading level0 col1" >win probability</th> 
    </tr>    <tr> 
        <th class="index_name level0" >1</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row0" class="row_heading level0 row0" >1</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row0_col0" class="data row0 col0" >3.49891e+06</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row0_col1" class="data row0 col1" >0.249911</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row1" class="row_heading level0 row1" >2</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row1_col0" class="data row1 col0" >2.18546e+06</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row1_col1" class="data row1 col1" >0.156097</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row2" class="row_heading level0 row2" >3</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row2_col0" class="data row2 col0" >712026</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row2_col1" class="data row2 col1" >0.0508568</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row3" class="row_heading level0 row3" >4</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row3_col0" class="data row3 col0" >199091</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row3_col1" class="data row3 col1" >0.0142202</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row4" class="row_heading level0 row4" >5</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row4_col0" class="data row4 col0" >52039</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row4_col1" class="data row4 col1" >0.00371691</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row5" class="row_heading level0 row5" >6</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row5_col0" class="data row5 col0" >13297</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row5_col1" class="data row5 col1" >0.000949745</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row6" class="row_heading level0 row6" >7</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row6_col0" class="data row6 col0" >3438</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row6_col1" class="data row6 col1" >0.000245561</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row7" class="row_heading level0 row7" >8</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row7_col0" class="data row7 col0" >816</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row7_col1" class="data row7 col1" >5.82832e-05</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row8" class="row_heading level0 row8" >9</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row8_col0" class="data row8 col0" >215</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row8_col1" class="data row8 col1" >1.53565e-05</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row9" class="row_heading level0 row9" >10</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row9_col0" class="data row9 col0" >65</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row9_col1" class="data row9 col1" >4.64266e-06</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row10" class="row_heading level0 row10" >11</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row10_col0" class="data row10 col0" >17</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row10_col1" class="data row10 col1" >1.21423e-06</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row11" class="row_heading level0 row11" >12</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row11_col0" class="data row11 col0" >5</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row11_col1" class="data row11 col1" >3.57127e-07</td> 
    </tr>    <tr> 
        <th id="T_011a1c7a_7002_11e9_9cd0_8c8590804826level0_row12" class="row_heading level0 row12" >13</th> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row12_col0" class="data row12 col0" >1</td> 
        <td id="T_011a1c7a_7002_11e9_9cd0_8c8590804826row12_col1" class="data row12 col1" >7.14255e-08</td> 
    </tr></tbody> 
</table> 



<input type="button" id="button_316164186" value="Download">



<script>
    document.getElementById("button_316164186").addEventListener("click", function(event){
        var filename = "Thanos win times in 14000605 times.csv";
        var data = "data:text/csv;base64,MSx3aW4gdGltZSx3aW4gcHJvYmFiaWxpdHkKMSwzNDk4OTEwLjAsMC4yNDk5MTEzNDMxMTY5NTgxNwoyLDIxODU0NTUuMCwwLjE1NjA5NzE4Mjk0MzE2NTY2CjMsNzEyMDI2LjAsMC4wNTA4NTY4MDIyNTk2MTY2NAo0LDE5OTA5MS4wLDAuMDE0MjIwMTcxMTk5NzQ0NTgyCjUsNTIwMzkuMCwwLjAwMzcxNjkxMDgwNDkyNTkzCjYsMTMyOTcuMCwwLjAwMDk0OTc0NDY3MTc0ODExMzcKNywzNDM4LjAsMC4wMDAyNDU1NjA4MTY4MzYxMjk2CjgsODE2LjAsNS44MjgzMTk1NjE5MDQ2NDY1ZS0wNQo5LDIxNS4wLDEuNTM1NjQ3OTIzNzg2MTUwNmUtMDUKMTAsNjUuMCw0LjY0MjY1NjUxMzc3MjA4NGUtMDYKMTEsMTcuMCwxLjIxNDIzMzI0MjA2MzQ2OGUtMDYKMTIsNS4wLDMuNTcxMjc0MjQxMzYzMTQxZS0wNwoxMywxLjAsNy4xNDI1NDg0ODI3MjYyODJlLTA4Cg==";
        const element = document.createElement('a');
        element.setAttribute('href', data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    });
</script>



```python
print ('Avengers win {}, Thanos win {}'.format(1 - summarize_data[:,1].sum()/test_time,summarize_data[:,1].sum()/test_time) )
```

    Avengers win 0.523922359069483, Thanos win 0.4760776409305169



```python
np.savetxt('summarize_data_num_2.txt',summarize_data.astype('uint8'),fmt='%1d')
```
