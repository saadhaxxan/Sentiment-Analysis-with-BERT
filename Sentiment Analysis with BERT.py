#!/usr/bin/env python
# coding: utf-8

# ### What is BERT
# 
# BERT is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.
# 
# For more information, the original paper can be found [here](https://arxiv.org/abs/1810.04805). 
# 
# [HuggingFace documentation](https://huggingface.co/transformers/model_doc/bert.html)
# 
# [Bert documentation](https://characters.fandom.com/wiki/Bert_(Sesame_Street) ;)

# <img src="BERT_diagrams.png" width="1000">

# ## Exploratory Data Analysis and Preprocessing

# We will use the SMILE Twitter dataset.
# 
# _Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2_

# In[72]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[73]:


df = pd.read_csv('smile-annotations-final.csv',names=['id','text','category'])
df.set_index('id',inplace=True)


# In[74]:


df.head()


# In[75]:


# get unique values from category column
df.category.value_counts()


# In[76]:


# remove the multiple and nocode category as they may misslead our model
df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']


# In[77]:


df.category.value_counts()


# In[78]:


# create a dictionary for labeling the data
labels = {}
for index,value in enumerate(df.category.unique()):
    labels[value] = index


# In[79]:


labels


# In[80]:


df['label'] = df.category.replace(labels)


# In[81]:


df.head()


# ## Training/Validation Split

# As you see above that we have imbalanced data because the difference between happy and disgust labels is quite high so we use stratify technique of spliting the data which is built into sklearn

# In[82]:


from sklearn.model_selection import train_test_split


# In[83]:


x_train,x_val,y_train,y_val = train_test_split(df.index.values,df.label.values,test_size=0.15,random_state=17,stratify=df.label.values)


# In[84]:


df['data_type'] = ['not_known']*df.shape[0]


# In[85]:


df.head()


# In[86]:


df.loc[x_train,'data_type'] = 'train'
df.loc[x_val,'data_type'] = 'val'


# In[87]:


df.head()


# In[88]:


df.groupby(['category','label','data_type']).count()


# ## Loading Tokenizer and Encoding our Data

# In[89]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[90]:


tokenizer = BertTokenizer.from_pretrained(
                        'bert-base-uncased',
                        do_lower_case=True)


# In[91]:


encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type=='train'].text.values,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=256,
                                                return_tensors='pt',
                                                pad_to_max_length= True)
encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=256,
                                                return_tensors='pt',
                                                pad_to_max_length= True)


# In[92]:


input_id_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
label_train = torch.tensor(df[df.data_type=='train'].label.values)

input_id_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
label_val= torch.tensor(df[df.data_type=='val'].label.values)


# In[93]:


Dataset_train = TensorDataset(input_id_train,attention_masks_train,label_train)
Dataset_val = TensorDataset(input_id_val,attention_masks_val,label_val)


# ## Setting up BERT Pretrained Model

# In[94]:


from transformers import BertForSequenceClassification


# In[95]:


model = BertForSequenceClassification.from_pretrained(
                                    'bert-base-uncased',
                                    num_labels = len(labels),
                                    output_attentions = False,
                                    output_hidden_states = False)


# ## Creating Data Loaders

# In[96]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[97]:


dataloader_train = DataLoader(Dataset_train,
                             sampler=RandomSampler(Dataset_train),
                             batch_size=4)
dataloader_val = DataLoader(Dataset_val,
                             sampler=RandomSampler(Dataset_val),
                             batch_size=32)


# ## Setting Up Optimizer and Scheduler

# In[99]:


from transformers import AdamW, get_linear_schedule_with_warmup


# In[111]:


optimizer = AdamW(model.parameters(),
                 lr=1e-5,
                 eps=1e-8)


# In[115]:


epochs = 10
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=0,
                                           num_training_steps=len(dataloader_train)*epochs)


# ##  Defining our Performance Metrics

# In[102]:


import numpy as np


# In[103]:


from sklearn.metrics import f1_score


# In[104]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(preds_flat,labels_flat,average='weighted')


# In[124]:


def accuracy_per_class(preds, labels):
    labels_inverse = {v:k for k,v in labels.item()}
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    for labels in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==labels]
        y_true = labels_flat[labels_flat==labels]
        print(f'Class:{labels_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==labels])}/{len(y_true)}\n')


# ## Creating our Training Loop

# Approach adapted from an older version of HuggingFace's `run_glue.py` script. Accessible [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128).

# In[106]:


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[107]:


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[108]:


device = torch.device('cuda' if torch.cuda.is_available() else:'cpu')
model.to(device)
print(device)


# In[ ]:


for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train = 0
    progress_bar = tqdm(dataloader_train,desc="Epochs {:1d}".format(epoch),
                       leave=False,
                       disable=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels':batch[2]
        }
        
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        
#         optimizer.setup()
#         scheduler.setup()
        
        progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
        
    torch.save(model.state_dict(),f'finetuned_bert_epoch_1_gpu.model')
    tqdm.write('\nEpoch {epoch}')
    
    loss_train_avg = loss_train/len(dataloader)
    tqdm.write(f'Traning Loss: {loss_train_avg}')
    
    val_loss , predictions , true_values = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions,true_values)
    tqdm.write(f'validation_loss:{val_loss}')
    tqdm.write(f'F1 Score : {val_f1}')


# In[ ]:


print(torch.__version__)


# ## Traning will take 30-35 minutes per epoch on GPU

# ## Loading and Evaluating our Model

# #### Loading a trained model which is trained  on GPU

# ### Download it from Here https://mega.nz/file/9qY3BIrC#3Pd8eqpzp_qDo_VDOFChIekL8P0nNFYgZ_-TrWyYsSA

# In[119]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(labels),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# In[120]:


model.to(device)


# In[121]:


model.load_state_dict(torch.load('finetuned_bert_epoch_1_gpu_trained.model',map_location=torch.device('cpu')))


# In[122]:


_,prediction,true_values = evaluate(dataloader_val)


# In[126]:


prediction


# In[127]:


true_values


# In[ ]:




