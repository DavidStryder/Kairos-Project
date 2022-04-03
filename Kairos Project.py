# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:00:25 2021

@author: David SoucieGarza
IDE: SpyderV3
"""

#%% 1. All imported packages. Complete list of packages in the anaconda environment available in Kairos-Project/All Packages

from typing import List, Tuple
import scipy
import numpy as np
import os.path
from pathlib import Path
import pickle
from io import open
import glob 
import os
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch #Py Torch
import torch.nn as nn
import torch.nn.functional as Func




from transformers import BertModel, BertTokenizer, BertConfig

#%% 2. Instantiation of BERT Tokenizer

#Bert model. Source: https://huggingface.co/transformers/model_doc/bert.html
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)



#%% 2.5 Training Data, including html links to their source


#Training and evaluation data, taken from text files
    #Sources: 
    #Github (text files themselves): What if I got a github
    #Library of Babel: https://libraryofbabel.info/
    #Random Text Generator: http://www.randomtextgenerator.com/
    #Project Gutenberg: https://www.gutenberg.org/
    #Books used: Heart of Darkness, Criminal Psychology, Mob Rule in New Orleans, History of the Impeachment of Andrew Johnson of The United States, Captain Singleton
    #Poetry Foundation: https://www.poetryfoundation.org/


    
labelled_data = {}

def findFiles(path): return glob.glob(path)

def readText(folder: str, category: str, read_dict: dict):
    
    read_list: List[str] = []
    
    for filename in findFiles(folder):
    
        text = open(filename, encoding='utf-8').read().strip()
        print (filename)
        read_list.append(text)
        
    read_dict[category] = read_list
    
readText('Kairos_Project_Training_Data/Chaos/*.txt', 'Chaos', labelled_data)
readText('Kairos_Project_Training_Data/Nonsense/*.txt', 'Nonsense', labelled_data)
readText('Kairos_Project_Training_Data/Gibberish/*.txt', 'Gibberish', labelled_data)
readText('Kairos_Project_Training_Data/English/*.txt', 'English', labelled_data)

labelled_evaluation_data = {}

readText('Kairos_Project_Evaluation_Data/Chaos/*.txt', 'Chaos', labelled_evaluation_data)
readText('Kairos_Project_Evaluation_Data/Nonsense/*.txt', 'Nonsense', labelled_evaluation_data)
readText('Kairos_Project_Evaluation_Data/Gibberish/*.txt', 'Gibberish', labelled_evaluation_data)
readText('Kairos_Project_Evaluation_Data/English/*.txt', 'English', labelled_evaluation_data)

#%% 3. All instantiated classes

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

#%% 4. All functions and constant variables

#Don't change this stuff. It'll make everything very messy
N_HIDDEN = 200
N_CATEGORIES = 4
INPUT_SIZE = 30522
all_categories: List[str] = ["Chaos", "Nonsense", "Gibberish", "English"]
LEARNING_RATE = 0.005 #Too high, explooooosion of backpropagation. Too low, doesn't backpropagate enough. Change at your own risk
criterion = nn.NLLLoss()


#Gets a token. Inputs a string of text and converts it into a list of tokens, then a Torch Tensor, via the Bert tokenizer
def get_tokens_tensor(text: str, tokenizer: BertTokenizer, 
               config: BertConfig) -> List[int]:
    
    tokens = tokenizer.tokenize(text)
    
    max_length = config.max_position_embeddings
    tokens = tokens[:max_length-1] # Will add special begin token
    
    tokens = [tokenizer.cls_token] + tokens
    
    token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
    token_ids_tensor = torch.tensor(token_ids)
    token_ids_tensor = torch.unsqueeze(token_ids_tensor, 1)
    token_ids_tensor = Func.one_hot(token_ids_tensor, num_classes = 30522)
    
    return token_ids_tensor

#A full batch of tokens, results in a list of tensors.
def enter_tokens(strings: List[str], length: int) -> List[List[int]]:
    
    token_ids_tensor_list: List[str] = []
    
    for string in strings:
        token_ids_tensor = get_tokens_tensor(str)
        token_ids_tensor_list.append(token_ids_tensor)
        Func.one_hot(token_ids_tensor, num_classes = 4)
        
    return token_ids_tensor_list

#The actual results. Takes the output Tensor, gets the highest probabillity value, tells ya what it is.
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

#Get number for random training example
def randomChoice(array): return array[random.randint(0, len(array) - 1)]

#Gets an actual random training example
def randomTrainingExample():
    category = randomChoice(all_categories)
    text = randomChoice(labelled_data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    data_tensor = get_tokens_tensor(text, tokenizer, config)
    return category, text, category_tensor, data_tensor

#Trains. Runs the model with backpropagation
def train(category_tensor, text_tensor):
    hidden = rnn.initHidden()
    
    rnn.zero_grad()
    
    for i in range(text_tensor.size()[0]):
        output, hidden = rnn(text_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-LEARNING_RATE)
    
    return output, loss.item()
        
#Tests. Runs the model without backpropagation
def evaluate(text_tensor):
    hidden = rnn.initHidden()
    
    for i in range(text_tensor.size()[0]):
        output, hidden = rnn(text_tensor[i], hidden)
        
    return output

#Timer for training intervals
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#The same as Training Sample, but specifically uses evaluation data
def randomEvaluationExample():
    category = randomChoice(all_categories)
    text = randomChoice(labelled_evaluation_data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    data_tensor = get_tokens_tensor(text, tokenizer, config)
    return category, text, category_tensor, data_tensor
    
def predict(input_line, n_predictions = 4):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(get_tokens_tensor(input_line))
        
        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            vategory_index = topi[0][i].tiem()
            


#%% 5. Instance of neural net. Pretty big file, so run this cell at your own risk, King

rnn = RNN(INPUT_SIZE, N_HIDDEN, N_CATEGORIES)
hidden = torch.zeros(1, N_HIDDEN)


#%% 5.5 Already ran it, and have some saved weights? Load it like this instead!

rnn.load_state_dict(torch.load('Kairos_RNN_Weights.pth'))


#%% 6. PUSH IT TO THE LIMIT! Training area. Run your training here. 

#output, next_hidden = rnn(input[0], hidden)
#print (output)

#print (categoryFromOutput(output))


#for i in range(15):
  #  category, text, category_tensor, data_tensor = randomTrainingExample()
   # print('category =', category, 'tensor =', data_tensor.size())
   
   
n_iters = 0
print_every = 5
plot_every = 5

   
current_loss = 0
all_losses = []



start = time.time()

for iter in range(1, n_iters + 1):
    category, text, category_tensor, text_tensor = randomTrainingExample()
    output, loss = train(category_tensor, text_tensor)
    current_loss += loss
    
    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        is_correct = 'Correct' if guess == category else 'Incorrect (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, text_tensor.size(), guess, is_correct))
        
    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
   




#%% 7. Training is hard. Take a break here and see the results.

plt.figure()
#plt.plot(all_losses)

confusion = torch.zeros(N_CATEGORIES, N_CATEGORIES)
n_confusion = 50

#Evaluate by going through bunch of examples, record correct guesses
for i in range(n_confusion):
    category, text, category_tensor, text_tensor = randomEvaluationExample()
    output = evaluate(text_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    

#Normalize by dividing every row by its sum
for i in range(N_CATEGORIES):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()



#%% 8. Always remember to save your work.

torch.save (rnn.state_dict(), 'Kairos_RNN_Weights.pth')
