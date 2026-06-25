# %% [markdown]
# # Transformer with Multi Head Attention

# %%
import torch
import torch.nn as nn
import math
import torch.functional as F

# %% [markdown]
# 
# ### Part 1: `self.embedding(x)` (Turning Words into Numbers)
# 
# Computers can't understand text strings like `"cat"` or `"dog"`. Before passing data into the model, tokenization turns each word into a unique ID number (for example, `"cat"` might become the integer `45`).
# 
# When you pass those word IDs (`x`) into `self.embedding(x)`, PyTorch looks up each ID inside a giant built-in dictionary matrix. It replaces every simple integer with a long row of numbers (a vector of size `d_model`, usually 512). These numbers capture the structural meaning of the word.
# 
# ---
# 
# ### Part 2: `* math.sqrt(self.d_model)` (The Scaling Trick)
# 
# This is the part that throws most people off. Why multiply your brand-new word numbers by the square root of 512 ($\sqrt{512} \approx 22.6$)?
# 
# It all comes down to what happens immediately **after** this step: **Positional Encoding**.
# 
# #### 1. Preventing the Waves from Drowning Out the Meaning
# 
# Remember that Positional Encoding adds a matrix of static sine and cosine waves to these vectors so the model knows the order of the words. The values of those sine and cosine waves naturally span between $-1.0$ and $1.0$.
# 
# When PyTorch initializes a standard `nn.Embedding` layer, it automatically scales the values to be quite small (often with a variance much less than 1.0). If you add a wave signal fluctuating between $-1.0$ and $1.0$ directly to those tiny embedding values, **the positional noise will overpower the actual meaning of the words**.
# 
# Multiplying by $\sqrt{d_{\text{model}}}$ scales up the embedding values so that the word's identity stays dominant, balancing out the positional values nicely.
# 
# #### 2. Keeping Variance Stable
# 
# As you go deeper into the model (especially during multi-head attention), you will be multiplying vectors together. Increasing the magnitude of the embedding weights at the absolute beginning helps keep the variance of the hidden states stable throughout the entire deep network architecture.
# 
# ---
# 
# ### Summary
# 
# The `forward` method takes a batch of raw tokenized word IDs, **swaps them out for deep meaning vectors** of length 512, and then **cranks up their volume** by multiplying them by $\sqrt{512}$ so they don't get drowned out when the positional wave marks are added right after!

# %%
# Step one embedding of imput
class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocabsize:int):
        super().__init__()
        self.d_model=d_model
        self.vocabsize=vocabsize
        self.embedding=nn.Embedding(vocabsize,d_model)
                                            
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

# %% [markdown]
# ### Step 1: The Blueprint Setup (`__init__`)
# 
# When the model first loads, the `__init__` constructor builds a static master map of positions.
# 
# ```python
# pe = torch.zeros(seq_len, d_model)
# 
# ```
# 
# 1. **Creates a Blank Canvas:** It builds a grid of zeros. The rows represent every possible position a word can take (`seq_len`), and the columns represent the embedding channels (`d_model`, usually 512).
# 
# ```python
# position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# 
# ```
# 
# 2. **Generates Coordinates:** * `position` becomes a vertical column listing word numbers: 
# `[[0],
#  [1],
#  [2],
#  ...]`.
# * `div_term` calculates a smooth row of 256 decreasing frequencies, scaling down from $1.0$ toward a tiny decimal.
# 
# 
# 
# ```python
# pe[:, 0::2] = torch.sin(position * div_term)  # Even columns (0, 2, 4...)
# pe[:, 1::2] = torch.cos(position * div_term)  # Odd columns (1, 3, 5...)
# 
# ```
# 
# 3. **Paints the Waves:** It multiplies the vertical word positions by the horizontal frequencies to create a massive pattern matrix. It applies a **Sine** wave to all the even-numbered columns and a **Cosine** wave to all the odd-numbered columns. This ensures that every single row has a completely unique, unrepeatable rhythmic pattern.
# 
# ```python
# pe = pe.unsqueeze(0)  # Shape becomes (1, seq_len, d_model)
# self.register_buffer('pe', pe)
# 
# ```
# 
# 4. **Locks It In:** It adds an extra placeholder dimension at the front to accommodate future sentence batches. Then, `register_buffer` tells PyTorch: *"Save this map, but do not calculate gradients for it. It is fixed math, not a trainable weight."*
# 
# ---
# 
# ## Step 2: The Live Pipeline (`forward`)
# 
# Every time a sentence passes through the network during training or translation, the `forward` pass is executed.
# 
# ```python
# def forward(self, x):
#     x = x + self.pe[:, :x.size(1), :]
#     return self.dropout(x)
# 
# ```
# 
# 1. **`x.size(1)`**: The incoming tensor `x` holds your word embeddings and has a shape of `(Batch_Size, Current_Sentence_Length, 512)`. This checks exactly how many words are in the current sentence.
# #2. **The Slice (`:x.size(1)`)**:
# If our master map `self.pe` was built to hold up to 100 words, but our current sentence only has 10 words, it crops the map to take only the first 10 rows.
# 3. **The Add (`+`)**: It adds the wave numbers directly into the word vectors. Thanks to PyTorch's broadcasting, that single master slice automatically copies and stamps itself perfectly across every single sentence in your batch simultaneously.
# 4. **The Dropout**: Finally, it randomly shuts off a few values to keep the model robust and flexible, then passes the context-ready vectors out to the Self-Attention layer.
# 
# 
# 
# 
# The `div_term` is simply a list of 256 speeds (frequencies) that start fast and gradually slow down to a crawl.
# 
# 
# ### Step-by-Step Breakdown of the Code Line
# 
# Let's look at the exact line of code:
# 
# ```python
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# 
# ```
# 
# Let's see what the computer creates at each step, assuming our total vector size (`d_model`) is 512:
# 
# #### Step 1: The Counter (`torch.arange(0, d_model, 2)`)
# 
# The number `2` at the end means "skip by 2". This simply generates a list of all the **even column numbers** in our matrix:
# 
# ```text
# [0, 2, 4, 6, 8, ..., 510]
# 
# ```
# 
# *(This gives us a list of 256 numbers).*
# 
# #### Step 2: The Negative Multiplier (`* (-math.log(10000.0) / d_model)`)
# 
# The computer calculates a single fixed number using standard math: ` -math.log(10000.0) / 512`. This equals roughly **`-0.018`**.
# 
# Now, it multiplies our list of even numbers by this negative value:
# 
# * $0 \times -0.018 = \mathbf{0}$
# * $2 \times -0.018 = \mathbf{-0.036}$
# * $4 \times -0.018 = \mathbf{-0.072}$
# * ...
# * $510 \times -0.018 = \mathbf{-9.21}$
# 
# Our list has now become a collection of increasingly negative numbers:
# 
# ```text
# [0, -0.036, -0.072, -0.108, ..., -9.21]
# 
# ```
# 
# #### Step 3: Turning it into Speeds (`torch.exp(...)`)
# 
# Finally, the code applies the exponential function ($e^x$) to our negative numbers.
# 
# There is a beautiful rule in math: **$e^0$ is exactly 1.0, and as a number becomes more and more negative, $e^{-\text{number}}$ shrinks closer and closer to 0.**
# 
# Watch how our list transforms:
# 
# * $e^{0} = \mathbf{1.0}$ (Maximum speed)
# * $e^{-0.036} = \mathbf{0.96}$
# * $e^{-0.072} = \mathbf{0.93}$
# * ...
# * $e^{-9.21} = \mathbf{0.0001}$ (Super slow speed)
# 
# ---
# 
# ### The Final Output
# 
# After running that single line of code, your `div_term` becomes a neat list of 256 gradually shrinking speeds:
# 
# ```text
# div_term = [1.0, 0.96, 0.93, 0.89, ..., 0.0001]
# 
# ```
# 
# When the code later does `position * div_term` inside the `sin` and `cos` functions, it uses these exact speeds to ensure that every single column across your 512-wide vector gets a wave that moves at a completely unique rhythm!

# %%
# Step 2 is add positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_l:int,dropout:float):
        super().__init__()
        self.seq_l=seq_l
        self.dropout=nn.Dropout(dropout)
        
        pe=torch.zeros(seq_l,d_model)
        
        position=torch.arange(0,seq_l).unsqueeze(1) # Create a vector of shape (seq_len, 1)
        
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        
        #even sin and odd cosine
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+self.pe[:, :x.size(1),:] # size(1) see what is size of current sentence so according to add pe
        return self.dropout(x)

# %%
# step 3 add normalization layer
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        x_mean=torch.mean(x,dim=-1,keepdim=True)
        x_std=torch.std(x,dim=-1,keepdim=True)
        normalized=(x-x_mean)/(x_std+self.eps)
        
        return self.alpha*normalized+self.bias

# %%
#step 4 Feed Forward NN
class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(d_ff,d_model)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
    
        return self.linear2(x)
     

# %%
#step 5 multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, 'model dimension not / by h'
        self.d_k=d_model//h
        
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout=nn.Dropout,):
        d_k=query.shape[-1] # Grabs the last dimension size (64)
        
        attention_score=(query@key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None:
            attention_score=attention_score.masked_fill(mask==0,-1e9)
        
        attention_score=torch.softmax(attention_score,dim=-1)
        
        if dropout is not None:
            attention_score=dropout(attention_score)
        
        return (attention_score @ value),attention_score
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        
        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, seq, h, d_k) -> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1],self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1],self.h, self.d_k).transpose(1,2)
        
        
        x,selfAttetion=MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        
        # Concatenate heads
        # (batch, h, seq, d_k) -> (batch, seq, h, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x)
    

# %%
#step 6 Residual connection
class ResidualConection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=dropout
        self.norm=LayerNormalization()
    def forward(self,x,sublayer):
        
        return x+self.dropout(sublayer(self.nom(x)))

# %%
# step 7 Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self,self_attention:MultiHeadAttention,feed_forward:FeedForwardNetwork):
        super().__init__()
        self.self_attention=self_attention
        self.feed_forward=feed_forward
        self.residual_connection=nn.ModuleList([
        ResidualConection(dropout) for _ in range(2)
    ])
    
    def forward(self,x):
    
        x=self.residual_connection[0](x,lambda y:self.self_attention(x,x,x,src_mask))
        x=self.residual_connection[1](x,self.feed_forward)
        
        return x

# %%
#step 8 Encoder Multiple encoder block in one class
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    
    def forward(self,x):
        for layer in self.layers:
            x=layer(x,mask)
            return self.norm(x)
    

# %%
#step 9 Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self,self_attention:MultiHeadAttention,cross_attention:MultiHeadAttention,
                 feed_foward:FeedForwardNetwork):
        super().__init__()
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.feed_foward=feed_foward
        self.residual_connections=nn.ModuleList([
            ResidualConection(nn.Dropout) for _ in range(3)
        ])
    def forward(self,x):
        x=self.residual_connections[0](
        x,lambda x:self_attention(x,x,x,tgt_mask)
        )
        x=self.cross_attention[1](
        x,lambda x:self_attention(x,encoder_outout,encoder_outout,src_mask)
        )
        x=self.residual_connections[2](
        x,lambda x:self_attention(x,self.feed_forward)
        )
        

# %%
#Step 10 Decoder Put all decoders block in one class
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    
    def forward(self,x):
        for layer in self.layers:
        
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    

# %%
# Put all in one Transformer

class Transfomer(nn.Module):
    def __init__(self,encdor:Encoder,decoder:Decoder,
                src_embeding:InputEmbedding,
                tgt_embeding:InputEmbedding,
                src_pos:PositionalEncoding,
                tgt_embedding:PositionalEncoding,
                projection_layer:nn.Linear
                ):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embeding=src_embeding
        self.tgt_embeding=tgt_embeding
        self.src_pos=src_pos
        self.tgt_embedding=tgt_embedding
        self.projection_layer=projection_layer
    
    def encoder(self,src,src_mask):
        src=self.src_embeding(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decoder(self,tgt,tgt_mask):
        tgt=self.tgt_embeding(src)
        tgt=self.tgt_pos(src)
        return self.encoder(tgt,tgt_mask)
    
    def projectionLayer(self,x):
        return self.projection_layer(x)

# %%



