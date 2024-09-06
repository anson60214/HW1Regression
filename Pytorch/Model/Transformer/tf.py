import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # ensure the embed_size can be divided by heads num
        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, queries, keys, values, mask):
        # go trough linear block to actually get queries, keys and values
        queries = self.queries(queries) # (N, queries_len, embed_size)
        keys = self.keys(keys)          # (N, keys_len, embed_size)
        values = self.values(values)    # (N, value_len, embed_size)

        # Split Queries, Keys, Values into multiple heads
        queries = queries.view(queries.size(0), -1, self.heads, self.head_dim)     # (batch_size, query_len, heads, heads_dim)
        keys    = keys.view(keys.size(0), -1, self.heads, self.head_dim)
        values  = values.view(values.size(0), -1, self.heads, self.head_dim)

        # calculate the attention
        attensionScore = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) # extract b,h and mut q x d * d x k => Q * K  
        # queries shape: (batch_size, query_len, heads, heads_dim)
        # keys shape: (batch_size, key_len, heads, heads_dim)
        # attensionScore shape (batch_size, heads, query_len, key_len)

        # apply the mask if needed
        if mask is not None:
            attensionScore = attensionScore.masked_fill(mask == 0, float("-1e20")) # set the upper triangular matrix as -inf

        attention = torch.softmax(attensionScore / (self.embed_size ** (1/2)), dim=3)
        # dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1). 
        # softmax(x_i) = exp(x_i)/ sum_j (exp (x_j) ) j is the dim which is the key

        # multply values to get attention
        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            queries.size(0), -1, self.heads*self.head_dim 
        ) 
        # key_len = value_len, extract b,h and mut q x l * l x d => A * V 
         
        # attention shape: (batch_size, heads, query_len, key_len)
        # values shape: (batch_size, value_len, heads, heads_dim)
        # after einsum: (batch_size, query_len, heads, value_len) -> 
        # out = flattern last 2 dimension: (batch_size, query_len, emb_size)
        
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)

        x = self.dropout(self.norm1(attention + query)) # add & norm, x dim: (batch_size, seq_length, embed_size)
        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward + x)) # add & norm
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size, # the input number of vocab in source
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)  # to run num_layers times of transformer block
            ]
        )
        # nn.ModuleList ensures that all the modules (layers) added to it are correctly registered 
        # as part of the model and their parameters are automatically tracked by PyTorch
        # (for things like optimization, saving, loading, etc.).

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length = x.shape

        # positional Encoding
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # Example: If seq_length = 5, this would give tensor([0, 1, 2, 3, 4])
        # If N = 3 (i.e., 3 sequences in the batch) and seq_length = 5, this would expand the tensor like this:
        # tensor([[0, 1, 2, 3, 4],   # First sequence
        #         [0, 1, 2, 3, 4],   # Second sequence
        #         [0, 1, 2, 3, 4]])  # Third sequence
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)  
            # becasue query, key and value are all the same so we use out x 3 
            # and this is the input for transformer forward
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, trg_mask): # trg: target
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x)) # add & norm
        out = self.transformer_block(query, key, value, src_mask) 
        # In decoder, there is a repeat of encoder type of transfomer but key and value are from encoder
        return out
    
class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size, # the input number of vocab in target
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape

        # positional Encoding
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device='cuda',
            max_length=100
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, # the input number of vocab in source
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size, # the input number of vocab in target
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # to find the 0 in the source which 0 represent the padding words
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        ) # torch.tril: triangle low and default all 0 
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        # print('the src_mask', src_mask)
        trg_mask = self.make_trg_mask(trg)
        # print('the trg_mask', trg_mask)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # for example, this is the input for the sentenses need to be translated
    # batch=2, and seq_len(word_len)=9
    # 0 in the x sequence represent the padding token to ensure that 2 batchs have same number of word 
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    # this is the target sentense after translating
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    
    out = model(x, trg[:, :-1])
    print(out)
    print(out.shape)