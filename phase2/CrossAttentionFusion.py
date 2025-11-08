class CrossAttentionFusion(nn.Module):
    def __init__(self,img_dim,meta_dim,hidden_dim=512,num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.img_proj = nn.Linear(img_dim,hidden_dim)
        self.meta_proj = nn.Linear(meta_dim,hidden_dim)

        self.query = nn.Linear(hidden_dim,hidden_dim)
        self.key = nn.Linear(hidden_dim,hidden_dim)
        self.value = nn.Linear(hidden_dim,hidden_dim)

        self.out_proj = nn.Linear(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,img_features,meta_features):
        batch_size = img_features.size(0)

        img_proj = self.img_proj(img_features)
        meta_proj = self.meta_proj(meta_features)

        Q =self.query(meta_proj).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(img_proj).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(img_proj).view(batch_size, self.num_heads, self.head_dim)

        attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs,V)
        attn_output = attn_output.view(batch_size, self.hidden_dim)

        output = self.out_proj(attn_output)
        output = self.layer_norm(output+meta_proj)
        
        return output
