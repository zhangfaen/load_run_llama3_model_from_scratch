# git clone https://github.com/zhangfaen/load_run_llama3_model_from_scratch
# cd load_run_llama3_model_from_scratch
# pip install torch pdbpp
# huggingface-cli download zhangfaen/Meta-Llama-3-8B_checkpoint  --local-dir Meta-Llama-3-8B/
# enjoy!


import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json

# import pdb
# pdb.set_trace()

device = "cuda:0"
# device = "cpu"

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

# Below is definition of load_tiktoken_bpe function
#  143  def load_tiktoken_bpe(                                                                                                                                                                                
#  144         tiktoken_bpe_file: str, expected_hash: Optional[str] = None                                                                                                                                       
#  145     ) -> dict[bytes, int]:                                                                                                                                                                                
#  146         # NB: do not add caching to this function                                                                                                                                                         
#  147         contents = read_file_cached(tiktoken_bpe_file, expected_hash)                                                                                                                                     
#  148         return {                                                                                                                                                                                          
#  149             base64.b64decode(token): int(rank)                                                                                                                                                            
#  150             for token, rank in (line.split() for line in contents.splitlines() if line)                                                                                                                   
#  151         }  
mergeable_ranks = load_tiktoken_bpe(tokenizer_path) # type(mergeable_ranks) is dict. len(mergeable_ranks) is 128000.

tokenizer = tiktoken.Encoding(
    name="tokenizer.model",
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

# (Pdb++) tokenizer.encode("中国")
# [59795]
# (Pdb++) [k for k,v in mergeable_ranks.items() if v == 59795]
# [b'\xe4\xb8\xad\xe5\x9b\xbd']

# in python interpreter 
# >>> "中国".encode()
# b'\xe4\xb8\xad\xe5\x9b\xbd'
# >>> import base64
# >>> base64.b64encode(b'\xe4\xb8\xad\xe5\x9b\xbd')
# b'5Lit5Zu9'

# In tokenizer_path model file, there is a line: (it is text file, just vim it). 
# every line is token def pair: b64encode of that token str into utf8 bytes and its rank.
# b'5Lit5Zu9' 59795

# Here, tokenizer.n_vocab is 128256

model = torch.load("Meta-Llama-3-8B/consolidated.00.pth", map_location=device) # type(model) is dict. len(model) is 291.

total_params = sum([torch.prod(torch.tensor(p.shape)) for p in model.values()])
print(f"total_params:{total_params}")
for k,v in model.items():
    print(k, v.shape, type(v), v.device, f"{torch.prod(torch.tensor(v.shape)) / total_params:.2%}")
# total_params:8030261248
# tok_embeddings.weight torch.Size([128256, 4096]) <class 'torch.Tensor'> cuda:0 6.54%
# layers.0.attention.wq.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.0.attention.wk.weight torch.Size([1024, 4096]) <class 'torch.Tensor'> cuda:0 0.05%
# layers.0.attention.wv.weight torch.Size([1024, 4096]) <class 'torch.Tensor'> cuda:0 0.05%
# layers.0.attention.wo.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.0.feed_forward.w1.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.0.feed_forward.w3.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.0.feed_forward.w2.weight torch.Size([4096, 14336]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.0.attention_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# layers.0.ffn_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# layers.1.attention.wq.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.1.attention.wk.weight torch.Size([1024, 4096]) <class 'torch.Tensor'> cuda:0 0.05%
# layers.1.attention.wv.weight torch.Size([1024, 4096]) <class 'torch.Tensor'> cuda:0 0.05%
# layers.1.attention.wo.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.1.feed_forward.w1.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.1.feed_forward.w3.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.1.feed_forward.w2.weight torch.Size([4096, 14336]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.1.attention_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# layers.1.ffn_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# layers.2.attention.wq.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.2.attention.wk.weight torch.Size([1024, 4096]) <class 'torch.Tensor'> cuda:0 0.05%
# ......
# layers.31.attention.wo.weight torch.Size([4096, 4096]) <class 'torch.Tensor'> cuda:0 0.21%
# layers.31.feed_forward.w1.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.31.feed_forward.w3.weight torch.Size([14336, 4096]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.31.feed_forward.w2.weight torch.Size([4096, 14336]) <class 'torch.Tensor'> cuda:0 0.73%
# layers.31.attention_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# layers.31.ffn_norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# norm.weight torch.Size([4096]) <class 'torch.Tensor'> cuda:0 0.00%
# output.weight torch.Size([128256, 4096]) <class 'torch.Tensor'> cuda:0 6.54%

with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
print(config)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens, device=device)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

embedding_layer = torch.nn.Embedding(vocab_size, dim, device=device)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)


def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

# We just use fixed number 64, as there are 32 heads for each layer, and hidden dim is 4096
# 4096 / 32 = 128. A complex number has 2 parts, so 128 / 2 = 64.
zero_to_one_split_into_64_parts = torch.tensor(range(64), device=device)/64

freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts) # freqs.shape = (64,)

freqs_for_each_token = torch.outer(torch.arange(17, device=device), freqs) # freqs_for_each_token.shape = (17, 64); freqs_for_each_token.dtype = torch.float32
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token) # freqs_cis.shape = (17, 64); freqs_cis.dtype = torch.complex64

final_embedding = token_embeddings_unnormalized # final_embedding.shape = torch.Size([17, 4096])
for layer in range(n_layers): # n_layers = 32
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads): # n_heads = 32
        q_layer_head = q_layer[head] # (Pdb++) q_layer_head.shape = torch.Size([128, 4096]); (Pdb++) q_layer_head.dtype = torch.bfloat16
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T) # q_per_token.shape = torch.Size([17, 128])
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T) # k_per_token.shape = torch.Size([17, 128])
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T) # v_per_token.shape = torch.Size([17, 128])
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) # q_per_token_split_into_pairs.shape = torch.Size([17, 64, 2])
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs) # q_per_token_as_complex_numbers.shape = torch.Size([17, 64])
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis) # q_per_token_split_into_pairs_rotated.shape = torch.Size([17, 64, 2])
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape) # q_per_token_rotated.shape = torch.Size([17, 128])
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2) # k_per_token_split_into_pairs.shape = torch.Size([17, 64, 2])
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs) # k_per_token_as_complex_numbers.shape = torch.Size([17, 64])
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis) # k_per_token_split_into_pairs_rotated.shape = torch.Size([17, 64, 2])
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape) # k_per_token_rotated.shape = torch.Size([17, 128])
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5 # qk_per_token.shape = torch.Size([17, 17])
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1) # mask.shape = torch.Size([17, 17])
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1) # stacked_qkv_attention.shape = torch.Size([17, 4096])
    w_layer = model[f"layers.{layer}.attention.wo.weight"] # w_layer.shape = torch.Size([4096, 4096])
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T) # embedding_delta.shape = torch.Size([17, 4096])
    embedding_after_edit = final_embedding + embedding_delta # embedding_after_edit.shape = torch.Size([17, 4096])
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"] # w1.shape = torch.Size([14336, 4096])
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"] # w2.shape = torch.Size([4096, 14336])
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"] # w3.shape = torch.Size([14336, 4096])
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T) # output_after_feedforward.shape = torch.Size([17, 4096])
    final_embedding = embedding_after_edit+output_after_feedforward

final_embedding = rms_norm(final_embedding, model["norm.weight"])

logits = torch.matmul(final_embedding[-1], model["output.weight"].T)

next_token = torch.argmax(logits, dim=-1)

print(tokenizer.decode([next_token.item()]))