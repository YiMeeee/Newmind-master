from model.lite.rnewmind_base import RNewMindConfig, RNewMindForCausalLM

# Pro-3B: 2560 hidden, 32 layers, 32 heads, 8 KV heads, 8192 FFN dim, 6400 vocab, 4096 ctx
PRO_3B_CONFIG = RNewMindConfig(
    hidden_size=2560,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=8192,
    vocab_size=6400,
    max_position_embeddings=4096,
)


def build_pro_3b() -> RNewMindForCausalLM:
    return RNewMindForCausalLM(PRO_3B_CONFIG)
