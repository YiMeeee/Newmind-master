from model.lite.rnewmind_base import RNewMindConfig, RNewMindForCausalLM

# Pro-1B: 1536 hidden, 24 layers, 24 heads, 6 KV heads, 4096 FFN dim, 6400 vocab, 8192 ctx
PRO_1B_CONFIG = RNewMindConfig(
    hidden_size=1536,
    num_hidden_layers=24,
    num_attention_heads=24,
    num_key_value_heads=6,
    intermediate_size=4096,
    vocab_size=6400,
    max_position_embeddings=8192,
)


def build_pro_1b() -> RNewMindForCausalLM:
    return RNewMindForCausalLM(PRO_1B_CONFIG)
