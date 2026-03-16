from model.lite.rnewmind_base import RNewMindConfig

# Preset configurations for Pro models

LITE_CONFIGS = {
    "pro-1b": RNewMindConfig(
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=6,
        intermediate_size=4096,
        vocab_size=6400,
        max_position_embeddings=8192,
    ),
    "pro-3b": RNewMindConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=8192,
        vocab_size=6400,
        max_position_embeddings=4096,
    ),
}
