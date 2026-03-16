from model.lite.rnewmind_base import RNewMindConfig

# Preset configurations for Lite models

LITE_CONFIGS = {
    "small": RNewMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=6400,
        max_position_embeddings=32768,
    ),
    "base": RNewMindConfig(
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=4,
        vocab_size=6400,
        max_position_embeddings=32768,
    ),
}
