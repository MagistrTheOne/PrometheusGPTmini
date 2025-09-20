"""
PrometheusGPT Mini - Main Transformer Model
Author: MagistrTheOne, Krasnodar, 2025

Полная seq2seq модель на базе Transformer для генерации текста.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .encoder import Encoder
from .decoder import Decoder
from .config import model_config, training_config


class PrometheusGPTMini(nn.Module):
    """Основная модель PrometheusGPT Mini"""

    def __init__(self, config: model_config = None):
        super().__init__()

        if config is None:
            config = model_config

        # Encoder и Decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Сохраняем конфигурацию для доступа
        self.config = config

        # Инициализация весов
        self._init_weights(config.init_method)

        # Сохраняем конфигурацию
        self.config = config
        self.training_config = training_config

    def _init_weights(self, init_method: str = "xavier"):
        """Инициализация весов модели"""

        if init_method == "xavier":
            init_fn = nn.init.xavier_uniform_
        elif init_method == "kaiming":
            init_fn = nn.init.kaiming_uniform_
        else:
            init_fn = nn.init.normal_

        # Инициализируем все линейные слои
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method in ["xavier", "kaiming"]:
                    init_fn(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Прямой проход модели

        Args:
            src_tokens: исходные токены [batch_size, src_seq_len]
            tgt_tokens: целевые токены [batch_size, tgt_seq_len]
            src_mask: маска для encoder
            tgt_mask: causal маска для decoder

        Returns:
            logits: [batch_size, tgt_seq_len, vocab_size]
        """

        # Encoder
        encoder_output = self.encoder(src_tokens, src_mask)

        # Decoder
        decoder_output = self.decoder(tgt_tokens, encoder_output, src_mask, tgt_mask)

        return decoder_output

    def encode(self, src_tokens: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Только encoder (для feature extraction)"""
        return self.encoder(src_tokens, src_mask)

    def decode(self, tgt_tokens: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Только decoder (для autoregressive генерации)"""
        return self.decoder(tgt_tokens, encoder_output, src_mask, tgt_mask)

    def generate(self, src_tokens: torch.Tensor, max_length: int = 50,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.9, do_sample: bool = True) -> torch.Tensor:
        """
        Автоматическая генерация текста

        Args:
            src_tokens: исходные токены [batch_size, src_seq_len]
            max_length: максимальная длина генерации
            temperature: температура для sampling
            top_k: top-k sampling
            top_p: nucleus sampling
            do_sample: использовать sampling или greedy

        Returns:
            generated_tokens: [batch_size, max_length]
        """

        batch_size = src_tokens.size(0)
        device = src_tokens.device

        # Encoder
        with torch.no_grad():
            encoder_output = self.encode(src_tokens)
            src_mask = self._create_padding_mask(src_tokens)

        # Начинаем генерацию с BOS токена (или [PAD] если нет BOS)
        bos_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        generated = bos_token

        # Генерируем токен за токеном
        for _ in range(max_length - 1):
            tgt_mask = self._create_causal_mask(generated.size(1))

            # Decoder
            with torch.no_grad():
                logits = self.decode(generated, encoder_output, src_mask, tgt_mask)
                next_token_logits = logits[:, -1, :]  # последний токен

            # Sampling
            if do_sample:
                # Temperature
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1].unsqueeze(1)
                    next_token_logits[indices_to_remove] = -float('inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Добавляем токен к последовательности
            generated = torch.cat([generated, next_token], dim=1)

            # Остановка если сгенерирован EOS токен
            if next_token.item() == self.config.vocab_size - 1:  # EOS token
                break

        return generated

    def _create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Создать маску для паддинг токенов"""
        return (tokens != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Создать causal маску для decoder"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.unsqueeze(0)  # [1, seq_len, seq_len]

    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": "PrometheusGPT Mini",
            "author": "MagistrTheOne, Krasnodar, 2025",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.config.__dict__,
            "device": next(self.parameters()).device
        }

    def save_model(self, path: str):
        """Сохранить модель и конфигурацию"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_config': self.training_config,
            'author': 'MagistrTheOne, Krasnodar, 2025'
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """Загрузить модель из файла"""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
