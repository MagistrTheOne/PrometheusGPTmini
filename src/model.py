"""
Архитектура PrometheusGPT Mini
Гибридная мультиязычная LLM с HM-MoE (Hybrid Multilingual Mixture-of-Experts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig


class HybridMoE(nn.Module):
    """
    Hybrid Multilingual Mixture-of-Experts
    Гибридный механизм с языковыми экспертами для мультиязычности
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts

        # Роутер для выбора экспертов
        self.router = nn.Linear(config.d_model, config.num_experts)

        # Эксперты (по языковым парам)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.expert_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.expert_size, config.d_model)
            ) for _ in range(config.num_experts)
        ])

        # Языковой детектор (встроенный в роутер)
        self.language_detector = nn.Linear(config.d_model, len(config.languages))

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов по Xavier"""
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.xavier_uniform_(self.language_detector.weight)
        nn.init.zeros_(self.language_detector.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с роутингом по экспертам

        Args:
            x: Входной тензор [batch_size, seq_len, d_model]

        Returns:
            Выход после MoE [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Языковой детектор для контекстного роутинга
        lang_logits = self.language_detector(x)  # [batch, seq, num_langs]
        lang_probs = F.softmax(lang_logits, dim=-1)

        # Роутер для выбора экспертов
        router_logits = self.router(x)  # [batch, seq, num_experts]

        # Добавляем языковой контекст к роутеру
        # Каждый эксперт соответствует паре языков
        expert_weights = []
        for i in range(self.num_experts):
            # Для каждого эксперта учитываем релевантные языки
            lang_pair_idx = i % len(self.config.languages)
            lang_weight = lang_probs[..., lang_pair_idx:lang_pair_idx+2].sum(dim=-1)
            expert_weights.append(lang_weight.unsqueeze(-1))

        expert_weights = torch.cat(expert_weights, dim=-1)  # [batch, seq, num_experts]
        router_logits = router_logits + expert_weights * 0.1  # мягкое усиление

        # Top-k роутинг
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Нормализация top-k вероятностей
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Вычисление выхода экспертов
        expert_outputs = []
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)  # [batch, seq, d_model]
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, d_model, num_experts]

        # Взвешенное суммирование top-k экспертов
        output = torch.zeros_like(x)  # [batch, seq, d_model]

        for k in range(self.top_k):
            indices = top_k_indices[..., k]  # [batch, seq]
            probs = top_k_probs[..., k]  # [batch, seq]

            # Gather по экспертам
            selected_expert = torch.gather(
                expert_outputs,
                dim=-1,
                index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d_model, 1)
            ).squeeze(-1)  # [batch, seq, d_model]

            output += probs.unsqueeze(-1) * selected_expert

        return output


class MultiHeadAttention(nn.Module):
    """Многоголовое внимание с sparse оптимизациями"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Линейные проекции
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Внимание: QK^T / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применение внимания к значениям
        context = torch.matmul(attn_weights, v)

        # Перестановка и проекция
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(context)

        return output


class TransformerBlock(nn.Module):
    """Трансформерный блок с HM-MoE"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.moe = HybridMoE(config)

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Self-attention с residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # HM-MoE feed-forward с residual connection
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)

        return x


class PrometheusGPT(nn.Module):
    """Основная модель PrometheusGPT Mini"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Токен эмбеддинг
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Позиционное кодирование
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Трансформерные слои
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Финальная нормализация
        self.norm = nn.LayerNorm(config.d_model)

        # Выходной линейный слой
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Инициализация
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели"""
        # Эмбеддинги
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Выходной слой (weight tying)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, labels=None) -> dict:
        """
        Forward pass модели

        Args:
            input_ids: Токены входа [batch_size, seq_len]
            labels: Метки для вычисления loss (опционально)

        Returns:
            Словарь с логитами и loss
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Создание позиционных индексов
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Эмбеддинги
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb

        # Маска для causal attention (треугольная)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(device).unsqueeze(0).unsqueeze(0)

        # Трансформерные слои
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                try:
                    x = torch.utils.checkpoint.checkpoint(layer, x, causal_mask, use_reentrant=False)
                except AttributeError:
                    # Fallback для старых версий PyTorch
                    x = torch.utils.checkpoint.checkpoint(layer, x, causal_mask)
            else:
                x = layer(x, causal_mask)

        # Финальная нормализация
        x = self.norm(x)

        # Выходные логиты
        logits = self.lm_head(x)

        result = {'logits': logits}

        # Вычисление loss если переданы метки
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result['loss'] = loss

        return result

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Генерация текста с nucleus sampling

        Args:
            input_ids: Начальные токены [batch_size, seq_len]
            max_new_tokens: Максимальное количество новых токенов
            temperature: Температура для sampling
            top_k: Top-k для ограничения словаря

        Returns:
            Сгенерированные токены [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Получаем предсказания для последнего токена
                outputs = self(input_ids)
                next_token_logits = outputs['logits'][:, -1, :]

                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Temperature scaling
                next_token_logits = next_token_logits / temperature

                # Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Добавляем к последовательности
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Подсчет общего количества параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Тест модели
    config = ModelConfig()
    config.print_config()

    model = PrometheusGPT(config)
    total_params = count_parameters(model)
    print(f"Actual parameters: {total_params:,}")

    # Тест forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output shape: {outputs['logits'].shape}")

    print("Model test completed successfully!")
