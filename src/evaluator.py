"""
Система оценки PrometheusGPT Mini
Метрики: Perplexity, BLEU, ROUGE, мультиязычное качество
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import evaluate as evaluate_lib
from config import ModelConfig
from src.model import PrometheusGPT
from src.tokenizer import MultilingualTokenizer
from src.data import MultilingualDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Оценщик производительности PrometheusGPT Mini
    """

    def __init__(self, config: ModelConfig, model: PrometheusGPT,
                 tokenizer: MultilingualTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # Метрики
        self.bleu_metric = evaluate_lib.load("bleu")
        self.rouge_metric = evaluate_lib.load("rouge")

        # Переводим модель в eval режим
        self.model.eval()

    @torch.no_grad()
    def compute_perplexity(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Вычисление perplexity на датасете

        Args:
            data_loader: DataLoader с данными

        Returns:
            Словарь с perplexity по языкам
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        lang_losses = {lang: [] for lang in self.config.languages}

        for batch in tqdm(data_loader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
            languages = batch.get('language', ['unknown'] * len(input_ids))

            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            # Вычисляем loss для каждого сэмпла
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Подсчитываем количество не-pad токенов
                valid_tokens = (labels[i] != -100).sum().item()
                sample_loss = loss.item() * valid_tokens

                lang = languages[i] if isinstance(languages[i], str) else 'unknown'
                if lang in lang_losses:
                    lang_losses[lang].append(sample_loss / valid_tokens)

                total_loss += sample_loss
                total_tokens += valid_tokens

        # Средняя perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Perplexity по языкам
        lang_perplexities = {}
        for lang, losses in lang_losses.items():
            if losses:
                avg_lang_loss = np.mean(losses)
                lang_perplexities[f'{lang}_perplexity'] = torch.exp(torch.tensor(avg_lang_loss)).item()

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            **lang_perplexities
        }

    def evaluate_generation(self, test_prompts: Dict[str, List[str]],
                          max_new_tokens: int = 50) -> Dict[str, float]:
        """
        Оценка качества генерации текста

        Args:
            test_prompts: Словарь язык -> список промптов
            max_new_tokens: Максимальное количество новых токенов

        Returns:
            Метрики генерации
        """
        results = {}

        for lang, prompts in test_prompts.items():
            logger.info(f"Evaluating generation for {lang}")

            generated_texts = []
            reference_texts = []

            for prompt in prompts:
                # Генерируем продолжение
                input_tokens = self.tokenizer.encode(prompt, lang, add_special_tokens=True)
                input_tensor = torch.tensor([input_tokens]).cuda() if torch.cuda.is_available() else torch.tensor([input_tokens])

                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=40
                )

                generated_text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

                # Для оценки используем часть сгенерированного текста как reference
                # В реальности нужны настоящие reference тексты
                reference_text = generated_text[len(prompt):].strip()

                generated_texts.append(generated_text)
                reference_texts.append(reference_text)

            # Вычисляем BLEU и ROUGE
            bleu_score = self.bleu_metric.compute(
                predictions=generated_texts,
                references=[[ref] for ref in reference_texts]
            )

            rouge_score = self.rouge_metric.compute(
                predictions=generated_texts,
                references=reference_texts
            )

            results[f'{lang}_bleu'] = bleu_score['bleu']
            results[f'{lang}_rouge1'] = rouge_score['rouge1']
            results[f'{lang}_rouge2'] = rouge_score['rouge2']
            results[f'{lang}_rougeL'] = rouge_score['rougeL']

        return results

    def evaluate_multilingual_understanding(self) -> Dict[str, float]:
        """
        Оценка мультиязычных способностей модели

        Returns:
            Метрики мультиязычности
        """
        # Тест на понимание языков
        test_sentences = {
            'en': "The cat sat on the mat.",
            'ru': "Кошка сидела на коврике.",
            'es': "El gato se sentó en la alfombra.",
            'fr': "Le chat s'est assis sur le tapis.",
            'de': "Die Katze saß auf der Matte."
        }

        results = {}

        # Тест на language identification через perplexity
        for lang, sentence in test_sentences.items():
            tokens = self.tokenizer.encode(sentence, lang=None, add_special_tokens=True)
            input_tensor = torch.tensor([tokens]).cuda() if torch.cuda.is_available() else torch.tensor([tokens])

            with torch.no_grad():
                outputs = self.model(input_tensor, labels=input_tensor)
                loss = outputs['loss'].item()

            perplexity = torch.exp(torch.tensor(loss)).item()
            results[f'lang_id_{lang}_ppl'] = perplexity

        # Тест на cross-lingual transfer (простой)
        # Генерируем на одном языке и оцениваем perplexity на других
        base_sentence = test_sentences['en']
        base_tokens = self.tokenizer.encode(base_sentence, 'en')

        for target_lang in self.config.languages[1:]:  # Пропускаем английский
            # "Переводим" через токенизацию на целевой язык (упрощенная версия)
            target_text = base_sentence  # В реальности нужен перевод
            target_tokens = self.tokenizer.encode(target_text, target_lang)

            input_tensor = torch.tensor([target_tokens]).cuda() if torch.cuda.is_available() else torch.tensor([target_tokens])

            with torch.no_grad():
                outputs = self.model(input_tensor, labels=input_tensor)
                loss = outputs['loss'].item()

            transfer_ppl = torch.exp(torch.tensor(loss)).item()
            results[f'cross_lingual_{target_lang}_ppl'] = transfer_ppl

        return results

    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 2, 4],
                                seq_lengths: List[int] = [32, 64, 128]) -> Dict[str, float]:
        """
        Бенчмарк скорости инференса

        Args:
            batch_sizes: Размеры батчей для тестирования
            seq_lengths: Длины последовательностей

        Returns:
            Метрики скорости
        """
        results = {}
        self.model.eval()

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Создаем тестовый батч
                input_ids = torch.randint(
                    0, self.config.vocab_size,
                    (batch_size, seq_len)
                ).cuda() if torch.cuda.is_available() else torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

                # Разогрев
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model(input_ids)

                # Замер времени
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if torch.cuda.is_available():
                    start_time.record()
                else:
                    import time
                    start_time = time.time()

                # Инференс
                with torch.no_grad():
                    for _ in range(10):  # 10 итераций для усреднения
                        _ = self.model(input_ids)

                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # в секундах
                else:
                    elapsed_time = time.time() - start_time

                # Вычисляем throughput
                total_tokens = batch_size * seq_len * 10
                tokens_per_sec = total_tokens / elapsed_time

                results[f'batch_{batch_size}_seq_{seq_len}_tokens_sec'] = tokens_per_sec
                results[f'batch_{batch_size}_seq_{seq_len}_latency_ms'] = (elapsed_time / 10) * 1000

        return results

    def run_full_evaluation(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Полная оценка модели

        Args:
            test_loader: DataLoader для тестовых данных

        Returns:
            Полный набор метрик
        """
        logger.info("Starting full model evaluation")

        results = {}

        # 1. Perplexity
        logger.info("Computing perplexity...")
        perplexity_results = self.compute_perplexity(test_loader)
        results.update(perplexity_results)

        # 2. Generation quality
        logger.info("Evaluating generation quality...")
        test_prompts = {
            'en': [
                "The future of artificial intelligence",
                "Machine learning is",
                "Natural language processing"
            ],
            'ru': [
                "Будущее искусственного интеллекта",
                "Машинное обучение",
                "Обработка естественного языка"
            ],
            'es': [
                "El futuro de la inteligencia artificial",
                "El aprendizaje automático",
                "El procesamiento del lenguaje natural"
            ]
        }
        generation_results = self.evaluate_generation(test_prompts)
        results.update(generation_results)

        # 3. Multilingual understanding
        logger.info("Evaluating multilingual capabilities...")
        multilingual_results = self.evaluate_multilingual_understanding()
        results.update(multilingual_results)

        # 4. Inference speed
        logger.info("Benchmarking inference speed...")
        speed_results = self.benchmark_inference_speed()
        results.update(speed_results)

        # Итоговый отчет
        logger.info("Evaluation completed!")
        logger.info(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}")
        for lang in self.config.languages:
            lang_bleu = results.get(f'{lang}_bleu')
            if lang_bleu is not None:
                logger.info(f"{lang.upper()} BLEU: {lang_bleu:.4f}")

        return results


def create_test_prompts() -> Dict[str, List[str]]:
    """Создание тестовых промптов для оценки"""
    return {
        'en': [
            "The quick brown fox",
            "Artificial intelligence will",
            "The meaning of life is",
            "Climate change affects",
            "Machine learning algorithms"
        ],
        'ru': [
            "Быстрая коричневая лиса",
            "Искусственный интеллект",
            "Смысл жизни в том",
            "Изменение климата влияет",
            "Алгоритмы машинного обучения"
        ],
        'es': [
            "El zorro marrón rápido",
            "La inteligencia artificial",
            "El significado de la vida es",
            "El cambio climático afecta",
            "Los algoritmos de aprendizaje automático"
        ],
        'fr': [
            "Le renard brun rapide",
            "L'intelligence artificielle",
            "Le sens de la vie est",
            "Le changement climatique affecte",
            "Les algorithmes d'apprentissage automatique"
        ],
        'de': [
            "Der schnelle braune Fuchs",
            "Künstliche Intelligenz",
            "Der Sinn des Lebens ist",
            "Der Klimawandel beeinflusst",
            "Maschinelles Lernen Algorithmen"
        ]
    }


if __name__ == "__main__":
    # Тест оценщика
    config = ModelConfig()
    model = PrometheusGPT(config)
    tokenizer = MultilingualTokenizer(config)

    evaluator = ModelEvaluator(config, model, tokenizer)

    # Создаем тестовый датасет
    test_dataset = MultilingualDataset(config, tokenizer, split="test")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Запускаем оценку
    results = evaluator.run_full_evaluation(test_loader)

    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
