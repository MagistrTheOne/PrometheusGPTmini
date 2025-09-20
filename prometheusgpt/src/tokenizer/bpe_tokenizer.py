"""
PrometheusGPT Mini - BPE Tokenizer
Author: MagistrTheOne, Krasnodar, 2025

BPE токенизатор на базе SentencePiece для мультиязычных текстов.
"""

import os
import sentencepiece as spm
from typing import List, Dict, Union, Optional
import tempfile
import shutil


class BPETokenizer:
    """BPE токенизатор с поддержкой мультиязычности"""

    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 16000):
        """
        Args:
            model_path: путь к обученной модели SentencePiece
            vocab_size: размер словаря (используется только при обучении)
        """

        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp_model = None
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<author>': 4  # Специальный токен для авторства
        }

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, texts: List[str], model_prefix: str = "prometheus_tokenizer",
              vocab_size: int = 16000, character_coverage: float = 0.995) -> str:
        """
        Обучить токенизатор на текстах

        Args:
            texts: список текстов для обучения
            model_prefix: префикс для файлов модели
            vocab_size: размер словаря
            character_coverage: покрытие символов (0.995 = 99.5%)

        Returns:
            путь к обученной модели
        """

        # Создаем временную директорию для файлов
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, model_prefix)

        # Создаем файл с текстами для обучения
        train_file = os.path.join(temp_dir, "train.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        # Конфигурация для SentencePiece
        user_symbols = ','.join([f'<author_{i}>' for i in range(10)])

        spm_config = (
            f"--input={train_file} "
            f"--model_prefix={model_path} "
            f"--vocab_size={vocab_size} "
            f"--character_coverage={character_coverage} "
            "--model_type=bpe "
            f"--pad_id={self.special_tokens['<pad>']} "
            f"--unk_id={self.special_tokens['<unk>']} "
            f"--bos_id={self.special_tokens['<bos>']} "
            f"--eos_id={self.special_tokens['<eos>']} "
            f"--user_defined_symbols={user_symbols}"
        )

        # Обучаем модель
        spm.SentencePieceTrainer.train(spm_config)

        # Загружаем обученную модель
        model_file = f"{model_path}.model"
        self.load(model_file)

        # Копируем модель в постоянное место
        final_model_path = f"{model_prefix}.model"
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        shutil.copy2(model_file, final_model_path)

        # Очищаем временные файлы
        shutil.rmtree(temp_dir)

        return final_model_path

    def load(self, model_path: str):
        """Загрузить обученную модель"""
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)

        # Обновляем словарь специальных токенов
        self.special_tokens = {
            '<pad>': self.sp_model.pad_id(),
            '<unk>': self.sp_model.unk_id(),
            '<bos>': self.sp_model.bos_id(),
            '<eos>': self.sp_model.eos_id(),
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Закодировать текст в токены

        Args:
            text: входной текст
            add_bos: добавить токен начала последовательности
            add_eos: добавить токен конца последовательности

        Returns:
            список токенов (int)
        """

        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")

        # Кодируем текст
        tokens = self.sp_model.encode(text, out_type=int)

        # Добавляем специальные токены
        if add_bos:
            tokens = [self.special_tokens['<bos>']] + tokens
        if add_eos:
            tokens = tokens + [self.special_tokens['<eos>']]

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Декодировать токены обратно в текст

        Args:
            tokens: список токенов (int)

        Returns:
            декодированный текст
        """

        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Call load() or train() first.")

        return self.sp_model.decode(tokens)

    def encode_batch(self, texts: List[str], add_bos: bool = False,
                    add_eos: bool = False) -> List[List[int]]:
        """Закодировать батч текстов"""
        return [self.encode(text, add_bos, add_eos) for text in texts]

    def decode_batch(self, token_batches: List[List[int]]) -> List[str]:
        """Декодировать батч токенов"""
        return [self.decode(tokens) for tokens in token_batches]

    def get_vocab_size(self) -> int:
        """Получить размер словаря"""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Получить словарь токен -> id"""
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded.")

        vocab = {}
        for i in range(self.sp_model.get_piece_size()):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i

        return vocab

    def add_author_token(self, author_name: str) -> int:
        """
        Добавить специальный токен для автора

        Args:
            author_name: имя автора (например, "MagistrTheOne")

        Returns:
            ID нового токена
        """

        # Создаем токен в формате <author_MagistrTheOne>
        author_token = f"<author_{author_name}>"

        # Проверяем, существует ли токен
        try:
            token_id = self.sp_model.piece_to_id(author_token)
            return token_id
        except:
            # Если токен не существует, возвращаем <unk>
            return self.special_tokens['<unk>']

    def prepare_text_with_author(self, text: str, author: str = "MagistrTheOne",
                               city: str = "Krasnodar", year: int = 2025) -> str:
        """
        Подготовить текст с информацией об авторе

        Args:
            text: исходный текст
            author: имя автора
            city: город
            year: год

        Returns:
            текст с добавленной информацией об авторе
        """

        author_info = f"<author_{author}> <author_{city}> <author_{year}>"
        return f"{author_info} {text}"

    def __len__(self) -> int:
        """Получить размер словаря"""
        return self.get_vocab_size()

    def __call__(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Удобный вызов для кодирования"""
        return self.encode(text, add_bos=add_special_tokens, add_eos=add_special_tokens)


def create_demo_dataset() -> List[str]:
    """Создать демо датасет с текстами на разных языках"""

    texts = [
        # Русский
        "Привет, мир! Это демонстрация работы токенизатора.",
        "Меня зовут MagistrTheOne, я из Краснодара.",
        "Сегодня 2025 год, и мы создаем собственную LLM модель.",
        "Transformer - это мощная архитектура для обработки последовательностей.",

        # Английский
        "Hello, world! This is a tokenizer demonstration.",
        "My name is MagistrTheOne, I'm from Krasnodar.",
        "Today is 2025, and we're building our own LLM model.",
        "Transformer is a powerful architecture for sequence processing.",

        # Смешанный
        "Привет! Hello! Это mixed language text для testing.",
        "MagistrTheOne создает PrometheusGPT Mini в 2025 году.",

        # Математика и код
        "def forward(self, x): return x + 1",
        "y = mx + b, где m=2, b=3",
        "SELECT * FROM users WHERE id > 100",

        # Длинные тексты
        """
        Искусственный интеллект - это область компьютерных наук,
        которая занимается созданием машин, способных выполнять
        задачи, требующие человеческого интеллекта.
        """,

        """
        Artificial Intelligence is a field of computer science
        that deals with creating machines capable of performing
        tasks that require human intelligence.
        """
    ]

    return texts


if __name__ == "__main__":
    # Демонстрация работы токенизатора
    print("=== PrometheusGPT Mini BPE Tokenizer Demo ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")

    # Создаем демо датасет
    demo_texts = create_demo_dataset()
    print(f"\nDemo dataset size: {len(demo_texts)} texts")

    # Обучаем токенизатор
    print("\nTraining tokenizer...")
    tokenizer = BPETokenizer()
    model_path = tokenizer.train(demo_texts, "demo_tokenizer", vocab_size=1000)

    print(f"Tokenizer trained and saved to: {model_path}")

    # Тестируем кодирование
    test_text = "Привет! Меня зовут MagistrTheOne из Краснодара."
    author_text = tokenizer.prepare_text_with_author(test_text)

    print(f"\nOriginal text: {test_text}")
    print(f"Text with author: {author_text}")

    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

    # Информация о словаре
    print(f"\nVocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.special_tokens}")

    print("\n✅ Tokenizer demo completed!")
