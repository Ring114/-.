#!/usr/bin/env python3
"""Генератор тестовых данных пользователей с LLM-биографиями (Grok API)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from faker import Faker
except ImportError:  # Опциональная зависимость.
    Faker = None

FIRST_NAMES = [
    "Алексей",
    "Мария",
    "Иван",
    "Екатерина",
    "Дмитрий",
    "Ольга",
    "Сергей",
    "Анна",
    "Михаил",
    "Наталья",
    "Павел",
    "Елена",
    "Роман",
    "Татьяна",
    "Никита",
    "Юлия",
]

LAST_NAMES = [
    "Иванов",
    "Петров",
    "Смирнов",
    "Кузнецов",
    "Попов",
    "Васильев",
    "Соколов",
    "Морозов",
    "Новиков",
    "Фёдоров",
    "Михайлов",
    "Белова",
    "Семенова",
    "Орлова",
    "Павлова",
    "Волкова",
]

EMAIL_DOMAINS = ["example.com", "mail.test", "demo.local", "qa-company.dev"]

INTERESTS = [
    "бег",
    "чтение нон-фикшн",
    "фотография",
    "настольные игры",
    "путешествия по России",
    "изучение языков",
    "волонтёрство",
    "городские прогулки",
    "кулинария",
    "йога",
]

CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

MAX_BIO_LENGTH = 320

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class User:
    """Модель тестового пользователя."""

    name: str
    email: str
    age: int
    interest: str
    biography: str


class LLMBiographyGenerator:
    """Генерирует биографии через Grok API или fallback-режим."""

    def __init__(
        self,
        model: str = "grok-2-latest",
        timeout: int = 25,
        disabled: bool = False,
    ) -> None:
        self.api_key = os.getenv("XAI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.disabled = disabled

    @property
    def available(self) -> bool:
        """Проверяет, можно ли отправлять запросы к API."""
        return bool(self.api_key) and not self.disabled

    def generate(self, name: str, age: int, interest: str) -> str:
        """Возвращает биографию; при сбое — fallback без падения программы."""
        if not self.available:
            return self._fallback(name, age, interest)

        prompt = textwrap.dedent(
            f"""
            Создай короткую реалистичную биографию на русском языке (2-3 предложения)
            для тестового пользователя.

            Имя: {name}
            Возраст: {age}
            Интерес: {interest}

            Требования:
            - Нейтральный и правдоподобный стиль.
            - Без чувствительных данных и без выдуманных контактов.
            - Только текст биографии, без списков и пояснений.
            """
        ).strip()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты пишешь реалистичные короткие биографии для "
                        "тестовых данных."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
        }

        request = urllib.request.Request(
            "https://api.x.ai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                if not content:
                    return self._fallback(name, age, interest)
                return self._trim_biography(content)
        except (
            KeyError,
            IndexError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
        ) as error:
            LOGGER.warning("Ошибка при генерации биографии через LLM: %s", error)
            return self._fallback(name, age, interest)

    @staticmethod
    def _trim_biography(text: str, max_len: int = MAX_BIO_LENGTH) -> str:
        """Обрезает слишком длинную биографию до разумного размера."""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_len:
            return cleaned
        shortened = cleaned[: max_len - 1].rstrip(" ,;:")
        return f"{shortened}…"

    @staticmethod
    def _fallback(name: str, age: int, interest: str) -> str:
        """Возвращает безопасную шаблонную биографию."""
        bio = (
            f"{name}, {age} лет, работает в сфере цифровых сервисов и "
            f"живет в крупном городе. В свободное время увлекается "
            f"направлением «{interest}» и любит развивать практические "
            "навыки. Коллеги отмечают ответственность и спокойный "
            "подход к задачам."
        )
        return LLMBiographyGenerator._trim_biography(bio)


def transliterate_for_email(text: str) -> str:
    """Транслитерирует строку в ASCII-представление для email local-part."""
    normalized = text.lower().replace(" ", ".")
    result: list[str] = []
    for char in normalized:
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        elif char.isascii() and (char.isalnum() or char in ".-_"):
            result.append(char)
    return "".join(result).strip(".-_") or "user"


def is_email_valid(email: str) -> bool:
    """Проверяет базовую корректность email: наличие и позицию '@'."""
    if "@" not in email:
        return False
    local, domain = email.split("@", 1)
    return bool(local) and bool(domain)


def make_email(name: str) -> str:
    """Генерирует email и гарантирует базовую валидность адреса."""
    base = transliterate_for_email(name)
    for _ in range(10):
        suffix = random.randint(10, 999)
        email = f"{base}{suffix}@{random.choice(EMAIL_DOMAINS)}"
        if is_email_valid(email):
            return email
    return f"{base}@example.com"


def generate_name(fake: Faker | None = None) -> str:
    """Генерирует имя: через Faker при наличии, иначе из локальных списков."""
    if fake is not None:
        return fake.name()
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate_users(
    count: int,
    min_age: int,
    max_age: int,
    bio_generator: LLMBiographyGenerator,
    use_faker: bool = False,
) -> list[User]:
    """Создает список пользователей с уникальными email."""
    users: list[User] = []
    used_emails: set[str] = set()
    fake = Faker("ru_RU") if (use_faker and Faker is not None) else None

    for _ in range(count):
        name = generate_name(fake)
        age = random.randint(min_age, max_age)
        interest = random.choice(INTERESTS)
        email = make_email(name)

        while email in used_emails:
            email = make_email(name)
        used_emails.add(email)

        biography = bio_generator.generate(name=name, age=age, interest=interest)
        users.append(
            User(
                name=name,
                email=email,
                age=age,
                interest=interest,
                biography=biography,
            )
        )

    return users


def write_csv(path: Path, users: list[User]) -> None:
    """Сохраняет пользователей в CSV со структурой, равной JSON-выводу."""
    fieldnames = ["name", "email", "age", "interest", "biography"]
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            writer.writerow(asdict(user))


def write_json(path: Path, users: list[User]) -> None:
    """Сохраняет пользователей в JSON со структурой, равной CSV-выводу."""
    with path.open("w", encoding="utf-8") as jsonfile:
        json.dump([asdict(user) for user in users], jsonfile, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Генератор тестовых пользователей для БД"
    )
    parser.add_argument("--count", type=int, default=20, help="Количество пользователей")
    parser.add_argument("--min-age", type=int, default=18, help="Минимальный возраст")
    parser.add_argument("--max-age", type=int, default=65, help="Максимальный возраст")
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Формат вывода",
    )
    parser.add_argument(
        "--output",
        default="generated_users.json",
        help="Имя выходного файла",
    )
    parser.add_argument("--seed", type=int, help="Фиксированный seed для повторяемости")
    parser.add_argument(
        "--model",
        default="grok-2-latest",
        help="LLM модель Grok (xAI)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Отключить запросы к LLM и использовать только fallback-биографии",
    )
    parser.add_argument(
        "--use-faker",
        action="store_true",
        help="Использовать Faker для более реалистичных имен (если установлен)",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа: валидирует параметры, генерирует и сохраняет данные."""
    args = parse_args()

    if args.count <= 0:
        raise SystemExit("--count должен быть > 0")
    if args.min_age < 0 or args.max_age < 0:
        raise SystemExit("Возраст не может быть отрицательным")
    if args.min_age > args.max_age:
        raise SystemExit("--min-age не может быть больше --max-age")

    if args.seed is not None:
        random.seed(args.seed)

    generator = LLMBiographyGenerator(
        model=args.model,
        disabled=args.no_llm,
    )
    users = generate_users(
        count=args.count,
        min_age=args.min_age,
        max_age=args.max_age,
        bio_generator=generator,
        use_faker=args.use_faker,
    )

    output_path = Path(args.output)
    if args.format == "csv":
        write_csv(output_path, users)
    else:
        write_json(output_path, users)

    print(f"Готово: сгенерировано {len(users)} пользователей в файл {output_path}")
    if args.no_llm:
        print("LLM отключен флагом --no-llm, использованы fallback-биографии.")
    elif not generator.available:
        print("XAI_API_KEY не задан, использованы шаблонные fallback-биографии.")


if __name__ == "__main__":
    main()
