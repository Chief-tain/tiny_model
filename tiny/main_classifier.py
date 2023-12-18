from typing import Tuple, List
from transformers import pipeline


classifier = pipeline("text-classification", model='./model', device=0)
print("[Classifier] Модель успешно инициализирована!")


def classify_dataset(text: List[str], batch_size=16) -> Tuple[str, float]:
    r = classifier(text, batch_size=batch_size, truncation="only_first")
    return [x['label'] for x in r]


classify_dataset(['Футболист забил гол'])
