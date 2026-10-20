from transformers import AutoModelForSequenceClassification

from src.config import BASE_MODEL_NAME, ID_TO_LABEL, STUDENT_NUM_LAYERS
from src.optimization.distill import build_student


def test_build_student_has_half_the_layers() -> None:
    teacher = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=len(ID_TO_LABEL)
    )
    student = build_student(teacher)
    assert len(student.bert.encoder.layer) == STUDENT_NUM_LAYERS
    assert student.config.hidden_size == teacher.config.hidden_size
