from pathlib import Path
import csv
from datetime import datetime, timedelta
import math


DATA_PATH = Path(__file__).with_name("wind_dataset.csv")
THRESHOLD_MS = 4.0
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURE_COLUMNS = (
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_before_ms",
    "wind_speed_2h_before_ms",
)
PREDICTIONS_PATH = Path(__file__).with_name("classification_predictions.csv")
SCORES_PATH = Path(__file__).with_name("classification_scores.csv")


def main() -> None:
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    required_columns = {"prediction_time", "target_time"}
    required_columns.update(FEATURE_COLUMNS)
    missing_columns = required_columns - set(data[0]) if data else required_columns
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")

    target_speed_column = next(
        (
            column
            for column in ("wind_next1h", "target_wind_speed_ms")
            if column in data[0]
        ),
        None,
    )
    if target_speed_column is None:
        raise ValueError(
            "정답 풍속 열이 없습니다. wind_next1h 또는 target_wind_speed_ms가 필요합니다."
        )

    for row in data:
        row["_prediction_dt"] = datetime.strptime(
            row["prediction_time"], TIME_FORMAT
        )
        row["_target_dt"] = datetime.strptime(row["target_time"], TIME_FORMAT)

    data.sort(key=lambda row: row["_target_dt"])
    invalid_rows = [
        row
        for row in data
        if row["_target_dt"] - row["_prediction_dt"] != timedelta(hours=1)
    ]
    if invalid_rows:
        raise ValueError(
            "prediction_time과 target_time이 정확히 1시간 차이가 아닌 행이 있습니다:\n"
            + "\n".join(
                f"{row['prediction_time']}, {row['target_time']}"
                for row in invalid_rows
            )
        )

    for row in data:
        row["answer"] = int(float(row[target_speed_column]) >= THRESHOLD_MS)

    train_start = datetime(2023, 1, 1)
    train_end = datetime(2024, 12, 1)
    evaluation_start = datetime(2024, 12, 1)
    evaluation_end = datetime(2025, 1, 1)

    train = [
        row
        for row in data
        if train_start <= row["_target_dt"] < train_end
    ]
    evaluation = [
        row
        for row in data
        if evaluation_start <= row["_target_dt"] < evaluation_end
    ]

    def label_count(rows: list[dict[str, str]], label: int) -> int:
        return sum(row["answer"] == label for row in rows)

    print(f"정답 풍속 열: {target_speed_column}")
    print("prediction_time과 target_time의 1시간 차이 확인: 통과")
    print(
        f"학습: 전체 {len(train)}행, "
        f"0 {label_count(train, 0)}개, "
        f"1 {label_count(train, 1)}개"
    )
    print(
        f"평가: 전체 {len(evaluation)}행, "
        f"0 {label_count(evaluation, 0)}개, "
        f"1 {label_count(evaluation, 1)}개"
    )
    if not evaluation:
        raise ValueError("평가 기간에 해당하는 행이 없습니다.")

    print("첫 평가 행:")
    print(
        {
            key: value
            for key, value in evaluation[0].items()
            if not key.startswith("_")
        }
    )

    feature_rows = [
        [float(row[column]) for column in FEATURE_COLUMNS] for row in train
    ]
    evaluation_features = [
        [float(row[column]) for column in FEATURE_COLUMNS] for row in evaluation
    ]
    means = [
        sum(row[index] for row in feature_rows) / len(feature_rows)
        for index in range(len(FEATURE_COLUMNS))
    ]
    standard_deviations = [
        math.sqrt(
            sum((row[index] - means[index]) ** 2 for row in feature_rows)
            / len(feature_rows)
        )
        for index in range(len(FEATURE_COLUMNS))
    ]
    if any(deviation == 0 for deviation in standard_deviations):
        raise ValueError("학습 자료에서 표준편차가 0인 입력 특성이 있습니다.")

    def standardize(features: list[list[float]]) -> list[list[float]]:
        return [
            [
                (value - means[index]) / standard_deviations[index]
                for index, value in enumerate(row)
            ]
            for row in features
        ]

    standardized_train = standardize(feature_rows)
    standardized_evaluation = standardize(evaluation_features)

    class LogisticRegressionModel:
        def __init__(self) -> None:
            self.weights = [0.0] * (len(FEATURE_COLUMNS) + 1)

        def fit(self, features: list[list[float]], labels: list[int]) -> None:
            design = [[1.0, *row] for row in features]
            for _ in range(100):
                probabilities = [
                    self._sigmoid(sum(weight * value for weight, value in zip(self.weights, row)))
                    for row in design
                ]
                gradient = [
                    sum(
                        (probabilities[row_index] - labels[row_index]) * design[row_index][column]
                        for row_index in range(len(design))
                    )
                    + (self.weights[column] if column else 0.0)
                    for column in range(len(self.weights))
                ]
                hessian = [
                    [
                        sum(
                            probabilities[row_index]
                            * (1.0 - probabilities[row_index])
                            * design[row_index][left]
                            * design[row_index][right]
                            for row_index in range(len(design))
                        )
                        + (1.0 if left == right and left else 0.0)
                        for right in range(len(self.weights))
                    ]
                    for left in range(len(self.weights))
                ]
                update = self._solve(hessian, gradient)
                self.weights = [
                    weight - change
                    for weight, change in zip(self.weights, update)
                ]

        def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
            probabilities = [
                self._sigmoid(
                    self.weights[0]
                    + sum(weight * value for weight, value in zip(self.weights[1:], row))
                )
                for row in features
            ]
            return [[1.0 - probability, probability] for probability in probabilities]

        @staticmethod
        def _sigmoid(value: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-36.0, min(36.0, value))))

        @staticmethod
        def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
            augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
            size = len(vector)
            for column in range(size):
                pivot = max(
                    range(column, size),
                    key=lambda row: abs(augmented[row][column]),
                )
                augmented[column], augmented[pivot] = (
                    augmented[pivot],
                    augmented[column],
                )
                for row in range(column + 1, size):
                    factor = augmented[row][column] / augmented[column][column]
                    for value in range(column, size + 1):
                        augmented[row][value] -= factor * augmented[column][value]
            solution = [0.0] * size
            for row in range(size - 1, -1, -1):
                solution[row] = (
                    augmented[row][size]
                    - sum(
                        augmented[row][column] * solution[column]
                        for column in range(row + 1, size)
                    )
                ) / augmented[row][row]
            return solution

    model = LogisticRegressionModel()
    model.fit(standardized_train, [row["answer"] for row in train])

    predictions = []
    for row, features in zip(evaluation, standardized_evaluation):
        wind_speed = float(row["wind_speed_ms"])
        model_probability = model.predict_proba([features])[0][1]
        predictions.append(
            {
                "prediction_time": row["prediction_time"],
                "target_time": row["target_time"],
                "actual_wind_speed_ms": row[target_speed_column],
                "actual_answer": str(row["answer"]),
                "current_wind_speed_ms": row["wind_speed_ms"],
                "baseline_always_0": "0",
                "baseline_current_wind": str(int(wind_speed >= THRESHOLD_MS)),
                "strong_wind_probability": f"{model_probability:.6f}",
                "logistic_regression_prediction": str(
                    int(model_probability >= 0.50)
                ),
                "logistic_regression_prediction_0.20": str(
                    int(model_probability >= 0.20)
                ),
            }
        )

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(predictions[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    methods = {
        "항상 0": "baseline_always_0",
        "현재 상태 유지": "baseline_current_wind",
        "로지스틱 회귀 0.50": "logistic_regression_prediction",
        "로지스틱 회귀 0.20": "logistic_regression_prediction_0.20",
    }
    evaluation_keys = [
        (prediction["prediction_time"], prediction["target_time"])
        for prediction in predictions
    ]
    same_evaluation_rows = (
        len(predictions) == len(evaluation)
        and len(set(evaluation_keys)) == len(evaluation)
    )
    if not same_evaluation_rows:
        raise ValueError("세 예측 방법이 동일한 평가 행을 사용하지 않습니다.")

    def evaluate_method(
        predictions_to_evaluate: list[dict[str, str]],
        prediction_column: str,
    ) -> dict[str, str | int]:
        true_positive = sum(
            prediction["actual_answer"] == "1"
            and prediction[prediction_column] == "1"
            for prediction in predictions_to_evaluate
        )
        false_negative = sum(
            prediction["actual_answer"] == "1"
            and prediction[prediction_column] == "0"
            for prediction in predictions_to_evaluate
        )
        false_positive = sum(
            prediction["actual_answer"] == "0"
            and prediction[prediction_column] == "1"
            for prediction in predictions_to_evaluate
        )
        true_negative = sum(
            prediction["actual_answer"] == "0"
            and prediction[prediction_column] == "0"
            for prediction in predictions_to_evaluate
        )
        predicted_strong = true_positive + false_positive
        precision = (
            f"{true_positive / predicted_strong:.6f}"
            if predicted_strong
            else "해당 없음"
        )
        return {
            "TP": str(true_positive),
            "FN": str(false_negative),
            "FP": str(false_positive),
            "TN": str(true_negative),
            "accuracy": (
                f"{(true_positive + true_negative) / len(predictions_to_evaluate):.6f}"
            ),
            "strong_wind_precision": precision,
            "strong_wind_recall": (
                f"{true_positive / (true_positive + false_negative):.6f}"
            ),
        }

    scores = []
    for method, prediction_column in methods.items():
        score = evaluate_method(predictions, prediction_column)
        scores.append({"method": method, **score})

    with SCORES_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(scores[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)

    print(f"\n저장 파일: {PREDICTIONS_PATH.name}")
    print(f"입력 열: {', '.join(FEATURE_COLUMNS)}")
    print("모델: LogisticRegression, L2 규제(C=1), class_weight=None")
    print("강풍 확률: predict_proba()의 클래스 1 확률")
    print(f"저장된 전체 행 수: {len(predictions)}")
    print("저장된 CSV 첫 5행:")
    for prediction in predictions[:5]:
        print(prediction)
    print(f"\n동일한 평가 행 사용 여부: {'예' if same_evaluation_rows else '아니오'}")
    actual_strong_wind = sum(
        prediction["actual_answer"] == "1" for prediction in predictions
    )
    print(f"실제 강풍 건수: {actual_strong_wind}")
    print("방법별 적중 + 놓침:")
    for score in scores:
        print(
            f"{score['method']}: "
            f"{int(score['TP'])} + {int(score['FN'])} = "
            f"{int(score['TP']) + int(score['FN'])}"
        )
    print(f"\n저장 파일: {SCORES_PATH.name}")
    print("평가 결과 표:")
    print(
        "방법\tTP\tFN\tFP\tTN\t정확도\t강풍 정밀도\t강풍 재현율"
    )
    for score in scores:
        print(
            f"{score['method']}\t{score['TP']}\t{score['FN']}\t"
            f"{score['FP']}\t{score['TN']}\t{score['accuracy']}\t"
            f"{score['strong_wind_precision']}\t"
            f"{score['strong_wind_recall']}"
        )
    print("\n로지스틱 회귀 판정 기준 비교:")
    print("판정 기준\tTP\tFN\tFP\t강풍 정밀도\t강풍 재현율")
    for method, prediction_column in (
        (
            "0.50",
            "logistic_regression_prediction",
        ),
        (
            "0.20",
            "logistic_regression_prediction_0.20",
        ),
    ):
        score = evaluate_method(predictions, prediction_column)
        print(
            f"{method}\t{score['TP']}\t{score['FN']}\t{score['FP']}\t"
            f"{score['strong_wind_precision']}\t"
            f"{score['strong_wind_recall']}"
        )


if __name__ == "__main__":
    main()
