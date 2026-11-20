import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter
from datetime import timedelta


DATA_PATH = Path(__file__).with_name("classification_predictions.csv")
OUTPUT_PATH = Path(__file__).with_name("strong_wind_december.png")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def main() -> None:
    korean_font = Path("C:/Windows/Fonts/malgun.ttf")
    if korean_font.is_file():
        font_manager.fontManager.addfont(str(korean_font))
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    with DATA_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    start = datetime(2024, 12, 1, 0, 0, 0)
    end = datetime(2024, 12, 7, 23, 0, 0)

    selected = []
    for row in rows:
        target_time = datetime.strptime(row["target_time"], TIME_FORMAT)
        if start <= target_time <= end:
            row["_target_dt"] = target_time
            selected.append(row)

    selected.sort(key=lambda row: row["_target_dt"])
    if not selected:
        raise ValueError("2024년 12월 1~7일 범위의 행이 없습니다.")

    times = [row["_target_dt"] for row in selected]
    actual_wind = [float(row["actual_wind_speed_ms"]) for row in selected]

    actual_strong = [
        index for index, row in enumerate(selected)
        if int(row["actual_answer"]) == 1
    ]
    prediction_050 = [
        index for index, row in enumerate(selected)
        if int(row["logistic_regression_prediction"]) == 1
    ]
    prediction_020 = [
        index for index, row in enumerate(selected)
        if int(row["logistic_regression_prediction_0.20"]) == 1
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(times, actual_wind, color="tab:blue", label="실제 풍속")
    axes[0].axhline(
        4.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="강풍 기준선 4.0 m/s",
    )
    axes[0].set_ylabel("풍속 (m/s)")
    axes[0].set_title("2024년 12월 1~7일 강풍 관측 및 예측")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        [times[index] for index in actual_strong],
        [1] * len(actual_strong),
        color="black",
        marker="o",
        s=24,
        label="실제 강풍",
    )
    axes[1].scatter(
        [times[index] for index in prediction_050],
        [0.66] * len(prediction_050),
        color="tab:orange",
        marker="^",
        s=24,
        label="로지스틱 회귀 0.50",
    )
    axes[1].scatter(
        [times[index] for index in prediction_020],
        [0.32] * len(prediction_020),
        color="tab:red",
        marker="x",
        s=28,
        label="로지스틱 회귀 0.20",
    )
    axes[1].set_yticks([0.32, 0.66, 1])
    axes[1].set_yticklabels(
        ["0.20 예측", "0.50 예측", "실제 강풍"]
    )
    axes[1].set_ylim(0.15, 1.15)
    axes[1].set_xlabel("정답 시각 (KST)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, axis="x", alpha=0.3)

    tick_times = []
    tick_time = start
    while tick_time <= end:
        tick_times.append(tick_time)
        tick_time += timedelta(hours=12)
    axes[1].set_xticks(tick_times)
    axes[1].xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"사용한 행 수: {len(selected)}")
    print(f"PNG 저장 결과: {OUTPUT_PATH}")
    print(f"PNG 존재 여부: {OUTPUT_PATH.is_file()}")
    if OUTPUT_PATH.is_file():
        print(f"PNG 파일 크기: {OUTPUT_PATH.stat().st_size} bytes")

    december_2_mismatches = [
        row
        for row in selected
        if row["_target_dt"].date() == datetime(2024, 12, 2).date()
        and int(row["actual_answer"]) != int(row["logistic_regression_prediction"])
    ]
    if december_2_mismatches:
        row = december_2_mismatches[0]
        print("12월 2일 0.50 판정 틀린 사례:")
        print(f"정답 시각: {row['target_time']}")
        print(f"실제 풍속: {row['actual_wind_speed_ms']} m/s")
        print(f"강풍 확률: {row['strong_wind_probability']}")
        print(f"실제 정답: {row['actual_answer']}")
        print(f"0.50 예측값: {row['logistic_regression_prediction']}")
    else:
        print("12월 2일에 0.50 판정이 틀린 사례가 없습니다.")


if __name__ == "__main__":
    main()
