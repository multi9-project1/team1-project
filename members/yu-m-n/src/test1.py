import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_dashboard():
    print("Dashboard 실행")

    # 데이터 (나중에 모델 결과로 교체)
    df = pd.DataFrame({
        "model": ["A", "B", "C"],
        "accuracy": [0.82, 0.87, 0.79]
    })

    print("\n데이터 미리보기:")
    print(df)

    # 그래프
    plt.figure()
    sns.barplot(x="model", y="accuracy", data=df)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")

    plt.show()

    # 간단한 결론
    best_model = df.loc[df["accuracy"].idxmax()]

    print("\n결론:")
    print(f"가장 성능이 좋은 모델: {best_model['model']}")
    print(f"정확도: {best_model['accuracy']}")

if __name__ == "__main__":
    run_dashboard()