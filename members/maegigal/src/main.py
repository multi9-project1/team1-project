"""
maegigal - 개인 프로젝트 메인 파일

이 파일을 시작점으로 코딩을 시작하세요!
"""


def main():
    print("Hello, World! 🚀")
    print("여기서부터 코딩을 시작하세요.")
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    # =========================
    # 1. 데이터 불러오기
    # =========================
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")


    # =========================
    # 2. 타깃값 / 식별자 변수 분리
    # =========================
    target_col = "returned"
    id_col = "order_id"

    # X, y 분리
    X = train.drop(columns=[target_col]).copy()
    y = train[target_col].copy()
    X_test = test.copy()

    # 식별자 변수 제거
    X.drop(columns=[id_col], inplace=True)
    X_test.drop(columns=[id_col], inplace=True)


    # =========================
    # 3. train / validation 분리
    # =========================
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 인덱스 재정렬
    X_tr = X_tr.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)


    # =========================
    # 4. 수치형 변수 / 범주형 변수 구분
    # =========================
    num_cols = X_tr.select_dtypes(exclude=["object", "string"]).columns
    obj_cols = X_tr.select_dtypes(include=["object", "string"]).columns


    # =========================
    # 5. 이상값 개수 확인
    # =========================
    print(f"""[이상값 개수 확인 - train]
    product_price <= 0 : {(X_tr["product_price"] <= 0).sum()}
    discount_percent < 0 : {(X_tr["discount_percent"] < 0).sum()}
    discount_percent > 100 : {(X_tr["discount_percent"] > 100).sum()}
    product_rating < 0 : {(X_tr["product_rating"] < 0).sum()}
    product_rating > 5 : {(X_tr["product_rating"] > 5).sum()}
    past_return_rate < 0 : {(X_tr["past_return_rate"] < 0).sum()}
    past_return_rate > 1 : {(X_tr["past_return_rate"] > 1).sum()}
    session_length_minutes < 0 : {(X_tr["session_length_minutes"] < 0).sum()}
    num_product_views < 0 : {(X_tr["num_product_views"] < 0).sum()}
    delivery_delay_days < 0 : {(X_tr["delivery_delay_days"] < 0).sum()}
    """)


    # =========================
    # 6. 이상값 -> NaN 변환
    # =========================
    for df in [X_tr, X_val, X_test]:
        df.loc[df["product_price"] <= 0, "product_price"] = np.nan
        df.loc[(df["discount_percent"] < 0) | (df["discount_percent"] > 100), "discount_percent"] = np.nan
        df.loc[(df["product_rating"] < 0) | (df["product_rating"] > 5), "product_rating"] = np.nan
        df.loc[(df["past_return_rate"] < 0) | (df["past_return_rate"] > 1), "past_return_rate"] = np.nan
        df.loc[df["session_length_minutes"] < 0, "session_length_minutes"] = np.nan
        df.loc[df["num_product_views"] < 0, "num_product_views"] = np.nan
        df.loc[df["delivery_delay_days"] < 0, "delivery_delay_days"] = np.nan


    # =========================
    # 7. 이상값 / 결측치 대체
    # =========================
    # product_price -> product_category별 중앙값 대체
    price_median_by_cat = X_tr.groupby("product_category")["product_price"].median()

    X_tr["product_price"] = X_tr["product_price"].fillna(
        X_tr["product_category"].map(price_median_by_cat)
    )
    X_val["product_price"] = X_val["product_price"].fillna(
        X_val["product_category"].map(price_median_by_cat)
    )
    X_test["product_price"] = X_test["product_price"].fillna(
        X_test["product_category"].map(price_median_by_cat)
    )

    X_tr["product_price"] = X_tr["product_price"].fillna(X_tr["product_price"].median())
    X_val["product_price"] = X_val["product_price"].fillna(X_tr["product_price"].median())
    X_test["product_price"] = X_test["product_price"].fillna(X_tr["product_price"].median())

    # discount_percent -> 전체 중앙값 대체
    X_tr["discount_percent"] = X_tr["discount_percent"].fillna(X_tr["discount_percent"].median())
    X_val["discount_percent"] = X_val["discount_percent"].fillna(X_tr["discount_percent"].median())
    X_test["discount_percent"] = X_test["discount_percent"].fillna(X_tr["discount_percent"].median())

    # product_rating -> product_category별 중앙값 대체
    rating_median_by_cat = X_tr.groupby("product_category")["product_rating"].median()

    X_tr["product_rating"] = X_tr["product_rating"].fillna(
        X_tr["product_category"].map(rating_median_by_cat)
    )
    X_val["product_rating"] = X_val["product_rating"].fillna(
        X_val["product_category"].map(rating_median_by_cat)
    )
    X_test["product_rating"] = X_test["product_rating"].fillna(
        X_test["product_category"].map(rating_median_by_cat)
    )

    X_tr["product_rating"] = X_tr["product_rating"].fillna(X_tr["product_rating"].median())
    X_val["product_rating"] = X_val["product_rating"].fillna(X_tr["product_rating"].median())
    X_test["product_rating"] = X_test["product_rating"].fillna(X_tr["product_rating"].median())

    # 나머지 컬럼 -> 전체 중앙값 대체
    for col in ["past_return_rate", "session_length_minutes", "num_product_views", "delivery_delay_days"]:
        X_tr[col] = X_tr[col].fillna(X_tr[col].median())
        X_val[col] = X_val[col].fillna(X_tr[col].median())
        X_test[col] = X_test[col].fillna(X_tr[col].median())


    # =========================
    # 8. 이상값 대체 후 확인
    # =========================

    print(f"""[이상값 개수 확인 - train]
    product_price <= 0 : {(X_tr["product_price"] <= 0).sum()}
    discount_percent < 0 : {(X_tr["discount_percent"] < 0).sum()}
    discount_percent > 100 : {(X_tr["discount_percent"] > 100).sum()}
    product_rating < 0 : {(X_tr["product_rating"] < 0).sum()}
    product_rating > 5 : {(X_tr["product_rating"] > 5).sum()}
    past_return_rate < 0 : {(X_tr["past_return_rate"] < 0).sum()}
    past_return_rate > 1 : {(X_tr["past_return_rate"] > 1).sum()}
    session_length_minutes < 0 : {(X_tr["session_length_minutes"] < 0).sum()}
    num_product_views < 0 : {(X_tr["num_product_views"] < 0).sum()}
    delivery_delay_days < 0 : {(X_tr["delivery_delay_days"] < 0).sum()}
    """)


    # =========================
    # 9. 범주형 변수 원-핫 인코딩
    # =========================
    for col in obj_cols:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_tr_enc = pd.DataFrame(
        encoder.fit_transform(X_tr[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_tr.index,
    )

    X_val_enc = pd.DataFrame(
        encoder.transform(X_val[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_val.index,
    )

    X_test_enc = pd.DataFrame(
        encoder.transform(X_test[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_test.index,
    )

    X_tr = pd.concat([X_tr.drop(columns=obj_cols), X_tr_enc], axis=1)
    X_val = pd.concat([X_val.drop(columns=obj_cols), X_val_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=obj_cols), X_test_enc], axis=1)


    # =========================
    # 10. 수치형 변수 스케일링
    # =========================
    scale_cols = [col for col in num_cols if col != "used_coupon"]

    # 스케일링 안 한 버전 저장
    X_tr_unscaled = X_tr.copy()
    X_val_unscaled = X_val.copy()
    X_test_unscaled = X_test.copy()

    # 스케일링한 버전 생성
    scaler = StandardScaler()

    X_tr_scaled = X_tr.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_tr_scaled[scale_cols] = scaler.fit_transform(X_tr[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])


    # =========================
    # 11. 최종 확인
    # =========================
    print("\n[최종 shape 확인]")
    print(X_tr_unscaled.shape, X_val_unscaled.shape, X_test_unscaled.shape)
    print(X_tr_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)

    print("\n[컬럼 일치 여부]")
    print("train/val :", X_tr_unscaled.columns.equals(X_val_unscaled.columns))
    print("train/test:", X_tr_unscaled.columns.equals(X_test_unscaled.columns))

    print("\n[결측치 확인 - unscaled]")
    print(X_tr_unscaled.isnull().sum().sum())
    print(X_val_unscaled.isnull().sum().sum())
    print(X_test_unscaled.isnull().sum().sum())

    print("\n[결측치 확인 - scaled]")
    print(X_tr_scaled.isnull().sum().sum())
    print(X_val_scaled.isnull().sum().sum())
    print(X_test_scaled.isnull().sum().sum())


    # =========================
    # 12. csv 저장
    # =========================
    X_tr_unscaled.to_csv("X_tr_unscaled.csv", index=False)
    X_val_unscaled.to_csv("X_val_unscaled.csv", index=False)
    X_test_unscaled.to_csv("X_test_unscaled.csv", index=False)

    X_tr_scaled.to_csv("X_tr_scaled.csv", index=False)
    X_val_scaled.to_csv("X_val_scaled.csv", index=False)
    X_test_scaled.to_csv("X_test_scaled.csv", index=False)

    y_tr.to_csv("y_tr.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)

    # test order_id 저장
    test[["order_id"]].to_csv("test_order_id.csv", index=False)

    print("\ncsv 저장 완료")


if __name__ == "__main__":
    main()
