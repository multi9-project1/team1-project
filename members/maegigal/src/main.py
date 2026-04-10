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

    X = train.drop(columns=[target_col]).copy()
    y = train[target_col].copy()
    X_test = test.copy()

    # test order_id 저장용
    test_order_id = test[[id_col]].copy()

    # 식별자 제거
    X.drop(columns=[id_col], inplace=True)
    X_test.drop(columns=[id_col], inplace=True)


    # =========================
    # 3. 이상값 개수 확인 (split 전)
    # delivery_delay_days는 이상치로 보지 않음
    # =========================
    print(f"""[이상값 개수 확인 - train 전체]
    product_price <= 0 : {(X["product_price"] <= 0).sum()}
    discount_percent < 0 : {(X["discount_percent"] < 0).sum()}
    discount_percent > 100 : {(X["discount_percent"] > 100).sum()}
    product_rating < 0 : {(X["product_rating"] < 0).sum()}
    product_rating > 5 : {(X["product_rating"] > 5).sum()}
    past_return_rate < 0 : {(X["past_return_rate"] < 0).sum()}
    past_return_rate > 1 : {(X["past_return_rate"] > 1).sum()}
    session_length_minutes < 0 : {(X["session_length_minutes"] < 0).sum()}
    num_product_views < 0 : {(X["num_product_views"] < 0).sum()}
    """)

    print(f"""[이상값 개수 확인 - test]
    product_price <= 0 : {(X_test["product_price"] <= 0).sum()}
    discount_percent < 0 : {(X_test["discount_percent"] < 0).sum()}
    discount_percent > 100 : {(X_test["discount_percent"] > 100).sum()}
    product_rating < 0 : {(X_test["product_rating"] < 0).sum()}
    product_rating > 5 : {(X_test["product_rating"] > 5).sum()}
    past_return_rate < 0 : {(X_test["past_return_rate"] < 0).sum()}
    past_return_rate > 1 : {(X_test["past_return_rate"] > 1).sum()}
    session_length_minutes < 0 : {(X_test["session_length_minutes"] < 0).sum()}
    num_product_views < 0 : {(X_test["num_product_views"] < 0).sum()}
    """)


    # =========================
    # 4. 이상값 -> NaN 변환 (split 전)
    # delivery_delay_days는 그대로 유지
    # =========================
    for df in [X, X_test]:
        df.loc[df["product_price"] <= 0, "product_price"] = np.nan
        df.loc[(df["discount_percent"] < 0) | (df["discount_percent"] > 100), "discount_percent"] = np.nan
        df.loc[(df["product_rating"] < 0) | (df["product_rating"] > 5), "product_rating"] = np.nan
        df.loc[(df["past_return_rate"] < 0) | (df["past_return_rate"] > 1), "past_return_rate"] = np.nan
        df.loc[df["session_length_minutes"] < 0, "session_length_minutes"] = np.nan
        df.loc[df["num_product_views"] < 0, "num_product_views"] = np.nan


    # =========================
    # 5. 파생변수 생성 (split 전)
    # 결측 허용 상태로 먼저 생성
    # =========================
    for df in [X, X_test]:
        df["discount_amount"] = df["product_price"] * (df["discount_percent"] / 100.0)
        df["paid_amount"] = df["product_price"] - df["discount_amount"]

        df["is_delayed"] = (df["delivery_delay_days"] > 0).astype(int)
        df["is_early"] = (df["delivery_delay_days"] < 0).astype(int)

        df["price_per_view"] = df["product_price"] / (df["num_product_views"] + 1)
        df["high_discount"] = (df["discount_percent"] >= 50).astype(int)
        df["rating_x_return"] = df["product_rating"] * df["past_return_rate"]
        df["view_x_session"] = df["num_product_views"] * df["session_length_minutes"]
        df["discount_x_coupon"] = df["discount_percent"] * df["used_coupon"]
        df["price_x_rating"] = df["product_price"] * df["product_rating"]
        df["price_x_discount"] = df["product_price"] * df["discount_percent"]
        df["views_x_rating"] = df["num_product_views"] * df["product_rating"]
        df["delay_x_rating"] = df["delivery_delay_days"] * df["product_rating"]
        df["paid_amount_per_view"] = df["paid_amount"] / (df["num_product_views"] + 1)

        df["log_product_price"] = np.log1p(np.clip(df["product_price"], 0, None))
        df["log_num_product_views"] = np.log1p(np.clip(df["num_product_views"], 0, None))
        df["log_session_length"] = np.log1p(np.clip(df["session_length_minutes"], 0, None))
        df["log_paid_amount"] = np.log1p(np.clip(df["paid_amount"], 0, None))


    # =========================
    # 6. train / validation 분리
    # =========================
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_tr = X_tr.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)


    # =========================
    # 7. 결측치 대체 (split 후, X_tr 기준)
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

    global_price_median = X_tr["product_price"].median()
    X_tr["product_price"] = X_tr["product_price"].fillna(global_price_median)
    X_val["product_price"] = X_val["product_price"].fillna(global_price_median)
    X_test["product_price"] = X_test["product_price"].fillna(global_price_median)

    # discount_percent -> 전체 중앙값 대체
    discount_median = X_tr["discount_percent"].median()
    X_tr["discount_percent"] = X_tr["discount_percent"].fillna(discount_median)
    X_val["discount_percent"] = X_val["discount_percent"].fillna(discount_median)
    X_test["discount_percent"] = X_test["discount_percent"].fillna(discount_median)

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

    global_rating_median = X_tr["product_rating"].median()
    X_tr["product_rating"] = X_tr["product_rating"].fillna(global_rating_median)
    X_val["product_rating"] = X_val["product_rating"].fillna(global_rating_median)
    X_test["product_rating"] = X_test["product_rating"].fillna(global_rating_median)

    # 나머지 원본 수치 컬럼 결측치 대체
    base_fill_cols = [
        "past_return_rate",
        "session_length_minutes",
        "num_product_views",
        "delivery_delay_days"
    ]

    for col in base_fill_cols:
        median_value = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(median_value)
        X_val[col] = X_val[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)


    # =========================
    # 8. 파생변수 재계산
    # 원본 수치 결측 대체 후 파생변수 다시 계산해서 NaN 제거
    # =========================
    for df in [X_tr, X_val, X_test]:
        df["discount_amount"] = df["product_price"] * (df["discount_percent"] / 100.0)
        df["paid_amount"] = df["product_price"] - df["discount_amount"]

        df["is_delayed"] = (df["delivery_delay_days"] > 0).astype(int)
        df["is_early"] = (df["delivery_delay_days"] < 0).astype(int)

        df["price_per_view"] = df["product_price"] / (df["num_product_views"] + 1)
        df["high_discount"] = (df["discount_percent"] >= 50).astype(int)
        df["rating_x_return"] = df["product_rating"] * df["past_return_rate"]
        df["view_x_session"] = df["num_product_views"] * df["session_length_minutes"]
        df["discount_x_coupon"] = df["discount_percent"] * df["used_coupon"]
        df["price_x_rating"] = df["product_price"] * df["product_rating"]
        df["price_x_discount"] = df["product_price"] * df["discount_percent"]
        df["views_x_rating"] = df["num_product_views"] * df["product_rating"]
        df["delay_x_rating"] = df["delivery_delay_days"] * df["product_rating"]
        df["paid_amount_per_view"] = df["paid_amount"] / (df["num_product_views"] + 1)

        df["log_product_price"] = np.log1p(np.clip(df["product_price"], 0, None))
        df["log_num_product_views"] = np.log1p(np.clip(df["num_product_views"], 0, None))
        df["log_session_length"] = np.log1p(np.clip(df["session_length_minutes"], 0, None))
        df["log_paid_amount"] = np.log1p(np.clip(df["paid_amount"], 0, None))


    # =========================
    # 9. 결측치 최종 확인
    # =========================
    print("\n[결측치 확인 - 파생변수 재계산 후]")
    print("X_tr :", X_tr.isnull().sum().sum())
    print("X_val:", X_val.isnull().sum().sum())
    print("X_test:", X_test.isnull().sum().sum())


    # =========================
    # 10. 수치형 / 범주형 변수 다시 구분
    # =========================
    num_cols = X_tr.select_dtypes(exclude=["object", "string"]).columns
    obj_cols = X_tr.select_dtypes(include=["object", "string"]).columns


    # =========================
    # 11. 범주형 변수 원-핫 인코딩
    # =========================
    for col in obj_cols:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_tr_enc = pd.DataFrame(
        encoder.fit_transform(X_tr[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_tr.index
    )

    X_val_enc = pd.DataFrame(
        encoder.transform(X_val[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_val.index
    )

    X_test_enc = pd.DataFrame(
        encoder.transform(X_test[obj_cols]),
        columns=encoder.get_feature_names_out(obj_cols),
        index=X_test.index
    )

    X_tr = pd.concat([X_tr.drop(columns=obj_cols), X_tr_enc], axis=1)
    X_val = pd.concat([X_val.drop(columns=obj_cols), X_val_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=obj_cols), X_test_enc], axis=1)


    # =========================
    # 12. 수치형 변수 스케일링
    # used_coupon, is_delayed, is_early, high_discount는 스케일링 제외
    # =========================
    scale_exclude = ["used_coupon", "is_delayed", "is_early", "high_discount"]
    scale_cols = [col for col in num_cols if col not in scale_exclude]

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
    # 13. 최종 확인
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
    # 14. csv 저장
    # =========================
    X_tr_unscaled.to_csv("X_tr_unscaled.csv", index=False)
    X_val_unscaled.to_csv("X_val_unscaled.csv", index=False)
    X_test_unscaled.to_csv("X_test_unscaled.csv", index=False)

    X_tr_scaled.to_csv("X_tr_scaled.csv", index=False)
    X_val_scaled.to_csv("X_val_scaled.csv", index=False)
    X_test_scaled.to_csv("X_test_scaled.csv", index=False)

    y_tr.to_csv("y_tr.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)

    test_order_id.to_csv("test_order_id.csv", index=False)

    print("\ncsv 저장 완료")


if __name__ == "__main__":
    main()
