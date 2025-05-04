
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso # Thêm mô hình hồi quy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Thêm mô hình hồi quy
from xgboost import XGBClassifier, XGBRegressor # Thêm mô hình hồi quy
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score # Thêm metric hồi quy
from scipy.stats import skew
from imblearn.over_sampling import SMOTE
# Có thể cần thêm: pip install imbalanced-learn

# --- Helper Function for Outlier Detection ---
def detect_outliers_iqr(series, threshold=0.1):
    """Phát hiện tỷ lệ ngoại lệ bằng phương pháp IQR."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0, False # Không phải số thì không có ngoại lệ
    series = series.dropna()
    if series.empty:
        return 0, False
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0: # Tránh trường hợp tất cả giá trị giống nhau
        return 0, False
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_ratio = len(outliers) / len(series) if len(series) > 0 else 0
    has_significant_outliers = outlier_ratio >= threshold
    return outlier_ratio, has_significant_outliers

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 Ứng dụng phân tích dữ liệu & huấn luyện mô hình")

# Upload CSV file
uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu của bạn", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")
            return None
    df = load_data(uploaded_file)

    if df is not None: # Chỉ tiếp tục nếu load data thành công
        st.subheader("🔍 Thông tin tổng quan về dữ liệu")
        st.dataframe(df.head())
        st.write("Kích thước dữ liệu:", df.shape)
        st.write("Các kiểu dữ liệu:")
        st.dataframe(df.dtypes.astype(str))

        st.subheader("📉 Phân tích Missing Values")
        missing_percent = df.isnull().mean() * 100
        missing_percent_filtered = missing_percent[missing_percent > 0].sort_values(ascending=False)

        if not missing_percent_filtered.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=missing_percent_filtered.index, y=missing_percent_filtered.values, ax=ax)
            ax.set_ylabel("Tỷ lệ thiếu dữ liệu (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.success("✅ Dữ liệu không có giá trị thiếu!")

        with st.expander("📊 Xem biểu đồ phân phối (Categorical & Numerical)"):
            st.subheader("Biểu đồ tỷ lệ giá trị theo từng cột (Categorical)")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                for col in cat_cols:
                    try:
                        fig, ax = plt.subplots()
                        value_counts = df[col].value_counts(normalize=True)
                        limit = 30 # Giới hạn số lượng thanh
                        if len(value_counts) > limit:
                            value_counts = value_counts.head(limit)
                            st.caption(f"Cột '{col}': Chỉ hiển thị {limit} giá trị phổ biến nhất.")
                        value_counts.plot(kind='bar', ax=ax)
                        ax.set_title(f"Tỷ lệ giá trị trong cột {col}")
                        ax.set_ylabel("Tỷ lệ")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Lỗi khi vẽ biểu đồ cho cột '{col}': {e}")
            else:
                st.info("Không có cột dạng Categorical để vẽ biểu đồ.")

            st.subheader("Boxplot các biến số (Numerical)")
            num_cols_plot = df.select_dtypes(include=np.number).columns
            if len(num_cols_plot) > 0:
                for col in num_cols_plot:
                    try:
                        fig, ax = plt.subplots()
                        sns.boxplot(x=df[col], ax=ax)
                        ax.set_title(f"Boxplot của {col}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Lỗi khi vẽ boxplot cho cột '{col}': {e}")
            else:
                st.info("Không có cột dạng Numerical để vẽ boxplot.")

        # --- BẮT ĐẦU PHẦN THÊM LẠI ---
        st.markdown("---")
        st.subheader("🎯 Chọn biến mục tiêu và loại bỏ cột")

        # 1. Chọn biến mục tiêu (target)
        all_columns = df.columns.tolist()
        # Đặt cột cuối cùng làm mặc định nếu có thể
        default_target_index = len(all_columns) - 1 if len(all_columns) > 0 else 0
        target_col = st.selectbox(
            "Chọn biến mục tiêu (target):",
            options=all_columns,
            index=default_target_index,
            key="target_select"
        )

        if target_col:
            st.success(f"Bạn đã chọn '{target_col}' làm biến mục tiêu.")
            # Tạo bản sao để xử lý, giữ lại df gốc nếu cần
            df_processed = df.copy()

            # 2. Tách X và y
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
            st.write(f"Kích thước X ban đầu: {X.shape}")
            st.write(f"Kích thước y ban đầu: {y.shape}")

            # 3. Chọn cột muốn loại bỏ khỏi X
            available_cols_for_removal = X.columns.tolist()
            remove_cols = st.multiselect(
                "Chọn các cột muốn loại bỏ khỏi tập đặc trưng (X):",
                options=available_cols_for_removal,
                key="remove_cols_multiselect"
            )

            if remove_cols:
                X = X.drop(columns=remove_cols)
                st.success(f"Đã loại bỏ các cột: {', '.join(remove_cols)}")
                st.write(f"Kích thước X sau khi loại bỏ cột: {X.shape}")
            else:
                st.info("Không có cột nào được chọn để loại bỏ.")

        else:
            st.warning("Vui lòng chọn biến mục tiêu để tiếp tục.")
            st.stop() # Dừng thực thi nếu chưa chọn target

        # --- KẾT THÚC PHẦN THÊM LẠI ---


        # --- Xử lý dữ liệu (Tiếp tục từ đây với X và y đã được xác định) ---
        st.markdown("---")
        st.subheader("🧹 Xử lý Missing Values trong X") # Chỉ xử lý X ở đây, y sẽ xử lý riêng nếu cần
        missing_columns_X = X.columns[X.isnull().any()].tolist()
        if missing_columns_X:
            st.write("Các cột trong X có giá trị thiếu:", missing_columns_X)
            for col in missing_columns_X:
                if pd.api.types.is_numeric_dtype(X[col]):
                    fill_value = X[col].median() # Dùng median an toàn hơn với outliers
                    X[col].fillna(fill_value, inplace=True)
                    # st.write(f"- Cột số '{col}': Điền bằng Median ({fill_value:.2f})")
                elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
                    try:
                        fill_value = X[col].mode()[0]
                        X[col].fillna(fill_value, inplace=True)
                        # st.write(f"- Cột phân loại '{col}': Điền bằng Mode ('{fill_value}')")
                    except IndexError:
                         st.warning(f"- Cột phân loại '{col}': Không tìm thấy mode (có thể cột toàn NaN). Bỏ qua.")
                    except Exception as e:
                         st.error(f"Lỗi khi điền mode cho cột '{col}': {e}")

                else:
                     st.warning(f"- Cột '{col}' trong X có kiểu dữ liệu {X[col].dtype} không được tự động xử lý missing value.")

            missing_total_after_X = X.isnull().sum().sum()
            if missing_total_after_X == 0:
                st.success("✅ Đã xử lý toàn bộ missing values trong X.")
            else:
                st.warning(f"⚠️ Vẫn còn {missing_total_after_X} giá trị thiếu trong X.")
        else:
            st.success("✅ Tập dữ liệu X không có giá trị thiếu.")

        # Xử lý missing values trong y (nếu có và nếu y là số)
        if y is not None and y.isnull().any():
             st.subheader("🧹 Xử lý Missing Values trong y (Biến mục tiêu)")
             if pd.api.types.is_numeric_dtype(y):
                  # Đối với hồi quy, điền mean/median có thể chấp nhận được
                  y_fill_value = y.median()
                  y.fillna(y_fill_value, inplace=True)
                  st.write(f"Đã điền giá trị thiếu trong biến mục tiêu y bằng median ({y_fill_value:.2f}).")
             else:
                  # Đối với phân loại, thường loại bỏ hàng có y bị thiếu
                  original_len = len(y)
                  not_na_indices = y.notna()
                  y = y[not_na_indices]
                  X = X[not_na_indices] # Quan trọng: phải lọc cả X tương ứng
                  st.warning(f"Đã loại bỏ {original_len - len(y)} hàng do giá trị thiếu trong biến mục tiêu phân loại '{target_col}'.")
                  st.write(f"Kích thước X sau khi lọc y: {X.shape}")
                  st.write(f"Kích thước y sau khi lọc: {y.shape}")


        # Encode y nếu là phân loại và chưa phải số / Xác định loại bài toán
        le_target = None # Khởi tạo le_target
        is_classification = False # Xác định là bài toán phân loại hay hồi quy
        if y is not None: # Chỉ xử lý nếu y tồn tại
            if pd.api.types.is_numeric_dtype(y):
                 # Kiểm tra số lượng giá trị duy nhất để đoán là phân loại hay hồi quy
                 unique_count_y = y.nunique()
                 if unique_count_y < 2:
                      st.error(f"Biến mục tiêu '{target_col}' chỉ có {unique_count_y} giá trị duy nhất. Không thể huấn luyện mô hình.")
                      st.stop()
                 elif unique_count_y <= 20: # Ngưỡng để đoán là phân loại (có thể điều chỉnh)
                      st.write(f"Biến mục tiêu '{target_col}' là số nhưng có ít giá trị duy nhất ({unique_count_y}). Giả định là bài toán phân loại.")
                      is_classification = True
                      # Có thể cần encode lại thành 0, 1, 2... nếu giá trị không phải vậy
                      if not np.array_equal(np.sort(y.unique()), np.arange(unique_count_y)):
                           st.write("Tiến hành Label Encoding lại cho biến mục tiêu số để đảm bảo nhãn là 0, 1, 2...")
                           le_target = LabelEncoder()
                           y = le_target.fit_transform(y)
                           try:
                               # Sửa lỗi hiển thị mapping cho LabelEncoder
                               target_mapping = {str(label): int(index) for index, label in enumerate(le_target.classes_)}
                               st.json(target_mapping)
                           except Exception as e:
                               st.warning(f"Không thể hiển thị mapping: {e}")
                 else:
                      st.write(f"Biến mục tiêu '{target_col}' là số và có nhiều giá trị duy nhất ({unique_count_y}). Giả định là bài toán hồi quy.")
                      is_classification = False
            else: # Nếu không phải số, chắc chắn là phân loại
                 st.write(f"Biến mục tiêu '{target_col}' không phải dạng số. Tiến hành Label Encoding.")
                 is_classification = True
                 le_target = LabelEncoder()
                 y = le_target.fit_transform(y)
                 st.write("Mapping của Label Encoder cho biến mục tiêu:")
                 try:
                      # Sửa lỗi hiển thị mapping cho LabelEncoder
                      target_mapping = {str(label): int(index) for index, label in enumerate(le_target.classes_)}
                      st.json(target_mapping)
                 except Exception as e:
                      st.warning(f"Không thể hiển thị mapping: {e}")


        st.markdown("---")
        st.subheader("🧮 Xử lý các biến đầu vào (Features) trong X")

        # --- Scaling biến số (Linh hoạt theo từng cột) ---
        st.markdown("#### 1. Scaling biến số (Numerical Scaling)")
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        scaling_config = {} # Lưu trữ lựa chọn scaling cho từng cột

        if num_cols:
            st.write("Cấu hình Scaling cho từng cột số:")
            cols_per_row = 3
            col_objs = st.columns(cols_per_row)
            col_idx = 0

            default_scaler_no_outlier = "MinMaxScaler"
            outlier_threshold = 0.05
            st.caption(f"(Tự động đề xuất RobustScaler nếu tỷ lệ outliers >= {outlier_threshold*100}%)")

            for col_name in num_cols:
                with col_objs[col_idx % cols_per_row]:
                    outlier_ratio, has_outliers = detect_outliers_iqr(X[col_name], threshold=outlier_threshold)
                    suggested_scaler = "RobustScaler" if has_outliers else default_scaler_no_outlier

                    label = f"Cột '{col_name}'"
                    label_detail = f" (Outliers: {outlier_ratio:.1%})"

                    options = ["Tự động", "MinMaxScaler", "StandardScaler", "RobustScaler", "Không Scale"]
                    try:
                        suggested_index = options.index(suggested_scaler)
                    except ValueError:
                        suggested_index = 1 # Mặc định là MinMaxScaler nếu có lỗi

                    user_choice = st.selectbox(
                        label + label_detail,
                        options=options,
                        index=0, # Luôn mặc định là "Tự động"
                        key=f"scaler_{col_name}"
                    )

                    if user_choice == "Tự động":
                        final_method = suggested_scaler
                    else:
                        final_method = user_choice

                    scaling_config[col_name] = final_method
                    st.caption(f"Đề xuất: {suggested_scaler} -> Chọn: {final_method}")
                    st.markdown("---") # Ngăn cách giữa các cột

                col_idx += 1

            # Áp dụng scaling dựa trên config
            st.write("🔄 **Áp dụng Scaling:**")
            scaled_cols_count = 0
            for col_name, method in scaling_config.items():
                if method != "Không Scale":
                    try:
                        if method == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        elif method == "StandardScaler":
                            scaler = StandardScaler()
                        elif method == "RobustScaler":
                            scaler = RobustScaler()

                        X[[col_name]] = scaler.fit_transform(X[[col_name]])
                        # st.write(f"- Cột '{col_name}': Áp dụng {method}") # Giảm bớt output
                        scaled_cols_count += 1
                    except Exception as e:
                        st.error(f"Lỗi khi scale cột '{col_name}' bằng {method}: {e}")
                # else:
                     # st.write(f"- Cột '{col_name}': Bỏ qua Scaling") # Giảm bớt output

            if scaled_cols_count > 0:
                 st.success(f"✅ Đã áp dụng scaling cho {scaled_cols_count}/{len(num_cols)} cột số.")
            else:
                 st.info("ℹ️ Không có cột số nào được scale.")

        else:
            st.write("Không có cột số để scale.")

        # --- Encoding biến phân loại ---
        st.markdown("#### 2. Mã hóa biến phân loại (Categorical Encoding)")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
       

        if categorical_cols:
            st.write("Các cột phân loại được phát hiện:", categorical_cols)
            encoding_method = st.radio(
                "Chọn phương pháp mã hóa cho tất cả các cột trên:",
                ("One-Hot Encoding", "Label Encoding (Cẩn thận!)"),
                index=0,
                key="encoding_method"
            )

            if encoding_method == "One-Hot Encoding":
                try:
                    original_cols = X.shape[1]
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
                    new_cols = X.shape[1]
                    st.success(f"✅ Đã áp dụng One-Hot Encoding. Số cột tăng từ {original_cols} lên {new_cols}.")
                except Exception as e:
                     st.error(f"Lỗi khi thực hiện One-Hot Encoding: {e}")

            elif encoding_method == "Label Encoding (Cẩn thận!)":
                st.warning("⚠️ Lưu ý: Label Encoding chỉ nên dùng cho biến có thứ tự hoặc mô hình cây.")
                le = LabelEncoder()
                encoded_cols_le = []
                for col in categorical_cols:
                     try:
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoded_cols_le.append(col)
                     except Exception as e:
                        st.error(f"Lỗi khi Label Encoding cột '{col}': {e}")
                if encoded_cols_le:
                    st.success(f"✅ Đã áp dụng Label Encoding cho các cột: {encoded_cols_le}")

            remaining_objects = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if remaining_objects:
                st.warning(f"⚠️ Vẫn còn {len(remaining_objects)} cột dạng object/category chưa được mã hóa: {remaining_objects}")

        else:
            st.write("Không có cột phân loại để mã hóa.")
            

        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

        st.subheader("Các cột boolean sau khi encode:")
        st.write(bool_cols)

        if bool_cols:
            X[bool_cols] = X[bool_cols].astype(int)
            st.success("✅ Đã chuyển đổi các cột boolean thành int.")
        else:
            st.info("Không có cột boolean nào cần chuyển đổi.")

        st.write(f"🔢 Số lượng đặc trưng (features) cuối cùng trong X: {X.shape[1]}")
        st.write("Xem trước 5 dòng dữ liệu X đã xử lý:")
        st.dataframe(X.head())


        # --- Chia Train/Test ---
        st.markdown("---")
        st.subheader("🚀 Chuẩn bị huấn luyện mô hình")
        test_size = st.slider("Chọn tỷ lệ tập kiểm tra (Test set size):", 0.1, 0.5, 0.2, 0.05, key="test_size")
        random_state = st.number_input("Nhập Random State:", value=42, key="random_state")

        try:
            # Stratify chỉ dùng cho phân loại
            stratify_option = y if is_classification and y is not None else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_option
            )
            st.write(f"Kích thước tập huấn luyện (Train): {X_train.shape}, {y_train.shape}")
            st.write(f"Kích thước tập kiểm tra (Test): {X_test.shape}, {y_test.shape}")

            # --- TÙY CHỌN SMOTE ---
            if is_classification: # Chỉ hiển thị SMOTE nếu là bài toán phân loại
                st.markdown("#### ⚖️ Cân bằng dữ liệu huấn luyện (SMOTE - Tùy chọn)")
                apply_smote = st.checkbox("Áp dụng SMOTE cho tập huấn luyện?", key="apply_smote")

                if apply_smote:
                    st.write("Phân phối lớp (Train) trước SMOTE:", np.bincount(y_train))
                    smote = SMOTE(random_state=random_state)
                    try:
                        if not X_train.select_dtypes(exclude=np.number).empty:
                             st.error("Lỗi SMOTE: Tập dữ liệu huấn luyện vẫn còn cột không phải số. Vui lòng kiểm tra lại bước Encoding.")
                        else:
                            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                            st.success("✅ Áp dụng SMOTE thành công!")
                            st.write("Kích thước tập huấn luyện sau SMOTE:", X_train_resampled.shape, y_train_resampled.shape)
                            st.write("Phân phối lớp (Train) sau SMOTE:", np.bincount(y_train_resampled))
                            X_train = X_train_resampled
                            y_train = y_train_resampled
                    except Exception as e:
                        st.error(f"Lỗi khi áp dụng SMOTE: {e}")
                else:
                    st.info("ℹ️ Không áp dụng SMOTE.")
            # --- KẾT THÚC PHẦN SMOTE ---

            # --- Chọn và Huấn luyện Mô hình ---
            st.markdown("---")
            st.subheader("🤖 Chọn và Huấn luyện Mô hình")

            if is_classification:
                model_options = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
                st.write("Chọn mô hình phân loại:")
            else:
                # Thêm các mô hình hồi quy ở đây nếu muốn
                model_options = ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor", "XGBoost Regressor"] # Ví dụ
                st.write("Chọn mô hình hồi quy:")
                # Cần import các mô hình hồi quy tương ứng (đã import ở đầu)


            model_choice = st.selectbox("Chọn mô hình:", model_options, key="model_choice")

            if st.button(f"🚀 Huấn luyện mô hình {model_choice}"):
                with st.spinner(f"Đang huấn luyện {model_choice}..."):
                    model = None
                    try:
                        # --- Khởi tạo mô hình ---
                        if model_choice == "Logistic Regression":
                            if not is_classification: st.error("Logistic Regression chỉ dùng cho phân loại."); st.stop()
                            model = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
                        elif model_choice == "Decision Tree":
                            if not is_classification: st.error("Decision Tree Classifier chỉ dùng cho phân loại."); st.stop()
                            model = DecisionTreeClassifier(random_state=random_state)
                        elif model_choice == "Random Forest":
                             if is_classification:
                                 model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
                             else:
                                 model = RandomForestRegressor(random_state=random_state, n_jobs=-1) # RF cho hồi quy
                        elif model_choice == "XGBoost":
                             if is_classification:
                                 num_classes_final = len(np.unique(y_train))
                                 model = XGBClassifier(
                                     random_state=random_state, use_label_encoder=False,
                                     eval_metric='logloss' if num_classes_final == 2 else 'mlogloss',
                                     n_jobs=-1
                                 )
                             else:
                                 model = XGBRegressor(random_state=random_state, n_jobs=-1, objective='reg:squarederror') # XGB cho hồi quy
                        # Thêm các mô hình hồi quy khác
                        elif model_choice == "Linear Regression":
                             if is_classification: st.error("Linear Regression chỉ dùng cho hồi quy."); st.stop()
                             model = LinearRegression(n_jobs=-1)
                        elif model_choice == "Ridge":
                             if is_classification: st.error("Ridge chỉ dùng cho hồi quy."); st.stop()
                             model = Ridge(random_state=random_state)
                        elif model_choice == "Lasso":
                             if is_classification: st.error("Lasso chỉ dùng cho hồi quy."); st.stop()
                             model = Lasso(random_state=random_state)
                        # Thêm các mô hình khác nếu cần...
                        else:
                             st.error("Mô hình chưa được hỗ trợ")
                             st.stop()

                        # --- Huấn luyện ---
                        model.fit(X_train, y_train)
                        st.success(f"✅ Huấn luyện mô hình {model_choice} hoàn tất!")

                        # --- Đánh giá mô hình ---
                        st.subheader("📈 Kết quả đánh giá mô hình (trên tập Test)")
                        y_pred = model.predict(X_test)

                        if is_classification:
                            st.text("Classification Report:")
                            report = classification_report(y_test, y_pred, zero_division=0)
                            st.text(report)

                            st.text("Confusion Matrix:")
                            try:
                                fig, ax = plt.subplots()
                                # Lấy nhãn gốc nếu đã encode
                                display_labels = le_target.classes_ if le_target is not None else np.unique(y_test)
                                ConfusionMatrixDisplay.from_estimator(
                                    model, X_test, y_test,
                                    ax=ax, cmap="Blues",
                                    display_labels=display_labels
                                )
                                plt.xticks(rotation=45, ha='right')
                                plt.yticks(rotation=0)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Lỗi khi vẽ Confusion Matrix: {e}")
                                from sklearn.metrics import confusion_matrix
                                cm = confusion_matrix(y_test, y_pred)
                                st.text("Ma trận nhầm lẫn (dạng số):")
                                st.dataframe(cm)
                        else: # Đánh giá hồi quy
                             mse = mean_squared_error(y_test, y_pred)
                             mae = mean_absolute_error(y_test, y_pred)
                             r2 = r2_score(y_test, y_pred)
                             st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                             st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                             st.write(f"R-squared (R²): {r2:.4f}")

                             # Biểu đồ dự đoán vs thực tế
                             fig, ax = plt.subplots()
                             ax.scatter(y_test, y_pred, alpha=0.5)
                             ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
                             ax.set_xlabel("Giá trị thực tế")
                             ax.set_ylabel("Giá trị dự đoán")
                             ax.set_title("Thực tế vs. Dự đoán")
                             st.pyplot(fig)


                    except Exception as train_error:
                        st.error(f"Lỗi trong quá trình huấn luyện hoặc đánh giá: {train_error}")

        except ValueError as ve:
             st.error(f"Lỗi khi chia dữ liệu Train/Test: {ve}")
             st.warning("Kiểm tra lại biến mục tiêu và các đặc trưng. Đảm bảo X và y có cùng số hàng.")
        except Exception as e:
             st.error(f"Đã xảy ra lỗi không mong muốn: {e}")

else:
    st.info("💡 Vui lòng tải lên file CSV để bắt đầu.")
    st.warning("Lưu ý: Bạn có thể cần cài đặt thư viện `imbalanced-learn` để sử dụng SMOTE: `pip install imbalanced-learn`")

