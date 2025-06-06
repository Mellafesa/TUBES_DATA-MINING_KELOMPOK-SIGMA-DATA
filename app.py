
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
import traceback 

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Prediksi Penyakit",
    page_icon="🩺",
)
st.title("Aplikasi Prediksi Penyakit Berdasarkan Gejala")
st.write("Masukkan data gejala pasien untuk memprediksi kemungkinan penyakit.")

def process_model_probabilities(prob_output, n_classes, n_samples=None):
    """Process model probability output silently"""
    try:
        if isinstance(prob_output, list):
            if len(prob_output) == 1 and isinstance(prob_output[0], np.ndarray):
                probs = prob_output[0]
            else:
                probs = np.array(prob_output)
        else:
            probs = prob_output

        if probs.ndim == 3: 
            if probs.shape[0] == n_classes and probs.shape[2] == 2: 
                probs = probs[:, :, 1].T
            elif probs.shape[1] == 1 and probs.shape[2] == 2: 
                probs = probs[:, 0, 1] 
                if n_samples == 1:
                    probs = probs.reshape(1, -1)    
            elif probs.shape[0] > 0 and probs.shape[2] == n_classes:

                probs = probs[:, -1, :] 

        # Standardize to 2D (n_samples, n_classes) or 1D (n_classes,) for single sample
        if probs.ndim == 2 and probs.shape[0] == 1 and n_samples ==1: 
            return probs.flatten()
        elif probs.ndim == 1 and n_samples == 1: 
            return probs
        elif probs.ndim == 2: 
            return probs
        else: 

            return np.ones(n_classes) / n_classes if n_classes > 0 else np.array([])

    except Exception as e:

        return np.ones(n_classes) / n_classes if n_classes > 0 else np.array([])

def process_model_predictions(pred_output):
    """Process model prediction output silently to ensure 1D array of labels."""
    try:
        if not isinstance(pred_output, np.ndarray): 
            pred_output = np.array(pred_output)

        if pred_output.ndim == 1: 
            return pred_output
        elif pred_output.ndim == 2:
            if pred_output.shape[1] == 1: # (n_samples, 1)
                return pred_output.flatten()
            elif pred_output.shape[1] > 1: 
                return np.argmax(pred_output, axis=1)
        elif np.isscalar(pred_output): 
             return np.array([pred_output])

        try:
            return pred_output.flatten() if hasattr(pred_output, 'flatten') else np.array([pred_output])
        except:
            return np.array([]) 

    except Exception:
        return np.array([])


# Load Model and Data
@st.cache_resource
def load_resources():
    model_res, df_res, desc_df_res, precaution_df_res, symptom_severity_df_res = None, None, None, None, None
    all_loaded_successfully = True
    try:
        model_res = joblib.load('diseaseprediction.joblib')
        df_res = pd.read_csv('dataset.csv')
        desc_df_res = pd.read_csv('symptom_Description.csv')
        precaution_df_res = pd.read_csv('symptom_precaution.csv')
    except FileNotFoundError:
        st.error("Satu atau lebih file data inti tidak ditemukan (diseaseprediction.joblib, dataset.csv, symptom_Description.csv, symptom_precaution.csv). Aplikasi tidak dapat berjalan.")
        all_loaded_successfully = False

    except Exception as e:
        st.error(f"Gagal memuat sumber daya inti: {e}. Aplikasi tidak dapat berjalan.")
        all_loaded_successfully = False

    try:
        symptom_severity_df_res = pd.read_csv('Symptom-severity.csv')
        symptom_severity_df_res.columns = [col.strip().lower() for col in symptom_severity_df_res.columns]
        if 'symptom' not in symptom_severity_df_res.columns or 'weight' not in symptom_severity_df_res.columns:
            st.warning("File 'Symptom-severity.csv' tidak memiliki kolom 'symptom' atau 'weight'. Skor keparahan tidak dapat dihitung/digunakan.")
            symptom_severity_df_res = None # Important to set to None
        else:
            symptom_severity_df_res['symptom'] = symptom_severity_df_res['symptom'].astype(str).str.strip().str.lower().str.replace(' ', '_')
    except FileNotFoundError:
        st.warning("File 'Symptom-severity.csv' tidak ditemukan. Skor keparahan tidak akan dihitung/digunakan oleh model atau ditampilkan.")
        symptom_severity_df_res = None
    except Exception as e:
        st.warning(f"Gagal memuat 'Symptom-severity.csv': {e}. Skor keparahan tidak akan dihitung/digunakan atau ditampilkan.")
        symptom_severity_df_res = None

    return model_res, df_res, desc_df_res, precaution_df_res, symptom_severity_df_res, all_loaded_successfully

model, df_orig, desc_df, precaution_df, symptom_severity_df, load_success = load_resources()

if not load_success or model is None or df_orig is None or desc_df is None or precaution_df is None:
    st.error("Pemuatan sumber daya inti gagal. Aplikasi tidak dapat melanjutkan.")
    st.stop() # Stop execution if core files are missing

# --- Data Preprocessing ---
df = df_orig.copy()

cols_to_drop = ['Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)


SYMPTOM_PLACEHOLDER = 'unknown_symptom_placeholder' 
symptom_string_cols = [col for col in df.columns if 'Symptom' in col and col != 'severity_score'] 

symptoms_for_multiselect_set = set()
for col in symptom_string_cols:
    if col in df.columns: 
        df[col] = df[col].astype(str).str.strip().str.lower().str.replace(' ', '_')
        df[col] = df[col].replace(['nan', '', 'none', 'null'], SYMPTOM_PLACEHOLDER) 
        symptoms_for_multiselect_set.update(s for s in df[col].unique() if s != SYMPTOM_PLACEHOLDER)
available_symptoms_for_multiselect = sorted(list(symptoms_for_multiselect_set))


# Hitung severity score
weights_dict = {}
if symptom_severity_df is not None: 
    weights_dict = symptom_severity_df.set_index('symptom')['weight'].to_dict()

def calculate_df_severity(row, symptom_cols_list, current_weights_dict):
    score = 0
    for col_name in symptom_cols_list:
        if col_name in row: 
            symptom_key = row[col_name] 
            if symptom_key != SYMPTOM_PLACEHOLDER:
                score += current_weights_dict.get(symptom_key, 0)
    return score

severity_q1_threshold = 0 
severity_q2_threshold = 0

if symptom_severity_df is not None and weights_dict: 
    df['severity_score'] = df.apply(lambda row: calculate_df_severity(row, symptom_string_cols, weights_dict), axis=1)
    if not df['severity_score'].empty:
        if df['severity_score'].nunique() > 1: 
            severity_q1_threshold = df['severity_score'].quantile(0.33)
            severity_q2_threshold = df['severity_score'].quantile(0.66)
            if severity_q1_threshold == severity_q2_threshold:
                max_score_in_df = df['severity_score'].max()
                if max_score_in_df > 0:
                    severity_q1_threshold = max_score_in_df / 3
                    severity_q2_threshold = 2 * max_score_in_df / 3
                else: 
                     severity_q1_threshold = 1 
                     severity_q2_threshold = 2 
        elif df['severity_score'].nunique() == 1: 
            unique_score = df['severity_score'].iloc[0]
            if unique_score > 0:
                severity_q1_threshold = unique_score / 3
                severity_q2_threshold = 2 * unique_score / 3
            else: 
                severity_q1_threshold = 1
                severity_q2_threshold = 2
        else: 
            severity_q1_threshold = 1 
            severity_q2_threshold = 2
    else: 
        df['severity_score'] = 0 
        st.warning("Kolom 'severity_score' kosong setelah kalkulasi. Menggunakan nilai default.")
        severity_q1_threshold = 1 
        severity_q2_threshold = 2
else: 
    df['severity_score'] = 0 
    st.warning("Data keparahan gejala (Symptom-severity.csv) tidak tersedia atau kosong. 'severity_score' diatur ke 0.")
    severity_q1_threshold = 1 
    severity_q2_threshold = 2

def categorize_severity(score, q1, q2):
    if score <= q1:
        return "Rendah"
    elif score <= q2:
        return "Sedang"
    else:
        return "Tinggi"

# Prepare X (features) and y (target)
symptom_cols_to_encode = symptom_string_cols.copy()

all_symptom_values_for_encoder = [SYMPTOM_PLACEHOLDER]
for col in symptom_cols_to_encode:
    if col in df.columns:
        all_symptom_values_for_encoder.extend(df[col].unique())
unique_symptom_encoder_values = sorted(list(set(all_symptom_values_for_encoder)))

symptom_encoder = LabelEncoder()
symptom_encoder.fit(unique_symptom_encoder_values)
symptom_mapping = dict(zip(symptom_encoder.classes_, symptom_encoder.transform(symptom_encoder.classes_)))


if 'Disease' not in df.columns:
    st.error("Kolom target 'Disease' tidak ditemukan dalam dataset. Tidak dapat melanjutkan.")
    st.stop()

X = df.drop('Disease', axis=1).copy() 
for col in symptom_cols_to_encode:
    if col in X.columns:
        X[col] = X[col].map(symptom_mapping).fillna(symptom_mapping.get(SYMPTOM_PLACEHOLDER, 0))


y_series = df['Disease']
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y_series)
n_classes = len(disease_encoder.classes_)
if n_classes == 0:
    st.error("Tidak ada kelas penyakit yang terdeteksi setelah encoding. Periksa kolom 'Disease' Anda.")
    st.stop()


# --- Feature Scaling and Clustering ---
expected_model_features_before_cluster = list(X.columns) 

scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X[expected_model_features_before_cluster])
except ValueError as e_scale:
    st.error(f"Error saat scaling fitur: {e_scale}. Ini bisa terjadi jika ada nilai non-numerik atau NaN yang tersisa di X.")
    st.error(f"Kolom di X sebelum scaling: {X.columns.tolist()}")
    st.error(f"Contoh data X:\n{X.head()}")
    st.stop()


# Clustering
clusters = np.zeros(X_scaled.shape[0], dtype=int) 
kmeans_model = None 
if X_scaled.shape[0] > 0:
    try:
        n_cluster_val = min(3, n_classes, X_scaled.shape[0]) 
        n_cluster_val = max(1, n_cluster_val)

        kmeans_model = KMeans(n_clusters=n_cluster_val, random_state=42, n_init='auto')
        clusters = kmeans_model.fit_predict(X_scaled)
    except ValueError as e_kmeans: 
        st.warning(f"Gagal melakukan K-Means clustering ({e_kmeans}). Menggunakan klaster default (0).")
        clusters = np.zeros(X_scaled.shape[0], dtype=int) 
else:
    st.warning("Tidak ada data untuk K-Means clustering.")


X_final = pd.DataFrame(X_scaled, columns=expected_model_features_before_cluster)
X_final['cluster'] = clusters 

# Train-test split
if X_final.empty or len(y) == 0:
    st.warning("Tidak cukup data untuk train-test split dan evaluasi model.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([])
else:
    try:
        stratify_y = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42, stratify=stratify_y)
    except ValueError as e_split:
        st.warning(f"Gagal melakukan stratified split ({e_split}). Melakukan split standar.")
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)


st.markdown("---")
st.subheader("🧪 Evaluasi Model Penyakit")

if X_test.empty or len(y_test) == 0:
    st.info("Evaluasi model tidak dapat dilakukan karena tidak ada data test yang cukup.")
else:
    try:
        pred_raw = model.predict(X_test) 
        y_pred = process_model_predictions(pred_raw)

        if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1 or (len(y_test) > 0 and len(y_pred) != len(y_test)):
             st.error(f"Output y_pred tidak valid. Shape: {getattr(y_pred, 'shape', 'N/A')}")
             y_pred = np.array([-1] * len(y_test)) if len(y_test) > 0 else np.array([]) 

        prob_raw = model.predict_proba(X_test)
        y_prob = process_model_probabilities(prob_raw, n_classes, len(X_test))

        accuracy = accuracy_score(y_test, y_pred) if len(y_pred) == len(y_test) and len(y_test) > 0 else 0.0

        target_names_str = [str(cls_name) for cls_name in disease_encoder.classes_]
        report = classification_report(y_test, y_pred, target_names=target_names_str, zero_division=0) if len(y_pred) == len(y_test) and len(y_test) > 0 else "Laporan tidak tersedia."

        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes)) if len(y_pred) == len(y_test) and len(y_test) > 0 else np.array([[0]*n_classes]*n_classes)

        auc_score = 0.0
        can_calc_auc = isinstance(y_prob, np.ndarray) and y_prob.ndim == 2 and y_prob.shape[0] == len(y_test) and y_prob.shape[1] == n_classes and len(np.unique(y_test)) > 1

        if can_calc_auc:
            try:
                y_bin = label_binarize(y_test, classes=range(n_classes))
                if n_classes == 2: 
                    auc_score = roc_auc_score(y_test, y_prob[:, 1]) 
                elif y_bin.shape[1] == n_classes : 
                    auc_score = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
            except ValueError: 
                pass 
            except Exception:
                pass 

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Accuracy", f"{accuracy:.2%}")
        with col2: st.metric("Classes", f"{n_classes}")
        with col3: st.metric("Test Samples", f"{len(y_test)}")
        with col4: st.metric("ROC AUC", f"{auc_score:.3f}" if auc_score > 0 else "N/A")

        viz_option = st.selectbox("Pilih visualisasi:", ["Tidak Ada", "ROC Curve", "Confusion Matrix", "Classification Report"])

        if viz_option == "ROC Curve" and can_calc_auc and auc_score > 0:
            st.subheader("ROC Curves")
            fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
            y_bin_roc = label_binarize(y_test, classes=range(n_classes))

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                ax_roc.plot(fpr, tpr, label=f'{target_names_str[1]} (AUC = {auc_score:.2f})')
            elif y_bin_roc.shape[1] == n_classes :
                for i in range(n_classes):
                    if i < len(target_names_str):
                        try:
                            fpr, tpr, _ = roc_curve(y_bin_roc[:, i], y_prob[:, i])
                            auc_val_class = roc_auc_score(y_bin_roc[:, i], y_prob[:, i])
                            ax_roc.plot(fpr, tpr, label=f'{target_names_str[i]} (AUC = {auc_val_class:.2f})')
                        except ValueError: pass 

            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curves (OVR)'); 
            if n_classes > 1 : ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_roc.grid(True, alpha=0.3); st.pyplot(fig_roc); plt.close(fig_roc)

        elif viz_option == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(max(8, n_classes*0.6), max(6, n_classes*0.5)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=target_names_str, yticklabels=target_names_str)
            ax_cm.set_title('Confusion Matrix'); ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)

        elif viz_option == "Classification Report":
            st.subheader("Classification Report"); st.text_area("Laporan:",report, height=max(300, n_classes*25) )

    except Exception as e_eval:
        st.error(f"Gagal melakukan evaluasi model: {e_eval}")
        st.error(traceback.format_exc())


st.markdown("---")
st.subheader("🔍 Prediksi Penyakit")

with st.form("prediction_form"):
    selected_symptoms_raw = st.multiselect( 
        "Pilih gejala yang dialami:",
        options=available_symptoms_for_multiselect, 
        help="Pilih satu atau lebih gejala dari daftar"
    )
    predict_btn = st.form_submit_button("🧠 Prediksi Penyakit", use_container_width=True)

if predict_btn:
    if selected_symptoms_raw:
        try:
            user_input_features_dict = {} 


            for i, col_name in enumerate(symptom_string_cols): 
                if col_name in X_final.columns: 
                    if i < len(selected_symptoms_raw):

                        user_input_features_dict[col_name] = symptom_mapping.get(selected_symptoms_raw[i], symptom_mapping[SYMPTOM_PLACEHOLDER])
                    else:
                        user_input_features_dict[col_name] = symptom_mapping[SYMPTOM_PLACEHOLDER]


            current_user_severity_score = 0
            if symptom_severity_df is not None and weights_dict: 
                for symptom_name in selected_symptoms_raw: 
                    current_user_severity_score += weights_dict.get(symptom_name, 0)
            user_input_features_dict['severity_score'] = current_user_severity_score


            user_df_list = []
            for col_feature_name in expected_model_features_before_cluster:

                default_val = symptom_mapping.get(SYMPTOM_PLACEHOLDER, 0) if 'Symptom' in col_feature_name else 0
                user_df_list.append(user_input_features_dict.get(col_feature_name, default_val))

            user_df_pre_scale = pd.DataFrame([user_df_list], columns=expected_model_features_before_cluster)


            user_scaled_values = scaler.transform(user_df_pre_scale)

            user_cluster = 0 
            if kmeans_model is not None: 
                try:
                    user_cluster = kmeans_model.predict(user_scaled_values)[0]
                except Exception: 
                    user_cluster = 0 

            user_features_df_final = pd.DataFrame(user_scaled_values, columns=expected_model_features_before_cluster)
            user_features_df_final['cluster'] = user_cluster

            user_features_for_model = user_features_df_final[X_final.columns] 

            pred_raw_user = model.predict(user_features_for_model)
            prob_raw_user = model.predict_proba(user_features_for_model)
            probabilities = process_model_probabilities(prob_raw_user, n_classes, 1) 

            if probabilities is not None and len(probabilities) == n_classes:
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                disease_name = disease_encoder.inverse_transform([predicted_idx])[0]

                st.success("### 📋 Hasil Prediksi")
                col1_res, col2_res = st.columns([2,1]) 
                with col1_res: st.markdown(f"**Penyakit:** <span style='font-size: 1.2em; color: #28a745;'>**{str(disease_name).strip()}**</span>", unsafe_allow_html=True)
                with col2_res: st.markdown(f"**Keyakinan:** <span style='font-size: 1.2em;'>{confidence:.1%}</span>", unsafe_allow_html=True)
                st.markdown("---")

                # Display Description
                if desc_df is not None:
                    desc_row = desc_df[desc_df['Disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not desc_row.empty: st.info(f"**📖 Deskripsi:** {desc_row['Description'].values[0]}")
                    else: st.caption(f"Deskripsi tidak tersedia untuk '{str(disease_name).strip()}'.")

                # Display Precautions
                if precaution_df is not None:
                    precaution_row = precaution_df[precaution_df['Disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not precaution_row.empty:
                        precautions = [str(p).strip() for p in precaution_row.iloc[0, 1:].values if pd.notna(p) and str(p).strip().lower() not in ['nan', '']]
                        if precautions:
                            st.warning("**🛡️ Tindakan Pencegahan & Saran:**")
                            for i, p_text in enumerate(precautions, 1): st.write(f"{i}. {p_text}")
                        else: st.caption(f"Tindakan pencegahan tidak spesifik atau tidak tersedia untuk '{str(disease_name).strip()}'.")
                    else: st.caption(f"Informasi tindakan pencegahan tidak ditemukan untuk '{str(disease_name).strip()}'.")

                st.markdown("---") 

                # Top 5 Diagnoses
                if len(probabilities) > 1 and n_classes > 1:
                    st.markdown("### 📊 Top 5 Kemungkinan Diagnosis")
                    top_indices = np.argsort(probabilities)[::-1][:min(5, n_classes)] # Show up to 5 or n_classes
                    top_diseases = disease_encoder.inverse_transform(top_indices)
                    top_probs = probabilities[top_indices]

                    results_data = {'Penyakit': [str(d).strip() for d in top_diseases], 'Probabilitas': top_probs}
                    results_df = pd.DataFrame(results_data)

                    chart_data = results_df.set_index('Penyakit')
                    st.bar_chart(chart_data, height=300)
                    st.dataframe(results_df.style.format({'Probabilitas': '{:.1%}'}), use_container_width=True, hide_index=True)

                # Display user's calculated severity score and category
                if symptom_severity_df is not None and weights_dict: 
                    severity_category = categorize_severity(current_user_severity_score, severity_q1_threshold, severity_q2_threshold)
                    st.info(f"ℹ️ **Total Skor Keparahan Gejala (berdasarkan input):** {current_user_severity_score} (Kategori: **{severity_category}**).\n\nAmbang batas kategori: Rendah ≤ {severity_q1_threshold:.1f}, Sedang ≤ {severity_q2_threshold:.1f}, Tinggi > {severity_q2_threshold:.1f}. Skor ini adalah salah satu dari berbagai faktor yang dipertimbangkan model.")
                else:
                    st.info("ℹ️ Skor keparahan gejala tidak dapat dihitung (data 'Symptom-severity.csv' tidak tersedia atau kosong), sehingga kategori tidak dapat ditentukan.")

            else:
                st.error("Tidak dapat menghitung probabilitas prediksi. Model mungkin tidak mengembalikan output yang diharapkan atau jumlah kelas tidak sesuai.")

        except KeyError as e_key:
            st.error(f"Terjadi kesalahan pemetaan fitur (KeyError): '{e_key}'. Ini mungkin bug internal atau ketidaksesuaian data/kolom.")
            st.error(traceback.format_exc())
        except ValueError as e_val:
            st.error(f"Terjadi kesalahan dalam pemrosesan input atau prediksi (ValueError): {e_val}.")
            st.error(traceback.format_exc())
        except Exception as e_pred:
            st.error(f"Terjadi kesalahan umum saat memproses prediksi: {e_pred}")
            st.error(traceback.format_exc())

    else:
        st.warning("⚠️ Silakan pilih minimal satu gejala untuk melakukan prediksi.")

st.markdown("---")
st.caption("⚕️ *Aplikasi ini untuk tujuan informasi dan edukasi saja. Selalu konsultasikan dengan profesional medis atau dokter untuk diagnosis dan perawatan yang akurat.*")
