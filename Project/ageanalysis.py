# =========================================================
# IMPORT LIBRARY
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(r"C:\Users\Michael Marcel\OneDrive\Desktop\TA\Data\Dataset\student_lifestyle_100k.csv")


# =========================================================
# MEMBUAT KELOMPOK UMUR
# =========================================================

bins_age = [18, 19, 22, 25]

labels_age = [
    'Mahasiswa Baru',
    'Mahasiswa Menengah/Junior',
    'Mahasiswa Akhir'
]

df['Kelompok_Umur'] = pd.cut(
    df['Age'],
    bins=bins_age,
    labels=labels_age,
    include_lowest=True
)

# =========================================================
# MENAMPILKAN JUMLAH DATA PER KELOMPOK UMUR
# =========================================================

print("\n=== DISTRIBUSI KELOMPOK UMUR ===")

print(
    df['Kelompok_Umur']
    .value_counts()
)

# =========================================================
# MENGAMBIL DATA MAHASISWA DEPRESI
# =========================================================

df_depresi = df[
    df['Depression'] == True
]

# =========================================================
# ANALISIS RATA-RATA VARIABEL
# PADA MAHASISWA DEPRESI
# BERDASARKAN KELOMPOK UMUR
# =========================================================

analisis_age = df_depresi.groupby('Kelompok_Umur')[
    [
        'CGPA',
        'Sleep_Duration',
        'Study_Hours',
        'Social_Media_Hours',
        'Physical_Activity',
        'Stress_Level'
    ]
].mean().round(2)

print("\n=== KARAKTERISTIK MAHASISWA DEPRESI BERDASARKAN KELOMPOK UMUR ===")

print(analisis_age)

# =========================================================
# ANALISIS NILAI MINIMUM DAN MAXIMUM
# =========================================================

analisis_minmax = df_depresi.groupby('Kelompok_Umur')[
    [
        'CGPA',
        'Sleep_Duration',
        'Study_Hours',
        'Social_Media_Hours',
        'Physical_Activity',
        'Stress_Level'
    ]
].agg(['min', 'max']).round(2)

print("\n=== BATAS MINIMUM DAN MAXIMUM VARIABEL ===")

print(analisis_minmax)

# =========================================================
# VISUALISASI STRESS LEVEL
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df_depresi,
    x='Kelompok_Umur',
    y='Stress_Level'
)

plt.title(
    'Rata-rata Stress Level Mahasiswa Depresi Berdasarkan Kelompok Umur'
)

plt.xlabel('Kelompok Umur')

plt.ylabel('Stress Level')

plt.tight_layout()

plt.show()

# =========================================================
# VISUALISASI DURASI TIDUR
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df_depresi,
    x='Kelompok_Umur',
    y='Sleep_Duration'
)

plt.title(
    'Rata-rata Durasi Tidur Mahasiswa Depresi Berdasarkan Kelompok Umur'
)

plt.xlabel('Kelompok Umur')

plt.ylabel('Durasi Tidur')

plt.tight_layout()

plt.show()

# =========================================================
# VISUALISASI STUDY HOURS
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df_depresi,
    x='Kelompok_Umur',
    y='Study_Hours'
)

plt.title(
    'Rata-rata Study Hours Mahasiswa Depresi Berdasarkan Kelompok Umur'
)

plt.xlabel('Kelompok Umur')

plt.ylabel('Study Hours')

plt.tight_layout()

plt.show()

# =========================================================
# VISUALISASI SOCIAL MEDIA HOURS
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df_depresi,
    x='Kelompok_Umur',
    y='Social_Media_Hours'
)

plt.title(
    'Rata-rata Social Media Hours Mahasiswa Depresi Berdasarkan Kelompok Umur'
)

plt.xlabel('Kelompok Umur')

plt.ylabel('Social Media Hours')

plt.tight_layout()

plt.show()

# =========================================================
# VISUALISASI PHYSICAL ACTIVITY
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df_depresi,
    x='Kelompok_Umur',
    y='Physical_Activity'
)

plt.title(
    'Rata-rata Physical Activity Mahasiswa Depresi Berdasarkan Kelompok Umur'
)

plt.xlabel('Kelompok Umur')

plt.ylabel('Physical Activity')

plt.tight_layout()

plt.show()

# =========================================================
# MENYIMPAN HASIL ANALISIS
# =========================================================

analisis_age.to_csv(
    'analisis_karakteristik_depresi_umur.csv'
)

analisis_minmax.to_csv(
    'batas_variabel_depresi_umur.csv'
)

print("\nFile analisis berhasil disimpan.")