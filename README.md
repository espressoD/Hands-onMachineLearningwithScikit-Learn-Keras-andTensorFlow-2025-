# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2025)
<p align="center"><img src="https://imgur.com/T599V5B.png" width="1000"></p>
Repository ini berisi catatan, reproduksi kode, dan ringkasan teori untuk membantu memahami konsep inti Machine Learning. Materi dan contoh kode mengacu pada buku "Hands-On Machine Learning with Scikit-Learn, Keras, dan TensorFlow" (O’Reilly) sebagai referensi utama. Tujuan repository ini adalah memperdalam keterampilan praktis dalam mengimplementasikan algoritma ML melalui contoh nyata dan penjelasan terstruktur.

## Ringkasan Bab 1 — The Machine Learning Landscape

Bab 1 membahas gambaran umum Machine Learning, mulai dari definisi dasar dan contoh aplikasinya seperti pengenalan gambar, sistem rekomendasi, hingga prediksi waktu-seri. Bab ini menguraikan klasifikasi tugas ML, yaitu supervised learning (regresi dan klasifikasi), unsupervised learning (clustering dan reduksi dimensi), serta sekilas tentang reinforcement learning. Selain itu, dijelaskan pipeline proyek ML yang meliputi pengumpulan data, preprocessing, pemilihan model, pelatihan, evaluasi, dan deployment. Tantangan seperti overfitting, underfitting, pemilihan metrik evaluasi, dan pentingnya data yang representatif juga dibahas. Bab ini memperkenalkan alat populer seperti Scikit-Learn, Keras, dan TensorFlow, serta menekankan pentingnya pemetaan masalah nyata ke pendekatan ML yang tepat.

## Ringkasan Bab 2 — End-to-End Machine Learning Project

Bab 2 membahas langkah-langkah praktis dalam membangun proyek Machine Learning secara end-to-end. Dimulai dari pengambilan dan eksplorasi data, bab ini menekankan pentingnya memahami data sebelum melakukan pemodelan. Proses selanjutnya meliputi pembersihan data, penanganan nilai hilang, dan transformasi fitur agar siap digunakan oleh algoritma ML. Bab ini juga menguraikan cara membagi data menjadi set pelatihan dan pengujian, memilih serta melatih model, melakukan evaluasi dengan metrik yang sesuai, dan melakukan tuning hyperparameter untuk meningkatkan performa. Di akhir, dijelaskan proses deployment model ke lingkungan produksi. Bab ini memberikan gambaran nyata workflow ML yang sistematis dan terstruktur, serta menekankan pentingnya dokumentasi dan reproducibility dalam setiap tahapan proyek.

## Ringkasan Bab 3 — Klasifikasi

Bab 3 memperdalam topik klasifikasi, mulai dari membangun pengklasifikasi biner sederhana (misalnya detektor angka 5 di MNIST) hingga perluasan ke klasifikasi multikelas, multilabel, dan multioutput. Bab ini menekankan pentingnya evaluasi yang tepat di dataset tidak seimbang dengan metrik seperti precision, recall, F1 score, serta kurva PR dan ROC/AUC. Konsep trade-off precision–recall melalui pengaturan threshold juga dibahas, termasuk penggunaan confusion matrix untuk analisis kesalahan. Di bagian akhir, diperkenalkan teknik praktis seperti cross-validation, pemilihan strategi OvR/OvO, dan contoh penggunaan KNN untuk tugas multilabel serta pembersihan noise pada gambar.

## Ringkasan Bab 4 — Melatih Model

Bab 4 membahas cara melatih model, terutama model linear, dari pendekatan solusi tertutup seperti Persamaan Normal hingga metode iteratif seperti Gradient Descent (Batch, Stochastic, dan Mini-batch). Bab ini menunjukkan bagaimana memperluas model linear menjadi regresi polinomial untuk menangani pola non-linear, serta cara membaca kurva belajar guna mendiagnosis underfitting dan overfitting. Teknik regularisasi seperti Ridge (L2), Lasso (L1), Elastic Net, dan Early Stopping diperkenalkan untuk mengendalikan kompleksitas model. Di bagian akhir, bab ini merangkum regresi logistik untuk klasifikasi biner dan regresi Softmax untuk multikelas, menekankan pemilihan solver, regularisasi, dan evaluasi yang tepat.

## Ringkasan Bab 5 — Pohon Keputusan & Ensemble Methods

Bab 5 memperkenalkan algoritma berbasis pohon keputusan dan teknik ensemble yang sering digunakan untuk meningkatkan stabilitas dan akurasi model. Dimulai dengan konsep dasar Decision Tree (struktur, pembagian fitur, kriteria pemisahan seperti Gini/entropy), bab ini menunjukkan bagaimana pohon cenderung overfit dan bagaimana pruning atau pembatasan kedalaman dapat membantu. Selanjutnya dibahas ensemble methods: bagging (contoh: Random Forest) untuk mengurangi variansi, serta boosting (mis. AdaBoost, Gradient Boosting Machines) untuk memperbaiki bias dengan menumpuk model lemah. Bab ini juga menyentuh hyperparameter penting (jumlah estimator, kedalaman, learning rate), interpretabilitas model berbasis pohon, dan penggunaan cross-validation serta metrik yang sesuai untuk memilih konfigurasi terbaik.

## Ringkasan Bab 6 — Decision Trees

Bab 6 mengulas Decision Trees secara lebih mendalam: prinsip kerja algoritma CART (pemilihan fitur dan threshold yang meminimalkan impurity), teknik visualisasi pohon untuk interpretabilitas, serta cara membaca aturan keputusan untuk prediksi dan estimasi probabilitas. Bab ini membahas kerentanan pohon terhadap overfitting dan instabilitas (sensitif terhadap rotasi data dan perubahan kecil pada data), serta strategi regularisasi praktis seperti membatasi `max_depth`, `min_samples_split`, `min_samples_leaf`, atau `max_leaf_nodes`. Selain itu dibahas juga penggunaan Decision Tree untuk tugas regresi, contoh visualisasi batas keputusan, dan catatan kapan pohon sebaiknya digunakan sendiri atau digabungkan dalam ensemble (mis. Random Forest, boosting) untuk meningkatkan stabilitas dan performa.

## Ringkasan Bab 7 — Ensemble Learning & Random Forests

Bab 7 membahas Ensemble Learning, teknik menggabungkan prediksi dari beberapa model untuk menghasilkan akurasi lebih baik. Konsep "Wisdom of the Crowd" menunjukkan bahwa gabungan prediksi seringkali lebih akurat daripada model terbaik sendirian. Bab ini menguraikan Voting Classifiers (Hard & Soft Voting), Bagging dan Pasting (Bootstrap Aggregating untuk mengurangi variance), Random Forests (ensemble decision trees dengan randomisasi fitur), serta metode Boosting seperti AdaBoost dan Gradient Boosting yang secara iteratif memperbaiki kesalahan model sebelumnya. Dibahas pula teknik Stacking, hyperparameter penting (n_estimators, max_depth, learning rate), serta trade-off antara bias-variance. Bab ini menekankan bahwa ensemble methods sering menjadi pilihan utama untuk meningkatkan performa model di kompetisi dan aplikasi nyata.

## Ringkasan Bab 8 — Dimensionality Reduction

Bab 8 membahas teknik reduksi dimensi untuk mengatasi "Curse of Dimensionality" pada dataset dengan banyak fitur. PCA (Principal Component Analysis) diperkenalkan sebagai metode proyeksi linear yang mencari sumbu dengan varians terbesar menggunakan SVD (Singular Value Decomposition). Bab ini menjelaskan cara memilih jumlah dimensi optimal berdasarkan preserved variance (biasanya 95%), serta varian PCA untuk data besar seperti Incremental PCA dan Randomized PCA. Kernel PCA dibahas untuk menangani data non-linear dengan memanfaatkan kernel trick. Teknik manifold learning seperti LLE (Locally Linear Embedding) dan t-SNE juga diperkenalkan, dengan penekanan bahwa t-SNE sangat efektif untuk visualisasi data berdimensi tinggi ke 2D/3D tetapi tidak cocok untuk reduksi dimensi umum. Bab ini menekankan pentingnya mempertahankan informasi penting sambil menyederhanakan representasi data.

## Ringkasan Bab 9 — Unsupervised Learning

Bab 9 membahas pembelajaran tak terawasi (unsupervised learning) yang bekerja pada data tanpa label. Fokus utama adalah clustering dan Gaussian Mixtures. K-Means diperkenalkan sebagai algoritma clustering iteratif yang menemukan k centroids dan meminimalkan inersia (jarak kuadrat rata-rata ke centroid terdekat). Bab ini menjelaskan cara menentukan jumlah cluster optimal menggunakan Elbow Method (plot inersia) dan Silhouette Score (mengukur kualitas clustering). DBSCAN dibahas sebagai alternatif density-based clustering yang dapat menemukan cluster bentuk arbitrary dan mengidentifikasi outlier. Gaussian Mixture Models (GMM) diperkenalkan sebagai pendekatan probabilistik soft clustering yang mengasumsikan data berasal dari campuran distribusi Gaussian, serta algoritma Expectation-Maximization (EM) untuk pelatihan. Aplikasi praktis meliputi segmentasi pelanggan, deteksi anomali, dan generative modeling.

## Struktur singkat repository

- `bab1/TugasML_1.ipynb` — salinan/narasi terkait Bab 1 dan latihan.
- `bab2/TugasML_2.ipynb` — Teori terkait Bab 2 dan latihan.
- `bab3/TugasML3.ipynb` — teori dan praktik klasifikasi (Bab 3) menggunakan dataset MNIST.
- `bab4/TugasML4.ipynb` — teori dan praktik melatih model (Bab 4): linear/polinomial, GD, regularisasi, logistic & softmax.
- `bab5/TugasML5.ipynb` — teori dan praktik pohon keputusan & ensemble (Bab 5): decision trees, random forest, boosting.
- `bab6/TugasML6.ipynb` — teori dan praktik Decision Trees (Bab 6): CART, visualisasi, regresi pohon, regularisasi, instabilitas.
- `bab7/Tugas_ML7.ipynb` — teori dan praktik Ensemble Learning (Bab 7): voting, bagging, pasting, random forests, boosting, stacking.
- `bab8/Tugas_ML8.ipynb` — teori dan praktik Dimensionality Reduction (Bab 8): PCA, kernel PCA, LLE, t-SNE, preserved variance.
- `bab9/Tugas_ML9.ipynb` — teori dan praktik Unsupervised Learning (Bab 9): K-Means, DBSCAN, Gaussian Mixtures, clustering evaluation.

Jika ingin mulai, buka salah satu notebook di atas dan jalankan sel-selnya (pastikan dependensi di `requirements.txt` terpasang).

---

Diperbarui: 2025-12-28
