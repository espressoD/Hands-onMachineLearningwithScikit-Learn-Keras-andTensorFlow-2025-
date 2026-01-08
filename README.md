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

## Ringkasan Bab 10 — Introduction to Artificial Neural Networks with Keras

Bab 10 memperkenalkan fondasi Artificial Neural Networks (ANN) dan implementasinya menggunakan Keras API. Dimulai dari sejarah perceptron hingga multi-layer perceptron (MLP), bab ini menjelaskan cara kerja forward propagation dan backpropagation untuk training. Konsep fungsi aktivasi seperti ReLU, sigmoid, dan softmax dibahas, serta arsitektur jaringan untuk regresi dan klasifikasi. Bab ini mencakup implementasi praktis menggunakan Sequential API dan Functional API Keras, teknik regularisasi (dropout, batch normalization), optimizer (SGD, Adam, RMSprop), dan callbacks untuk monitoring training. Dibahas pula hyperparameter tuning, learning rate scheduling, dan best practices untuk building dan training neural networks. Notebook mencakup implementasi lengkap dari Fashion MNIST classification dan California Housing regression dengan berbagai teknik optimasi.

## Ringkasan Bab 11 — Training Deep Neural Networks

Bab 11 membahas tantangan training deep neural networks dan solusinya. Masalah vanishing/exploding gradients dijelaskan beserta solusinya menggunakan weight initialization (Xavier, He), batch normalization, dan gradient clipping. Teknik regularisasi advanced seperti dropout, L1/L2 regularization, dan early stopping dibahas untuk mencegah overfitting. Bab ini memperkenalkan berbagai optimizer modern (Momentum, AdaGrad, RMSprop, Adam, Nadam) dan learning rate scheduling (exponential decay, piecewise constant, 1cycle policy). Transfer learning dan fine-tuning model pretrained juga dibahas sebagai cara efisien melatih model dengan data terbatas. Implementasi praktis mencakup custom callbacks, model checkpointing, dan TensorBoard untuk visualisasi training. Best practices untuk debugging dan monitoring training process juga tercakup lengkap.

## Ringkasan Bab 12 — Custom Models and Training with TensorFlow

Bab 12 mendalami TensorFlow API tingkat rendah untuk membangun model dan training loop custom. Dimulai dari operasi tensor dasar, variables, dan automatic differentiation dengan GradientTape, bab ini menunjukkan cara membuat custom layers, models, losses, dan metrics dari scratch. Konsep subclassing API dijelaskan untuk building complex architectures yang tidak bisa dibuat dengan Sequential/Functional API. Custom training loops dengan `@tf.function` decorator untuk performance optimization juga dibahas. Bab ini mencakup implementasi custom activation functions, initializers, regularizers, dan constraints. TensorFlow datasets dan data augmentation pipeline untuk performance optimization juga dijelaskan. Notebook berisi contoh lengkap custom ResNet implementation, custom loss functions untuk specialized tasks, dan integration dengan TensorBoard untuk monitoring.

## Ringkasan Bab 13 — Loading and Preprocessing Data with TensorFlow

Bab 13 fokus pada data pipeline yang efisien menggunakan tf.data API untuk handling large datasets. Dijelaskan cara membuat dataset dari berbagai sumber (arrays, files, generators, TFRecord), transformasi data (map, filter, batch, shuffle, prefetch), dan optimasi performance dengan parallel processing dan caching. Bab ini mencakup TFRecord format untuk efficient storage dan loading, feature engineering dengan preprocessing layers (normalization, discretization, hashing), dan data augmentation untuk computer vision tasks. Teknik handling imbalanced datasets, stratified sampling, dan windowing untuk time series juga dibahas. Implementasi praktis meliputi building efficient input pipelines untuk training, creating custom preprocessing layers, dan best practices untuk production-ready data loading dengan performance profiling menggunakan tf.data.experimental.

## Ringkasan Bab 14 — Deep Computer Vision Using Convolutional Neural Networks

Bab 14 membahas Convolutional Neural Networks (CNN) untuk computer vision tasks. Dimulai dari konsep convolution operation, pooling layers, dan receptive fields, bab ini menjelaskan bagaimana CNN dapat secara otomatis belajar hierarchical features dari raw pixels. Arsitektur CNN klasik seperti LeNet-5, AlexNet, VGG, ResNet, dan Inception dibahas beserta inovasi masing-masing. Implementasi praktis mencakup image classification, object detection basics, dan semantic segmentation. Teknik data augmentation (rotation, flipping, cropping, color jittering) dijelaskan untuk meningkatkan generalisasi. Transfer learning dengan pretrained models (VGG16, ResNet50, EfficientNet) dari Keras Applications dibahas sebagai cara efisien building state-of-the-art models. Notebook berisi implementasi lengkap CNN dari scratch, fine-tuning pretrained models, dan visualisasi feature maps untuk interpretabilitas.

## Ringkasan Bab 15 — Processing Sequences Using RNNs and CNNs

Bab 15 membahas arsitektur neural networks untuk data sequential seperti time series dan text. Recurrent Neural Networks (RNN) diperkenalkan dengan varian SimpleRNN, LSTM (Long Short-Term Memory), dan GRU (Gated Recurrent Unit) untuk mengatasi masalah vanishing gradients dan long-term dependencies. Deep RNNs dengan stacking layers dan layer normalization untuk sequence modeling yang lebih kompleks dijelaskan lengkap. Bab ini juga membahas CNN untuk sequences dengan 1D convolutions dan WaveNet architecture menggunakan dilated convolutions untuk large receptive fields. Bidirectional RNNs untuk sequence classification, encoder-decoder untuk sequence-to-sequence tasks, dan stateful RNNs untuk very long sequences dibahas dengan implementasi praktis. Teknik regularisasi khusus RNN (dropout, recurrent dropout, gradient clipping) dan best practices untuk training stability juga tercakup. Notebook berisi complete implementation time series forecasting dengan comparison 7 model berbeda, visualisasi predictions, dan performance benchmarking.

## Ringkasan Bab 16 — Natural Language Processing with RNNs and Attention

Bab 16 membahas evolusi NLP dari RNN klasik hingga Transformer modern. Character-level RNN (Char-RNN) diperkenalkan untuk text generation dengan Shakespeare dataset. Word embeddings dijelaskan sebagai dense vector representations untuk semantic similarity, diimplementasikan dalam sentiment analysis task menggunakan IMDB dataset dengan Bidirectional LSTM. Attention mechanism (Bahdanau/Luong attention) dibahas sebagai solusi bottleneck problem dalam encoder-decoder architecture, dengan custom implementation dari scratch. Transformer architecture diperkenalkan sebagai revolution dalam NLP: positional encoding untuk sequence order information, multi-head self-attention untuk capturing different aspects of relationships, dan complete encoder-decoder blocks dijelaskan dengan visualisasi. Integration dengan Hugging Face Transformers library untuk using pretrained models (BERT, GPT-2, DistilBERT) dibahas lengkap dengan examples sentiment analysis dan text generation. Transfer learning dalam NLP dan best practices untuk fine-tuning large language models juga tercakup.

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
- `bab10/Tugas_ML10.ipynb` — teori dan praktik Artificial Neural Networks (Bab 10): MLP, Sequential/Functional API, regularisasi, optimizer, callbacks.
- `bab11/Tugas_ML11.ipynb` — teori dan praktik Training Deep Neural Networks (Bab 11): vanishing gradients, batch normalization, advanced optimizers, transfer learning.
- `bab12/Tugas_ML12.ipynb` — teori dan praktik Custom Models (Bab 12): TensorFlow low-level API, custom layers/losses/metrics, @tf.function, GradientTape.
- `bab13/Tugas_ML13.ipynb` — teori dan praktik Data Loading (Bab 13): tf.data API, TFRecord, preprocessing layers, efficient pipelines, data augmentation.
- `bab14/Tugas_ML14.ipynb` — teori dan praktik Computer Vision (Bab 14): CNN architectures, convolution/pooling, ResNet/VGG/Inception, transfer learning, feature visualization.
- `bab15/Tugas_ML15.ipynb` — teori dan praktik Processing Sequences (Bab 15): SimpleRNN/LSTM/GRU, 1D CNN, WaveNet, Bidirectional RNN, Seq2Seq, Stateful RNN, regularisasi RNN.
- `bab16/Tugas_ML16.ipynb` — teori dan praktik NLP (Bab 16): Char-RNN, word embeddings, sentiment analysis, attention mechanism, Transformer architecture, Hugging Face integration.


Jika ingin mulai, buka salah satu notebook di atas dan jalankan sel-selnya (pastikan dependensi di `requirements.txt` terpasang).

---

**Diperbarui**: 2026-01-08  