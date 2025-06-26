# 第4章 敵対的生成ネットワーク
**敵対的生成ネットワーク (Generative Adversarial Network: GAN)** は、2014年に Ian Goodfellow らによって提案されたモデル[1]である。
このモデルの核となるアイデアは、現在の生成モデリングにも引き継がれている。  
[1] Ian J. Goodfellow et al., "Generative Adversarial Nets," June 10, 2014, https://arxiv.org/abs/1406.2661

## 4.1 イントロダクション
GAN は **生成器 (generator)** と **識別器 (discriminator)** から構成される。

- 生成器：ランダムの椅子をもとのデータセットからサンプリングした観測に変換しようとする
- 識別機：観測がもとのデータセットから得られたものか、生成器が生成したものかを予測する

<div align="center">
    <img src="../../images/gan_input_output.png" width=500>
</div>

学習では、生成器はランダムノイズから画像を生成し、識別機はランダムな予測を行う。
生成器がよりうまく識別機を騙し、識別機が正しく判定する能力をより高めるには、この2種類のネットワークの学習を、どう行えばよいかが重要となる。

## 4.2 深層畳み込み GAN (DCGAN)
まずは、2015年に発表された Alec Radford らの論文[2] (DCGAN: Deep Convolutional GAN) の流れを厳密にたどる。  
[2] Alec Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," January 7, 2016, https://arxiv.org/abs/1511.06434

### 4.2.1 Bricks データセット
学習用のデータセットとして、4万枚の LEGO 画像データセット [Bricks](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images) を利用する。

<div align="center">
    <img src="../../images/bricks_dataset.png" width=1000>
</div>

オリジナルの画像サイズは 400x400 や 200x200 ピクセルだが、64x64 ピクセルにリサイズして利用する。
また、ピクセル値の値域は [0, 255] であるが、[-1, 1] の範囲にスケーリングし、最終層の活性化関数に tanh 関数を使えるようにする。

### 4.2.2 識別器
識別器の目的は、画像が本物か偽物かを判定する教師あり画像分類問題である。
そのため、次のようなアーキテクチャのネットワークを実装する。

| 層の種類 | 出力形状 | パラメータ数 |
| :-- | :-- | :-- |
| InputLayer | (None, 64, 64, 1) | 0 |
| 4x4 Conv2D | (None, 32, 32, 64) | 1,024 |
| LeakyReLU | (None, 32, 32, 64) | 0 |
| Dropout | (None, 32, 32, 64) | 0 |
| 4x4 Conv2D | (None, 16, 16, 128) | 131,072 |
| BatchNormalization | (None, 16, 16, 128) | 512 |
| LeakyReLU | (None, 16, 16, 128) | 0 |
| Dropout | (None, 16, 16, 128) | 0 |
| 4x4 Conv2D | (None, 8, 8, 256) | 524,288 |
| BatchNormalization | (None, 8, 8, 256) | 1,024 |
| LeakyReLU | (None, 8, 8, 256) | 0 |
| Dropout | (None, 8, 8, 256) | 0 |
| 4x4 Conv2D | (None, 4, 4, 512) | 2,097,152 |
| BatchNormalization | (None, 4, 4, 512) | 2048 |
| LeakyReLU | (None, 4, 4, 512) | 0 |
| Dropout | (None, 4, 4, 512) | 0 |
| 4x4 Conv2D | (None, 1, 1, 1) | 8,192 |
| Flatten | (None, 1) | 0 |

最終層の畳み込み層では、活性化関数に Sigmoid 関数を用いて [0, 1] の値を出力する。

### 生成器
生成器には、多変量正規分布から生成したベクトルを入力として、変分オートエンコーダのデコーダと似たアーキテクチャのネットワークを用いる。
このようなアーキテクチャを用いることで、潜在空間のベクトルを操作するだけで、元の領域内の画像が持つ高いレベルの特徴を変更できる。

| 層の種類 | 出力形状 | パラメータ数 |
| :-- | :-- | :-- |
| InputLayer | (None, 100) | 0 |
| Reshape | (None, 1, 1, 100) | 0 |
| 4x4 Conv2DTranspose | (None, 4, 4, 512) | 819,200 |
| BatchNormalization | (None, 4, 4, 512) | 2048 |
| LeakyReLU | (None, 4, 4, 512) | 0 |
| 4x4 Conv2DTranspose | (None, 8, 8, 256) | 2,097,152 |
| BatchNormalization | (None, 8, 8, 256) | 1024 |
| LeakyReLU | (None, 8, 8, 256) | 0 |
| 4x4 Conv2DTranspose | (None, 16, 16, 128) | 524,288 |
| BatchNormalization | (None, 16, 16, 128) | 512 |
| LeakyReLU | (None, 16, 16, 128) | 0 |
| 4x4 Conv2DTranspose | (None, 32, 32, 64) | 131,072 |
| BatchNormalization | (None, 32, 32, 64) | 256 |
| LeakyReLU | (None, 32, 32, 64) | 0 |
| 4x4 Conv2DTranpose | (None, 64, 64, 1) | 1,024 |

転置畳み込み層の代わりに、アップサンプリング層 (UpSampling2D) とストライドが1の畳み込み層を用いても良い。

### DCGAN を訓練する