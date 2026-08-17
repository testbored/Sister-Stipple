# Sister Stipple

Program C++17 untuk menghasilkan gambar stippling dari citra masukan menggunakan **Weighted Lloyd's Algorithm**. Proyek menyediakan tiga backend komputasi:

- Serial CPU
- Paralel CPU dengan OpenMP
- GPU dengan CUDA

Selain output PNG, program dapat membuat GIF progres iterasi dan menyediakan GUI berbasis OpenCV.

## Jurnal Progress
Hal pertama yang saya lakukan adalah mencari tahu apa itu . Selama proses pencarian tersebut, saya menemukan paper berikut Weighted Voronoi Stippling, yang menjelaskan penggunaan weighted centroidal Voronoi diagram untuk melakukan stippling. Saya mencari dari artikel hingga video youtube sampai menemukan penjelasan yang tepat. 

https://dahtah.github.io/imager/stippling.html
https://mfkasim1.github.io/2016/12/06/stippling-pictures-with-lloyds-algorithm/

Kedua sumber diatas memberikan saya pemahaman paling besar diantara ketiganya

Ide dasar weighhed centered voronoi/lloyd algorithm adalah dengan memperlakukan setiap titik stipple sebagai site Voronoi. Setiap pixel akan diberikan kepada titik terdekat, setiap pixel pada titik terdekat akan dihitung pusat dari seluruh titik dan diambil centroid dari seluruh pixel pada titik tersebut. Titik akan dipindahkan lokasinya ke centroid dan mengulang hal tersebut hingga mencapai konvergens atau jumlah iterasi berakhir

Program mengubah citra grayscale menjadi *density map*. Bobot setiap pixel dihitung dari tingkat kegelapan, lalu diperkaya dengan informasi tepi (Sobel):

```text
darkness(x, y) = (255 - gray(x, y)) / 255
density(x, y) = 255 × (darkness(x, y)^gamma + edgeWeight × edge(x, y))
```

`gamma` mengontrol penekanan area gelap dan `edgeWeight` mempertahankan detail tepi objek.

## Implementasi Lloyd's Algorithm

Setiap titik adalah centroid dari sebuah wilayah Voronoi. Satu iterasi terdiri dari dua tahap.

1. Untuk setiap pixel, tentukan titik terdekat.
2. Hitung ulang posisi titik sebagai centroid berbobot density dari semua pixel pada wilayahnya.

Rumus centroid berbobot untuk titik ke-`i`:

```text
newX[i] = Σ(x × density(x, y)) / Σ(density(x, y))
newY[i] = Σ(y × density(x, y)) / Σ(density(x, y))
```

Implementasi serial berada pada `calculateNewCentroid()`.

```cpp
for (int y = 0; y < density.rows; ++y) {
    for (int x = 0; x < density.cols; ++x) {
        const int point = findNearestPoint(x, y);
        const float weight = density.at<float>(y, x);
        sumX[point] += x * weight;
        sumY[point] += y * weight;
        sumWeight[point] += weight;
    }
}

for (int i = 0; i < pointCount; ++i) {
    if (sumWeight[i] > 0.0) {
        pointX[i] = static_cast<float>(sumX[i] / sumWeight[i]);
        pointY[i] = static_cast<float>(sumY[i] / sumWeight[i]);
    }
}
```

Proses berhenti saat jumlah iterasi maksimum tercapai atau pergeseran titik terbesar sudah lebih kecil/sama dengan `epsilon`.

```cpp
const float shift = calculateNewCentroid();
if (shift <= epsilon) {
    statistics.converged = true;
    break;
}
```

### Inisialisasi Berbobot

Titik awal tidak dipilih uniform. `std::discrete_distribution` memilih pixel berdasarkan density sehingga titik awal sudah lebih banyak pada area penting.

```cpp
std::discrete_distribution<int> weightedPixel(weights.begin(), weights.end());
const int pixelIndex = weightedPixel(rng);
const int x = pixelIndex % density.cols;
const int y = pixelIndex / density.cols;
```

Untuk benchmark yang adil, semua backend memakai `seed` yang sama.

## Paralelisasi CPU dengan OpenMP

Loop pixel merupakan bagian paling mahal dengan kompleksitas kira-kira:

```text
O(jumlahIterasi × lebarGambar × tinggiGambar × jumlahTitik)
```

Menulis langsung ke `sumX[point]`, `sumY[point]`, dan `sumWeight[point]` dari banyak thread akan menyebabkan *race condition*. Karena itu, setiap thread memakai buffer akumulasi privat. Indeks buffer memuat kombinasi `threadId` dan nomor titik.

```cpp
const int threadId = omp_get_thread_num();
const std::size_t offset = static_cast<std::size_t>(threadId) * pointCount;

#pragma omp for schedule(static)
for (int y = 0; y < density.rows; ++y) {
    for (int x = 0; x < density.cols; ++x) {
        const int point = findNearestPoint(x, y);
        const std::size_t index = offset + point;
        const float weight = density.at<float>(y, x);

        partialSumX[index] += x * weight;
        partialSumY[index] += y * weight;
        partialSumWeight[index] += weight;
    }
}
```

Setelah semua thread selesai, setiap titik direduksi dengan menjumlahkan buffer seluruh thread. Tidak ada dua thread yang menulis indeks titik yang sama pada tahap pembaruan centroid.

```cpp
#pragma omp parallel for reduction(max:maximumShift) schedule(static)
for (int i = 0; i < pointCount; ++i) {
    for (int thread = 0; thread < threadCount; ++thread) {
        const std::size_t index = static_cast<std::size_t>(thread) * pointCount + i;
        sumX += partialSumX[index];
        sumY += partialSumY[index];
        sumWeight += partialSumWeight[index];
    }
}
```

Atur jumlah thread saat menjalankan program:

```bash
OMP_NUM_THREADS=8 ./build/sister-stipple --backend omp --input gambar.png --points 5000 --iterations 100 --epsilon 0.05 --output hasil.png
```

## Akselerasi GPU dengan CUDA

Backend CUDA menyimpan density map dan koordinat titik di GPU selama seluruh iterasi. Pada setiap iterasi GPU:

1. Mereset buffer `sumX`, `sumY`, dan `sumWeight`.
2. Menjalankan satu thread untuk setiap pixel untuk mencari titik terdekat.
3. Mengakumulasi centroid dengan `atomicAdd` karena banyak pixel dapat mengubah region yang sama.
4. Menjalankan kernel kedua untuk memperbarui koordinat centroid.

Kernel assignment pixel:

```cpp
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
const float weight = density[y * width + x];

int nearest = 0;
for (int point = 1; point < pointCount; ++point) {
    const float dx = pointX[point] - x;
    const float dy = pointY[point] - y;
    if (dx * dx + dy * dy < nearestDistance) {
        nearest = point;
    }
}

atomicAdd(&sumX[nearest], static_cast<float>(x) * weight);
atomicAdd(&sumY[nearest], static_cast<float>(y) * weight);
atomicAdd(&sumWeight[nearest], weight);
```

Kernel pembaruan centroid dan pergeseran maksimum:

```cpp
const float newX = sumX[point] / sumWeight[point];
const float newY = sumY[point] / sumWeight[point];
const float dx = newX - pointX[point];
const float dy = newY - pointY[point];

atomicMax(maximumShiftSquaredBits, __float_as_uint(dx * dx + dy * dy));
pointX[point] = newX;
pointY[point] = newY;
```

`atomicMax` tersebut dipakai untuk memeriksa kondisi `epsilon` tanpa perlu memindahkan semua titik dari GPU ke CPU pada setiap iterasi.

## Build

Kebutuhan utama:

- CMake 3.24+
- Compiler C++17
- OpenCV 4
- OpenMP
- CUDA Toolkit untuk backend CUDA
- ImageMagick `convert` bila ingin menghasilkan GIF

Build CUDA dengan arsitektur GPU yang terdeteksi otomatis:

```bash
cmake -S . -B build \
  -DSISTER_STIPPLE_ENABLE_CUDA=ON \
  -DSISTER_STIPPLE_CUDA_ARCHITECTURES=native

cmake --build build --parallel
```

Jika mengetahui compute capability GPU, misalnya `86`, gunakan:

```bash
cmake -S . -B build -DSISTER_STIPPLE_CUDA_ARCHITECTURES=86
```

## Menjalankan Program

Benchmark ketiga backend:

```bash
./build/sister-stipple --backend all --input assets/gambar.png --points 5000 --iterations 150 --epsilon 0.05 --gamma 1.3 --edge-weight 0.5 --render-scale 3 --output output/hasil.png
```

Menjalankan CUDA saja:

```bash
./build/sister-stipple --backend cuda --input assets/gambar.png --points 5000 --iterations 150 --epsilon 0.05 --output output/hasil-cuda.png
```

Membuat GIF progres:

```bash
./build/sister-stipple --backend omp --input assets/gambar.png --points 5000 --iterations 150 --epsilon 0.05 --output output/hasil.png --gif output/proses.gif
```

Membuka GUI:

```bash
./build/sister-stipple --gui
```

## Parameter Kualitas

| Parameter | Default | Dampak |
|---|---:|---|
| `--gamma` | `1.6` | Nilai lebih tinggi memusatkan titik pada area gelap. Rekomendasi awal: `1.5`–`2.2`. |
| `--edge-weight` | `0.25` | Menjaga tepi objek. Gunakan `0.2`–`0.4` untuk detail lebih tegas. |
| `--render-scale` | `3` | Supersampling. Nilai lebih tinggi menghasilkan titik lebih halus, tetapi render lebih lambat. |
| `--points` | wajib | Lebih banyak titik meningkatkan detail dan biaya komputasi. |
| `--epsilon` | wajib | Nilai kecil membuat konvergensi lebih ketat. |

## Tabel Benchmark

Isi tabel berikut setelah menjalankan benchmark pada input, parameter, dan seed yang sama.

**Konfigurasi pengujian**

| Item | Nilai |
|---|---|
| Jumlah titik | 1000 |
| Maksimum iterasi |10|
| Epsilon | 0.5 |
| Gamma | 1.5 |
| Edge weight | 0.5 |
| Seed | 42 |
**Hasil benchmark**

| Backend | Waktu eksekusi (ms) | Iterasi aktual | Speedup terhadap serial | Konvergen | Catatan |
|---|---:|---:|---:|---|---|
| Serial | 174251.222 | 10  | 1.00x | | tidak |
| OpenMP | 17333.760 |  10  | 1.00x | | tidak |
| CUDA | 509.646 | 10  | 1.00x | | tidak |

Speedup dihitung dengan rumus:

```text
speedup = waktu serial / waktu backend
```

## Kendala yang dialami

1. Mereset buffer `sumX`, `sumY`, dan `sumWeight`.
2. Menjalankan satu thread untuk setiap pixel untuk mencari titik terdekat.
3. Mengakumulasi centroid dengan `atomicAdd` karena banyak pixel dapat mengubah region yang sama.
4. Menjalankan kernel kedua untuk memperbarui koordinat centroid.
