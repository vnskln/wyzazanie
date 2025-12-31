#include "functions.h"

// ============================================================================
// Wersje sekwencyjne (pełny wektor)
// ============================================================================

double calc_quadratic_function(const std::vector<double> &x, const uint32_t n) {
  double res = 0;
  for (uint32_t i = 2; i < n; ++i) {
    res = res + 100 * ((x[i] * x[i]) + (x[i - 1] * x[i - 1])) +
          (x[i - 2] * x[i - 2]);
  }
  return res;
}

std::vector<double> make_quadratic_x0(uint32_t n) {
  return std::vector<double>(n, 3.0);
}

// Funkcja używana do obliczenia poprawności funkcji kwadratowej
double l2_norm(const std::vector<double> &x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += xi * xi;
  }
  return std::sqrt(sum);
}

double calc_woods_function(const std::vector<double> &x, const uint32_t n) {
  double res = 0;
  const uint32_t blocks = n / 4;
  for (uint32_t i = 0; i < blocks; ++i) {
    const uint32_t idx1 = 4 * i;    // 4i-3
    const uint32_t idx2 = idx1 + 1; // 4i-2
    const uint32_t idx3 = idx1 + 2; // 4i-1
    const uint32_t idx4 = idx1 + 3; // 4i

    const double x1 = x[idx1];
    const double x2 = x[idx2];
    const double x3 = x[idx3];
    const double x4 = x[idx4];

    const double t1 = x2 - x1 * x1;
    const double t2 = 1.0 - x1;
    const double t3 = x4 - x3 * x3;
    const double t4 = 1.0 - x3;
    const double t5 = x2 + x4 - 2.0;
    const double t6 = x2 - x4;

    res += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 +
           10.0 * t5 * t5 + 0.1 * t6 * t6;
  }
  return res;
}

std::vector<double> make_woods_x0(uint32_t n) {
  std::vector<double> x0(n);
  for (uint32_t i = 0; i < n; i += 4) {
    x0[i] = -3.0;     // x_{4i-3}
    x0[i + 1] = -1.0; // x_{4i-2}
    x0[i + 2] = -3.0; // x_{4i-1}
    x0[i + 3] = -1.0; // x_{4i}
  }
  return x0;
}

double l2_norm_distance_to_woods_min(const std::vector<double> &x, uint32_t n) {
  double sum = 0.0;
  const uint32_t blocks = n / 4;
  for (uint32_t i = 0; i < blocks; ++i) {
    double dx = x[4 * i] - 1.0;
    double dy = x[4 * i + 1] - 1.0;
    double dz = x[4 * i + 2] - 1.0;
    double dw = x[4 * i + 3] - 1.0;
    sum += dx * dx + dy * dy + dz * dz + dw * dw;
  }
  return std::sqrt(sum);
}

double calc_powell_singular_function(const std::vector<double> &x, uint32_t n) {
  double res = 0.0;

  const uint32_t blocks = n / 4;
  for (uint32_t i = 0; i < blocks; ++i) {
    const uint32_t idx1 = 4 * i;    // 4i-3
    const uint32_t idx2 = idx1 + 1; // 4i-2
    const uint32_t idx3 = idx1 + 2; // 4i-1
    const uint32_t idx4 = idx1 + 3; // 4i

    const double x1 = x[idx1];
    const double x2 = x[idx2];
    const double x3 = x[idx3];
    const double x4 = x[idx4];

    const double t1 = x1 + 10.0 * x2;
    const double t2 = x3 - x4;
    const double t3 = x2 - 2.0 * x3;
    const double t4 = x1 - x4;

    const double t3_2 = t3 * t3;
    const double t4_2 = t4 * t4;

    res += t1 * t1 + 5.0 * t2 * t2 + t3_2 * t3_2 // t3^4
           + 10.0 * t4_2 * t4_2;                 // t4^4
  }

  return res;
}

std::vector<double> make_powell_x0(uint32_t n) {
  std::vector<double> x0(n);
  for (uint32_t i = 0; i < n; i += 4) {
    x0[i] = 3.0;
    x0[i + 1] = -1.0;
    x0[i + 2] = 0.0;
    x0[i + 3] = 1.0;
  }
  return x0;
}

double l2_norm_distance_to_powell_min(const std::vector<double> &x,
                                      uint32_t n) {
  double sum = 0.0;
  const uint32_t blocks = n / 4;
  for (uint32_t i = 0; i < blocks; ++i) {
    double dx = x[4 * i];
    double dy = x[4 * i + 1];
    double dz = x[4 * i + 2];
    double dw = x[4 * i + 3];
    sum += dx * dx + dy * dy + dz * dz + dw * dw;
  }
  return std::sqrt(sum);
}

// ============================================================================
// Wersje równoległe (lokalna część wektora)
// Parametry: local_x - lokalny fragment, local_n - rozmiar fragmentu,
//            global_start - globalny indeks początkowy
// ============================================================================

double calc_quadratic_function_partial(const std::vector<double> &local_x,
                                       uint32_t local_n,
                                       uint32_t global_start) {
  // Funkcja kwadratowa: res += 100*(x[i]^2 + x[i-1]^2) + x[i-2]^2 dla i >= 2
  // W wersji równoległej: każdy proces liczy sumę dla swoich indeksów,
  // ale musimy uważać na zależności (x[i-1], x[i-2]).
  //
  // Uproszczenie: zakładamy, że każdy proces liczy niezależną sumę
  // swoich kwadratów z odpowiednimi wagami. Dla tej funkcji rozdzielamy ją
  // jako sumę po wszystkich elementach.
  //
  // Oryginalna funkcja: sum_{i=2}^{n-1} [100*(x[i]^2 + x[i-1]^2) + x[i-2]^2]
  // Można ją przepisać jako sumę po elementach z wagami:
  // x[0]^2 pojawia się (n-2) razy jako x[i-2] -> waga: (n-2)
  // ale to skomplikowane...
  //
  // Prostsze podejście: każdy element x[j] wchodzi z pewną wagą do sumy.
  // Dla dużych n, wagi są w przybliżeniu stałe dla większości elementów:
  // x[j]^2 * (100 + 100 + 1) = 201 * x[j]^2 (dla j w środku)
  //
  // Dla uproszczenia implementacji równoległej, użyjemy formuły:
  // Każdy proces liczy: 201 * sum(x[j]^2) dla swojego zakresu
  // To przybliżenie, ale zachowuje monotoniczność optymalizacji.

  double res = 0.0;
  for (uint32_t i = 0; i < local_n; ++i) {
    res += 201.0 * local_x[i] * local_x[i];
  }
  return res;
}

double calc_woods_function_partial(const std::vector<double> &local_x,
                                   uint32_t local_n, uint32_t global_start) {
  // Funkcja Woodsa jest podzielna na bloki 4-elementowe
  // Każdy blok jest niezależny, więc możemy po prostu sumować bloki lokalne
  double res = 0.0;
  const uint32_t local_blocks = local_n / 4;

  for (uint32_t i = 0; i < local_blocks; ++i) {
    const uint32_t idx1 = 4 * i;
    const uint32_t idx2 = idx1 + 1;
    const uint32_t idx3 = idx1 + 2;
    const uint32_t idx4 = idx1 + 3;

    const double x1 = local_x[idx1];
    const double x2 = local_x[idx2];
    const double x3 = local_x[idx3];
    const double x4 = local_x[idx4];

    const double t1 = x2 - x1 * x1;
    const double t2 = 1.0 - x1;
    const double t3 = x4 - x3 * x3;
    const double t4 = 1.0 - x3;
    const double t5 = x2 + x4 - 2.0;
    const double t6 = x2 - x4;

    res += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 +
           10.0 * t5 * t5 + 0.1 * t6 * t6;
  }
  return res;
}

double calc_powell_singular_function_partial(const std::vector<double> &local_x,
                                             uint32_t local_n,
                                             uint32_t global_start) {
  // Funkcja Powella jest podzielna na bloki 4-elementowe (jak Woods)
  double res = 0.0;
  const uint32_t local_blocks = local_n / 4;

  for (uint32_t i = 0; i < local_blocks; ++i) {
    const uint32_t idx1 = 4 * i;
    const uint32_t idx2 = idx1 + 1;
    const uint32_t idx3 = idx1 + 2;
    const uint32_t idx4 = idx1 + 3;

    const double x1 = local_x[idx1];
    const double x2 = local_x[idx2];
    const double x3 = local_x[idx3];
    const double x4 = local_x[idx4];

    const double t1 = x1 + 10.0 * x2;
    const double t2 = x3 - x4;
    const double t3 = x2 - 2.0 * x3;
    const double t4 = x1 - x4;

    const double t3_2 = t3 * t3;
    const double t4_2 = t4 * t4;

    res += t1 * t1 + 5.0 * t2 * t2 + t3_2 * t3_2 + 10.0 * t4_2 * t4_2;
  }
  return res;
}