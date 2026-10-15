import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Для воспроизводимости результатов
np.random.seed(42)

# Настройки графиков
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def print_stats(name, sample):
    sample = np.asarray(sample)
    print(f"\n=== {name} ===")
    print(f"Объём выборки:              {len(sample)}")
    print(f"Выборочное среднее:         {np.mean(sample):.4f}")
    print(f"Выборочная дисперсия:       {np.var(sample):.4f}")
    print(f"Выборочное СКО:             {np.std(sample):.4f}")
    print(f"Минимальное значение:       {np.min(sample):.4f}")
    print(f"Максимальное значение:      {np.max(sample):.4f}")


def analyze_discrete(name, sample, dist, xlim=None):
    """Анализ дискретной случайной величины."""
    sample = np.asarray(sample)
    print_stats(name, sample)

    if xlim is None:
        xmin, xmax = int(np.min(sample)), int(np.max(sample))
    else:
        xmin, xmax = xlim

    x = np.arange(xmin, xmax + 1)
    pmf = dist.pmf(x)
    cdf = dist.cdf(x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Теоретические вероятности
    axes[0, 0].stem(x, pmf, basefmt=" ")
    axes[0, 0].set_title(f"{name}: теоретические вероятности")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("P(X=k)")

    # Теоретическая функция распределения
    axes[0, 1].step(x, cdf, where="post")
    axes[0, 1].set_title(f"{name}: теоретическая CDF")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("F(k)")

    # График выборки
    axes[1, 0].plot(sample, ".", markersize=3)
    axes[1, 0].set_title("Выборка")
    axes[1, 0].set_xlabel("Номер элемента")
    axes[1, 0].set_ylabel("Значение")

    # Гистограмма и теоретические вероятности
    edges = np.arange(xmin - 0.5, xmax + 1.5, 1)
    axes[1, 1].hist(sample, bins=edges, density=True, alpha=0.5,
                    label="Гистограмма")
    axes[1, 1].plot(x, pmf, "ro-", label="Теоретические вероятности")
    axes[1, 1].set_title(f"{name}: гистограмма и теоретические вероятности")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Эмпирическая и теоретическая CDF
    x_sorted = np.sort(sample)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    plt.step(x_sorted, y, where="post", label="Эмпирическая CDF")
    plt.step(x, cdf, where="post", color="r", label="Теоретическая CDF")
    plt.title(f"{name}: эмпирическая и теоретическая CDF")
    plt.xlabel("k")
    plt.ylabel("F(k)")
    plt.legend()
    plt.show()


def analyze_continuous(name, sample, dist, xlim=None, bins=30):
    """Анализ непрерывной случайной величины."""
    sample = np.asarray(sample)
    print_stats(name, sample)

    if xlim is None:
        xmin, xmax = float(np.min(sample)), float(np.max(sample))
        margin = 0.1 * (xmax - xmin) if xmax > xmin else 1.0
        xmin -= margin
        xmax += margin
    else:
        xmin, xmax = xlim

    x = np.linspace(xmin, xmax, 500)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Теоретическая плотность
    axes[0, 0].plot(x, pdf)
    axes[0, 0].set_title(f"{name}: теоретическая плотность")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("f(x)")

    # Теоретическая CDF
    axes[0, 1].plot(x, cdf)
    axes[0, 1].set_title(f"{name}: теоретическая CDF")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("F(x)")

    # График выборки
    axes[1, 0].plot(sample, ".", markersize=3)
    axes[1, 0].set_title("Выборка")
    axes[1, 0].set_xlabel("Номер элемента")
    axes[1, 0].set_ylabel("Значение")

    # Гистограмма и теоретическая плотность
    axes[1, 1].hist(sample, bins=bins, density=True, alpha=0.5,
                    label="Гистограмма")
    axes[1, 1].plot(x, pdf, "r-", label="Теоретическая плотность")
    axes[1, 1].set_title(f"{name}: гистограмма и плотность")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Эмпирическая и теоретическая CDF
    x_sorted = np.sort(sample)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    plt.step(x_sorted, y, where="post", label="Эмпирическая CDF")
    plt.plot(x, cdf, "r-", label="Теоретическая CDF")
    plt.title(f"{name}: эмпирическая и теоретическая CDF")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.show()


def study_sample_size_continuous(name, sampler, dist,
                                 sizes=(100, 1000, 10000),
                                 bins=30, xlim=None):
    """Исследование зависимости гистограммы от объёма выборки."""
    plt.figure(figsize=(10, 5))

    for N in sizes:
        s = sampler(N)
        plt.hist(s, bins=bins, density=True, alpha=0.4, label=f"N={N}")

    if xlim is None:
        s_all = sampler(max(sizes))
        xmin, xmax = np.min(s_all), np.max(s_all)
    else:
        xmin, xmax = xlim

    x = np.linspace(xmin, xmax, 500)
    plt.plot(x, dist.pdf(x), "k-", lw=2, label="Теоретическая плотность")
    plt.title(f"{name}: зависимость гистограммы от объёма выборки")
    plt.legend()
    plt.show()


def study_bins_continuous(name, sample, dist,
                          bins_list=(5, 15, 30, 50), xlim=None):
    """Исследование зависимости гистограммы от числа интервалов."""
    plt.figure(figsize=(10, 5))

    for bins in bins_list:
        plt.hist(sample, bins=bins, density=True, alpha=0.4,
                 label=f"bins={bins}")

    if xlim is None:
        xmin, xmax = np.min(sample), np.max(sample)
    else:
        xmin, xmax = xlim

    x = np.linspace(xmin, xmax, 500)
    plt.plot(x, dist.pdf(x), "k-", lw=2, label="Теоретическая плотность")
    plt.title(f"{name}: зависимость гистограммы от числа интервалов")
    plt.legend()
    plt.show()


def study_sample_size_discrete(name, sampler, dist,
                               sizes=(100, 1000, 10000), xlim=(0, 20)):
    """Исследование зависимости гистограммы от объёма выборки для дискретной СВ."""
    plt.figure(figsize=(10, 5))

    xmin, xmax = xlim
    edges = np.arange(xmin - 0.5, xmax + 1.5, 1)

    for N in sizes:
        s = sampler(N)
        plt.hist(s, bins=edges, density=True, alpha=0.4, label=f"N={N}")

    x = np.arange(xmin, xmax + 1)
    plt.plot(x, dist.pmf(x), "ro-", label="Теоретические вероятности")
    plt.title(f"{name}: зависимость гистограммы от объёма выборки")
    plt.legend()
    plt.show()


# ============================================================
# ОСНОВНАЯ ЧАСТЬ
# ============================================================

def main():
    N = 1000

    # --------------------------------------------------------
    # ДИСКРЕТНЫЕ РАСПРЕДЕЛЕНИЯ
    # --------------------------------------------------------

    # ---------- 1. Геометрическое распределение ----------
    p = 0.3
    geom_dist = stats.geom(p)
    sample_geom = geom_dist.rvs(size=N, random_state=42)
    analyze_discrete("Геометрическое p=0.3", sample_geom, geom_dist, xlim=(1, 20))
    study_sample_size_discrete(
        "Геометрическое p=0.3",
        lambda n: geom_dist.rvs(size=n, random_state=42),
        geom_dist,
        xlim=(1, 20)
    )

    # ---------- 2. Биномиальное распределение ----------
    n_binom = 10
    p_binom = 0.4
    binom_dist = stats.binom(n_binom, p_binom)
    sample_binom = binom_dist.rvs(size=N, random_state=42)
    analyze_discrete("Биномиальное n=10, p=0.4", sample_binom, binom_dist,
                     xlim=(0, 10))

    # ---------- 3. Распределение Пуассона ----------
    mu_pois = 3.0
    pois_dist = stats.poisson(mu_pois)
    sample_pois = pois_dist.rvs(size=N, random_state=42)
    analyze_discrete("Пуассон mu=3", sample_pois, pois_dist, xlim=(0, 15))
    study_sample_size_discrete(
        "Пуассон mu=3",
        lambda n: pois_dist.rvs(size=n, random_state=42),
        pois_dist,
        xlim=(0, 15)
    )

    # --------------------------------------------------------
    # НЕПРЕРЫВНЫЕ РАСПРЕДЕЛЕНИЯ
    # --------------------------------------------------------

    # ---------- 4. Равномерное распределение через метод обратной функции ----------
    a, b = 2, 8
    u = np.random.rand(N)
    sample_uniform = a + (b - a) * u
    uniform_dist = stats.uniform(loc=a, scale=b - a)

    analyze_continuous("Равномерное [2, 8]", sample_uniform, uniform_dist,
                       xlim=(a, b))
    study_sample_size_continuous(
        "Равномерное",
        lambda n: a + (b - a) * np.random.rand(n),
        uniform_dist,
        xlim=(a, b)
    )
    study_bins_continuous("Равномерное", sample_uniform, uniform_dist,
                          xlim=(a, b))

    # ---------- 5. Экспоненциальное распределение через метод обратной функции ----------
    lam = 2.0
    u = np.random.rand(N)
    sample_exp = -np.log(1 - u) / lam
    exp_dist = stats.expon(scale=1 / lam)

    analyze_continuous("Экспоненциальное lambda=2", sample_exp, exp_dist,
                       xlim=(0, np.percentile(sample_exp, 99)))
    study_sample_size_continuous(
        "Экспоненциальное",
        lambda n: -np.log(1 - np.random.rand(n)) / lam,
        exp_dist,
        xlim=(0, np.percentile(sample_exp, 99))
    )

    # ---------- 6. Нормальное распределение через метод суммирования ----------
    mu, sigma = 0, 1

    def normal_by_sum(n, mu=0, sigma=1, k=12):
        """
        Метод суммирования: сумма 12 равномерных U(0,1)
        даёт приближённо N(0,1).
        """
        u = np.random.rand(n, k)
        z = np.sum(u, axis=1) - k / 2  # для k=12: mean=0, var=1
        return mu + sigma * z

    sample_norm = normal_by_sum(N, mu, sigma)
    norm_dist = stats.norm(mu, sigma)

    analyze_continuous("Нормальное mu=0, sigma=1", sample_norm, norm_dist,
                       xlim=(-4, 4))
    study_sample_size_continuous(
        "Нормальное",
        lambda n: normal_by_sum(n, mu, sigma),
        norm_dist,
        xlim=(-4, 4)
    )

    # ---------- 7. Гамма-распределение ----------
    shape, scale = 2.0, 2.0
    sample_gamma = np.random.gamma(shape, scale, N)
    gamma_dist = stats.gamma(a=shape, scale=scale)

    analyze_continuous("Гамма shape=2, scale=2", sample_gamma, gamma_dist,
                       xlim=(0, np.percentile(sample_gamma, 99)))

    # ---------- 8. Распределение Стьюдента ----------
    df_t = 5
    sample_t = stats.t.rvs(df_t, size=N, random_state=42)
    t_dist = stats.t(df_t)

    analyze_continuous("Стьюдент df=5", sample_t, t_dist, xlim=(-5, 5))

    # ---------- 9. Распределение хи-квадрат ----------
    df_chi = 3
    sample_chi = stats.chi2.rvs(df_chi, size=N, random_state=42)
    chi_dist = stats.chi2(df_chi)

    analyze_continuous("Хи-квадрат df=3", sample_chi, chi_dist,
                       xlim=(0, np.percentile(sample_chi, 99)))

    # ---------- 10. Распределение Фишера ----------
    dfn, dfd = 5, 10
    sample_f = stats.f.rvs(dfn, dfd, size=N, random_state=42)
    f_dist = stats.f(dfn, dfd)

    analyze_continuous("Фишер dfn=5, dfd=10", sample_f, f_dist,
                       xlim=(0, np.percentile(sample_f, 99)))

    # --------------------------------------------------------
    # ПРИМЕРЫ ДЛЯ РАЗНЫХ ЗНАЧЕНИЙ ПАРАМЕТРОВ
    # --------------------------------------------------------

    # Геометрическое с разными p
    for p in [0.2, 0.5, 0.8]:
        d = stats.geom(p)
        s = d.rvs(size=N, random_state=42)
        analyze_discrete(f"Геометрическое p={p}", s, d, xlim=(1, 20))

    # Пуассон с разными mu
    for mu in [1, 5]:
        d = stats.poisson(mu)
        s = d.rvs(size=N, random_state=42)
        analyze_discrete(f"Пуассон mu={mu}", s, d, xlim=(0, 15))

    # Нормальное с разными параметрами
    for mu, sigma in [(0, 1), (2, 0.5), (-1, 2)]:
        s = normal_by_sum(N, mu, sigma)
        d = stats.norm(mu, sigma)
        analyze_continuous(f"Нормальное mu={mu}, sigma={sigma}", s, d,
                           xlim=(mu - 4 * sigma, mu + 4 * sigma))


if __name__ == "__main__":
    main()