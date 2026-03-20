#pragma once

#include <cmath>

// Rank-2 symmetric 3x3 tensor templated on element type.
// Stores 6 unique elements (xx, xy, xz, yy, yz, zz).
// tensor[i][j] == tensor[j][i] — no redundant storage.
// Simple aggregate (no default constructor) so it is safe inside memset'd structs.
template<typename T>
struct SymmetricTensor2 {
    // Storage: lower triangle, row-major: [0]=xx [1]=xy [2]=xz [3]=yy [4]=yz [5]=zz
    T data[6];

    static constexpr int flat_index(int i, int j) noexcept {
        if (i < j) { int tmp = i; i = j; j = tmp; }
        return i * (i + 1) / 2 + j;
    }

    struct Row {
        SymmetricTensor2 &t;
        int i;
        T &operator[](int j) noexcept { return t.data[flat_index(i, j)]; }
        const T &operator[](int j) const noexcept { return t.data[flat_index(i, j)]; }
    };

    struct ConstRow {
        const SymmetricTensor2 &t;
        int i;
        const T &operator[](int j) const noexcept { return t.data[flat_index(i, j)]; }
    };

    Row operator[](int i) noexcept { return {*this, i}; }
    ConstRow operator[](int i) const noexcept { return {*this, i}; }

    // Squared Frobenius norm: sum of squares of all 9 entries (off-diagonal count twice).
    double frobenius_norm_sq() const noexcept {
        const double diag    = data[0]*data[0] + data[3]*data[3] + data[5]*data[5];
        const double offdiag = data[1]*data[1] + data[2]*data[2] + data[4]*data[4];
        return diag + 2.0 * offdiag;
    }

    // Frobenius norm: sqrt(sum of squares of all 9 entries).
    double frobenius_norm() const noexcept { return std::sqrt(frobenius_norm_sq()); }
};
