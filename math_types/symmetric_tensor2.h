#pragma once

#include <cmath>
#include "vec3.h"
#include "mat3.h"

// Rank-2 symmetric 3x3 tensor templated on element type.
// Stores 6 unique elements: [0]=xx [1]=yy [2]=zz [3]=xy [4]=yz [5]=xz
// tensor[i][j] == tensor[j][i] — no redundant storage.
// Simple aggregate (no default constructor) so it is safe inside memset'd structs.
template<typename T>
struct SymmetricTensor2 {
    // Storage: diagonals first, then off-diagonals: [0]=xx [1]=yy [2]=zz [3]=xy [4]=yz [5]=xz
    T data[6];

    static constexpr int flat_index(int i, int j) noexcept {
        if (i > j) { int tmp = i; i = j; j = tmp; }
        if (i == j) return i; // (0,0)->0, (1,1)->1, (2,2)->2
        if (i == 0 && j == 1) return 3; // xy
        if (i == 1 && j == 2) return 4; // yz
        return 5; // (0,2) -> xz
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

    // Trace: sum of diagonal elements.
    T trace() const noexcept { return data[0] + data[1] + data[2]; }

    // Scalar multiply in-place.
    SymmetricTensor2& operator*=(T s) noexcept {
        for(int k = 0; k < 6; k++) { data[k] *= s; }
        return *this;
    }

    // Element-wise addition in-place.
    SymmetricTensor2& operator+=(const SymmetricTensor2& other) noexcept {
        for(int k = 0; k < 6; k++) { data[k] += other.data[k]; }
        return *this;
    }

    // Element-wise subtraction in-place.
    SymmetricTensor2& operator-=(const SymmetricTensor2& other) noexcept {
        for(int k = 0; k < 6; k++) { data[k] -= other.data[k]; }
        return *this;
    }

    // Squared Frobenius norm: sum of squares of all 9 entries (off-diagonal count twice).
    T frobenius_norm_sq() const noexcept {
        const T diag    = data[0]*data[0] + data[1]*data[1] + data[2]*data[2];
        const T offdiag = data[3]*data[3] + data[4]*data[4] + data[5]*data[5];
        return diag + T(2) * offdiag;
    }

    // Frobenius norm: sqrt(sum of squares of all 9 entries).
    T frobenius_norm() const noexcept { return std::sqrt(frobenius_norm_sq()); }

    // Scalar divide in-place.
    SymmetricTensor2& operator/=(T s) noexcept {
        T inv = T(1)/s;
        for(int k = 0; k < 6; k++) { data[k] *= inv; }
        return *this;
    }

    // Matrix-vector product: returns this * v.  xx=0 yy=1 zz=2 xy=3 yz=4 xz=5
    Vec3<T> matvec(const Vec3<T>& v) const noexcept {
        return Vec3<T>{data[0]*v[0] + data[3]*v[1] + data[5]*v[2],
                       data[3]*v[0] + data[1]*v[1] + data[4]*v[2],
                       data[5]*v[0] + data[4]*v[1] + data[2]*v[2]};
    }

    // Double contraction with a general 3x3 matrix: sum_ij S_ij * M_ij
    // For symmetric S, this equals S_ii*M_ii + S_ij*(M_ij+M_ji) for i<j
    template<typename U>
    T double_contract(const Mat3<U>& M) const noexcept {
        return data[0]*M[0][0] + data[1]*M[1][1] + data[2]*M[2][2]
             + data[3]*(M[0][1]+M[1][0]) + data[4]*(M[1][2]+M[2][1]) + data[5]*(M[0][2]+M[2][0]);
    }

    // Set to scalar multiple of identity: diag = val, off-diag = 0
    void set_isotropic(T val) noexcept {
        data[0] = val; data[1] = val; data[2] = val;
        data[3] = T(0); data[4] = T(0); data[5] = T(0);
    }
};

template<typename T>
inline SymmetricTensor2<T> operator*(T s, SymmetricTensor2<T> t) noexcept { t *= s; return t; }

template<typename T>
inline SymmetricTensor2<T> operator*(SymmetricTensor2<T> t, T s) noexcept { t *= s; return t; }

template<typename T>
inline SymmetricTensor2<T> operator+(SymmetricTensor2<T> a, const SymmetricTensor2<T>& b) noexcept { a += b; return a; }

template<typename T>
inline SymmetricTensor2<T> operator-(SymmetricTensor2<T> a, const SymmetricTensor2<T>& b) noexcept { a -= b; return a; }

template<typename T>
inline SymmetricTensor2<T> operator/(SymmetricTensor2<T> t, T s) noexcept { t /= s; return t; }

// Symmetric outer product v ⊗ v
// Storage: [0]=xx [1]=yy [2]=zz [3]=xy [4]=yz [5]=xz
template<typename T>
inline SymmetricTensor2<T> outer_product(const Vec3<T>& v) noexcept {
    return { v[0]*v[0], v[1]*v[1], v[2]*v[2],
             v[0]*v[1], v[1]*v[2], v[0]*v[2] };
}
