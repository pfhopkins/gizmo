#pragma once

#include "vec3.h"

// 3x3 matrix templated on element type.
// Stores 3 row-vectors contiguously — layout-identical to Vec3<T>[3].
// Trivial aggregate (no user-provided constructors) — safe for memset/memcpy/MPI.
template<typename T>
struct Mat3 {
    Vec3<T> row[3];

    Vec3<T>& operator[](int i) noexcept { return row[i]; }
    const Vec3<T>& operator[](int i) const noexcept { return row[i]; }

    T trace() const noexcept { return row[0][0] + row[1][1] + row[2][2]; }

    T frobenius_norm_sq() const noexcept {
        return row[0].norm_sq() + row[1].norm_sq() + row[2].norm_sq();
    }
    T frobenius_norm() const noexcept { return std::sqrt(frobenius_norm_sq()); }

    // Matrix-vector product: M * v
    Vec3<T> matvec(const Vec3<T>& v) const noexcept {
        return Vec3<T>{ dot(row[0], v), dot(row[1], v), dot(row[2], v) };
    }

    // Transpose-multiply: M^T * v
    Vec3<T> Tmatvec(const Vec3<T>& v) const noexcept {
        return Vec3<T>{ row[0][0]*v[0] + row[1][0]*v[1] + row[2][0]*v[2],
                        row[0][1]*v[0] + row[1][1]*v[1] + row[2][1]*v[2],
                        row[0][2]*v[0] + row[1][2]*v[1] + row[2][2]*v[2] };
    }

    // Curl of vector field whose gradient is this matrix: (dv2/dx1 - dv1/dx2, dv0/dx2 - dv2/dx0, dv1/dx0 - dv0/dx1)
    // where row[i][j] = dv_i/dx_j
    Vec3<T> curl() const noexcept {
        return Vec3<T>{ row[2][1] - row[1][2], row[0][2] - row[2][0], row[1][0] - row[0][1] };
    }

    // Determinant of the matrix.
    T det() const noexcept {
        return row[0][0] * (row[1][1] * row[2][2] - row[1][2] * row[2][1])
             - row[0][1] * (row[1][0] * row[2][2] - row[1][2] * row[2][0])
             + row[0][2] * (row[1][0] * row[2][1] - row[1][1] * row[2][0]);
    }

    // Invert the matrix, writing the result into `inv`.
    // Returns the determinant. If the matrix is singular (det==0 or NaN),
    // `inv` is zeroed and 0 is returned.
    T invert(Mat3& inv) const noexcept {
        T d = det();
        if(d == T(0) || d != d) { // d != d catches NaN
            inv = Mat3{};
            return T(0);
        }
        T inv_d = T(1) / d;
        inv[0][0] = (row[1][1] * row[2][2] - row[1][2] * row[2][1]) * inv_d;
        inv[0][1] = (row[0][2] * row[2][1] - row[0][1] * row[2][2]) * inv_d;
        inv[0][2] = (row[0][1] * row[1][2] - row[0][2] * row[1][1]) * inv_d;
        inv[1][0] = (row[1][2] * row[2][0] - row[1][0] * row[2][2]) * inv_d;
        inv[1][1] = (row[0][0] * row[2][2] - row[0][2] * row[2][0]) * inv_d;
        inv[1][2] = (row[0][2] * row[1][0] - row[0][0] * row[1][2]) * inv_d;
        inv[2][0] = (row[1][0] * row[2][1] - row[1][1] * row[2][0]) * inv_d;
        inv[2][1] = (row[0][1] * row[2][0] - row[0][0] * row[2][1]) * inv_d;
        inv[2][2] = (row[0][0] * row[1][1] - row[0][1] * row[1][0]) * inv_d;
        return d;
    }

    Mat3& operator+=(const Mat3& o) noexcept { row[0]+=o.row[0]; row[1]+=o.row[1]; row[2]+=o.row[2]; return *this; }
    Mat3& operator-=(const Mat3& o) noexcept { row[0]-=o.row[0]; row[1]-=o.row[1]; row[2]-=o.row[2]; return *this; }
    Mat3& operator*=(T s) noexcept { row[0]*=s; row[1]*=s; row[2]*=s; return *this; }
    Mat3& operator/=(T s) noexcept { T inv=T(1)/s; row[0]*=inv; row[1]*=inv; row[2]*=inv; return *this; }
};

template<typename T> inline Mat3<T> operator+(Mat3<T> a, const Mat3<T>& b) noexcept { a += b; return a; }
template<typename T> inline Mat3<T> operator-(Mat3<T> a, const Mat3<T>& b) noexcept { a -= b; return a; }
template<typename T> inline Mat3<T> operator*(T s, Mat3<T> m) noexcept { m *= s; return m; }
template<typename T> inline Mat3<T> operator*(Mat3<T> m, T s) noexcept { m *= s; return m; }
template<typename T> inline Mat3<T> operator/(Mat3<T> m, T s) noexcept { m /= s; return m; }

template<typename T>
inline Mat3<T> transpose(const Mat3<T>& m) noexcept {
    return Mat3<T>{{ {m[0][0], m[1][0], m[2][0]},
                     {m[0][1], m[1][1], m[2][1]},
                     {m[0][2], m[1][2], m[2][2]} }};
}

static_assert(sizeof(Mat3<double>) == 9*sizeof(double), "Mat3<double> layout check failed");
static_assert(sizeof(Mat3<float>)  == 9*sizeof(float),  "Mat3<float> layout check failed");
