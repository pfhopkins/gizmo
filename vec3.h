#pragma once

#include <cmath>

// 3-vector templated on element type.
// Stores 3 elements contiguously — layout-identical to T[3].
// Trivial aggregate (no user-provided constructors) — safe for memset/memcpy/MPI.
template<typename T>
struct Vec3 {
    T data[3];

    T& operator[](int i) noexcept { return data[i]; }
    const T& operator[](int i) const noexcept { return data[i]; }

    T* data_ptr() noexcept { return data; }
    const T* data_ptr() const noexcept { return data; }

    T* begin() noexcept { return data; }
    T* end() noexcept { return data + 3; }
    const T* begin() const noexcept { return data; }
    const T* end() const noexcept { return data + 3; }

    Vec3& operator+=(const Vec3& o) noexcept { data[0]+=o.data[0]; data[1]+=o.data[1]; data[2]+=o.data[2]; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { data[0]-=o.data[0]; data[1]-=o.data[1]; data[2]-=o.data[2]; return *this; }
    Vec3& operator*=(T s) noexcept { data[0]*=s; data[1]*=s; data[2]*=s; return *this; }
    Vec3& operator/=(T s) noexcept { T inv=T(1)/s; data[0]*=inv; data[1]*=inv; data[2]*=inv; return *this; }

    T norm_sq() const noexcept { return data[0]*data[0] + data[1]*data[1] + data[2]*data[2]; }
    T norm() const noexcept { return std::sqrt(norm_sq()); }
};

template<typename T> inline Vec3<T> operator+(Vec3<T> a, const Vec3<T>& b) noexcept { a += b; return a; }
template<typename T> inline Vec3<T> operator-(Vec3<T> a, const Vec3<T>& b) noexcept { a -= b; return a; }
template<typename T> inline Vec3<T> operator-(Vec3<T> a) noexcept { a.data[0]=-a.data[0]; a.data[1]=-a.data[1]; a.data[2]=-a.data[2]; return a; }
template<typename T> inline Vec3<T> operator*(T s, Vec3<T> v) noexcept { v *= s; return v; }
template<typename T> inline Vec3<T> operator*(Vec3<T> v, T s) noexcept { v *= s; return v; }
template<typename T> inline Vec3<T> operator/(Vec3<T> v, T s) noexcept { v /= s; return v; }

template<typename T> inline T dot(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return a.data[0]*b.data[0] + a.data[1]*b.data[1] + a.data[2]*b.data[2];
}

template<typename T> inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return Vec3<T>{ a.data[1]*b.data[2] - a.data[2]*b.data[1],
                    a.data[2]*b.data[0] - a.data[0]*b.data[2],
                    a.data[0]*b.data[1] - a.data[1]*b.data[0] };
}

static_assert(sizeof(Vec3<double>) == 3*sizeof(double), "Vec3<double> layout check failed");
static_assert(sizeof(Vec3<float>)  == 3*sizeof(float),  "Vec3<float> layout check failed");
