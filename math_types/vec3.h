#pragma once

#include <cmath>
#ifndef GIZMO_GPU_FUNCTION
#define GIZMO_GPU_FUNCTION
#endif

// 3-vector templated on element type.
// Stores 3 elements contiguously — layout-identical to T[3].
// Trivial aggregate (no user-provided constructors) — safe for memset/memcpy/MPI.
template<typename T>
struct Vec3 {
    T data[3];

    GIZMO_GPU_FUNCTION T& operator[](int i) noexcept { return data[i]; }
    GIZMO_GPU_FUNCTION const T& operator[](int i) const noexcept { return data[i]; }

    GIZMO_GPU_FUNCTION T* data_ptr() noexcept { return data; }
    GIZMO_GPU_FUNCTION const T* data_ptr() const noexcept { return data; }

    GIZMO_GPU_FUNCTION T* begin() noexcept { return data; }
    GIZMO_GPU_FUNCTION T* end() noexcept { return data + 3; }
    GIZMO_GPU_FUNCTION const T* begin() const noexcept { return data; }
    GIZMO_GPU_FUNCTION const T* end() const noexcept { return data + 3; }

    GIZMO_GPU_FUNCTION Vec3& operator+=(const Vec3& o) noexcept { data[0]+=o.data[0]; data[1]+=o.data[1]; data[2]+=o.data[2]; return *this; }
    GIZMO_GPU_FUNCTION Vec3& operator-=(const Vec3& o) noexcept { data[0]-=o.data[0]; data[1]-=o.data[1]; data[2]-=o.data[2]; return *this; }
    GIZMO_GPU_FUNCTION Vec3& operator*=(T s) noexcept { data[0]*=s; data[1]*=s; data[2]*=s; return *this; }
    GIZMO_GPU_FUNCTION Vec3& operator/=(T s) noexcept { T inv=T(1)/s; data[0]*=inv; data[1]*=inv; data[2]*=inv; return *this; }

    GIZMO_GPU_FUNCTION T norm_sq() const noexcept { return data[0]*data[0] + data[1]*data[1] + data[2]*data[2]; }
    GIZMO_GPU_FUNCTION T norm() const noexcept { return sqrt(norm_sq()); }
};

template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator+(Vec3<T> a, const Vec3<T>& b) noexcept { a += b; return a; }
template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator-(Vec3<T> a, const Vec3<T>& b) noexcept { a -= b; return a; }
template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator-(Vec3<T> a) noexcept { a.data[0]=-a.data[0]; a.data[1]=-a.data[1]; a.data[2]=-a.data[2]; return a; }
template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator*(T s, Vec3<T> v) noexcept { v *= s; return v; }
template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator*(Vec3<T> v, T s) noexcept { v *= s; return v; }
template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> operator/(Vec3<T> v, T s) noexcept { v /= s; return v; }

template<typename T> GIZMO_GPU_FUNCTION inline T dot(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return a.data[0]*b.data[0] + a.data[1]*b.data[1] + a.data[2]*b.data[2];
}

template<typename T> GIZMO_GPU_FUNCTION inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) noexcept {
    return Vec3<T>{ a.data[1]*b.data[2] - a.data[2]*b.data[1],
                    a.data[2]*b.data[0] - a.data[0]*b.data[2],
                    a.data[0]*b.data[1] - a.data[1]*b.data[0] };
}

static_assert(sizeof(Vec3<double>) == 3*sizeof(double), "Vec3<double> layout check failed");
static_assert(sizeof(Vec3<float>)  == 3*sizeof(float),  "Vec3<float> layout check failed");
