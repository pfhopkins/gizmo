#pragma once

#include <cmath>
#include <type_traits>

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

    // Cross-precision conversion and assignment (for float-stored fields, PRECISION_DIET). No converting
    // constructor: it would cost Vec3 its aggregate status and break `Vec3<double> v = {a,b,c}`.
    // Compound operators narrow on store; the free binary operators below promote.
    template<typename U> operator Vec3<U>() const noexcept {
        return Vec3<U>{static_cast<U>(data[0]), static_cast<U>(data[1]), static_cast<U>(data[2])};
    }
    template<typename U> Vec3& operator=(const Vec3<U>& o) noexcept {
        data[0]=static_cast<T>(o.data[0]); data[1]=static_cast<T>(o.data[1]); data[2]=static_cast<T>(o.data[2]); return *this;
    }
    template<typename U> Vec3& operator+=(const Vec3<U>& o) noexcept {
        for(int k=0;k<3;k++) {data[k] = static_cast<T>(data[k] + o.data[k]);} return *this;
    }
    template<typename U> Vec3& operator-=(const Vec3<U>& o) noexcept {
        for(int k=0;k<3;k++) {data[k] = static_cast<T>(data[k] - o.data[k]);} return *this;
    }

    Vec3& operator+=(const Vec3& o) noexcept { data[0]+=o.data[0]; data[1]+=o.data[1]; data[2]+=o.data[2]; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { data[0]-=o.data[0]; data[1]-=o.data[1]; data[2]-=o.data[2]; return *this; }
    Vec3& operator*=(T s) noexcept { data[0]*=s; data[1]*=s; data[2]*=s; return *this; }
    Vec3& operator/=(T s) noexcept { T inv=T(1)/s; data[0]*=inv; data[1]*=inv; data[2]*=inv; return *this; }

    T norm_sq() const noexcept { return data[0]*data[0] + data[1]*data[1] + data[2]*data[2]; }
    T norm() const noexcept { return std::sqrt(norm_sq()); }
};

// Binary operators evaluate in the common type of the operands, so a float-stored field meeting a double
// computes in double. With T==U they are the plain same-type operators (bitwise unchanged).
template<typename T, typename U> using Vec3Promote = typename std::common_type<T,U>::type;

template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> operator+(const Vec3<T>& a, const Vec3<U>& b) noexcept {
    using R = Vec3Promote<T,U>;
    return Vec3<R>{ R(a.data[0])+R(b.data[0]), R(a.data[1])+R(b.data[1]), R(a.data[2])+R(b.data[2]) };
}
template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> operator-(const Vec3<T>& a, const Vec3<U>& b) noexcept {
    using R = Vec3Promote<T,U>;
    return Vec3<R>{ R(a.data[0])-R(b.data[0]), R(a.data[1])-R(b.data[1]), R(a.data[2])-R(b.data[2]) };
}
template<typename T> inline Vec3<T> operator-(Vec3<T> a) noexcept { a.data[0]=-a.data[0]; a.data[1]=-a.data[1]; a.data[2]=-a.data[2]; return a; }

template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> operator*(U s, const Vec3<T>& v) noexcept {
    using R = Vec3Promote<T,U>;
    return Vec3<R>{ R(s)*R(v.data[0]), R(s)*R(v.data[1]), R(s)*R(v.data[2]) };
}
template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> operator*(const Vec3<T>& v, U s) noexcept {
    using R = Vec3Promote<T,U>;
    return Vec3<R>{ R(v.data[0])*R(s), R(v.data[1])*R(s), R(v.data[2])*R(s) };
}
template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> operator/(const Vec3<T>& v, U s) noexcept {
    using R = Vec3Promote<T,U>; R inv = R(1)/R(s);
    return Vec3<R>{ R(v.data[0])*inv, R(v.data[1])*inv, R(v.data[2])*inv };
}

template<typename T, typename U> inline Vec3Promote<T,U> dot(const Vec3<T>& a, const Vec3<U>& b) noexcept {
    using R = Vec3Promote<T,U>;
    return R(a.data[0])*R(b.data[0]) + R(a.data[1])*R(b.data[1]) + R(a.data[2])*R(b.data[2]);
}

template<typename T, typename U> inline Vec3<Vec3Promote<T,U>> cross(const Vec3<T>& a, const Vec3<U>& b) noexcept {
    using R = Vec3Promote<T,U>;
    return Vec3<R>{ R(a.data[1])*R(b.data[2]) - R(a.data[2])*R(b.data[1]),
                    R(a.data[2])*R(b.data[0]) - R(a.data[0])*R(b.data[2]),
                    R(a.data[0])*R(b.data[1]) - R(a.data[1])*R(b.data[0]) };
}

static_assert(sizeof(Vec3<double>) == 3*sizeof(double), "Vec3<double> layout check failed");
static_assert(sizeof(Vec3<float>)  == 3*sizeof(float),  "Vec3<float> layout check failed");
