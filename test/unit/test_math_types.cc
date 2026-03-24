// Unit tests for Vec3, Mat3, and SymmetricTensor2.
#include "test_harness.h"
#include "../../math_types/vec3.h"
#include "../../math_types/mat3.h"
#include "../../math_types/symmetric_tensor2.h"

static constexpr double EPS = 1e-14;

// ============================================================
//  Vec3
// ============================================================

TEST_CASE("Vec3: brace init and element access") {
    Vec3<double> v{1.0, 2.0, 3.0};
    CHECK_CLOSE(v[0], 1.0, EPS);
    CHECK_CLOSE(v[1], 2.0, EPS);
    CHECK_CLOSE(v[2], 3.0, EPS);
}

TEST_CASE("Vec3: data_ptr") {
    Vec3<double> v{4.0, 5.0, 6.0};
    double* p = v.data_ptr();
    CHECK(p == &v[0]);
    CHECK_CLOSE(p[0], 4.0, EPS);
    CHECK_CLOSE(p[1], 5.0, EPS);
    CHECK_CLOSE(p[2], 6.0, EPS);
}

TEST_CASE("Vec3: begin/end iteration") {
    Vec3<double> v{1.0, 2.0, 3.0};
    double sum = 0;
    for (double x : v) sum += x;
    CHECK_CLOSE(sum, 6.0, EPS);
}

TEST_CASE("Vec3: operator+= -= *= /=") {
    Vec3<double> a{1.0, 2.0, 3.0};
    Vec3<double> b{4.0, 5.0, 6.0};

    a += b;
    CHECK_CLOSE(a[0], 5.0, EPS); CHECK_CLOSE(a[1], 7.0, EPS); CHECK_CLOSE(a[2], 9.0, EPS);

    a -= b;
    CHECK_CLOSE(a[0], 1.0, EPS); CHECK_CLOSE(a[1], 2.0, EPS); CHECK_CLOSE(a[2], 3.0, EPS);

    a *= 2.0;
    CHECK_CLOSE(a[0], 2.0, EPS); CHECK_CLOSE(a[1], 4.0, EPS); CHECK_CLOSE(a[2], 6.0, EPS);

    a /= 2.0;
    CHECK_CLOSE(a[0], 1.0, EPS); CHECK_CLOSE(a[1], 2.0, EPS); CHECK_CLOSE(a[2], 3.0, EPS);
}

TEST_CASE("Vec3: binary + - operators") {
    Vec3<double> a{1.0, 2.0, 3.0}, b{10.0, 20.0, 30.0};
    Vec3<double> c = a + b;
    CHECK_CLOSE(c[0], 11.0, EPS); CHECK_CLOSE(c[1], 22.0, EPS); CHECK_CLOSE(c[2], 33.0, EPS);
    Vec3<double> d = b - a;
    CHECK_CLOSE(d[0], 9.0, EPS); CHECK_CLOSE(d[1], 18.0, EPS); CHECK_CLOSE(d[2], 27.0, EPS);
}

TEST_CASE("Vec3: unary negation") {
    Vec3<double> a{1.0, -2.0, 3.0};
    Vec3<double> b = -a;
    CHECK_CLOSE(b[0], -1.0, EPS); CHECK_CLOSE(b[1], 2.0, EPS); CHECK_CLOSE(b[2], -3.0, EPS);
}

TEST_CASE("Vec3: scalar * and / (both orderings)") {
    Vec3<double> v{2.0, 4.0, 6.0};
    Vec3<double> a = 3.0 * v;
    Vec3<double> b = v * 3.0;
    CHECK_CLOSE(a[0], 6.0, EPS); CHECK_CLOSE(a[1], 12.0, EPS); CHECK_CLOSE(a[2], 18.0, EPS);
    CHECK_CLOSE(b[0], 6.0, EPS); CHECK_CLOSE(b[1], 12.0, EPS); CHECK_CLOSE(b[2], 18.0, EPS);
    Vec3<double> c = v / 2.0;
    CHECK_CLOSE(c[0], 1.0, EPS); CHECK_CLOSE(c[1], 2.0, EPS); CHECK_CLOSE(c[2], 3.0, EPS);
}

TEST_CASE("Vec3: norm and norm_sq") {
    Vec3<double> v{3.0, 4.0, 0.0};
    CHECK_CLOSE(v.norm_sq(), 25.0, EPS);
    CHECK_CLOSE(v.norm(), 5.0, EPS);

    Vec3<double> zero{0.0, 0.0, 0.0};
    CHECK_CLOSE(zero.norm(), 0.0, EPS);
}

TEST_CASE("Vec3: dot product") {
    Vec3<double> a{1.0, 2.0, 3.0}, b{4.0, -5.0, 6.0};
    CHECK_CLOSE(dot(a, b), 1*4 + 2*(-5) + 3*6, EPS);  // 4 - 10 + 18 = 12
}

TEST_CASE("Vec3: cross product") {
    Vec3<double> x{1,0,0}, y{0,1,0}, z{0,0,1};
    Vec3<double> c = cross(x, y);
    CHECK_CLOSE(c[0], 0.0, EPS); CHECK_CLOSE(c[1], 0.0, EPS); CHECK_CLOSE(c[2], 1.0, EPS);
    c = cross(y, z);
    CHECK_CLOSE(c[0], 1.0, EPS); CHECK_CLOSE(c[1], 0.0, EPS); CHECK_CLOSE(c[2], 0.0, EPS);
    c = cross(z, x);
    CHECK_CLOSE(c[0], 0.0, EPS); CHECK_CLOSE(c[1], 1.0, EPS); CHECK_CLOSE(c[2], 0.0, EPS);
    // cross product is anti-symmetric
    Vec3<double> a{1,2,3}, b{4,5,6};
    Vec3<double> ab = cross(a, b), ba = cross(b, a);
    CHECK_CLOSE(ab[0], -ba[0], EPS); CHECK_CLOSE(ab[1], -ba[1], EPS); CHECK_CLOSE(ab[2], -ba[2], EPS);
    // a x a = 0
    Vec3<double> aa = cross(a, a);
    CHECK_CLOSE(aa[0], 0.0, EPS); CHECK_CLOSE(aa[1], 0.0, EPS); CHECK_CLOSE(aa[2], 0.0, EPS);
}

TEST_CASE("Vec3: sizeof layout") {
    CHECK(sizeof(Vec3<double>) == 3 * sizeof(double));
    CHECK(sizeof(Vec3<float>) == 3 * sizeof(float));
}

// ============================================================
//  Mat3
// ============================================================

TEST_CASE("Mat3: brace init and element access") {
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};
    CHECK_CLOSE(m[0][0], 1.0, EPS); CHECK_CLOSE(m[0][1], 2.0, EPS); CHECK_CLOSE(m[0][2], 3.0, EPS);
    CHECK_CLOSE(m[1][0], 4.0, EPS); CHECK_CLOSE(m[1][1], 5.0, EPS); CHECK_CLOSE(m[1][2], 6.0, EPS);
    CHECK_CLOSE(m[2][0], 7.0, EPS); CHECK_CLOSE(m[2][1], 8.0, EPS); CHECK_CLOSE(m[2][2], 9.0, EPS);
}

TEST_CASE("Mat3: trace") {
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};
    CHECK_CLOSE(m.trace(), 15.0, EPS); // 1 + 5 + 9
}

TEST_CASE("Mat3: frobenius_norm_sq and frobenius_norm") {
    Mat3<double> m{{{1,0,0},{0,0,0},{0,0,0}}};
    CHECK_CLOSE(m.frobenius_norm_sq(), 1.0, EPS);
    CHECK_CLOSE(m.frobenius_norm(), 1.0, EPS);

    // identity
    Mat3<double> I{{{1,0,0},{0,1,0},{0,0,1}}};
    CHECK_CLOSE(I.frobenius_norm_sq(), 3.0, EPS);
}

TEST_CASE("Mat3: matvec") {
    Mat3<double> I{{{1,0,0},{0,1,0},{0,0,1}}};
    Vec3<double> v{3,4,5};
    Vec3<double> r = I.matvec(v);
    CHECK_CLOSE(r[0], 3.0, EPS); CHECK_CLOSE(r[1], 4.0, EPS); CHECK_CLOSE(r[2], 5.0, EPS);

    // non-trivial
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};
    Vec3<double> u{1,0,0};
    r = m.matvec(u);
    CHECK_CLOSE(r[0], 1.0, EPS); CHECK_CLOSE(r[1], 4.0, EPS); CHECK_CLOSE(r[2], 7.0, EPS);

    r = m.matvec(v);
    CHECK_CLOSE(r[0], 1*3+2*4+3*5, EPS); // 26
    CHECK_CLOSE(r[1], 4*3+5*4+6*5, EPS); // 62
    CHECK_CLOSE(r[2], 7*3+8*4+9*5, EPS); // 98
}

TEST_CASE("Mat3: Tmatvec (transpose-multiply)") {
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};
    Vec3<double> v{1,1,1};
    // M^T * v: column sums
    Vec3<double> r = m.Tmatvec(v);
    CHECK_CLOSE(r[0], 1+4+7, EPS); // 12
    CHECK_CLOSE(r[1], 2+5+8, EPS); // 15
    CHECK_CLOSE(r[2], 3+6+9, EPS); // 18

    // For symmetric matrix, matvec == Tmatvec
    Mat3<double> s{{{1,2,3},{2,5,6},{3,6,9}}};
    Vec3<double> u{1,2,3};
    Vec3<double> a = s.matvec(u), b = s.Tmatvec(u);
    CHECK_CLOSE(a[0], b[0], EPS); CHECK_CLOSE(a[1], b[1], EPS); CHECK_CLOSE(a[2], b[2], EPS);
}

TEST_CASE("Mat3: curl") {
    // curl of (x, y, z) velocity field with gradient = identity -> curl = 0
    Mat3<double> I{{{1,0,0},{0,1,0},{0,0,1}}};
    Vec3<double> c = I.curl();
    CHECK_CLOSE(c[0], 0.0, EPS); CHECK_CLOSE(c[1], 0.0, EPS); CHECK_CLOSE(c[2], 0.0, EPS);

    // solid body rotation about z-axis: v = (-y, x, 0), grad = {{0,-1,0},{1,0,0},{0,0,0}}
    // curl = (0-0, 0-0, 1-(-1)) = (0, 0, 2)
    Mat3<double> rot{{{0,-1,0},{1,0,0},{0,0,0}}};
    c = rot.curl();
    CHECK_CLOSE(c[0], 0.0, EPS); CHECK_CLOSE(c[1], 0.0, EPS); CHECK_CLOSE(c[2], 2.0, EPS);
}

TEST_CASE("Mat3: arithmetic operators") {
    Mat3<double> a{{{1,2,3},{4,5,6},{7,8,9}}};
    Mat3<double> b{{{9,8,7},{6,5,4},{3,2,1}}};

    Mat3<double> c = a + b;
    CHECK_CLOSE(c[0][0], 10.0, EPS); CHECK_CLOSE(c[1][1], 10.0, EPS); CHECK_CLOSE(c[2][2], 10.0, EPS);

    Mat3<double> d = a - b;
    CHECK_CLOSE(d[0][0], -8.0, EPS); CHECK_CLOSE(d[1][1], 0.0, EPS); CHECK_CLOSE(d[2][2], 8.0, EPS);

    Mat3<double> e = 2.0 * a;
    CHECK_CLOSE(e[0][0], 2.0, EPS); CHECK_CLOSE(e[1][2], 12.0, EPS);

    Mat3<double> f = a * 2.0;
    CHECK_CLOSE(f[0][0], 2.0, EPS); CHECK_CLOSE(f[1][2], 12.0, EPS);

    Mat3<double> g = a / 2.0;
    CHECK_CLOSE(g[0][0], 0.5, EPS); CHECK_CLOSE(g[2][2], 4.5, EPS);
}

TEST_CASE("Mat3: transpose") {
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};
    Mat3<double> mt = transpose(m);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK_CLOSE(mt[i][j], m[j][i], EPS);
}

TEST_CASE("Mat3: sizeof layout") {
    CHECK(sizeof(Mat3<double>) == 9 * sizeof(double));
    CHECK(sizeof(Mat3<float>) == 9 * sizeof(float));
}

// ============================================================
//  SymmetricTensor2
// ============================================================

TEST_CASE("SymmetricTensor2: storage layout [xx,yy,zz,xy,yz,xz]") {
    SymmetricTensor2<double> s{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // data[0]=xx, data[1]=yy, data[2]=zz, data[3]=xy, data[4]=yz, data[5]=xz
    CHECK_CLOSE(s.data[0], 1.0, EPS); // xx
    CHECK_CLOSE(s.data[1], 2.0, EPS); // yy
    CHECK_CLOSE(s.data[2], 3.0, EPS); // zz
    CHECK_CLOSE(s.data[3], 4.0, EPS); // xy
    CHECK_CLOSE(s.data[4], 5.0, EPS); // yz
    CHECK_CLOSE(s.data[5], 6.0, EPS); // xz
}

TEST_CASE("SymmetricTensor2: flat_index mapping") {
    // Diagonals
    CHECK(SymmetricTensor2<double>::flat_index(0, 0) == 0);
    CHECK(SymmetricTensor2<double>::flat_index(1, 1) == 1);
    CHECK(SymmetricTensor2<double>::flat_index(2, 2) == 2);
    // Off-diagonals (both orderings should give same index)
    CHECK(SymmetricTensor2<double>::flat_index(0, 1) == 3);
    CHECK(SymmetricTensor2<double>::flat_index(1, 0) == 3);
    CHECK(SymmetricTensor2<double>::flat_index(1, 2) == 4);
    CHECK(SymmetricTensor2<double>::flat_index(2, 1) == 4);
    CHECK(SymmetricTensor2<double>::flat_index(0, 2) == 5);
    CHECK(SymmetricTensor2<double>::flat_index(2, 0) == 5);
}

TEST_CASE("SymmetricTensor2: [i][j] proxy access matches data layout") {
    SymmetricTensor2<double> s{11.0, 22.0, 33.0, 12.0, 23.0, 13.0};
    // Diagonals
    CHECK_CLOSE(s[0][0], 11.0, EPS);
    CHECK_CLOSE(s[1][1], 22.0, EPS);
    CHECK_CLOSE(s[2][2], 33.0, EPS);
    // Off-diagonals — symmetric access
    CHECK_CLOSE(s[0][1], 12.0, EPS); CHECK_CLOSE(s[1][0], 12.0, EPS);
    CHECK_CLOSE(s[1][2], 23.0, EPS); CHECK_CLOSE(s[2][1], 23.0, EPS);
    CHECK_CLOSE(s[0][2], 13.0, EPS); CHECK_CLOSE(s[2][0], 13.0, EPS);
}

TEST_CASE("SymmetricTensor2: [i][j] write goes to correct storage") {
    SymmetricTensor2<double> s{};
    s[0][0] = 1; s[1][1] = 2; s[2][2] = 3;
    s[0][1] = 4; s[1][2] = 5; s[0][2] = 6;
    CHECK_CLOSE(s.data[0], 1.0, EPS);
    CHECK_CLOSE(s.data[1], 2.0, EPS);
    CHECK_CLOSE(s.data[2], 3.0, EPS);
    CHECK_CLOSE(s.data[3], 4.0, EPS);
    CHECK_CLOSE(s.data[4], 5.0, EPS);
    CHECK_CLOSE(s.data[5], 6.0, EPS);
    // Writing via transposed index hits same storage
    s[1][0] = 99.0;
    CHECK_CLOSE(s.data[3], 99.0, EPS); // xy slot
    CHECK_CLOSE(s[0][1], 99.0, EPS);   // reading back via (0,1)
}

TEST_CASE("SymmetricTensor2: const [i][j] access") {
    const SymmetricTensor2<double> s{1,2,3,4,5,6};
    CHECK_CLOSE(s[0][0], 1.0, EPS);
    CHECK_CLOSE(s[1][0], 4.0, EPS);
    CHECK_CLOSE(s[0][1], 4.0, EPS);
}

TEST_CASE("SymmetricTensor2: trace") {
    SymmetricTensor2<double> s{1.0, 2.0, 3.0, 99.0, 99.0, 99.0};
    CHECK_CLOSE(s.trace(), 6.0, EPS); // 1 + 2 + 3
}

TEST_CASE("SymmetricTensor2: set_isotropic") {
    SymmetricTensor2<double> s{99,99,99,99,99,99};
    s.set_isotropic(1.0 / 3.0);
    CHECK_CLOSE(s[0][0], 1.0/3.0, EPS);
    CHECK_CLOSE(s[1][1], 1.0/3.0, EPS);
    CHECK_CLOSE(s[2][2], 1.0/3.0, EPS);
    CHECK_CLOSE(s[0][1], 0.0, EPS);
    CHECK_CLOSE(s[1][2], 0.0, EPS);
    CHECK_CLOSE(s[0][2], 0.0, EPS);
    CHECK_CLOSE(s.trace(), 1.0, EPS);
}

TEST_CASE("SymmetricTensor2: operator*= and scalar * (both orderings)") {
    SymmetricTensor2<double> s{1,2,3,4,5,6};
    s *= 2.0;
    CHECK_CLOSE(s.data[0], 2.0, EPS); CHECK_CLOSE(s.data[3], 8.0, EPS);

    SymmetricTensor2<double> a{1,2,3,4,5,6};
    SymmetricTensor2<double> b = 3.0 * a;
    SymmetricTensor2<double> c = a * 3.0;
    for (int k = 0; k < 6; k++) {
        CHECK_CLOSE(b.data[k], 3.0 * a.data[k], EPS);
        CHECK_CLOSE(c.data[k], b.data[k], EPS);
    }
}

TEST_CASE("SymmetricTensor2: operator/= and /") {
    SymmetricTensor2<double> s{2,4,6,8,10,12};
    s /= 2.0;
    CHECK_CLOSE(s.data[0], 1.0, EPS); CHECK_CLOSE(s.data[5], 6.0, EPS);

    SymmetricTensor2<double> t = SymmetricTensor2<double>{2,4,6,8,10,12} / 2.0;
    for (int k = 0; k < 6; k++) CHECK_CLOSE(t.data[k], s.data[k], EPS);
}

TEST_CASE("SymmetricTensor2: operator+= -= and binary + -") {
    SymmetricTensor2<double> a{1,2,3,4,5,6};
    SymmetricTensor2<double> b{6,5,4,3,2,1};

    SymmetricTensor2<double> c = a + b;
    CHECK_CLOSE(c.data[0], 7.0, EPS); CHECK_CLOSE(c.data[5], 7.0, EPS);

    SymmetricTensor2<double> d = a - b;
    CHECK_CLOSE(d.data[0], -5.0, EPS); CHECK_CLOSE(d.data[5], 5.0, EPS);

    a += b;
    for (int k = 0; k < 6; k++) CHECK_CLOSE(a.data[k], c.data[k], EPS);

    a -= b;
    CHECK_CLOSE(a.data[0], 1.0, EPS); CHECK_CLOSE(a.data[5], 6.0, EPS);
}

TEST_CASE("SymmetricTensor2: frobenius_norm_sq counts off-diag twice") {
    // Identity: diag = 3*(1^2) = 3, offdiag = 0 -> total = 3
    SymmetricTensor2<double> I{}; I.set_isotropic(1.0);
    CHECK_CLOSE(I.frobenius_norm_sq(), 3.0, EPS);

    // Single off-diagonal element: s[0][1] = s[1][0] = 1, both contribute
    SymmetricTensor2<double> s{0,0,0,1,0,0}; // only xy = 1
    CHECK_CLOSE(s.frobenius_norm_sq(), 2.0, EPS); // 1^2 counted twice

    // Full tensor: compare with explicit 3x3 sum
    SymmetricTensor2<double> t{1,2,3,4,5,6};
    double expected = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = t[i][j];
            expected += v * v;
        }
    CHECK_CLOSE(t.frobenius_norm_sq(), expected, EPS);
}

TEST_CASE("SymmetricTensor2: frobenius_norm") {
    SymmetricTensor2<double> I{}; I.set_isotropic(1.0);
    CHECK_CLOSE(I.frobenius_norm(), std::sqrt(3.0), EPS);
}

TEST_CASE("SymmetricTensor2: matvec — identity") {
    SymmetricTensor2<double> I{}; I.set_isotropic(1.0);
    Vec3<double> v{3, 4, 5};
    Vec3<double> r = I.matvec(v);
    CHECK_CLOSE(r[0], 3.0, EPS); CHECK_CLOSE(r[1], 4.0, EPS); CHECK_CLOSE(r[2], 5.0, EPS);
}

TEST_CASE("SymmetricTensor2: matvec — general, matches brute-force [i][j]") {
    SymmetricTensor2<double> s{1,2,3,4,5,6}; // xx=1,yy=2,zz=3,xy=4,yz=5,xz=6
    Vec3<double> v{7, 8, 9};
    Vec3<double> r = s.matvec(v);

    // brute-force reference
    for (int i = 0; i < 3; i++) {
        double ref = 0;
        for (int j = 0; j < 3; j++) ref += s[i][j] * v[j];
        CHECK_CLOSE(r[i], ref, EPS);
    }
}

TEST_CASE("SymmetricTensor2: matvec is symmetric (v . S . w == w . S . v)") {
    SymmetricTensor2<double> s{2,3,5,7,11,13};
    Vec3<double> v{1,2,3}, w{4,5,6};
    double vSw = dot(v, s.matvec(w));
    double wSv = dot(w, s.matvec(v));
    CHECK_CLOSE(vSw, wSv, EPS);
}

TEST_CASE("SymmetricTensor2: outer_product storage order") {
    Vec3<double> v{2.0, 3.0, 5.0};
    SymmetricTensor2<double> op = outer_product(v);
    CHECK_CLOSE(op.data[0], 4.0, EPS);  // xx = 2*2
    CHECK_CLOSE(op.data[1], 9.0, EPS);  // yy = 3*3
    CHECK_CLOSE(op.data[2], 25.0, EPS); // zz = 5*5
    CHECK_CLOSE(op.data[3], 6.0, EPS);  // xy = 2*3
    CHECK_CLOSE(op.data[4], 15.0, EPS); // yz = 3*5
    CHECK_CLOSE(op.data[5], 10.0, EPS); // xz = 2*5
}

TEST_CASE("SymmetricTensor2: outer_product [i][j] matches v[i]*v[j]") {
    Vec3<double> v{2.0, 3.0, 5.0};
    SymmetricTensor2<double> op = outer_product(v);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            CHECK_CLOSE(op[i][j], v[i] * v[j], EPS);
}

TEST_CASE("SymmetricTensor2: outer_product matvec gives (v.w)*v") {
    Vec3<double> v{2.0, 3.0, 5.0}, w{7.0, 11.0, 13.0};
    SymmetricTensor2<double> op = outer_product(v);
    Vec3<double> r = op.matvec(w);
    double vdotw = dot(v, w);
    CHECK_CLOSE(r[0], vdotw * v[0], EPS);
    CHECK_CLOSE(r[1], vdotw * v[1], EPS);
    CHECK_CLOSE(r[2], vdotw * v[2], EPS);
}

TEST_CASE("SymmetricTensor2: double_contract with identity") {
    SymmetricTensor2<double> s{1,2,3,4,5,6};
    Mat3<double> I{{{1,0,0},{0,1,0},{0,0,1}}};
    // S : I = trace(S) = 1+2+3 = 6
    CHECK_CLOSE(s.double_contract(I), 6.0, EPS);
}

TEST_CASE("SymmetricTensor2: double_contract with general matrix") {
    SymmetricTensor2<double> s{1,2,3,4,5,6}; // xx=1,yy=2,zz=3,xy=4,yz=5,xz=6
    Mat3<double> m{{{1,2,3},{4,5,6},{7,8,9}}};

    // brute-force: sum_ij s[i][j] * m[i][j]
    double expected = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            expected += s[i][j] * m[i][j];
    CHECK_CLOSE(s.double_contract(m), expected, EPS);
}

TEST_CASE("SymmetricTensor2: double_contract with symmetric M equals S:M = S:M^T") {
    SymmetricTensor2<double> s{2,3,5,7,11,13};
    Mat3<double> M{{{1,4,7},{2,5,8},{3,6,9}}};
    // For symmetric S: S:M == S:M^T
    Mat3<double> Mt = transpose(M);
    CHECK_CLOSE(s.double_contract(M), s.double_contract(Mt), EPS);
}

TEST_CASE("SymmetricTensor2: zero initialization") {
    SymmetricTensor2<double> s{};
    for (int k = 0; k < 6; k++) CHECK_CLOSE(s.data[k], 0.0, EPS);
    CHECK_CLOSE(s.trace(), 0.0, EPS);
    CHECK_CLOSE(s.frobenius_norm(), 0.0, EPS);
}

// ============================================================
//  Cross-type: SymmetricTensor2 matvec vs Mat3 matvec
// ============================================================

TEST_CASE("SymmetricTensor2 vs Mat3: matvec agrees for symmetric matrix") {
    SymmetricTensor2<double> s{2,3,5,7,11,13};
    Mat3<double> m{};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            m[i][j] = s[i][j];

    Vec3<double> v{17, 19, 23};
    Vec3<double> rs = s.matvec(v);
    Vec3<double> rm = m.matvec(v);
    CHECK_CLOSE(rs[0], rm[0], EPS);
    CHECK_CLOSE(rs[1], rm[1], EPS);
    CHECK_CLOSE(rs[2], rm[2], EPS);
}

// ---- Mat3 det and invert ----

TEST_CASE("Mat3: det of identity is 1") {
    Mat3<double> I = {{{1,0,0},{0,1,0},{0,0,1}}};
    CHECK_CLOSE(I.det(), 1.0, EPS);
}

TEST_CASE("Mat3: det of known matrix") {
    Mat3<double> A = {{{1,2,3},{0,4,5},{1,0,6}}};
    // det = 1*(4*6-5*0) - 2*(0*6-5*1) + 3*(0*0-4*1) = 24+10-12 = 22
    CHECK_CLOSE(A.det(), 22.0, EPS);
}

TEST_CASE("Mat3: invert of identity is identity") {
    Mat3<double> I = {{{1,0,0},{0,1,0},{0,0,1}}};
    Mat3<double> Iinv;
    double d = I.invert(Iinv);
    CHECK_CLOSE(d, 1.0, EPS);
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        CHECK_CLOSE(Iinv[i][j], (i==j ? 1.0 : 0.0), EPS);
}

TEST_CASE("Mat3: A * A^{-1} = I") {
    Mat3<double> A = {{{1,2,3},{0,4,5},{1,0,6}}};
    Mat3<double> Ainv;
    double d = A.invert(Ainv);
    CHECK(d != 0.0);
    // multiply A * Ainv, check it's identity
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
        double sum = 0;
        for(int k=0;k<3;k++) sum += A[i][k]*Ainv[k][j];
        CHECK_CLOSE(sum, (i==j ? 1.0 : 0.0), 1e-12);
    }
}

TEST_CASE("Mat3: invert of singular matrix returns 0") {
    Mat3<double> S = {{{1,2,3},{2,4,6},{1,1,1}}}; // row 1 = 2*row 0
    Mat3<double> Sinv;
    double d = S.invert(Sinv);
    CHECK_CLOSE(d, 0.0, EPS);
}

TEST_MAIN()
