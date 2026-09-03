//! blackhole - a relativistic black hole for the terminal
//!
//! Light is integrated along real null geodesics of the Schwarzschild metric.
//! In units of the Schwarzschild radius the spatial path of a photon obeys
//!
//!     d2x/dl2 = -3/2 * h2 * x / |x|^5,      h2 = |x x dx/dl|^2
//!
//! which is exact (it is the u'' + u = 3mu^2 orbit equation written in cartesian
//! form).  Everything the picture shows - the shadow, the photon ring, the arc
//! of the far side of the disk over the top of the hole - comes out of that one
//! line.  The thin disk is sampled where a ray crosses the equatorial plane;
//! on top of the geometry there is special-relativistic Doppler beaming and
//! gravitational redshift, both driven by the local Keplerian velocity.
//!
//! No external crates.

use std::env;
use std::io::{Read, Write};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

// ------------------------------------------------------------------ scene
// 1 unit == one Schwarzschild radius.
// event horizon  r = 1, photon sphere r = 1.5, ISCO r = 3,
// apparent shadow radius 3*sqrt(3)/2 = 2.6.

const RS: f64 = 1.0;
const R_IN: f64 = 3.05; // inner rim of the disk (just outside the ISCO)
const R_OUT: f64 = 11.0; // outer edge
const CAM_D: f64 = 18.0; // camera distance from the hole
const CAM_TILT: f64 = 0.100; // degrees above the disk plane: near edge-on,
                             // the classic look - the disk's thickness on screen is mostly lensing
const VIEW: f64 = 0.30; // half height of the frustum (tangent of half fov)
const DISK_OPA: f64 = 0.05; // transmittance of one disk crossing (opaque disk)
const ESCAPE: f64 = 32.0; // where a ray is considered free again
const MAX_STEPS: usize = 900;
const EXPOSURE: f64 = 1.0; // tone map exposure
const BRIGHT: f64 = 3.0; // disk emission scale
const STAR_DENSITY: f64 = 0.74; // probability a cell is empty; lower = more stars
/// target Gaussian radius of a star core, in ray-grid pixels; the actual
/// angular sigma is derived from it so a star is a crisp dot at every
/// resolution instead of a fixed-angle blob
const STAR_CORE: f64 = 0.6;
const STAR_SCALE: [f64; 3] = [44.0, 74.0, 120.0];
const STAR_BRI: [f64; 3] = [1.0, 0.55, 0.30];
/// Soft cap on how fast the disk turbulence may drift, in device pixels per
/// second. The Keplerian pattern near the inner rim moves at ~1.6 rad/s,
/// which is over a thousand pixels per second on a sixel grid - fine noise
/// granules then jump dozens of pixels between frames and the disk turns into
/// strobing static. Character grids are coarse enough that the cap never
/// engages there; their calm look is what this reproduces at device size.
const TURB_MAX_PX: f64 = 300.0;

// ------------------------------------------------------------------- maths
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Blackbody-ish ramp: 0 = deep red, 1 = blue white.
fn heat(t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        1.0 - 0.45 * t * t,
        0.30 + 0.68 * t * t,
        0.10 + 0.95 * t * t * t,
    ]
}

fn hash3i(x: i64, y: i64, z: i64) -> f64 {
    let mut h = 0x9E3779B97F4A7C15u64
        ^ (x as u64).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (y as u64).wrapping_mul(0x94D049BB133111EB)
        ^ (z as u64).wrapping_mul(0xD6E8FEB86659FD93);
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 29;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn vnoise3(x: f64, y: f64, z: f64) -> f64 {
    let xi = x.floor();
    let yi = y.floor();
    let zi = z.floor();
    let sv = |t: f64| t * t * (3.0 - 2.0 * t);
    let fx = sv(x - xi);
    let fy = sv(y - yi);
    let fz = sv(z - zi);
    let (xi, yi, zi) = (xi as i64, yi as i64, zi as i64);
    let n000 = hash3i(xi, yi, zi);
    let n100 = hash3i(xi + 1, yi, zi);
    let n010 = hash3i(xi, yi + 1, zi);
    let n110 = hash3i(xi + 1, yi + 1, zi);
    let n001 = hash3i(xi, yi, zi + 1);
    let n101 = hash3i(xi + 1, yi, zi + 1);
    let n011 = hash3i(xi, yi + 1, zi + 1);
    let n111 = hash3i(xi + 1, yi + 1, zi + 1);
    let x00 = mix(n000, n100, fx);
    let x10 = mix(n010, n110, fx);
    let x01 = mix(n001, n101, fx);
    let x11 = mix(n011, n111, fx);
    mix(mix(x00, x10, fy), mix(x01, x11, fy), fz)
}

fn fbm3(x: f64, y: f64, z: f64) -> f64 {
    let mut a = 0.5;
    let mut f = 1.0;
    let mut s = 0.0;
    let mut n = 0.0;
    for _ in 0..4 {
        s += a * vnoise3(x * f, y * f, z * f);
        n += a;
        a *= 0.52;
        f *= 2.05;
    }
    s / n
}

// -------------------------------------------------------------------- V3

#[derive(Copy, Clone, Debug)]
struct V3 {
    x: f64,
    y: f64,
    z: f64,
}

impl V3 {
    const fn new(x: f64, y: f64, z: f64) -> V3 {
        V3 { x, y, z }
    }
    fn dot(self, o: V3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: V3) -> V3 {
        V3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn norm(self) -> V3 {
        let l = self.len();
        V3::new(self.x / l, self.y / l, self.z / l)
    }
}

impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl std::ops::Mul<f64> for V3 {
    type Output = V3;
    fn mul(self, k: f64) -> V3 {
        V3::new(self.x * k, self.y * k, self.z * k)
    }
}

// ---------------------------------------------------------------- camera

#[derive(Copy, Clone)]
struct Cam {
    p: V3,
    f: V3,
    r: V3,
    u: V3,
}

impl Cam {
    /// Camera on a circle around the hole, slightly above the disk plane,
    /// always looking at the hole. `orbit` turns slowly with time.
    fn new(orbit: f64, tilt: f64) -> Cam {
        let st = tilt.sin();
        let ct = tilt.cos();
        let p = V3::new(ct * orbit.cos(), st, ct * orbit.sin()) * CAM_D;
        let f = (p * -1.0).norm();
        let up0 = V3::new(0.0, 1.0, 0.0);
        let r = f.cross(up0).norm();
        let u = r.cross(f);
        Cam { p, f, r, u }
    }

    /// Screen-space ray through pixel (x, y).
    ///
    /// `nx` runs -1..+1 left to right, `ny` runs +1..-1 top to bottom, so the
    /// aim direction `f` (which points at the hole) lands exactly in the middle
    /// of the frame: the black hole sits at the vertical centre of the screen.
    /// `shift` moves the whole picture up/down in units of half the frame height.
    fn ray(&self, x: usize, y: usize, w: usize, h: usize, zoom: f64, shift: f64) -> V3 {
        let nx = ((x as f64 + 0.5) / w as f64 - 0.5) * 2.0;
        let ny = (0.5 - (y as f64 + 0.5) / h as f64) * 2.0 + shift;
        let asp = w as f64 / h as f64;
        let s = VIEW / zoom;
        (self.f + self.r * (nx * s * asp) + self.u * (ny * s)).norm()
    }
}

/// Schwarzschild "force" on a photon (exact for the shape of the orbit).
#[inline]
fn accel(p: V3, h2: f64) -> V3 {
    let r2 = p.len2();
    let r = r2.sqrt();
    let r5 = r2 * r2 * r; // |p|^5
    p * (-1.5 * h2 / r5)
}

impl V3 {
    fn len2(self) -> f64 {
        self.dot(self)
    }
}

// --------------------------------------------------------- sampled tables
//
// Three per-frame transcendental costs dominated the profile, and all three
// are smooth enough to pre-sample:
//  - the disk turbulence is fbm3 over (cos(phi), sin(phi), nz(rr)) - a 2D
//    field in (phi, radius), rasterised once into a texture;
//  - the tone curve (1 - e^-x)^(1/1.85) has infinite slope at 0; a larger
//    uniform table keeps the interpolation error below terminal resolution
//    while removing the per-channel sqrt;
//  - the star twinkle needs one sin per sky pixel - a 7th-order polynomial
//    is indistinguishable at the 0.18 amplitude it modulates.

const TURB_K: f64 = 2.1; // turbulence noise frequency (see disk_em / shade)
const TAU: f64 = std::f64::consts::TAU;
const TURB_PHI: usize = 1024;
const TURB_RR: usize = 512;
static TURB_TEX: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();

/// Fill the turbulence texture once. About half a million noise lookups -
/// a tenth of a second single-threaded, a few milliseconds across cores,
/// and afterwards every frame samples it instead of recomputing fbm3. The
/// caller supplies the worker limit so `--threads 1` stays serial end-to-end.
fn build_turb_tex(nthreads: usize) {
    let mut tex = vec![0.0f64; TURB_RR * TURB_PHI];
    let rows_per = TURB_RR.div_ceil(nthreads);
    thread::scope(|sc| {
        for (n, band) in tex.chunks_mut(rows_per * TURB_PHI).enumerate() {
            sc.spawn(move || {
                for (r, row) in band.chunks_mut(TURB_PHI).enumerate() {
                    let rr = R_IN + (n * rows_per + r) as f64 / TURB_RR as f64 * (R_OUT - R_IN);
                    let nz = rr * 0.55 + (rr * 0.21).sin() * 0.8;
                    for (i, v) in row.iter_mut().enumerate() {
                        let phi = (i as f64 + 0.5) / TURB_PHI as f64 * TAU;
                        *v = fbm3(phi.cos() * TURB_K, phi.sin() * TURB_K, nz);
                    }
                }
            });
        }
    });
    let _ = TURB_TEX.set(tex);
}

/// Bilinear sample of the turbulence at pattern angle `phi` (any range) and
/// disk radius `rr`. This is what a frame pays per disk crossing.
#[inline]
fn turb(phi: f64, rr: f64) -> f64 {
    let tex = TURB_TEX.get().expect("turbulence texture");
    let mut u = (phi * (TURB_PHI as f64 / TAU)).fract();
    if u < 0.0 {
        u += 1.0;
    }
    let v = ((rr - R_IN) / (R_OUT - R_IN)).clamp(0.0, 1.0) * TURB_RR as f64;
    let fu = u * TURB_PHI as f64;
    let iu = fu as usize % TURB_PHI;
    let au = fu - fu.floor();
    let iv = (v as usize).min(TURB_RR - 1);
    let av = v - v.floor();
    let iv1 = (iv + 1).min(TURB_RR - 1);
    let iu1 = (iu + 1) % TURB_PHI;
    let a = tex[iv * TURB_PHI + iu] + (tex[iv * TURB_PHI + iu1] - tex[iv * TURB_PHI + iu]) * au;
    let b = tex[iv1 * TURB_PHI + iu] + (tex[iv1 * TURB_PHI + iu1] - tex[iv1 * TURB_PHI + iu]) * au;
    a + (b - a) * av
}

/// tone curve, sampled on a uniform x grid. The larger table removes a sqrt
/// from every shaded disk channel while retaining terminal-level precision.
const TONE_N: usize = 16_384;
const TONE_MAX: f64 = 16.0;
static TONE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();

fn build_tone() {
    let t: Vec<f64> = (0..=TONE_N)
        .map(|i| {
            let x = TONE_MAX * i as f64 / TONE_N as f64;
            (1.0 - (-x).exp()).powf(1.0 / 1.85)
        })
        .collect();
    let _ = TONE.set(t);
}

#[inline]
fn tone(x: f64) -> f64 {
    let t = TONE.get().expect("tone table");
    let s = (x.max(0.0) / TONE_MAX) * TONE_N as f64;
    let i = (s as usize).min(TONE_N - 1);
    let a = s - i as f64;
    t[i] + (t[i + 1] - t[i]) * a
}

/// sin to 1e-4 - all the twinkle needs (it modulates brightness by +-18%)
#[inline]
fn tw_sin(x: f64) -> f64 {
    let q = (x / std::f64::consts::PI).round();
    let y = x - q * std::f64::consts::PI;
    let s = if (q as i64) & 1 == 0 { 1.0 } else { -1.0 };
    let y2 = y * y;
    s * y * (1.0 - y2 * (1.0 / 6.0 - y2 * (1.0 / 120.0 - y2 * (1.0 / 5040.0))))
}

// ------------------------------------------------------------ the disk
//
#[inline]
fn local_orbit_beta(rr: f64) -> (f64, f64) {
    let beta2 = (0.5 * RS) / (rr - RS);
    (beta2.sqrt().min(0.85), beta2)
}

/// Time-independent part of the disk emission at one crossing: colour times
/// intensity with the turbulence streak factored out, because the streak is
/// the only thing that still moves once the geodesics are traced. Returns the
/// emission and the angular drift rate of the pattern at this radius, so a
/// frame only pays for one noise lookup per crossing.
fn disk_em(rr: f64, hp: V3, vd: V3) -> ([f64; 3], f64) {
    let u = (rr - R_IN) / (R_OUT - R_IN); // 0 at the inner rim, 1 at the outer edge

    let edge_in = smoothstep(0.0, 0.05, u); // hot, sharp inner rim
    let edge_out = 1.0 - smoothstep(0.72, 1.0, u); // cool, ragged outer edge
    let rad = (R_IN / rr).powf(1.55) * edge_in * edge_out;

    // local Keplerian speed seen by a static observer: v^2 = M/(r-2M), M=RS/2
    let rs_r = RS / rr;
    let (beta, beta2) = local_orbit_beta(rr);
    let gamma = 1.0 / (1.0 - beta2).max(1e-3).sqrt();

    // prograde orbital direction
    let d = (hp.x * hp.x + hp.z * hp.z).sqrt().max(1e-9);
    let bvec = V3::new(-hp.z / d, 0.0, hp.x / d) * beta;

    // g = (gravitational redshift) / (Doppler)
    let g = (1.0 - rs_r).sqrt() / (gamma * (1.0 + bvec.dot(vd)));
    let g = g.clamp(0.05, 4.0);

    let temp = ((R_IN / rr).powf(0.72) * (0.55 + 0.55 * g)).clamp(0.0, 1.0);
    // g^3 beaming, softened by an emissivity floor: a fully beaming disk leaves
    // the receding half of the frame empty, which reads as a bug, not as physics.
    let inten = rad * (0.35 + 0.65 * g * g * g) * BRIGHT;
    let c = heat(temp);
    let om = 1.6 * (R_IN / rr).powf(1.5); // pattern drift rate at this radius
    ([c[0] * inten, c[1] * inten, c[2] * inten], om)
}

// ------------------------------------------------------------- deep space

/// Deep space, split the same way as the disk: everything is computed once
/// per ray and only the twinkle phase is left for the frame. Display-referred
/// (0..1) on purpose - the tone curve below must not lift the void, or the
/// hole drowns in grey noise.
#[derive(Clone, Copy)]
struct Sky {
    /// stars at twinkle = 1 (tint already folded in)
    star: [f64; 3],
    /// the galactic band, which does not twinkle at all
    band: [f64; 3],
    /// twinkle = 0.82 + 0.18 sin(t * freq + phase), taken from the brightest
    /// contributing star cell
    freq: f64,
    phase: f64,
}

/// The galactic band weight exp(-(|y|·2.6)^1.6·1.4)·0.01 as a function of
/// |d.y|. It is a smooth 1D curve, so it is sampled once into a small table:
/// the powf+exp pair cost more than everything else in the sky lookup, and
/// with an orbiting camera the lookup runs for every sky pixel every frame.
static BAND_LUT: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
fn band_w(ay: f64) -> f64 {
    let t = BAND_LUT.get_or_init(|| {
        (0..=512)
            .map(|i| {
                let y = i as f64 / 512.0;
                (-((y * 2.6).powf(1.6)) * 1.4).exp() * 0.010
            })
            .collect()
    });
    let f = (ay * 512.0).clamp(0.0, 511.999);
    let i = f as usize;
    let a = f - i as f64;
    t[i] * (1.0 - a) + t[i + 1] * a
}

const BAND_FACE_N: usize = 128;
static BAND_TEX: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

#[cfg(test)]
fn band_exact(d: V3) -> f64 {
    let band = if d.y.abs() < 0.75 {
        band_w(d.y.abs())
    } else {
        0.0
    };
    if band > 1e-3 {
        band * band_noise(d)
    } else {
        0.0
    }
}

#[inline]
fn band_noise(d: V3) -> f64 {
    0.4 + 1.2 * fbm3(d.x * 3.0 + 11.0, d.y * 3.0, d.z * 3.0 - 7.0)
}

#[inline]
fn band_face(d: V3) -> (usize, f64, f64) {
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    if ax >= ay && ax >= az {
        (usize::from(d.x < 0.0), d.y / ax, d.z / ax)
    } else if ay >= az {
        (2 + usize::from(d.y < 0.0), d.x / ay, d.z / ay)
    } else {
        (4 + usize::from(d.z < 0.0), d.x / az, d.y / az)
    }
}

fn build_band_tex() -> Vec<f32> {
    let side = BAND_FACE_N + 1;
    let mut tex = vec![0.0; 6 * side * side];
    for face in 0..6 {
        let sign = if face & 1 == 0 { 1.0 } else { -1.0 };
        for y in 0..side {
            let v = y as f64 / BAND_FACE_N as f64 * 2.0 - 1.0;
            for x in 0..side {
                let u = x as f64 / BAND_FACE_N as f64 * 2.0 - 1.0;
                let d = match face / 2 {
                    0 => V3::new(sign, u, v),
                    1 => V3::new(u, sign, v),
                    _ => V3::new(u, v, sign),
                }
                .norm();
                tex[face * side * side + y * side + x] = band_noise(d) as f32;
            }
        }
    }
    tex
}

#[inline]
fn band_sample(d: V3) -> f64 {
    let band = if d.y.abs() < 0.75 {
        band_w(d.y.abs())
    } else {
        0.0
    };
    if band <= 1e-3 {
        return 0.0;
    }
    let tex = BAND_TEX.get_or_init(build_band_tex);
    let (face, u, v) = band_face(d);
    let side = BAND_FACE_N + 1;
    let fx = ((u + 1.0) * 0.5 * BAND_FACE_N as f64).clamp(0.0, BAND_FACE_N as f64);
    let fy = ((v + 1.0) * 0.5 * BAND_FACE_N as f64).clamp(0.0, BAND_FACE_N as f64);
    let x = (fx as usize).min(BAND_FACE_N - 1);
    let y = (fy as usize).min(BAND_FACE_N - 1);
    let (tx, ty) = (fx - x as f64, fy - y as f64);
    let i = face * side * side + y * side + x;
    let a = tex[i] as f64 + (tex[i + 1] as f64 - tex[i] as f64) * tx;
    let b = tex[i + side] as f64 + (tex[i + side + 1] as f64 - tex[i + side] as f64) * tx;
    band * (a + (b - a) * ty)
}

/// Star contribution (at twinkle = 1) and its twinkle parameters.
fn star_layer(d: V3, ppc0: f64) -> ([f64; 3], f64, f64) {
    let mut lum = 0.0;
    let mut tint = 0.0;
    let mut wmax = 0.0f64;
    let mut freq = 2.0;
    let mut phase = 0.0;
    for k in 0..3 {
        let p = d * STAR_SCALE[k as usize];
        let ci = [p.x.round() as i64, p.y.round() as i64, p.z.round() as i64];
        let h0 = hash3i(ci[0], ci[1], ci[2]);
        if h0 < STAR_DENSITY {
            continue; // empty cell
        }
        let dx = p.x - (ci[0] as f64 + (hash3i(ci[0] + 7, ci[1], ci[2]) - 0.5) * 0.6);
        let dy = p.y - (ci[1] as f64 + (hash3i(ci[0], ci[1] + 7, ci[2]) - 0.5) * 0.6);
        let dz = p.z - (ci[2] as f64 + (hash3i(ci[0], ci[1] + 7, ci[2] + 7) - 0.5) * 0.6);
        let r2 = dx * dx + dy * dy + dz * dz;
        // Gaussian point spread sized to the sampling pitch: `ppc0` is how many
        // ray pixels one layer-0 cell spans, so sigma in cells makes the core
        // ~STAR_CORE pixels wide. A fixed-angle falloff instead renders as soft
        // hand-sized blobs at device resolution - a film over the sky, not
        // stars - and buries the encoder in near-black colour strips.
        let ppc = ppc0 * STAR_SCALE[0] / STAR_SCALE[k as usize];
        let sigma = (STAR_CORE / ppc).clamp(0.02, 0.30);
        let s2 = r2 / (2.0 * sigma * sigma);
        if s2 > 30.0 {
            continue; // far out in the Gaussian tail
        }
        let f2 = (-s2).exp();
        let w = f2 * STAR_BRI[k as usize];
        lum += w * (0.4 + 0.6 * h0);
        tint += f2 * hash3i(ci[0] + 3, ci[1] + 11, ci[2] + 5);
        if w > wmax {
            // this layer dominates: its twinkle drives the whole pixel
            wmax = w;
            freq = 1.5 + h0 * 2.0;
            phase = h0 * 40.0;
        }
    }
    (
        [lum * (0.85 + 0.4 * tint), lum * (0.88 + 0.3 * tint), lum],
        freq,
        phase,
    )
}

fn stars(d: V3, ppc0: f64) -> Sky {
    let (star, freq, phase) = star_layer(d, ppc0);
    // World-fixed galactic band. A small cube map keeps orbiting-camera
    // lookups cheap while preserving its longitudinal fbm texture.
    let g = band_sample(d);
    Sky {
        star,
        band: [g * 0.7, g * 0.8, g],
        freq,
        phase,
    }
}

// --------------------------------------------------------- ray tracing
//
// Split into an expensive geometry pass and a cheap shading pass.
//
// The geometry - integrating the null geodesics - does not depend on time
// at all: with a still camera the ray paths are identical frame after frame.
// What changes with t is only the shading: the disk's turbulence drifts and
// the stars twinkle. So trace once per camera setup, keep for every pixel
// the plane crossings and the escape direction, and let each animation frame
// re-evaluate only the emission. Video codecs call these I-frames and
// P-frames; here a P-frame costs a fraction of the I-frame.

// ------------------------------------------------------------- infall star
// A star on its way through the disk and into the hole. The orbit is
// Newtonian in a Paczynski-Wiita pseudo-potential (a = -GM p / ((r-RS)^2 r)),
// which puts its innermost stable circle at 3 RS - the same radius the
// disk's inner rim sits at - so the plunge happens where it should. A
// whisper of drag eats angular momentum and turns the fly-by into an
// inspiral. The picture is a string of glowing points (the head, a cooling
// trail, and a remnant that lingers by the photon sphere): every traced ray
// picks up emission where it passes close to one, exactly like the disk
// crossings.

/// gravitational parameter of the pseudo-potential; v_circ(12) ~ 3.5, so a
/// whole orbit at the spawn radius takes a few tens of seconds on screen
const INFALL_GM: f64 = 120.0;
/// drag per unit time - the reason the orbit decays instead of holding
const INFALL_DRAG: f64 = 0.0045;
/// spawn radius range
const INFALL_R0: (f64, f64) = (11.0, 15.0);
/// spawn speed as a fraction of the local circular velocity (below 1: the
/// star arrives already doomed)
const INFALL_F0: (f64, f64) = (0.60, 0.85);
/// spawn inclination range, radians off the disk plane
const INFALL_INC: (f64, f64) = (0.15, 0.45);
/// below this radius the star is gone
const INFALL_SWALLOW: f64 = RS * 1.15;
/// the remnant a swallowed star leaves fades over about this much time
const INFALL_FADE: f64 = 2.6;
/// the trail drops a point whenever the head has moved this far
const INFALL_TRAIL_STEP: f64 = 1.1;
/// trail length in points
const INFALL_TRAIL_N: usize = 14;
/// Shed material loses angular momentum and feels more drag than the intact
/// star. That makes the old end of the trail spiral into the hole instead of
/// remaining frozen along the orbit the head has already travelled.
const INFALL_TRAIL_TANGENTIAL: f64 = 0.90;
const INFALL_TRAIL_DRAG: f64 = 0.030;
/// at most this many stars at once
const INFALL_MAX: usize = 3;
/// gaussian radii of head / trail point / remnant, in scene units
const INFALL_SIG: f64 = 0.85;
const INFALL_TRAIL_SIG: f64 = 0.38;
const INFALL_REM_SIG: f64 = 0.5;
/// emission weights of head / trail / remnant (soft-clipped when shaded)
const INFALL_HEAD_BRI: f64 = 15.0;
const INFALL_TRAIL_BRI: f64 = 3.2;
const INFALL_REM_BRI: f64 = 10.0;
/// the massive star (3x the hole): spawns far out on a near-circular
/// orbit and barely feels the drag, so the infall takes its time
const BIG_R0: (f64, f64) = (24.0, 33.0);
const BIG_F0: (f64, f64) = (0.58, 0.70);
const BIG_DRAG: f64 = INFALL_DRAG * 0.25;
/// the superstar: a parked giant several times the massive star's size,
/// held dead-still in the world frame while its envelope pours into the
/// hole through one long funnel. The default and spiral profiles keep a
/// coherent arc; the tidal profile gives each debris parcel a small spread
/// in energy and angular momentum, so it reads as a continuous sheared stream
/// rather than a row of projectiles. The frame never rotates: only the flow
/// inside the funnel moves, and the drain plays out over tens of minutes.
const SUPER_SC: f64 = 6.0;
/// where the donor sits (world frame): on the far side of the hole and
/// well off to the side of the view axis, so its glow stays a compact
/// blob at the frame's edge while the funnel sweeps across the picture
const SUPER_PARK: V3 = V3::new(-17.1, 2.0, -19.2);
/// |SUPER_PARK|, spelled out for the funnel's width taper
const SUPER_PARK_R: f64 = 25.8;
/// envelope mass bled into the funnel per second: at this rate the star
/// still keeps most of itself after ten minutes of watching
const SUPER_SHED_RATE: f64 = 0.00045;
/// The simulation keeps a tiny numerical core, but it is no longer rendered
/// as a star once the donor has been drained.
const SUPER_MIN_MASS: f64 = 0.05;
/// mass (and glow weight) of one shed parcel, and how many may be in
/// flight at once. Smaller parcels make the visible ribbon denser without
/// increasing the number of live glow deposits (they are grouped below).
const SUPER_SHED_W: f64 = 0.0002;
const SUPER_STREAM_MAX: usize = 160;
/// Number of neighbouring funnel parcels represented by one deposited glow.
/// Keeping this small preserves the shape while avoiding a full-grid deposit
/// for every microscopic parcel.
const SUPER_STREAM_GROUP: usize = 4;
/// The tidal profile is a debris stream, not a set of luminous bullets. Its
/// sheared launch distribution and bounded grouping keep the mass distribution
/// visually continuous while retaining the same total transfer rate.
const TIDAL_SHED_W: f64 = 0.0002;
const TIDAL_STREAM_MAX: usize = 160;
const TIDAL_STREAM_GROUP: usize = 4;
const TIDAL_STREAM_BRI: f64 = 260.0;
/// A grouped tidal glow is only a good approximation while its parcels are
/// near one another. Once they occupy separate orbital branches, gradually
/// replace it with individual deposits so that one fading parcel cannot
/// teleport the group's weighted centroid to another branch.
const TIDAL_GROUP_SPLIT_START: f64 = 2.0;
const TIDAL_GROUP_MAX_SPREAD: f64 = 3.0;
/// Guard the approximation against widely separated/unstable parcels. A
/// funnel glow larger than this would illuminate most of the cached ray grid
/// and both wash out the image and destroy the frame time.
const SUPER_STREAM_SIG_MAX: f64 = 2.4;
/// A Gaussian contribution this far from a glow is below terminal colour
/// quantisation for stream-sized sources. Bright stellar glows use a wider
/// safety margin; the exact integrator remains in use for every retained
/// segment.
const GLOW_CUTOFF_DIM: f64 = 4.0;
const GLOW_CUTOFF_BRIGHT: f64 = 5.0;
/// The traced adaptive step is at most 1.1 scene units; this conservative
/// half-length keeps the midpoint reject from excluding any retained segment.
const GLOW_SEG_HALF_MAX: f64 = 1.0;
/// how the funnel leaves the donor: mostly tangential (perpendicular to
/// the star-hole line), a touch toward the hole - enough sideways speed
/// to swing a long arc around, little enough to keep diving inward
const SUPER_FUN_F: f64 = 0.90;
const SUPER_FUN_G: f64 = 0.10;
/// the bleed is diffuse shock-heated gas rather than a compact clump: it
/// couples to the field (drag shrinks the arc wrap by wrap until it
/// pours past the horizon) and shines far brighter per unit mass, which
/// is the only way a deliberately slow drain stays visible at all
const SUPER_STREAM_DRAG: f64 = 0.010;
// The default and spiral streams are rendered as paired, enlarged glows
// (see `glow_list`), so this is brightness per pair rather than brightness
// of an individual dot. The tidal profile supplies its own dimmer value.
const SUPER_STREAM_BRI: f64 = 420.0;
/// the funnel's gaussian radius at the spout (by the hole) and at the
/// throat (by the star)
const SUPER_SIG_TIP: f64 = 0.5;
const SUPER_SIG_ROOT: f64 = 1.25;
/// the donor's near-side envelope is stretched toward the hole as a smooth
/// lobe before the moving parcels take over
const SUPER_STRETCH_LEN: f64 = 6.5;
const SUPER_STRETCH_N: usize = 4;
/// how fast the massive star is torn apart inside its tidal radius
const INFALL_STRIP: f64 = 2.5;
/// how long a normal shed stream particle keeps glowing
const STREAM_LIFE: f64 = 6.0;
/// Superstar parcels are material in the persistent funnel, not fading
/// after-images: they stay until they cross the horizon. The test uses its
/// own finite horizon below.
const SUPER_STREAM_LIFE: f64 = f64::INFINITY;
/// stream particles alive at once, tops
const STREAM_MAX: usize = 24;
const STREAM_BRI: f64 = 6.0;
const STREAM_SIG: f64 = 0.45;
const MATTER_STEP: f64 = 0.02;

/// One glowing point of the stream, in the trace (camera-azimuth-0) frame,
/// its colour already weighted: head hot and bright, trail dim and red,
/// remnant fading. `sig` is the gaussian radius in scene units.
#[derive(Clone, Copy)]
struct Glow {
    p: V3,
    c: [f64; 3],
    sig: f64,
    /// Funnel groups are broad, low-contrast render primitives. They use a
    /// midpoint quadrature fast path; stellar heads/trails retain the exact
    /// finite-segment integral below.
    fast: bool,
}

/// Every glow lives at radius <= BIG_R0.1 and reaches at most 3.5 sigmas
/// times the largest star scale (3x for the massive star; the superstar's
/// glow radius is capped at the same size, see `glow_list`) plus the
/// shell's radial thickness past it, so a segment whose midpoint is
/// inside this radius is worth recording for the deposition to chew on.
const GLOW_R: f64 = BIG_R0.1 + 3.5 * INFALL_SIG * 3.0 + 0.06 * BIG_R0.1;

/// One recorded trace segment inside the glow shell, pre-rotated into the
/// glow's own coordinate frame: the midpoint `m` (the position the bin index
/// and the tail cutoff work on), the unit axis `u`, the length `dl` and the
/// midpoint's squared radius `r2` for the cheap radial pre-reject, the
/// transmittance that was in effect when it was traced, and the pixel it
/// belongs to. `sub` is the 4x4x4 fast-field cell, filled after binning, so
/// the per-frame replay does not redo three coordinate divisions. Everything
/// a glow needs about the segment is already here, so
/// the hot deposition loop is pure arithmetic: no square root, no division,
/// no re-derivation of the axis from the endpoints.
///
/// Storing these is what lets the glows move every frame without
/// re-integrating the geodesics: the deposition just replays the gaussian
/// sums over the bins it touches. `u` and `dl` are rounded to f32 at record
/// time; the resulting perpendicular-distance error is of order 1e-7
/// relative, the same magnitude as the f32 endpoint quantization the record
/// always had, and far below the display quantisation the cutoffs keep.
#[derive(Clone, Copy)]
struct Seg {
    m: [f32; 3],
    u: [f32; 3],
    dl: f32,
    /// squared midpoint radius, for the cheap radial pre-reject
    r2: f32,
    tr: f32,
    px: u32,
    sub: u8,
}

impl Seg {
    /// Record one step of a ray as a glow segment, unless it is degenerate
    /// or lies entirely outside the shell any glow can light.
    fn from_endpoints(p0: V3, p1: V3, tr: f64, px: usize) -> Option<Seg> {
        let sg = p1 - p0;
        let dl = sg.len();
        if dl <= 1e-12 {
            return None;
        }
        // one reciprocal instead of three divisions: the axis is rounded to
        // f32 right after anyway
        let inv_dl = dl.recip();
        let m = p0 + sg * 0.5;
        let rm = m.len();
        if rm >= GLOW_R {
            return None;
        }
        Some(Seg {
            m: [m.x as f32, m.y as f32, m.z as f32],
            u: [
                (sg.x * inv_dl) as f32,
                (sg.y * inv_dl) as f32,
                (sg.z * inv_dl) as f32,
            ],
            dl: dl as f32,
            r2: (rm * rm) as f32,
            tr: tr as f32,
            px: px as u32,
            sub: 0,
        })
    }
}

/// The axial factor of a segment's glow integral is separable: it is the
/// sum of one 1D function evaluated at the glow's axial distances from
/// the segment's two endpoints (see `segment_glow_weight`). So a single
/// small table of sqrt(pi/2) * erf(x / sqrt 2) with linear interpolation
/// covers every (dl, along) pair exactly - no clamped length axis, no zero
/// cutoff, no nearest-neighbour steps like the old 2D grid - and at 16 KB
/// it stays resident in L1 instead of streaming from L2.
const ERF_N: usize = 2048;
const ERF_X: f64 = 7.0; // erf(7/sqrt 2) = 1 - 2e-12: saturated
static ERF_LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

/// The field setup evaluates the same positive Gaussian exponent at every
/// active bin/source/subcell. A small linear table is faster than rebuilding
/// the degree-six exp polynomial for each setup sample; the stored f32 field
/// is already much coarser than this interpolation error.
const FAST_EXP_N: usize = 8192;
const FAST_EXP_X: f64 = 12.0;
static FAST_EXP_LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn build_fast_exp_lut() -> Vec<f32> {
    (0..=FAST_EXP_N)
        .map(|i| (-(i as f64 * FAST_EXP_X / FAST_EXP_N as f64)).exp() as f32)
        .collect()
}

#[inline(always)]
fn fast_exp_lut(x: f64, table: &[f32]) -> f32 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= FAST_EXP_X {
        return 0.0;
    }
    let f = x * FAST_EXP_N as f64 / FAST_EXP_X;
    let i = f as usize;
    let lo = table[i];
    lo + (table[i + 1] - lo) * (f - i as f64) as f32
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let p = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - p * (-x * x).exp())
}

fn build_erf_lut() -> Vec<f32> {
    (0..=ERF_N)
        .map(|i| {
            let x = i as f64 / ERF_N as f64 * ERF_X;
            ((std::f64::consts::PI / 2.0).sqrt() * erf_approx(x / std::f64::consts::SQRT_2)) as f32
        })
        .collect()
}

/// e^(-x) for x >= 0 without the libm call: reduce to 2^(-x/ln 2), split
/// into an integer part (folded straight into the exponent bits) and a
/// reduced argument in [-ln2/2, ln2/2], and evaluate exp with the Taylor
/// polynomial of degree 6. The relative error stays below 2e-7 - two orders
/// under the display quantisation the glow cutoffs already enforce - and
/// the hot deposition loop saves the indirect call per touched segment.
#[inline(always)]
fn fast_exp_neg(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let y = -x * std::f64::consts::LOG2_E;
    let n = (y + 0.5).floor() as i32;
    let z = (y - n as f64) * std::f64::consts::LN_2;
    let p = 1.0
        + z * (1.0
            + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z * (1.0 / 120.0 + z / 720.0)))));
    p * f64::from_bits(((n + 1023) as u64) << 52)
}

/// sqrt(pi/2) * erf(x / sqrt 2), odd in x, linear between nodes. Past
/// ERF_X the remaining tail is below 2e-12, so the saturated endpoint is
/// exact to well below the table's own resolution.
#[inline]
fn axial_end(x: f64, t: &[f32]) -> f64 {
    let a = x.abs();
    let v = if a >= ERF_X {
        (std::f64::consts::PI / 2.0).sqrt()
    } else {
        let f = a * (ERF_N as f64 / ERF_X);
        let i = (f as usize).min(ERF_N - 1);
        let lo = t[i] as f64;
        lo + (t[i + 1] as f64 - lo) * (f - i as f64)
    };
    if x < 0.0 {
        -v
    } else {
        v
    }
}

/// The exact line integral of the glow's gaussian exp(-d^2 / 2 sig^2)
/// along the segment, times the transmittance in effect when it was
/// traced: split the glow offset into its axial and perpendicular parts
/// (relative to the midpoint, which the tail cutoff already needs, so the
/// two share every product), keep the perpendicular factor as-is and
/// integrate the axial gaussian in closed form - erf at the distances from
/// both endpoints. The segment's stored unit axis and length make this
/// root-free and division-free.
/// Reference form of the exact integral, used by the quadrature test; the
/// deposition loop calls `segment_glow_weight_at` with the per-glow
/// constants it already has.
#[cfg(test)]
#[inline]
fn segment_glow_weight(gl: &Glow, s: &Seg, erf_t: &[f32]) -> f64 {
    let inv = gl.sig.recip();
    segment_glow_weight_at(
        gl,
        s,
        erf_t,
        inv,
        inv * inv * 0.5,
        gl.p.x - s.m[0] as f64,
        gl.p.y - s.m[1] as f64,
        gl.p.z - s.m[2] as f64,
    )
}

/// The exact line integral above, with the per-glow constants and the
/// midpoint offset already computed by the caller (the tail cutoff needs
/// the same products, so nothing is evaluated twice).
#[cfg(test)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn segment_glow_weight_at(
    gl: &Glow,
    s: &Seg,
    erf_t: &[f32],
    inv: f64,
    inv2h: f64,
    dmx: f64,
    dmy: f64,
    dmz: f64,
) -> f64 {
    let dm2 = dmx * dmx + dmy * dmy + dmz * dmz;
    let q = dmx * s.u[0] as f64 + dmy * s.u[1] as f64 + dmz * s.u[2] as f64;
    let perp2 = (dm2 - q * q).max(0.0);
    let dl = s.dl as f64;
    let along = q + 0.5 * dl;
    let axial = axial_end(along * inv, erf_t) + axial_end((dl - along) * inv, erf_t);
    fast_exp_neg(perp2 * inv2h) * gl.sig * axial * s.tr as f64
}

/// Fast quadrature for the broad, low-contrast funnel groups. Their Gaussian
/// radius is comparable to a traced segment, so sampling the segment midpoint
/// is sufficient at terminal precision and avoids the two axial LUT lookups.
/// The midpoint offset and the folded inverse variance are exactly what the
/// tail cutoff and the exact path already computed.
#[inline(always)]
fn segment_glow_midpoint(dm2: f64, inv2h: f64, s: &Seg) -> f64 {
    fast_exp_neg(dm2 * inv2h) * s.dl as f64 * s.tr as f64
}

/// Constants used while replaying one glow over the static segment arena.
/// Keeping these separate from `Glow` makes the segment-centric hot loop
/// independent of the per-frame setup work (radius, cutoff and reciprocals).
#[derive(Clone, Copy)]
struct GlowParams {
    p: [f64; 3],
    c: [f64; 3],
    sig: f64,
    inv: f64,
    inv2h: f64,
    inner2: f64,
    outer2: f64,
    reach2: f64,
    range: isize,
    fast: bool,
}

impl GlowParams {
    fn new(gl: &Glow) -> GlowParams {
        let gr = gl.p.len();
        let rej = gl.sig * 3.5 + 0.06 * gr;
        let peak = gl.c[0].max(gl.c[1]).max(gl.c[2]);
        let cutoff = if peak < 1.0 {
            GLOW_CUTOFF_DIM
        } else {
            GLOW_CUTOFF_BRIGHT
        };
        let inner = (gr - rej - 1e-4).max(0.0);
        let outer = gr + rej + 1e-4;
        let reach = cutoff * gl.sig + GLOW_SEG_HALF_MAX;
        let inv = gl.sig.recip();
        GlowParams {
            p: [gl.p.x, gl.p.y, gl.p.z],
            c: gl.c,
            sig: gl.sig,
            inv,
            inv2h: inv * inv * 0.5,
            inner2: inner * inner,
            outer2: outer * outer,
            reach2: reach * reach,
            range: (rej / BIN_W).ceil() as isize + 1,
            fast: gl.fast,
        }
    }
}

/// The exact finite-segment integral for the segment-centric deposition path.
/// The midpoint offset and all glow constants are shared with the tail test,
/// so the hot loop does not repeat setup work for a segment/glow pair.
#[inline(always)]
fn segment_glow_weight_params(
    gl: &GlowParams,
    s: &Seg,
    erf_t: &[f32],
    dmx: f64,
    dmy: f64,
    dmz: f64,
) -> f64 {
    let dm2 = dmx * dmx + dmy * dmy + dmz * dmz;
    let q = dmx * s.u[0] as f64 + dmy * s.u[1] as f64 + dmz * s.u[2] as f64;
    let perp2 = (dm2 - q * q).max(0.0);
    let dl = s.dl as f64;
    let along = q + 0.5 * dl;
    let axial = axial_end(along * gl.inv, erf_t) + axial_end((dl - along) * gl.inv, erf_t);
    fast_exp_neg(perp2 * gl.inv2h) * gl.sig * axial * s.tr as f64
}

/// Add a glow to every conservative bin its tail can reach. The bit mask is
/// the key change from the old glow-major loop: a bin is visited once in the
/// frame even when many stream glows overlap it.
fn mark_glow_bins(gl: &GlowParams, bit: u64, bin_glows: &mut [u64], active_bins: &mut Vec<u32>) {
    let cell = |v: f64| (((v / BIN_W).floor() as isize) + BIN_C).clamp(0, BIN_N as isize - 1);
    let (gx, gy, gz) = (cell(gl.p[0]), cell(gl.p[1]), cell(gl.p[2]));
    let last = BIN_N as isize - 1;

    let axis_bounds = |i: isize| {
        if i == 0 || i == last {
            (0.0, GLOW_R * GLOW_R)
        } else {
            let lo = (i - BIN_C) as f64 * BIN_W;
            let hi = lo + BIN_W;
            let lo2 = lo * lo;
            let hi2 = hi * hi;
            let min2 = if lo <= 0.0 && hi >= 0.0 {
                0.0
            } else {
                lo2.min(hi2)
            };
            (min2, lo2.max(hi2))
        }
    };
    let point_axis_min = |i: isize, p: f64| {
        if i == 0 || i == last {
            0.0
        } else {
            let lo = (i - BIN_C) as f64 * BIN_W;
            let hi = lo + BIN_W;
            if p < lo {
                (p - lo) * (p - lo)
            } else if p > hi {
                (p - hi) * (p - hi)
            } else {
                0.0
            }
        }
    };
    let mut xdist = [0.0; BIN_N];
    let mut ydist = [0.0; BIN_N];
    let mut zdist = [0.0; BIN_N];
    for i in 0..BIN_N {
        xdist[i] = point_axis_min(i as isize, gl.p[0]);
        ydist[i] = point_axis_min(i as isize, gl.p[1]);
        zdist[i] = point_axis_min(i as isize, gl.p[2]);
    }

    for z in (gz - gl.range).max(0)..=(gz + gl.range).min(last) {
        let (zmin, zmax) = axis_bounds(z);
        for y in (gy - gl.range).max(0)..=(gy + gl.range).min(last) {
            let (ymin, ymax) = axis_bounds(y);
            for x in (gx - gl.range).max(0)..=(gx + gl.range).min(last) {
                if xdist[x as usize] + ydist[y as usize] + zdist[z as usize] > gl.reach2 {
                    continue;
                }
                let (xmin, xmax) = axis_bounds(x);
                if ymin + zmin + xmin > gl.outer2 || ymax + zmax + xmax < gl.inner2 {
                    continue;
                }
                let b = (x + BIN_N as isize * (y + BIN_N as isize * z)) as usize;
                if bin_glows[b] == 0 {
                    active_bins.push(b as u32);
                }
                bin_glows[b] |= bit;
            }
        }
    }
}

/// Evaluate all glows that can reach one segment. The segment is loaded once,
/// and its pixel accumulator is updated once after all glow contributions have
/// been summed. This removes the random pixel read/modify/write from the
/// innermost glow loop.
#[inline(always)]
fn segment_glow_add(s: &Seg, mask: u64, glows: &[GlowParams], erf_t: &[f32]) -> [f64; 3] {
    let r2 = s.r2 as f64;
    let sx = s.m[0] as f64;
    let sy = s.m[1] as f64;
    let sz = s.m[2] as f64;
    let mut bits = mask;
    let mut add = [0.0; 3];
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let gl = &glows[i];
        if r2 < gl.inner2 || r2 > gl.outer2 {
            continue;
        }
        let dmx = gl.p[0] - sx;
        let dmy = gl.p[1] - sy;
        let dmz = gl.p[2] - sz;
        let dm2 = dmx * dmx + dmy * dmy + dmz * dmz;
        if dm2 > gl.reach2 {
            continue;
        }
        let e = if gl.fast {
            segment_glow_midpoint(dm2, gl.inv2h, s)
        } else {
            segment_glow_weight_params(gl, s, erf_t, dmx, dmy, dmz)
        };
        add[0] += gl.c[0] * e;
        add[1] += gl.c[1] * e;
        add[2] += gl.c[2] * e;
    }
    add
}

/// Process a range of active spatial bins. Each segment belongs to exactly
/// one bin, so partitioning bins gives workers disjoint input without locks.
#[inline(always)]
fn add_fast_pixel(buf: &mut GlowBuf, px: u32, energy: f32) {
    // Fast field values are non-negative. Do not maintain a touched list for
    // this dense path; the reduction scans the small output grid once, which
    // is cheaper than a branch and push for every path segment.
    let dst = &mut buf.fast_px[px as usize];
    // All fast funnel primitives share this fixed shock-heated colour ratio;
    // keep only scalar energy in both the field and the worker buffer.
    *dst += energy;
}

#[inline(always)]
fn add_exact_pixel(buf: &mut GlowBuf, px: u32, add: [f64; 3]) {
    if add == [0.0; 3] {
        return;
    }
    let dst = &mut buf.px[px as usize];
    if *dst == [0.0; 3] {
        buf.touched.push(px);
    }
    dst[0] += add[0];
    dst[1] += add[1];
    dst[2] += add[2];
}

/// Merge one worker's scalar f32 funnel field and exact f64 stellar field into
/// the frame buffer. Keeping the two formats separate makes the hot funnel
/// loop cheap without weakening the exact source kernel's accumulation.
fn reduce_glow_buf(buf: &mut GlowBuf, st: &mut [[f64; 3]], lit: &mut Vec<u32>) {
    for (i, slot) in buf.fast_px.iter_mut().enumerate() {
        let energy = *slot;
        *slot = 0.0;
        if energy == 0.0 {
            continue;
        }
        let dst = &mut st[i];
        if *dst == [0.0; 3] {
            lit.push(i as u32);
        }
        let add = energy as f64;
        dst[0] += add;
        dst[1] += add * 0.75;
        dst[2] += add * 0.45;
    }
    for px in buf.touched.drain(..) {
        let i = px as usize;
        let add = buf.px[i];
        buf.px[i] = [0.0; 3];
        let dst = &mut st[i];
        if *dst == [0.0; 3] {
            lit.push(px);
        }
        dst[0] += add[0];
        dst[1] += add[1];
        dst[2] += add[2];
    }
}

#[allow(clippy::too_many_arguments)]
fn deposit_bin_range(
    active_bins: &[u32],
    begin: usize,
    end: usize,
    bin_field: &[u32],
    bin_fast: &[FastBin],
    fast_segs: &[FastSeg],
    fast_off: &[u32],
    segs: &[Seg],
    bin_off: &[u32],
    glows: &[GlowParams],
    erf_t: &[f32],
    replay_fast: bool,
    buf: &mut GlowBuf,
) {
    for &bin in &active_bins[begin..end] {
        let b = bin as usize;
        let slot = bin_field[b];
        let exact = if slot == u32::MAX {
            0
        } else {
            bin_fast[slot as usize].exact
        };
        if replay_fast && slot != u32::MAX && bin_fast[slot as usize].peak >= FAST_BIN_LOD_MIN {
            let field = &bin_fast[slot as usize].field;
            // The compact arena is built before this kernel whenever a fast
            // glow exists. Keeping the fallback out of the hot loop makes
            // the common sparse-matrix representation branch-free.
            for fs in &fast_segs[fast_off[b] as usize..fast_off[b + 1] as usize] {
                let fast = field[(fs.px_sub & FAST_SUB_MASK) as usize];
                add_fast_pixel(buf, fs.px_sub >> FAST_SUB_BITS, fast * fs.weight);
            }
        }
        if exact != 0 {
            for s in &segs[bin_off[b] as usize..bin_off[b + 1] as usize] {
                add_exact_pixel(buf, s.px, segment_glow_add(s, exact, glows, erf_t));
            }
        }
    }
}

#[inline(always)]
fn fast_subcell(s: &Seg, b: usize) -> usize {
    let bx = b % BIN_N;
    let by = (b / BIN_N) % BIN_N;
    let bz = b / (BIN_N * BIN_N);
    let lo_x = (bx as isize - BIN_C) as f64 * BIN_W;
    let lo_y = (by as isize - BIN_C) as f64 * BIN_W;
    let lo_z = (bz as isize - BIN_C) as f64 * BIN_W;
    let ix = (((s.m[0] as f64 - lo_x) / BIN_W * FAST_SUBDIV as f64) as usize).min(FAST_SUBDIV - 1);
    let iy = (((s.m[1] as f64 - lo_y) / BIN_W * FAST_SUBDIV as f64) as usize).min(FAST_SUBDIV - 1);
    let iz = (((s.m[2] as f64 - lo_z) / BIN_W * FAST_SUBDIV as f64) as usize).min(FAST_SUBDIV - 1);
    ix + FAST_SUBDIV * iy + FAST_SUBDIV * FAST_SUBDIV * iz
}

/// Split active bins by segment count rather than by bin count. Photon-sphere
/// bins are much denser than the outer bins, so equal bin ranges otherwise
/// leave one worker with almost all of the frame's work.
fn split_bin_work(active_bins: &[u32], bin_off: &[u32], workers: usize) -> Vec<(usize, usize)> {
    let total: u64 = active_bins
        .iter()
        .map(|&b| (bin_off[b as usize + 1] - bin_off[b as usize]) as u64)
        .sum();
    let mut ranges = Vec::with_capacity(workers);
    let mut begin = 0;
    let mut remaining = total;
    for worker in 0..workers {
        if begin == active_bins.len() {
            break;
        }
        let left = (workers - worker) as u64;
        let target = remaining.div_ceil(left);
        let start = begin;
        let mut work = 0u64;
        while begin < active_bins.len() && (work < target || begin == start) {
            let b = active_bins[begin] as usize;
            work += (bin_off[b + 1] - bin_off[b]) as u64;
            begin += 1;
        }
        ranges.push((start, begin));
        remaining -= work;
    }
    ranges
}

/// Uniform bins over the cube of side BIN_N * BIN_W centred on the hole -
/// the whole region any glow can light up, out past where the massive
/// star spawns - so a glow only ever meets the segments physically near
/// it instead of all of them.
const BIN_N: usize = 75;
const BIN_W: f64 = 1.2;
/// Fast funnel fields are sampled on a small sub-grid inside each spatial bin.
/// Four samples per axis keep the broad tidal glow smooth without evaluating
/// one Gaussian for every traced segment.
const FAST_SUBDIV: usize = 4;
const FAST_SUBCELLS: usize = FAST_SUBDIV * FAST_SUBDIV * FAST_SUBDIV;
const FAST_SUB_BITS: u32 = 6;
const FAST_SUB_MASK: u32 = (1 << FAST_SUB_BITS) - 1;
/// Importance threshold for the adaptive fast-field LOD experiment. Zero is
/// the reference path; non-zero values drop whole low-energy spatial bins
/// whose contribution cannot normally change a terminal pixel.
const FAST_BIN_LOD_MIN: f32 = 1.0e-8;
/// the cube's centre: the bin index of the coordinate origin
const BIN_C: isize = (BIN_N as isize) / 2;

/// The bin a segment's midpoint falls in, clamped to the cube's edge.
fn bin_of(s: &Seg) -> usize {
    let c = |m: f32| ((((m as f64) / BIN_W).floor() as isize) + BIN_C).clamp(0, BIN_N as isize - 1);
    let x = c(s.m[0]);
    let y = c(s.m[1]);
    let z = c(s.m[2]);
    (x + BIN_N as isize * (y + BIN_N as isize * z)) as usize
}

/// Sort the segments into their bins: count per bin, turn the counts into
/// exclusive prefix offsets, then permute in place - each swap drops one
/// segment into its bin's next free slot, so no second arena is needed.
/// The bin index computed by the counting pass is kept in `bins` so the
/// permutation does not have to derive it again for every element (and for
/// every element swapped in): at ten million segments that is the better
/// part of the sort's cost. The scratch buffer is retained between
/// re-traces so its pages stay warm.
fn build_bins(segs: &mut [Seg], bin_off: &mut Vec<u32>, bins: &mut Vec<u32>, cur: &mut Vec<u32>) {
    bin_off.clear();
    bin_off.resize(BIN_N * BIN_N * BIN_N + 1, 0);
    bins.clear();
    bins.reserve(segs.len());
    for s in segs.iter() {
        let b = bin_of(s) as u32;
        bins.push(b);
        bin_off[b as usize + 1] += 1;
    }
    for i in 1..bin_off.len() {
        bin_off[i] += bin_off[i - 1];
    }
    let mut i = 0;
    // `cur` is a caller-owned scratch buffer: re-allocating the bin-offset
    // copy on every re-trace would fault in the same megabytes again and
    // again for no benefit.
    cur.clear();
    cur.extend_from_slice(&bin_off[..bin_off.len() - 1]);
    while i < segs.len() {
        let b = bins[i] as usize;
        if (i as u32) >= bin_off[b] && (i as u32) < bin_off[b + 1] {
            i += 1;
        } else {
            let t = cur[b] as usize;
            segs.swap(i, t);
            bins.swap(i, t);
            cur[b] += 1;
        }
    }
    // This is an I-frame-only pass. Cache the fast-field subcell while the
    // segment arena is already hot; P-frames can then select it with one byte
    // load instead of recomputing three bin-local coordinate divisions.
    for b in 0..bin_off.len() - 1 {
        for s in &mut segs[bin_off[b] as usize..bin_off[b + 1] as usize] {
            s.sub = fast_subcell(s, b) as u8;
        }
    }
}

/// Copy only the metadata needed by the fast funnel replay into a compact,
/// bin-aligned arena. The full segment records stay cold and are touched only
/// by bins containing exact stellar glows.
fn build_fast_arena(segs: &[Seg], bin_off: &[u32], out: &mut Vec<FastSeg>, off: &mut Vec<u32>) {
    out.clear();
    off.clear();
    off.reserve(bin_off.len());
    off.push(0);
    for b in 0..bin_off.len() - 1 {
        let start = out.len();
        for s in &segs[bin_off[b] as usize..bin_off[b + 1] as usize] {
            out.push(FastSeg {
                px_sub: (s.px << FAST_SUB_BITS) | s.sub as u32,
                weight: s.dl * s.tr,
            });
        }
        // Keep the field vector hot by retaining bin-major traversal, but
        // transpose each bin's scatter list into screen order. The output
        // writes then advance through nearby pixels instead of following the
        // arbitrary order left by the spatial permutation.
        out[start..].sort_unstable_by_key(|s| s.px_sub);
        // A ray can visit the same bin/subcell more than once near the
        // photon sphere. Fold those repeated matrix entries while the bin's
        // cache line is already hot; this is the sparse analogue of packing
        // a dense micro-tile before feeding it to the multiply pipeline.
        let mut write = start;
        for read in start..out.len() {
            let rec = out[read];
            if write > start && out[write - 1].px_sub == rec.px_sub {
                out[write - 1].weight += rec.weight;
            } else {
                out[write] = rec;
                write += 1;
            }
        }
        out.truncate(write);
        off.push(write as u32);
    }
}

/// Add one glow's light to `out`, indexed by pixel. This is the same sum the
/// trace used to carry inline - radial pre-reject and the gaussian line
/// integral with transmittance - but only over
/// the bins within the glow's reach, a few thousand segments instead of
/// every segment tested against every glow.
#[cfg(test)]
fn deposit_one_impl<F, const FAST: bool>(gl: &Glow, segs: &[Seg], bin_off: &[u32], out: &mut F)
where
    F: FnMut(usize, [f64; 3]),
{
    let erf_t = ERF_LUT.get_or_init(build_erf_lut);
    let gr = gl.p.len();
    let rej = gl.sig * 3.5 + 0.06 * gr;
    let range = (rej / BIN_W).ceil() as isize + 1;
    let cell = |v: f64| (((v / BIN_W).floor() as isize) + BIN_C).clamp(0, BIN_N as isize - 1);
    let (gx, gy, gz) = (cell(gl.p.x), cell(gl.p.y), cell(gl.p.z));
    let last = BIN_N as isize - 1;

    // `bin_of` indexes a segment by its midpoint. Reject a bin when its
    // axis-aligned box cannot intersect the same radial shell as the glow.
    // This is conservative (and therefore does not change the deposited
    // image), but avoids walking the segment slice in most of the cube bins
    // that the old rectangular neighbourhood included merely by shape.
    let axis_bounds = |i: isize| {
        if i == 0 || i == last {
            // The bin index is clamped at the cube boundary. Recorded
            // segments can consequently lie beyond the nominal edge.
            (0.0, GLOW_R * GLOW_R)
        } else {
            let lo = (i - BIN_C) as f64 * BIN_W;
            let hi = lo + BIN_W;
            let lo2 = lo * lo;
            let hi2 = hi * hi;
            let min2 = if lo <= 0.0 && hi >= 0.0 {
                0.0
            } else {
                lo2.min(hi2)
            };
            (min2, lo2.max(hi2))
        }
    };
    let inner = (gr - rej - 1e-4).max(0.0);
    let outer = gr + rej + 1e-4;
    let inner2 = inner * inner;
    let outer2 = outer * outer;
    let peak = gl.c[0].max(gl.c[1]).max(gl.c[2]);
    let cutoff = if peak < 1.0 {
        GLOW_CUTOFF_DIM
    } else {
        GLOW_CUTOFF_BRIGHT
    };
    let reach = cutoff * gl.sig + GLOW_SEG_HALF_MAX;
    let reach2 = reach * reach;
    // The same conservative query, this time around the glow's actual
    // position rather than around the hole. It removes corner bins from the
    // rectangular neighbourhood before touching their segment slices.
    let point_axis_min = |i: isize, p: f64| {
        if i == 0 || i == last {
            // Clamped edge bins may contain points beyond the nominal cube.
            0.0
        } else {
            let lo = (i - BIN_C) as f64 * BIN_W;
            let hi = lo + BIN_W;
            if p < lo {
                (p - lo) * (p - lo)
            } else if p > hi {
                (p - hi) * (p - hi)
            } else {
                0.0
            }
        }
    };
    let mut xdist = [0.0; BIN_N];
    let mut ydist = [0.0; BIN_N];
    let mut zdist = [0.0; BIN_N];
    for i in 0..BIN_N {
        xdist[i] = point_axis_min(i as isize, gl.p.x);
        ydist[i] = point_axis_min(i as isize, gl.p.y);
        zdist[i] = point_axis_min(i as isize, gl.p.z);
    }
    // The Gaussian scale and the colour are glow-constant: fold them once
    // per glow instead of once per touched segment. Neither kernel needs a
    // further scale factor - the exact integral folds `gl.sig` in
    // analytically, the midpoint quadrature has none.
    let inv = gl.sig.recip();
    let inv2h = inv * inv * 0.5;
    let cs = gl.c;
    for z in (gz - range).max(0)..=(gz + range).min(last) {
        let (zmin, zmax) = axis_bounds(z);
        for y in (gy - range).max(0)..=(gy + range).min(last) {
            let (ymin, ymax) = axis_bounds(y);
            for x in (gx - range).max(0)..=(gx + range).min(last) {
                if xdist[x as usize] + ydist[y as usize] + zdist[z as usize] > reach2 {
                    continue;
                }
                let (xmin, xmax) = axis_bounds(x);
                if ymin + zmin + xmin > outer2 || ymax + zmax + xmax < inner2 {
                    continue;
                }
                let b = (x + BIN_N as isize * (y + BIN_N as isize * z)) as usize;
                for s in &segs[bin_off[b] as usize..bin_off[b + 1] as usize] {
                    // cheap radial pre-reject: the glow lives in a shell
                    // around the hole at its own radius. The squared form
                    // avoids the square root the record used to pay just to
                    // store an unsquared radius.
                    let r2 = s.r2 as f64;
                    if r2 < inner2 || r2 > outer2 {
                        continue;
                    }
                    // The Gaussian never reaches mathematical zero. The
                    // midpoint bound drops only tails below display
                    // quantisation, before paying for the two erf lookups
                    // and the exponential in the exact weight. The same
                    // offset then feeds the integral itself.
                    let (dmx, dmy, dmz) = (
                        gl.p.x - s.m[0] as f64,
                        gl.p.y - s.m[1] as f64,
                        gl.p.z - s.m[2] as f64,
                    );
                    let dm2 = dmx * dmx + dmy * dmy + dmz * dmz;
                    if dm2 > reach2 {
                        continue;
                    }
                    // The exact kernel folds in `gl.sig` analytically; the
                    // midpoint quadrature has no scale factor of its own, so
                    // both paths just multiply their kernel by the colour.
                    let e = if FAST {
                        segment_glow_midpoint(dm2, inv2h, s)
                    } else {
                        segment_glow_weight_at(gl, s, erf_t, inv, inv2h, dmx, dmy, dmz)
                    };
                    out(s.px as usize, [cs[0] * e, cs[1] * e, cs[2] * e]);
                }
            }
        }
    }
}

/// Dispatch once per glow, so the hot segment loop is monomorphized: funnel
/// groups contain no branch or axial-integral work, while stellar glows keep
/// the exact path.
#[cfg(test)]
#[inline]
fn deposit_one<F>(gl: &Glow, segs: &[Seg], bin_off: &[u32], out: &mut F)
where
    F: FnMut(usize, [f64; 3]),
{
    if gl.fast {
        deposit_one_impl::<F, true>(gl, segs, bin_off, out);
    } else {
        deposit_one_impl::<F, false>(gl, segs, bin_off, out);
    }
}

#[derive(Clone, Copy)]
struct FastSeg {
    /// Upper bits are the output pixel, low six bits the 4x4x4 subcell.
    /// Packing makes the hot sparse record 8 bytes instead of 12.
    px_sub: u32,
    weight: f32,
}

#[derive(Clone, Copy)]
struct FastBin {
    /// All fast funnel sources share one colour ratio, so the replay only
    /// needs their scalar energy field.
    field: [f32; FAST_SUBCELLS],
    peak: f32,
    exact: u64,
}

impl FastBin {
    fn zero() -> FastBin {
        FastBin {
            field: [0.0; FAST_SUBCELLS],
            peak: 0.0,
            exact: 0,
        }
    }
}

/// Apply one sparse matrix row's change directly to the persistent output
/// accumulator. `FastBin::field` is the 64-wide sampled source vector and the
/// packed `FastSeg` arena is the row's non-zero `(pixel, subcell)` entries.
/// This is the temporal analogue of a CSR SpMV update: unchanged rows are not
/// replayed at all, while a changed row subtracts its old vector and adds its
/// new vector in one cache-friendly walk.
#[inline]
fn apply_fast_bin_delta(
    bin: u32,
    old: &FastBin,
    new: &FastBin,
    fast_segs: &[FastSeg],
    fast_off: &[u32],
    fast_accum: &mut [f32],
) {
    let b = bin as usize;
    let Some((&start, &end)) = fast_off.get(b).zip(fast_off.get(b + 1)) else {
        return;
    };
    let old_enabled = old.peak >= FAST_BIN_LOD_MIN;
    let new_enabled = new.peak >= FAST_BIN_LOD_MIN;
    if !old_enabled && !new_enabled {
        return;
    }
    for fs in &fast_segs[start as usize..end as usize] {
        let sub = (fs.px_sub & FAST_SUB_MASK) as usize;
        let before = if old_enabled { old.field[sub] } else { 0.0 };
        let after = if new_enabled { new.field[sub] } else { 0.0 };
        fast_accum[(fs.px_sub >> FAST_SUB_BITS) as usize] += (after - before) * fs.weight;
    }
}

/// Merge the persistent scalar fast result into the frame's RGB side buffer.
/// The dense scan is over the small output grid, not the millions of cold
/// segment records. A negative value can only be round-off from subtract/add
/// deltas, so it is ignored rather than turning a glow into negative light.
fn merge_fast_accum(fast_accum: &[f32], st: &mut [[f64; 3]], lit: &mut Vec<u32>) {
    for (i, &energy) in fast_accum.iter().enumerate() {
        if energy <= 0.0 {
            continue;
        }
        let dst = &mut st[i];
        if *dst == [0.0; 3] {
            lit.push(i as u32);
        }
        let add = energy as f64;
        dst[0] += add;
        dst[1] += add * 0.75;
        dst[2] += add * 0.45;
    }
}

/// Fingerprint the inputs that affect one scalar fast source. A compact source
/// generation mask can then mark only bins touched by changed sources instead
/// of hashing every source again for every active bin.
#[inline]
fn fast_source_key(gl: &GlowParams) -> u64 {
    if !gl.fast {
        return 0;
    }
    let mut key = 0xcbf2_9ce4_8422_2325u64;
    for value in [
        gl.p[0].to_bits(),
        gl.p[1].to_bits(),
        gl.p[2].to_bits(),
        gl.c[0].to_bits(),
        gl.sig.to_bits(),
    ] {
        key ^= value;
        key = key.wrapping_mul(0x1000_0000_01b3);
        key = key.rotate_left(7);
    }
    key
}

#[inline]
fn exact_glow_mask(mask: u64, params: &[GlowParams]) -> u64 {
    let mut exact = 0;
    let mut bits = mask;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if !params[i].fast {
            exact |= 1u64 << i;
        }
    }
    exact
}

fn build_fast_bin(bin: u32, mask: u64, params: &[GlowParams], exp_t: &[f32]) -> FastBin {
    let b = bin as usize;
    let bx = b % BIN_N;
    let by = (b / BIN_N) % BIN_N;
    let bz = b / (BIN_N * BIN_N);
    let lo = [
        (bx as isize - BIN_C) as f64 * BIN_W,
        (by as isize - BIN_C) as f64 * BIN_W,
        (bz as isize - BIN_C) as f64 * BIN_W,
    ];
    let mut bits = mask;
    let mut out = FastBin::zero();
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let gl = &params[i];
        if !gl.fast {
            out.exact |= 1u64 << i;
            continue;
        }
        for z in 0..FAST_SUBDIV {
            for y in 0..FAST_SUBDIV {
                for x in 0..FAST_SUBDIV {
                    let sample = [
                        lo[0] + (x as f64 + 0.5) * BIN_W / FAST_SUBDIV as f64,
                        lo[1] + (y as f64 + 0.5) * BIN_W / FAST_SUBDIV as f64,
                        lo[2] + (z as f64 + 0.5) * BIN_W / FAST_SUBDIV as f64,
                    ];
                    let dx = gl.p[0] - sample[0];
                    let dy = gl.p[1] - sample[1];
                    let dz = gl.p[2] - sample[2];
                    let dm2 = dx * dx + dy * dy + dz * dz;
                    if dm2 <= gl.reach2 {
                        let e = fast_exp_lut(dm2 * gl.inv2h, exp_t);
                        let k = x + FAST_SUBDIV * y + FAST_SUBDIV * FAST_SUBDIV * z;
                        out.field[k] += gl.c[0] as f32 * e;
                    }
                }
            }
        }
    }
    out.peak = out.field.iter().copied().fold(0.0, f32::max);
    out
}

fn build_fast_fields(
    active_bins: &[u32],
    bin_glows: &[u64],
    params: &[GlowParams],
    exp_t: &[f32],
    fields: &mut [FastBin],
    par: bool,
    nthreads: usize,
) {
    let workers = if par {
        nthreads.min(active_bins.len()).max(1)
    } else {
        1
    };
    let chunk = active_bins.len().div_ceil(workers).max(1);
    if workers == 1 {
        for (&bin, out) in active_bins.iter().zip(fields.iter_mut()) {
            *out = build_fast_bin(bin, bin_glows[bin as usize], params, exp_t);
        }
    } else {
        std::thread::scope(|scope| {
            for (bins, out) in active_bins.chunks(chunk).zip(fields.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (&bin, dst) in bins.iter().zip(out.iter_mut()) {
                        *dst = build_fast_bin(bin, bin_glows[bin as usize], params, exp_t);
                    }
                });
            }
        });
    }
}

/// Per-worker glow accumulation. `px` is dense so updates need no hashing;
/// `touched` makes clearing and reduction proportional to the number of lit
/// pixels rather than the size of the entire ray grid.
struct GlowBuf {
    /// Exact stellar contributions retain f64 accumulation.
    px: Vec<[f64; 3]>,
    touched: Vec<u32>,
    /// The sampled funnel field is already scalar f32; RGB is restored only
    /// during reduction, avoiding three channel updates per segment.
    fast_px: Vec<f32>,
}

impl GlowBuf {
    fn new() -> GlowBuf {
        GlowBuf {
            px: Vec::new(),
            touched: Vec::new(),
            fast_px: Vec::new(),
        }
    }

    fn resize(&mut self, len: usize) {
        if self.px.len() != len {
            self.px = vec![[0.0; 3]; len];
            self.touched.clear();
        }
        if self.fast_px.len() != len {
            self.fast_px = vec![0.0; len];
        }
    }
}

struct GlowCache {
    /// Glow emission is kept separate from the large, mostly cold `Geo`
    /// record. The deposition kernel writes this compact array at random
    /// pixel indices, so it no longer drags the sky and crossing data into
    /// cache for every segment contribution.
    st: Vec<[f64; 3]>,
    /// One bit per glow for every spatial bin. A segment can therefore find
    /// all possible sources with one compact load.
    bin_glows: Vec<u64>,
    /// Sparse fast-field samples, indexed through `bin_field`. They are kept
    /// only for active bins so the 75^3 query cube does not consume hundreds
    /// of megabytes for empty cells.
    bin_field: Vec<u32>,
    bin_fast: Vec<FastBin>,
    /// Persistent fields, addressed by the dense `bin_field` slot map. Slots
    /// survive frames so only bins whose source generation changed rebuild.
    fast_masks: Vec<u64>,
    /// Per-source fingerprints from the previous frame; their XOR-free bit
    /// mask is used to dirty dependent bins without scanning every source in
    /// every bin.
    fast_param_keys: Vec<u64>,
    /// Compact hot-path copy of the segment metadata needed by the sampled
    /// funnel field. The full `Seg` arena remains available only for exact
    /// stellar bins.
    fast_segs: Vec<FastSeg>,
    fast_off: Vec<u32>,
    /// Persistent scalar result of the fast sparse multiply, one value per
    /// output pixel. Dirty bins update this buffer by subtraction/addition;
    /// unchanged bins never revisit their segment records.
    fast_accum: Vec<f32>,
    active_bins: Vec<u32>,
    bufs: Vec<GlowBuf>,
    /// Pixels lit by the current deposition.
    lit: Vec<u32>,
    /// Pixels lit by the previous deposition and therefore dirty after clear.
    previous_lit: Vec<u32>,
    /// Sparse glow pixels are dynamic too, but do not belong in the trace mask.
    mask: Vec<bool>,
    grid_len: usize,
}

impl GlowCache {
    fn new() -> GlowCache {
        GlowCache {
            st: Vec::new(),
            bin_glows: Vec::new(),
            bin_field: Vec::new(),
            bin_fast: Vec::new(),
            fast_masks: Vec::new(),
            fast_param_keys: Vec::new(),
            fast_segs: Vec::new(),
            fast_off: Vec::new(),
            fast_accum: Vec::new(),
            active_bins: Vec::new(),
            bufs: Vec::new(),
            lit: Vec::new(),
            previous_lit: Vec::new(),
            mask: Vec::new(),
            grid_len: 0,
        }
    }
}

/// Upper bound for all dense worker accumulators together. At very large ray
/// grids deposition uses fewer workers instead of multiplying memory use by
/// the machine's CPU count.
const GLOW_BUF_BUDGET: usize = 32 * 1024 * 1024;
/// Bound persistent fast-field slots so temporal coherence cannot turn a
/// long-running orbit into an unbounded cache. A slot is 272 bytes at the
/// current 4x4x4 sampling, so this is about 34 MiB plus the index table.
const FAST_FIELD_CACHE_MAX: usize = 131_072;

/// Lay the frame's glows over the cached geometry. Parallel workers retain
/// their dense accumulation buffers between frames, but reduce and clear only
/// pixels they actually touched. `lit` similarly identifies the Geo entries
/// that need clearing at the beginning of the next glow frame.
/// Deposit one <=64-glow chunk with a bin-major traversal. The old path was
/// glow-major: the same dense photon-sphere bin was streamed once per glow.
/// Here a segment is loaded once, the bin's glow bitmask selects its sources,
/// and the compact pixel accumulator is updated once.
#[allow(clippy::too_many_arguments)]
fn deposit_glow_chunk(
    segs: &[Seg],
    bin_off: &[u32],
    glows: &[Glow],
    grid_len: usize,
    cache: &mut GlowCache,
    par: bool,
    nthreads: usize,
    delta_fast: bool,
) {
    debug_assert!(glows.len() <= u64::BITS as usize);
    let bin_len = bin_off.len().saturating_sub(1);
    let same_bin_layout = cache.bin_glows.len() == bin_len;
    if !same_bin_layout {
        cache.bin_glows.resize(bin_len, 0);
        cache.bin_field.clear();
        cache.bin_fast.clear();
        cache.fast_masks.clear();
        cache.fast_param_keys.clear();
    }
    if cache.bin_field.len() != bin_len {
        cache.bin_field.resize(bin_len, u32::MAX);
    }

    // The masks are sparse in practice. Clear only the bins used by the
    // preceding chunk/frame instead of streaming the whole 421k-bin array.
    // The slot map and fields survive: they form the temporal cache. Keep the
    // old list long enough to subtract rows that disappear this frame.
    let previous_bins = if same_bin_layout {
        std::mem::take(&mut cache.active_bins)
    } else {
        cache.active_bins.clear();
        Vec::new()
    };
    for &bin in &previous_bins {
        cache.bin_glows[bin as usize] = 0;
    }

    let params_storage: Vec<GlowParams> = glows.iter().map(GlowParams::new).collect();
    let params: &[GlowParams] = &params_storage;
    let exp_t = FAST_EXP_LUT.get_or_init(build_fast_exp_lut);
    let mut changed_fast_mask = 0u64;
    let mut current_param_keys = Vec::with_capacity(params.len());
    for (i, gl) in params.iter().enumerate() {
        let key = fast_source_key(gl);
        if cache.fast_param_keys.get(i).copied() != Some(key) && gl.fast {
            changed_fast_mask |= 1u64 << i;
        }
        current_param_keys.push(key);
    }
    for (i, gl) in params.iter().enumerate() {
        mark_glow_bins(gl, 1u64 << i, &mut cache.bin_glows, &mut cache.active_bins);
    }
    // `mark_glow_bins` is deliberately conservative and also marks empty
    // bins. Removing those here keeps the work splitter focused on real
    // segments without changing the set of accepted contributions.
    cache.active_bins.retain(|&b| {
        let b = b as usize;
        bin_off[b] < bin_off[b + 1]
    });

    // If a moving source would grow the persistent cache past its budget,
    // discard old slots and rebuild the current working set. This bounds the
    // long-running process instead of trading temporal coherence for a leak.
    let persistent = cache.active_bins.len() <= FAST_FIELD_CACHE_MAX;
    let cache_reset = !persistent
        || cache.bin_fast.len().saturating_add(cache.active_bins.len()) > FAST_FIELD_CACHE_MAX;
    if cache_reset {
        // The old rows are no longer addressable after a reset. Start the
        // persistent output from zero and add the new working set below.
        cache.fast_accum.fill(0.0);
        cache.bin_field.fill(u32::MAX);
        cache.bin_fast.clear();
        cache.fast_masks.clear();
    }
    let use_delta = delta_fast && persistent;
    if cache.active_bins.is_empty() {
        if use_delta {
            // With no current rows the mathematical result is exactly zero.
            // Reset the dense accumulator directly instead of spending a
            // full sparse walk subtracting rows whose output is disappearing.
            cache.fast_accum.fill(0.0);
            let zero = FastBin::zero();
            for &bin in &previous_bins {
                let slot = cache.bin_field[bin as usize];
                if slot == u32::MAX {
                    continue;
                }
                cache.bin_fast[slot as usize] = zero;
                cache.fast_masks[slot as usize] = 0;
            }
            merge_fast_accum(&cache.fast_accum, &mut cache.st, &mut cache.lit);
        }
        cache.fast_param_keys = current_param_keys;
        return;
    }
    if use_delta {
        // A row that leaves the active set must be removed from the output
        // accumulator. Clear its cached field too, so reappearing in a later
        // frame is treated as a new row and gets added back.
        let zero = FastBin::zero();
        for &bin in &previous_bins {
            let b = bin as usize;
            if cache.bin_glows[b] != 0 {
                continue;
            }
            let slot = cache.bin_field[b];
            if slot == u32::MAX {
                continue;
            }
            let old = cache.bin_fast[slot as usize];
            apply_fast_bin_delta(
                bin,
                &old,
                &zero,
                &cache.fast_segs,
                &cache.fast_off,
                &mut cache.fast_accum,
            );
            cache.bin_fast[slot as usize] = zero;
            cache.fast_masks[slot as usize] = 0;
        }
    }

    if persistent {
        let mut dirty_bins = Vec::new();
        let mut dirty_slots = Vec::new();
        for &bin in &cache.active_bins {
            let b = bin as usize;
            let mask = cache.bin_glows[b];
            let exact = exact_glow_mask(mask, params);
            let fast_mask = mask & !exact;
            let slot = if cache.bin_field[b] == u32::MAX {
                let slot = cache.bin_fast.len() as u32;
                cache.bin_field[b] = slot;
                cache.bin_fast.push(FastBin::zero());
                cache.fast_masks.push(0);
                slot
            } else {
                cache.bin_field[b]
            };
            let slot_i = slot as usize;
            let new_slot = cache.fast_masks[slot_i] == 0 && cache.bin_fast[slot_i].exact == 0;
            if new_slot
                || cache.fast_masks[slot_i] != fast_mask
                || (fast_mask & changed_fast_mask) != 0
            {
                cache.fast_masks[slot_i] = fast_mask;
                dirty_bins.push(bin);
                dirty_slots.push(slot_i);
            } else {
                // Exact masks are cheap to refresh even when the scalar field
                // is reusable: an exact glow may move inside its current bin.
                cache.bin_fast[slot_i].exact = exact;
            }
        }

        // Funnel groups are broad, low-contrast midpoint primitives. Rebuild
        // only dirty fields; the existing bin worker splitter still
        // parallelizes the misses without locks, while hits are just a slot
        // lookup.
        if !dirty_bins.is_empty() {
            let mut dirty_fields = vec![FastBin::zero(); dirty_bins.len()];
            build_fast_fields(
                &dirty_bins,
                &cache.bin_glows,
                params,
                exp_t,
                &mut dirty_fields,
                par,
                nthreads,
            );
            for ((bin, slot), field) in dirty_bins.into_iter().zip(dirty_slots).zip(dirty_fields) {
                let old = cache.bin_fast[slot];
                if use_delta {
                    apply_fast_bin_delta(
                        bin,
                        &old,
                        &field,
                        &cache.fast_segs,
                        &cache.fast_off,
                        &mut cache.fast_accum,
                    );
                }
                cache.bin_fast[slot] = field;
            }
        }
    } else {
        // The active set itself is larger than the persistent cache budget.
        // Use the old bounded working-set path for this frame and release it
        // after reduction instead of allowing the cache to grow unbounded.
        cache
            .bin_fast
            .resize(cache.active_bins.len(), FastBin::zero());
        for (i, &bin) in cache.active_bins.iter().enumerate() {
            cache.bin_field[bin as usize] = i as u32;
        }
        build_fast_fields(
            &cache.active_bins,
            &cache.bin_glows,
            params,
            exp_t,
            &mut cache.bin_fast,
            par,
            nthreads,
        );
    }

    let replay_exact = glows.iter().any(|gl| !gl.fast);
    if use_delta {
        // Fast rows have already updated the persistent scalar result. Merge
        // it once over the output grid; if the chunk contains only fast
        // sources there is no worker pass at all.
        merge_fast_accum(&cache.fast_accum, &mut cache.st, &mut cache.lit);
    }
    if !use_delta || replay_exact {
        if cache.grid_len != grid_len {
            // A size change is an I-frame event, so release all old worker
            // capacities once instead of letting several grid sizes accumulate.
            cache.bufs.clear();
            cache.grid_len = grid_len;
        }
        let bytes_per_buf = grid_len
            .saturating_mul(std::mem::size_of::<[f64; 3]>())
            .saturating_add(grid_len.saturating_mul(std::mem::size_of::<f32>()))
            .max(1);
        let memory_workers = (GLOW_BUF_BUDGET / bytes_per_buf).max(1);
        let nt = if par {
            nthreads.min(memory_workers).min(cache.active_bins.len())
        } else {
            1
        };
        let erf_t = ERF_LUT.get_or_init(build_erf_lut);

        let workers = nt.max(1);
        while cache.bufs.len() < workers {
            cache.bufs.push(GlowBuf::new());
        }
        for buf in cache.bufs.iter_mut().take(workers) {
            buf.resize(grid_len);
        }

        let replay_fast = !use_delta;
        if nt <= 1 {
            {
                let active_bins = &cache.active_bins;
                let bin_field = &cache.bin_field;
                let bin_fast = &cache.bin_fast;
                let fast_segs = &cache.fast_segs;
                let fast_off = &cache.fast_off;
                let buf = &mut cache.bufs[0];
                deposit_bin_range(
                    active_bins,
                    0,
                    active_bins.len(),
                    bin_field,
                    bin_fast,
                    fast_segs,
                    fast_off,
                    segs,
                    bin_off,
                    params,
                    erf_t,
                    replay_fast,
                    buf,
                );
            }
            reduce_glow_buf(&mut cache.bufs[0], &mut cache.st, &mut cache.lit);
        } else {
            let ranges = split_bin_work(&cache.active_bins, bin_off, nt);
            let active_bins = &cache.active_bins;
            let bin_field = &cache.bin_field;
            let bin_fast = &cache.bin_fast;
            let fast_segs = &cache.fast_segs;
            let fast_off = &cache.fast_off;
            let bufs = &mut cache.bufs;
            std::thread::scope(|sc| {
                for ((begin, end), buf) in ranges.iter().copied().zip(bufs.iter_mut().take(nt)) {
                    sc.spawn(move || {
                        deposit_bin_range(
                            active_bins,
                            begin,
                            end,
                            bin_field,
                            bin_fast,
                            fast_segs,
                            fast_off,
                            segs,
                            bin_off,
                            params,
                            erf_t,
                            replay_fast,
                            buf,
                        );
                    });
                }
            });
            // Reduction remains deterministic in worker/range order. Fast field
            // sums are accumulated in f32; exact stellar sums remain f64.
            for buf in cache.bufs.iter_mut().take(ranges.len()) {
                reduce_glow_buf(buf, &mut cache.st, &mut cache.lit);
            }
        }
    }

    for i in 0..cache.active_bins.len() {
        let b = cache.active_bins[i] as usize;
        cache.bin_glows[b] = 0;
    }
    if !persistent {
        for &bin in &cache.active_bins {
            cache.bin_field[bin as usize] = u32::MAX;
        }
        cache.bin_fast.clear();
        cache.fast_masks.clear();
        cache.fast_param_keys.clear();
        cache.active_bins.clear();
    }
    cache.fast_param_keys = current_param_keys;
}

/// Lay the frame's glows over the cached geometry. Previous glow pixels are
/// cleared first, then all current sources are deposited through one
/// bin-major pass. The compact `GlowCache::st` keeps this random-write path
/// out of the large cold geometry records.
fn deposit_glows(
    segs: &[Seg],
    bin_off: &[u32],
    glows: &[Glow],
    grid_len: usize,
    cache: &mut GlowCache,
    par: bool,
    nthreads: usize,
) {
    if cache.st.len() != grid_len {
        cache.st = vec![[0.0; 3]; grid_len];
        cache.fast_accum = vec![0.0; grid_len];
        cache.lit.clear();
        cache.previous_lit.clear();
        cache.mask.clear();
    }
    if cache.fast_accum.len() != grid_len {
        cache.fast_accum = vec![0.0; grid_len];
    }
    if cache.mask.len() != grid_len {
        cache.mask.clear();
        cache.mask.resize(grid_len, false);
    }

    // Clear the previous glow, but retain its coordinates until after this
    // frame is shaded: those pixels must be redrawn once without the glow.
    cache.previous_lit.clear();
    cache.previous_lit.append(&mut cache.lit);
    for &px in &cache.previous_lit {
        cache.st[px as usize] = [0.0; 3];
        cache.mask[px as usize] = true;
    }
    if glows.is_empty() || segs.is_empty() {
        // Run the empty chunk once so rows from the preceding frame are
        // subtracted from the persistent accumulator instead of leaving a
        // temporal glow behind.
        deposit_glow_chunk(segs, bin_off, &[], grid_len, cache, par, nthreads, true);
        return;
    }
    if glows.iter().any(|glow| glow.fast) && cache.fast_off.len() != bin_off.len() {
        build_fast_arena(segs, bin_off, &mut cache.fast_segs, &mut cache.fast_off);
    }
    if cache.grid_len != grid_len {
        cache.bufs.clear();
        cache.grid_len = grid_len;
    }

    // The current application has fewer than 64 glow primitives (even with
    // three ordinary stars). Chunking keeps the representation safe if a
    // future profile exceeds that limit, at the cost of another bin pass only
    // in that unusual case.
    let delta_fast = glows.len() <= u64::BITS as usize;
    if !delta_fast {
        // Multi-chunk source lists use the bounded replay path. Do not leave
        // the previous persistent result to be merged into this frame.
        cache.fast_accum.fill(0.0);
        cache.fast_masks.fill(0);
        cache.bin_fast.fill(FastBin::zero());
    }
    for chunk in glows.chunks(u64::BITS as usize) {
        deposit_glow_chunk(
            segs, bin_off, chunk, grid_len, cache, par, nthreads, delta_fast,
        );
    }
}

/// 0 for x <= 0, x/(1+x) above: lets the glow blaze without a hard clip.
fn softclip(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x / (1.0 + x)
    }
}

/// Tidal brightening: the stream shines harder the deeper it falls.
fn tide(r: f64) -> f64 {
    1.0 + 9.0 * (RS / r) * (RS / r)
}

/// A parcel should fade before crossing the horizon instead of disappearing
/// at full tidal brightness on one frame. This prevents the direct tidal
/// profile from producing a pulse each time a discrete parcel is swallowed.
fn funnel_horizon_fade(r: f64) -> f64 {
    smoothstep(INFALL_SWALLOW, 2.6, r)
}

/// One parcel shed into a star's trail. Unlike a screen-space after-image it
/// keeps moving under gravity, with reduced tangential speed and extra drag,
/// until it crosses the horizon.
struct Trail {
    p: V3,
    v: V3,
}

impl Trail {
    fn shed(p: V3, v: V3) -> Trail {
        let radial = p.norm();
        let vr = radial * v.dot(radial);
        let vt = v - vr;
        Trail {
            p,
            v: vr + vt * INFALL_TRAIL_TANGENTIAL,
        }
    }

    /// Advance one frame-sized slice, subdividing it near the horizon.
    /// Returns false once this parcel has been swallowed.
    fn advance(&mut self, mut dt: f64, gm: f64) -> bool {
        while dt > 1e-9 {
            let r = self.p.len();
            let h = (if r < 3.0 { 0.004_f64 } else { 0.02_f64 }).min(dt);
            self.v = self.v + Infall::acc(self.p, gm) * h;
            self.v = self.v * (1.0 - INFALL_TRAIL_DRAG * h);
            self.p = self.p + self.v * h;
            dt -= h;
            if self.p.len() <= INFALL_SWALLOW {
                return false;
            }
        }
        true
    }
}

/// A live star: position, velocity, the trail of where it has been and -
/// for the massive one - how much mass it has left plus the bookkeeping
/// for the streams it sheds on the way down.
struct Infall {
    p: V3,
    v: V3,
    tr: Vec<Trail>,
    /// position of the head when the last trail parcel was shed
    tr_at: V3,
    /// false after the head crosses the horizon; the Infall remains until
    /// its last trail parcel has followed it in
    alive: bool,
    /// size scale: 3.0 for the massive star, else 1.0
    sc: f64,
    /// remaining mass fraction: the massive star is stripped inside its
    /// tidal radius and the glow tracks sc * m^(1/3)
    m: f64,
    /// drag this star feels (a quarter of the usual for the massive one)
    drag: f64,
    /// streams shed so far, and the mass shed since the last one
    ns: u64,
    debt: f64,
    /// the superstar is parked: held at a fixed position so the frame
    /// never moves, and only its funnel flows. No orbital integration,
    /// no trail, no horizon crossing - just the bleed.
    parked: bool,
    /// Profile used by the superstar's shed material.
    funnel: FunnelMode,
}

impl Infall {
    /// Deterministic spawn from a seed, so `--star` always looks the same.
    fn spawn(seed: i64, sc: f64, d: V3, a: V3, b: V3, sf: Option<f64>) -> Infall {
        let rnd = |k: i64| hash3i(seed, k, 0x51A7);
        let (r0, f0, drag) = if sc > 1.5 {
            (BIG_R0, BIG_F0, BIG_DRAG)
        } else {
            (INFALL_R0, INFALL_F0, INFALL_DRAG)
        };
        let r = mix(r0.0, r0.1, rnd(1));
        let cw = if rnd(2) < 0.5 { -1.0 } else { 1.0 };
        let inc = mix(INFALL_INC.0, INFALL_INC.1, rnd(3));
        // --star-speed overrides the draw: the fraction of the local
        // circular speed the star starts at - 1 = a circular orbit,
        // 0 = a dead drop, negative = the orbit run backwards
        let f = sf.unwrap_or_else(|| mix(f0.0, f0.1, rnd(4)));
        // circular speed in the static pseudo-potential
        let vc = (INFALL_GM * r).sqrt() / (r - RS);
        let p = d * r;
        let v = a * (f * vc * inc.cos() * cw) + b * (f * vc * inc.sin());
        Infall {
            p,
            v,
            tr: vec![Trail::shed(p, v)],
            tr_at: p,
            alive: true,
            sc,
            m: 1.0,
            drag,
            ns: 0,
            debt: 0.0,
            parked: false,
            funnel: FunnelMode::Current,
        }
    }

    fn acc(p: V3, gm: f64) -> V3 {
        let r = p.len();
        p * (-gm / ((r - RS) * (r - RS) * r))
    }

    /// Symplectic Euler with the step shrunk near the hole, plus this
    /// star's drag - a quarter of the usual for the massive one. Inside
    /// its tidal radius the massive star is torn apart: every 0.02 of
    /// shed mass becomes a glowing stream particle. After the head crosses
    /// the horizon, keep advancing its trail until every parcel follows it.
    fn advance(&mut self, mut dt: f64, gm: f64, streams: &mut Vec<Stream>) -> bool {
        let rt = 0.9 * self.sc * (2.0f64 / 3.0).cbrt();
        while dt > 1e-9 {
            let r = self.p.len();
            let h = (if self.alive && r < 3.0 {
                0.004_f64
            } else {
                0.02_f64
            })
            .min(dt);

            // Existing trail material advances over the same time slice as
            // the head. Newly shed material is appended afterwards, at the
            // end of the slice, so large --frame time jumps stay consistent.
            self.tr.retain_mut(|q| q.advance(h, gm));

            if self.alive && !self.parked {
                self.v = self.v + Self::acc(self.p, gm) * h;
                self.v = self.v * (1.0 - self.drag * h);
                self.p = self.p + self.v * h;
                if self.sc > 1.5 && r < rt {
                    // inside the tidal radius the hole strips mass away
                    let requested =
                        self.m * INFALL_STRIP * ((rt / r) * (rt / r) * (rt / r) - 1.0) * h;
                    let next_mass = (self.m - requested).max(0.05);
                    self.debt += self.m - next_mass;
                    self.m = next_mass;
                }
                if (self.p - self.tr_at).len() > INFALL_TRAIL_STEP {
                    self.tr.push(Trail::shed(self.p, self.v));
                    self.tr_at = self.p;
                    if self.tr.len() > INFALL_TRAIL_N {
                        self.tr.remove(0);
                    }
                }
                if self.p.len() <= INFALL_SWALLOW {
                    self.alive = false;
                }
            }
            if self.alive && self.sc > 5.5 {
                // Roche-lobe overflow: the parked donor's envelope bleeds
                // into the funnel - slowly, so the transfer lasts
                let bleed = SUPER_SHED_RATE * h;
                let next_mass = (self.m - bleed).max(SUPER_MIN_MASS);
                self.debt += self.m - next_mass;
                self.m = next_mass;
            }
            if self.alive && self.sc > 5.5 && self.m <= SUPER_MIN_MASS {
                // The donor is exhausted. Let already-launched material fade
                // out briefly, then remove it normally; do not leave an
                // immortal line after the star has finished feeding.
                for stream in streams.iter_mut().filter(|stream| stream.fun) {
                    stream.life = stream.life.min(stream.age + STREAM_LIFE);
                }
                self.alive = false;
            }
            if self.alive && self.sc > 1.5 {
                let (w0, cap, fun) = if self.sc > 5.5 {
                    (self.funnel.shed_weight(), self.funnel.stream_cap(), true)
                } else {
                    (0.02, STREAM_MAX, false)
                };
                // Keep unmaterialized shed mass in `debt` while the visual
                // particle pool is full; never create or discard mass just
                // because the pool was reached.
                while self.debt > w0 && streams.len() < cap {
                    self.debt -= w0;
                    self.ns += 1;
                    // The funnel leaves the donor through L1. Each profile
                    // controls its orbital momentum; the tidal profile adds
                    // a deterministic physical spread to the debris below.
                    let (v, launch_offset) = if fun {
                        let r = self.p.len();
                        let vc = (INFALL_GM * r).sqrt() / (r - RS);
                        let rad = self.p.norm();
                        let (fun_f, fun_g) = self.funnel.launch_factors();
                        let tangent = V3::new(0.0, 1.0, 0.0).cross(rad);
                        // top/bottom origins are parallel to world-up; use
                        // world-x there so the funnel still has a tangent
                        let tan = if tangent.len2() < 1e-12 {
                            V3::new(1.0, 0.0, 0.0).cross(rad).norm()
                        } else {
                            tangent.norm()
                        };
                        if self.funnel == FunnelMode::Tidal {
                            // A disrupted star does not fire identical
                            // projectiles. The debris inherits a small
                            // spread in orbital energy, angular momentum and
                            // height; gravity then shears that spread into a
                            // broad, continuous stream. Keep the perturbation
                            // deterministic so pausing or changing FPS does
                            // not make the scene shimmer.
                            let signed =
                                |key: i64| hash3i(self.ns as i64, key, 0x071D_A1A1) * 2.0 - 1.0;
                            let energy = 1.0 + 0.12 * signed(1);
                            let angular = 1.0 + 0.10 * signed(2);
                            let normal = rad.cross(tan);
                            let v = tan * (fun_f * vc * energy * angular)
                                - rad * (fun_g * vc * energy)
                                + normal * (0.045 * vc * signed(3));
                            let offset = tan * (0.10 * signed(4)) + normal * (0.10 * signed(5));
                            (v, offset)
                        } else {
                            (
                                tan * (fun_f * vc) - rad * (fun_g * vc),
                                V3::new(0.0, 0.0, 0.0),
                            )
                        }
                    } else {
                        (
                            self.v * (0.95 + 0.10 * (self.ns % 4) as f64),
                            V3::new(0.0, 0.0, 0.0),
                        )
                    };
                    streams.push(Stream {
                        p: if fun {
                            self.p * (1.0 - 0.02) + launch_offset
                        } else {
                            self.p
                        },
                        v,
                        w: w0,
                        age: 0.0,
                        life: if fun { SUPER_STREAM_LIFE } else { STREAM_LIFE },
                        drag: if fun {
                            self.funnel.stream_drag()
                        } else {
                            INFALL_DRAG
                        },
                        bri: if fun {
                            self.funnel.stream_brightness()
                        } else {
                            STREAM_BRI
                        },
                        sig: STREAM_SIG,
                        fun,
                        funnel: self.funnel,
                        group: (self.ns - 1) / self.funnel.stream_group() as u64,
                    });
                }
            }
            dt -= h;
        }
        self.alive || !self.tr.is_empty()
    }
}

/// One particle of the mass a massive star shed: it feels the same static
/// pseudo-Newtonian pull (with its own drag: the superstar's diffuse bleed
/// couples to the field harder than a compact clump) and glows until it
/// crosses the horizon or cools off.
struct Stream {
    p: V3,
    v: V3,
    /// how much star mass it carries; `bri` is how brightly that mass
    /// shines - far above STREAM_BRI for the shock-heated funnel - `sig`
    /// its glow radius, and `fun` marks a funnel parcel (whose radius
    /// tapers toward the spout instead)
    w: f64,
    /// Stable aggregation bucket. It prevents removing the oldest parcel
    /// from shifting every subsequent group and making the line jump.
    group: u64,
    age: f64,
    life: f64,
    drag: f64,
    bri: f64,
    sig: f64,
    fun: bool,
    funnel: FunnelMode,
}

impl Stream {
    /// Returns None while the stream lives; Some(w) once it is gone,
    fn advance(&mut self, mut dt: f64, gm: f64) -> bool {
        let remaining = self.life - self.age;
        if remaining <= 0.0 {
            return false;
        }
        let cooled = dt >= remaining;
        dt = dt.min(remaining);
        self.age += dt;
        while dt > 1e-9 {
            let r = self.p.len();
            let h = (if r < 3.0_f64 { 0.004_f64 } else { 0.02_f64 }).min(dt);
            self.v = self.v + Infall::acc(self.p, gm) * h;
            self.v = self.v * (1.0 - self.drag * h);
            self.p = self.p + self.v * h;
            dt -= h;
            if self.p.len() <= INFALL_SWALLOW {
                return false;
            }
        }
        !cooled
    }
}

struct Remnant {
    p: V3,
    b: f64,
    sc: f64,
}

/// All the infall state the animation owns.
struct Stars {
    live: Vec<Infall>,
    /// the streams the massive stars are shedding
    streams: Vec<Stream>,
    /// a swallowed star leaves a glow parked just outside the photon sphere
    /// (the lensing smears it into an arc hugging the shadow), with its
    /// remaining brightness and its (mass-shrunk) glow scale
    rem: Vec<Remnant>,
    seed: i64,
}

impl Stars {
    fn new() -> Stars {
        Stars {
            live: Vec::new(),
            streams: Vec::new(),
            rem: Vec::new(),
            seed: 0,
        }
    }

    /// Pick the screen side the next star dives in from and freeze the
    /// screen basis at spawn time, so the star enters from the side the
    /// user asked for; with no --origin every star picks a side at random,
    /// except the first one, which enters from the left.
    fn origin_basis(&mut self, o: &Opt) -> (V3, V3, V3) {
        self.seed += 1;
        let cam = Cam::new(o.azi, o.tilt.to_radians());
        let origin = o.origin.unwrap_or(if self.seed == 1 {
            Origin::Left
        } else {
            match (hash3i(self.seed, 0xC0DE, 0x51A7) * 6.0) as usize {
                0 => Origin::Left,
                1 => Origin::Right,
                2 => Origin::Top,
                3 => Origin::Bottom,
                4 => Origin::Front,
                _ => Origin::Back,
            }
        });
        origin.basis(&cam)
    }

    fn spawn(&mut self, big: bool, o: &Opt) {
        if self.live.iter().filter(|inf| inf.alive).count() < INFALL_MAX {
            let (d, a, b) = self.origin_basis(o);
            self.live.push(Infall::spawn(
                self.seed,
                if big { 3.0 } else { 1.0 },
                d,
                a,
                b,
                o.star_speed,
            ));
        }
    }

    /// The superstar is meant to be the whole show, and the show is a
    /// diorama: the donor is parked at a fixed spot in the world frame,
    /// so the frame never moves - only the funnel between it and the hole
    /// does. `--origin` chooses the side on which that fixed donor is parked.
    fn spawn_super(&mut self, origin: Option<Origin>, azi: f64, tilt: f64, funnel: FunnelMode) {
        if self.live.iter().any(|inf| inf.sc > 5.5) {
            return;
        }
        let p = super_park(origin, azi, tilt);
        self.live.push(Infall {
            p,
            v: V3::new(0.0, 0.0, 0.0),
            tr: Vec::new(),
            tr_at: p,
            alive: true,
            sc: SUPER_SC,
            m: 1.0,
            drag: 0.0,
            ns: 0,
            debt: 0.0,
            parked: true,
            funnel,
        });
    }

    fn clear(&mut self) {
        self.live.clear();
        self.streams.clear();
        self.rem.clear();
    }

    /// Advance every star and stream in the static background potential; each
    /// star that crosses the horizon plants its own fading remnant glow.
    fn advance(&mut self, mut dt: f64) {
        while dt > 1e-9 {
            let h = dt.min(MATTER_STEP);
            let gm = INFALL_GM;

            // Advance streams and existing remnants before stars. Material
            // created by a star during this slice starts at the slice boundary
            // instead of being aged before birth. A parcel vanishes cleanly at
            // the horizon; it does not create a flickering flash at the hole.
            self.streams.retain_mut(|stream| stream.advance(h, gm));
            let fade = (-h / (INFALL_FADE * 0.35)).exp();
            self.rem.retain_mut(|rem| {
                rem.b *= fade;
                rem.b >= 0.02
            });

            let mut i = 0;
            while i < self.live.len() {
                let was_alive = self.live[i].alive;
                let keep = self.live[i].advance(h, gm, &mut self.streams);
                if was_alive && !self.live[i].alive && !self.live[i].parked {
                    let inf = &self.live[i];
                    let r = inf.p.len();
                    let p = if r > 1e-9 {
                        inf.p * (1.6 / r)
                    } else {
                        V3::new(1.6, 0.0, 0.0)
                    };
                    self.rem.push(Remnant {
                        p,
                        b: 1.0,
                        sc: inf.sc * inf.m.cbrt(),
                    });
                }
                if keep {
                    i += 1;
                } else {
                    self.live.remove(i);
                }
            }
            dt -= h;
        }
    }
}

/// The frame's glow list: heads, trails and any remnant, rotated from the
/// world frame into the trace frame - the same rotation the shading undoes
/// when it re-looks-up the sky for an orbiting camera.
fn glow_list(st: &Stars, orb: f64) -> Vec<Glow> {
    let (c, s) = (orb.cos(), orb.sin());
    let rot = |p: V3| V3::new(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
    let mut g: Vec<Glow> = Vec::new();
    for inf in &st.live {
        // the glow tracks the star's size and whatever mass it has left;
        // brightness follows the mass up to half again past the massive
        // star's, and the radius is capped at the massive star's - a bigger
        // or brighter blob would wash the frame whenever it sweeps near the
        // camera, and a bigger radius would reach past the shell of recorded
        // segments (GLOW_R) that the binned deposition scans
        let mass_scale = inf.m.cbrt();
        let scl = inf.sc * mass_scale;
        // The exhausted parked donor is a numerical bookkeeping core, not a
        // visible object. Its launched stream is allowed to finish fading.
        if inf.parked && inf.sc > 5.5 && inf.m <= SUPER_MIN_MASS {
            continue;
        }
        // The old caps made the superstar look frozen: its six-unit size was
        // clamped to the same visual size until almost all mass was gone.
        // Keep the initial cap for performance, but scale from frame one.
        let (vis, sig_scl) = if inf.sc > 5.5 {
            (4.5 * mass_scale, 2.2 * mass_scale)
        } else {
            (scl.min(4.5), scl.min(2.2))
        };
        if inf.alive {
            let head = heat(0.85);
            let w = INFALL_HEAD_BRI * tide(inf.p.len()) * vis;
            g.push(Glow {
                p: rot(inf.p),
                c: [head[0] * w, head[1] * w, head[2] * w],
                sig: INFALL_SIG * sig_scl,
                fast: false,
            });
            if inf.sc > 5.5 {
                // Stretch the donor's near-side envelope itself toward the
                // hole. These overlapping gaussians form one tapered lobe,
                // so the funnel visibly grows out of the star instead of
                // beginning as a disconnected bead.
                let radial = inf.p.norm();
                let stretch_len = inf.funnel.stretch_len() * mass_scale;
                for i in 0..inf.funnel.stretch_n() {
                    let u = (i + 1) as f64 / inf.funnel.stretch_n() as f64;
                    let distance = 0.65 * mass_scale + stretch_len * u;
                    let p = inf.p - radial * distance;
                    let lobe_w = w * inf.funnel.stretch_weight(u);
                    g.push(Glow {
                        p: rot(p),
                        c: [head[0] * lobe_w, head[1] * lobe_w, head[2] * lobe_w],
                        sig: mix(SUPER_SIG_ROOT, SUPER_SIG_TIP, u) * mass_scale,
                        fast: false,
                    });
                }
            }
        }
        // the trail: older parcels are dimmer and cooler
        let n = inf.tr.len().max(1);
        for (i, q) in inf.tr.iter().enumerate() {
            let u = i as f64 / n as f64; // 0 = oldest, 1 = newest
            let col = heat(0.25 + 0.5 * u);
            let w = mix(0.25, 1.0, u) * INFALL_TRAIL_BRI * tide(q.p.len()) * vis;
            g.push(Glow {
                p: rot(q.p),
                c: [col[0] * w, col[1] * w, col[2] * w],
                sig: INFALL_TRAIL_SIG * sig_scl,
                fast: false,
            });
        }
    }
    // The mass the massive star sheds: a cooler string of glows, each fading
    // with age. Adjacent funnel parcels are combined into one enlarged glow;
    // the tidal profile uses a sheared launch distribution and the group's
    // positional variance so the result is a diffuse debris ribbon instead
    // of luminous bullets. Grouping keeps the expensive glow-deposition work
    // bounded.
    let mut si = 0;
    while si < st.streams.len() {
        let first = &st.streams[si];
        if !first.fun {
            let b = 1.0 - first.age / first.life;
            let w = first.bri * first.w * b * tide(first.p.len());
            g.push(Glow {
                p: rot(first.p),
                c: [w, 0.75 * w, 0.45 * w],
                sig: first.sig,
                fast: false,
            });
            si += 1;
            continue;
        }

        // Group by the parcel's stable bucket, not by its current index. If
        // the oldest parcel is swallowed, index-based groups would all shift
        // and make the entire visible line jump sideways.
        let mut end = si + 1;
        while end < st.streams.len()
            && end - si < first.funnel.stream_group()
            && st.streams[end].fun
            && st.streams[end].group == first.group
        {
            end += 1;
        }
        let group = &st.streams[si..end];
        let mut spread2: f64 = 0.0;
        for (i, a) in group.iter().enumerate() {
            for b in &group[i + 1..] {
                spread2 = spread2.max((a.p - b.p).len2());
            }
        }
        let split_factor = if first.funnel != FunnelMode::Tidal {
            0.0
        } else if group.len() < first.funnel.stream_group()
            || group
                .iter()
                .any(|stm| funnel_horizon_fade(stm.p.len()) < 0.5)
        {
            1.0
        } else {
            smoothstep(
                TIDAL_GROUP_SPLIT_START,
                TIDAL_GROUP_MAX_SPREAD,
                spread2.sqrt(),
            )
        };

        let mut weight = 0.0;
        let mut position = V3::new(0.0, 0.0, 0.0);
        let mut sig2 = 0.0;
        let mut second_moment = 0.0;
        for stm in group {
            let b = 1.0 - stm.age / stm.life;
            let horizon = funnel_horizon_fade(stm.p.len());
            let w = stm.bri * stm.w * b * stm.funnel.stream_gain(stm.p.len()) * horizon;
            let sig = stm.funnel.stream_sig(stm.p.len());
            weight += w;
            position = position + stm.p * w;
            sig2 += sig * sig * w;
            second_moment += stm.p.len2() * w;
        }
        // Keep total brightness conserved while transitioning from the
        // grouped approximation to individual branch glows.
        let grouped_weight = weight * (1.0 - split_factor);
        if grouped_weight > 0.0 {
            position = position * weight.recip();
            // The tidal profile's grouped glow also carries the positional
            // variance of its parcels. This smooths the denser simulation
            // into a diffuse tube without depositing one glow per particle;
            // the cap keeps a wrapped group from lighting the whole frame.
            let variance = (second_moment * weight.recip() - position.len2()).max(0.0);
            let group_spread = if first.funnel == FunnelMode::Tidal {
                0.75 * variance
            } else {
                0.0
            };
            let sig = (sig2 * weight.recip() + group_spread)
                .sqrt()
                .clamp(0.35, SUPER_STREAM_SIG_MAX.min(1.5));
            g.push(Glow {
                p: rot(position),
                c: [grouped_weight, 0.75 * grouped_weight, 0.45 * grouped_weight],
                sig,
                fast: true,
            });
        }
        if split_factor > 0.0 {
            // A sheared stream can put neighbouring launch parcels on
            // different orbital branches. Keep each branch anchored to its
            // own position instead of letting a horizon fade move one large
            // weighted centroid across the frame.
            for stm in group {
                let b = 1.0 - stm.age / stm.life;
                let horizon = funnel_horizon_fade(stm.p.len());
                let w = stm.bri * stm.w * b * stm.funnel.stream_gain(stm.p.len()) * horizon;
                let w = w * split_factor;
                if w > 0.0 {
                    g.push(Glow {
                        p: rot(stm.p),
                        c: [w, 0.75 * w, 0.45 * w],
                        sig: stm.funnel.stream_sig(stm.p.len()),
                        fast: true,
                    });
                }
            }
        }
        si = end;
    }
    for rem in &st.rem {
        let col = heat(0.95);
        let w = INFALL_REM_BRI * rem.b * rem.sc;
        g.push(Glow {
            p: rot(rem.p),
            c: [col[0] * w, col[1] * w, col[2] * w],
            sig: INFALL_REM_SIG * rem.sc,
            fast: false,
        });
    }
    g
}

/// One crossing of the equatorial plane inside the disk, reduced to what a
/// frame actually needs: the position (for the noise phase), the radius, the
/// pattern drift rate and the fully pre-weighted static emission.
#[derive(Clone, Copy)]
struct Cross {
    /// azimuth of the crossing point, atan2(hp.z, hp.x) - a geometric
    /// constant of the cached geodesic, so the per-frame shading needs no
    /// atan2 at all (it only drifts the pattern by orb - t*om)
    phi: f64,
    /// angular drift rate of the turbulence pattern at this radius
    om: f64,
    /// static emission (colour, profile, Doppler beaming, transmittance and
    /// grazing fade all folded together); multiply by the streak and done
    em: [f64; 3],
    rr: f64,
}

/// Cached geometry of one pixel: where its ray crosses the disk, and the sky
/// behind it (None = captured or stuck near the photon sphere - either way no
/// sky shows). `esc` keeps the un-rotated escape direction so an orbiting
/// camera can re-look-up the star field without re-tracing.
#[derive(Clone, Copy)]
struct Geo {
    sky: Option<Sky>,
    esc: V3,
    n: u8,
    cr: [Cross; 3],
}

impl Geo {
    /// A pixel that cannot change while the camera holds still: no disk
    /// crossings and no star to twinkle (the band is static). Its value from
    /// the previous frame is still correct, so shading can skip it entirely.
    /// Glow state is kept in the compact side buffer, not in this cold record.
    fn is_static(&self) -> bool {
        self.n == 0 && !self.sky.is_some_and(|s| s.star != [0.0, 0.0, 0.0])
    }

    fn empty() -> Geo {
        Geo {
            sky: None,
            esc: V3::new(0.0, 0.0, 0.0),
            n: 0,
            cr: [Cross {
                phi: 0.0,
                om: 0.0,
                rr: 0.0,
                em: [0.0; 3],
            }; 3],
        }
    }
}

/// Geometry cache, invalidated whenever anything that bends the rays changes
/// (frame size, zoom, tilt, shift, camera orbit angle). The star no longer
/// takes part: its light is deposited per frame from the segment record.
struct GeoCache {
    key: (usize, usize, u64, u64, u64),
    geo: Vec<Geo>,
    /// one byte per pixel: true = can change between frames. Scanning this
    /// instead of the 90 MB of Geo structs is what makes the skip cheap.
    mask: Vec<bool>,
    /// the glow-shell segments of the last trace, binned by position: the
    /// per-frame glow deposition scans these instead of re-integrating
    segs: Vec<Seg>,
    bin_off: Vec<u32>,
    /// the bin index of every segment in `segs`, kept from the last
    /// `build_bins` counting pass so a re-trace's permutation reuses them
    bins: Vec<u32>,
    /// the per-bin write cursor of the last permutation, retained with its
    /// pages so a re-trace does not re-allocate it
    bin_cur: Vec<u32>,
    /// per-band segment arenas of the last parallel trace; retained (with
    /// their pages) so a re-trace does not fault in fresh hundreds of
    /// megabytes just to throw them away after the concatenation
    band_bufs: Vec<Vec<Seg>>,
    /// reusable per-worker glow accumulators and the Geo pixels lit in the
    /// previous frame; both avoid full-grid work during sparse deposition
    glow: GlowCache,
    /// whether the last frame had glows (the deposition must run once more
    /// to clear them after the star is gone)
    glow_was: bool,
}

impl GeoCache {
    fn new() -> GeoCache {
        // a key no camera setup can produce
        GeoCache {
            key: (0, 0, 1, 1, 1),
            geo: Vec::new(),
            mask: Vec::new(),
            segs: Vec::new(),
            bin_off: Vec::new(),
            bins: Vec::new(),
            bin_cur: Vec::new(),
            band_bufs: Vec::new(),
            glow: GlowCache::new(),
            glow_was: false,
        }
    }

    /// Reserve every segment-side arena once, before the first frame. All
    /// of these come out of the allocator as anonymous mmaps - free until
    /// touched - so sizing them up front costs nothing now, guarantees the
    /// (huge) reservation succeeds while the process is young, and keeps
    /// the pages warm for every re-trace after that. `bins` holds one
    /// `bin_of` result per segment; `bin_off` and `bin_cur` address the
    /// fixed bin cube.
    fn prime(&mut self, o: &Opt) {
        let (w, h) = grid_size(o);
        let cap = w * h * MAX_STEPS;
        self.segs.reserve(cap);
        self.bins.reserve(cap);
        self.bin_off.reserve(BIN_N * BIN_N * BIN_N + 1);
        self.bin_cur.reserve(BIN_N * BIN_N * BIN_N);
    }
}

#[inline]
fn record_glow_segment(p0: V3, p1: V3, tr: f64, px: usize, segs: &mut Vec<Seg>) {
    if let Some(s) = Seg::from_endpoints(p0, p1, tr, px) {
        segs.push(s);
    }
}

/// Integrate one ray backwards from the camera and record what the shading
/// will need later. Time-independent by construction; `ppc0` is the ray-grid
/// pitch in pixels per star cell, needed to size the star cores. Segments
/// inside the glow shell are appended to `segs` for the per-frame glow
/// deposition; `px` is the pixel the ray (and those segments) belong to.
fn trace_geo(
    cam: &Cam,
    dir: V3,
    ppc0: f64,
    om_max: f64,
    out: &mut Geo,
    px: usize,
    segs: &mut Vec<Seg>,
) {
    let mut p = cam.p;
    let mut v = dir;
    let h2 = p.cross(v).len2();
    let mut a = accel(p, h2);
    let mut tr = 1.0f64; // remaining transmittance
    let mut n = 0u8;
    let mut cr = [Cross {
        phi: 0.0,
        om: 0.0,
        rr: 0.0,
        em: [0.0; 3],
    }; 3];

    for _ in 0..MAX_STEPS {
        let r = p.len();
        if r <= RS {
            // swallowed: this pixel is the shadow
            *out = Geo {
                sky: None,
                esc: V3::new(0.0, 0.0, 0.0),
                n,
                cr,
            };
            return;
        }
        if r > ESCAPE && p.dot(v) > 0.0 {
            let esc = v.norm();
            *out = Geo {
                sky: Some(stars(esc, ppc0)),
                esc,
                n,
                cr,
            };
            return;
        }
        // adaptive step: fine near the hole, coarse far away. The cap may be
        // generous - out there the orbit is straight and only the sky lookup
        // is left to pay for.
        let dt = (0.045 * r).clamp(0.012, 1.1);
        let pn = p + v * dt + a * (0.5 * dt * dt);
        let an = accel(pn, h2);
        let vn = v + (a + an) * (0.5 * dt);

        // did we cross the equatorial plane?
        let mut disk_hit = None;
        if p.y * pn.y < 0.0 {
            let k = p.y / (p.y - pn.y);
            let hp = p + (pn - p) * k;
            let rr = (hp.x * hp.x + hp.z * hp.z).sqrt();
            // The disk is optically thick: the near side silhouettes everything
            // behind it, which is what carves the shadow out of the glow. Each
            // crossing lets only a few per cent of the light from behind
            // through, so three recorded crossings already cover everything
            // the eye can see (the fourth would arrive at 0.01% strength).
            if rr > R_IN && rr < R_OUT {
                let vd = (v + (vn - v) * k).norm(); // direction of travel (away from us)
                                                    // Grazing crossings pile an enormous projected area into a
                                                    // single row of pixels - fade them the way a real thin disk's
                                                    // photosphere limb-darkens edge-on.
                let graze = smoothstep(0.0, 0.045, vd.y.abs());
                if n < 3 {
                    let (mut em, om_raw) = disk_em(rr, hp, vd);
                    // soft speed cap (see TURB_MAX_PX): keeps the differential
                    // rotation but stops the inner rim from strobing
                    let om = om_max * (om_raw / om_max).tanh();
                    for e in em.iter_mut() {
                        *e *= tr * graze;
                    }
                    cr[n as usize] = Cross {
                        phi: hp.z.atan2(hp.x),
                        om,
                        rr,
                        em,
                    };
                    n += 1;
                }
                disk_hit = Some(hp);
            }
        }
        // Cache the spatial line element for moving glows. A disk crossing
        // splits it so light on the far side receives the post-disk opacity.
        if let Some(hp) = disk_hit {
            record_glow_segment(p, hp, tr, px, segs);
            tr *= DISK_OPA;
            record_glow_segment(hp, pn, tr, px, segs);
        } else {
            record_glow_segment(p, pn, tr, px, segs);
        }
        p = pn;
        v = vn;
        a = an;
    }
    // ran out of steps circling the photon sphere: no sky behind this pixel
    *out = Geo {
        sky: None,
        esc: V3::new(0.0, 0.0, 0.0),
        n,
        cr,
    };
}

/// Sky colour at time `t` - the twinkle is the only thing that moves.
fn sky_rgb(s: &Sky, t: f64) -> [f64; 3] {
    if s.star == [0.0, 0.0, 0.0] {
        return s.band; // no star: nothing to twinkle, the pixel is static
    }
    let tw = 0.82 + 0.18 * tw_sin(t * s.freq + s.phase);
    [
        s.star[0] * tw + s.band[0],
        s.star[1] * tw + s.band[1],
        s.star[2] * tw + s.band[2],
    ]
}

/// Everything a frame's shading needs, precomputed once per frame so the
/// per-pixel loop is pure arithmetic (no transcendentals buried inside).
struct ShCtx {
    t: f64,
    /// camera azimuth in radians (0 for a still camera)
    orb: f64,
    c: f64,
    s: f64,
    ppc0: f64,
}

/// Emission of one pixel at time `t` from cached geometry. This is all an
/// animation frame pays for once the geodesics are traced: one sine per sky
/// pixel, one noise lookup per disk crossing.
fn shade(g: &Geo, st: &[f64; 3], ctx: &ShCtx) -> [f64; 3] {
    let mut col = [0.0f64; 3]; // HDR disk light, tone mapped at the end
    for i in 0..g.n as usize {
        let c = &g.cr[i];
        // Schwarzschild is axially symmetric: the geometry seen from azimuth
        // `orb` is the azimuth-0 geometry rotated about the disk axis, so the
        // radius, Doppler and emission are unchanged and the turbulence phase
        // simply gains `orb`. The pattern itself lives in a pre-sampled
        // texture (see build_turb_tex).
        let phi = c.phi + ctx.orb - ctx.t * c.om;
        let streak = 0.42 + 1.25 * turb(phi, c.rr);
        col[0] += c.em[0] * streak;
        col[1] += c.em[1] * streak;
        col[2] += c.em[2] * streak;
    }
    // tonemap(0) = 0, so pure-sky pixels skip the exponentials entirely
    let c = if g.n > 0 { tonemap(col) } else { [0.0; 3] };
    let bg = match &g.sky {
        Some(cached) => {
            if ctx.orb == 0.0 {
                sky_rgb(cached, ctx.t)
            } else {
                // the star field does not rotate with the camera: look it up
                // again at the rotated escape direction (cheap - the geodesics
                // stay cached)
                let d = V3::new(
                    g.esc.x * ctx.c - g.esc.z * ctx.s,
                    g.esc.y,
                    g.esc.x * ctx.s + g.esc.z * ctx.c,
                );
                sky_rgb(&stars(d, ctx.ppc0), ctx.t)
            }
        }
        None => [0.0; 3],
    };
    // the infalling star's glow is steady; the funnel animation comes from
    // moving parcels and changing mass, not an artificial brightness pulse
    let mut gl = [0.0f64; 3];
    if *st != [0.0, 0.0, 0.0] {
        gl = [softclip(st[0]), softclip(st[1]), softclip(st[2])];
    }
    [
        (c[0] + bg[0] + gl[0]).clamp(0.0, 1.0),
        (c[1] + bg[1] + gl[1]).clamp(0.0, 1.0),
        (c[2] + bg[2] + gl[2]).clamp(0.0, 1.0),
    ]
}

fn tonemap(c: [f64; 3]) -> [f64; 3] {
    // exposure + filmic shoulder + display gamma, sampled (see build_tone)
    [
        tone(EXPOSURE * c[0]),
        tone(EXPOSURE * c[1]),
        tone(EXPOSURE * c[2]),
    ]
}

fn lum(c: &[f64]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

// ---------------------------------------------------------------- options

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Ascii,
    Braille,
    Sixel,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum FunnelMode {
    /// The geometry currently used before the selectable profiles were added.
    Current,
    /// A mostly radial tidal stream: broad at the donor and gently twisted.
    Tidal,
    /// A high-angular-momentum stream that wraps more tightly around the hole.
    Spiral,
}

impl FunnelMode {
    fn parse(value: &str) -> Option<FunnelMode> {
        match value {
            "current" | "default" | "0" => Some(FunnelMode::Current),
            "tidal" | "tidal-stream" | "1" => Some(FunnelMode::Tidal),
            "spiral" | "spiral-funnel" | "2" => Some(FunnelMode::Spiral),
            _ => None,
        }
    }

    /// Fractions of the local circular speed: tangential and inward radial.
    fn launch_factors(self) -> (f64, f64) {
        match self {
            // Preserve the original almost-tangential funnel exactly.
            FunnelMode::Current => (SUPER_FUN_F, SUPER_FUN_G),
            // A tidal stream keeps more of the star's orbital momentum. It
            // therefore curves around the hole instead of looking like a
            // nozzle aimed straight at it.
            FunnelMode::Tidal => (0.62, 0.32),
            // Nearly circular launch: the stream spends longer wrapping
            // around the hole before the drag brings it in.
            FunnelMode::Spiral => (0.98, 0.04),
        }
    }

    /// Keep the simulation parcel size bounded; the tidal profile gets its
    /// smooth appearance from launch shear and grouped spatial variance, not
    /// from multiplying the number of render sources.
    fn shed_weight(self) -> f64 {
        match self {
            FunnelMode::Tidal => TIDAL_SHED_W,
            FunnelMode::Current | FunnelMode::Spiral => SUPER_SHED_W,
        }
    }

    fn stream_cap(self) -> usize {
        match self {
            FunnelMode::Tidal => TIDAL_STREAM_MAX,
            FunnelMode::Current | FunnelMode::Spiral => SUPER_STREAM_MAX,
        }
    }

    fn stream_group(self) -> usize {
        match self {
            FunnelMode::Tidal => TIDAL_STREAM_GROUP,
            FunnelMode::Current | FunnelMode::Spiral => SUPER_STREAM_GROUP,
        }
    }

    /// Tidal debris is mostly ballistic; the other profiles use the original
    /// stronger coupling so their deliberately compact funnels settle inward.
    fn stream_drag(self) -> f64 {
        match self {
            FunnelMode::Tidal => 0.003,
            FunnelMode::Current | FunnelMode::Spiral => SUPER_STREAM_DRAG,
        }
    }

    /// Lower per-particle emissivity prevents the debris from reading as a
    /// set of energy bolts while keeping its total luminosity tied to the
    /// same mass-transfer rate.
    fn stream_brightness(self) -> f64 {
        match self {
            FunnelMode::Tidal => TIDAL_STREAM_BRI,
            FunnelMode::Current | FunnelMode::Spiral => SUPER_STREAM_BRI,
        }
    }

    /// A real tidal stream can shock and brighten near pericentre, but it
    /// should not have the very sharp inverse-square pulse used by the
    /// theatrical funnel profiles.
    fn stream_gain(self, radius: f64) -> f64 {
        match self {
            FunnelMode::Tidal => 1.0 + 3.0 * (RS / radius) * (RS / radius),
            FunnelMode::Current | FunnelMode::Spiral => tide(radius),
        }
    }

    /// Gaussian radius along the funnel, wider at the donor than at the tip.
    fn stream_sig(self, radius: f64) -> f64 {
        let fraction = (radius / SUPER_PARK_R).clamp(0.0, 1.0);
        match self {
            FunnelMode::Current => mix(SUPER_SIG_TIP, SUPER_SIG_ROOT, fraction),
            FunnelMode::Tidal => mix(0.38, 1.10, fraction),
            FunnelMode::Spiral => mix(0.42, 0.95, fraction),
        }
    }

    /// The direct donor lobe should not hide the spiral profile. The tidal
    /// profile gets only a short, dim envelope extension; its debris stream
    /// should be the visible continuation rather than a straight beam.
    fn stretch_len(self) -> f64 {
        match self {
            FunnelMode::Current => SUPER_STRETCH_LEN,
            FunnelMode::Tidal => 2.8,
            FunnelMode::Spiral => 2.8,
        }
    }

    fn stretch_n(self) -> usize {
        match self {
            FunnelMode::Current => SUPER_STRETCH_N,
            FunnelMode::Tidal => 6,
            FunnelMode::Spiral => 6,
        }
    }

    fn stretch_weight(self, u: f64) -> f64 {
        match self {
            FunnelMode::Current | FunnelMode::Spiral => mix(0.46, 0.10, u),
            FunnelMode::Tidal => mix(0.22, 0.04, u),
        }
    }

    fn name(self) -> &'static str {
        match self {
            FunnelMode::Current => "current",
            FunnelMode::Tidal => "tidal",
            FunnelMode::Spiral => "spiral",
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Origin {
    Left,
    Right,
    Top,
    Bottom,
    Front,
    Back,
}

impl Origin {
    /// Screen-frame basis at spawn time: `d` is the direction the star
    /// dives in from, `a` the in-plane tangent and `b` the out-of-plane
    /// lift - an orthonormal triple either way, so the orbit keeps its
    /// shape no matter which side the star enters from.
    fn basis(self, cam: &Cam) -> (V3, V3, V3) {
        let w = cam.p.norm(); // toward the camera, out of the screen
        match self {
            Origin::Left => (cam.r * -1.0, cam.u, w),
            Origin::Right => (cam.r, cam.u, w),
            Origin::Top => (cam.u, cam.r, w),
            Origin::Bottom => (cam.u * -1.0, cam.r, w),
            Origin::Front => (w, cam.r, cam.u),
            Origin::Back => (w * -1.0, cam.r, cam.u),
        }
    }
}

/// Park the superstar according to `--origin`. With no origin keep the
/// composed default position; for horizontal sides lift it just above the
/// disk, while front/back get a small sideways offset so the donor does not
/// sit exactly on the camera axis and flood the frame.
fn super_park(origin: Option<Origin>, azi: f64, tilt: f64) -> V3 {
    let Some(side) = origin else {
        return SUPER_PARK;
    };
    let cam = Cam::new(azi, tilt.to_radians());
    let (d, _, _) = side.basis(&cam);
    let d = match side {
        Origin::Left | Origin::Right => (d + V3::new(0.0, 0.08, 0.0)).norm(),
        Origin::Top | Origin::Bottom => d,
        Origin::Front | Origin::Back => (d + cam.r * 0.35 + V3::new(0.0, 0.04, 0.0)).norm(),
    };
    d * SUPER_PARK_R
}

struct Opt {
    mode: Mode,
    /// Shape of the funnel created by --super-star.
    funnel: FunnelMode,
    fps: f64,
    zoom: f64,
    speed: f64,
    orbit: f64,
    /// Current camera azimuth in radians. Accumulated by the animation loop
    /// (not recomputed as orbit*t) so that changing the orbit rate mid-flight
    /// never jumps the view.
    azi: f64,
    tilt: f64,
    shift: f64,
    color: bool,
    cols: usize,
    rows: usize,
    /// emit target in device pixels (sixel) or character sub-pixels (rest)
    tpw: usize,
    tph: usize,
    /// upper bound on rays cast per frame
    rays: usize,
    /// worker limit; zero means automatic, one is a fully serial render path
    threads: usize,
    one_shot: Option<f64>,
    /// start with a star spiralling into the hole
    star: bool,
    /// start with a massive star, 3x the size of the hole
    big_star: bool,
    /// start with a superstar bleeding onto the hole over a very long time
    super_star: bool,
    /// which side of the screen the infalling star dives in from; None =
    /// a random side per star (the first star enters from the left)
    origin: Option<Origin>,
    /// initial star speed as a fraction of the local circular speed;
    /// negative runs the orbit backwards; None = the usual random draw
    star_speed: Option<f64>,
    ramp: Vec<char>,
}

fn num(args: &[String], i: &mut usize) -> f64 {
    *i += 1;
    args.get(*i)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn parse_opt() -> Opt {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut o = Opt {
        mode: Mode::Ascii,
        funnel: FunnelMode::Current,
        fps: 30.0,
        zoom: 1.0,
        speed: 1.0,
        orbit: 0.0,
        azi: 0.0,
        tilt: CAM_TILT,
        shift: 0.0,
        color: true,
        cols: 0,
        rows: 0,
        tpw: 0,
        tph: 0,
        rays: RAY_BUDGET,
        threads: 0,
        one_shot: None,
        star: false,
        big_star: false,
        super_star: false,
        origin: None,
        star_speed: None,
        ramp: " .·:;+=*xX#%@█".chars().collect(),
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--star" => o.star = true,
            "--big-star" | "--big-start" => o.big_star = true,
            "--super-star" | "--superstar" => o.super_star = true,
            "--funnel" => {
                i += 1;
                match args.get(i).and_then(|v| FunnelMode::parse(v)) {
                    Some(funnel) => o.funnel = funnel,
                    None => {
                        eprintln!(
                            "unknown funnel: {:?} (use current|tidal|spiral)",
                            args.get(i)
                        );
                        std::process::exit(2);
                    }
                }
            }
            "-m" | "--mode" => {
                i += 1;
                match args.get(i).map(|v| v.as_str()) {
                    Some("ascii") | Some("text") => o.mode = Mode::Ascii,
                    Some("braille") | Some("bra") => o.mode = Mode::Braille,
                    Some("sixel") => o.mode = Mode::Sixel,
                    other => {
                        eprintln!("unknown mode: {other:?} (use ascii|braille|sixel)");
                        std::process::exit(2);
                    }
                }
            }
            "--origin" => {
                i += 1;
                match args.get(i).map(|v| v.as_str()) {
                    Some("left") => o.origin = Some(Origin::Left),
                    Some("right") => o.origin = Some(Origin::Right),
                    Some("top") => o.origin = Some(Origin::Top),
                    Some("bottom") => o.origin = Some(Origin::Bottom),
                    Some("front") => o.origin = Some(Origin::Front),
                    Some("back") => o.origin = Some(Origin::Back),
                    other => {
                        eprintln!(
                            "unknown origin: {other:?} (use left|right|top|bottom|front|back)"
                        );
                        std::process::exit(2);
                    }
                }
            }
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--fps" => o.fps = num(&args, &mut i).clamp(1.0, 240.0),
            "--zoom" => o.zoom = num(&args, &mut i).clamp(0.25, 6.0),
            "--speed" => o.speed = num(&args, &mut i),
            "--orbit" => o.orbit = num(&args, &mut i),
            "--tilt" => o.tilt = num(&args, &mut i),
            "--shift" => o.shift = num(&args, &mut i),
            "--star-speed" => {
                let v = num(&args, &mut i);
                o.star_speed = if v.is_nan() { None } else { Some(v) };
            }
            "--cols" => o.cols = num(&args, &mut i) as usize,
            "--rows" => o.rows = num(&args, &mut i) as usize,
            "--rays" => o.rays = num(&args, &mut i).clamp(40_000.0, 4_000_000.0) as usize,
            "--threads" => o.threads = num(&args, &mut i).clamp(0.0, 32.0) as usize,
            "--frame" => o.one_shot = Some(num(&args, &mut i)),
            "--no-color" | "--ascii-only" => o.color = false,
            _ => {
                eprintln!("unknown argument: {a}\n\n{HELP}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if o.speed.is_nan() {
        o.speed = 1.0;
    }
    if o.orbit.is_nan() {
        o.orbit = 0.0;
    }
    if o.tilt.is_nan() {
        o.tilt = CAM_TILT;
    }
    if o.shift.is_nan() {
        o.shift = 0.0;
    }
    if o.rays == 0 {
        o.rays = RAY_BUDGET;
    }
    let (c, r) = term_size();
    if o.cols == 0 {
        o.cols = c;
    }
    if o.rows == 0 {
        o.rows = r;
    }
    o.cols = o.cols.clamp(20, 600);
    o.rows = o.rows.clamp(10, 300);
    // ascii/braille draw one character per SUB_X x SUB_Y block of rays; sixel
    // has no such grid - it paints device pixels, so the picture must be
    // rendered at the real pixel size of cols x rows cells or it comes out a
    // couple of times smaller than the window.
    let (cw, ch) = if o.mode == Mode::Sixel {
        cell_pixels()
    } else {
        (SUB_X, SUB_Y)
    };
    o.tpw = o.cols * cw;
    o.tph = o.rows * ch;
    o
}

const HELP: &str = "\
blackhole - a relativistic black hole for your terminal

USAGE: blackhole [OPTIONS]

MODES
  -m, --mode ascii|braille|sixel   renderer (default: ascii)
      ascii   one character per block, truecolour ANSI
      braille higher apparent resolution via braille dots
      sixel   real graphics; the image is rendered at device-pixel size
              (the cell size is asked from the terminal via CSI 16 t)

OPTIONS
      --fps <n>         frame rate (default: 30)
      --speed <n>       flow speed of the disk (default: 1)
      --zoom <n>        zoom, >1 is closer (default: 1)
      --orbit <deg/s>   slow camera orbit rate (default: 0)
      --tilt <deg>      camera elevation above the disk plane
      --shift <n>       move the picture up (+) / down (-), half-frame units
      --cols <n>        override terminal width
      --rows <n>        override terminal height
      --rays <n>        ray budget per frame (default 200000); lower = faster
                        and coarser, higher = sharper and slower
      --threads <n>     worker limit; 0 = automatic, 1 = fully single-threaded
      --frame <n>       render a single frame at time n/fps and exit
      --no-color        no ANSI colours (pure ASCII output, good for pipes)
      --star            add a star that gets swallowed by the hole
      --big-star        start with a massive star, 3x the size of the hole
      --super-star     start with a superstar: a giant parked dead-still in the
                        frame, pouring one long luminous funnel into the hole -
                        the frame never rotates, only the flow moves, and the
                        drain takes tens of minutes (s/S add normal stars)
      --funnel <shape>  superstar funnel: current|tidal|spiral (default: current)
                        tidal = diffuse sheared debris; spiral = stronger wrap
      --origin <side>   side the star dives in from: left|right|top|bottom|front|back
                        (default: random per star; the first star dives in from the left)
      --star-speed <n>  initial star speed as a fraction of the local circular
                        speed: 1 = circular orbit, 0 = dropped from rest,
                        negative = the orbit run backwards (default: random)

KEYS        q/Esc quit    +/- zoom    up/down tilt    left/right orbit rate    space pause
            < / > simulation speed slower/faster (same physical keys in any
            keyboard layout)
            s spawn star    S spawn big star    b spawn super star    x clear stars
";

// ---------------------------------------------------------------- terminal

/// Size of one character cell in device pixels, asked from the terminal with
/// XTWINOPS `CSI 16 t` (reply `CSI 6 ; height ; width t`). Only sixel needs it.
/// A terminal that does not answer gets the usual 10x20 guess.
fn cell_pixels() -> (usize, usize) {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return (10, 20);
    }
    let _ = Command::new("stty")
        .args(["raw", "-echo", "min", "0", "time", "1"])
        .status();
    let mut so = std::io::stdout();
    let _ = so.write_all(b"\x1b[16t");
    let _ = so.flush();
    let mut got = Vec::new();
    let mut buf = [0u8; 32];
    for _ in 0..3 {
        // `time 1` above makes each read give up after 100 ms
        match std::io::stdin().read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(k) => {
                got.extend_from_slice(&buf[..k]);
                if got.ends_with(b"t") {
                    break;
                }
            }
        }
    }
    let _ = Command::new("stty").args(["sane"]).status();
    let s = String::from_utf8_lossy(&got);
    let body = s.trim_start_matches('\x1b').trim_start_matches('[');
    let body = body.trim_end_matches('t');
    let mut it = body.split(';');
    if it.next() == Some("6") {
        if let (Some(h), Some(w)) = (it.next(), it.next()) {
            if let (Ok(h), Ok(w)) = (h.parse::<usize>(), w.parse::<usize>()) {
                if h > 0 && w > 0 {
                    return (w, h);
                }
            }
        }
    }
    (10, 20)
}

fn term_size() -> (usize, usize) {
    // ask the tty first (works when stdin is a terminal)
    if let Ok(out) = Command::new("stty")
        .arg("size")
        .stdin(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::null())
        .output()
    {
        if let Some((c, r)) = parse_size(&String::from_utf8_lossy(&out.stdout)) {
            return (c, r);
        }
    }
    if let Ok(out) = Command::new("stty")
        .args(["-F", "/dev/tty", "size"])
        .output()
    {
        if let Some((c, r)) = parse_size(&String::from_utf8_lossy(&out.stdout)) {
            return (c, r);
        }
    }
    if let (Some(c), Some(r)) = (
        env::var("COLUMNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
        env::var("LINES").ok().and_then(|v| v.parse::<usize>().ok()),
    ) {
        if c > 0 && r > 0 {
            return (c, r);
        }
    }
    (100, 32)
}

fn parse_size(s: &str) -> Option<(usize, usize)> {
    let mut it = s.split_whitespace();
    let r = it.next()?.parse::<usize>().ok()?;
    let c = it.next()?.parse::<usize>().ok()?;
    if r > 0 && c > 0 {
        Some((c, r))
    } else {
        None
    }
}

struct RawTerm;

impl RawTerm {
    fn new() -> RawTerm {
        let _ = Command::new("stty")
            .args(["raw", "-echo", "min", "0", "time", "0"])
            .status();
        RawTerm
    }
}

impl Drop for RawTerm {
    fn drop(&mut self) {
        let _ = Command::new("stty").args(["sane"]).status();
    }
}

/// A keypress, with arrow escape sequences told apart from a plain Esc.
enum Key {
    Char(char),
    Esc,
    Up,
    Down,
    Left,
    Right,
}

/// Camera tilt step per arrow press, in degrees (terminals auto-repeat held
/// keys, so this is also the slew rate: ~1 deg per press at ~25 cps).
const TILT_STEP: f64 = 1.0;
/// Stay away from exactly +-90 deg: there the camera basis degenerates
/// (forward becomes parallel to the world-up axis).
const TILT_LIMIT: f64 = 80.0;

/// Orbit rate step per arrow press, in degrees per second (held keys
/// auto-repeat, so this is also how fast the rate itself slews).
const ORBIT_STEP: f64 = 1.0;
const ORBIT_MAX: f64 = 90.0;

/// Simulation-speed multiplier per < / > press and the range it is clamped
/// to (held keys auto-repeat here too, exactly like the zoom and tilt keys).
const SPEED_STEP: f64 = 1.15;
const SPEED_MIN: f64 = 1.0 / 64.0;
const SPEED_MAX: f64 = 128.0;

/// One press of the speed keys: multiplicative like the zoom keys, clamped
/// to the range above, and preserving the direction of a negative
/// `--speed -n` (time running backwards) instead of flipping it.
fn step_speed(speed: f64, up: bool) -> f64 {
    let s = if up {
        (speed.abs() * SPEED_STEP).min(SPEED_MAX)
    } else {
        (speed.abs() / SPEED_STEP).max(SPEED_MIN)
    };
    if speed < 0.0 {
        -s
    } else {
        s
    }
}

/// The first UTF-8 character of a keypress, or the raw byte as a fallback
/// for anything that is not valid UTF-8. A terminal in a non-Latin layout
/// sends multi-byte characters for the very physical keys whose ASCII a
/// Latin layout would produce, so decoding properly (instead of taking the
/// first byte) is what lets a binding follow the key, not the layout.
fn first_char(b: &[u8]) -> char {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.chars().next())
        .unwrap_or_else(|| b[0] as char)
}

/// Poll stdin for one key. Arrow keys arrive as 3-byte bursts (`ESC [ A`, or
/// `ESC O A` in application cursor mode); a lone ESC is either the Esc key
/// or a burst that came in fragmented, so give the terminal a few
/// milliseconds to finish it before deciding.
fn poll_key() -> Option<Key> {
    let seq = |c: u8| match c {
        b'A' => Some(Key::Up),
        b'B' => Some(Key::Down),
        b'C' => Some(Key::Right),
        b'D' => Some(Key::Left),
        _ => None,
    };
    let mut b = [0u8; 8];
    let n = match std::io::stdin().read(&mut b) {
        Ok(0) | Err(_) => return None,
        Ok(n) => n,
    };
    if b[0] != 0x1b {
        return Some(Key::Char(first_char(&b[..n])));
    }
    if n >= 3 && (b[1] == b'[' || b[1] == b'O') {
        return seq(b[2]);
    }
    if n == 1 {
        let mut m = 0;
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(1));
            m += std::io::stdin().read(&mut b[1 + m..]).unwrap_or(0);
            if m >= 2 {
                break;
            }
        }
        if m >= 2 && (b[1] == b'[' || b[1] == b'O') {
            return seq(b[2]);
        }
        return Some(Key::Esc);
    }
    None
}

// ---------------------------------------------------------------- renderers

const SUB_X: usize = 2;
const SUB_Y: usize = 4;
/// Text cells are independent while they are being reduced from the ray
/// grid. Keep small terminal windows single-threaded: the scoped-thread setup
/// costs more than the handful of cells in those frames.
const TEXT_PAR_MIN_CELLS: usize = 8_192;

struct Frame {
    w: usize,
    h: usize,
    px: Vec<[f64; 3]>,
}

/// Rays are cast on a grid no denser than RAY_BUDGET; if the emit target is
/// larger the renderers upsample with nearest neighbour. Keeps sixel frames
/// (device-pixel sized) from costing seconds each.
const RAY_BUDGET: usize = 200_000;

/// Renders into `f`, reusing its pixel buffer between frames - a sixel frame
/// is a 14 MB allocation and re-allocating it every frame shows.
/// The ray grid for the current options: `rays` caps how many of the
/// terminal's sub-cells get traced, with a floor so tiny windows still fill
/// the picture. Both the per-frame sizing and the one-off arena priming
/// below share this, so the reservation can never disagree with what a
/// frame actually needs.
fn grid_size(o: &Opt) -> (usize, usize) {
    let s = (o.rays as f64 / (o.tpw as f64 * o.tph as f64))
        .min(1.0)
        .sqrt();
    let w = ((o.tpw as f64 * s).round() as usize).max(80);
    let h = ((o.tph as f64 * s).round() as usize).max(40);
    (w, h)
}

fn worker_count(requested: usize) -> usize {
    if requested > 0 {
        requested.clamp(1, 32)
    } else {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 32)
    }
}

/// Fill a text-cell buffer in row-aligned disjoint slices. The callback only
/// reads the frame/options and writes its own slice, so the result remains in
/// row-major order without locks or a second frame-sized buffer. Keeping the
/// terminal diff/output stage outside this helper preserves the existing
/// single-owner `Screen` architecture.
fn fill_text_cells<F>(cells: &mut [Cell], cw: usize, ch: usize, nthreads: usize, fill: F)
where
    F: Fn(usize, &mut [Cell]) + Sync,
{
    if cells.is_empty() || cw == 0 || ch == 0 {
        return;
    }
    debug_assert_eq!(cells.len(), cw * ch);

    let nthreads = nthreads.min(ch);
    if nthreads < 2 || cells.len() < TEXT_PAR_MIN_CELLS {
        fill(0, cells);
        return;
    }

    let rows_per = ch.div_ceil(nthreads);
    thread::scope(|scope| {
        for (band_no, band) in cells.chunks_mut(rows_per * cw).enumerate() {
            let fill = &fill;
            let y0 = band_no * rows_per;
            scope.spawn(move || fill(y0, band));
        }
    });
}

fn render_frame(o: &Opt, t: f64, f: &mut Frame, cache: &mut GeoCache, glows: &[Glow]) {
    let (w, h) = grid_size(o);
    f.w = w;
    f.h = h;
    // keep the previous frame's pixels: every shaded pixel is overwritten and
    // static ones (see below) rely on their old value being still there.
    // Re-zeroing 8 MB per frame was pure waste.
    let resized = f.px.len() != w * h;
    if resized {
        f.px.clear();
        f.px.resize(w * h, [0.0; 3]);
    }
    let px = &mut f.px;
    let orbit = o.azi; // camera azimuth right now (kept by the caller)
                       // The camera may orbit, but the hole is axially symmetric: geometry is
                       // traced once at azimuth zero and shaded at the current azimuth (see
                       // `shade`), so an orbiting camera never pays for a re-trace.
    let cam = Cam::new(0.0, o.tilt.to_radians());
    let zoom = o.zoom;
    let shift = o.shift;

    let nthreads = worker_count(o.threads);
    // Small frames run faster on a single thread: spawning a worker costs
    // more than the shading it would do.
    let par = nthreads > 1 && w * h >= 96_000;
    let rows_per = h.div_ceil(nthreads);

    // pixels per star cell at layer 0: w pixels span 2*atan(VIEW/zoom*aspect)
    // radians, a layer-0 cell is 1/STAR_SCALE[0] radians wide
    let aspect = w as f64 / h as f64;
    let fov = 2.0 * (VIEW / zoom * aspect).atan();
    let ppc0 = w as f64 / (fov * STAR_SCALE[0]);
    let om_max = TURB_MAX_PX * fov / o.tpw as f64;

    // I-frame: full geodesic pass, only when the camera setup changes. The
    // orbit is deliberately absent from the key: rotation of the view is
    // handled in the shading, not by moving the camera. The stars are absent
    // for the same reason - their light is re-deposited every frame onto the
    // cached path segments by the glow pass below, so a moving star never
    // costs a re-trace.
    let key = (w, h, zoom.to_bits(), o.tilt.to_bits(), shift.to_bits());
    let mut invalidated = false;
    if cache.key != key || cache.geo.len() != w * h {
        invalidated = true;
        cache.key = key;
        // The re-trace overwrites every Geo entry, so old glow indices no
        // longer identify state that needs an explicit clear.
        cache.glow.lit.clear();
        cache.glow.previous_lit.clear();
        cache.glow.mask.clear();
        // The geometry is being replaced, so the side-buffered glow values
        // no longer belong to these pixels even when the grid size is stable.
        cache.glow.st.clear();
        cache.glow.fast_accum.clear();
        cache.glow.bin_glows.clear();
        cache.glow.bin_field.clear();
        cache.glow.bin_fast.clear();
        cache.glow.fast_masks.clear();
        cache.glow.fast_param_keys.clear();
        cache.glow.active_bins.clear();
        cache.glow.fast_segs.clear();
        cache.glow.fast_off.clear();
        // same reuse story as the pixel buffer: a stable-size re-trace (zoom)
        // overwrites every entry anyway
        if cache.geo.len() != w * h {
            cache.geo.clear();
            cache.geo.resize_with(w * h, Geo::empty);
            cache.mask.clear();
            cache.mask.resize(w * h, false);
        }
        let geo = &mut cache.geo;
        let mask = &mut cache.mask;
        // The arena can hold every ray at its step budget; reserving that
        // up front is free (untouched pages of the large allocation are
        // never faulted in) and removes the repeated doubling copies a
        // growing Vec of ten million segments otherwise pays. The arena -
        // and the per-band arenas below - are taken out of the cache and
        // put back, so a re-trace (zoom, tilt) reuses warm pages instead
        // of faulting in fresh hundreds of megabytes every time.
        let mut segs = std::mem::take(&mut cache.segs);
        segs.clear();
        segs.reserve(w * h * MAX_STEPS);
        if par {
            // each band collects the glow-lit path segments into its own
            // arena; concatenated in band order they arrive sorted by pixel,
            // which keeps the deposition pass cache-friendly
            let nband = h.div_ceil(rows_per);
            let mut local = std::mem::take(&mut cache.band_bufs);
            local.resize_with(nband, Vec::new);
            for buf in local.iter_mut() {
                buf.clear();
                buf.reserve(rows_per * w * MAX_STEPS);
            }
            thread::scope(|sc| {
                for (n, ((band, mband), slot)) in geo
                    .chunks_mut(rows_per * w)
                    .zip(mask.chunks_mut(rows_per * w))
                    .zip(local.iter_mut())
                    .enumerate()
                {
                    let cam = &cam;
                    sc.spawn(move || {
                        let y0 = n * rows_per;
                        for (j, (rowgeo, rowmask)) in
                            band.chunks_mut(w).zip(mband.chunks_mut(w)).enumerate()
                        {
                            let y = y0 + j;
                            for (x, (g, m)) in rowgeo.iter_mut().zip(rowmask.iter_mut()).enumerate()
                            {
                                let dir = cam.ray(x, y, w, h, zoom, shift);
                                trace_geo(cam, dir, ppc0, om_max, g, y * w + x, slot);
                                // remember which pixels can ever change - the
                                // P-frames scan this byte mask instead of
                                // streaming the whole geometry cache
                                *m = !g.is_static();
                            }
                        }
                    });
                }
            });
            for v in local.iter() {
                segs.extend_from_slice(v);
            }
            cache.band_bufs = local;
        } else {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let dir = cam.ray(x, y, w, h, zoom, shift);
                    trace_geo(&cam, dir, ppc0, om_max, &mut geo[i], i, &mut segs);
                    mask[i] = !geo[i].is_static();
                }
            }
        }
        // index the segment arena by position (in-place counting sort into
        // 1.2-unit cells): the per-frame glow pass then touches only the few
        // cells a glow can light instead of streaming the whole arena
        build_bins(
            &mut segs,
            &mut cache.bin_off,
            &mut cache.bins,
            &mut cache.bin_cur,
        );
        cache.segs = segs;
    }

    // Glow pass: the per-frame price of moving stars. The trace above pays
    // no attention to the glows - it only cached the path segments a glow
    // could ever light, binned by position. Every frame with live glows (and
    // once more after the last of them dies, to wipe the residual light)
    // re-deposits their light onto those segments: a small binned
    // neighbourhood scan per glow, instead of the full re-trace a moving
    // star used to force on every frame.
    let glow_now = !glows.is_empty();
    if glow_now || cache.glow_was {
        deposit_glows(
            &cache.segs,
            &cache.bin_off,
            glows,
            cache.geo.len(),
            &mut cache.glow,
            par,
            nthreads,
        );
        // `deposit_glows` has populated `lit`; make both old and new glow
        // pixels eligible for the cheap shading pass below.
        for &px in &cache.glow.lit {
            cache.glow.mask[px as usize] = true;
        }
    }

    // P-frame: re-shade the cached geometry for the current time and azimuth.
    // The one-off sample tables are wanted by the shading (and by nothing
    // else), so they are built here, once.
    if TURB_TEX.get().is_none() {
        build_turb_tex(nthreads);
    }
    if TONE.get().is_none() {
        build_tone();
    }
    let ctx = ShCtx {
        t,
        orb: orbit,
        c: orbit.cos(),
        s: orbit.sin(),
        ppc0,
    };
    // pixels that cannot change (no disk, no star, still camera) keep their
    // previous value; on the first frame or after a re-trace everything is
    // shaded so the buffer is fully written. The mask carries the decision
    // so this pass streams 320 KB, not the whole geometry cache.
    // Glow pixels are sparse dynamic pixels. Keep the normal trace mask for
    // disk/star animation, and add only the pixels touched by the glow pass.
    let skip_static = !invalidated && !resized && orbit == 0.0;
    // glow-lit pixels are not in the trace-time mask, so while any glow is
    // live (and for one frame after the last one dies) everything is
    // re-shaded; once the residual is cleared the cheap masked path is
    // valid again
    cache.glow_was = glow_now;
    if cache.glow.mask.len() != cache.geo.len() {
        cache.glow.mask.clear();
        cache.glow.mask.resize(cache.geo.len(), false);
    }
    let geo = &cache.geo;
    let glow_mask = &cache.glow.mask;
    let glow_st = &cache.glow.st;
    if par {
        thread::scope(|sc| {
            for (n, (((band, mband), gmband), stband)) in px
                .chunks_mut(rows_per * w)
                .zip(cache.mask.chunks(rows_per * w))
                .zip(glow_mask.chunks(rows_per * w))
                .zip(glow_st.chunks(rows_per * w))
                .enumerate()
            {
                let src = &geo[n * rows_per * w..];
                let ctx = &ctx;
                sc.spawn(move || {
                    if skip_static {
                        for ((((p, m), gm), g), st) in
                            band.iter_mut().zip(mband).zip(gmband).zip(src).zip(stband)
                        {
                            if !(*m || *gm) {
                                continue;
                            }
                            *p = shade(g, st, ctx);
                        }
                    } else {
                        for ((p, g), st) in band.iter_mut().zip(src).zip(stband) {
                            *p = shade(g, st, ctx);
                        }
                    }
                });
            }
        });
    } else if skip_static {
        for ((((p, m), gm), g), st) in px
            .iter_mut()
            .zip(cache.mask.iter())
            .zip(glow_mask.iter())
            .zip(geo.iter())
            .zip(glow_st.iter())
        {
            if !(*m || *gm) {
                continue;
            }
            *p = shade(g, st, &ctx);
        }
    } else {
        for ((p, g), st) in px.iter_mut().zip(geo.iter()).zip(glow_st.iter()) {
            *p = shade(g, st, &ctx);
        }
    }

    // The previous glow pixels were redrawn above after their old `st` value
    // was cleared. Keep only the current deposition in the sparse mask.
    for &px in &cache.glow.previous_lit {
        cache.glow.mask[px as usize] = false;
    }
    cache.glow.previous_lit.clear();
}

fn push_u32(out: &mut String, mut v: u32) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut n = 0;
    while v > 0 {
        buf[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    for i in (0..n).rev() {
        out.push(buf[i] as char);
    }
}

// -------------------------------------------------------- incremental UI
//
// The classic fast-terminal recipe (see e.g. Andrew Kelley's `tcout` demo):
// never hand the terminal a whole screen per frame. Keep the previous frame's
// cells, send only what changed, in runs, with a colour set only when it
// actually changes, and push it all with a single write(). A naive full
// repaint with SGR + reset per cell costs ~25 bytes per cell and makes the
// terminal, not the renderer, the bottleneck.

#[derive(Clone, PartialEq)]
struct Cell {
    ch: char,
    rgb: [u8; 3],
}

struct Screen {
    prev: Vec<Cell>,
    w: usize,
    h: usize,
}

impl Screen {
    fn new() -> Screen {
        Screen {
            prev: Vec::new(),
            w: 0,
            h: 0,
        }
    }

    /// Append the shortest byte sequence that turns the screen from `prev`
    /// into `cells` (a `cw` x `chh` grid, row major). With `o.color` off no SGR
    /// is emitted at all and colours take no part in the diff.
    fn emit(&mut self, o: &Opt, cells: &[Cell], cw: usize, chh: usize, out: &mut String) {
        let full = cw != self.w || chh != self.h || cells.len() != self.prev.len();
        let mut cur: Option<[u8; 3]> = None; // colour the terminal is in now
        if full {
            for y in 0..chh {
                for x in 0..cw {
                    let c = &cells[y * cw + x];
                    set_colour(o, out, &mut cur, c.rgb);
                    out.push(c.ch);
                }
                if y + 1 < chh {
                    out.push_str("\r\n");
                }
            }
        } else {
            for y in 0..chh {
                let row = y * cw;
                let mut x = 0;
                while x < cw {
                    if cells[row + x] == self.prev[row + x] {
                        x += 1;
                        continue;
                    }
                    // changed run start: extend it over gaps of a few equal
                    // cells - re-addressing the cursor costs about as much as
                    // rewriting three cells
                    let mut end = x + 1;
                    let mut gap = 0;
                    let mut j = x + 1;
                    while j < cw {
                        if cells[row + j] != self.prev[row + j] {
                            end = j + 1;
                            gap = 0;
                        } else {
                            gap += 1;
                            if gap > 3 {
                                break;
                            }
                        }
                        j += 1;
                    }
                    out.push_str("\x1b[");
                    push_u32(out, (y + 1) as u32);
                    out.push(';');
                    push_u32(out, (x + 1) as u32);
                    out.push('H');
                    for c in &cells[row + x..row + end] {
                        set_colour(o, out, &mut cur, c.rgb);
                        out.push(c.ch);
                    }
                    x = end;
                }
            }
        }
        if o.color {
            out.push_str("\x1b[0m");
        }
        self.prev = cells.to_vec();
        self.w = cw;
        self.h = chh;
    }
}

fn set_colour(o: &Opt, out: &mut String, cur: &mut Option<[u8; 3]>, rgb: [u8; 3]) {
    if !o.color || *cur == Some(rgb) {
        return;
    }
    out.push_str("\x1b[38;2;");
    push_u32(out, rgb[0] as u32);
    out.push(';');
    push_u32(out, rgb[1] as u32);
    out.push(';');
    push_u32(out, rgb[2] as u32);
    out.push('m');
    *cur = Some(rgb);
}

/// Cell colour, quantised to 32 levels per channel: fewer SGR switches, and
/// tiny float wobble between frames no longer counts as a change.
fn cell_rgb(o: &Opt, c: &[f64]) -> [u8; 3] {
    if !o.color {
        return [0, 0, 0];
    }
    let q = |x: f64| (((x.max(0.0) * 31.0).round() as u8).min(31)) * 8;
    [q(c[0]), q(c[1]), q(c[2])]
}

fn draw_ascii(o: &Opt, f: &Frame, out: &mut String, scr: &mut Screen) {
    let ramp = &o.ramp;
    let cw = f.w / SUB_X;
    // Keep the terminal's last row for the unobtrusive status line.
    let ch = (f.h / SUB_Y).min(o.rows.saturating_sub(1));
    let mut cells = vec![
        Cell {
            ch: ' ',
            rgb: [0; 3],
        };
        cw * ch
    ];
    fill_text_cells(&mut cells, cw, ch, worker_count(o.threads), |y0, band| {
        for (row_no, row) in band.chunks_mut(cw).enumerate() {
            let cy = y0 + row_no;
            for (cx, cell) in row.iter_mut().enumerate() {
                let mut acc = [0.0; 3];
                for sy in 0..SUB_Y {
                    for sx in 0..SUB_X {
                        let p = f.px[(cy * SUB_Y + sy) * f.w + cx * SUB_X + sx];
                        acc[0] += p[0];
                        acc[1] += p[1];
                        acc[2] += p[2];
                    }
                }
                let n = (SUB_X * SUB_Y) as f64;
                let c = [acc[0] / n, acc[1] / n, acc[2] / n];
                let l = lum(&c);
                let idx = ((ramp.len() - 1) as f64 * l).round() as usize;
                *cell = Cell {
                    ch: ramp[idx],
                    rgb: cell_rgb(o, &c),
                };
            }
        }
    });
    scr.emit(o, &cells, cw, ch, out);
}

fn dot_bit(sx: usize, sy: usize) -> u8 {
    match (sx, sy) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (0, 3) => 0x40,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        _ => 0x80,
    }
}

fn draw_braille(o: &Opt, f: &Frame, out: &mut String, scr: &mut Screen) {
    let cw = f.w / SUB_X;
    // Keep the terminal's last row for the unobtrusive status line.
    let ch = (f.h / SUB_Y).min(o.rows.saturating_sub(1));
    let mut cells = vec![
        Cell {
            ch: ' ',
            rgb: [0; 3],
        };
        cw * ch
    ];
    fill_text_cells(&mut cells, cw, ch, worker_count(o.threads), |y0, band| {
        for (row_no, row) in band.chunks_mut(cw).enumerate() {
            let cy = y0 + row_no;
            for (cx, cell) in row.iter_mut().enumerate() {
                let mut acc = [0.0; 3];
                let mut lums = [0.0f64; 8];
                for sy in 0..SUB_Y {
                    for sx in 0..SUB_X {
                        let p = f.px[(cy * SUB_Y + sy) * f.w + cx * SUB_X + sx];
                        acc[0] += p[0];
                        acc[1] += p[1];
                        acc[2] += p[2];
                        lums[sy * SUB_X + sx] = lum(&p);
                    }
                }
                let n = (SUB_X * SUB_Y) as f64;
                let c = [acc[0] / n, acc[1] / n, acc[2] / n];
                let l = lum(&c);
                // punch the brightest sub-cells first: grey level -> number of dots
                let dots = (l * 8.0).round() as usize;
                let mut bits = 0u8;
                for _ in 0..dots.min(8) {
                    let mut best = 0usize;
                    for k in 0..8 {
                        if lums[k] > lums[best] {
                            best = k;
                        }
                    }
                    lums[best] = -1.0;
                    bits |= dot_bit(best % SUB_X, best / SUB_X);
                }
                // faint stars still get a single dot
                if bits == 0 && l > 0.03 {
                    bits = dot_bit(0, 0);
                }
                let ch4 = char::from_u32(0x2800 + bits as u32).unwrap_or(' ');
                *cell = Cell {
                    ch: ch4,
                    rgb: cell_rgb(o, &c),
                };
            }
        }
    });
    scr.emit(o, &cells, cw, ch, out);
}

/// Draw a quiet one-line status bar in the last terminal row. It deliberately
/// has no background or inverse attribute, and uses a dim neutral grey so it
/// does not compete with the scene. The measured frame rate leads the line
/// in a lighter grey - the one number that changes every frame deserves the
/// one bit of emphasis the bar has.
fn draw_status(o: &Opt, out: &mut String, t: f64, paused: bool, fps: f64) {
    let state = if paused { " | paused" } else { "" };
    let head = format!(" fps:{:.1}", fps);
    let text = format!(
        " | funnel:{} | speed:{:.2}x | zoom:{:.2} | tilt:{:+.1}° | orbit:{:+.1}°/s{} | t:{:.1}",
        o.funnel.name(),
        o.speed,
        o.zoom,
        o.tilt,
        o.orbit,
        state,
        t
    );
    let width = o.cols.max(1);
    let head_vis: String = head.chars().take(width).collect();
    let tail_vis: String = text
        .chars()
        .take(width - head_vis.chars().count())
        .collect();
    out.push_str("\x1b[");
    push_u32(out, o.rows.max(1) as u32);
    out.push_str(";1H");
    if o.color {
        out.push_str("\x1b[38;2;168;168;168m");
    }
    out.push_str(&head_vis);
    if o.color {
        out.push_str("\x1b[38;2;72;72;72m");
    }
    out.push_str(&tail_vis);
    for _ in head_vis.chars().count() + tail_vis.chars().count()..width {
        out.push(' ');
    }
    if o.color {
        out.push_str("\x1b[0m");
    }
}

/// Sixel sky cutoff: pixels dimmer than this are dropped outright. The faint
/// specks it removes are near-invisible but each one costs a colour strip and
/// broken run-length compression in every band it touches.
const SIXEL_SKY_CUT: f64 = 0.17;

fn draw_sixel(o: &Opt, f: &Frame, out: &mut String, t: f64, paused: bool, fps: f64) {
    // Sixel paints device pixels, so map the ray grid up to the target size
    // (nearest neighbour) instead of drawing the picture a fifth of the
    // window wide. Reserve the last terminal row for the status line.
    let tw = o.tpw;
    let cell_h = (o.tph / o.rows.max(1)).max(1);
    let th = o.tph.saturating_sub(cell_h).max(6);
    let mut col: Vec<usize> = vec![0; tw];
    (0..tw).for_each(|x| {
        col[x] = x * f.w / tw.max(1);
    });

    // Quantise every source pixel to its cube register once (255 = dropped =
    // sky). Two kinds of sky detail are pure waste and get cut here:
    //  - pixels that land in the black register anyway (level 0 in every
    //    channel) - they would paint black over the black underlay;
    //  - anything dimmer than SIXEL_SKY_CUT: specks that are barely visible
    //    yet each one opens a colour strip and shreds the RLE runs of the
    //    whole band.
    let mut idx_map = vec![255u8; f.w * f.h];
    for (p, q) in f.px.iter().zip(idx_map.iter_mut()) {
        if lum(p) >= SIXEL_SKY_CUT {
            let i = pixel_index(p);
            if i != 0 {
                *q = i as u8;
            }
        }
    }

    // register 0 = black, registers 16..231 = a 6x6x6 colour cube.
    //
    // Registers use the RGB form `#idx;2;r;g;b` with components in per cent
    // (0..=100).  The colourspace selector `2;` is mandatory: a bare
    // `#idx;r;g;b` makes a strict parser read r as the colourspace and throw the
    // whole image away, and components above 100 are rejected just as loudly.
    let mut used = [false; 216];
    for &q in &idx_map {
        if q != 255 {
            used[q as usize] = true;
        }
    }
    out.push_str("\x1bPq#0;2;0;0;0");
    for b in 0..6usize {
        for g in 0..6usize {
            for r in 0..6usize {
                if !used[36 * r + 6 * g + b] {
                    continue;
                }
                let idx = 16 + 36 * r + 6 * g + b;
                out.push('#');
                push_u32(out, idx as u32);
                out.push_str(";2");
                for c in [r, g, b] {
                    out.push(';');
                    push_u32(out, (c * 20) as u32);
                }
            }
        }
    }

    let bands = th.div_ceil(6);
    let mut row: Vec<u8> = vec![0; tw];
    // one reusable strip buffer per cube colour + the list of colours in use:
    // building a map per band (and allocating a mask per strip) costs more
    // than the encoding itself
    let mut strips: Vec<Vec<u8>> = (0..216).map(|_| Vec::new()).collect();
    let mut seen = [false; 216];
    let mut used_cols: Vec<usize> = Vec::new();
    for band in 0..bands {
        for &c in &used_cols {
            seen[c] = false;
        }
        used_cols.clear();
        for sy in 0..6usize {
            let ty = band * 6 + sy;
            if ty >= th {
                break;
            }
            let y = ty * f.h / th.max(1);
            let srow = &idx_map[y * f.w..(y + 1) * f.w];
            for x in 0..tw {
                let q = srow[col[x]];
                if q == 255 {
                    continue;
                }
                let idx = q as usize;
                if !seen[idx] {
                    seen[idx] = true;
                    used_cols.push(idx);
                    strips[idx].clear();
                    strips[idx].resize(tw, 0);
                }
                strips[idx][x] |= 1 << sy;
            }
        }
        // opaque black background first so nothing of the previous frame leaks
        row.fill(0x3f);
        out.push_str("#0");
        write_row(out, &row);
        out.push('$');
        for idx in used_cols.iter() {
            out.push('#');
            push_u32(out, (16 + idx) as u32);
            write_row(out, &strips[*idx]);
            // `$` rewinds to the left edge of the band: without it the next
            // colour strip would be drawn to the right of this one.
            out.push('$');
        }
        if band + 1 < bands {
            // `-` is the raster advance: next band, left margin.
            out.push('-');
        }
    }
    out.push_str("\x1b\\");
    draw_status(o, out, t, paused, fps);
}

/// 216-cube index of a colour (0..215)
fn pixel_index(p: &[f64; 3]) -> usize {
    let q = |x: f64| ((x * 5.0).round() as usize).clamp(0, 5);
    36 * q(p[0]) + 6 * q(p[1]) + q(p[2])
}

/// Emit one colour row of a sixel band. Runs longer than one pixel are
/// compressed with the `!` repeat introducer - a bare digit after a register
/// number or after data is not a count, it is a parameter and gets swallowed.
/// Transparent runs (all bits clear, `?`) must still be written out: dropping
/// them would slide everything after them to the left.
fn write_row(out: &mut String, row: &[u8]) {
    let mut x = 0;
    while x < row.len() {
        let v = row[x];
        let mut n = 1;
        while x + n < row.len() && row[x + n] == v {
            n += 1;
        }
        if n > 1 {
            out.push('!');
            push_u32(out, n as u32);
        }
        out.push((0x3f + v) as char);
        x += n;
    }
}

// ---------------------------------------------------------------- main

fn main() {
    let mut o = parse_opt();
    let mut out = String::with_capacity(1 << 22);

    if let Some(n) = o.one_shot {
        let t = n / o.fps * o.speed;
        let final_azi = o.orbit.to_radians() * t;
        let mut f = Frame {
            w: 0,
            h: 0,
            px: Vec::new(),
        };
        let mut cache = GeoCache::new();
        cache.prime(&o);
        let mut stars = Stars::new();
        if o.super_star {
            stars.spawn_super(o.origin, o.azi, o.tilt, o.funnel);
            stars.advance(t);
        } else if o.big_star || o.star {
            stars.spawn(o.big_star, &o);
            stars.advance(t);
        }
        o.azi = final_azi;
        let glows = glow_list(&stars, o.azi);
        render_frame(&o, t, &mut f, &mut cache, &glows);
        let mut scr = Screen::new();
        draw_into(&o, &f, &mut out, &mut scr, t, false, o.fps);
        println!("{out}");
        return;
    }

    let _raw = RawTerm::new();
    let mut so = std::io::stdout();
    print!("\x1b[?1049h\x1b[?25l\x1b[2J"); // alt screen, hide cursor, clear
    let _ = so.flush();

    let mut t = 0.0;
    let mut paused = false;
    let mut drawn = false; // a paused, up-to-date frame needs no work at all
    let mut scr = Screen::new();
    let mut f = Frame {
        w: 0,
        h: 0,
        px: Vec::new(),
    };
    let mut cache = GeoCache::new();
    cache.prime(&o);
    let mut stars = Stars::new();
    if o.super_star {
        stars.spawn_super(o.origin, o.azi, o.tilt, o.funnel);
    } else if o.big_star || o.star {
        stars.spawn(o.big_star, &o);
    }
    let mut last = Instant::now();
    // Measured frame rate of actually drawn frames, smoothed with an EMA so
    // the status readout does not flicker. While paused nothing is produced,
    // so the reading freezes; resuming re-arms it without counting the gap.
    let mut fps = 0.0f64;
    let mut last_draw: Option<Instant> = None;
    loop {
        let step = last.elapsed().as_secs_f64();
        last = Instant::now();
        if !paused {
            t += step * o.speed;
            o.azi += o.orbit.to_radians() * step * o.speed;
            stars.advance(step * o.speed);
        }
        if !paused || !drawn {
            let glows = glow_list(&stars, o.azi);
            render_frame(&o, t, &mut f, &mut cache, &glows);
            out.clear();
            out.push_str("\x1b[H");
            draw_into(&o, &f, &mut out, &mut scr, t, paused, fps);
            let _ = so.write_all(out.as_bytes());
            let _ = so.flush();
            let now = Instant::now();
            if let Some(prev) = last_draw {
                let dt = now.duration_since(prev).as_secs_f64();
                if dt > 1e-9 {
                    fps += (1.0 / dt - fps) * 0.2;
                }
            }
            last_draw = Some(now);
            drawn = true;
        }

        // frame pacing: sleep the remaining budget in one call. Keys are
        // polled once per frame either way (see below), so coarse 1 ms
        // sleep-polling here bought nothing - it just burned several
        // milliseconds of CPU per frame waking up over and over.
        let budget = Duration::from_secs_f64(1.0 / o.fps);
        let left = budget.saturating_sub(last.elapsed());
        if !left.is_zero() {
            std::thread::sleep(left);
        }
        if let Some(k) = poll_key() {
            match k {
                Key::Esc | Key::Char('q') | Key::Char('c') => break,
                Key::Char(' ') => {
                    paused = !paused;
                    last_draw = None;
                }
                Key::Char('+') | Key::Char('=') => {
                    o.zoom = (o.zoom * 1.15).clamp(0.25, 6.0);
                    drawn = false;
                }
                Key::Char('-') | Key::Char('_') => {
                    o.zoom = (o.zoom / 1.15).clamp(0.25, 6.0);
                    drawn = false;
                }
                // simulation speed: < slower, > faster. Terminals report
                // whatever the current layout puts on the key, so match the
                // unshifted latin , / . and the cyrillic letters living on
                // the same physical keys (б / ю, shifted Б / Ю) - neither
                // shift state nor layout changes the binding
                Key::Char('<') | Key::Char(',') | Key::Char('б') | Key::Char('Б') => {
                    o.speed = step_speed(o.speed, false);
                }
                Key::Char('>') | Key::Char('.') | Key::Char('ю') | Key::Char('Ю') => {
                    o.speed = step_speed(o.speed, true);
                }
                // tilt the camera over/under the disk plane; tilt is part of
                // the geometry cache key, so the next frame re-traces
                Key::Up => {
                    o.tilt = (o.tilt + TILT_STEP).min(TILT_LIMIT);
                    drawn = false;
                }
                Key::Down => {
                    o.tilt = (o.tilt - TILT_STEP).max(-TILT_LIMIT);
                    drawn = false;
                }
                // orbit rate: faster / slower (through zero into reverse).
                // The azimuth itself is accumulated state, so the view never
                // jumps when the rate changes. The rate is not part of the
                // geometry cache key (axial symmetry), so this costs nothing.
                Key::Right => {
                    o.orbit = (o.orbit + ORBIT_STEP).min(ORBIT_MAX);
                }
                Key::Left => {
                    o.orbit = (o.orbit - ORBIT_STEP).max(-ORBIT_MAX);
                }
                // drop another star into the well / clear the ones in flight
                Key::Char('s') => {
                    stars.spawn(false, &o);
                    drawn = false;
                }
                Key::Char('S') => {
                    stars.spawn(true, &o);
                    drawn = false;
                }
                Key::Char('b') => {
                    stars.spawn_super(o.origin, o.azi, o.tilt, o.funnel);
                    drawn = false;
                }
                Key::Char('x') => {
                    stars.clear();
                    drawn = false;
                }
                _ => {}
            }
        }
    }
    print!("\x1b[0m\x1b[0m\x1b[?25h\x1b[?1049l");
    let _ = so.flush();
}

fn draw_into(
    o: &Opt,
    f: &Frame,
    out: &mut String,
    scr: &mut Screen,
    t: f64,
    paused: bool,
    fps: f64,
) {
    match o.mode {
        Mode::Ascii => {
            draw_ascii(o, f, out, scr);
            draw_status(o, out, t, paused, fps);
        }
        Mode::Braille => {
            draw_braille(o, f, out, scr);
            draw_status(o, out, t, paused, fps);
        }
        Mode::Sixel => draw_sixel(o, f, out, t, paused, fps),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_stars(sc: f64, speed: f64, p: V3) -> Stars {
        let mut stars = Stars::new();
        let d = p.norm();
        let a = V3::new(-d.y, d.x, 0.0).norm();
        stars.live.push(Infall::spawn(
            1,
            sc,
            d,
            a,
            V3::new(0.0, 0.0, 1.0),
            Some(speed),
        ));
        stars
    }

    fn evolve(mut stars: Stars, total: f64, step: f64) -> Stars {
        let mut elapsed = 0.0;
        while elapsed < total - 1e-12 {
            let h = step.min(total - elapsed);
            stars.advance(h);
            elapsed += h;
        }
        stars
    }

    fn stripping_stars() -> Stars {
        let p = V3::new(2.2, 0.0, 0.0);
        let v = V3::new(0.0, 2.5, 0.0);
        let mut stars = Stars::new();
        stars.live.push(Infall {
            p,
            v,
            tr: vec![Trail::shed(p, v)],
            tr_at: p,
            alive: true,
            sc: 3.0,
            m: 1.0,
            drag: BIG_DRAG,
            ns: 0,
            debt: 0.0,
            parked: false,
            funnel: FunnelMode::Current,
        });
        stars
    }

    #[test]
    fn empty_glow_frame_clears_previous_deposition() {
        let mut cache = GlowCache::new();
        cache.st = vec![[1.0, 0.5, 0.25]];
        cache.lit.push(0);

        deposit_glows(&[], &[], &[], 1, &mut cache, false, 1);

        assert_eq!(cache.st[0], [0.0; 3]);
        assert!(cache.lit.is_empty());
    }

    #[test]
    fn fast_delta_tracks_source_motion_and_clear() {
        let o = Opt {
            mode: Mode::Ascii,
            funnel: FunnelMode::Spiral,
            fps: 30.0,
            zoom: 1.0,
            speed: 1.0,
            orbit: 0.0,
            azi: 0.0,
            tilt: CAM_TILT,
            shift: 0.0,
            color: false,
            cols: 40,
            rows: 20,
            tpw: 80,
            tph: 80,
            rays: 40_000,
            threads: 1,
            one_shot: None,
            star: false,
            big_star: false,
            super_star: false,
            origin: None,
            star_speed: None,
            ramp: " .·:;+=*xX#%@█".chars().collect(),
        };
        let glow = |x| {
            vec![Glow {
                p: V3::new(x, 0.2, 0.0),
                c: [100.0, 75.0, 45.0],
                sig: 1.0,
                fast: true,
            }]
        };
        let first = glow(2.8);
        let moved = glow(3.1);
        let mut delta_frame = Frame {
            w: 0,
            h: 0,
            px: Vec::new(),
        };
        let mut delta_cache = GeoCache::new();
        render_frame(&o, 0.0, &mut delta_frame, &mut delta_cache, &first);
        assert!(delta_cache.glow.fast_accum.iter().any(|&e| e > 0.0));
        render_frame(&o, 1.0, &mut delta_frame, &mut delta_cache, &moved);

        let mut fresh_frame = Frame {
            w: 0,
            h: 0,
            px: Vec::new(),
        };
        let mut fresh_cache = GeoCache::new();
        render_frame(&o, 1.0, &mut fresh_frame, &mut fresh_cache, &moved);
        let max_error = delta_frame
            .px
            .iter()
            .zip(&fresh_frame.px)
            .map(|(a, b)| {
                (a[0] - b[0])
                    .abs()
                    .max((a[1] - b[1]).abs())
                    .max((a[2] - b[2]).abs())
            })
            .fold(0.0, f64::max);
        assert!(max_error < 1e-5, "delta error: {max_error}");

        render_frame(&o, 2.0, &mut delta_frame, &mut delta_cache, &[]);
        let residual = delta_cache
            .glow
            .fast_accum
            .iter()
            .map(|&energy| energy.abs())
            .fold(0.0, f32::max);
        println!("delta residual: {residual}");
        assert!(residual < 1e-5);
    }

    #[test]
    fn tone_table_stays_close_to_the_curve() {
        build_tone();
        let mut max_error = 0.0f64;
        for i in 0..100_001 {
            let x = TONE_MAX * i as f64 / 100_000.0;
            let exact = (1.0 - (-x).exp()).powf(1.0 / 1.85);
            max_error = max_error.max((tone(x) - exact).abs());
        }
        println!("tone LUT max error: {max_error:.6}");
        assert!(max_error < 0.01, "tone LUT max error: {max_error}");
    }

    #[test]
    fn gaussian_segment_integral_matches_numerical_quadrature() {
        // covers: glow mid-segment, glow near an end of a segment much
        // longer than sig (the old table clamped its length axis at 6 sig
        // and halved this), and glow just past an end (where the old table
        // hit its hard zero cutoff)
        let cases = [
            (
                V3::new(-0.8, 0.2, -0.1),
                V3::new(1.1, -0.3, 0.4),
                V3::new(0.15, 0.45, 0.2),
                0.38,
            ),
            (
                V3::new(-6.0, 0.2, -0.1),
                V3::new(6.0, -0.3, 0.4),
                V3::new(3.0, 0.2328, 0.4328),
                0.38,
            ),
            (
                V3::new(-6.0, 0.2, -0.1),
                V3::new(6.0, -0.3, 0.4),
                V3::new(6.6094, -0.0426, 0.7082),
                0.38,
            ),
        ];
        let erf_t = ERF_LUT.get_or_init(build_erf_lut);
        for (p0, p1, gp, sig) in cases {
            let seg = Seg::from_endpoints(p0, p1, 0.37, 0).expect("test segment in range");
            let glow = Glow {
                p: gp,
                c: [1.0; 3],
                sig,
                fast: false,
            };
            let n = 200_000;
            let mut numerical = 0.0;
            for i in 0..n {
                let u = (i as f64 + 0.5) / n as f64;
                let d = p0 + (p1 - p0) * u - glow.p;
                numerical += (-d.len2() / (2.0 * glow.sig * glow.sig)).exp();
            }
            numerical *= (p1 - p0).len() / n as f64 * seg.tr as f64;

            let analytic = segment_glow_weight(&glow, &seg, erf_t);
            let error = (analytic - numerical).abs();
            assert!(
                error < 1e-6,
                "integral error: {error}, analytic={analytic}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn disk_crossing_splits_glow_transmittance() {
        let cam = Cam::new(0.0, 12.0_f64.to_radians());
        let target = V3::new(0.0, 0.0, 5.0);
        let mut geo = Geo::empty();
        let mut segs = Vec::new();
        trace_geo(
            &cam,
            (target - cam.p).norm(),
            3.0,
            10.0,
            &mut geo,
            0,
            &mut segs,
        );

        let split = segs.windows(2).any(|pair| {
            let a = &pair[0];
            let b = &pair[1];
            // segment a ends at its midpoint plus half its axis vector
            let ay = a.m[1] as f64 + 0.5 * a.dl as f64 * a.u[1] as f64;
            ay.abs() < 1e-5
                && (b.m[1] as f64 - 0.5 * b.dl as f64 * b.u[1] as f64).abs() < 1e-5
                && (b.tr as f64 - a.tr as f64 * DISK_OPA).abs() < 1e-6
        });

        assert!(geo.n > 0);
        assert!(split);
    }

    #[test]
    #[ignore = "perf smoke: cargo test --release glow_deposit_throughput -- --ignored --nocapture"]
    fn glow_deposit_throughput() {
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let next = |rng: &mut u64| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            *rng as f64 / u64::MAX as f64
        };
        let v3 = |rng: &mut u64| {
            V3::new(
                next(rng) * 10.0 - 5.0,
                next(rng) * 10.0 - 5.0,
                next(rng) * 10.0 - 5.0,
            )
        };
        let mut segs = Vec::with_capacity(120_000);
        for i in 0..120_000 {
            let p0 = v3(&mut rng);
            let p1 = p0 + v3(&mut rng) * 0.5;
            let tr = 0.2 + 0.8 * next(&mut rng);
            if let Some(s) = Seg::from_endpoints(p0, p1, tr, i % 4096) {
                segs.push(s);
            }
        }
        let mut bin_off = Vec::new();
        let mut bins = Vec::new();
        let mut cur = Vec::new();
        build_bins(&mut segs, &mut bin_off, &mut bins, &mut cur);
        let glows: Vec<Glow> = (0..40)
            .map(|_| Glow {
                p: v3(&mut rng) * 0.8,
                c: [1.0; 3],
                sig: 0.38 + 0.5 * next(&mut rng),
                fast: false,
            })
            .collect();
        let mut acc = vec![0.0f64; 4096];
        let t0 = std::time::Instant::now();
        for gl in &glows {
            deposit_one(gl, &segs, &bin_off, &mut |px, add| {
                acc[px] += add[0] + add[1] + add[2];
            });
        }
        let dt = t0.elapsed();
        let total: f64 = acc.iter().sum();
        println!(
            "deposit 40 glows x 120k segs: {dt:?} ({:.0} us/glow), sum={total:.3}",
            dt.as_micros() / 40
        );
    }

    #[test]
    #[ignore = "perf smoke: cargo test --release render_frame_throughput -- --ignored --nocapture"]
    fn render_frame_throughput() {
        let o = Opt {
            mode: Mode::Sixel,
            funnel: FunnelMode::Spiral,
            fps: 240.0,
            zoom: 1.0,
            speed: 1.0,
            orbit: 0.0,
            azi: 0.0,
            tilt: CAM_TILT,
            shift: 0.0,
            color: true,
            cols: 80,
            rows: 24,
            tpw: 800,
            tph: 480,
            rays: 200_000,
            threads: 1,
            one_shot: None,
            star: false,
            big_star: false,
            super_star: true,
            origin: None,
            star_speed: None,
            ramp: " .·:;+=*xX#%@█".chars().collect(),
        };
        let mut stars = Stars::new();
        stars.spawn_super(None, 0.0, o.tilt, FunnelMode::Spiral);
        // Advance to a deliberately dense, steady superstar funnel before
        // tracing. The measured loop below replays the same cached geometry
        // and source list, so variants are compared at identical state.
        stars.advance(256.0);
        let glows = glow_list(&stars, 0.0);
        let mut frame = Frame {
            w: 0,
            h: 0,
            px: Vec::new(),
        };
        let mut cache = GeoCache::new();
        cache.prime(&o);
        render_frame(&o, 123.0, &mut frame, &mut cache, &glows);
        let t0 = Instant::now();
        for _ in 0..64 {
            render_frame(&o, 123.0, &mut frame, &mut cache, &glows);
        }
        let dt = t0.elapsed();
        println!(
            "render 64 dense frames: {dt:?} ({:.0} us/frame), streams={}, glows={}, segs={}, fast_segs={}, field_slots={}",
            dt.as_micros() as f64 / 64.0,
            stars.streams.len(),
            glows.len(),
            cache.segs.len(),
            cache.glow.fast_segs.len(),
            cache.glow.bin_fast.len(),
        );
        std::hint::black_box(frame.px[0]);
    }

    #[test]
    fn shed_trail_loses_angular_momentum_and_falls_inward() {
        let p = V3::new(10.0, 0.0, 0.0);
        let circular = (INFALL_GM * p.len()).sqrt() / (p.len() - RS);
        let mut trail = Trail::shed(p, V3::new(0.0, circular, 0.0));

        assert!(trail.advance(2.0, INFALL_GM));
        assert!(trail.p.len() < p.len());
    }

    #[test]
    fn large_time_jump_does_not_pre_age_a_remnant() {
        let one_jump = evolve(test_stars(1.0, 0.72, V3::new(1.0, 0.0, 0.0)), 20.0, 20.0);
        let many_frames = evolve(
            test_stars(1.0, 0.72, V3::new(1.0, 0.0, 0.0)),
            20.0,
            1.0 / 120.0,
        );
        let one_brightness = one_jump.rem.first().expect("one-jump remnant").b;
        let frame_brightness = many_frames.rem.first().expect("frame-step remnant").b;

        assert!(one_brightness > 0.8);
        assert!((one_brightness - frame_brightness).abs() < 0.05);
    }

    #[test]
    fn cooling_stream_integrates_until_its_lifetime() {
        let mut stream = Stream {
            p: V3::new(10.0, 0.0, 0.0),
            v: V3::new(0.0, 1.0, 0.0),
            w: 0.02,
            age: STREAM_LIFE - 0.01,
            life: STREAM_LIFE,
            drag: INFALL_DRAG,
            bri: STREAM_BRI,
            sig: STREAM_SIG,
            fun: false,
            funnel: FunnelMode::Current,
            group: 0,
        };
        let before = stream.p;

        assert!(!stream.advance(0.02, INFALL_GM));
        assert!((stream.age - STREAM_LIFE).abs() < 1e-12);
        assert!((stream.p - before).len() > 0.0);
    }

    #[test]
    fn stripping_conserves_unabsorbed_mass() {
        let mut stars = stripping_stars();
        let mut infall = stars.live.remove(0);
        infall.p = V3::new(1.3, 0.0, 0.0);
        let mut streams = Vec::new();

        infall.advance(0.02, INFALL_GM, &mut streams);
        let represented = infall.m + infall.debt + streams.iter().map(|s| s.w).sum::<f64>();

        assert!((represented - 1.0).abs() < 1e-12);
    }

    #[test]
    fn simultaneous_infalls_keep_separate_remnants() {
        let make = |p: V3| Infall {
            p,
            v: p.norm() * -1.0,
            tr: vec![Trail::shed(p, p.norm() * -1.0)],
            tr_at: p,
            alive: true,
            sc: 1.0,
            m: 1.0,
            drag: INFALL_DRAG,
            ns: 0,
            debt: 0.0,
            parked: false,
            funnel: FunnelMode::Current,
        };
        let mut stars = Stars::new();
        stars.live.push(make(V3::new(1.16, 0.0, 0.0)));
        stars.live.push(make(V3::new(0.0, 1.16, 0.0)));

        stars.advance(0.02);

        assert_eq!(stars.rem.len(), 2);
    }

    #[test]
    fn disk_uses_local_orbital_speed_not_its_square() {
        let (beta, beta2) = local_orbit_beta(3.05);

        assert!((beta2 - 0.24390243902439027).abs() < 1e-14);
        assert!((beta - beta2.sqrt()).abs() < 1e-14);
        assert!((beta - 0.49386479832479485).abs() < 1e-14);
    }

    #[test]
    fn escaped_ray_is_stored_as_a_direction() {
        let cam = Cam::new(0.0, 12.0_f64.to_radians());
        let mut geo = Geo::empty();
        let mut segs = Vec::new();
        trace_geo(
            &cam,
            cam.ray(0, 0, 8, 8, 1.0, 0.0),
            3.0,
            10.0,
            &mut geo,
            0,
            &mut segs,
        );

        assert!(geo.sky.is_some());
        assert!((geo.esc.len() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn orbiting_sky_matches_a_full_lookup() {
        let esc = V3::new(0.31, -0.27, 0.91).norm();
        let orb = 0.73_f64;
        let ctx = ShCtx {
            t: 1.2345,
            orb,
            c: orb.cos(),
            s: orb.sin(),
            ppc0: 3.0,
        };
        let mut geo = Geo::empty();
        geo.sky = Some(stars(esc, ctx.ppc0));
        geo.esc = esc;
        let d = V3::new(
            esc.x * ctx.c - esc.z * ctx.s,
            esc.y,
            esc.x * ctx.s + esc.z * ctx.c,
        );
        let mut expected = sky_rgb(&stars(d, ctx.ppc0), ctx.t);
        for channel in &mut expected {
            *channel = channel.clamp(0.0, 1.0);
        }

        assert_eq!(shade(&geo, &[0.0; 3], &ctx), expected);
    }

    #[test]
    fn galactic_band_texture_tracks_procedural_field() {
        let mut max_error = 0.0_f64;
        let mut total_error = 0.0_f64;
        let mut samples = 0usize;
        for yi in 0..81 {
            let y = -0.74 + 1.48 * yi as f64 / 80.0;
            let xz = (1.0 - y * y).sqrt();
            for ai in 0..256 {
                let a = TAU * ai as f64 / 256.0;
                let d = V3::new(xz * a.cos(), y, xz * a.sin());
                let error = (band_sample(d) - band_exact(d)).abs();
                max_error = max_error.max(error);
                total_error += error;
                samples += 1;
            }
        }

        let mean_error = total_error / samples as f64;
        assert!(
            max_error < 1e-4 && mean_error < 1e-5,
            "band error: max={max_error}, mean={mean_error}"
        );
    }

    #[test]
    fn named_origins_match_screen_sides() {
        let cam = Cam::new(0.0, 0.0);
        let (left, _, _) = Origin::Left.basis(&cam);
        let (right, _, _) = Origin::Right.basis(&cam);
        let (front, _, _) = Origin::Front.basis(&cam);
        let (back, _, _) = Origin::Back.basis(&cam);
        let toward_camera = cam.p.norm();

        assert!(left.dot(cam.r) < 0.0);
        assert!(right.dot(cam.r) > 0.0);
        assert!(front.dot(toward_camera) > 0.0);
        assert!(back.dot(toward_camera) < 0.0);
    }

    #[test]
    fn tidal_profile_uses_dense_sheared_debris() {
        assert!(FunnelMode::Tidal.shed_weight() <= FunnelMode::Current.shed_weight());
        assert!(FunnelMode::Tidal.stream_group() <= FunnelMode::Current.stream_group());
        assert!(FunnelMode::Tidal.stream_gain(2.0) < tide(2.0));

        let mut stars = Stars::new();
        stars.spawn_super(None, 0.0, CAM_TILT, FunnelMode::Tidal);
        stars.advance(4.0);

        assert!(stars.streams.len() > 8, "debris stream too sparse");
        assert!(
            stars
                .streams
                .windows(2)
                .any(|pair| (pair[0].v - pair[1].v).len() > 1e-6),
            "tidal parcels all follow the same ballistic path"
        );
    }

    #[test]
    fn tidal_glow_splits_separated_or_incomplete_groups() {
        let parcel = |p| Stream {
            p,
            v: V3::new(0.0, 1.0, 0.0),
            w: TIDAL_SHED_W,
            group: 0,
            age: 0.0,
            life: SUPER_STREAM_LIFE,
            drag: FunnelMode::Tidal.stream_drag(),
            bri: FunnelMode::Tidal.stream_brightness(),
            sig: STREAM_SIG,
            fun: true,
            funnel: FunnelMode::Tidal,
        };
        let mut stars = Stars::new();
        stars.streams = vec![
            parcel(V3::new(10.0, 0.0, 0.0)),
            parcel(V3::new(10.5, 0.0, 0.0)),
            parcel(V3::new(10.5, 0.5, 0.0)),
            parcel(V3::new(11.0, 1.0, 0.0)),
        ];

        assert_eq!(
            glow_list(&stars, 0.0).len(),
            1,
            "compact group should stay merged"
        );

        stars.streams[3].p = V3::new(14.0, 0.0, 0.0);
        assert_eq!(
            glow_list(&stars, 0.0).len(),
            4,
            "separated tidal branches must not share one centroid"
        );

        stars.streams.pop();
        assert_eq!(
            glow_list(&stars, 0.0).len(),
            3,
            "an incomplete group must not jump when a parcel disappears"
        );
    }

    #[test]
    fn super_star_parks_and_bleeds_slowly_through_the_funnel() {
        let mut stars = Stars::new();
        stars.spawn_super(None, 0.0, CAM_TILT, FunnelMode::Current);
        assert_eq!(stars.live.len(), 1);
        // a second superstar never joins the first
        stars.spawn_super(None, 0.0, CAM_TILT, FunnelMode::Current);
        assert_eq!(stars.live.len(), 1);

        // ten simulated minutes: the donor never moves an inch - the frame
        // holds still - while the funnel keeps drinking from it
        let p0 = stars.live[0].p;
        let st = evolve(stars, 600.0, 0.1);
        let inf = &st.live[0];
        assert!(inf.alive, "the star should outlast the drain");
        assert!((inf.p - p0).len() < 1e-12, "the parked star drifted");
        assert!(inf.m > 0.6, "drained too fast: m={}", inf.m);
        assert!(st.streams.len() > 20, "funnel too sparse");
        let flown: f64 = st.streams.iter().map(|s| s.w).sum();
        assert!(inf.m + flown <= 1.0 + 1e-9);
        assert!(1.0 - inf.m > flown, "some mass must already be swallowed");
        assert!(st.rem.is_empty(), "the hole must not flash when it feeds");
    }

    #[test]
    fn super_star_origin_selects_the_parking_side() {
        let cam = Cam::new(0.0, CAM_TILT.to_radians());
        let left = super_park(Some(Origin::Left), 0.0, CAM_TILT);
        let right = super_park(Some(Origin::Right), 0.0, CAM_TILT);
        let top = super_park(Some(Origin::Top), 0.0, CAM_TILT);
        let bottom = super_park(Some(Origin::Bottom), 0.0, CAM_TILT);
        let front = super_park(Some(Origin::Front), 0.0, CAM_TILT);
        let back = super_park(Some(Origin::Back), 0.0, CAM_TILT);

        assert!(left.dot(cam.r) < 0.0 && right.dot(cam.r) > 0.0);
        assert!(top.dot(cam.u) > 0.0 && bottom.dot(cam.u) < 0.0);
        assert!(front.dot(cam.p.norm()) > 0.0 && back.dot(cam.p.norm()) < 0.0);
        assert!((super_park(None, 0.0, CAM_TILT) - SUPER_PARK).len() < 1e-12);
    }

    #[test]
    fn superstar_glow_shrinks_with_remaining_mass() {
        let mut stars = Stars::new();
        stars.spawn_super(None, 0.0, CAM_TILT, FunnelMode::Current);
        let full = glow_list(&stars, 0.0);
        stars.live[0].m = 0.125;
        let reduced = glow_list(&stars, 0.0);
        assert!(reduced[0].sig < full[0].sig);
        assert!(reduced[0].c[0] < full[0].c[0]);
        assert!(reduced[1].sig < full[1].sig);
    }

    #[test]
    fn super_star_near_side_is_stretched_toward_the_hole() {
        let mut stars = Stars::new();
        stars.spawn_super(None, 0.0, CAM_TILT, FunnelMode::Current);
        let glows = glow_list(&stars, 0.0);
        assert_eq!(glows.len(), 1 + SUPER_STRETCH_N);

        let radial = SUPER_PARK.norm();
        let head_projection = glows[0].p.dot(radial);
        let deepest_projection = glows
            .iter()
            .skip(1)
            .map(|glow| glow.p.dot(radial))
            .fold(f64::INFINITY, f64::min);
        assert!(
            deepest_projection < head_projection - SUPER_STRETCH_LEN,
            "near side was not elongated toward the hole"
        );
    }

    #[test]
    fn funnel_parcels_take_the_long_way_round() {
        // one parcel launched exactly the way the parked donor sheds it: it
        // must swing a long arc (the funnel is the show) yet still arrive
        // within its own lifetime, so no shed mass is lost to cooling
        let r = SUPER_PARK.len();
        let vc = (INFALL_GM * r).sqrt() / (r - RS);
        let rad = SUPER_PARK.norm();
        let tan = V3::new(0.0, 1.0, 0.0).cross(rad).norm();
        let mut s = Stream {
            p: SUPER_PARK * (1.0 - 0.02),
            v: tan * (SUPER_FUN_F * vc) - rad * (SUPER_FUN_G * vc),
            w: SUPER_SHED_W,
            age: 0.0,
            life: SUPER_STREAM_LIFE,
            drag: SUPER_STREAM_DRAG,
            bri: SUPER_STREAM_BRI,
            sig: STREAM_SIG,
            fun: true,
            funnel: FunnelMode::Current,
            group: 0,
        };
        // near the horizon the PW speeds are so high that a 0.25 s sample
        // can hop straight past the swallow radius, so the early exit is
        // the arrival proof, not a sampled minimum radius
        let mut t = 0.0;
        const TEST_HORIZON: f64 = 300.0;
        while t < TEST_HORIZON {
            if !s.advance(0.25, INFALL_GM) {
                break;
            }
            t += 0.25;
        }
        assert!(t > 20.0, "fell in too fast: t={t}");
        assert!(t < TEST_HORIZON, "parcel did not arrive: t={t}");
    }

    #[test]
    fn swallowed_parcels_do_not_flash_at_the_hole() {
        let mut stars = Stars::new();
        stars.streams.push(Stream {
            p: V3::new(2.0, 0.0, 0.0),
            v: V3::new(0.0, 0.0, 0.4),
            w: 0.01,
            age: 0.0,
            life: 100.0,
            drag: INFALL_DRAG,
            bri: STREAM_BRI,
            sig: STREAM_SIG,
            fun: false,
            funnel: FunnelMode::Current,
            group: 0,
        });
        stars.advance(0.5);
        assert!(stars.streams.is_empty());
        assert!(stars.rem.is_empty());
    }

    #[test]
    fn speed_keys_step_multiplicatively_and_clamp() {
        assert!((step_speed(1.0, true) - SPEED_STEP).abs() < 1e-12);
        assert!((step_speed(1.0, false) - 1.0 / SPEED_STEP).abs() < 1e-12);
        assert_eq!(step_speed(1000.0, true), SPEED_MAX);
        assert_eq!(step_speed(1e-9, false), SPEED_MIN);
        // a backwards-running simulation stays backwards
        assert!((step_speed(-2.0, true) + 2.0 * SPEED_STEP).abs() < 1e-12);
        assert!(step_speed(-0.5, false) < 0.0);
    }

    #[test]
    fn speed_keys_follow_the_physical_key_in_any_layout() {
        // a latin layout sends , / . (shifted: < / >); a russian layout sends
        // the cyrillic letters living on the same physical keys as two-byte
        // UTF-8 - first_char must yield the character, not the first byte
        assert_eq!(first_char(b","), ',');
        assert_eq!(first_char(b"<"), '<');
        assert_eq!(first_char(b"."), '.');
        assert_eq!(first_char(b">"), '>');
        // invalid UTF-8 falls back to the raw byte instead of garbage
        assert_eq!(first_char("\u{0431}".as_bytes()), '\u{0431}'); // б
        assert_eq!(first_char("\u{0411}".as_bytes()), '\u{0411}'); // Б
        assert_eq!(first_char("\u{044e}".as_bytes()), '\u{044e}'); // ю
        assert_eq!(first_char("\u{042e}".as_bytes()), '\u{042e}'); // Ю
        assert_eq!(first_char(&[0xff]), char::from(0xff_u8));
    }
}
