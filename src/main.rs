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
    fn new(x: f64, y: f64, z: f64) -> V3 {
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
//  - the tone curve (1 - e^-x)^(1/1.85) has infinite slope at 0, so it is
//    sampled on a sqrt-spaced grid;
//  - the star twinkle needs one sin per sky pixel - a 7th-order polynomial
//    is indistinguishable at the 0.18 amplitude it modulates.

const TURB_K: f64 = 2.1; // turbulence noise frequency (see disk_em / shade)
const TAU: f64 = std::f64::consts::TAU;
const TURB_PHI: usize = 1024;
const TURB_RR: usize = 512;
static TURB_TEX: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();

/// Fill the turbulence texture once. About half a million noise lookups -
/// a tenth of a second single-threaded, a few milliseconds across cores,
/// and afterwards every frame samples it instead of recomputing fbm3.
fn build_turb_tex() {
    let mut tex = vec![0.0f64; TURB_RR * TURB_PHI];
    let nthreads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 32);
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

/// tone curve, sampled sqrt-spaced in x (dense where the slope blows up)
const TONE_N: usize = 4096;
const TONE_MAX: f64 = 16.0;
static TONE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();

fn build_tone() {
    let t: Vec<f64> = (0..=TONE_N)
        .map(|i| {
            let x = TONE_MAX * (i as f64 / TONE_N as f64) * (i as f64 / TONE_N as f64);
            (1.0 - (-x).exp()).powf(1.0 / 1.85)
        })
        .collect();
    let _ = TONE.set(t);
}

#[inline]
fn tone(x: f64) -> f64 {
    let t = TONE.get().expect("tone table");
    let s = (x.max(0.0) / TONE_MAX).sqrt() * TONE_N as f64;
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
/// how fast the massive star is torn apart inside its tidal radius
const INFALL_STRIP: f64 = 2.5;
/// how long a shed stream particle keeps glowing
const STREAM_LIFE: f64 = 6.0;
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
}

/// Every glow lives at radius <= BIG_R0.1 and reaches at most 3.5 sigmas
/// times the largest star scale (3x for the massive star) plus the shell's
/// radial thickness past it, so a segment whose midpoint is inside this
/// radius is worth recording for the deposition to chew on.
const GLOW_R: f64 = BIG_R0.1 + 3.5 * INFALL_SIG * 3.0 + 0.06 * BIG_R0.1;

/// One recorded trace segment inside the glow shell: the endpoints (start
/// position and the step vector), the midpoint's radius for the cheap radial
/// pre-reject, the integration step and the transmittance that was in effect
/// when it was traced, and the pixel it belongs to. Storing these is what
/// lets the glows move every frame without re-integrating the geodesics:
/// the deposition just replays the gaussian sums over the bins it touches.
#[derive(Clone, Copy)]
struct Seg {
    p0: [f32; 3],
    sg: [f32; 3],
    r: f32,
    dt: f32,
    tr: f32,
    px: u32,
}

/// Uniform bins over the cube of side BIN_N * BIN_W centred on the hole -
/// the whole region any glow can light up, out past where the massive
/// star spawns - so a glow only ever meets the segments physically near
/// it instead of all of them.
const BIN_N: usize = 75;
const BIN_W: f64 = 1.2;
/// the cube's centre: the bin index of the coordinate origin
const BIN_C: isize = (BIN_N as isize) / 2;

/// The bin a segment's midpoint falls in, clamped to the cube's edge.
fn bin_of(s: &Seg) -> usize {
    let c = |a: f32, v: f32| {
        let m = a as f64 + 0.5 * v as f64;
        (((m / BIN_W).floor() as isize) + BIN_C).clamp(0, BIN_N as isize - 1)
    };
    let x = c(s.p0[0], s.sg[0]);
    let y = c(s.p0[1], s.sg[1]);
    let z = c(s.p0[2], s.sg[2]);
    (x + BIN_N as isize * (y + BIN_N as isize * z)) as usize
}

/// Sort the segments into their bins: count per bin, turn the counts into
/// exclusive prefix offsets, then permute in place - each swap drops one
/// segment into its bin's next free slot, so no second arena is needed.
fn build_bins(segs: &mut [Seg], bin_off: &mut Vec<u32>) {
    bin_off.clear();
    bin_off.resize(BIN_N * BIN_N * BIN_N + 1, 0);
    for s in segs.iter() {
        bin_off[bin_of(s) + 1] += 1;
    }
    for i in 1..bin_off.len() {
        bin_off[i] += bin_off[i - 1];
    }
    let mut cur = bin_off[..bin_off.len() - 1].to_vec();
    let mut i = 0;
    while i < segs.len() {
        let b = bin_of(&segs[i]);
        if (i as u32) >= bin_off[b] && (i as u32) < bin_off[b + 1] {
            i += 1;
        } else {
            let t = cur[b] as usize;
            segs.swap(i, t);
            cur[b] += 1;
        }
    }
}

/// Add one glow's light to `out`, indexed by pixel. This is the same sum the
/// trace used to carry inline - radial pre-reject, closest approach of the
/// segment, gaussian weight times step times transmittance - but only over
/// the bins within the glow's reach, a few thousand segments instead of
/// every segment tested against every glow.
fn deposit_one(gl: &Glow, segs: &[Seg], bin_off: &[u32], out: &mut dyn FnMut(usize, [f64; 3])) {
    let gr = gl.p.len();
    let rej = gl.sig * 3.5 + 0.06 * gr;
    let range = (rej / BIN_W).ceil() as isize + 1;
    let cell = |v: f64| (((v / BIN_W).floor() as isize) + BIN_C).clamp(0, BIN_N as isize - 1);
    let (gx, gy, gz) = (cell(gl.p.x), cell(gl.p.y), cell(gl.p.z));
    let s2 = 2.0 * gl.sig * gl.sig;
    let last = BIN_N as isize - 1;
    for z in (gz - range).max(0)..=(gz + range).min(last) {
        for y in (gy - range).max(0)..=(gy + range).min(last) {
            for x in (gx - range).max(0)..=(gx + range).min(last) {
                let b = (x + BIN_N as isize * (y + BIN_N as isize * z)) as usize;
                for s in &segs[bin_off[b] as usize..bin_off[b + 1] as usize] {
                    // cheap radial pre-reject: the glow lives in a shell
                    // around the hole at its own radius
                    if (s.r as f64 - gr).abs() > rej + 1e-4 {
                        continue;
                    }
                    // closest approach of the segment to the glow
                    let (px, py, pz) = (s.p0[0] as f64, s.p0[1] as f64, s.p0[2] as f64);
                    let (sx, sy, sz) = (s.sg[0] as f64, s.sg[1] as f64, s.sg[2] as f64);
                    let du = (gl.p.x - px) * sx + (gl.p.y - py) * sy + (gl.p.z - pz) * sz;
                    let u = (du / (sx * sx + sy * sy + sz * sz)).clamp(0.0, 1.0);
                    let dx = gl.p.x - (px + sx * u);
                    let dy = gl.p.y - (py + sy * u);
                    let dz = gl.p.z - (pz + sz * u);
                    let e = (-(dx * dx + dy * dy + dz * dz) / s2).exp() * s.dt as f64 * s.tr as f64;
                    out(s.px as usize, [gl.c[0] * e, gl.c[1] * e, gl.c[2] * e]);
                }
            }
        }
    }
}

/// Per-worker glow accumulation. `px` is dense so updates need no hashing;
/// `touched` makes clearing and reduction proportional to the number of lit
/// pixels rather than the size of the entire ray grid.
struct GlowBuf {
    px: Vec<[f64; 3]>,
    touched: Vec<u32>,
}

impl GlowBuf {
    fn new() -> GlowBuf {
        GlowBuf {
            px: Vec::new(),
            touched: Vec::new(),
        }
    }

    fn resize(&mut self, len: usize) {
        if self.px.len() != len {
            self.px = vec![[0.0; 3]; len];
            self.touched.clear();
        }
    }
}

struct GlowCache {
    bufs: Vec<GlowBuf>,
    lit: Vec<u32>,
    grid_len: usize,
}

impl GlowCache {
    fn new() -> GlowCache {
        GlowCache {
            bufs: Vec::new(),
            lit: Vec::new(),
            grid_len: 0,
        }
    }
}

/// Upper bound for all dense worker accumulators together. At very large ray
/// grids deposition uses fewer workers instead of multiplying memory use by
/// the machine's CPU count.
const GLOW_BUF_BUDGET: usize = 32 * 1024 * 1024;

/// Lay the frame's glows over the cached geometry. Parallel workers retain
/// their dense accumulation buffers between frames, but reduce and clear only
/// pixels they actually touched. `lit` similarly identifies the Geo entries
/// that need clearing at the beginning of the next glow frame.
fn deposit_glows(
    segs: &[Seg],
    bin_off: &[u32],
    glows: &[Glow],
    geo: &mut [Geo],
    cache: &mut GlowCache,
    par: bool,
    nthreads: usize,
) {
    // This also runs on the first empty frame. Otherwise the final deposited
    // trail remains in Geo forever after its glows disappear.
    for px in cache.lit.drain(..) {
        geo[px as usize].st = [0.0; 3];
    }
    if glows.is_empty() || segs.is_empty() {
        return;
    }
    if cache.grid_len != geo.len() {
        // A size change is an I-frame event, so release all old worker
        // capacities once instead of letting several grid sizes accumulate.
        cache.bufs.clear();
        cache.grid_len = geo.len();
    }
    let bytes_per_buf = geo
        .len()
        .saturating_mul(std::mem::size_of::<[f64; 3]>())
        .max(1);
    let memory_workers = (GLOW_BUF_BUDGET / bytes_per_buf).max(1);
    let nt = if par {
        nthreads.min(glows.len()).min(memory_workers)
    } else {
        1
    };
    if nt <= 1 {
        for gl in glows {
            deposit_one(gl, segs, bin_off, &mut |px, add| {
                let g = &mut geo[px];
                if g.st == [0.0; 3] {
                    cache.lit.push(px as u32);
                }
                g.st[0] += add[0];
                g.st[1] += add[1];
                g.st[2] += add[2];
            });
        }
    } else {
        while cache.bufs.len() < nt {
            cache.bufs.push(GlowBuf::new());
        }
        for buf in cache.bufs.iter_mut().take(nt) {
            buf.resize(geo.len());
        }
        std::thread::scope(|sc| {
            for (chunk, buf) in glows
                .chunks(glows.len().div_ceil(nt))
                .zip(cache.bufs.iter_mut().take(nt))
            {
                sc.spawn(move || {
                    for gl in chunk {
                        deposit_one(gl, segs, bin_off, &mut |px, add| {
                            let dst = &mut buf.px[px];
                            if *dst == [0.0; 3] {
                                buf.touched.push(px as u32);
                            }
                            dst[0] += add[0];
                            dst[1] += add[1];
                            dst[2] += add[2];
                        });
                    }
                });
            }
        });
        // Worker order matches glow-chunk order, exactly as in the previous
        // full-buffer reduction. This preserves floating-point association.
        for buf in cache.bufs.iter_mut().take(nt) {
            for px in buf.touched.drain(..) {
                let px = px as usize;
                let add = buf.px[px];
                buf.px[px] = [0.0; 3];
                let g = &mut geo[px];
                if g.st == [0.0; 3] {
                    cache.lit.push(px as u32);
                }
                g.st[0] += add[0];
                g.st[1] += add[1];
                g.st[2] += add[2];
            }
        }
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

            if self.alive {
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
                if self.sc > 1.5 {
                    // Keep unmaterialized shed mass in `debt` while the visual
                    // particle pool is full; never create or discard mass just
                    // because STREAM_MAX was reached.
                    while self.debt > 0.02 && streams.len() < STREAM_MAX {
                        self.debt -= 0.02;
                        self.ns += 1;
                        streams.push(Stream {
                            p: self.p,
                            v: self.v * (0.95 + 0.10 * (self.ns % 4) as f64),
                            w: 0.02,
                            age: 0.0,
                        });
                    }
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
            dt -= h;
        }
        self.alive || !self.tr.is_empty()
    }
}

/// One particle of the mass a massive star shed: it feels the same static
/// pseudo-Newtonian pull with the full drag, and glows until it crosses the
/// horizon or cools off.
struct Stream {
    p: V3,
    v: V3,
    /// how much star mass it carries, i.e. how brightly it glows
    w: f64,
    age: f64,
}

impl Stream {
    /// Returns None while the stream lives; Some(w) once it is gone,
    fn advance(&mut self, mut dt: f64, gm: f64) -> bool {
        let remaining = STREAM_LIFE - self.age;
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
            self.v = self.v * (1.0 - INFALL_DRAG * h);
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

    fn spawn(&mut self, big: bool, o: &Opt) {
        if self.live.iter().filter(|inf| inf.alive).count() < INFALL_MAX {
            self.seed += 1;
            // freeze the screen basis at spawn time so the star dives in
            // from the side the user asked for; with no --origin every
            // star picks a side at random, except the first one, which
            // enters from the left
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
            let (d, a, b) = origin.basis(&cam);
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

            // Advance streams and existing remnants before stars. Material or
            // remnants created by a star during this slice therefore start at
            // the slice boundary instead of being aged by time before birth.
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
                if was_alive && !self.live[i].alive {
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
        // the glow tracks the star's size and whatever mass it has left
        let scl = inf.sc * inf.m.cbrt();
        if inf.alive {
            let head = heat(0.85);
            let w = INFALL_HEAD_BRI * tide(inf.p.len()) * scl;
            g.push(Glow {
                p: rot(inf.p),
                c: [head[0] * w, head[1] * w, head[2] * w],
                sig: INFALL_SIG * scl,
            });
        }
        // the trail: older parcels are dimmer and cooler
        let n = inf.tr.len().max(1);
        for (i, q) in inf.tr.iter().enumerate() {
            let u = i as f64 / n as f64; // 0 = oldest, 1 = newest
            let col = heat(0.25 + 0.5 * u);
            let w = mix(0.25, 1.0, u) * INFALL_TRAIL_BRI * tide(q.p.len()) * scl;
            g.push(Glow {
                p: rot(q.p),
                c: [col[0] * w, col[1] * w, col[2] * w],
                sig: INFALL_TRAIL_SIG * scl,
            });
        }
    }
    // the mass the massive star sheds: a cooler string of glows, each
    // fading with age
    for stm in &st.streams {
        let b = 1.0 - stm.age / STREAM_LIFE;
        let w = STREAM_BRI * stm.w * b * tide(stm.p.len());
        g.push(Glow {
            p: rot(stm.p),
            c: [w, 0.75 * w, 0.45 * w],
            sig: STREAM_SIG,
        });
    }
    for rem in &st.rem {
        let col = heat(0.95);
        let w = INFALL_REM_BRI * rem.b * rem.sc;
        g.push(Glow {
            p: rot(rem.p),
            c: [col[0] * w, col[1] * w, col[2] * w],
            sig: INFALL_REM_SIG * rem.sc,
        });
    }
    g
}

/// One crossing of the equatorial plane inside the disk, reduced to what a
/// frame actually needs: the position (for the noise phase), the radius, the
/// pattern drift rate and the fully pre-weighted static emission.
#[derive(Clone, Copy)]
struct Cross {
    x: f64,
    z: f64,
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
    /// emission deposited by the infalling star's glow along this ray
    st: [f64; 3],
}

impl Geo {
    /// A pixel that cannot change while the camera holds still: no disk
    /// crossings and no star to twinkle (the band is static). Its value from
    /// the previous frame is still correct, so shading can skip it entirely.
    fn is_static(&self) -> bool {
        self.n == 0
            && self.st == [0.0, 0.0, 0.0]
            && !self.sky.is_some_and(|s| s.star != [0.0, 0.0, 0.0])
    }

    fn empty() -> Geo {
        Geo {
            sky: None,
            esc: V3::new(0.0, 0.0, 0.0),
            n: 0,
            cr: [Cross {
                x: 0.0,
                z: 0.0,
                om: 0.0,
                rr: 0.0,
                em: [0.0; 3],
            }; 3],
            st: [0.0; 3],
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
            glow: GlowCache::new(),
            glow_was: false,
        }
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
    let st = [0.0f64; 3]; // glow is deposited after the trace, not here
    let mut n = 0u8;
    let mut cr = [Cross {
        x: 0.0,
        z: 0.0,
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
                st,
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
                st,
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

        // the infalling star's glows are deposited after the trace, from a
        // compact record of every segment that threads the glow shell:
        // endpoints, step and the transmittance in effect here
        let seg = pn - p;
        let rm = (p + seg * 0.5).len();
        if rm < GLOW_R {
            segs.push(Seg {
                p0: [p.x as f32, p.y as f32, p.z as f32],
                sg: [seg.x as f32, seg.y as f32, seg.z as f32],
                r: rm as f32,
                dt: dt as f32,
                tr: tr as f32,
                px: px as u32,
            });
        }

        // did we cross the equatorial plane?
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
                        x: hp.x,
                        z: hp.z,
                        om,
                        rr,
                        em,
                    };
                    n += 1;
                }
                tr *= DISK_OPA;
            }
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
        st,
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
fn shade(g: &Geo, ctx: &ShCtx) -> [f64; 3] {
    let mut col = [0.0f64; 3]; // HDR disk light, tone mapped at the end
    for i in 0..g.n as usize {
        let c = &g.cr[i];
        // Schwarzschild is axially symmetric: the geometry seen from azimuth
        // `orb` is the azimuth-0 geometry rotated about the disk axis, so the
        // radius, Doppler and emission are unchanged and the turbulence phase
        // simply gains `orb`. The pattern itself lives in a pre-sampled
        // texture (see build_turb_tex).
        let phi = c.z.atan2(c.x) + ctx.orb - ctx.t * c.om;
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
    // the infalling star's glow: soft-clipped so it can blaze near the hole
    // without flattening anything else, with a slow shimmer on top
    let mut gl = [0.0f64; 3];
    if g.st != [0.0, 0.0, 0.0] {
        let fl = 0.86 + 0.14 * tw_sin(8.0 * ctx.t);
        gl = [
            softclip(g.st[0] * fl),
            softclip(g.st[1] * fl),
            softclip(g.st[2] * fl),
        ];
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

struct Opt {
    mode: Mode,
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
    one_shot: Option<f64>,
    /// start with a star spiralling into the hole
    star: bool,
    /// start with a massive star, 3x the size of the hole
    big_star: bool,
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
        one_shot: None,
        star: false,
        big_star: false,
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
      --frame <n>       render a single frame at time n/fps and exit
      --no-color        no ANSI colours (pure ASCII output, good for pipes)
      --star            add a star that gets swallowed by the hole
      --big-star        start with a massive star, 3x the size of the hole
      --origin <side>   side the star dives in from: left|right|top|bottom|front|back
                        (default: random per star; the first star dives in from the left)
      --star-speed <n>  initial star speed as a fraction of the local circular
                        speed: 1 = circular orbit, 0 = dropped from rest,
                        negative = the orbit run backwards (default: random)

KEYS        q/Esc quit    +/- zoom    up/down tilt    left/right orbit rate    space pause
            s spawn star    S spawn big star    x clear stars
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
        return Some(Key::Char(b[0] as char));
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
fn render_frame(o: &Opt, t: f64, f: &mut Frame, cache: &mut GeoCache, glows: &[Glow]) {
    let s = (o.rays as f64 / (o.tpw as f64 * o.tph as f64))
        .min(1.0)
        .sqrt();
    let w = ((o.tpw as f64 * s).round() as usize).max(80);
    let h = ((o.tph as f64 * s).round() as usize).max(40);
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

    let nthreads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 32);
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
        let mut segs: Vec<Seg> = Vec::new();
        if par {
            // each band collects the glow-lit path segments into its own
            // arena; concatenated in band order they arrive sorted by pixel,
            // which keeps the deposition pass cache-friendly
            let nband = h.div_ceil(rows_per);
            let mut local: Vec<Vec<Seg>> = (0..nband).map(|_| Vec::new()).collect();
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
            for v in local {
                segs.extend_from_slice(&v);
            }
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
        build_bins(&mut segs, &mut cache.bin_off);
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
            &mut cache.geo,
            &mut cache.glow,
            par,
            nthreads,
        );
    }

    // P-frame: re-shade the cached geometry for the current time and azimuth.
    // The one-off sample tables are wanted by the shading (and by nothing
    // else), so they are built here, once.
    if TURB_TEX.get().is_none() {
        build_turb_tex();
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
    let skip_static = !invalidated && !resized && orbit == 0.0 && !(glow_now || cache.glow_was);
    // glow-lit pixels are not in the trace-time mask, so while any glow is
    // live (and for one frame after the last one dies) everything is
    // re-shaded; once the residual is cleared the cheap masked path is
    // valid again
    cache.glow_was = glow_now;
    let geo = &cache.geo;
    if par {
        thread::scope(|sc| {
            for (n, band) in px.chunks_mut(rows_per * w).enumerate() {
                let src = &geo[n * rows_per * w..];
                let mband = &cache.mask[n * rows_per * w..];
                let ctx = &ctx;
                sc.spawn(move || {
                    if skip_static {
                        for ((p, m), g) in band.iter_mut().zip(mband).zip(src) {
                            if !*m {
                                continue;
                            }
                            *p = shade(g, ctx);
                        }
                    } else {
                        for (p, g) in band.iter_mut().zip(src) {
                            *p = shade(g, ctx);
                        }
                    }
                });
            }
        });
    } else if skip_static {
        for ((p, m), g) in px.iter_mut().zip(cache.mask.iter()).zip(geo.iter()) {
            if !*m {
                continue;
            }
            *p = shade(g, &ctx);
        }
    } else {
        for (p, g) in px.iter_mut().zip(geo.iter()) {
            *p = shade(g, &ctx);
        }
    }
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
    let ch = f.h / SUB_Y;
    let mut cells = Vec::with_capacity(cw * ch);
    for cy in 0..ch {
        for cx in 0..cw {
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
            cells.push(Cell {
                ch: ramp[idx],
                rgb: cell_rgb(o, &c),
            });
        }
    }
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
    let ch = f.h / SUB_Y;
    let mut cells = Vec::with_capacity(cw * ch);
    for cy in 0..ch {
        for cx in 0..cw {
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
            cells.push(Cell {
                ch: ch4,
                rgb: cell_rgb(o, &c),
            });
        }
    }
    scr.emit(o, &cells, cw, ch, out);
}

/// Sixel sky cutoff: pixels dimmer than this are dropped outright. The faint
/// specks it removes are near-invisible but each one costs a colour strip and
/// broken run-length compression in every band it touches.
const SIXEL_SKY_CUT: f64 = 0.17;

fn draw_sixel(o: &Opt, f: &Frame, out: &mut String) {
    // Sixel paints device pixels, so map the ray grid up to the target size
    // (nearest neighbour) instead of drawing the picture a fifth of the
    // window wide.
    let tw = o.tpw;
    let th = o.tph;
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
        let mut stars = Stars::new();
        if o.big_star || o.star {
            stars.spawn(o.big_star, &o);
            stars.advance(t);
        }
        o.azi = final_azi;
        let glows = glow_list(&stars, o.azi);
        render_frame(&o, t, &mut f, &mut cache, &glows);
        let mut scr = Screen::new();
        draw_into(&o, &f, &mut out, &mut scr);
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
    let mut stars = Stars::new();
    if o.big_star || o.star {
        stars.spawn(o.big_star, &o);
    }
    let mut last = Instant::now();
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
            draw_into(&o, &f, &mut out, &mut scr);
            let _ = so.write_all(out.as_bytes());
            let _ = so.flush();
            drawn = true;
        }

        // frame pacing (also lets us stay responsive on slow terminals)
        let budget = Duration::from_secs_f64(1.0 / o.fps);
        while last.elapsed() < budget {
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(k) = poll_key() {
            match k {
                Key::Esc | Key::Char('q') | Key::Char('c') => break,
                Key::Char(' ') => paused = !paused,
                Key::Char('+') | Key::Char('=') => {
                    o.zoom = (o.zoom * 1.15).clamp(0.25, 6.0);
                    drawn = false;
                }
                Key::Char('-') | Key::Char('_') => {
                    o.zoom = (o.zoom / 1.15).clamp(0.25, 6.0);
                    drawn = false;
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

fn draw_into(o: &Opt, f: &Frame, out: &mut String, scr: &mut Screen) {
    match o.mode {
        Mode::Ascii => draw_ascii(o, f, out, scr),
        Mode::Braille => draw_braille(o, f, out, scr),
        Mode::Sixel => draw_sixel(o, f, out),
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
        });
        stars
    }

    #[test]
    fn empty_glow_frame_clears_previous_deposition() {
        let mut geo = vec![Geo::empty()];
        geo[0].st = [1.0, 0.5, 0.25];
        let mut cache = GlowCache::new();
        cache.lit.push(0);

        deposit_glows(&[], &[], &[], &mut geo, &mut cache, false, 1);

        assert_eq!(geo[0].st, [0.0; 3]);
        assert!(cache.lit.is_empty());
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

        assert_eq!(shade(&geo, &ctx), expected);
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
}
