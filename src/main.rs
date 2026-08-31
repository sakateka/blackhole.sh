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

fn clamp(x: f64, a: f64, b: f64) -> f64 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Blackbody-ish ramp: 0 = deep red, 1 = blue white.
fn heat(t: f64) -> [f64; 3] {
    let t = clamp(t, 0.0, 1.0);
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
    let rows_per = (TURB_RR + nthreads - 1) / nthreads;
    thread::scope(|sc| {
        for (n, band) in tex.chunks_mut(rows_per * TURB_PHI).enumerate() {
            sc.spawn(move || {
                for (r, row) in band.chunks_mut(TURB_PHI).enumerate() {
                    let rr =
                        R_IN + (n * rows_per + r) as f64 / TURB_RR as f64 * (R_OUT - R_IN);
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
    let b = tex[iv1 * TURB_PHI + iu]
        + (tex[iv1 * TURB_PHI + iu1] - tex[iv1 * TURB_PHI + iu]) * au;
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
    let beta2 = (0.5 * RS) / (rr - RS);
    let beta = beta2.min(0.85);
    let gamma = 1.0 / (1.0 - beta2).max(1e-3).sqrt();

    // prograde orbital direction
    let d = (hp.x * hp.x + hp.z * hp.z).sqrt().max(1e-9);
    let bvec = V3::new(-hp.z / d, 0.0, hp.x / d) * beta;

    // g = (gravitational redshift) / (Doppler)
    let g = (1.0 - rs_r).sqrt() / (gamma * (1.0 + bvec.dot(vd)));
    let g = clamp(g, 0.05, 4.0);

    let temp = clamp((R_IN / rr).powf(0.72) * (0.55 + 0.55 * g), 0.0, 1.0);
    // g^3 beaming, softened by an emissivity floor: a fully beaming disk leaves
    // the receding half of the frame empty, which reads as a bug, not as physics.
    let inten = rad * (0.35 + 0.65 * g * g * g) * BRIGHT;
    let c = heat(temp);
    let om = 1.6 * (R_IN / rr).powf(1.5); // pattern drift rate at this radius
    (
        [c[0] * inten, c[1] * inten, c[2] * inten],
        om,
    )
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
    let f = (ay * 512.0).min(511.999).max(0.0);
    let i = f as usize;
    let a = f - i as f64;
    t[i] * (1.0 - a) + t[i + 1] * a
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
        let sigma = clamp(STAR_CORE / ppc, 0.02, 0.30);
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
    // whisper of a galactic band so the void is not perfectly flat; it is a
    // function of d.y alone, so a yawing camera leaves it untouched
    let band = if d.y.abs() < 0.75 { band_w(d.y.abs()) } else { 0.0 };
    let g = if band > 1e-3 {
        band * (0.4 + 1.2 * fbm3(d.x * 3.0 + 11.0, d.y * 3.0, d.z * 3.0 - 7.0))
    } else {
        0.0
    };
    Sky {
        star,
        band: [g * 0.7, g * 0.8, g],
        freq,
        phase,
    }
}

/// Sky for an orbiting camera: only the star positions move. The band is
/// exactly invariant under yaw (it depends on d.y alone) and the twinkle
/// parameters belong to the pixel's neighbourhood, so both are reused.
fn stars_moved(d: V3, ppc0: f64, cached: &Sky) -> Sky {
    let (star, _, _) = star_layer(d, ppc0);
    Sky { star, band: cached.band, freq: cached.freq, phase: cached.phase }
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
}

impl Geo {
    /// A pixel that cannot change while the camera holds still: no disk
    /// crossings and no star to twinkle (the band is static). Its value from
    /// the previous frame is still correct, so shading can skip it entirely.
    fn is_static(&self) -> bool {
        self.n == 0 && !self.sky.map_or(false, |s| s.star != [0.0, 0.0, 0.0])
    }

    fn empty() -> Geo {
        Geo {
            sky: None,
            esc: V3::new(0.0, 0.0, 0.0),
            n: 0,
            cr: [Cross { x: 0.0, z: 0.0, om: 0.0, rr: 0.0, em: [0.0; 3] }; 3],
        }
    }
}

/// Geometry cache, invalidated whenever anything that bends the rays changes
/// (frame size, zoom, tilt, shift, camera orbit angle).
struct GeoCache {
    key: (usize, usize, u64, u64, u64),
    geo: Vec<Geo>,
    /// one byte per pixel: true = can change between frames. Scanning this
    /// instead of the 90 MB of Geo structs is what makes the skip cheap.
    mask: Vec<bool>,
}

impl GeoCache {
    fn new() -> GeoCache {
        // a key no camera setup can produce
        GeoCache { key: (0, 0, 1, 1, 1), geo: Vec::new(), mask: Vec::new() }
    }
}

/// Integrate one ray backwards from the camera and record what the shading
/// will need later. Time-independent by construction; `ppc0` is the ray-grid
/// pitch in pixels per star cell, needed to size the star cores.
fn trace_geo(cam: &Cam, dir: V3, ppc0: f64, om_max: f64, out: &mut Geo) {
    let mut p = cam.p;
    let mut v = dir;
    let h2 = p.cross(v).len2();
    let mut a = accel(p, h2);
    let mut tr = 1.0f64; // remaining transmittance
    let mut n = 0u8;
    let mut cr = [Cross { x: 0.0, z: 0.0, om: 0.0, rr: 0.0, em: [0.0; 3] }; 3];

    for _ in 0..MAX_STEPS {
        let r = p.len();
        if r <= RS {
            // swallowed: this pixel is the shadow
            *out = Geo { sky: None, esc: V3::new(0.0, 0.0, 0.0), n, cr };
            return;
        }
        if r > ESCAPE && p.dot(v) > 0.0 {
            *out = Geo { sky: Some(stars(v, ppc0)), esc: v, n, cr };
            return;
        }
        // adaptive step: fine near the hole, coarse far away. The cap may be
        // generous - out there the orbit is straight and only the sky lookup
        // is left to pay for.
        let dt = clamp(0.045 * r, 0.012, 1.1);
        let pn = p + v * dt + a * (0.5 * dt * dt);
        let an = accel(pn, h2);
        let vn = v + (a + an) * (0.5 * dt);

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
                    cr[n as usize] = Cross { x: hp.x, z: hp.z, om, rr, em };
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
    *out = Geo { sky: None, esc: V3::new(0.0, 0.0, 0.0), n, cr };
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
                // stay cached, and the band/twinkle are reused, see
                // stars_moved)
                let d = V3::new(
                    g.esc.x * ctx.c - g.esc.z * ctx.s,
                    g.esc.y,
                    g.esc.x * ctx.s + g.esc.z * ctx.c,
                );
                sky_rgb(&stars_moved(d, ctx.ppc0, cached), ctx.t)
            }
        }
        None => [0.0; 3],
    };
    [
        clamp(c[0] + bg[0], 0.0, 1.0),
        clamp(c[1] + bg[1], 0.0, 1.0),
        clamp(c[2] + bg[2], 0.0, 1.0),
    ]
}

fn tonemap(c: [f64; 3]) -> [f64; 3] {
    // exposure + filmic shoulder + display gamma, sampled (see build_tone)
    [tone(EXPOSURE * c[0]), tone(EXPOSURE * c[1]), tone(EXPOSURE * c[2])]
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

struct Opt {
    mode: Mode,
    fps: f64,
    zoom: f64,
    speed: f64,
    orbit: f64,
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
        tilt: CAM_TILT,
        shift: 0.0,
        color: true,
        cols: 0,
        rows: 0,
        tpw: 0,
        tph: 0,
        rays: RAY_BUDGET,
        one_shot: None,
        ramp: " .·:;+=*xX#%@█".chars().collect(),
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
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

KEYS        q/Esc quit    +/- zoom    up/down tilt    space pause
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
        env::var("COLUMNS").ok().and_then(|v| v.parse::<usize>().ok()),
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
fn render_frame(o: &Opt, t: f64, f: &mut Frame, cache: &mut GeoCache) {
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
    let orbit = o.orbit.to_radians() * t; // camera azimuth right now
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
    let rows_per = (h + nthreads - 1) / nthreads;

    // pixels per star cell at layer 0: w pixels span 2*atan(VIEW/zoom*aspect)
    // radians, a layer-0 cell is 1/STAR_SCALE[0] radians wide
    let aspect = w as f64 / h as f64;
    let fov = 2.0 * (VIEW / zoom * aspect).atan();
    let ppc0 = w as f64 / (fov * STAR_SCALE[0]);
    let om_max = TURB_MAX_PX * fov / o.tpw as f64;

    // I-frame: full geodesic pass, only when the camera setup changes. The
    // orbit is deliberately absent from the key: rotation of the view is
    // handled in the shading, not by moving the camera.
    let key = (w, h, zoom.to_bits(), o.tilt.to_bits(), shift.to_bits());
    let mut invalidated = false;
    if cache.key != key || cache.geo.len() != w * h {
        invalidated = true;
        cache.key = key;
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
        if par {
            thread::scope(|sc| {
                for (n, (band, mband)) in geo
                    .chunks_mut(rows_per * w)
                    .zip(mask.chunks_mut(rows_per * w))
                    .enumerate()
                {
                    let cam = &cam;
                    sc.spawn(move || {
                        let y0 = n * rows_per;
                        for (j, (rowgeo, rowmask)) in band
                            .chunks_mut(w)
                            .zip(mband.chunks_mut(w))
                            .enumerate()
                        {
                            let y = y0 + j;
                            for (x, (g, m)) in rowgeo
                                .iter_mut()
                                .zip(rowmask.iter_mut())
                                .enumerate()
                            {
                                let dir = cam.ray(x, y, w, h, zoom, shift);
                                trace_geo(&cam, dir, ppc0, om_max, g);
                                // remember which pixels can ever change - the
                                // P-frames scan this byte mask instead of
                                // streaming the whole geometry cache
                                *m = !g.is_static();
                            }
                        }
                    });
                }
            });
        } else {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let dir = cam.ray(x, y, w, h, zoom, shift);
                    trace_geo(&cam, dir, ppc0, om_max, &mut geo[i]);
                    mask[i] = !geo[i].is_static();
                }
            }
        }
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
    let ctx = ShCtx { t, orb: orbit, c: orbit.cos(), s: orbit.sin(), ppc0 };
    // pixels that cannot change (no disk, no star, still camera) keep their
    // previous value; on the first frame or after a re-trace everything is
    // shaded so the buffer is fully written. The mask carries the decision
    // so this pass streams 320 KB, not the whole geometry cache.
    let skip_static = !invalidated && !resized && orbit == 0.0;
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
        Screen { prev: Vec::new(), w: 0, h: 0 }
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
            cells.push(Cell { ch: ramp[idx], rgb: cell_rgb(o, &c) });
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
            cells.push(Cell { ch: ch4, rgb: cell_rgb(o, &c) });
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
    for x in 0..tw {
        col[x] = x * f.w / tw.max(1);
    }

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

    let bands = (th + 5) / 6;
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
        let mut f = Frame { w: 0, h: 0, px: Vec::new() };
        let mut cache = GeoCache::new();
        render_frame(&o, t, &mut f, &mut cache);
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
    let mut f = Frame { w: 0, h: 0, px: Vec::new() };
    let mut cache = GeoCache::new();
    let mut last = Instant::now();
    loop {
        let step = last.elapsed().as_secs_f64();
        last = Instant::now();
        if !paused {
            t += step * o.speed;
        }
        if !paused || !drawn {
            render_frame(&o, t, &mut f, &mut cache);
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
                // Left/Right are parsed but not bound to anything yet
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
