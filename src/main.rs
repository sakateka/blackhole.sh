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

use std::collections::BTreeMap;
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
const CAM_TILT: f64 = 0.100; // radians above the disk plane
const VIEW: f64 = 0.30; // half height of the frustum (tangent of half fov)
const DISK_OPA: f64 = 0.05; // transmittance of one disk crossing (opaque disk)
const ESCAPE: f64 = 32.0; // where a ray is considered free again
const MAX_STEPS: usize = 900;
const EXPOSURE: f64 = 1.0; // tone map exposure
const BRIGHT: f64 = 3.0; // disk emission scale
const STAR_DENSITY: f64 = 0.74; // probability a cell is empty; lower = more stars
const STAR_SHARP: f64 = 5.0; // higher = tighter points
const STAR_SCALE: [f64; 3] = [44.0, 74.0, 120.0];
const STAR_BRI: [f64; 3] = [1.0, 0.55, 0.30];

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

// ------------------------------------------------------------ the disk
//
/// Emission of the thin disk at radius `rr`, hit point `hp`, photon direction
/// `vd` (integrated backwards, i.e. away from the camera).
fn disk(rr: f64, hp: V3, vd: V3, t: f64) -> [f64; 3] {
    if rr <= R_IN || rr >= R_OUT {
        return [0.0; 3];
    }
    let u = (rr - R_IN) / (R_OUT - R_IN); // 0 at the inner rim, 1 at the outer edge

    // turbulence, sheared by the Keplerian rotation
    let mut phi = hp.z.atan2(hp.x);
    phi -= t * 1.6 * (R_IN / rr).powf(1.5);
    let k = 2.1;
    let n = fbm3(
        phi.cos() * k,
        phi.sin() * k,
        rr * 0.55 + (rr * 0.21).sin() * 0.8,
    );
    let streak = 0.42 + 1.25 * n;

    let edge_in = smoothstep(0.0, 0.05, u); // hot, sharp inner rim
    let edge_out = 1.0 - smoothstep(0.72, 1.0, u); // cool, ragged outer edge
    let prof = (R_IN / rr).powf(1.55) * edge_in * edge_out * streak;

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
    let inten = prof * (0.35 + 0.65 * g * g * g) * BRIGHT;
    let c = heat(temp);
    [c[0] * inten, c[1] * inten, c[2] * inten]
}

// ------------------------------------------------------------- deep space

/// Deep space. Returned already display-referred (0..1) on purpose: the tone
/// curve below must not lift the void, or the hole drowns in grey noise.
fn stars(d: V3, t: f64) -> [f64; 3] {
    let mut lum = 0.0;
    let mut tint = 0.0;
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
        let f = (1.0 - r2 * STAR_SHARP).max(0.0);
        if f <= 0.0 {
            continue;
        }
        let f2 = f * f * f;
        let tw = 0.82 + 0.18 * (t * (1.5 + h0 * 2.0) + h0 * 40.0).sin();
        lum += f2 * (0.4 + 0.6 * h0) * tw * STAR_BRI[k as usize];
        tint += f2 * hash3i(ci[0] + 3, ci[1] + 11, ci[2] + 5);
    }
    // whisper of a galactic band so the void is not perfectly flat; the noise
    // is only worth sampling inside the band, the rest of the sky is flat 0
    let band = (-((d.y * 2.6).abs().powf(1.6)) * 1.4).exp() * 0.010;
    let g = if band > 1e-3 {
        band * (0.4 + 1.2 * fbm3(d.x * 3.0 + 11.0, d.y * 3.0, d.z * 3.0 - 7.0))
    } else {
        0.0
    };
    [
        lum * (0.85 + 0.4 * tint) + g * 0.7,
        lum * (0.88 + 0.3 * tint) + g * 0.8,
        lum + g,
    ]
}

// ------------------------------------------------------------- ray tracing

/// Integrate one ray backwards from the camera. Returns linear RGB.
fn trace(cam: &Cam, dir: V3, t: f64) -> [f64; 3] {
    let mut p = cam.p;
    let mut v = dir;
    let h2 = p.cross(v).len2();
    let mut col = [0.0; 3]; // HDR disk light, tone mapped at the end
    let mut bg = [0.0; 3]; // background sky, already display referred
    // The disk is optically thick: the near side silhouettes everything behind
    // it, which is what carves the shadow out of the glow. Each crossing lets
    // only a few per cent of the light from behind through.
    let mut tr = 1.0; // remaining transmittance
    let mut a = accel(p, h2);
    let mut steps = 0;

    loop {
        steps += 1;
        let r = p.len();
        if r <= RS {
            // swallowed: this pixel is the shadow
            break;
        }
        if r > ESCAPE && p.dot(v) > 0.0 {
            bg = stars(v, t);
            break;
        }
        if steps > MAX_STEPS {
            break;
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
            if rr > R_IN && rr < R_OUT {
                let vd = (v + (vn - v) * k).norm(); // direction of travel (away from us)
                let e = disk(rr, hp, vd, t);
                // Grazing crossings pile an enormous projected area into a single
                // row of pixels (the plane's horizon line) - fade those out, the
                // way a real thin disk's photosphere limb-darkens edge-on.
                let graze = smoothstep(0.0, 0.045, v.y.abs());
                for i in 0..3 {
                    col[i] += tr * e[i] * graze;
                }
                tr *= DISK_OPA;
            }
        }
        p = pn;
        v = vn;
        a = an;
    }
    let c = tonemap(col);
    [
        clamp(c[0] + bg[0], 0.0, 1.0),
        clamp(c[1] + bg[1], 0.0, 1.0),
        clamp(c[2] + bg[2], 0.0, 1.0),
    ]
}

fn tonemap(c: [f64; 3]) -> [f64; 3] {
    // exposure + filmic shoulder + display gamma
    let f = |x: f64| {
        let x = 1.0 - (-x.max(0.0) * EXPOSURE).exp();
        x.powf(1.0 / 1.85)
    };
    [f(c[0]), f(c[1]), f(c[2])]
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
      --frame <n>       render a single frame at time n/fps and exit
      --no-color        no ANSI colours (pure ASCII output, good for pipes)

KEYS        q/Esc quit    +/- zoom    space pause
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

fn poll_key() -> Option<char> {
    let mut b = [0u8; 8];
    match std::io::stdin().read(&mut b) {
        Ok(0) | Err(_) => None,
        Ok(_n) => {
            // skip escape sequences (arrow keys & friends)
            if b[0] == 0x1b {
                return Some('\x1b');
            }
            Some(b[0] as char)
        }
    }
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

fn render_frame(o: &Opt, t: f64) -> Frame {
    let s = (RAY_BUDGET as f64 / (o.tpw as f64 * o.tph as f64))
        .min(1.0)
        .sqrt();
    let w = ((o.tpw as f64 * s).round() as usize).max(80);
    let h = ((o.tph as f64 * s).round() as usize).max(40);
    let mut px = vec![[0.0f64; 3]; w * h];
    let orbit = o.orbit.to_radians() * t; // radians per second
    let cam = Cam::new(orbit, o.tilt.to_radians());
    let zoom = o.zoom;
    let shift = o.shift;

    let nthreads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let rows_per = (h + nthreads - 1) / nthreads;
    thread::scope(|sc| {
        for (n, band) in px.chunks_mut(rows_per * w).enumerate() {
            sc.spawn(move || {
                let y0 = n * rows_per;
                for (j, row) in band.chunks_mut(w).enumerate() {
                    let y = y0 + j;
                    for (x, pxl) in row.iter_mut().enumerate() {
                        let dir = cam.ray(x, y, w, h, zoom, shift);
                        *pxl = trace(&cam, dir, t);
                    }
                }
            });
        }
    });
    Frame { w, h, px }
}

fn colour(o: &Opt, out: &mut String, c: &[f64]) {
    if !o.color {
        return;
    }
    let r = (c[0] * 255.0) as u32;
    let g = (c[1] * 255.0) as u32;
    let b = (c[2] * 255.0) as u32;
    out.push_str("\x1b[38;2;");
    push_u32(out, r);
    out.push(';');
    push_u32(out, g);
    out.push(';');
    push_u32(out, b);
    out.push('m');
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

fn draw_ascii(o: &Opt, f: &Frame, out: &mut String) {
    let ramp = &o.ramp;
    let cw = f.w / SUB_X;
    let ch = f.h / SUB_Y;
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
            colour(o, out, &c);
            out.push(ramp[idx]);
            if o.color {
                out.push_str("\x1b[0m");
            }
        }
        if cy + 1 < ch {
            out.push_str("\r\n");
        }
    }
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

fn draw_braille(o: &Opt, f: &Frame, out: &mut String) {
    let cw = f.w / SUB_X;
    let ch = f.h / SUB_Y;
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
            colour(o, out, &c);
            out.push(char::from_u32(0x2800 + bits as u32).unwrap_or(' '));
            if o.color {
                out.push_str("\x1b[0m");
            }
        }
        if cy + 1 < ch {
            out.push_str("\r\n");
        }
    }
}

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

    // register 0 = black, registers 16..231 = a 6x6x6 colour cube.
    //
    // Registers use the RGB form `#idx;2;r;g;b` with components in per cent
    // (0..=100).  The colourspace selector `2;` is mandatory: a bare
    // `#idx;r;g;b` makes a strict parser read r as the colourspace and throw the
    // whole image away, and components above 100 are rejected just as loudly.
    let mut used = [false; 216];
    for p in &f.px {
        used[pixel_index(p)] = true;
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
    let mut strip: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for band in 0..bands {
        strip.clear();
        for sy in 0..6usize {
            let ty = band * 6 + sy;
            if ty >= th {
                break;
            }
            let y = ty * f.h / th.max(1);
            for x in 0..tw {
                let p = f.px[y * f.w + col[x]];
                if lum(&p) < 0.02 {
                    continue;
                }
                let idx = pixel_index(&p);
                let e = strip.entry(idx).or_insert_with(|| vec![0; tw]);
                e[x] |= 1 << sy;
            }
        }
        // opaque black background first so nothing of the previous frame leaks
        row.fill(0x3f);
        out.push_str("#0");
        write_row(out, &row);
        out.push('$');
        for (idx, mask) in strip.iter() {
            out.push('#');
            push_u32(out, (16 + idx) as u32);
            write_row(out, mask);
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
        let f = render_frame(&o, t);
        draw_into(&o, &f, &mut out);
        println!("{out}");
        return;
    }

    let _raw = RawTerm::new();
    let mut so = std::io::stdout();
    print!("\x1b[?1049h\x1b[?25l\x1b[2J"); // alt screen, hide cursor, clear
    let _ = so.flush();

    let mut t = 0.0;
    let mut paused = false;
    let mut last = Instant::now();
    loop {
        let step = last.elapsed().as_secs_f64();
        last = Instant::now();
        if !paused {
            t += step * o.speed;
        }
        let f = render_frame(&o, t);
        out.clear();
        out.push_str("\x1b[H");
        draw_into(&o, &f, &mut out);
        let _ = so.write_all(out.as_bytes());
        let _ = so.flush();

        // frame pacing (also lets us stay responsive on slow terminals)
        let budget = Duration::from_secs_f64(1.0 / o.fps);
        while last.elapsed() < budget {
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(k) = poll_key() {
            match k {
                'q' | '\x1b' | 'c' => break,
                ' ' => paused = !paused,
                '+' | '=' => o.zoom = (o.zoom * 1.15).clamp(0.25, 6.0),
                '-' | '_' => o.zoom = (o.zoom / 1.15).clamp(0.25, 6.0),
                _ => {}
            }
        }
    }
    print!("\x1b[0m\x1b[0m\x1b[?25h\x1b[?1049l");
    let _ = so.flush();
}

fn draw_into(o: &Opt, f: &Frame, out: &mut String) {
    match o.mode {
        Mode::Ascii => draw_ascii(o, f, out),
        Mode::Braille => draw_braille(o, f, out),
        Mode::Sixel => draw_sixel(o, f, out),
    }
}
