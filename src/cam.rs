//! Renderer-neutral camera data and a tiny quaternion fly-camera controller.
//!
//! Input backends keep ownership of key and pointer events. Feed their current
//! WASD state to [`FlyCam::step`] and pointer deltas to [`FlyCam::look`].
//!
//! Blueprint applications can use [`FlyCam::step_blueprint`] when the camera
//! is driven by the TRUEOS VLayer HID broker. That adapter is Blueprint-feature
//! gated; the reusable camera contract remains allocation-free.

/// Projection data shared by authored glTF cameras and runtime cameras.
#[cfg_attr(feature = "host", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective {
        /// Vertical field of view, in radians.
        yfov: f32,
        znear: f32,
        zfar: Option<f32>,
        /// `None` means the renderer supplies its current viewport aspect.
        aspect_ratio: Option<f32>,
    },
    Orthographic {
        xmag: f32,
        ymag: f32,
        znear: f32,
        zfar: f32,
    },
}

/// Quaternion in x, y, z, w order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion(pub [f32; 4]);

impl Quaternion {
    pub const IDENTITY: Self = Self([0.0, 0.0, 0.0, 1.0]);

    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Self {
        let half = angle * 0.5;
        let sin = libm::sinf(half);
        let cos = libm::cosf(half);
        Self([axis[0] * sin, axis[1] * sin, axis[2] * sin, cos])
    }

    pub fn normalized(self) -> Self {
        let [x, y, z, w] = self.0;
        let length = libm::sqrtf(x * x + y * y + z * z + w * w);
        if length <= f32::EPSILON {
            Self::IDENTITY
        } else {
            Self([x / length, y / length, z / length, w / length])
        }
    }

    pub fn rotate(self, vector: [f32; 3]) -> [f32; 3] {
        let [x, y, z, w] = self.normalized().0;
        let [vx, vy, vz] = vector;
        let tx = 2.0 * (y * vz - z * vy);
        let ty = 2.0 * (z * vx - x * vz);
        let tz = 2.0 * (x * vy - y * vx);
        [
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ]
    }
}

impl core::ops::Mul for Quaternion {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [ax, ay, az, aw] = self.0;
        let [bx, by, bz, bw] = rhs.0;
        Self([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: [f32; 3],
    pub rotation: Quaternion,
    pub projection: Projection,
}

/// Current state of the four movement keys.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Wasd {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyCam {
    pub camera: Camera,
    speed: f32,
    look_sensitivity: f32,
    #[cfg(feature = "blueprint")]
    blueprint_debug: BlueprintDebugState,
}

#[cfg(feature = "blueprint")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BlueprintDebugState {
    last_keys: Wasd,
    last_keyboard: [u32; 3],
    last_combo: u32,
    last_mouse_count: u32,
    no_keyboard: bool,
    no_mouse: bool,
    mouse_log_frames: u8,
}

impl FlyCam {
    pub const fn new(camera: Camera, speed: f32) -> Self {
        Self {
            camera,
            speed,
            look_sensitivity: 0.002,
            #[cfg(feature = "blueprint")]
            blueprint_debug: BlueprintDebugState {
                last_keys: Wasd {
                    w: false,
                    a: false,
                    s: false,
                    d: false,
                },
                last_keyboard: [0; 3],
                last_combo: 0,
                last_mouse_count: 0,
                no_keyboard: false,
                no_mouse: false,
                mouse_log_frames: 0,
            },
        }
    }

    pub const fn speed(&self) -> f32 {
        self.speed
    }

    pub fn set_speed(&mut self, units_per_second: f32) {
        self.speed = units_per_second.max(0.0);
    }

    pub const fn look_sensitivity(&self) -> f32 {
        self.look_sensitivity
    }

    pub fn set_look_sensitivity(&mut self, radians_per_pixel: f32) {
        self.look_sensitivity = radians_per_pixel.max(0.0);
    }

    /// Applies a pointer delta. Positive x turns right; positive y turns down,
    /// matching the usual window-coordinate convention.
    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        let yaw = Quaternion::from_axis_angle([0.0, 1.0, 0.0], -delta_x * self.look_sensitivity);
        let pitch = Quaternion::from_axis_angle([1.0, 0.0, 0.0], -delta_y * self.look_sensitivity);
        self.camera.rotation = (yaw * self.camera.rotation * pitch).normalized();
    }

    /// Moves in camera-local X/Z. Opposing keys cancel and diagonals are
    /// normalized so they are not faster than movement along one axis.
    pub fn step(&mut self, keys: Wasd, delta_seconds: f32) {
        let x = (keys.d as u8 as f32) - (keys.a as u8 as f32);
        let z = (keys.s as u8 as f32) - (keys.w as u8 as f32);
        let length = libm::sqrtf(x * x + z * z);
        if length <= f32::EPSILON || delta_seconds <= 0.0 {
            return;
        }
        let movement = self.camera.rotation.rotate([x / length, 0.0, z / length]);
        let distance = self.speed * delta_seconds;
        for (position, direction) in self.camera.position.iter_mut().zip(movement) {
            *position += direction * distance;
        }
    }

    /// Consume the default Blueprint camera bindings from TRUEOS VLayer.
    ///
    /// The first routed keyboard/mouse pair is used (preferentially a pair
    /// with the same `combo_id`). WASD uses USB HID Keyboard/Keypad usages and
    /// the primary mouse button enables look. Mouse samples are read from the
    /// exact endpoint advertised by VLayer, so unrelated devices do not move
    /// this camera.
    #[cfg(feature = "blueprint")]
    pub fn step_blueprint(&mut self, delta_seconds: f32) -> BlueprintInputFrame {
        let keyboards = trueos_bp::hid::hid_hut_keyboards();
        let mice = trueos_bp::hid::hid_hut_mice();
        let keyboard = keyboards
            .iter()
            .find(|keyboard| mice.iter().any(|mouse| mouse.combo_id == keyboard.combo_id))
            .or_else(|| keyboards.first());
        let keys = keyboard.map_or(Wasd::default(), |keyboard| Wasd {
            w: hid_key_down(&keyboard.key_down_bits, HID_KEY_W),
            a: hid_key_down(&keyboard.key_down_bits, HID_KEY_A),
            s: hid_key_down(&keyboard.key_down_bits, HID_KEY_S),
            d: hid_key_down(&keyboard.key_down_bits, HID_KEY_D),
        });

        let keys_changed = keys != self.blueprint_debug.last_keys;
        if keys_changed {
            trueos_bp::logl::log(
                trueos_bp::logl::level::DEBUG,
                format_args!(
                    "picasso flycam: WASD w={} a={} s={} d={}",
                    keys.w as u8, keys.a as u8, keys.s as u8, keys.d as u8
                ),
            );
            self.blueprint_debug.last_keys = keys;
        }
        self.step(keys, delta_seconds);

        let Some(keyboard) = keyboard else {
            if !self.blueprint_debug.no_keyboard {
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!("picasso flycam: no HID keyboard available"),
                );
                self.blueprint_debug.no_keyboard = true;
                self.blueprint_debug.no_mouse = false;
            }
            if keys_changed {
                let [x, y, z] = self.camera.position;
                let [qx, qy, qz, qw] = self.camera.rotation.0;
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!(
                        "picasso flycam: camera=({:.3},{:.3},{:.3}) quat=({:.3},{:.3},{:.3},{:.3})",
                        x, y, z, qx, qy, qz, qw
                    ),
                );
            }
            return BlueprintInputFrame {
                keys,
                mouse_samples: 0,
                dropped_mouse_samples: 0,
            };
        };
        self.blueprint_debug.no_keyboard = false;
        if keys_changed {
            let [x, y, z] = self.camera.position;
            let [qx, qy, qz, qw] = self.camera.rotation.0;
            trueos_bp::logl::log(
                trueos_bp::logl::level::DEBUG,
                format_args!(
                    "picasso flycam: camera=({:.3},{:.3},{:.3}) quat=({:.3},{:.3},{:.3},{:.3})",
                    x, y, z, qx, qy, qz, qw
                ),
            );
        }

        let keyboard_endpoint = [keyboard.controller_id, keyboard.slot_id, keyboard.ep_target];
        let matching_mouse_count = mice
            .iter()
            .filter(|mouse| mouse.combo_id == keyboard.combo_id)
            .count() as u32;
        if self.blueprint_debug.last_combo != keyboard.combo_id
            || self.blueprint_debug.last_keyboard != keyboard_endpoint
            || self.blueprint_debug.last_mouse_count != matching_mouse_count
        {
            trueos_bp::logl::log(
                trueos_bp::logl::level::DEBUG,
                format_args!(
                    "picasso flycam: input combo={} keyboard={}:{}:{} mice={}",
                    keyboard.combo_id,
                    keyboard.controller_id,
                    keyboard.slot_id,
                    keyboard.ep_target,
                    matching_mouse_count
                ),
            );
            for mouse in mice
                .iter()
                .filter(|mouse| mouse.combo_id == keyboard.combo_id)
            {
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!(
                        "picasso flycam: mouse endpoint={}:{}:{}",
                        mouse.controller_id, mouse.slot_id, mouse.ep_target
                    ),
                );
            }
            self.blueprint_debug.last_combo = keyboard.combo_id;
            self.blueprint_debug.last_keyboard = keyboard_endpoint;
            self.blueprint_debug.last_mouse_count = matching_mouse_count;
        }
        if matching_mouse_count == 0 {
            if !self.blueprint_debug.no_mouse {
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!(
                        "picasso flycam: no mouse matched keyboard combo={}",
                        keyboard.combo_id
                    ),
                );
                self.blueprint_debug.no_mouse = true;
            }
        } else {
            self.blueprint_debug.no_mouse = false;
        }
        let mut mouse_samples = 0u32;
        let mut dropped_mouse_samples = 0u32;
        for mouse in mice
            .iter()
            .filter(|mouse| mouse.combo_id == keyboard.combo_id)
        {
            let (samples, dropped) = trueos_bp::hid::hid_mouse_read(
                mouse.controller_id,
                mouse.slot_id,
                mouse.ep_target,
                BLUEPRINT_MOUSE_SAMPLE_CAP,
            );
            dropped_mouse_samples = dropped_mouse_samples.saturating_add(dropped);
            for sample in samples {
                mouse_samples = mouse_samples.saturating_add(1);
                if sample.buttons & MOUSE_BUTTON_LEFT != 0 {
                    self.look(sample.dx as f32, sample.dy as f32);
                }
            }
        }
        if mouse_samples != 0 || dropped_mouse_samples != 0 {
            let periodic = self.blueprint_debug.mouse_log_frames == 0;
            self.blueprint_debug.mouse_log_frames =
                self.blueprint_debug.mouse_log_frames.wrapping_add(1) % 30;
            if periodic || dropped_mouse_samples != 0 {
                let [x, y, z] = self.camera.position;
                let [qx, qy, qz, qw] = self.camera.rotation.0;
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!(
                        "picasso flycam: mouse samples={} dropped={} camera=({:.3},{:.3},{:.3}) quat=({:.3},{:.3},{:.3},{:.3})",
                        mouse_samples, dropped_mouse_samples, x, y, z, qx, qy, qz, qw
                    ),
                );
            }
        } else {
            self.blueprint_debug.mouse_log_frames = 0;
        }
        BlueprintInputFrame {
            keys,
            mouse_samples,
            dropped_mouse_samples,
        }
    }
}

/// USB HID Keyboard/Keypad usages used by the default fly-camera binding.
pub const HID_KEY_A: u8 = 0x04;
pub const HID_KEY_D: u8 = 0x07;
pub const HID_KEY_S: u8 = 0x16;
pub const HID_KEY_W: u8 = 0x1a;
pub const MOUSE_BUTTON_LEFT: u8 = 1;
#[cfg(feature = "blueprint")]
const BLUEPRINT_MOUSE_SAMPLE_CAP: u32 = 64;

#[cfg(feature = "blueprint")]
fn hid_key_down(bits: &[u32; 8], usage: u8) -> bool {
    let index = usage as usize;
    bits[index / 32] & (1u32 << (index % 32)) != 0
}

/// Input accounting returned by [`FlyCam::step_blueprint`].
#[cfg(feature = "blueprint")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlueprintInputFrame {
    pub keys: Wasd,
    pub mouse_samples: u32,
    pub dropped_mouse_samples: u32,
}
