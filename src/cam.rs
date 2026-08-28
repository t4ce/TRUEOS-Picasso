//! Renderer-neutral camera data and a tiny quaternion fly-camera controller.
//!
//! Input backends keep ownership of key and pointer events. Feed their current
//! WASD state to [`FlyCam::step`] and pointer deltas to [`FlyCam::look`].
//!
//! Blueprint applications can use [`FlyCam::step_ui4`] together with
//! [`FlyCam::handle_ui4_pointer_event`]. UI4 remains the owner of physical
//! HID, focus, selection, and pointer capture; the adapter only observes the
//! events that the application has already chosen to drain.

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
    ui4: Ui4FlyState,
}

#[cfg(feature = "blueprint")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Ui4FlyState {
    last_keys: Wasd,
    route: Option<Ui4RouteIdentity>,
    last_combo: u32,
    pointer_log_frames: u8,
}

#[cfg(feature = "blueprint")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Ui4RouteIdentity {
    cursor: trueos_bp::ui4_scene::CursorSource,
    combo_id: u32,
}

impl FlyCam {
    pub const fn new(camera: Camera, speed: f32) -> Self {
        Self {
            camera,
            speed,
            look_sensitivity: 0.002,
            #[cfg(feature = "blueprint")]
            ui4: Ui4FlyState {
                last_keys: Wasd {
                    w: false,
                    a: false,
                    s: false,
                    d: false,
                },
                route: None,
                last_combo: 0,
                pointer_log_frames: 0,
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

    /// Applies camera-local quaternion yaw and pitch. Positive x turns toward
    /// the camera's visual right and positive y turns down at any camera roll.
    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        let yaw = Quaternion::from_axis_angle([0.0, 1.0, 0.0], -delta_x * self.look_sensitivity);
        let pitch = Quaternion::from_axis_angle([1.0, 0.0, 0.0], -delta_y * self.look_sensitivity);
        self.camera.rotation = (self.camera.rotation * yaw * pitch).normalized();
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

    /// Sample WASD from UI4's focused route. This never drains frame events.
    #[cfg(feature = "blueprint")]
    pub fn step_ui4(
        &mut self,
        frame: &trueos_bp::ui4_scene::Frame,
        delta_seconds: f32,
    ) -> Result<Ui4InputFrame, trueos_bp::ui4_scene::Error> {
        let routes = frame.input_routes()?;
        let focused_keyboard = routes
            .iter()
            .any(|route| route.application_focus)
            .then(|| frame.keyboard_state())
            .transpose()?
            .flatten();
        let route = focused_keyboard
            .and_then(|keyboard| {
                routes.iter().find(|route| {
                    ui4_route_eligible(route) && ui4_keyboard_matches(route, keyboard)
                })
            })
            .or_else(|| {
                self.ui4.route.and_then(|id| {
                    routes.iter().find(|route| {
                        ui4_route_eligible(route) && Ui4RouteIdentity::from(*route) == id
                    })
                })
            })
            .or_else(|| routes.iter().find(|route| ui4_route_eligible(route)));
        let identity = route.map(Ui4RouteIdentity::from);
        if identity != self.ui4.route {
            self.ui4.route = identity;
            if let Some(route) = route {
                self.ui4.last_combo = route.combo_id;
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!(
                        "picasso flycam: UI4 route combo={} cursor={}:{}:{} focus={} selected={} virtual={} keyboard={}",
                        route.combo_id,
                        route.cursor.controller_id,
                        route.cursor.slot_id,
                        route.cursor.ep_target,
                        route.application_focus as u8,
                        route.selected_for_window as u8,
                        route.vcursor as u8,
                        route.keyboard.is_some() as u8
                    ),
                );
                if let Some(keyboard) = route.keyboard {
                    trueos_bp::logl::log(
                        trueos_bp::logl::level::DEBUG,
                        format_args!(
                            "picasso flycam: UI4 keyboard={}:{}:{} combo={} virtual={}",
                            keyboard.controller_id,
                            keyboard.slot_id,
                            keyboard.ep_target,
                            keyboard.combo_id,
                            keyboard.virtual_keyboard as u8,
                        ),
                    );
                }
            } else {
                trueos_bp::logl::log(
                    trueos_bp::logl::level::DEBUG,
                    format_args!("picasso flycam: no focused UI4 input route"),
                );
            }
        }
        let keyboard = identity.and_then(|id| {
            routes
                .iter()
                .find(|route| Ui4RouteIdentity::from(*route) == id)
                .and_then(|route| route.keyboard)
        });
        let keys = keyboard.map_or(Wasd::default(), wasd_from_keyboard);
        let changed = keys != self.ui4.last_keys;
        if changed {
            trueos_bp::logl::log(
                trueos_bp::logl::level::DEBUG,
                format_args!(
                    "picasso flycam: UI4 WASD w={} a={} s={} d={}",
                    keys.w as u8, keys.a as u8, keys.s as u8, keys.d as u8
                ),
            );
            self.ui4.last_keys = keys;
        }
        self.step(keys, delta_seconds);
        if changed {
            self.log_ui4_pose("WASD");
        }
        Ok(Ui4InputFrame {
            keys,
            route_count: routes.len() as u32,
            focused: identity.is_some(),
        })
    }

    /// Consume one pointer event already drained by the application. Camera
    /// look is active only while the middle mouse button is held. `allow_look`
    /// preserves application-owned hit regions such as a resize grip.
    #[cfg(feature = "blueprint")]
    pub fn handle_ui4_pointer_event(
        &mut self,
        event: &trueos_bp::ui4_scene::PointerEvent,
        allow_look: bool,
    ) -> bool {
        let identity = Ui4RouteIdentity {
            cursor: event.source,
            combo_id: event.combo_id,
        };
        let look_gesture = event.buttons_down & trueos_bp::ui4_scene::POINTER_BUTTON_MIDDLE != 0;
        let active = allow_look
            && Some(identity) == self.ui4.route
            && look_gesture
            && (event.dx != 0 || event.dy != 0);
        if !active {
            return false;
        }
        self.look(event.dx as f32, event.dy as f32);
        let periodic = self.ui4.pointer_log_frames == 0;
        self.ui4.pointer_log_frames = self.ui4.pointer_log_frames.wrapping_add(1) % 30;
        if periodic {
            trueos_bp::logl::log(
                trueos_bp::logl::level::DEBUG,
                format_args!(
                    "picasso flycam: UI4 pointer combo={} dx={} dy={} virtual={} dropped=unavailable",
                    event.combo_id, event.dx, event.dy, event.vcursor as u8,
                ),
            );
            self.log_ui4_pose("pointer");
        }
        true
    }

    #[cfg(feature = "blueprint")]
    fn log_ui4_pose(&self, source: &str) {
        let [x, y, z] = self.camera.position;
        let [qx, qy, qz, qw] = self.camera.rotation.0;
        trueos_bp::logl::log(
            trueos_bp::logl::level::DEBUG,
            format_args!(
                "picasso flycam: UI4 {} combo={} camera=({:.3},{:.3},{:.3}) quat=({:.3},{:.3},{:.3},{:.3})",
                source, self.ui4.last_combo, x, y, z, qx, qy, qz, qw
            ),
        );
    }
}

/// USB HID Keyboard/Keypad usages used by the default fly-camera binding.
pub const HID_KEY_A: u8 = 0x04;
pub const HID_KEY_D: u8 = 0x07;
pub const HID_KEY_S: u8 = 0x16;
pub const HID_KEY_W: u8 = 0x1a;
#[cfg(feature = "blueprint")]
impl From<&trueos_bp::ui4_scene::InputRoute> for Ui4RouteIdentity {
    fn from(route: &trueos_bp::ui4_scene::InputRoute) -> Self {
        Self {
            cursor: route.cursor,
            combo_id: route.combo_id,
        }
    }
}

#[cfg(feature = "blueprint")]
fn ui4_route_eligible(route: &trueos_bp::ui4_scene::InputRoute) -> bool {
    route.application_focus && route.selected_for_window
}

#[cfg(feature = "blueprint")]
fn ui4_keyboard_matches(
    route: &trueos_bp::ui4_scene::InputRoute,
    keyboard: trueos_bp::ui4_scene::KeyboardState,
) -> bool {
    route.keyboard.is_some_and(|candidate| {
        candidate.controller_id == keyboard.controller_id
            && candidate.slot_id == keyboard.slot_id
            && candidate.ep_target == keyboard.ep_target
            && candidate.combo_id == keyboard.combo_id
    })
}

#[cfg(feature = "blueprint")]
fn wasd_from_keyboard(keyboard: trueos_bp::ui4_scene::KeyboardState) -> Wasd {
    Wasd {
        w: keyboard.is_down(HID_KEY_W),
        a: keyboard.is_down(HID_KEY_A),
        s: keyboard.is_down(HID_KEY_S),
        d: keyboard.is_down(HID_KEY_D),
    }
}

/// UI4-routed input accounting. UI4 exposes no pointer-drop count in this ABI.
#[cfg(feature = "blueprint")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ui4InputFrame {
    pub keys: Wasd,
    pub route_count: u32,
    pub focused: bool,
}
