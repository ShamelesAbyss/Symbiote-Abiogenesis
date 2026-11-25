//! Professional aspect-aware Simulation Camera & World Bounds (`WorldBounds`, `SimCamera`).
//!
//! Strictly separates simulation topology (toroidal wrapping on `[0, width) x [0, height)`)
//! from camera/rendering topology so full-world overview NEVER displays repeated copies
//! of the ecosystem. Provides a single source of truth for screen<->world projection,
//! cursor-centered zoom, frustum culling, and minimap navigation.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldBounds {
    pub width: f32,
    pub height: f32,
}

impl Default for WorldBounds {
    fn default() -> Self {
        Self {
            width: 16.0,
            height: 9.0,
        }
    }
}

impl WorldBounds {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width: width.clamp(4.0, 64.0),
            height: height.clamp(4.0, 64.0),
        }
    }

    #[inline]
    pub fn wrap_x(self, x: f32) -> f32 {
        let mut r = x.rem_euclid(self.width);
        if r >= self.width {
            r = 0.0;
        }
        r
    }

    #[inline]
    pub fn wrap_y(self, y: f32) -> f32 {
        let mut r = y.rem_euclid(self.height);
        if r >= self.height {
            r = 0.0;
        }
        r
    }

    /// Shortest signed toroidal delta along X on `[0, width)`.
    #[inline]
    pub fn delta_x(self, dx: f32) -> f32 {
        let half = self.width * 0.5;
        if dx > half {
            dx - self.width
        } else if dx < -half {
            dx + self.width
        } else {
            dx
        }
    }

    /// Shortest signed toroidal delta along Y on `[0, height)`.
    #[inline]
    pub fn delta_y(self, dy: f32) -> f32 {
        let half = self.height * 0.5;
        if dy > half {
            dy - self.height
        } else if dy < -half {
            dy + self.height
        } else {
            dy
        }
    }

    #[inline]
    pub fn contains(self, x: f32, y: f32) -> bool {
        x >= 0.0 && x <= self.width && y >= 0.0 && y <= self.height
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SimCamera {
    pub center_x: f32,
    pub center_y: f32,
    /// Half-height of the visible viewport in world units.
    /// Smaller `half_h` = zoomed in microscopically; larger `half_h` = zoomed out overview.
    pub half_h: f32,
    pub show_bounds: bool,
}

impl SimCamera {
    pub fn fit_world(bounds: WorldBounds, aspect: f32) -> Self {
        let half_h = Self::max_half_h(bounds, aspect);
        Self {
            center_x: bounds.width * 0.5,
            center_y: bounds.height * 0.5,
            half_h,
            show_bounds: true,
        }
    }

    /// Maximum half-height showing the entire world cleanly with a slight 4% breathing margin,
    /// without ever exposing repeated world copies.
    pub fn max_half_h(bounds: WorldBounds, aspect: f32) -> f32 {
        let safe_aspect = aspect.max(0.25);
        let by_h = bounds.height * 0.52;
        let by_w = (bounds.width * 0.52) / safe_aspect;
        by_h.max(by_w)
    }

    pub fn min_half_h() -> f32 {
        0.22
    }

    /// Relative zoom factor (1.0 = full world overview, higher = zoomed in).
    pub fn zoom_factor(self, bounds: WorldBounds, aspect: f32) -> f32 {
        (Self::max_half_h(bounds, aspect) / self.half_h.max(Self::min_half_h())).max(0.5)
    }

    pub fn set_zoom_factor(&mut self, zoom: f32, bounds: WorldBounds, aspect: f32) {
        let max_h = Self::max_half_h(bounds, aspect);
        self.half_h = (max_h / zoom.clamp(1.0, 28.0)).clamp(Self::min_half_h(), max_h);
        self.clamp_to_bounds(bounds, aspect);
    }

    /// Convert physical screen pixel `(px, py)` in `[0..screen_w, 0..screen_h]` to world coordinates.
    #[inline]
    pub fn screen_to_world(
        self,
        px: f32,
        py: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> (f32, f32) {
        let w = screen_w.max(1.0);
        let h = screen_h.max(1.0);
        let aspect = w / h;
        let ndc_x = (px / w - 0.5) * 2.0;
        let ndc_y = (py / h - 0.5) * 2.0;
        let wx = self.center_x + ndc_x * self.half_h * aspect;
        let wy = self.center_y + ndc_y * self.half_h;
        (wx, wy)
    }

    /// Convert world coordinate `(wx, wy)` to normalized viewport UV `[0..1, 0..1]`.
    #[inline]
    pub fn world_to_uv(self, wx: f32, wy: f32, aspect: f32) -> (f32, f32) {
        let half_w = (self.half_h * aspect.max(0.25)).max(1e-5);
        let u = 0.5 + (wx - self.center_x) / (2.0 * half_w);
        let v = 0.5 + (wy - self.center_y) / (2.0 * self.half_h.max(1e-5));
        (u, v)
    }

    /// Cursor-centered zoom: keeps the world point under `(px, py)` fixed during scroll zoom.
    pub fn zoom_at_cursor(
        &mut self,
        factor: f32,
        px: f32,
        py: f32,
        screen_w: f32,
        screen_h: f32,
        bounds: WorldBounds,
    ) {
        let aspect = screen_w.max(1.0) / screen_h.max(1.0);
        let (before_x, before_y) = self.screen_to_world(px, py, screen_w, screen_h);
        let max_h = Self::max_half_h(bounds, aspect);
        self.half_h = (self.half_h / factor.clamp(0.5, 2.0)).clamp(Self::min_half_h(), max_h);
        let (after_x, after_y) = self.screen_to_world(px, py, screen_w, screen_h);
        self.center_x += before_x - after_x;
        self.center_y += before_y - after_y;
        self.clamp_to_bounds(bounds, aspect);
    }

    pub fn pan(&mut self, dx_frac: f32, dy_frac: f32, bounds: WorldBounds, aspect: f32) {
        self.center_x += dx_frac * self.half_h * aspect.max(0.25);
        self.center_y += dy_frac * self.half_h;
        self.clamp_to_bounds(bounds, aspect);
    }

    pub fn clamp_to_bounds(&mut self, bounds: WorldBounds, aspect: f32) {
        let max_h = Self::max_half_h(bounds, aspect);
        self.half_h = self.half_h.clamp(Self::min_half_h(), max_h);
        let half_w = self.half_h * aspect.max(0.25);

        if half_w >= bounds.width * 0.5 {
            self.center_x = bounds.width * 0.5;
        } else {
            self.center_x = self.center_x.clamp(half_w * 0.5, bounds.width - half_w * 0.5);
        }

        if self.half_h >= bounds.height * 0.5 {
            self.center_y = bounds.height * 0.5;
        } else {
            self.center_y = self
                .center_y
                .clamp(self.half_h * 0.5, bounds.height - self.half_h * 0.5);
        }
    }

    /// Visible world rectangle `(min_x, min_y, max_x, max_y)` for frustum culling & minimap.
    pub fn visible_rect(self, aspect: f32) -> (f32, f32, f32, f32) {
        let half_w = self.half_h * aspect.max(0.25);
        (
            self.center_x - half_w,
            self.center_y - self.half_h,
            self.center_x + half_w,
            self.center_y + self.half_h,
        )
    }

    /// Frustum culling check with seam-fragment support for organisms near world boundaries.
    pub fn is_visible(self, x: f32, y: f32, radius: f32, bounds: WorldBounds, aspect: f32) -> bool {
        let (min_x, min_y, max_x, max_y) = self.visible_rect(aspect);
        let in_x = (x + radius >= min_x && x - radius <= max_x)
            || (x - bounds.width + radius >= min_x && x - bounds.width - radius <= max_x)
            || (x + bounds.width + radius >= min_x && x + bounds.width - radius <= max_x);
        let in_y = (y + radius >= min_y && y - radius <= max_y)
            || (y - bounds.height + radius >= min_y && y - bounds.height - radius <= max_y)
            || (y + bounds.height + radius >= min_y && y + bounds.height - radius <= max_y);
        in_x && in_y
    }
}
