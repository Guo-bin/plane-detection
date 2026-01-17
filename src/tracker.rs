use crate::fast::KeyPoint;
use nalgebra::{Matrix2, Vector2};

pub fn track_points(
    prev_img: &[u8],
    curr_img: &[u8],
    width: u32,
    height: u32,
    old_points: &[KeyPoint],
) -> (Vec<KeyPoint>, f32, f32) {
    let w = width as i32;
    let h = height as i32;
    // Window size 15x15
    let half_win = 7;
    let mut new_points = Vec::with_capacity(old_points.len());
    let mut total_dx = 0.0;
    let mut total_dy = 0.0;
    let mut tracked_count = 0;

    for point in old_points {
        let px = point.x as f32;
        let py = point.y as f32;

        let mut cur_x = px;
        let mut cur_y = py;

        let max_iter = 10;
        let epsilon = 0.001; // Squared epsilon

        let mut lost = false;

        // Ensure we are inside bounds with margin for gradient (1) and window (7) -> 8
        if px < 8.0 || px >= (w - 8) as f32 || py < 8.0 || py >= (h - 8) as f32 {
            continue;
        }

        for _ in 0..max_iter {
            if cur_x < 8.0 || cur_x >= (w - 8) as f32 ||
               cur_y < 8.0 || cur_y >= (h - 8) as f32 {
                lost = true;
                break;
            }

            let mut g = Matrix2::<f32>::zeros();
            let mut b = Vector2::<f32>::zeros();

            for wy in -half_win..=half_win {
                for wx in -half_win..=half_win {
                    let px_idx = (px as i32 + wx) as usize;
                    let py_idx = (py as i32 + wy) as usize;

                    let prev_idx = py_idx * width as usize + px_idx;

                    // Gradient on PREV image
                    let val_x_plus = prev_img[prev_idx + 1] as f32;
                    let val_x_minus = prev_img[prev_idx - 1] as f32;
                    let ix = (val_x_plus - val_x_minus) * 0.5;

                    let val_y_plus = prev_img[prev_idx + width as usize] as f32;
                    let val_y_minus = prev_img[prev_idx - width as usize] as f32;
                    let iy = (val_y_plus - val_y_minus) * 0.5;

                    let val_prev = prev_img[prev_idx] as f32;

                    let target_x = cur_x + wx as f32;
                    let target_y = cur_y + wy as f32;

                    let val_curr = get_pixel_bilinear(curr_img, w, h, target_x, target_y);

                    let diff = val_prev - val_curr;

                    g[(0, 0)] += ix * ix;
                    g[(0, 1)] += ix * iy;
                    g[(1, 0)] += ix * iy;
                    g[(1, 1)] += iy * iy;

                    b[0] += ix * diff;
                    b[1] += iy * diff;
                }
            }

            if let Some(inv_g) = g.try_inverse() {
                 let d = inv_g * b;
                 cur_x += d[0];
                 cur_y += d[1];

                 if d.norm_squared() < epsilon {
                     break;
                 }
            } else {
                lost = true;
                break;
            }
        }

        if !lost {
            new_points.push(KeyPoint { x: cur_x as usize, y: cur_y as usize });
            total_dx += cur_x - px;
            total_dy += cur_y - py;
            tracked_count += 1;
        }
    }

    let avg_dx = if tracked_count > 0 { total_dx / tracked_count as f32 } else { 0.0 };
    let avg_dy = if tracked_count > 0 { total_dy / tracked_count as f32 } else { 0.0 };

    (new_points, avg_dx, avg_dy)
}

fn get_pixel_bilinear(img: &[u8], w: i32, h: i32, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Clamp to ensure valid indices (though loop checks should prevent this usually)
    let x0c = x0.clamp(0, w - 1);
    let x1c = x1.clamp(0, w - 1);
    let y0c = y0.clamp(0, h - 1);
    let y1c = y1.clamp(0, h - 1);

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let v00 = img[(y0c * w + x0c) as usize] as f32;
    let v10 = img[(y0c * w + x1c) as usize] as f32;
    let v01 = img[(y1c * w + x0c) as usize] as f32;
    let v11 = img[(y1c * w + x1c) as usize] as f32;

    let top = v00 * (1.0 - wx) + v10 * wx;
    let bottom = v01 * (1.0 - wx) + v11 * wx;

    top * (1.0 - wy) + bottom * wy
}
