use wasm_bindgen::prelude::*;

mod fast;
use fast::{detect_features, KeyPoint};

mod tracker;
use tracker::track_points;

#[wasm_bindgen]
pub struct TrackingResult {
    pub dx: f32,
    pub dy: f32,
}

#[wasm_bindgen]
pub struct AREngine {
    width: u32,
    height: u32,
    prev_frame: Vec<u8>,
    points: Vec<KeyPoint>,
    gray_buffer: Vec<u8>,
}

#[wasm_bindgen]
impl AREngine {
    pub fn init(width: u32, height: u32) -> AREngine {
        console_error_panic_hook::set_once();
        AREngine {
            width,
            height,
            prev_frame: Vec::new(),
            points: Vec::new(),
            gray_buffer: Vec::new(),
        }
    }

    pub fn process_frame(&mut self, image_data: &[u8]) -> TrackingResult {
        let expected_gray = (self.width * self.height) as usize;
        let expected_rgba = expected_gray * 4;

        // We need a way to reference the grayscale data (either slice or buffer)
        // Since we can't easily return a reference to self.gray_buffer from a block that borrows self mutably
        // while also mutating self.gray_buffer, we have to be careful.
        // Actually, we can mutate gray_buffer first, then take a slice.

        let use_buffer_conversion = image_data.len() == expected_rgba;

        if image_data.len() != expected_gray && !use_buffer_conversion {
            // Invalid input size
            return TrackingResult { dx: 0.0, dy: 0.0 };
        }

        if use_buffer_conversion {
            if self.gray_buffer.len() != expected_gray {
                self.gray_buffer.resize(expected_gray, 0);
            }

            for (i, chunk) in image_data.chunks_exact(4).enumerate() {
                let r = chunk[0] as u16;
                let g = chunk[1] as u16;
                let b = chunk[2] as u16;
                // Y = 0.299 R + 0.587 G + 0.114 B
                let y = (77 * r + 150 * g + 29 * b) >> 8;
                self.gray_buffer[i] = y as u8;
            }
        }

        // Now we can get the slice.
        // Note: We need to be careful with borrowing.
        // &self.gray_buffer is borrowed from self.
        // We pass it to methods.

        // Let's defer getting the slice until needed, or copy logic.
        // But we want to avoid copy.
        // We can't hold `gray_data = &self.gray_buffer` and then mutate `self.points`.
        // So we might need to index or clone? No, detect_features takes slice.
        // We can pass `if use_buffer { &self.gray_buffer } else { image_data }` to calls.

        // But `track_points` needs `prev_frame` (from self) and `curr_frame`.
        // `prev_frame` is in self.
        // `curr_frame` is either `image_data` or `&self.gray_buffer`.

        let mut dx = 0.0;
        let mut dy = 0.0;

        if self.prev_frame.is_empty() {
            // First frame
            let curr_slice = if use_buffer_conversion { &self.gray_buffer } else { image_data };
            self.points = detect_features(curr_slice, self.width, self.height, 30);

            if self.prev_frame.len() != expected_gray {
                self.prev_frame.resize(expected_gray, 0);
            }
            self.prev_frame.copy_from_slice(curr_slice);
        } else {
            // Track
            if self.points.len() < 5 {
                 let curr_slice = if use_buffer_conversion { &self.gray_buffer } else { image_data };
                 self.points = detect_features(curr_slice, self.width, self.height, 30);

                 if self.prev_frame.len() != expected_gray {
                    self.prev_frame.resize(expected_gray, 0);
                 }
                 self.prev_frame.copy_from_slice(curr_slice);
            } else {
                 // We need to borrow prev_frame and points from self, AND get curr_slice.
                 // curr_slice borrows from self (if buffer) or is argument.
                 // If curr_slice borrows self.gray_buffer, we are borrowing self immutable.
                 // track_points takes `&prev` and `&curr` and `&points`.
                 // All immutable borrows. This is fine.

                 // But wait, `track_points` returns `new_points`. We assign to `self.points` AFTER call.
                 // So we are fine.

                 let (new_points, computed_dx, computed_dy) = {
                     let curr_slice = if use_buffer_conversion { &self.gray_buffer } else { image_data };
                     track_points(
                         &self.prev_frame,
                         curr_slice,
                         self.width,
                         self.height,
                         &self.points
                     )
                 };

                 self.points = new_points;
                 dx = computed_dx;
                 dy = computed_dy;

                 // Update prev_frame
                 let curr_slice = if use_buffer_conversion { &self.gray_buffer } else { image_data };
                 if self.prev_frame.len() != expected_gray {
                    self.prev_frame.resize(expected_gray, 0);
                 }
                 self.prev_frame.copy_from_slice(curr_slice);
            }
        }

        TrackingResult { dx, dy }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_gray() {
        let width = 40;
        let height = 40;
        let mut engine = AREngine::init(width, height);

        let mut frame1 = vec![0u8; (width * height) as usize];
        for y in 18..28 {
            for x in 18..28 {
                frame1[(y * width + x) as usize] = 200;
            }
        }

        let res1 = engine.process_frame(&frame1);
        println!("Frame 1: dx={}, dy={}, points={}", res1.dx, res1.dy, engine.points.len());
        assert_eq!(res1.dx, 0.0);
        assert_eq!(res1.dy, 0.0);

        let mut frame2 = vec![0u8; (width * height) as usize];
        for y in 18..28 {
            for x in 18..28 {
                frame2[((y + 1) * width + (x + 2)) as usize] = 200;
            }
        }

        let res2 = engine.process_frame(&frame2);
        println!("Frame 2: dx={}, dy={}, points={}", res2.dx, res2.dy, engine.points.len());

        // If we reset, points.len might be non-zero but dx/dy is 0.
        // We expect tracking to succeed.
        assert!((res2.dx - 2.0).abs() < 0.5, "Expected dx ~ 2.0, got {}", res2.dx);
        assert!((res2.dy - 1.0).abs() < 0.5, "Expected dy ~ 1.0, got {}", res2.dy);
    }

    #[test]
    fn test_tracking_rgba() {
        let width = 40;
        let height = 40;
        let mut engine = AREngine::init(width, height);

        // Frame 1 RGBA
        let mut frame1 = vec![0u8; (width * height * 4) as usize];
        for y in 18..23 {
            for x in 18..23 {
                let idx = (y * width + x) as usize * 4;
                frame1[idx] = 200;   // R
                frame1[idx+1] = 200; // G
                frame1[idx+2] = 200; // B
                frame1[idx+3] = 255; // A
            }
        }

        let res1 = engine.process_frame(&frame1);
        assert_eq!(res1.dx, 0.0);

        // Frame 2 RGBA
        let mut frame2 = vec![0u8; (width * height * 4) as usize];
        for y in 18..23 {
            for x in 18..23 {
                let idx = ((y + 1) * width + (x + 2)) as usize * 4;
                frame2[idx] = 200;
                frame2[idx+1] = 200;
                frame2[idx+2] = 200;
                frame2[idx+3] = 255;
            }
        }

        let res2 = engine.process_frame(&frame2);
        assert!((res2.dx - 2.0).abs() < 0.5);
        assert!((res2.dy - 1.0).abs() < 0.5);
    }
}
